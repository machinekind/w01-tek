"""Search space: genome <-> physical parameters <-> batched MJX model.

A genome is a vector u in [0,1]^dim. Each block decodes to one physical
parameter (log- or linear-scaled inside its bounds). Joint parameters
(kp/kd/damping/armature/frictionloss) carry one value per group — shared,
per joint type (first/second/third within a leg), or per joint. Scalar
parameters are torque_scale (forcerange multiplier) and latency; latency
is not a model field, so batched_model returns it separately for the
rollout to apply as a command-stream delay.

Model batching follows randomize.py: a vmapped builder returns
per-candidate field copies, tree_replace produces the batched model plus
an in_axes template for vmapping the rollout.
"""

import jax
import jax.numpy as jnp
import numpy as np

JOINT_PARAMS = ("kp", "kd", "damping", "armature", "frictionloss")
SCALAR_PARAMS = ("torque_scale", "latency")
ALL_PARAMS = JOINT_PARAMS + SCALAR_PARAMS
# CLI default: only parameters an ordinary excitation bag can actually pin.
# torque_scale needs deliberate torque saturation; without that it only
# sponges window-init error. Opt in when the bag really saturates.
DEFAULT_PARAMS = JOINT_PARAMS + ("latency",)
GROUPINGS = ("shared", "per_type", "per_joint")

# name -> (lo, hi, scale). Bounds bracket the current model values
# (kp=20, kd=1, damping=0.05, armature=0.01, frictionloss=0.01) by a wide
# margin; log scaling keeps the low decades searchable. latency is in
# control-grid steps (sim-dt grid by default, so 1 step = 4 ms).
DEFAULT_BOUNDS = {
    "kp": (5.0, 80.0, "log"),
    "kd": (0.1, 5.0, "log"),
    "damping": (0.005, 0.5, "log"),
    "armature": (0.001, 0.05, "log"),
    "frictionloss": (0.001, 0.3, "log"),
    "torque_scale": (0.5, 1.5, "lin"),
    "latency": (0.0, 8.0, "lin"),
}


class ParamSpace:
    def __init__(self, mj_model, params=ALL_PARAMS, grouping="per_type", bounds=None):
        unknown = set(params) - set(ALL_PARAMS)
        if unknown:
            raise ValueError(f"unknown params {sorted(unknown)}; pick from {ALL_PARAMS}")
        if grouping not in GROUPINGS:
            raise ValueError(f"grouping must be one of {GROUPINGS}, got {grouping!r}")
        self.params = tuple(p for p in ALL_PARAMS if p in params)
        self.grouping = grouping
        self.bounds = {**DEFAULT_BOUNDS, **(bounds or {})}

        nu = mj_model.nu
        if grouping == "shared":
            self._gmap = np.zeros(nu, dtype=int)
            self.group_names = ("all",)
        elif grouping == "per_type":
            # Actuators are declared per leg as (first, second, third).
            self._gmap = np.arange(nu) % 3
            self.group_names = ("first", "second", "third")
        else:
            self._gmap = np.arange(nu)
            self.group_names = tuple(mj_model.actuator(i).name for i in range(nu))
        self._ngroups = int(self._gmap.max()) + 1

        trn = mj_model.actuator_trnid[:, 0]
        self.qadr = np.array([mj_model.jnt_qposadr[j] for j in trn])
        self.vadr = np.array([mj_model.jnt_dofadr[j] for j in trn])
        self._base_forcerange = np.array(mj_model.actuator_forcerange)

        self._slices = {}
        d = 0
        for p in self.params:
            n = self._ngroups if p in JOINT_PARAMS else 1
            self._slices[p] = slice(d, d + n)
            d += n
        self.dim = d

        # Current-model values, group-averaged: the CMA-ES start mean.
        counts = np.bincount(self._gmap, minlength=self._ngroups)

        def group_mean(per_act):
            return np.bincount(self._gmap, per_act, self._ngroups) / counts

        self._defaults = {
            "kp": group_mean(mj_model.actuator_gainprm[:, 0]),
            "kd": group_mean(-mj_model.actuator_biasprm[:, 2]),
            "damping": group_mean(mj_model.dof_damping[self.vadr]),
            "armature": group_mean(mj_model.dof_armature[self.vadr]),
            "frictionloss": group_mean(mj_model.dof_frictionloss[self.vadr]),
            "torque_scale": np.array([1.0]),
            "latency": np.array([0.0]),
        }

    def _to_phys(self, p, u):
        lo, hi, scale = self.bounds[p]
        u = u.clip(0.0, 1.0)
        if scale == "log":
            return lo * (hi / lo) ** u
        return lo + (hi - lo) * u

    def _to_u(self, p, val):
        lo, hi, scale = self.bounds[p]
        val = np.asarray(val, dtype=float)
        if scale == "log":
            u = np.log(val / lo) / np.log(hi / lo)
        else:
            u = (val - lo) / (hi - lo)
        return np.clip(u, 0.0, 1.0)

    def default_genome(self):
        """Genome of the current model's values, nudged inside the box so
        it is a valid CMA-ES mean even for bound-edge defaults (latency=0)."""
        u = np.concatenate(
            [self._to_u(p, self._defaults[p]) for p in self.params]
        )
        return np.clip(u, 0.02, 0.98)

    def encode(self, vals):
        """Genome for a dict of physical values (missing params keep defaults)."""
        u = self.default_genome()
        for p, v in vals.items():
            n = self._slices[p].stop - self._slices[p].start
            u[self._slices[p]] = self._to_u(p, np.broadcast_to(v, (n,)))
        return u

    def describe(self, u):
        """JSON-friendly nested dict of the physical values a genome encodes."""
        u = np.asarray(u)
        out = {}
        for p in self.params:
            vals = np.asarray(self._to_phys(p, u[self._slices[p]]))
            if p in JOINT_PARAMS:
                out[p] = {g: float(v) for g, v in zip(self.group_names, vals)}
            else:
                out[p] = float(vals[0])
        return out

    def _fields(self, mjx_model, u):
        """Model-field dict for one genome (traceable, used under vmap)."""
        out = {}
        gmap = self._gmap
        vals = {p: self._to_phys(p, u[self._slices[p]]) for p in self.params}
        biasprm = mjx_model.actuator_biasprm
        if "kp" in vals:
            kp = vals["kp"][gmap]
            out["actuator_gainprm"] = mjx_model.actuator_gainprm.at[:, 0].set(kp)
            biasprm = biasprm.at[:, 1].set(-kp)
        if "kd" in vals:
            biasprm = biasprm.at[:, 2].set(-vals["kd"][gmap])
        if "kp" in vals or "kd" in vals:
            out["actuator_biasprm"] = biasprm
        if "damping" in vals:
            out["dof_damping"] = mjx_model.dof_damping.at[self.vadr].set(
                vals["damping"][gmap]
            )
        if "armature" in vals:
            out["dof_armature"] = mjx_model.dof_armature.at[self.vadr].set(
                vals["armature"][gmap]
            )
        if "frictionloss" in vals:
            out["dof_frictionloss"] = mjx_model.dof_frictionloss.at[self.vadr].set(
                vals["frictionloss"][gmap]
            )
        if "torque_scale" in vals:
            out["actuator_forcerange"] = (
                self._base_forcerange * vals["torque_scale"][0]
            )
        return out

    def _latency(self, U):
        # Traceable (works on tracers inside jit), so no numpy conversion.
        if "latency" not in self.params:
            return jnp.zeros(U.shape[0])
        return self._to_phys("latency", U[:, self._slices["latency"]])[:, 0]

    def apply(self, mjx_model, u):
        """(model, latency) for a single genome — unbatched, for replays."""
        u = jnp.asarray(u)
        model = mjx_model.tree_replace(self._fields(mjx_model, u))
        return model, self._latency(u[None])[0]

    def batched_model(self, mjx_model, U):
        """(P, dim) genomes -> (batched model, in_axes template, latency (P,))."""
        U = jnp.asarray(U)
        out = jax.vmap(lambda u: self._fields(mjx_model, u))(U)
        in_axes = jax.tree_util.tree_map(lambda x: None, mjx_model)
        in_axes = in_axes.tree_replace({k: 0 for k in out})
        return mjx_model.tree_replace(out), in_axes, self._latency(U)
