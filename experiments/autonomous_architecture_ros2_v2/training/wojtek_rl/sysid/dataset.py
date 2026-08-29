"""Bag signals -> windowed, sim-grid-aligned identification dataset.

The bag is resampled onto a uniform control grid (default: one command per
physics step, so latency resolves at substep granularity), then cut into
overlapping windows. Each window re-initializes the sim from the measured
state at its start, so model error cannot compound across the whole bag;
the first warmup_sec of every window is simulated but excluded from the
loss while the four-bar closure settles.

The model comes from mount.air_model (fixed base, no free joint), so a
window is fully determined by the joint encoders: positions/velocities
for the actuated joints from the bag, four-bar passive joints from the
bag when present (sim recordings) or the home keyframe otherwise. An
earlier on-ground mode reconstructed the floating base from IMU + a
foot-on-floor solve; it was removed because the unobservable contact
state let the optimizer launder init error into physics parameters.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SysidDataset:
    ctrl_dt: float
    n_substeps: int  # physics steps per control-grid step
    warmup_steps: int  # grid steps excluded from the loss at window start
    cmd: np.ndarray  # (W, T, nu) commanded positions, MJC convention
    meas: np.ndarray  # (W, T, nu) measured positions to fit against
    qpos0: np.ndarray  # (W, nq) initial state per window
    qvel0: np.ndarray  # (W, nv)
    t0: np.ndarray  # (W,) window start time in bag seconds


def _interp_cols(t, tp, fp):
    return np.stack([np.interp(t, tp, fp[:, j]) for j in range(fp.shape[1])], axis=1)


def build_dataset(
    mj_model,
    sig,
    ctrl_dt=None,
    window_sec=2.0,
    stride_sec=1.0,
    warmup_sec=0.3,
    max_windows=8,
):
    """Resample BagSignals onto the sim grid and cut identification windows."""
    dt_sim = float(mj_model.opt.timestep)
    ctrl_dt = ctrl_dt or dt_sim
    n_sub = int(round(ctrl_dt / dt_sim))
    if abs(n_sub * dt_sim - ctrl_dt) > 1e-9 or n_sub < 1:
        raise ValueError(
            f"ctrl_dt={ctrl_dt} must be a positive multiple of the model "
            f"timestep {dt_sim}"
        )

    trn = mj_model.actuator_trnid[:, 0]
    qadr = np.array([mj_model.jnt_qposadr[j] for j in trn])
    vadr = np.array([mj_model.jnt_dofadr[j] for j in trn])
    ctrlrange = mj_model.actuator_ctrlrange

    t_start = max(sig.t_cmd[0], sig.t_meas[0])
    t_end = min(sig.t_cmd[-1], sig.t_meas[-1])
    if t_end - t_start < window_sec:
        raise ValueError(
            f"bag overlap {t_end - t_start:.2f}s is shorter than one "
            f"{window_sec}s window"
        )
    grid = np.arange(t_start, t_end, ctrl_dt)

    hold = np.searchsorted(sig.t_cmd, grid, side="right") - 1  # zero-order hold
    cmd_g = np.clip(sig.cmd[hold], ctrlrange[:, 0], ctrlrange[:, 1])
    qpos_g = _interp_cols(grid, sig.t_meas, sig.qpos)
    qvel_g = _interp_cols(grid, sig.t_meas, sig.qvel)
    passive_g = {
        n: np.interp(grid, sig.t_meas, v) for n, v in sig.passive.items()
    }

    T = int(round(window_sec / ctrl_dt))
    stride = max(1, int(round(stride_sec / ctrl_dt)))
    warmup_steps = int(round(warmup_sec / ctrl_dt))
    if warmup_steps >= T:
        raise ValueError(f"warmup_sec={warmup_sec} leaves no scored steps in a window")
    starts = np.arange(0, len(grid) - T + 1, stride)
    if len(starts) > max_windows:
        starts = starts[
            np.unique(np.round(np.linspace(0, len(starts) - 1, max_windows)).astype(int))
        ]

    home_qpos = mj_model.key("home").qpos.copy()

    def init_state(s):
        # Fixed base: the encoders fully determine the state.
        qpos = home_qpos.copy()
        qvel = np.zeros(mj_model.nv)
        qpos[qadr] = qpos_g[s]
        for n, v in passive_g.items():
            qpos[mj_model.joint(n).qposadr[0]] = v[s]
        qvel[vadr] = qvel_g[s]
        return qpos, qvel

    inits = [init_state(s) for s in starts]
    return SysidDataset(
        ctrl_dt=ctrl_dt,
        n_substeps=n_sub,
        warmup_steps=warmup_steps,
        cmd=np.stack([cmd_g[s : s + T] for s in starts]),
        meas=np.stack([qpos_g[s : s + T] for s in starts]),
        qpos0=np.stack([q for q, _ in inits]),
        qvel0=np.stack([v for _, v in inits]),
        t0=grid[starts],
    )
