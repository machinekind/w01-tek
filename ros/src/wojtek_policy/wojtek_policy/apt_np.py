"""Pure-numpy APT-RL controller for deployment (Pi) and the export that feeds it.

    python apt_np.py export POLICY.pkl PRIOR.pkl OUT.npz      -> OUT.npz + OUT.json (everything the Pi needs)
    python apt_np.py selftest OUT.npz                          -> runs 100 steps on synthetic sensors, prints timing

Mirrors apt.CpuController.act step for step (same observation layout, buffers, latch, decoder, anchor) but takes
sensor values instead of an MjData, so the same class runs on the robot. No JAX, no MuJoCo.

Per control step (50 Hz) call  qc, tau_ff = ctl.step(gyro, grav, acc, q, dq, tau_rep, cmd)  with
  gyro    (3,)  body angular rate [rad/s]                       (IMU)
  grav    (3,)  gravity direction in the body frame, unit, = -R[2,:]  (from the IMU attitude; (0,0,-1) when level)
  acc     (3,)  specific force in the body frame [m/s^2]        (IMU accelerometer; (0,0,+9.81) when level and still)
  q       (12,) drive positions [rad], ABSOLUTE, order RL,RR,FR,FL x (first=abduction, second=hip, third=knee lever)
  dq      (12,) drive velocities [rad/s]
  tau_rep (12,) torque the drives REPORTED during the step that just ended [Nm, reported units] (0 on the first step)
  cmd     (3,)  vx [m/s], vy [m/s], wz [rad/s]
and send to the drives:  position target qc (12, rad, absolute), feed-forward torque tau_ff (12, Nm, TRUE units; divide by
S = 2.09 for the reported units the MD80 takes), with PD kp/kd in reported units as stored in the meta (80/60/60, 2).
"""
import json, math, os, pickle, sys, time
import numpy as np

NA, NZ, D, NH = 12, 16, 64, 8


# ------------------------------------------------------------------ numpy networks (apt.py / tnet.py twins)
def _relu(x): return np.maximum(x, 0.0)
def _elu(x): return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
def _gelu(x): return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x * x * x))))
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def mlp(layers, x, act):
    for i, (W, b) in enumerate(layers):
        x = x @ W + b
        if i < len(layers) - 1:
            x = act(x)
    return x


def ln(g, b, x):
    n = x.shape[-1]
    c = x - np.add.reduce(x, -1, keepdims=True) / n
    v = np.add.reduce(c * c, -1, keepdims=True) / n
    return c / np.sqrt(v + 1e-5) * g + b


def tlayer(p, x):
    """x (L, D) -> (L, D); post-LN transformer layer as tnet.layer (batch of one)."""
    L = x.shape[0]
    if "qkv.W" not in p:                                         # fuse q/k/v once (deploy speed)
        p["qkv.W"] = np.concatenate([p["q.W"], p["k.W"], p["v.W"]], 1)
        p["qkv.b"] = np.concatenate([p["q.b"], p["k.b"], p["v.b"]], 0)
    qkv = (x @ p["qkv.W"] + p["qkv.b"]).reshape(L, 3, NH, D // NH)
    q, k, v = (qkv[:, i].transpose(1, 0, 2) for i in range(3))
    a = _softmax(q @ k.transpose(0, 2, 1) / math.sqrt(D // NH))
    h = (a @ v).transpose(1, 0, 2).reshape(L, D)
    x = ln(p["n1.g"], p["n1.b"], x + h @ p["o.W"] + p["o.b"])
    return ln(p["n2.g"], p["n2.b"], x + _gelu(x @ p["f1.W"] + p["f1.b"]) @ p["f2.W"] + p["f2.b"])


def tstack(layers, x):
    for p in layers:
        x = tlayer(p, x)
    return x


def recon(p, z):
    """z (16,) -> 3 normalised feature frames (3, 69): t, t+1, t+2."""
    x = (z @ p["z.W"] + p["z.b"])[None] + p["qry"]
    return tstack(p["L"], x) @ p["out.W"] + p["out.b"]


def decoder(p, z):
    """z (16,) -> normalised torque (12,)."""
    x = np.concatenate([(z @ p["z.W"] + p["z.b"])[None], p["qry"]], 0)
    return (tstack(p["L"], x)[1:] @ p["out.W"] + p["out.b"])[:, 0]


def conv1d_valid_stride2(h, W, b):
    """h (T, C) -> (T', C), kernel W (4, C, C) as lax.conv_general_dilated NWC/WIO with stride 2, VALID."""
    T = h.shape[0]
    To = (T - 4) // 2 + 1
    # Vectorised (Pi deployment): windows (To, 4, C) -> one tensordot.
    idx = 2 * np.arange(To)[:, None] + np.arange(4)[None, :]
    return np.tensordot(h[idx], W, axes=([1, 2], [0, 1])) + b


# ------------------------------------------------------------------ controller
class AptNumpy:
    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=False)
        self.m = json.loads(str(z["meta_json"]))
        g = lambda k: np.asarray(z[k], np.float64)
        m = self.m
        self.H, self.F, self.NCUR, self.GE = m["H"], m["F"], m["NCUR"], m["GE"]
        self.home, self.lo, self.hi = g("home"), g("lo"), g("hi")
        self.kp_true, self.kd_true, self.S = g("kp_true"), g("kd_true"), m["S"]
        self.a_scale, self.anchor_mix = m["a_scale"], m["anchor_mix"]
        self.posonly = m["act_mode"] == "posonly"
        self.ns_mean, self.ns_std = g("ns_mean"), np.sqrt(g("ns_var") + 1e-8)
        self.enc = dict(fW=g("enc.f.W"), fb=g("enc.f.b"), c1W=g("enc.c1.W"), c1b=g("enc.c1.b"), c2W=g("enc.c2.W"), c2b=g("enc.c2.b"),
                        oW=g("enc.o.W"), ob=g("enc.o.b"))
        self.est = [(g(f"est.{i}.W"), g(f"est.{i}.b")) for i in range(m["n_est"])]
        self.act = [(g(f"act.{i}.W"), g(f"act.{i}.b")) for i in range(m["n_act"])]
        tl = lambda pre: [{k: g(f"{pre}.L.{i}.{k}") for k in ("q.W", "q.b", "k.W", "k.b", "v.W", "v.b", "o.W", "o.b", "f1.W", "f1.b",
                                                           "f2.W", "f2.b", "n1.g", "n1.b", "n2.g", "n2.b")} for i in range(2)]
        self.rec = {"z.W": g("rec.z.W"), "z.b": g("rec.z.b"), "qry": g("rec.qry"), "out.W": g("rec.out.W"), "out.b": g("rec.out.b"), "L": tl("rec")}
        self.dec = {"z.W": g("dec.z.W"), "z.b": g("dec.z.b"), "qry": g("dec.qry"), "out.W": g("dec.out.W"), "out.b": g("dec.out.b"), "L": tl("dec")}
        self.feat_mu, self.feat_sd, self.tau_scale = g("feat_mean"), g("feat_std"), g("tau_scale")
        self.QSL = slice(21, 33)
        self.sym = bool(m.get("sym_net", 0))       # actor mirror-equivariant by construction: mu(o) = (f(o) + M f(M o)) / 2
        if self.sym:
            self.obs_perm, self.obs_sign = np.asarray(z["obs_perm"], int), g("obs_sign")
            self.act_perm, self.act_sign = np.asarray(z["act_perm"], int), g("act_sign")
        self.lpf = float(m.get("act_lpf", 0.0))    # first-order low-pass on the drive commands (qc, ff), as in training
        self.reset()

    def reset(self):
        self.t = 0
        self.qb = self.qdb = None
        self.taub = np.zeros((3, NA)); self.auxb = np.zeros((3, NA)); self.zb = np.zeros((3, NZ))
        self.gsel = np.zeros(10); self.gout = np.full(10, 0.5); self.psi = 0.5; self.g = 0
        self.hist = None
        self.qc_prev, self.ff_prev = self.home.copy(), np.zeros(NA)

    # --- networks
    def policy(self, obs):
        mu, est = self._policy(obs)
        if self.sym:
            mum, _ = self._policy(self.obs_sign * obs[self.obs_perm])
            mu = 0.5 * (mu + self.act_sign * mum[self.act_perm])
        return mu, est

    def _policy(self, obs):
        xn = np.clip((obs - self.ns_mean) / self.ns_std, -10, 10)
        cur, seq = xn[:self.NCUR], xn[self.NCUR:].reshape(self.H, self.F)
        e = self.enc
        h = _elu(seq @ e["fW"] + e["fb"])
        h = _elu(conv1d_valid_stride2(h, e["c1W"], e["c1b"]))
        h = _elu(conv1d_valid_stride2(h, e["c2W"], e["c2b"]))
        zf = _elu(h.reshape(-1) @ e["oW"] + e["ob"])
        est = mlp(self.est, np.concatenate([cur, zf]), _elu)
        ein = np.concatenate([est[:8], _sigmoid(est[8:])])
        return mlp(self.act, np.concatenate([ein, cur]), _relu), est

    def prior_pose(self, z):
        f1 = recon(self.rec, z)[1] * self.feat_sd + self.feat_mu
        return self.home + self.anchor_mix * f1[self.QSL]

    def tau_dec(self, z):
        return decoder(self.dec, z) * self.tau_scale

    # --- one control step
    def step(self, gyro, grav, acc, q_abs, dq, tau_rep, cmd):
        q = np.asarray(q_abs, np.float64) - self.home
        dq = np.asarray(dq, np.float64)
        cmd = np.asarray(cmd, np.float64)
        if self.m.get("cmd_max"):  # hard command limits [vx, vy, wz] baked into the pack (e.g. wz 0.5 rad/s: Michal 2026-11-10, turning too fast)
            cmd = np.clip(cmd, -np.asarray(self.m["cmd_max"]), np.asarray(self.m["cmd_max"]))
        p3 = lambda b, x: np.concatenate([x[None], b[:-1]], 0)
        self.qb = np.tile(q[None], (3, 1)) if self.qb is None else p3(self.qb, q)
        self.qdb = np.tile(dq[None], (3, 1)) if self.qdb is None else p3(self.qdb, dq)
        if self.t > 0:
            self.taub = p3(self.taub, np.asarray(tau_rep, np.float64))
        fr = np.concatenate([gyro, grav, acc, q, dq, self.auxb[0]])
        self.hist = np.tile(fr[None], (self.H, 1)) if self.hist is None else np.concatenate([fr[None], self.hist[:-1]], 0)
        tick = float(self.t % self.GE == 0)
        cur = np.concatenate([gyro, cmd, grav, q, dq, self.qb[2], self.qdb[2], self.taub[0], self.taub[2], self.auxb[0], self.auxb[2],
                              self.gsel, self.gout, self.zb[0], self.zb[2], [tick], [self.psi]])
        a, est = self.policy(np.concatenate([cur, self.hist.reshape(-1)]))
        psi = _sigmoid(a[NZ + NA])
        g = 0 if self.m["gait"] == "trot" else (1 if self.m["gait"] == "bound" else (int(psi >= 0.5) if tick else self.g))
        z, aux = a[:NZ], a[NZ:NZ + NA] * self.m["aux"]
        td = self.tau_dec(z)                                    # true Nm
        qref = self.prior_pose(z) + self.a_scale * aux
        if self.posonly:
            qc, ff = np.clip(qref + td / self.kp_true, self.lo, self.hi), np.zeros(NA)
        else:
            qc, ff = np.clip(qref, self.lo, self.hi), td
        if self.lpf > 0:
            qc = self.qc_prev + self.lpf * (qc - self.qc_prev)
            ff = self.ff_prev + self.lpf * (ff - self.ff_prev)
        self.qc_prev, self.ff_prev = qc.copy(), ff.copy()
        if tick:
            self.gout = np.concatenate([[psi], self.gout[:-1]])
            self.gsel = np.concatenate([[float(g)], self.gsel[:-1]])
        self.psi, self.g, self.zb = psi, g, p3(self.zb, z)
        self.auxb = p3(self.auxb, aux)
        self.t += 1
        self.last = dict(z=z, aux=aux, td=td, est=est, qref=qref)
        return qc, ff


# ------------------------------------------------------------------ export
def _flat(d, pre, out):
    for k, v in d.items():
        kk = f"{pre}.{k}" if pre else k
        if isinstance(v, dict):
            _flat(v, kk, out)
        elif isinstance(v, (list, tuple)):
            for i, vi in enumerate(v):
                _flat(vi if isinstance(vi, dict) else {"": vi}, f"{kk}.{i}", out) if isinstance(vi, dict) else out.__setitem__(f"{kk}.{i}", np.asarray(vi))
        else:
            out[kk] = np.asarray(v)


def export(policy_pkl, prior_pkl, out_npz):
    b = pickle.load(open(policy_pkl, "rb"))
    P = pickle.load(open(prior_pkl, "rb"))
    cfg = b["cfg"]
    gait = cfg["gait"]
    assert gait in ("trot", "bound"), "export supports a fixed gait (the Pi runs one decoder)"
    arrs = {}
    pr = b["params"]
    for k, v in pr["enc"].items():
        arrs[f"enc.{k}.W"], arrs[f"enc.{k}.b"] = np.asarray(v["W"]), np.asarray(v["b"])
    for name in ("est", "act"):
        for i, l in enumerate(pr[name]):
            arrs[f"{name}.{i}.W"], arrs[f"{name}.{i}.b"] = np.asarray(l["W"]), np.asarray(l["b"])
    for name, net in (("rec", P["rec"]), ("dec", P["dec"][gait])):
        arrs[f"{name}.z.W"], arrs[f"{name}.z.b"], arrs[f"{name}.qry"] = np.asarray(net["z"]["W"]), np.asarray(net["z"]["b"]), np.asarray(net["qry"])
        arrs[f"{name}.out.W"], arrs[f"{name}.out.b"] = np.asarray(net["out"]["W"]), np.asarray(net["out"]["b"])
        for i, L in enumerate(net["L"]):
            for blk in ("q", "k", "v", "o", "f1", "f2"):
                arrs[f"{name}.L.{i}.{blk}.W"], arrs[f"{name}.L.{i}.{blk}.b"] = np.asarray(L[blk]["W"]), np.asarray(L[blk]["b"])
            for n_ in ("n1", "n2"):
                arrs[f"{name}.L.{i}.{n_}.g"], arrs[f"{name}.L.{i}.{n_}.b"] = np.asarray(L[n_]["g"]), np.asarray(L[n_]["b"])
    arrs.update(ns_mean=np.asarray(b["ns"]["mean"]), ns_var=np.asarray(b["ns"]["var"]), home=np.asarray(b["home"]), lo=np.asarray(b["lo"]),
                hi=np.asarray(b["hi"]), kp_true=np.asarray(b["kp"]), kd_true=np.asarray(b["kd"]), feat_mean=np.asarray(P["feat_mean"]),
                feat_std=np.asarray(P["feat_std"]), tau_scale=np.asarray(P["tau_scale"]))
    if cfg.get("sym_net"):
        import mirror_wrap
        od, zs = int(b["NCUR"] + b["H"] * b["F"]), np.array([1.0] * 8 + [-1.0] * 8)
        ro = mirror_wrap.mirror_obs(np.arange(1, od + 1, dtype=np.float64), zsign=zs)
        ra = mirror_wrap.mirror_act(np.arange(1, NZ + NA + 2, dtype=np.float64), zsign=zs)
        arrs.update(obs_perm=np.abs(ro).astype(np.int64) - 1, obs_sign=np.sign(ro), act_perm=np.abs(ra).astype(np.int64) - 1, act_sign=np.sign(ra))
    cmax = [float(x) for x in os.environ["CMD_MAX"].split(",")] if os.environ.get("CMD_MAX") else None
    meta = dict(name=cfg["name"] + (os.environ.get("NAME_SUFFIX", "")), cmd_max=cmax, sym_net=int(cfg.get("sym_net", 0)), act_lpf=float(cfg.get("act_lpf", 0.0)), kind="apt", gait=gait, act_mode=cfg["act_mode"], pd_anchor=cfg["pd_anchor"], anchor_mix=cfg["anchor_mix"],
                a_scale=cfg["a_scale"], aux=cfg["aux"], S=cfg["S"], kp_rep=cfg["kp_rep"], kd_rep=cfg["kd_rep"], cap_rep=cfg["cap_rep"],
                ctrl_dt=cfg["ctrl_dt"], H=int(b["H"]), F=int(b["F"]), NCUR=int(b["NCUR"]), GE=int(b["GE"]), NZ=NZ, NA=NA,
                n_est=len(pr["est"]), n_act=len(pr["act"]), obs_dim=int(b["NCUR"] + b["H"] * b["F"]),
                joint_order=["RL_first", "RL_second", "RL_third", "RR_first", "RR_second", "RR_third",
                             "FR_first", "FR_second", "FR_third", "FL_first", "FL_second", "FL_third"],
                train_delay_ms=[cfg["lat_sub"][0] * cfg["dt"] * 1000, cfg["lat_sub"][1] * cfg["dt"] * 1000],
                prior_morphology=P.get("morphology_hash", ""), feat_names=list(P.get("feat_names", [])),
                notes="q in obs = q_abs - home; tau_rep in obs = reported Nm (true / S); tau_ff output in TRUE Nm (send tau_ff / S to the MD80).")
    arrs["meta_json"] = np.array(json.dumps(meta))
    np.savez(out_npz, **arrs)
    json.dump(meta, open(out_npz.replace(".npz", ".json"), "w"), indent=1)
    print("wrote", out_npz, len(arrs), "arrays; obs", meta["obs_dim"])


def selftest(npz):
    ctl = AptNumpy(npz)
    rng = np.random.default_rng(0)
    t0 = time.time()
    tau_rep = np.zeros(NA)
    for i in range(100):
        qc, ff = ctl.step(rng.normal(0, 0.05, 3), np.array([0, 0, -1.0]), np.array([0, 0, 9.81]), ctl.home + rng.normal(0, 0.02, NA),
                          rng.normal(0, 0.2, NA), tau_rep, np.array([0.5, 0.0, 0.0]))
        tau_rep = ff / ctl.S
    dt = (time.time() - t0) / 100
    print(f"selftest ok: {dt * 1000:.2f} ms/step, qc-home range {np.round(qc - ctl.home, 2).tolist()}, tau_ff {np.round(ff, 2).tolist()}")


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export(*sys.argv[2:5])
    elif sys.argv[1] == "selftest":
        selftest(sys.argv[2])
