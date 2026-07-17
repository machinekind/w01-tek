"""Isaac Sim port of the MuJoCo click-to-walk demo (training/demo/app.py).

Same browser protocol as the MuJoCo app:
  client -> {"type":"target","x":..,"y":..} | {"type":"reset"}
            {"type":"command","text":"turn_left 15|forward 0.5|stop"}
  server -> per tick {"type":"state", x,y,yaw,tx,ty,reached,dist,cmd,
                      [frame b64 chase], [ego b64 bench], [exec status]}

Policy is ALWAYS on. Robot auto-recovers on falls. Nav via
navigation.command_to_target (verbatim copy of the MuJoCo one).
MAIN thread = Isaac sim loop; uvicorn daemon thread; queue between.
"""
import os, sys, json, math, time, queue, threading, base64
import numpy as np

from isaacsim import SimulationApp
CONFIG = {"headless": True, "renderer": "RayTracedLighting", "width": 1152, "height": 648}
sim = SimulationApp(CONFIG)

import cv2
import omni.usd
import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdLux, UsdPhysics, UsdGeom, UsdShade, Gf

sys.path.insert(0, os.path.expanduser("~/wojtek_asset/policy"))
from policy import WojtekPolicy
from navigation import NavConfig, command_to_target, quat_to_yaw

USD = os.path.expanduser("~/wojtek_asset/wojtek_mjx.usd")
ART_ROOT = "/World/Wojtek/root/root"
W, H = 896, 504           # chase/orbit stream resolution (native, no resize)
SPAWN = np.array([0.0, 0.0, 0.135])
CAM_OFF = np.array([1.6, 1.6, 0.9])
FOOT_RADIUS = 0.046
LEGS = ("rear_left", "rear_right", "front_right", "front_left")
WORLD_HALF = 5.0
CTRL_DT = 0.02
# Render every 2nd tick: EXACTLY even 40 ms frame spacing (25 fps). Even
# pacing looks smoother than more-but-jittery frames; DLAA render time
# (~43 ms) can't sustain 30+ unique frames anyway.
TARGET_FPS = 25.0
FPS_FLOOR = 25.0

# cheaper RTX: drop reflections/GI/AO — big fps win, mild visual cost in a
# dome+distant-lit scene. DLSS perf mode upscales from a lower internal res.
import carb.settings
_s = carb.settings.get_settings()
for _k, _v in [("/rtx/reflections/enabled", False),
               ("/rtx/indirectDiffuse/enabled", False),
               ("/rtx/ambientOcclusion/enabled", False),
               ("/rtx/raytracing/subsurface/enabled", False),
               ("/rtx/post/aa/op", 4),           # DLAA: native-res temporal AA,
                                                 # no DLSS upscale smear at 360p
               # preload ALL textures: streaming shows grey "loading"
               # placeholders whenever the robot walks into a new area
               ("/rtx-transient/resourcemanager/enableTextureStreaming", False),
               # GI is off, so lift pure-black shadow areas with flat ambient
               ("/rtx/sceneDb/ambientLightIntensity", 0.35),
               # sync rendering: async returns stale frames to the annotator
               # (~27% duplicates measured) -> judder in the stream
               ("/app/asyncRendering", False),
               ("/app/asyncRenderingLowLatency", False)]:
    _s.set(_k, _v)

# ---- stage ------------------------------------------------------------------
assets = get_assets_root_path()
ctx = omni.usd.get_context()
world = World(stage_units_in_meters=1.0, physics_dt=0.004, rendering_dt=0.02)
stage = ctx.get_stage()
UsdLux.DomeLight.Define(stage, "/World/DomeExtra").CreateIntensityAttr(400.0)
add_reference_to_stage(
    assets + "/Isaac/Environments/Simple_Warehouse/warehouse.usd", "/World/Warehouse")
from isaacsim.core.api.objects.ground_plane import GroundPlane
GroundPlane("/World/GroundBackstop", z_position=0.0, visible=False)

add_reference_to_stage(USD, "/World/Wojtek")
wb = stage.GetPrimAtPath("/World/Wojtek/worldBody")
if wb and wb.HasAPI(UsdPhysics.ArticulationRootAPI):
    wb.RemoveAPI(UsdPhysics.ArticulationRootAPI)

# recreate training contact model (importer drops all collision geoms)
mat = UsdShade.Material.Define(stage, "/World/FootMaterial")
pmat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
pmat.CreateStaticFrictionAttr(1.0); pmat.CreateDynamicFrictionAttr(1.0)
for leg in LEGS:
    body = stage.GetPrimAtPath(f"/World/Wojtek/root/{leg}_foot_link")
    sp = UsdGeom.Sphere.Define(stage, body.GetPath().AppendChild("foot_collider"))
    sp.CreateRadiusAttr(FOOT_RADIUS)
    UsdPhysics.CollisionAPI.Apply(sp.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(sp.GetPrim()).Bind(mat, materialPurpose="physics")
    UsdGeom.Imageable(sp.GetPrim()).MakeInvisible()
base = stage.GetPrimAtPath("/World/Wojtek/root/root")
bx = UsdGeom.Cube.Define(stage, base.GetPath().AppendChild("base_collider"))
bx.CreateSizeAttr(1.0)
UsdGeom.Xformable(bx).AddScaleOp().Set(Gf.Vec3f(0.14, 0.06, 0.045))
UsdPhysics.CollisionAPI.Apply(bx.GetPrim())
UsdGeom.Imageable(bx.GetPrim()).MakeInvisible()

robot = world.scene.add(SingleArticulation(prim_path=ART_ROOT, name="wojtek", position=SPAWN))
world.reset()

# ---- cameras ----------------------------------------------------------------
def make_cam(path, focal=None, haperture=None, vaperture=None):
    prim = UsdGeom.Camera.Define(stage, path)
    if focal: prim.CreateFocalLengthAttr(focal)
    if haperture: prim.CreateHorizontalApertureAttr(haperture)
    if vaperture: prim.CreateVerticalApertureAttr(vaperture)
    prim.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
    xf = UsdGeom.Xformable(prim.GetPrim())
    xf.ClearXformOpOrder()
    return prim, xf.AddTranslateOp(), xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)

def look_at_quat(eye, target, up=np.array([0.0, 0.0, 1.0])):
    fwd = np.asarray(target, float) - np.asarray(eye, float)
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    z = -fwd
    x = np.cross(up, z); x /= max(np.linalg.norm(x), 1e-9)
    y = np.cross(z, x)
    m = Gf.Matrix3d(x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2])
    q = m.ExtractRotation().GetQuat()
    return Gf.Quatf(q.GetReal(), Gf.Vec3f(*[float(v) for v in q.GetImaginary()]))

_, chase_t, chase_o = make_cam("/World/ChaseCam")
_, bench_t, bench_o = make_cam("/World/BenchCam", 10.0, 20.0, 15.0)  # VLN-CE 90deg
chase_t.Set(Gf.Vec3d(*(SPAWN + CAM_OFF))); chase_o.Set(look_at_quat(SPAWN + CAM_OFF, SPAWN))

chase_rp = rep.create.render_product("/World/ChaseCam", (W, H))
chase_rgb = rep.AnnotatorRegistry.get_annotator("rgb"); chase_rgb.attach(chase_rp)
bench_rp = rep.create.render_product("/World/BenchCam", (800, 600))
bench_rgb = rep.AnnotatorRegistry.get_annotator("rgb"); bench_rgb.attach(bench_rp)

# NOTE: gating the bench product via hydra_texture.set_updates_enabled starves
# the annotator (one enabled step yields no frame — pipeline latency) and
# measurements showed no fps gain from gating. Keep it always-on; only the
# ENCODE runs at 5 Hz.

CHASE_DIST = 2.4      # orbit radius (m) from the robot base
CHASE_PITCH = 0.26    # elevation angle (rad); ~15deg above horizontal, behind
CHASE_LOOK_Z = 0.25   # aim above the base: flatter angle keeps the horizon in frame
CHASE_YAW_ALPHA = 0.08  # heading EMA — slower than position; turns are sharp
# orbit limits
DIST_MIN, DIST_MAX = 1.3, 6.0
PITCH_MIN, PITCH_MAX = -0.12, 1.35

_cam_smooth = {"p": None, "hd": None}
# user-controllable orbit around the robot; yaw is an offset from "behind".
_orbit = {"yaw": 0.0, "pitch": CHASE_PITCH, "dist": CHASE_DIST}

def reset_orbit():
    _orbit["yaw"] = 0.0; _orbit["pitch"] = CHASE_PITCH; _orbit["dist"] = CHASE_DIST

def apply_orbit(msg):
    if msg.get("reset"):
        reset_orbit(); return
    _orbit["yaw"] += float(msg.get("dyaw", 0.0))
    _orbit["pitch"] = min(PITCH_MAX, max(PITCH_MIN, _orbit["pitch"] + float(msg.get("dpitch", 0.0))))
    _orbit["dist"] = min(DIST_MAX, max(DIST_MIN, _orbit["dist"] * float(msg.get("zoom", 1.0))))

def move_cams(pos, quat):
    p = np.asarray(pos, float)
    # low-pass the chase target: the base oscillates with the trot, and a
    # hard-locked camera turns that into shake + DLSS smear.
    sp = _cam_smooth["p"]
    sp = p.copy() if sp is None else 0.88 * sp + 0.12 * p
    _cam_smooth["p"] = sp
    w, x, y, z = [float(v) for v in quat]
    yaw = quat_to_yaw(w, x, y, z)
    # smooth heading as a unit vector — EMA on the angle itself breaks at the
    # +/-pi wrap, the vector form doesn't.
    hd = _cam_smooth["hd"]
    h_now = np.array([math.cos(yaw), math.sin(yaw)])
    hd = h_now if hd is None else (1.0 - CHASE_YAW_ALPHA) * hd + CHASE_YAW_ALPHA * h_now
    hd /= max(np.linalg.norm(hd), 1e-9)
    _cam_smooth["hd"] = hd
    # orbit around the base: azimuth = behind the heading + user yaw offset.
    az = math.atan2(hd[1], hd[0]) + math.pi + _orbit["yaw"]
    pit, dist = _orbit["pitch"], _orbit["dist"]
    horiz = dist * math.cos(pit)
    eye = sp + np.array([horiz * math.cos(az), horiz * math.sin(az),
                         dist * math.sin(pit)])
    eye[2] = max(eye[2], 0.12)  # never dip below the floor
    chase_t.Set(Gf.Vec3d(*eye))
    chase_o.Set(look_at_quat(eye, sp + np.array([0.0, 0.0, CHASE_LOOK_Z])))
    eye = np.array([p[0], p[1], 1.25])
    bench_t.Set(Gf.Vec3d(*eye))
    bench_o.Set(look_at_quat(eye, eye + np.array([math.cos(yaw), math.sin(yaw), 0.0])))

# ---- policy + physics parity ------------------------------------------------
pol = WojtekPolicy(os.path.expanduser("~/wojtek_asset/policy/policy.npz"))
# height packed into the 4-dim command: fbb runtime exposed default_height,
# the springy runtime renamed it command_height.
POLICY_HEIGHT = float(getattr(pol, "command_height",
                              getattr(pol, "default_height", 0.125)))
dof_names = list(robot.dof_names); ndof = robot.num_dof
act_map = np.array([dof_names.index(n) for n in pol.joint_names])
ctrl = robot.get_articulation_controller()
kps = np.zeros(ndof); kds = np.full(ndof, 0.05)
kps[act_map] = 20.0; kds[act_map] = 1.0
ctrl.set_gains(kps=kps, kds=kds)
ctrl.set_max_efforts(np.full(ndof, 9.0))
view = robot._articulation_view
with open(os.path.expanduser("~/wojtek_asset/policy/body_masses.json")) as f:
    mj_mass = json.load(f)
masses = np.asarray(view.get_body_masses()).reshape(-1)
for i, bn in enumerate(view.body_names):
    if bn in mj_mass: masses[i] = mj_mass[bn]
view.set_body_masses(masses.reshape(1, -1))
view.set_solver_position_iteration_counts(np.array([32]))
view.set_solver_velocity_iteration_counts(np.array([8]))
with open(os.path.expanduser("~/wojtek_asset/policy/home_qpos.json")) as f:
    _hm = json.load(f)
home_full = np.zeros(ndof)
for n, q in _hm.items():
    home_full[dof_names.index(n)] = float(q)
print("PARITY_OK mass=%.2f" % masses.sum(), flush=True)

# ---- nav / exec state (sim-thread only) --------------------------------------
navcfg = NavConfig(vx_max=0.6, vy_max=0.3, yaw_max=0.9, stop_radius=0.25)
target = None            # (tx, ty) or None
exec_queue = []          # discrete commands
exec_active = None       # dict(kind, amount, ref, deadline)
cmd_now = (0.0, 0.0, 0.0)

def reset_robot():
    global target, exec_queue, exec_active, cmd_now
    target = None; exec_queue = []; exec_active = None; cmd_now = (0.0, 0.0, 0.0)
    _cam_smooth["p"] = None; _cam_smooth["hd"] = None
    pol.reset()
    world.reset()
    ctrl.set_gains(kps=kps, kds=kds)
    robot.set_world_pose(position=SPAWN, orientation=np.array([1.0, 0, 0, 0]))
    robot.set_joint_positions(home_full)
    robot.set_joint_velocities(np.zeros(ndof))
    ctrl.set_gains(kps=kps * 10.0, kds=kds * 5.0)
    ctrl.apply_action(ArticulationAction(joint_positions=pol.home_ctrl, joint_indices=act_map))
    for _ in range(100): world.step(render=False)
    ctrl.set_gains(kps=kps, kds=kds)
    for _ in range(50): world.step(render=False)
    print("RESET_DONE", flush=True)

def parse_command(text):
    """'turn_left 15' | 'turn_right 30' | 'forward 0.5' | 'backward 0.3' | 'stop'"""
    parts = text.strip().lower().split()
    if not parts: raise ValueError("empty")
    kind = parts[0]
    if kind == "stop": return {"kind": "stop"}
    if kind in ("turn_left", "turn_right"):
        deg = float(parts[1]) if len(parts) > 1 else 15.0
        return {"kind": kind, "amount": math.radians(deg)}
    if kind in ("forward", "backward"):
        d = float(parts[1]) if len(parts) > 1 else 0.5
        return {"kind": kind, "amount": d}
    raise ValueError(f"unknown command {kind!r}")

def exec_step(x, y, yaw):
    """Velocity command for the active discrete command; pops when done."""
    global exec_active
    if exec_active is None and exec_queue:
        c = exec_queue.pop(0)
        ref = yaw if c["kind"].startswith("turn") else (x, y)
        exec_active = {**c, "ref": ref, "deadline": time.monotonic() + 15.0}
    if exec_active is None: return None
    c = exec_active
    if time.monotonic() > c["deadline"]:
        exec_active = None; return None
    k = c["kind"]
    if k.startswith("turn"):
        turned = abs((yaw - c["ref"] + math.pi) % (2 * math.pi) - math.pi)
        if turned >= c["amount"] - 0.05:
            exec_active = None; return None
        wz = 0.7 if k == "turn_left" else -0.7
        return (0.0, 0.0, wz, c["amount"] - turned)
    else:
        gone = math.hypot(x - c["ref"][0], y - c["ref"][1])
        if gone >= c["amount"] - 0.03:
            exec_active = None; return None
        vx = 0.4 if k == "forward" else -0.3
        return (vx, 0.0, 0.0, c["amount"] - gone)

# ---- shared with web thread ---------------------------------------------------
action_q = queue.Queue()
shared = {"state": None, "frame": None, "ego": None}   # frame/ego = b64 str

# JPEG encode + b64 off the sim thread (cv2 releases the GIL while encoding).
enc_q = queue.Queue(maxsize=3)
def _encoder():
    while True:
        kind, arr, q = enc_q.get()
        try:
            bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok:
                shared[kind] = base64.b64encode(buf.tobytes()).decode()
        except Exception as e:
            print("ENC_ERR", e, flush=True)
threading.Thread(target=_encoder, daemon=True).start()

_last_fp = {}
def submit_encode(kind, data, q):
    # skip stale frames: the RTX pipeline sometimes returns the previous
    # image again (async rendering) — re-encoding it just adds judder.
    arr = np.asarray(data)
    fp = arr[::48, ::48].tobytes()
    if _last_fp.get(kind) == fp:
        return
    _last_fp[kind] = fp
    try:
        enc_q.put_nowait((kind, np.array(arr, copy=True), q))
    except queue.Full:
        pass   # encoder busy: drop this frame, newer one comes next tick

# ---- web thread ---------------------------------------------------------------
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio

app = FastAPI()
HTML_PATH = os.path.expanduser("~/isaac_room.html")

@app.get("/")
async def index():
    with open(HTML_PATH) as f: return HTMLResponse(f.read())

@app.get("/api/info")
async def info():
    return JSONResponse({"world_half": WORLD_HALF, "backend": "isaac", "scene": "warehouse"})

@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    async def reader():
        try:
            while True:
                msg = json.loads(await sock.receive_text())
                if msg.get("type") == "command":
                    try:
                        parse_command(msg.get("text", ""))
                        await sock.send_text(json.dumps(
                            {"type": "command_ack", "ok": True, "command": msg["text"]}))
                    except Exception as e:
                        await sock.send_text(json.dumps(
                            {"type": "command_ack", "ok": False, "error": str(e)}))
                        continue
                if msg.get("type") == "goal":
                    await sock.send_text(json.dumps(
                        {"type": "goal_ack", "ok": False,
                         "error": "VLM nav not wired in Isaac yet"}))
                    continue
                action_q.put(msg)
        except Exception:
            pass
    rt = asyncio.create_task(reader())
    sent_frame = None
    sent_ego = None
    last_state_t = 0.0
    try:
        while True:
            st = shared["state"]
            f = shared["frame"]
            now = asyncio.get_event_loop().time()
            new_frame = f is not None and f is not sent_frame
            if st is not None and (new_frame or now - last_state_t > 0.1):
                payload = dict(st)
                if new_frame:
                    payload["frame"] = f
                    sent_frame = f
                    e = shared["ego"]
                    if e is not None and e is not sent_ego:
                        payload["ego"] = e
                        sent_ego = e
                await sock.send_text(json.dumps(payload))
                last_state_t = now
            await asyncio.sleep(0.005)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        rt.cancel()

def run_web():
    import uvicorn
    port = int(os.environ.get("WOJTEK_PORT", "8200"))
    print(f"SERVER_START :{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, ws="wsproto", log_level="warning")

# ---- init + main loop -----------------------------------------------------------
# wait until all referenced assets/textures finish loading before serving —
# rendering during streaming shows grey placeholder materials.
from isaacsim.core.utils.stage import is_stage_loading
_t0 = time.monotonic()
while is_stage_loading() and time.monotonic() - _t0 < 120:
    sim.update()
print("ASSETS_LOADED in %.1fs" % (time.monotonic() - _t0), flush=True)

reset_robot()
for _ in range(60): world.step(render=True)   # let DLSS/exposure converge too
threading.Thread(target=run_web, daemon=True).start()
print("SIM_LOOP_START", flush=True)

tick = 0
t_start = time.monotonic()
t_wall = t_start
render_acc = 0.0
last_ego_tick = -100
while True:
    while not action_q.empty():
        try:
            msg = action_q.get_nowait()
            t = msg.get("type")
            if t == "target":
                target = (float(msg["x"]), float(msg["y"]))
                exec_queue.clear(); exec_active = None
            elif t == "reset":
                reset_robot()
                t_start = time.monotonic() - tick * CTRL_DT
            elif t == "command":
                c = parse_command(msg.get("text", ""))
                if c["kind"] == "stop":
                    exec_queue.clear(); exec_active = None; target = None
                else:
                    target = None
                    exec_queue.append(c)
            elif t == "cam":
                apply_orbit(msg)
        except Exception as e:
            print("HANDLE_ERR", e, flush=True)

    pos, quat = robot.get_world_pose()
    x, yv, z = [float(v) for v in pos]
    w, qx, qy, qz = [float(v) for v in quat]
    yaw = quat_to_yaw(w, qx, qy, qz)

    # fall detection -> auto recover (body z-axis vs world up, or base too low)
    up = 1.0 - 2.0 * (qx * qx + qy * qy)
    if z < 0.05 or up < 0.4:
        print("AUTO_RECOVER z=%.3f up=%.2f" % (z, up), flush=True)
        tgt_keep = target
        reset_robot()
        target = tgt_keep
        t_start = time.monotonic() - tick * CTRL_DT
        continue

    # choose command: discrete exec > click target > stand
    reached = False; dist = 0.0
    es = exec_step(x, yv, yaw)
    if es is not None:
        cmd_now = es[:3]
    elif target is not None:
        # target persists after arrival (matches the MuJoCo demo): reached
        # re-evaluates every tick, a new click or stop/reset clears it.
        vx, vy, wz, dist, reached = command_to_target(x, yv, yaw, *target, navcfg)
        cmd_now = (vx, vy, wz)
    else:
        cmd_now = (0.0, 0.0, 0.0)

    cmd4 = np.array([*cmd_now, POLICY_HEIGHT], np.float32)
    jp = robot.get_joint_positions()[act_map]
    jv = robot.get_joint_velocities()[act_map]
    targets_q = pol.step(None, None, jp, jv, cmd4)
    ctrl.apply_action(ArticulationAction(joint_positions=targets_q, joint_indices=act_map))

    # fractional render cadence: accumulate TARGET_FPS worth of renders over
    # the 50 Hz tick grid (e.g. 30 fps = render on 3 of every 5 ticks).
    render_acc += TARGET_FPS * CTRL_DT
    do_render = render_acc >= 1.0
    if do_render:
        render_acc -= 1.0
    want_ego = do_render and (tick - last_ego_tick) >= 5   # ~10 Hz ego
    if do_render:
        move_cams(pos, quat)
    world.step(render=do_render)

    if True:   # state dict is cheap; keep the minimap as fresh as the stream
        exec_stat = None
        if exec_active is not None:
            exec_stat = {"active": exec_active["kind"], "remaining": round(float(0 if es is None else es[3]), 2),
                         "queued": len(exec_queue)}
        elif exec_queue:
            exec_stat = {"active": "queued", "remaining": 0, "queued": len(exec_queue)}
        shared["state"] = {
            "type": "state", "x": x, "y": yv, "yaw": yaw,
            "tx": None if target is None else target[0],
            "ty": None if target is None else target[1],
            "reached": bool(reached), "dist": float(dist),
            "cmd": [round(c, 3) for c in cmd_now],
            "exec": exec_stat, "mode": "isaac-warehouse",
        }
    if do_render:
        data = chase_rgb.get_data()
        if data is not None and np.asarray(data).size:
            submit_encode("frame", data, 80)
        if want_ego:         # ego (VLM view) at ~5 Hz is plenty
            data = bench_rgb.get_data()
            if data is not None and np.asarray(data).size:
                submit_encode("ego", data, 80)
                last_ego_tick = tick

    if tick % 500 == 0:
        now = time.monotonic()
        rtf = (500 * CTRL_DT) / max(now - t_wall, 1e-6) if tick else 0.0
        t_wall = now
        fps = TARGET_FPS * min(rtf, 1.0)
        print("TICK %d xyz=[%.2f %.2f %.3f] cmd %s tgt %s rtf=%.2f fps=%.0f tgtfps=%.0f"
              % (tick, x, yv, z, [round(c, 2) for c in cmd_now], target, rtf, fps,
                 TARGET_FPS), flush=True)
        if tick and rtf < 0.94 and TARGET_FPS > FPS_FLOOR:
            TARGET_FPS = max(FPS_FLOOR, TARGET_FPS - 4.0)
            print("FPS_BACKOFF ->", TARGET_FPS, flush=True)

    tick += 1
    nt = t_start + tick * CTRL_DT
    dt = nt - time.monotonic()
    if dt > 0: time.sleep(dt)
    elif dt < -1.0: t_start = time.monotonic() - tick * CTRL_DT
