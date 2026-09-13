// The deck panel. Two links and a worker:
//   gateway  ws://<host>/ws         commands out (sticks, buttons), status in
//   bridge   ws://<host>:<bridge>   telemetry in, decoded from CDR (bridge.js)
//   detector det_worker.js          YOLOX on this machine, fed frames off
//                                   the camera image already on screen
// The camera is the gateway's MJPEG stream in a plain <img>, filling the
// screen; the reticle, horizon and detections are drawn on the overlay
// canvas above it, and the instruments are laid over both.
//
// A tap on the picture locks onto the box under the finger (lock.js). The
// lock is held from frame to frame and sent to the gateway ten times a
// second as {t:"track"}, which is what the robot will follow. Sticks, stop
// and a second tap on the target let go.
//
// Query parameters: ?telemetry=on shows the instruments the bridge feeds
// (off by default: no bridge, no numbers), ?bridge=<url> for the
// telemetry bridge, ?det=off to
// switch detection off, ?det=cpu or ?det=gpu to pin the detector's backend,
// ?det=<ws url> for a detector in another process, ?detsrc=<url> to point
// the camera and the detector at a still image.
import { Bridge } from "./bridge.js";
import { Bars, Strip } from "./charts.js";
import { Lock, pick, stickMoved, toFrame } from "./lock.js";

// The instruments the telemetry bridge feeds are off unless ?telemetry=on.
const telemetry = new URLSearchParams(location.search).get("telemetry") === "on";
if (telemetry) document.body.classList.add("telemetry");

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const now = () => performance.now() / 1000;
const DASH = [[], [4, 3], [1, 3]];   // series identity: solid, dashed, dotted

// ---- state -----------------------------------------------------------------
let height = 0.125;
let armed = false, policyOn = false;
let rpy = [0, 0, 0];          // latest attitude
let det = null;               // {w, h, boxes, at}
let lock = null;              // the target the operator tapped (lock.js)
let lockNote = null;          // {text, until}: a word left on the HUD after a lock ends

function lamp(name, on, text) {
  const el = $(`lamp-${name}`);
  el.classList.toggle("on", !!on);
  if (text !== undefined) el.textContent = text;
}
function stamp() { return new Date().toTimeString().slice(0, 8); }
function log(text, cls) {
  const el = $("log"), d = document.createElement("div");
  d.textContent = `${stamp()}  ${text}`;
  if (cls) d.className = cls;
  el.prepend(d);
  while (el.children.length > 3) el.lastChild.remove();
}
function setDrive(state) {
  const d = $("drive"), v = $("viewport");
  d.textContent = state === "deadman" ? "dead-man" : state;
  for (const s of ["idle", "live", "deadman"]) { d.classList.toggle(s, state === s); v.classList.toggle(s, state === s); }
}

// ---- gateway ---------------------------------------------------------------
let gw = null, bridge = null;
function connectGateway() {
  gw = new WebSocket(`ws://${location.host}/ws`);
  gw.onopen = () => { lamp("link", true); log("gateway connected", "ok"); };
  gw.onclose = () => {
    lamp("link", false); setDrive("idle");
    // No link, no follow: the robot's own dead-man has already stopped it,
    // and a lock kept here would start it again the moment the link is back.
    if (lock) { lock = null; note("dropped · link down"); }
    setTimeout(connectGateway, 1000);
  };
  gw.onmessage = e => onGateway(JSON.parse(e.data));
}
function send(o) { if (gw && gw.readyState === 1) gw.send(JSON.stringify(o)); }
function onGateway(m) {
  if (m.t === "hello") {
    height = m.height_default;
    $("policy").textContent = m.policy || "";
    // Without ?telemetry=on there is nothing on screen for the bridge to
    // fill, so it is not opened at all -- no socket retrying every second
    // against a robot that has no bridge running.
    if (!bridge && telemetry) startBridge(params.get("bridge") || `ws://${location.hostname}:${m.bridge_port}`);
  } else if (m.t === "avail") {
    for (const b of document.querySelectorAll("[data-call]")) b.disabled = !m.svc[b.dataset.call];
  } else if (m.t === "svc") {
    if (m.success) {
      if (m.key === "arm") { armed = !!m.value; $("btn-arm").textContent = armed ? "armed" : "arm"; $("btn-arm").classList.toggle("armed", armed); }
      if (m.key === "enable") { policyOn = !!m.value; $("btn-enable").classList.toggle("on", policyOn); }
      log(`${m.key}: ${m.message || "ok"}`, "ok");
    } else log(`${m.key} refused: ${m.message}`, "bad");
  } else if (m.t === "status") {
    height = m.height; $("height").textContent = height.toFixed(3);
    setDrive(m.drive);
    lamp("cam", m.cam_hz > 0.5, m.cam_hz > 0.5 ? `cam ${m.cam_hz.toFixed(0)}` : "cam —");
    $("hud-cam").textContent = m.cam_hz > 0.5 ? `fwd cam · ${m.cam_hz.toFixed(0)} fps` : "fwd cam · —";
  }
}

// ---- charts and readouts -------------------------------------------------
const att = new Strip($("ch-att"), { series: [{ name: "r", dash: DASH[0] }, { name: "p", dash: DASH[1] }], min: -45, max: 45, fixed: 1 });
const gyro = new Strip($("ch-gyro"), { series: [{ name: "x", dash: DASH[0] }, { name: "y", dash: DASH[1] }, { name: "z", dash: DASH[2] }] });
const cmd = new Strip($("ch-cmd"), { series: [{ name: "vx", dash: DASH[0] }, { name: "vy", dash: DASH[1] }, { name: "wz", dash: DASH[2] }] });
const eff = new Bars($("ch-eff"), { max: 1 });
let effMax = 1;
const deg = r => r * 180 / Math.PI;

function startBridge(url) {
  bridge = new Bridge(url, {
    "/imu_sensor_broadcaster/imu": m => {
      const q = m.orientation, w = m.angular_velocity;
      const roll = Math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y));
      const pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (q.w * q.y - q.z * q.x))));
      const yaw = Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
      rpy = [roll, pitch, yaw];
      const t = now();
      att.push(t, [deg(roll), deg(pitch)]);
      gyro.push(t, [w.x, w.y, w.z]);
      $("att-r").textContent = `roll ${deg(roll).toFixed(1)}°`;
      $("att-p").textContent = `pitch ${deg(pitch).toFixed(1)}°`;
      $("gyro-z").textContent = `${w.z.toFixed(2)} rad/s`;
    },
    "/cmd_vel": m => {
      cmd.push(now(), [m.linear.x, m.linear.y, m.angular.z]);
      $("speed").textContent = m.linear.x.toFixed(2);
    },
    "/wojtek/policy_timing": m => { $("sys-tick").textContent = `${m.inference_ms.toFixed(1)} ms`; },
    "/wojtek/joint_targets": m => {
      // Torque when the contract has a torque head, otherwise the position
      // targets; the card says which. The largest one is named below.
      const useEffort = m.effort && m.effort.length === m.name.length && m.effort.some(v => v !== 0);
      const vals = useEffort ? m.effort : m.position;
      let k = 0;
      for (let i = 1; i < vals.length; i++) if (Math.abs(vals[i]) > Math.abs(vals[k])) k = i;
      effMax = Math.max(useEffort ? 0.5 : 1, effMax * 0.999, ...vals.map(Math.abs));
      eff.opts.max = effMax;
      eff.set(vals);
      $("eff-title").textContent = useEffort ? "joint effort" : "joint targets";
      $("eff-name").textContent = (m.name[k] || "").replace(/_joint$/, "").replace(/_/g, " ");
      $("eff-val").textContent = `${vals[k].toFixed(2)} ${useEffort ? "N·m" : "rad"}`;
    },
    "/wojtek/sysinfo": m => {
      const cpu = m.cpu_percent.length ? m.cpu_percent.reduce((a, b) => a + b, 0) / m.cpu_percent.length : NaN;
      const tx = m.wifi_tx_bytes_per_s / 1024;
      $("sys-cpu").textContent = `${cpu.toFixed(0)} %`;
      $("sys-soc").textContent = `${m.soc_temp_c.toFixed(0)} °C`;
      $("sys-wifi").textContent = `${tx.toFixed(0)} kB/s`;
      const bar = (id, f, hot) => { const b = $(id); b.style.width = `${Math.max(0, Math.min(100, f * 100))}%`; b.classList.toggle("hot", !!hot); };
      bar("bar-cpu", cpu / 100, cpu > 85);
      bar("bar-soc", (m.soc_temp_c - 20) / 70, m.soc_temp_c > 75);
      bar("bar-wifi", tx / 1500, false);
      // one line for the Pi's power/thermal flags: what is on now, else what was ever seen
      const flags = ["undervoltage", "throttled", "freq_capped", "soft_temp_limit"];
      const nowOn = flags.filter(f => m[`${f}_now`]), ever = flags.filter(f => m[`${f}_ever`]);
      const row = $("sys-flags");
      row.classList.toggle("now", nowOn.length > 0);
      row.classList.toggle("ever", nowOn.length === 0 && ever.length > 0);
      $("sys-flag").textContent = nowOn.length ? nowOn.join(" ").replace(/_/g, "-") + " now"
        : ever.length ? ever.join(" ").replace(/_/g, "-") + " seen" : "ok";
    },
  }, (up) => { lamp("bridge", up); if (up) log("bridge connected", "ok"); });
}

// ---- overlay: reticle with heading, horizon, detections -----------------
const overlay = $("overlay"), cam = $("cam");
function drawOverlay() {
  const dpr = window.devicePixelRatio || 1;
  const W = overlay.clientWidth, H = overlay.clientHeight;
  if (!W || !H) return;
  if (overlay.width !== W * dpr || overlay.height !== H * dpr) { overlay.width = W * dpr; overlay.height = H * dpr; }
  const ctx = overlay.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  // The overlay is drawn in the same colours as the rest of the panel:
  // white at two strengths for everything that just sits there, and the
  // one accent for the thing worth looking at.
  const dim = css("--on-image-2"), faint = css("--on-image-3"), accent = css("--accent-image"), mono = css("--mono");
  const [roll, pitch, yaw] = rpy;
  const cx = W / 2, cy = H / 2, R = Math.min(W, H) * 0.2;

  // reticle: dashed outer ring, inner ring, and an arc that points the way
  // the robot is heading (ROS +yaw is counter-clockwise), centre bars
  const hdg = ((-yaw * 180 / Math.PI) % 360 + 360) % 360;
  ctx.strokeStyle = faint; ctx.lineWidth = 1;
  ctx.setLineDash([3, 9]); ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
  ctx.setLineDash([]); ctx.beginPath(); ctx.arc(cx, cy, R * 0.7, 0, Math.PI * 2); ctx.stroke();
  const a0 = -Math.PI / 2 + hdg * Math.PI / 180;
  ctx.strokeStyle = accent; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(cx, cy, R, a0 - Math.PI / 4, a0 + Math.PI / 4); ctx.stroke();
  ctx.lineWidth = 1.5; ctx.strokeStyle = dim;
  ctx.beginPath();
  ctx.moveTo(cx - R * 0.5, cy); ctx.lineTo(cx - R * 0.15, cy); ctx.moveTo(cx + R * 0.15, cy); ctx.lineTo(cx + R * 0.5, cy);
  ctx.moveTo(cx, cy - R * 0.5); ctx.lineTo(cx, cy - R * 0.15);
  ctx.stroke();
  ctx.fillStyle = dim; ctx.font = `10px ${mono}`; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
  ctx.fillText(`HDG ${String(Math.round(hdg) % 360).padStart(3, "0")}`, cx, cy - R - 8);

  // horizon: a short line inside the inner ring that stays level with the ground
  ctx.save();
  ctx.translate(cx, cy + pitch * (H / 2) / (Math.PI / 4));
  ctx.rotate(-roll);
  ctx.strokeStyle = faint; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(-R * 1.6, 0); ctx.lineTo(-R * 1.1, 0); ctx.moveTo(R * 1.1, 0); ctx.lineTo(R * 1.6, 0); ctx.stroke();
  ctx.restore();

  // Boxes are in the pixels of the frame the detector saw, and the picture
  // is object-fit: cover, so a frame pixel lands on the viewport at the
  // cover scale of the camera's own size, times the ratio of the two frames
  // when they differ.
  const nw = cam.naturalWidth, nh = cam.naturalHeight;
  const place = (fw, fh) => {
    const iw = nw || fw, ih = nh || fh;
    const s = Math.max(W / iw, H / ih), ox = (W - iw * s) / 2, oy = (H - ih * s) / 2;
    const sx = s * iw / fw, sy = s * ih / fh;
    return b => ({ x: ox + b.x * sx, y: oy + b.y * sy, w: b.w * sx, h: b.h * sy });
  };
  ctx.textAlign = "start"; ctx.font = `10px ${mono}`;

  // detections
  if (det && now() - det.at <= 1.0) {
    const at = place(det.w, det.h);
    for (const b of det.boxes) {
      const { x, y, w, h } = at(b);
      const person = b.label === "person";
      ctx.strokeStyle = person ? accent : dim; ctx.lineWidth = 1.5;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = person ? accent : dim;
      ctx.fillText(`${b.label} ${(b.p * 100).toFixed(0)}`.toUpperCase(), x, y - 5);
    }
  }

  // the lock: brackets on the corners of the held box, in the accent, drawn
  // over whatever the detector drew there. Solid while a detection matches,
  // dashed while coasting on the last box, and a faint dashed square while
  // waiting for a detection to appear under a tap on empty picture.
  if (!lock) return;
  const state = lock.state(now());
  const { x, y, w, h } = place(lock.fw, lock.fh)(lock.box);
  const waiting = state === "waiting";
  ctx.strokeStyle = waiting ? dim : accent; ctx.lineWidth = waiting ? 1 : 2.5;
  ctx.setLineDash(state === "locked" ? [] : [6, 5]);
  const L = Math.min(w, h) * 0.3;
  ctx.beginPath();
  for (const [px, py, dx, dy] of [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]]) {
    ctx.moveTo(px + dx * L, py); ctx.lineTo(px, py); ctx.lineTo(px, py + dy * L);
  }
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = waiting ? dim : accent;
  const word = waiting ? "tap · waiting" : `lock ${lock.label} ${(lock.p * 100).toFixed(0)}`;
  ctx.fillText(word.toUpperCase(), x, y + h + 13);
}

// ---- detector -------------------------------------------------------------
// Detection runs here, in the browser, in a worker (det_worker.js). Nothing
// is asked of the robot: it already sends the camera, and the handheld is
// the machine with a GPU to spare.
let detRate = 0;             // detections per second, smoothed
const GPU_DEADLINE_MS = 8000; // first GPU answer must land within this
function onBoxes(w, h, boxes) {
  det = { w, h, boxes: boxes || [], at: now() };
  if (lock) lock.update(det.boxes, det.at);
  const people = det.boxes.filter(b => b.label === "person").length;
  $("hud-det").textContent = `${det.boxes.length} objects · ${people} people`;
  $("hud-det").classList.toggle("on", people > 0);
}

// ---- lock-in ----------------------------------------------------------------
// A tap on the picture locks onto the box under the finger, or on a square
// around the finger when there is no box there yet. The lock is held across
// frames by lock.js and told to the gateway ten times a second. It ends on
// a second tap on the target, on the sticks, on stop, and by itself when
// the target has been out of sight for a few seconds.
function note(text) { lockNote = { text, until: now() + 3 }; }
function unlock(why) {
  if (!lock) return;
  lock = null;
  send({ t: "unlock" });
  log(`lock: ${why}`);
  note(why);
}
cam.addEventListener("click", e => {
  if (!cam.naturalWidth) return;
  const r = cam.getBoundingClientRect();
  // The tap in the camera's pixels, then in the detector's frame, which is
  // the camera's frame unless a still image was put in the viewport.
  const fresh = det && now() - det.at <= 1.0;
  const fw = fresh ? det.w : cam.naturalWidth, fh = fresh ? det.h : cam.naturalHeight;
  const p = toFrame(e.clientX - r.left, e.clientY - r.top, r.width, r.height, cam.naturalWidth, cam.naturalHeight);
  const x = p.x * fw / cam.naturalWidth, y = p.y * fh / cam.naturalHeight;
  if (x < 0 || y < 0 || x > fw || y > fh) return;
  if (lock && lock.covers(x, y)) { unlock("released"); return; }
  lock = new Lock(pick(fresh ? det.boxes : [], x, y, fw, fh), fw, fh, now());
  lockNote = null;
  log(lock.label ? `lock: ${lock.label}` : "lock: waiting for a box under the tap");
});
setInterval(() => {
  const t = now();
  const tag = $("hud-lock");
  if (lock) {
    const s = lock.state(t);
    if (s === "lost") { unlock("lost"); return; }
    const m = lock.message(t);
    if (m) send(m);
    tag.hidden = false;
    tag.classList.toggle("on", s !== "waiting");
    tag.textContent = s === "waiting" ? "tap · waiting for a box"
      : s === "locked" ? `lock · ${lock.label}`
      : `lock · ${lock.label} · coasting ${lock.age(t).toFixed(1)} s`;
  } else if (lockNote && t < lockNote.until) {
    tag.hidden = false; tag.classList.remove("on"); tag.textContent = `lock · ${lockNote.text}`;
  } else tag.hidden = true;
}, 100);

function startDetector(backend) {
  const worker = new Worker(`det_worker.js?backend=${backend}`, { type: "module" });
  let ready = false, last = 0, seen = 0, msAvg = 0, feed = null, guard = null;
  const stopWorker = () => { clearInterval(feed); clearTimeout(guard); worker.terminate(); };
  // The GPU path can wedge: one Chrome build we met took the session and
  // then never answered a frame, and dragged the whole tab down with it.
  // So the first frame gets a deadline; miss it and the worker is thrown
  // away and started again on the CPU, which is slow but does not hang.
  const armGuard = () => {
    clearTimeout(guard);
    if (backend === "cpu") return;
    guard = setTimeout(() => {
      stopWorker();
      log("detector: gpu did not answer in time, switching to cpu", "bad");
      startDetector("cpu");
    }, GPU_DEADLINE_MS);
  };
  worker.onmessage = e => {
    const m = e.data;
    if (m.t === "ready") { ready = true; if (!seen) armGuard(); return; }
    if (m.t === "log") { log(m.msg); return; }
    if (m.t === "missing") {
      stopWorker();
      log("detector assets missing – run fetch_assets.sh", "bad");
      return;
    }
    if (m.t === "error") {
      stopWorker(); lamp("det", false, "det"); log(`detector: ${m.msg}`, "bad");
      if (backend !== "cpu") startDetector("cpu");
      return;
    }
    if (m.t !== "det") return;
    clearTimeout(guard);
    // Say once how it went. Not on the first frames: those carry the
    // warm-up and read many times too slow, so ignore three and average
    // the ten after them.
    seen++;
    if (seen > 3) msAvg += m.ms / 10;
    if (seen === 13) log(`detector on ${m.backend}, ${msAvg.toFixed(0)} ms/frame`, "ok");
    const t = now();
    // Detections per second, leaned on the last reading so the lamp shows a
    // rate instead of a flicker.
    if (last) detRate = detRate ? detRate * 0.8 + 0.2 / (t - last) : 1 / (t - last);
    last = t;
    lamp("det", true, `det ${m.backend} ${detRate.toFixed(0)}`);
    onBoxes(m.w, m.h, m.boxes);
  };
  worker.onerror = e => {
    stopWorker(); lamp("det", false, "det"); log(`detector: ${e.message}`, "bad");
    if (backend !== "cpu") startDetector("cpu");
  };

  // Take a frame off the camera image that is already on screen. No second
  // stream: the wifi carries one MJPEG and this reads the picture out of it.
  // Only when the worker has finished the last one, and no faster than
  // 15 Hz, because past that the network is the limit anyway.
  feed = setInterval(() => {
    if (!ready || !cam.naturalWidth) return;
    ready = false;
    const w = cam.naturalWidth, h = cam.naturalHeight;
    createImageBitmap(cam)
      .then(bmp => worker.postMessage({ t: "frame", bmp, w, h }, [bmp]))
      .catch(() => { ready = true; });
  }, 66);
}

// The escape hatch: a detector in some other process, sending the same
// {t:"det", w, h, boxes} frames over a websocket (?det=ws://...).
function connectDetector(url) {
  let ws;
  try { ws = new WebSocket(url); } catch { setTimeout(() => connectDetector(url), 10000); return; }
  ws.onopen = () => { lamp("det", true, "det"); $("hud-det").textContent = "detector on"; };
  ws.onclose = () => { lamp("det", false, "det"); $("hud-det").textContent = "no detector"; $("hud-det").classList.remove("on"); setTimeout(() => connectDetector(url), 5000); };
  ws.onerror = () => {};
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.t === "det") onBoxes(m.w, m.h, m.boxes);
  };
}

// ---- inputs ---------------------------------------------------------------
// One drive source at a time: a connected pad wins, otherwise the keys.
// Whoever drives streams frames at 20 Hz; when nobody drives, one "stop"
// tells the gateway to start its zeroing burst right away instead of
// waiting for the dead-man to notice.
const DEADZONE = 0.1;
function shape(v) {
  const s = Math.abs(v) < DEADZONE ? 0 : Math.min(1, (Math.abs(v) - DEADZONE) / (1 - DEADZONE));
  return v < 0 ? -s : s;
}
let padIndex = null, padPrev = {}, padButtons = null;
// Button numbers in the browser's "standard" layout (A B X Y, bumpers, d-pad).
const STANDARD_BUTTONS = { 0: "arm", 1: "lie_down", 3: "stand_up", 4: "h-", 5: "h+",
                           12: "trick_paw_wave", 13: "trick_shake", 14: "trick_bow", 15: "trick_sit" };
// The Steam Deck's own controller, read straight from the kernel driver when
// Steam is not running to translate it: the browser reports it without a
// standard layout, and its buttons come in the driver's order. Same actions,
// numbers measured on the device (A=3 B=4 X=5 Y=6, bumpers 9 10, d-pad
// 16..19 up down left right); the sticks sit on axes 0-3 like the standard.
const RAW_DECK_BUTTONS = { 3: "arm", 4: "lie_down", 6: "stand_up", 9: "h-", 10: "h+",
                           16: "trick_paw_wave", 17: "trick_shake", 18: "trick_bow", 19: "trick_sit" };
window.addEventListener("gamepadconnected", e => {
  if (padIndex !== null) return;
  const gp = e.gamepad;
  const standard = gp.mapping === "standard";
  padIndex = gp.index; padButtons = standard ? STANDARD_BUTTONS : RAW_DECK_BUTTONS;
  lamp("pad", true); log(`pad: ${gp.id}${standard ? "" : " (raw layout)"}`);
});
window.addEventListener("gamepaddisconnected", e => {
  if (e.gamepad.index !== padIndex) return;
  padIndex = null; lamp("pad", false); log("pad disconnected", "bad"); send({ t: "stop" });
});
function call(key) {
  if (key === "arm") send({ t: "call", key, value: !armed });
  else if (key === "enable") send({ t: "call", key, value: !policyOn });
  else if (key === "h-") send({ t: "height", delta: -0.005 });
  else if (key === "h+") send({ t: "height", delta: 0.005 });
  else send({ t: "call", key });
}
function padFrame() {
  const gp = navigator.getGamepads()[padIndex];
  if (!gp) return null;
  const pressed = {};
  for (const [i, key] of Object.entries(padButtons)) {
    pressed[i] = !!gp.buttons[i] && gp.buttons[i].pressed;
    if (pressed[i] && !padPrev[i]) call(key);
  }
  padPrev = pressed;
  // left stick: forward and turn; right stick: strafe. Left/CCW is
  // positive in ROS, screen right is positive on the pad, so both flip.
  return { vx: -shape(gp.axes[1]), vy: -shape(gp.axes[2]), yaw: -shape(gp.axes[0]) };
}

const keys = new Set();
const KEYMAP = { KeyW: 1, KeyS: 1, KeyA: 1, KeyD: 1, KeyQ: 1, KeyE: 1, ArrowUp: 1, ArrowDown: 1, ArrowLeft: 1, ArrowRight: 1 };
window.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.code === "Space") { keys.clear(); unlock("stop"); send({ t: "stop" }); e.preventDefault(); return; }
  if (KEYMAP[e.code]) { keys.add(e.code); e.preventDefault(); }
});
window.addEventListener("keyup", e => keys.delete(e.code));
window.addEventListener("blur", () => keys.clear());
function keyFrame() {
  if (!keys.size) return null;
  const k = c => keys.has(c) ? 1 : 0;
  return {
    vx: k("KeyW") + k("ArrowUp") - k("KeyS") - k("ArrowDown"),
    vy: k("KeyA") - k("KeyD"),
    yaw: k("KeyQ") + k("ArrowLeft") - k("KeyE") - k("ArrowRight"),
  };
}

let wasDriving = false;
setInterval(() => {
  const frame = padIndex !== null ? padFrame() : keyFrame();
  // The operator moving a stick ends the lock: the same rule the gateway
  // will apply on its side, so the two never disagree. A connected pad
  // sends a frame every tick even at rest, so it is the movement that
  // counts, not the frame.
  if (stickMoved(frame)) unlock("sticks");
  if (frame) { send({ t: "cmd", ...frame }); wasDriving = true; }
  else if (wasDriving) { send({ t: "stop" }); wasDriving = false; }
}, 50);
document.addEventListener("visibilitychange", () => { if (document.hidden) { keys.clear(); unlock("page hidden"); send({ t: "stop" }); } });

for (const b of document.querySelectorAll("[data-call]")) b.onclick = () => call(b.dataset.call);
for (const b of document.querySelectorAll("[data-height]")) b.onclick = () => send({ t: "height", delta: parseFloat(b.dataset.height) });
// Deliberately not data-calls: these two talk to the browser, not the robot.
document.getElementById("reload").onclick = () => location.reload();
const fullBtn = document.getElementById("fullscreen");
fullBtn.onclick = () => {
  // A page may only ask for the whole screen while a finger is on it, which
  // is why this cannot happen by itself at load.
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen().catch(() => {});
};
// The screen can also be given back by the browser (Escape on a keyboard),
// so the word follows the actual state rather than the last press.
document.addEventListener("fullscreenchange", () => {
  fullBtn.textContent = document.fullscreenElement ? "window" : "full";
  fullBtn.title = document.fullscreenElement ? "give the window back" : "fill the screen";
});

// ---- render loop -------------------------------------------------------------
setInterval(() => {
  const t = now();
  for (const s of [att, gyro, cmd]) s.draw(t);
  eff.draw();
  drawOverlay();
}, 33);
setInterval(() => { $("hud-clock").textContent = stamp(); }, 1000);

connectGateway();

// ?detsrc=<url> puts a still image in the viewport instead of the camera, and
// the detector then looks at that. It is how the detection path gets tested
// without a robot pointed at something interesting. Same-origin only: the
// page is cross-origin isolated, so a picture from elsewhere will not load.
// Drop one in the asset store and it is at /det/<name>.
const detsrc = params.get("detsrc");
if (detsrc) cam.src = detsrc;

const detParam = params.get("det");
if (detParam === "off") log("detector off (?det=off)");
else if (detParam === "cpu" || detParam === "gpu") startDetector(detParam);
else if (detParam) connectDetector(detParam);
else startDetector("auto");
