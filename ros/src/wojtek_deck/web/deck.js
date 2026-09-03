// The deck panel. Three links, three jobs:
//   gateway  ws://<host>/ws         commands out (sticks, buttons), status in
//   bridge   ws://<host>:<bridge>   telemetry in, decoded from CDR (bridge.js)
//   detector ws://localhost:8091    boxes from a detector running on the
//                                   handheld itself (optional, ?det=<url>)
// The camera is the gateway's MJPEG stream in a plain <img>; the reticle,
// horizon and detections are drawn on the overlay canvas above it.
import { Bridge } from "./bridge.js";
import { Bars, Strip } from "./charts.js";

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
  gw.onclose = () => { lamp("link", false); setDrive("idle"); setTimeout(connectGateway, 1000); };
  gw.onmessage = e => onGateway(JSON.parse(e.data));
}
function send(o) { if (gw && gw.readyState === 1) gw.send(JSON.stringify(o)); }
function onGateway(m) {
  if (m.t === "hello") {
    height = m.height_default;
    $("policy").textContent = m.policy || "";
    if (!bridge) startBridge(params.get("bridge") || `ws://${location.hostname}:${m.bridge_port}`);
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
  const cyan = css("--cyan"), magenta = css("--magenta"), ink = css("--ink"), mono = css("--mono");
  const [roll, pitch, yaw] = rpy;
  const cx = W / 2, cy = H / 2, R = Math.min(W, H) * 0.2;

  // reticle: dashed outer ring, inner ring, and an arc that points the way
  // the robot is heading (ROS +yaw is counter-clockwise), centre bars
  const hdg = ((-yaw * 180 / Math.PI) % 360 + 360) % 360;
  ctx.strokeStyle = cyan; ctx.lineWidth = 1;
  ctx.setLineDash([3, 9]); ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha = .6; ctx.beginPath(); ctx.arc(cx, cy, R * 0.7, 0, Math.PI * 2); ctx.stroke(); ctx.globalAlpha = 1;
  const a0 = -Math.PI / 2 + hdg * Math.PI / 180;
  ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(cx, cy, R, a0 - Math.PI / 4, a0 + Math.PI / 4); ctx.stroke();
  ctx.lineWidth = 1.5; ctx.strokeStyle = "#ffffff";
  ctx.beginPath();
  ctx.moveTo(cx - R * 0.5, cy); ctx.lineTo(cx - R * 0.15, cy); ctx.moveTo(cx + R * 0.15, cy); ctx.lineTo(cx + R * 0.5, cy);
  ctx.moveTo(cx, cy - R * 0.5); ctx.lineTo(cx, cy - R * 0.15);
  ctx.stroke();
  ctx.fillStyle = ink; ctx.font = `10px ${mono}`; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
  ctx.fillText(`HDG ${String(Math.round(hdg) % 360).padStart(3, "0")}`, cx, cy - R - 8);

  // horizon: a short line inside the inner ring that stays level with the ground
  ctx.save();
  ctx.translate(cx, cy + pitch * (H / 2) / (Math.PI / 4));
  ctx.rotate(-roll);
  ctx.strokeStyle = cyan; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(-R * 1.6, 0); ctx.lineTo(-R * 1.1, 0); ctx.moveTo(R * 1.1, 0); ctx.lineTo(R * 1.6, 0); ctx.stroke();
  ctx.restore();

  // detections, in the frame's pixel space mapped onto the shard (object-fit: cover)
  if (!det || now() - det.at > 1.0) return;
  const nw = cam.naturalWidth || det.w, nh = cam.naturalHeight || det.h;
  const s = Math.max(W / nw, H / nh), ox = (W - nw * s) / 2, oy = (H - nh * s) / 2;
  const sx = s * nw / det.w, sy = s * nh / det.h;
  ctx.textAlign = "start"; ctx.font = `10px ${mono}`;
  for (const b of det.boxes) {
    const x = ox + b.x * sx, y = oy + b.y * sy, w = b.w * sx, h = b.h * sy;
    const person = b.label === "person";
    ctx.strokeStyle = person ? magenta : cyan; ctx.lineWidth = 1.5;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = person ? magenta : cyan;
    ctx.fillText(`${b.label} ${(b.p * 100).toFixed(0)}`.toUpperCase(), x, y - 5);
  }
}

// ---- detector (optional, on the handheld) --------------------------------
function connectDetector() {
  const url = params.get("det") || "ws://localhost:8091";
  let ws;
  try { ws = new WebSocket(url); } catch { setTimeout(connectDetector, 10000); return; }
  ws.onopen = () => { lamp("det", true, "det"); $("hud-det").textContent = "detector on"; };
  ws.onclose = () => { lamp("det", false, "det"); $("hud-det").textContent = "no detector"; $("hud-det").classList.remove("on"); setTimeout(connectDetector, 5000); };
  ws.onerror = () => {};
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.t !== "det") return;
    det = { w: m.w, h: m.h, boxes: m.boxes || [], at: now() };
    const people = det.boxes.filter(b => b.label === "person").length;
    $("hud-det").textContent = `${det.boxes.length} objects · ${people} people`;
    $("hud-det").classList.toggle("on", people > 0);
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
let padIndex = null, padPrev = {};
const PAD_BUTTONS = { 0: "arm", 1: "lie_down", 3: "stand_up", 4: "h-", 5: "h+",
                      12: "trick_paw_wave", 13: "trick_shake", 14: "trick_bow", 15: "trick_sit" };
window.addEventListener("gamepadconnected", e => {
  if (padIndex !== null) return;
  padIndex = e.gamepad.index; lamp("pad", true); log(`pad: ${e.gamepad.id}`);
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
  for (const [i, key] of Object.entries(PAD_BUTTONS)) {
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
  if (e.code === "Space") { keys.clear(); send({ t: "stop" }); e.preventDefault(); return; }
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
  if (frame) { send({ t: "cmd", ...frame }); wasDriving = true; }
  else if (wasDriving) { send({ t: "stop" }); wasDriving = false; }
}, 50);
document.addEventListener("visibilitychange", () => { if (document.hidden) { keys.clear(); send({ t: "stop" }); } });

for (const b of document.querySelectorAll("[data-call]")) b.onclick = () => call(b.dataset.call);
for (const b of document.querySelectorAll("[data-height]")) b.onclick = () => send({ t: "height", delta: parseFloat(b.dataset.height) });

// ---- render loop -------------------------------------------------------------
setInterval(() => {
  const t = now();
  for (const s of [att, gyro, cmd]) s.draw(t);
  eff.draw();
  drawOverlay();
}, 33);
setInterval(() => { $("hud-clock").textContent = stamp(); }, 1000);

connectGateway();
connectDetector();
