// The deck panel. Three links, three jobs:
//   gateway  ws://<host>/ws         commands out (sticks, buttons), status in
//   bridge   ws://<host>:<bridge>   telemetry in, decoded from CDR (bridge.js)
//   detector ws://localhost:8091    boxes from a detector running on the
//                                   handheld itself (optional, ?det=<url>)
// The camera is the gateway's MJPEG stream in a plain <img> that fills the
// screen; the head-up symbology is drawn on the overlay canvas.
import { Bridge } from "./bridge.js";
import { Strip } from "./charts.js";

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const now = () => performance.now() / 1000;
const DASH = [[], [4, 3], [1, 3]];   // series identity: solid, dashed, dotted

// ---- state -----------------------------------------------------------------
let cmdLow = [-0.6, -0.4, -0.7], cmdHigh = [0.6, 0.4, 0.7];
let heightRange = [0.09, 0.17], height = 0.125;
let armed = false, policyOn = false;
let rpy = [0, 0, 0];          // latest attitude
let cmdLatest = [0, 0, 0];    // latest /cmd_vel, for the speed tape
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
    cmdLow = m.cmd_low; cmdHigh = m.cmd_high; heightRange = m.height_range; height = m.height_default;
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
    height = m.height;
    setDrive(m.drive);
    lamp("cam", m.cam_hz > 0.5, m.cam_hz > 0.5 ? `cam ${m.cam_hz.toFixed(0)}` : "cam —");
  }
}

// ---- strips and readouts -------------------------------------------------
const att = new Strip($("ch-att"), { series: [{ name: "r", dash: DASH[0] }, { name: "p", dash: DASH[1] }], min: -45, max: 45, fixed: 1 });
const gyro = new Strip($("ch-gyro"), { series: [{ name: "x", dash: DASH[0] }, { name: "y", dash: DASH[1] }, { name: "z", dash: DASH[2] }] });
const cmd = new Strip($("ch-cmd"), { series: [{ name: "vx", dash: DASH[0] }, { name: "vy", dash: DASH[1] }, { name: "wz", dash: DASH[2] }] });

function startBridge(url) {
  bridge = new Bridge(url, {
    "/imu_sensor_broadcaster/imu": m => {
      const q = m.orientation, w = m.angular_velocity;
      const roll = Math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y));
      const pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (q.w * q.y - q.z * q.x))));
      const yaw = Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
      rpy = [roll, pitch, yaw];
      const t = now();
      att.push(t, [roll * 180 / Math.PI, pitch * 180 / Math.PI]);
      gyro.push(t, [w.x, w.y, w.z]);
    },
    "/cmd_vel": m => {
      cmdLatest = [m.linear.x, m.linear.y, m.angular.z];
      cmd.push(now(), cmdLatest);
    },
    "/wojtek/policy_timing": m => { $("ro-tick").textContent = `${m.inference_ms.toFixed(1)} / ${m.period_ms.toFixed(1)} ms`; },
    "/wojtek/joint_targets": m => {
      // The joint working hardest right now: torque when the contract has a
      // torque head, otherwise the largest position target.
      const useEffort = m.effort && m.effort.length === m.name.length && m.effort.some(v => v !== 0);
      const vals = useEffort ? m.effort : m.position;
      let k = 0;
      for (let i = 1; i < vals.length; i++) if (Math.abs(vals[i]) > Math.abs(vals[k])) k = i;
      const short = (m.name[k] || "").replace(/_joint$/, "").split("_").map(p => p[0]).join("");
      $("ro-eff-label").textContent = useEffort ? "effort max" : "target max";
      $("ro-eff").textContent = `${short} ${vals[k].toFixed(2)} ${useEffort ? "N·m" : "rad"}`;
    },
    "/wojtek/sysinfo": m => {
      const cpu = m.cpu_percent.length ? m.cpu_percent.reduce((a, b) => a + b, 0) / m.cpu_percent.length : NaN;
      $("ro-sys").textContent = `${cpu.toFixed(0)} % · ${m.soc_temp_c.toFixed(0)} °C`;
      $("ro-wifi").textContent = `${(m.wifi_rx_bytes_per_s / 1024).toFixed(0)} · ${(m.wifi_tx_bytes_per_s / 1024).toFixed(0)} kB/s`;
      // one line for the Pi's power/thermal flags: what is on now, else what was ever seen
      const flags = ["undervoltage", "throttled", "freq_capped", "soft_temp_limit"];
      const nowOn = flags.filter(f => m[`${f}_now`]), ever = flags.filter(f => m[`${f}_ever`]);
      const row = $("ro-flags");
      row.classList.toggle("now", nowOn.length > 0);
      row.classList.toggle("ever", nowOn.length === 0 && ever.length > 0);
      $("ro-flag").textContent = nowOn.length ? nowOn.join(" ").replace(/_/g, "-") + " now"
        : ever.length ? ever.join(" ").replace(/_/g, "-") + " seen" : "ok";
    },
  }, (up) => { lamp("bridge", up); if (up) log("bridge connected", "ok"); });
}

// ---- overlay: heading tape, pitch ladder, tapes, detections -------------
const overlay = $("overlay"), cam = $("cam");
function drawOverlay() {
  const dpr = window.devicePixelRatio || 1;
  const W = overlay.clientWidth, H = overlay.clientHeight;
  if (!W || !H) return;
  if (overlay.width !== W * dpr || overlay.height !== H * dpr) { overlay.width = W * dpr; overlay.height = H * dpr; }
  const ctx = overlay.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const hud = css("--hud"), bg = css("--bg"), mono = css("--mono");
  ctx.strokeStyle = hud; ctx.fillStyle = hud; ctx.lineWidth = 1.25;
  ctx.font = `12px ${mono}`; ctx.textBaseline = "middle";
  const [roll, pitch, yaw] = rpy;
  const cx = W / 2, cy = H / 2;

  // heading tape: 120 degrees across 600 px, labels every 30, boxed readout
  {
    const hdg = ((-yaw * 180 / Math.PI) % 360 + 360) % 360;   // ROS +yaw is CCW
    const tw = 600, x0 = cx - tw / 2, y = 70, pxPerDeg = tw / 120;
    ctx.save(); ctx.beginPath(); ctx.rect(x0, 20, tw, 70); ctx.clip();
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + tw, y);
    ctx.textAlign = "center";
    for (let d = Math.floor((hdg - 60) / 10) * 10; d <= hdg + 60; d += 10) {
      const x = cx + (d - hdg) * pxPerDeg, major = ((d % 30) + 30) % 30 === 0;
      ctx.moveTo(x, y); ctx.lineTo(x, y - (major ? 10 : 6));
      if (major) {
        const deg = ((d % 360) + 360) % 360;
        ctx.fillText({ 0: "N", 90: "E", 180: "S", 270: "W" }[deg] || String(deg / 10).padStart(2, "0"), x, y - 18);
      }
    }
    ctx.stroke();
    ctx.fillStyle = bg; ctx.fillRect(cx - 24, y + 2, 48, 18);
    ctx.fillStyle = hud; ctx.font = `13px ${mono}`;
    ctx.fillText(String(Math.round(hdg) % 360).padStart(3, "0"), cx, y + 11);
    ctx.font = `12px ${mono}`;
    ctx.restore();
  }

  // pitch ladder: bars with end ticks, negatives dashed; 10 degrees = 60 px
  {
    const k = 6;   // px per degree
    ctx.save();
    ctx.translate(cx, cy + pitch * 180 / Math.PI * k);
    ctx.rotate(-roll);
    ctx.textAlign = "end";
    ctx.beginPath(); ctx.moveTo(-160, 0); ctx.lineTo(-50, 0); ctx.moveTo(50, 0); ctx.lineTo(160, 0); ctx.stroke();
    for (const d of [-20, -10, 10, 20]) {
      const y = -d * k, half = Math.abs(d) === 20 ? 30 : 40, tick = d > 0 ? 6 : -6;
      ctx.setLineDash(d < 0 ? [6, 4] : []);
      ctx.beginPath();
      ctx.moveTo(-half - 40, y + tick); ctx.lineTo(-half - 40, y); ctx.lineTo(-half, y);
      ctx.moveTo(half, y); ctx.lineTo(half + 40, y); ctx.lineTo(half + 40, y + tick);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillText(String(Math.abs(d)), -half - 48, y); ctx.textAlign = "start"; ctx.fillText(String(Math.abs(d)), half + 48, y); ctx.textAlign = "end";
    }
    ctx.restore();
    // flight-path marker
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - 24, cy); ctx.lineTo(cx - 6, cy); ctx.moveTo(cx + 6, cy); ctx.lineTo(cx + 24, cy); ctx.moveTo(cx, cy - 16); ctx.lineTo(cx, cy - 6); ctx.stroke();
    ctx.lineWidth = 1.25;
  }

  // tapes: speed (left, commanded vx over the trained box) and height (right)
  const tape = (x, top, len, lo, hi, value, side, fmt, title) => {
    const Y = v => top + len - (v - lo) / (hi - lo) * len;
    const dir = side === "left" ? -1 : 1;
    ctx.textAlign = side === "left" ? "end" : "start";
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + len);
    const span = hi - lo, step = span > 0.5 ? 0.3 : span > 0.05 ? 0.02 : 0.01;
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
      const y = Y(v), major = Math.round(v / step) % 2 === 0;
      ctx.moveTo(x, y); ctx.lineTo(x + dir * (major ? 10 : 6), y);
      if (major) ctx.fillText(fmt(v), x + dir * 18, y);
    }
    ctx.stroke();
    ctx.fillText(title, x + dir * 18, top - 22);
    const y = Y(Math.max(lo, Math.min(hi, value)));
    ctx.fillStyle = bg;
    ctx.beginPath();
    ctx.moveTo(x, y); ctx.lineTo(x + dir * 8, y - 10); ctx.lineTo(x + dir * 72, y - 10); ctx.lineTo(x + dir * 72, y + 10); ctx.lineTo(x + dir * 8, y + 10); ctx.closePath();
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = hud; ctx.font = `14px ${mono}`;
    ctx.fillText(fmt(value), x + dir * 14, y);
    ctx.font = `12px ${mono}`;
  };
  const tTop = H * 0.25, tLen = H * 0.5;
  tape(cx - 380, tTop, tLen, Math.min(0, cmdLow[0]), cmdHigh[0], cmdLatest[0], "left", v => v.toFixed(2), "vx m/s");
  if (heightRange[1] > heightRange[0])
    tape(cx + 380, tTop, tLen, heightRange[0], heightRange[1], height, "right", v => v.toFixed(3), "h m");
  else { ctx.textAlign = "start"; ctx.fillText(`h ${height.toFixed(3)}`, cx + 398, tTop - 22); }
  ctx.textAlign = "start";

  // detections: corner brackets in the frame's pixel space mapped onto the
  // screen (the picture is object-fit: cover, so the scale is the larger one)
  if (!det || now() - det.at > 1.0) return;
  const nw = cam.naturalWidth || det.w, nh = cam.naturalHeight || det.h;
  const s = Math.max(W / nw, H / nh), ox = (W - nw * s) / 2, oy = (H - nh * s) / 2;
  const sx = s * nw / det.w, sy = s * nh / det.h;
  ctx.lineWidth = 1.5; ctx.font = `12px ${mono}`; ctx.textBaseline = "bottom";
  for (const b of det.boxes) {
    const x = ox + b.x * sx, y = oy + b.y * sy, w = b.w * sx, h = b.h * sy, c = 16;
    ctx.beginPath();
    ctx.moveTo(x, y + c); ctx.lineTo(x, y); ctx.lineTo(x + c, y);
    ctx.moveTo(x + w - c, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + c);
    ctx.moveTo(x, y + h - c); ctx.lineTo(x, y + h); ctx.lineTo(x + c, y + h);
    ctx.moveTo(x + w - c, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - c);
    ctx.stroke();
    ctx.fillText(`${b.label} ${(b.p * 100).toFixed(0)}`.toUpperCase(), x, y - 6);
  }
  ctx.textBaseline = "middle";
}

// ---- detector (optional, on the handheld) --------------------------------
function connectDetector() {
  const url = params.get("det") || "ws://localhost:8091";
  let ws;
  try { ws = new WebSocket(url); } catch { setTimeout(connectDetector, 10000); return; }
  ws.onopen = () => lamp("det", true, "det");
  ws.onclose = () => { lamp("det", false, "det"); setTimeout(connectDetector, 5000); };
  ws.onerror = () => {};
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.t !== "det") return;
    det = { w: m.w, h: m.h, boxes: m.boxes || [], at: now() };
    lamp("det", true, `det ${det.boxes.length}`);
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
  drawOverlay();
}, 33);
setInterval(() => { $("hud-clock").textContent = stamp(); }, 1000);

connectGateway();
connectDetector();
