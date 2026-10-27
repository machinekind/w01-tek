// The operator page. Talks to server.py over one websocket (protocol in
// wire.py): JSON text frames for the log, the nav words, the robot's load
// and replies; binary frames (one tag byte first) for the camera JPEG, the
// brain's annotated JPEG (numbered for its ask line) and the raw depth.
"use strict";

const TAG_CAMERA = 0, TAG_ANNOTATED = 1, TAG_DEPTH = 2;
const DEPTH_MIN_MM = 200, DEPTH_MAX_MM = 4000;

const $ = (id) => document.getElementById(id);
let ws = null;

// ---- colour map for depth: jet-like, index 0 = red (near), 255 = blue (far) --
const LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = 1 - i / 255;  // index 0 -> t 1 (red) ... index 255 -> t 0 (blue)
    const r = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3))));
    const g = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2))));
    const b = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1))));
    lut[i * 3] = r; lut[i * 3 + 1] = g; lut[i * 3 + 2] = b;
  }
  return lut;
})();
function lutIndex(mm) {
  const x = (mm - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM);
  return Math.max(0, Math.min(255, Math.round(x * 255)));   // near -> 0 -> red
}
(function drawLegend() {
  const c = $("legend"), g = c.getContext("2d"), img = g.createImageData(256, 1);
  for (let i = 0; i < 256; i++) {
    const k = lutIndex(DEPTH_MIN_MM + (i / 255) * (DEPTH_MAX_MM - DEPTH_MIN_MM));
    img.data.set([LUT[k * 3], LUT[k * 3 + 1], LUT[k * 3 + 2], 255], i * 4);
  }
  for (let y = 0; y < c.height; y++) g.putImageData(img, 0, y);
})();

// ---- frame-rate meters ---------------------------------------------------
function meter(el) {
  const times = [];
  return () => {
    const now = performance.now();
    times.push(now);
    while (times.length && now - times[0] > 2000) times.shift();
    el.textContent = `${(times.length / 2).toFixed(1)} fps`;
  };
}
const camTick = meter($("cam-fps"));
const depthTick = meter($("depth-fps"));

// ---- pictures --------------------------------------------------------------
function showJpeg(img, bytes) {
  const url = URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }));
  const old = img.dataset.url;
  img.onload = () => { if (old) URL.revokeObjectURL(old); };
  img.dataset.url = url;
  img.src = url;
}

let lastCamAt = 0;
function onCamera(bytes) {
  showJpeg($("cam"), bytes);
  lastCamAt = performance.now();
  $("cam-info").textContent = `${Math.round(bytes.byteLength / 1024)} KB a frame`;
  camTick();
}

let depthFrame = null;  // {w, h, px} for the hover readout
function onDepth(buf) {
  const dv = new DataView(buf);
  const w = dv.getUint16(2, true), h = dv.getUint16(4, true);
  const px = new Uint16Array(buf, 8, w * h);
  const c = $("depth");
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  const g = c.getContext("2d");
  const img = g.createImageData(w, h);
  const out = img.data;
  for (let i = 0, j = 0; i < px.length; i++, j += 4) {
    const mm = px[i];
    if (mm === 0) { out[j] = out[j + 1] = out[j + 2] = 0; }
    else { const k = lutIndex(mm) * 3; out[j] = LUT[k]; out[j + 1] = LUT[k + 1]; out[j + 2] = LUT[k + 2]; }
    out[j + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  depthFrame = { w, h, px };
  depthTick();
}

$("depth").addEventListener("mousemove", (e) => {
  if (!depthFrame) return;
  const r = e.target.getBoundingClientRect();
  const x = Math.floor(((e.clientX - r.left) / r.width) * depthFrame.w);
  const y = Math.floor(((e.clientY - r.top) / r.height) * depthFrame.h);
  const mm = depthFrame.px[y * depthFrame.w + x];
  $("depth-info").textContent = mm ? `(${x}, ${y}): ${(mm / 1000).toFixed(2)} m` : `(${x}, ${y}): no depth`;
});

// ---- the chat: the operator's tasks and the brain's steps, newest at the bottom --
const chat = $("chat");
const hhmm = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
function nearBottom() { return chat.scrollHeight - chat.scrollTop - chat.clientHeight < 80; }
function addMsg(kind, title, detail, img) {
  const stick = nearBottom();
  const d = document.createElement("div");
  d.className = `msg ${kind}`;
  const t = document.createElement("span"); t.className = "time"; t.textContent = hhmm();
  const h = document.createElement("div"); h.className = "title"; h.textContent = title;
  d.append(t, h);
  if (detail) { const x = document.createElement("div"); x.className = "detail"; x.textContent = detail; d.append(x); }
  if (img) addImg(d, img, stick);
  chat.append(d);
  if (stick) chat.scrollTop = chat.scrollHeight; else $("jump").hidden = false;
  return d;
}
function addImg(d, url, stick) {
  const im = document.createElement("img");
  im.src = url; im.alt = "the picture the brain answered on";
  im.addEventListener("click", () => im.classList.toggle("zoom"));
  im.addEventListener("load", () => { if (stick) chat.scrollTop = chat.scrollHeight; });
  d.append(im);
}
function addDivider(text) {
  const d = document.createElement("div"); d.className = "divider"; d.textContent = text; chat.append(d);
}
chat.addEventListener("scroll", () => { if (nearBottom()) $("jump").hidden = true; });
$("jump").addEventListener("click", () => { chat.scrollTop = chat.scrollHeight; $("jump").hidden = true; });

// One brain status as a chat line: [css kind, friendly title].
function describe(s) {
  const a = s.answer || {};
  const lat = s.latency_s != null ? ` · ${s.latency_s}s` : "";
  switch (s.action) {
    case "ask":
      if (a.type === "goal") return ["brain goal", `I see ${a.label || "the target"} — heading there${lat}`];
      if (a.type === "not_visible") return ["brain look", `I don't see it yet${lat}`];
      if (a.type === "turn") return ["brain look", `I'd look ${a.deg > 0 ? "left" : "right"} ${Math.abs(a.deg || 0)}°${lat}`];
      if (a.type === "done") return ["brain look", `I think I'm there${lat}`];
      return ["brain fail", `The model's answer made no sense${lat}`];
    case "verify":
      return a.visible ? ["brain goal", `Double-checked: yes, that's it${lat}`] : ["brain look", `Double-checked: no, not it${lat}`];
    case "pixel_goal": {
      const d = s.distance_to_target_m != null ? `, target ${s.distance_to_target_m} m away` : "";
      const words = { reached: "Got to it", blocked: "Blocked on the way", timeout: "Walk timed out",
        no_depth: "It's too far for the depth camera", no_frame: "No depth frame for that picture",
        no_tf: "No pose for that picture", cancelled: "Walk cancelled" };
      return [s.result === "reached" ? "brain goal" : "brain move", `${words[s.result] || `Walk: ${s.result}`}${d}`];
    }
    case "turn": return ["brain move", `Turning ${s.deg > 0 ? "left" : "right"} ${Math.abs(s.deg)}° to look around (${Math.round(s.turned_total || 0)}° so far)`];
    case "approach": return ["brain move", `Walking ${s.metres} m towards it${s.detour_deg ? `, ${s.detour_deg}° detour` : ""} → ${s.result}${s.moved_m != null ? `, moved ${s.moved_m} m` : ""}`];
    case "explore": return ["brain move", `Exploring ${s.metres} m ahead → ${s.result}${s.moved_m != null ? `, moved ${s.moved_m} m` : ""}`];
    case "finished":
      if (s.result === "done") return ["brain done", "Done — I'm at the target"];
      if (s.result === "cancelled") return ["brain fail", "Stopped"];
      if (s.result === "replaced") return ["brain look", "Replaced by a new task"];
      return ["brain fail", `Gave up: ${s.result}${s.error ? ` — ${s.error}` : ""}`];
    default: return ["brain look", s.action];
  }
}

let current = null;      // the instruction whose lines are on screen
let shown = 0;           // how many of its lines are
let lastAsked = null;    // the operator's last sent text, to not echo it twice
// The server numbers each picture for its ask line, so the two meet by
// number whichever comes first: on a page load mid-task, after a
// reconnect, or when a line beats its picture by a moment.
let picture = null;      // the brain's newest picture, {n, url}, until its ask line shows it
let waiting = null;      // the newest ask line still without its picture, {n, el}
function onPicture(buf) {
  const n = new DataView(buf).getUint32(1, true);
  const url = URL.createObjectURL(new Blob([new Uint8Array(buf, 5)], { type: "image/jpeg" }));
  if (waiting && waiting.n === n) { addImg(waiting.el, url, nearBottom()); waiting = null; return; }
  if (picture) URL.revokeObjectURL(picture.url);   // never shown: free it
  picture = { n, url };
}
function onLog(m) {
  if (m.instruction !== current || m.lines.length < shown) {
    current = m.instruction; shown = 0;
    if (current && current !== lastAsked) addMsg("user", current, "sent from another console");
    lastAsked = null;
  }
  for (const line of m.lines.slice(shown)) {
    const [kind, title] = describe(line.s || { action: line.action });
    let img = null;
    if (line.action === "ask" && picture && line.picture === picture.n) { img = picture.url; picture = null; }
    const d = addMsg(kind, title, line.text, img);
    if (line.action === "ask" && line.picture != null && !img) waiting = { n: line.picture, el: d };
  }
  shown = m.lines.length;
}

function onNav(m) {
  $("nav").textContent = `goto ${m.goto || "-"} · pixel ${m.pixel || "-"}`;
}

function onSys(m) {
  $("sys-title").textContent = `Computer: ${m.host}`;
  const cores = $("cores");
  if (cores.children.length !== m.cores.length) {
    cores.replaceChildren(...m.cores.map((_, i) => {
      const d = document.createElement("div");
      d.className = "core";
      d.innerHTML = `<div class="row"><span></span><span></span></div><div class="bar"><i></i></div>`;
      return d;
    }));
  }
  m.cores.forEach((pct, i) => {
    const d = cores.children[i];
    const rt = m.isolated.includes(i);
    d.querySelector(".row span:first-child").textContent = `core ${i}${rt ? " (RT, reserved)" : ""}`;
    d.querySelector(".row span:last-child").textContent = `${Math.round(pct)}%`;
    d.querySelector(".bar > i").style.width = `${Math.max(0, Math.min(100, pct))}%`;
    d.className = "core" + (rt ? " rt" : pct >= 90 ? " hot" : "");
  });
  $("load").textContent = `${m.load[0].toFixed(1)} / ${m.load[1].toFixed(1)}`;
  $("mem").textContent = `${m.mem_used_mb} / ${m.mem_total_mb} MB`;
  $("temp").textContent = m.temp_c == null ? "-" : `${Math.round(m.temp_c)} °C`;
  $("sys-age").textContent = m.age_s > 5 ? `${Math.round(m.age_s)} s old` : "live";
}

// ---- the walker's guide: what the person carrying the camera does ---------
let lastGuideAt = 0;
function onGuide(m) {
  lastGuideAt = performance.now();
  const walk = $("walk");
  $("walk-role").textContent = m.role === "odometry" ? "Robot command (dead-reckoned pose)" : "Walk · you are the robot's legs";
  walk.className = `walk ${m.kind === "walk" ? "walk-on" : m.kind === "turn" ? "turn-on" : "still"}`;
  $("walk-text").textContent = m.text;
  // SVG y points down: a heading of +90 (left) rotates the arrow -90.
  let rot = -m.heading_deg;
  if (m.kind === "turn") rot = m.wz_deg_s > 0 ? -90 : 90;
  $("walk-arrow-g").setAttribute("transform", `rotate(${rot})`);
  if (m.kind === "turn") {
    $("walk-progress").textContent = `turned ${Math.abs(m.segment_deg).toFixed(0)} deg ${m.segment_deg >= 0 ? "left" : "right"} so far`;
  } else if (m.kind === "walk") {
    $("walk-progress").textContent = `walked ${m.segment_m.toFixed(2)} m so far`;
  } else {
    $("walk-progress").textContent = "";
  }
  $("walk-pose").textContent = `mock pose ${m.pose.x} m, ${m.pose.y} m, facing ${m.pose.yaw_deg}°`;
  const g = $("goal-g");
  if (m.goal) {
    g.classList.remove("none");
    g.setAttribute("transform", `rotate(${-m.goal.bearing_deg})`);
    const side = Math.abs(m.goal.bearing_deg) < 5 ? "straight ahead"
      : `${Math.abs(m.goal.bearing_deg).toFixed(0)} deg ${m.goal.bearing_deg > 0 ? "left" : "right"}`;
    $("goal-text").textContent = `goal ${m.goal.dist_m.toFixed(2)} m, ${side}`;
  } else {
    g.classList.add("none");
    $("goal-text").textContent = "no goal";
  }
}
setInterval(() => {
  const el = $("walk-age");
  if (!lastGuideAt) return;
  const age = (performance.now() - lastGuideAt) / 1000;
  el.textContent = age > 2 ? `· walker quiet ${age.toFixed(0)} s` : "· live";
}, 500);

function onReply(m) {
  if (m.ok && m.text.startsWith("sent: ")) return;   // the user bubble already says it
  const d = addMsg(`system${m.ok ? "" : " bad"}`, m.text);
  d.querySelector(".time").remove();
}

// ---- the socket --------------------------------------------------------------
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { $("link").textContent = "connected"; $("link").className = "pill ok"; };
  ws.onclose = () => {
    $("link").textContent = "disconnected, retrying"; $("link").className = "pill bad";
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      const m = JSON.parse(ev.data);
      if (m.t === "log") onLog(m);
      else if (m.t === "nav") onNav(m);
      else if (m.t === "sys") onSys(m);
      else if (m.t === "reply") onReply(m);
      else if (m.t === "guide") onGuide(m);
      return;
    }
    const buf = ev.data, tag = new Uint8Array(buf, 0, 1)[0];
    if (tag === TAG_CAMERA) onCamera(new Uint8Array(buf, 1));
    else if (tag === TAG_ANNOTATED) onPicture(buf);
    else if (tag === TAG_DEPTH) onDepth(buf);
  };
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  else onReply({ ok: false, text: "not connected to the page server" });
}

document.querySelectorAll("[data-svc]").forEach((b) =>
  b.addEventListener("click", () => send({ t: "svc", name: b.dataset.svc })));
function stopNow() { send({ t: "stop" }); chat.scrollTop = chat.scrollHeight; }
$("stop").addEventListener("click", stopNow);
// Esc freezes the robot from anywhere on the page, the task box included:
// the hand on the keyboard is closer than the mouse to the STOP button.
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !e.repeat) { e.preventDefault(); stopNow(); }
});
$("task-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("task-input").value.trim();
  if (!text) return;
  send({ t: "task", text });
  addDivider("new task");
  addMsg("user", text);
  lastAsked = text;
  chat.scrollTop = chat.scrollHeight;
  $("task-input").value = "";
});

// A camera that stops: say so rather than freezing on the last frame.
setInterval(() => {
  if (lastCamAt && performance.now() - lastCamAt > 2000)
    $("cam-info").textContent = `camera quiet for ${Math.round((performance.now() - lastCamAt) / 1000)} s`;
}, 1000);

connect();
