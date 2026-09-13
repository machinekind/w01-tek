// The lock-in: which box the operator tapped, and keeping hold of it while
// detections come and go.
//
// This is arithmetic only. No DOM, no socket, no clock of its own: the page
// (deck.js) hands in the tap, the detector's boxes and the time, and reads
// back a box to draw and a message to send. test/lock.test.js drives the
// same functions from node.
//
// A lock goes through these states, read with `state(t)`:
//   waiting   the tap landed on empty picture. The box is a square around
//             the finger and holds still until a detection appears inside
//             it. Nothing is sent to the robot in this state.
//   locked    a detection matched within the last COAST_S seconds.
//   coast     no match for longer than that. The last box is held, and the
//             message carries its age so the robot can decide what to do.
//   lost      no match for LOST_S seconds. The page drops the lock.
//
// Boxes are the detector's: {x, y, w, h, label, p}, top-left corner and
// size in pixels of the camera frame. The lock keeps the frame size it was
// started in, so the message can say which pixels it means.

export const COAST_S = 0.7;
export const LOST_S = 3.0;
// A tap on empty picture becomes a square this fraction of the frame's
// shorter side. At 480 rows that is 58 px, about a can at arm's length.
export const TAP_BOX = 0.12;
// A detection that covers this much of the held box is the same thing,
// whatever the network calls it this frame. YOLOX flips a can between
// bottle and cup; the box hardly moves.
export const SAME_SPOT_IOU = 0.5;

// The picture is object-fit: cover. The frame is scaled up until it fills
// the viewport, centred, and the overflow is cut off. This is the scale
// and the offset, in viewport pixels.
export function coverMap(W, H, fw, fh) {
  const s = Math.max(W / fw, H / fh);
  return { s, ox: (W - fw * s) / 2, oy: (H - fh * s) / 2 };
}

// A viewport point to frame pixels, and back.
export function toFrame(vx, vy, W, H, fw, fh) {
  const { s, ox, oy } = coverMap(W, H, fw, fh);
  return { x: (vx - ox) / s, y: (vy - oy) / s };
}
export function toViewport(x, y, W, H, fw, fh) {
  const { s, ox, oy } = coverMap(W, H, fw, fh);
  return { x: ox + x * s, y: oy + y * s };
}

function inside(b, x, y) {
  return x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h;
}
function centre(b) { return { x: b.x + b.w / 2, y: b.y + b.h / 2 }; }
function iou(a, b) {
  const x = Math.max(a.x, b.x), y = Math.max(a.y, b.y);
  const r = Math.min(a.x + a.w, b.x + b.w), t = Math.min(a.y + a.h, b.y + b.h);
  const over = Math.max(0, r - x) * Math.max(0, t - y);
  return over / (a.w * a.h + b.w * b.h - over);
}
// The smallest box under a point. Smallest, because a can in a hand sits
// inside the person's box, and the finger meant the can.
function under(boxes, x, y) {
  let best = null;
  for (const b of boxes) {
    if (inside(b, x, y) && (!best || b.w * b.h < best.w * best.h)) best = b;
  }
  return best;
}

// What a tap at frame pixel (x, y) means: the box under it, or a square
// around it when there is none. The square is kept inside the frame.
export function pick(boxes, x, y, fw, fh) {
  const b = under(boxes, x, y);
  if (b) return { x: b.x, y: b.y, w: b.w, h: b.h, label: b.label, p: b.p };
  const side = TAP_BOX * Math.min(fw, fh);
  const x0 = Math.max(0, Math.min(fw - side, x - side / 2));
  const y0 = Math.max(0, Math.min(fh - side, y - side / 2));
  return { x: x0, y: y0, w: side, h: side, label: null, p: 0 };
}

// Whether a drive frame carries an actual stick movement. A connected pad
// streams a frame every tick even at rest, all zeros after the dead zone,
// and a resting pad must not end the lock: only a moved stick does.
export function stickMoved(frame) {
  return !!frame && (frame.vx !== 0 || frame.vy !== 0 || frame.yaw !== 0);
}

export class Lock {
  // `box` is what pick() returned; `fw`, `fh` the frame it was picked in.
  constructor(box, fw, fh, t) {
    this.box = { x: box.x, y: box.y, w: box.w, h: box.h };
    this.label = box.label;
    this.p = box.p;
    this.fw = fw; this.fh = fh;
    this.started = t;
    this.seen = box.label ? t : null;   // last time a detection matched
  }

  state(t) {
    if (this.seen === null) return "waiting";
    const age = t - this.seen;
    return age <= COAST_S ? "locked" : age <= LOST_S ? "coast" : "lost";
  }

  // Seconds since a detection last matched. Zero while waiting: there is
  // nothing to be old.
  age(t) { return this.seen === null ? 0 : t - this.seen; }

  // How far a detection's centre may sit from the held box's centre and
  // still be the same thing. The box's own diagonal, at least a tenth of
  // the frame, and growing with the time since the last match, because a
  // target that walked during the coast is somewhere further along.
  gate(t) {
    const diag = Math.hypot(this.box.w, this.box.h);
    return Math.max(diag, 0.1 * Math.min(this.fw, this.fh)) * (1 + this.age(t));
  }

  // Fold in one pass of detections. Returns true when one matched.
  update(boxes, t) {
    if (this.seen === null) {
      // Waiting: the first detection that covers the tap adopts the lock.
      const c = centre(this.box);
      const b = under(boxes, c.x, c.y);
      if (!b) return false;
      this._take(b, t);
      return true;
    }
    const c = centre(this.box), gate = this.gate(t);
    let best = null, bestD = Infinity;
    for (const b of boxes) {
      if (b.label !== this.label && iou(b, this.box) < SAME_SPOT_IOU) continue;
      const bc = centre(b), d = Math.hypot(bc.x - c.x, bc.y - c.y);
      if (d <= gate && d < bestD) { best = b; bestD = d; }
    }
    if (!best) return false;
    this._take(best, t);
    return true;
  }

  _take(b, t) {
    this.box = { x: b.x, y: b.y, w: b.w, h: b.h };
    this.label = b.label; this.p = b.p;
    this.seen = t;
  }

  // Is a frame point on the held box. The page uses it to read a second
  // tap on the target as "let go".
  covers(x, y) { return inside(this.box, x, y); }

  // What the robot is told, ten times a second: the box's centre and size,
  // the frame those pixels belong to, and how stale the match is. Nothing
  // while waiting or lost.
  message(t) {
    const s = this.state(t);
    if (s === "waiting" || s === "lost") return null;
    const c = centre(this.box);
    return {
      t: "track",
      cx: Math.round(c.x), cy: Math.round(c.y),
      w: Math.round(this.box.w), h: Math.round(this.box.h),
      fw: this.fw, fh: this.fh,
      label: this.label,
      age: Math.round(this.age(t) * 100) / 100,
    };
  }
}
