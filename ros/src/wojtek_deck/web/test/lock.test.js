// node --test web/test   (lock.js is plain arithmetic, no browser needed)
//
// The lock has to do three things right: read a tap as the box the finger
// meant, keep hold of that box while the detector redraws it every frame,
// and let go at the right moment. Each test feeds hand-made boxes and a
// hand-set clock, so the answers can be checked exactly.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  COAST_S, LOST_S, Lock, TAP_BOX, coverMap, pick, stickMoved, toFrame, toViewport,
} from "../lock.js";

const FW = 640, FH = 480;
const person = { x: 200, y: 100, w: 120, h: 300, label: "person", p: 0.9 };
const can = { x: 250, y: 220, w: 30, h: 60, label: "bottle", p: 0.5 };
const chair = { x: 20, y: 200, w: 100, h: 150, label: "chair", p: 0.7 };
const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} is not ${b}`);

test("cover mapping fills a wider viewport and cuts the top and bottom", () => {
  // A 640x480 frame on a 1280x800 screen scales by 2 (the width is the
  // tighter fit), so the picture is 1280x960 and 80 rows are cut off each
  // end.
  const m = coverMap(1280, 800, FW, FH);
  close(m.s, 2); close(m.ox, 0); close(m.oy, -80);
});

test("viewport to frame and back is the identity", () => {
  const f = toFrame(700, 300, 1280, 800, FW, FH);
  close(f.x, 350); close(f.y, 190);
  const v = toViewport(f.x, f.y, 1280, 800, FW, FH);
  close(v.x, 700); close(v.y, 300);
});

test("a tap picks the box under it", () => {
  const b = pick([person, chair], 260, 150, FW, FH);
  assert.equal(b.label, "person");
  assert.equal(b.x, person.x);
});

test("a tap picks the smallest of the boxes under it", () => {
  // The can sits inside the person's box; the finger meant the can.
  const b = pick([person, can], 265, 250, FW, FH);
  assert.equal(b.label, "bottle");
});

test("a tap on empty picture makes a square around the finger", () => {
  const b = pick([person], 500, 400, FW, FH);
  const side = TAP_BOX * FH;
  assert.equal(b.label, null);
  close(b.w, side); close(b.h, side);
  close(b.x + b.w / 2, 500); close(b.y + b.h / 2, 400);
});

test("the tap square stays inside the frame", () => {
  const b = pick([], 2, 478, FW, FH);
  assert.equal(b.x, 0);
  close(b.y + b.h, FH);
});

test("a lock on a box is locked at once and says so to the robot", () => {
  const lk = new Lock(pick([person], 260, 150, FW, FH), FW, FH, 10);
  assert.equal(lk.state(10), "locked");
  const m = lk.message(10);
  assert.equal(m.t, "track");
  assert.equal(m.cx, 260); assert.equal(m.cy, 250);
  assert.equal(m.w, 120); assert.equal(m.h, 300);
  assert.equal(m.fw, FW); assert.equal(m.fh, FH);
  assert.equal(m.label, "person");
  assert.equal(m.age, 0);
});

test("the lock follows the same label to its new place", () => {
  const lk = new Lock(pick([person], 260, 150, FW, FH), FW, FH, 10);
  const moved = { ...person, x: 230 };
  assert.ok(lk.update([chair, moved], 10.1));
  assert.equal(lk.box.x, 230);
  assert.equal(lk.message(10.1).cx, 290);
});

test("the lock takes the nearest of two boxes with the label", () => {
  const lk = new Lock(pick([person], 260, 150, FW, FH), FW, FH, 10);
  const near = { ...person, x: 210 }, far = { ...person, x: 400 };
  lk.update([far, near], 10.1);
  assert.equal(lk.box.x, 210);
});

test("a box with another label somewhere else is not the target", () => {
  const lk = new Lock(pick([person], 260, 150, FW, FH), FW, FH, 10);
  assert.equal(lk.update([chair], 10.1), false);
  assert.equal(lk.box.x, person.x);
});

test("a box on the same spot keeps the lock whatever it is called", () => {
  // The network calls the can a cup this frame. Same place, same thing.
  const lk = new Lock(pick([can], 265, 250, FW, FH), FW, FH, 10);
  const cup = { ...can, x: 252, label: "cup" };
  assert.ok(lk.update([cup], 10.1));
  assert.equal(lk.label, "cup");
  assert.equal(lk.message(10.1).label, "cup");
});

test("a box with the label but far away is not the target", () => {
  const lk = new Lock(pick([can], 265, 250, FW, FH), FW, FH, 10);
  const other = { ...can, x: 550, y: 400 };
  assert.equal(lk.update([other], 10.1), false);
});

test("the gate widens while the target is out of sight", () => {
  // Gone for two seconds, the can may be well away from where it was.
  const lk = new Lock(pick([can], 265, 250, FW, FH), FW, FH, 10);
  const other = { ...can, x: 400, y: 300 };
  assert.equal(lk.update([other], 10.1), false);
  assert.ok(lk.update([other], 12.0));
});

test("without a match the lock coasts, then is lost", () => {
  const lk = new Lock(pick([person], 260, 150, FW, FH), FW, FH, 10);
  assert.equal(lk.state(10 + COAST_S), "locked");
  assert.equal(lk.state(10 + COAST_S + 0.1), "coast");
  const m = lk.message(11.5);
  assert.equal(m.cx, 260);            // the last box is held
  close(m.age, 1.5, 0.01);
  assert.equal(lk.state(10 + LOST_S + 0.1), "lost");
  assert.equal(lk.message(10 + LOST_S + 0.1), null);
});

test("a match during the coast makes it locked again", () => {
  const lk = new Lock(pick([person], 260, 150, FW, FH), FW, FH, 10);
  assert.equal(lk.state(12), "coast");
  lk.update([person], 12);
  assert.equal(lk.state(12), "locked");
  assert.equal(lk.message(12).age, 0);
});

test("a tap on empty picture waits and sends nothing", () => {
  const lk = new Lock(pick([], 500, 400, FW, FH), FW, FH, 10);
  assert.equal(lk.state(10), "waiting");
  assert.equal(lk.state(100), "waiting");   // it does not time out
  assert.equal(lk.message(10), null);
  assert.equal(lk.update([person], 10.1), false);
});

test("a detection over the tap adopts a waiting lock", () => {
  const lk = new Lock(pick([], 265, 250, FW, FH), FW, FH, 10);
  assert.ok(lk.update([person, can], 12));
  assert.equal(lk.label, "bottle");     // the smaller of the two
  assert.equal(lk.state(12), "locked");
  assert.equal(lk.message(12).label, "bottle");
});

test("a resting pad is not a stick movement, a nudged one is", () => {
  // A connected pad streams zeros every tick; that must not end a lock.
  assert.equal(stickMoved(null), false);
  assert.equal(stickMoved({ vx: 0, vy: 0, yaw: 0 }), false);
  assert.equal(stickMoved({ vx: 0, vy: 0, yaw: 0.3 }), true);
  assert.equal(stickMoved({ vx: -0.5, vy: 0, yaw: 0 }), true);
});

test("covers says whether a second tap landed on the target", () => {
  const lk = new Lock(pick([person], 260, 150, FW, FH), FW, FH, 10);
  assert.ok(lk.covers(210, 300));
  assert.equal(lk.covers(10, 10), false);
});
