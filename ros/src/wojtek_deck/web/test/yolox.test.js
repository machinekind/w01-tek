// node --test web/test   (no browser needed; yolox.js is plain arithmetic)
//
// The point of these tests is that the boxes land where the network meant
// them to. A decode that is off by half a cell, or that reads the grids in
// the wrong order, still produces plausible-looking boxes -- it just draws
// them next to the dog instead of on it. So the tests build an output
// tensor by hand, with candidates in cells whose pixel answer can be worked
// out on paper, and check the exact numbers.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { decode, letterboxScale, nms, postprocess, rowCount } from "../yolox.js";

const cfg = JSON.parse(readFileSync(new URL("../yolox.json", import.meta.url)));
const COLS = 85;                       // 4 box + 1 objectness + 80 classes

// An empty output tensor of the right size, and a way to fill one row.
function blank(input = cfg.input) {
  return new Float32Array(rowCount(input, cfg.strides) * COLS);
}
// The index of the row for a cell, counting stride by stride and then row
// by row within the grid -- the order the network answers in.
function rowOf(stride, gx, gy, input = cfg.input) {
  let row = 0;
  for (const s of cfg.strides) {
    if (s === stride) return row + gy * (input / s) + gx;
    row += (input / s) * (input / s);
  }
  throw new Error(`no stride ${stride}`);
}
function put(out, stride, gx, gy, { cx, cy, w, h, obj, cls, p }) {
  const o = rowOf(stride, gx, gy) * COLS;
  out[o] = cx; out[o + 1] = cy;
  out[o + 2] = Math.log(w); out[o + 3] = Math.log(h);   // sizes are logged
  out[o + 4] = obj;
  out[o + 5 + cls] = p;
  return out;
}
const close = (a, b, eps = 1e-4) =>
  assert.ok(Math.abs(a - b) < eps, `${a} is not ${b}`);

test("the tensor has one row per grid cell: 3549 at 416", () => {
  assert.equal(rowCount(416, [8, 16, 32]), 52 * 52 + 26 * 26 + 13 * 13);
  assert.equal(rowCount(416, [8, 16, 32]), 3549);
});

test("a cell decodes to the pixel box it stands for", () => {
  // Cell (10, 4) of the stride-8 grid, half a cell right and down of its
  // corner, four cells wide and two tall. Centre is therefore
  // (10.5 * 8, 4.5 * 8) = (84, 36), size (32, 16), so the corner is at
  // (84 - 16, 36 - 8) = (68, 28).
  const out = blank();
  put(out, 8, 10, 4, { cx: .5, cy: .5, w: 4, h: 2, obj: .9, cls: 16, p: .8 });
  const boxes = decode(out, cfg.input, cfg.strides, cfg.conf);
  assert.equal(boxes.length, 1);
  close(boxes[0].x, 68); close(boxes[0].y, 28);
  close(boxes[0].w, 32); close(boxes[0].h, 16);
  assert.equal(boxes[0].cls, 16);
  close(boxes[0].p, .72);
});

test("the coarse grid is read after the fine one", () => {
  // Cell (2, 3) of the stride-32 grid: centre (2.5 * 32, 3.5 * 32) =
  // (80, 112). Getting the grid order wrong puts this box somewhere in the
  // stride-8 grid instead, at a quarter of the coordinates.
  const out = blank();
  put(out, 32, 2, 3, { cx: .5, cy: .5, w: 3, h: 3, obj: 1, cls: 2, p: .9 });
  const boxes = decode(out, cfg.input, cfg.strides, cfg.conf);
  assert.equal(boxes.length, 1);
  close(boxes[0].x + boxes[0].w / 2, 80);
  close(boxes[0].y + boxes[0].h / 2, 112);
  close(boxes[0].w, 96);
});

test("a row below the threshold is dropped", () => {
  const out = blank();
  // 0.9 * 0.2 = 0.18, under the 0.3 the settings ask for.
  put(out, 8, 5, 5, { cx: 0, cy: 0, w: 2, h: 2, obj: .9, cls: 0, p: .2 });
  assert.equal(decode(out, cfg.input, cfg.strides, cfg.conf).length, 0);
  // and a low objectness with a certain class goes the same way
  const out2 = blank();
  put(out2, 8, 5, 5, { cx: 0, cy: 0, w: 2, h: 2, obj: .1, cls: 0, p: 1 });
  assert.equal(decode(out2, cfg.input, cfg.strides, cfg.conf).length, 0);
});

test("nms drops the weaker of two boxes on the same thing", () => {
  const a = { x: 0, y: 0, w: 100, h: 100, cls: 0, p: .9 };
  const b = { x: 5, y: 5, w: 100, h: 100, cls: 0, p: .6 };   // heavy overlap
  const kept = nms([b, a], cfg.iou);
  assert.equal(kept.length, 1);
  assert.equal(kept[0].p, .9);
});

test("nms keeps two classes standing on the same spot", () => {
  const person = { x: 0, y: 0, w: 100, h: 100, cls: 0, p: .9 };
  const bike = { x: 2, y: 2, w: 100, h: 100, cls: 1, p: .6 };
  assert.equal(nms([person, bike], cfg.iou).length, 2);
});

test("nms keeps two boxes of one class that barely touch", () => {
  const a = { x: 0, y: 0, w: 100, h: 100, cls: 0, p: .9 };
  const b = { x: 90, y: 0, w: 100, h: 100, cls: 0, p: .6 };  // 10% overlap
  assert.equal(nms([a, b], cfg.iou).length, 2);
});

test("a 640x360 frame scales back to its own pixels", () => {
  // 416/640 is the tighter fit, so the whole frame lands 0.65 the size and
  // the bottom of the square stays grey.
  const { scale, dw, dh } = letterboxScale(640, 360, cfg.input);
  close(scale, .65);
  assert.equal(dw, 0); assert.equal(dh, 0);

  // A box on cell (10, 4) of the fine grid was (68, 28, 32, 16) in the
  // square; divided by 0.65 that is (104.6..., 43.07..., 49.2..., 24.6...).
  const out = blank();
  put(out, 8, 10, 4, { cx: .5, cy: .5, w: 4, h: 2, obj: .9, cls: 16, p: .8 });
  const [box] = postprocess(out, cfg, scale, { w: 640, h: 360 });
  close(box.x, 68 / .65); close(box.y, 28 / .65);
  close(box.w, 32 / .65); close(box.h, 16 / .65);
  assert.equal(box.label, "dog");
  close(box.p, .72);
});

test("postprocess names the classes and suppresses the duplicate", () => {
  // Two cells firing on one dog, plus a car far away in the coarse grid.
  const out = blank();
  put(out, 8, 10, 4, { cx: .5, cy: .5, w: 4, h: 2, obj: .9, cls: 16, p: .8 });
  put(out, 8, 11, 4, { cx: -.4, cy: .4, w: 4, h: 2, obj: .8, cls: 16, p: .8 });
  put(out, 32, 10, 8, { cx: 0, cy: 0, w: 2, h: 2, obj: .9, cls: 2, p: .9 });
  // Both dog cells alone decode, so the pruning below is nms doing it.
  assert.equal(decode(out, cfg.input, cfg.strides, cfg.conf).length, 3);

  const boxes = postprocess(out, cfg, 1, { w: cfg.input, h: cfg.input });
  assert.deepEqual(boxes.map(b => b.label).sort(), ["car", "dog"]);
  const dog = boxes.find(b => b.label === "dog");
  close(dog.p, .72);          // the stronger of the two dog cells
  close(dog.x, 68);
});

test("a box hanging off the frame is cut to fit", () => {
  const out = blank();
  // Cell (0, 0) with a 60-cell box: 480 px across, centred on the frame's
  // top-left corner, so it pokes out on all four sides and what is left
  // after cutting is the frame itself.
  put(out, 8, 0, 0, { cx: 0, cy: 0, w: 60, h: 60, obj: 1, cls: 0, p: 1 });
  const [box] = postprocess(out, cfg, 1, { w: 200, h: 100 });
  close(box.x, 0); close(box.y, 0);
  close(box.w, 200); close(box.h, 100);
});

test("the settings file matches what the code assumes", () => {
  assert.equal(cfg.classes.length, 80);
  assert.equal(cfg.classes[0], "person");
  assert.equal(cfg.channels, "bgr");
  assert.equal(cfg.pad, 114);
});
