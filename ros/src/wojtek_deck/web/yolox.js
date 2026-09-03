// Turning YOLOX-nano's output numbers into boxes on the camera frame.
//
// This is the arithmetic half of the detector and nothing else: no DOM, no
// runtime, no fetch. det_worker.js runs the network and calls in here; the
// tests in test/yolox.test.js call the same functions from node.
//
// It follows the reference the YOLOX authors ship with the model --
// demo_postprocess and multiclass_nms in demo/ONNXRuntime/onnx_inference.py
// -- because a decode that disagrees with the network's training puts the
// boxes in the wrong place, and there is no way to tell by looking at one
// number. The settings live in yolox.json.

// How the frame was fitted into the square the network wants.
//
// YOLOX scales the frame down until it fits and leaves the rest of the
// square grey, with the picture in the top-left corner. So there is no
// padding to subtract later, only the scale to undo. dw and dh are here to
// say that out loud and to leave a place for a model that centres instead.
export function letterboxScale(w, h, input) {
  return { scale: Math.min(input / w, input / h), dw: 0, dh: 0 };
}

// How many rows the network answers with, for a given input size. One row
// per cell of each grid: 52x52 + 26x26 + 13x13 = 3549 at 416.
export function rowCount(input, strides) {
  return strides.reduce((n, s) => n + (input / s) * (input / s), 0);
}

// Read the raw output into candidate boxes, in the coordinates of the
// square the network saw.
//
// The output is one row per grid cell, [cx, cy, w, h, obj, ...80 classes].
// The rows run stride by stride, and within a stride row by row of the
// grid. A cell says where in itself the box sits (cx, cy are offsets in
// cells, w and h are logarithms of a size in cells), so undoing that is a
// shift by the cell's position and a multiply by the stride.
//
// The score of a box is how sure the network is that there is anything
// there at all, times how sure it is of the best class.
export function decode(output, input, strides, conf) {
  const out = [];
  // Row width from the data rather than a constant: 4 box numbers, one
  // objectness, then one score per class.
  const cols = output.length / rowCount(input, strides);
  const nc = cols - 5;
  let row = 0;
  for (const s of strides) {
    const cells = input / s;
    for (let gy = 0; gy < cells; gy++) {
      for (let gx = 0; gx < cells; gx++, row++) {
        const o = row * cols;
        const obj = output[o + 4];
        // The class scores cost 80 reads per row and only the winner
        // matters, so skip the row when objectness alone cannot clear the
        // threshold -- class scores are at most 1, they cannot rescue it.
        if (obj < conf) continue;
        let cls = 0, best = output[o + 5];
        for (let c = 1; c < nc; c++) {
          const v = output[o + 5 + c];
          if (v > best) { best = v; cls = c; }
        }
        const p = obj * best;
        if (p < conf) continue;
        const w = Math.exp(output[o + 2]) * s;
        const h = Math.exp(output[o + 3]) * s;
        out.push({
          x: (output[o] + gx) * s - w / 2,
          y: (output[o + 1] + gy) * s - h / 2,
          w, h, cls, p,
        });
      }
    }
  }
  return out;
}

function iou(a, b) {
  const x = Math.max(a.x, b.x), y = Math.max(a.y, b.y);
  const r = Math.min(a.x + a.w, b.x + b.w), t = Math.min(a.y + a.h, b.y + b.h);
  const over = Math.max(0, r - x) * Math.max(0, t - y);
  return over / (a.w * a.h + b.w * b.h - over);
}

// Drop the duplicates. The network fires on several cells around one
// object, so keep the most confident box and throw away the ones that sit
// on top of it. Class by class: a person standing in front of a car is two
// overlapping boxes and both should survive.
export function nms(boxes, iouLimit) {
  const kept = [];
  for (const b of [...boxes].sort((p, q) => q.p - p.p)) {
    if (!kept.some(k => k.cls === b.cls && iou(k, b) > iouLimit)) kept.push(b);
  }
  return kept;
}

// The whole trip: raw output in, boxes on the original frame out.
//
// `scale` is the letterbox scale that got the frame into the square, so
// dividing by it walks the boxes back to the frame's own pixels. `frame`
// is optional: give it {w, h} and boxes that hang off the edge are cut to
// fit, which is what the reference does and what the overlay wants.
export function postprocess(output, cfg, scale, frame) {
  const boxes = nms(decode(output, cfg.input, cfg.strides, cfg.conf), cfg.iou);
  return boxes.map(b => {
    let x = b.x / scale, y = b.y / scale;
    let w = b.w / scale, h = b.h / scale;
    if (frame) {
      const x1 = Math.min(frame.w, x + w), y1 = Math.min(frame.h, y + h);
      x = Math.max(0, x); y = Math.max(0, y);
      w = Math.max(0, x1 - x); h = Math.max(0, y1 - y);
    }
    return { x, y, w, h, label: cfg.classes[b.cls], p: b.p };
  });
}
