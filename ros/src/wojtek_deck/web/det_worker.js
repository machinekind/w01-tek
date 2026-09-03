// The detector, off the main thread.
//
// The page hands this worker camera frames and gets boxes back. Everything
// heavy happens here -- loading a 26 MB runtime, resizing a frame, running
// the network -- so the panel's charts and the pad keep their frame rate
// while the network thinks.
//
// This is a MODULE worker (`new Worker(url, {type: "module"})`), not the
// classic kind. onnxruntime-web 1.29 ships its entry as an ES module and
// loads its WASM half with a dynamic import, and a classic worker cannot do
// either. So: module worker, `import`, no importScripts.
//
// Messages in
//   {t: "frame", bmp, w, h}   an ImageBitmap of a camera frame, transferred
// Messages out
//   {t: "ready"}              send the next frame now
//   {t: "det", w, h, boxes, ms, backend}   backend is "gpu" or "cpu"
//   {t: "missing"}            the assets are not on the robot
//   {t: "error", msg}         anything else went wrong
//   {t: "log", msg}           worth a line in the panel's log, not a failure
//
// One frame at a time on purpose. The camera runs faster than the network,
// and a queue would only build a backlog of stale frames; "ready" is the
// worker saying it has caught up.
import * as ort from "/det/ort.webgpu.min.mjs";
import { letterboxScale, postprocess } from "./yolox.js";

let cfg = null, session = null, backend = "";
let canvas = null, ctx = null, input = 0;

// The tensor the network wants, built from one frame.
//
// Fit the frame into the square top-left and leave the rest flat grey, the
// same 114 the model was trained with -- an empty corner it has seen before
// is one it ignores. Then hand over the pixels as three planes, blue first:
// YOLOX was trained on images read by OpenCV, which reads blue first, and
// on raw 0-255 values with no rescaling (the version that divided by a mean
// and a deviation is older than this model).
function preprocess(bmp) {
  const r = Math.min(input / bmp.width, input / bmp.height);
  ctx.fillStyle = `rgb(${cfg.pad},${cfg.pad},${cfg.pad})`;
  ctx.fillRect(0, 0, input, input);
  ctx.drawImage(bmp, 0, 0, Math.round(bmp.width * r), Math.round(bmp.height * r));
  const px = ctx.getImageData(0, 0, input, input).data;
  const n = input * input;
  const data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    data[i] = px[i * 4 + 2];          // blue
    data[n + i] = px[i * 4 + 1];      // green
    data[2 * n + i] = px[i * 4];      // red
  }
  return new ort.Tensor("float32", data, [1, 3, input, input]);
}

async function load() {
  const res = await fetch("./yolox.json");
  cfg = await res.json();
  input = cfg.input;
  canvas = new OffscreenCanvas(input, input);
  // willReadFrequently: every frame is drawn and then read straight back,
  // which is the case the flag exists for.
  ctx = canvas.getContext("2d", { willReadFrequently: true });

  const model = await fetch(`/det/${cfg.model}`);
  if (!model.ok) { self.postMessage({ t: "missing" }); return; }
  const bytes = new Uint8Array(await model.arrayBuffer());

  // The runtime looks for its WASM half next to this path.
  ort.env.wasm.wasmPaths = "/det/";
  // One thread, deliberately. Left to itself the runtime opens a pool of
  // them, and a pool started from inside a worker never comes up: the
  // threads are workers of a worker, and they hang before the first one
  // reports for duty. The page is cross-origin isolated and shared memory
  // is there -- it is the nesting that breaks, and it hangs rather than
  // failing, which would leave the panel waiting for boxes forever.
  // Measured on a laptop at 416x416: 10 ms a frame on the GPU, 44 on one
  // CPU thread, both faster than the camera, so this costs nothing today.
  ort.env.wasm.numThreads = 1;
  const opts = { graphOptimizationLevel: "all" };

  // GPU first, CPU behind it. Two separate attempts rather than one list,
  // because a session built from a list does not say which provider it
  // ended up on, and the panel puts that word on a lamp. Asking twice costs
  // nothing: the first attempt only fails on a machine with no WebGPU, and
  // then it fails immediately.
  if (self.navigator.gpu) {
    try {
      session = await ort.InferenceSession.create(
        bytes, { ...opts, executionProviders: ["webgpu"] });
      backend = "gpu";
    } catch (err) {
      self.postMessage({ t: "log", msg: `webgpu unavailable (${err})` });
    }
  }
  if (!session) {
    session = await ort.InferenceSession.create(
      bytes, { ...opts, executionProviders: ["wasm"] });
    backend = "cpu";
  }
  self.postMessage({ t: "ready" });
}

async function detect(bmp, w, h) {
  const t0 = performance.now();
  const tensor = preprocess(bmp);
  bmp.close();
  const out = await session.run({ [session.inputNames[0]]: tensor });
  const raw = out[session.outputNames[0]].data;
  const { scale } = letterboxScale(w, h, input);
  const boxes = postprocess(raw, cfg, scale, { w, h });
  self.postMessage({
    t: "det", w, h, boxes, backend,
    ms: performance.now() - t0,
  });
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.t !== "frame") return;
  if (!session) { m.bmp.close(); return; }
  try {
    await detect(m.bmp, m.w, m.h);
  } catch (err) {
    self.postMessage({ t: "error", msg: String(err) });
  }
  // Ask for the next frame whether or not that one worked, otherwise one
  // bad frame stops the detector for good.
  self.postMessage({ t: "ready" });
};

load().catch(err => self.postMessage({ t: "error", msg: String(err) }));
