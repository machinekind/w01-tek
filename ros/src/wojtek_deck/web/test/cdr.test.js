// node --test web/test   (no browser needed; cdr.js is plain JS)
import { test } from "node:test";
import assert from "node:assert/strict";
import { CdrReader, decode, messageData, parseSchema } from "../cdr.js";

// A tiny CDR writer for building test vectors with the same rules the
// reader must follow: 4-byte header, alignment relative to the body.
class W {
  constructor() { this.bytes = [0, 1, 0, 0]; }
  align(n) { while ((this.bytes.length - 4) % n) this.bytes.push(0); }
  num(type, v) {
    const size = { uint8: 1, int32: 4, uint32: 4, float32: 4, float64: 8, uint64: 8 }[type];
    this.align(size);
    const b = new ArrayBuffer(size), dv = new DataView(b);
    if (type === "uint8") dv.setUint8(0, v);
    else if (type === "int32") dv.setInt32(0, v, true);
    else if (type === "uint32") dv.setUint32(0, v, true);
    else if (type === "float32") dv.setFloat32(0, v, true);
    else if (type === "float64") dv.setFloat64(0, v, true);
    else dv.setBigUint64(0, BigInt(v), true);
    this.bytes.push(...new Uint8Array(b));
    return this;
  }
  str(s) {
    const enc = new TextEncoder().encode(s);
    this.num("uint32", enc.length + 1);
    this.bytes.push(...enc, 0);
    return this;
  }
  buffer() { return new Uint8Array(this.bytes).buffer; }
}

const HEADER = `std_msgs/Header header
float32 inference_ms
float32 period_ms
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
`;

test("nested message with header and strings", () => {
  const schema = parseSchema(HEADER, "wojtek_telemetry/msg/PolicyTiming");
  const buf = new W().num("int32", 12).num("uint32", 34).str("base")
    .num("float32", 1.5).num("float32", 20.25).buffer();
  const msg = decode(schema, new CdrReader(buf));
  assert.deepEqual(msg, {
    header: { stamp: { sec: 12, nanosec: 34 }, frame_id: "base" },
    inference_ms: 1.5, period_ms: 20.25,
  });
});

test("alignment after an odd-length string", () => {
  // "abc" + NUL = 4 bytes after the length; "ab" + NUL = 3 -> the float64
  // that follows needs padding to an 8-byte boundary.
  const schema = parseSchema("string s\nfloat64 x\n", "t/msg/T");
  const buf = new W().str("ab").num("float64", 2.5).buffer();
  const msg = decode(schema, new CdrReader(buf));
  assert.equal(msg.s, "ab");
  assert.equal(msg.x, 2.5);
});

test("fixed arrays, sequences, bools and constants", () => {
  const text = `uint8 FLAG=1   # a constant, not a field
bool ok
float64[2] cov
float32[] vals
string[] names
uint64 big
`;
  const schema = parseSchema(text, "t/msg/T");
  const w = new W().num("uint8", 1).num("float64", 1).num("float64", -1)
    .num("uint32", 3).num("float32", 0.5).num("float32", 1.5).num("float32", 2.5)
    .num("uint32", 2).str("a").str("bcd").num("uint64", 4294967296);
  const msg = decode(schema, new CdrReader(w.buffer()));
  assert.equal(msg.ok, true);
  assert.deepEqual(msg.cov, [1, -1]);
  assert.deepEqual(msg.vals, [0.5, 1.5, 2.5]);
  assert.deepEqual(msg.names, ["a", "bcd"]);
  assert.equal(msg.big, 4294967296);
  assert.equal("FLAG" in msg, false);
});

test("unqualified dependency resolves within the package", () => {
  const text = `Vector3 linear
Vector3 angular
================================================================================
MSG: geometry_msgs/msg/Vector3
float64 x
float64 y
float64 z
`;
  const schema = parseSchema(text, "geometry_msgs/msg/Twist");
  const w = new W();
  for (const v of [1, 2, 3, 4, 5, 6]) w.num("float64", v);
  const msg = decode(schema, new CdrReader(w.buffer()));
  assert.deepEqual(msg.angular, { x: 4, y: 5, z: 6 });
});

test("foxglove MessageData framing", () => {
  const body = new W().num("int32", 7).buffer();
  const frame = new Uint8Array(13 + body.byteLength);
  const dv = new DataView(frame.buffer);
  dv.setUint8(0, 1);
  dv.setUint32(1, 42, true);
  dv.setBigUint64(5, 123456789n, true);
  frame.set(new Uint8Array(body), 13);
  const md = messageData(frame.buffer);
  assert.equal(md.subscriptionId, 42);
  assert.equal(md.timestampNs, 123456789);
  const schema = parseSchema("int32 v\n", "t/msg/T");
  assert.equal(decode(schema, md.reader).v, 7);
  assert.equal(messageData(new Uint8Array([2, 0, 0, 0, 0]).buffer), null);
});
