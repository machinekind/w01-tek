// ROS 2 messages in the browser, with no library: a ros2msg schema parser
// and a CDR reader. foxglove_bridge advertises every ROS 2 topic as
// encoding "cdr" with the message definition attached as text (schema
// encoding "ros2msg"), which is all we need to turn its binary frames into
// plain objects. Covers what the robot publishes: primitives, strings,
// fixed and variable arrays, nested messages, builtin_interfaces/Time.

const SIZES = {
  bool: 1, byte: 1, char: 1, int8: 1, uint8: 1, int16: 2, uint16: 2,
  int32: 4, uint32: 4, float32: 4, int64: 8, uint64: 8, float64: 8,
};
const decoder = new TextDecoder();

// "std_msgs/msg/Header" and "std_msgs/Header" are the same type.
function canon(name) { return name.replace("/msg/", "/"); }

// Parse a ros2msg schema (root definition first, dependencies after
// "====" separators with a "MSG: pkg/Name" line) into a lookup of
// {name -> [{type, name, array, length}]}.
export function parseSchema(text, rootName) {
  const defs = new Map();
  const sections = text.split(/^={10,}\s*$/m);
  sections.forEach((section, i) => {
    let name = i === 0 ? canon(rootName) : null;
    const fields = [];
    for (let line of section.split("\n")) {
      const m = /^MSG:\s*(\S+)/.exec(line);
      if (m) { name = canon(m[1]); continue; }
      line = line.replace(/#.*$/, "").trim();
      if (!line) continue;
      const f = /^([\w/]+)(?:<=\d+)?(?:\[(<=)?(\d*)\])?\s+(\w+)\s*(.*)$/.exec(line);
      if (!f) continue;
      const [, type, , len, fname, rest] = f;
      if (rest.startsWith("=")) continue;           // a constant, not a field
      const isArray = line.includes("[");
      fields.push({
        type: canon(type), name: fname, array: isArray,
        // fixed length or null for a sequence
        length: isArray && len !== "" ? parseInt(len, 10) : null,
      });
    }
    if (name) defs.set(name, fields);
  });
  return { root: canon(rootName), defs };
}

// Find a field's type: qualified, or by bare name within the package of the
// message that uses it, or by bare name anywhere (foxglove writes some
// dependencies unqualified).
function resolve(defs, type, fromType) {
  if (defs.has(type)) return type;
  const pkg = fromType.split("/")[0];
  if (defs.has(`${pkg}/${type}`)) return `${pkg}/${type}`;
  for (const key of defs.keys()) if (key.endsWith(`/${type}`)) return key;
  return null;
}

export class CdrReader {
  constructor(buffer, byteOffset = 0, byteLength) {
    this.dv = new DataView(buffer, byteOffset, byteLength);
    this.u8s = new Uint8Array(buffer, byteOffset, byteLength);
    // 4-byte encapsulation header: 00 01 = little-endian XCDR1.
    this.le = this.dv.getUint8(1) === 1;
    this.pos = 4;
  }
  align(n) {
    const rel = this.pos - 4;
    this.pos += (n - (rel % n)) % n;
  }
  prim(type) {
    const n = SIZES[type];
    if (n > 1) this.align(n);
    const p = this.pos; this.pos += n;
    switch (type) {
      case "bool": return this.dv.getUint8(p) !== 0;
      case "byte": case "uint8": case "char": return this.dv.getUint8(p);
      case "int8": return this.dv.getInt8(p);
      case "int16": return this.dv.getInt16(p, this.le);
      case "uint16": return this.dv.getUint16(p, this.le);
      case "int32": return this.dv.getInt32(p, this.le);
      case "uint32": return this.dv.getUint32(p, this.le);
      case "float32": return this.dv.getFloat32(p, this.le);
      case "float64": return this.dv.getFloat64(p, this.le);
      case "int64": return Number(this.dv.getBigInt64(p, this.le));
      case "uint64": return Number(this.dv.getBigUint64(p, this.le));
      default: throw new Error(`unknown primitive ${type}`);
    }
  }
  string() {
    this.align(4);
    const len = this.dv.getUint32(this.pos, this.le); this.pos += 4;
    const s = decoder.decode(this.u8s.subarray(this.pos, this.pos + Math.max(0, len - 1)));
    this.pos += len;
    return s;
  }
  count() { this.align(4); const n = this.dv.getUint32(this.pos, this.le); this.pos += 4; return n; }
}

export function decode(schema, reader, typeName = schema.root) {
  const fields = schema.defs.get(typeName);
  if (!fields) throw new Error(`no definition for ${typeName}`);
  const out = {};
  for (const f of fields) {
    const one = () => {
      if (f.type === "string" || f.type === "wstring") return reader.string();
      if (f.type in SIZES) return reader.prim(f.type);
      const t = resolve(schema.defs, f.type, typeName);
      if (!t) throw new Error(`unresolved type ${f.type} in ${typeName}`);
      return decode(schema, reader, t);
    };
    if (!f.array) { out[f.name] = one(); continue; }
    const n = f.length ?? reader.count();
    const arr = new Array(n);
    for (let i = 0; i < n; i++) arr[i] = one();
    out[f.name] = arr;
  }
  return out;
}

// Decode one foxglove MessageData frame (opcode 1): returns
// {subscriptionId, timestampNs, reader} or null for any other opcode.
export function messageData(arrayBuffer) {
  const dv = new DataView(arrayBuffer);
  if (dv.getUint8(0) !== 1) return null;
  return {
    subscriptionId: dv.getUint32(1, true),
    timestampNs: Number(dv.getBigUint64(5, true)),
    reader: new CdrReader(arrayBuffer, 13, arrayBuffer.byteLength - 13),
  };
}
