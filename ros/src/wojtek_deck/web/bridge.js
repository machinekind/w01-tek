// Telemetry straight from foxglove_bridge: subscribe to the topics the
// panel charts, decode each frame with cdr.js, hand the object to a
// callback. Reconnects on its own; topics that appear later (the bridge
// advertises them as they come up) are picked up from the advertise frames.
import { decode, messageData, parseSchema } from "./cdr.js";

export class Bridge {
  // wanted: {topic: (msg, timestampNs) => void}
  // onState: (connected: bool, note: string) => void
  constructor(url, wanted, onState) {
    this.url = url;
    this.wanted = wanted;
    this.onState = onState || (() => {});
    this.subs = new Map();     // subscription id -> {topic, schema, cb}
    this.channels = new Map(); // channel id -> subscription id
    this.nextId = 1;
    this.ws = null;
    this.closed = false;
    this.connect();
  }

  connect() {
    if (this.closed) return;
    let ws;
    try {
      // foxglove_bridge 3.x speaks the SDK protocol name, older bridges
      // the original one; the server picks whichever it knows.
      ws = new WebSocket(this.url, ["foxglove.sdk.v1", "foxglove.websocket.v1"]);
    } catch (e) {
      this.onState(false, String(e));
      setTimeout(() => this.connect(), 3000);
      return;
    }
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onopen = () => this.onState(true, "connected");
    ws.onclose = () => {
      this.subs.clear(); this.channels.clear();
      this.onState(false, "retrying");
      setTimeout(() => this.connect(), 2000);
    };
    ws.onerror = () => {};
    ws.onmessage = (e) => {
      if (typeof e.data === "string") this.onJson(JSON.parse(e.data));
      else this.onBinary(e.data);
    };
  }

  close() { this.closed = true; if (this.ws) this.ws.close(); }

  onJson(m) {
    if (m.op === "advertise") {
      const subscriptions = [];
      for (const ch of m.channels) {
        const cb = this.wanted[ch.topic];
        if (!cb || ch.encoding !== "cdr" || ch.schemaEncoding !== "ros2msg") continue;
        if (this.channels.has(ch.id)) continue;
        let schema;
        try { schema = parseSchema(ch.schema, ch.schemaName); }
        catch (err) { console.warn("schema", ch.topic, err); continue; }
        const id = this.nextId++;
        this.subs.set(id, { topic: ch.topic, schema, cb });
        this.channels.set(ch.id, id);
        subscriptions.push({ id, channelId: ch.id });
      }
      if (subscriptions.length)
        this.ws.send(JSON.stringify({ op: "subscribe", subscriptions }));
    } else if (m.op === "unadvertise") {
      for (const chId of m.channelIds) {
        const id = this.channels.get(chId);
        if (id !== undefined) { this.subs.delete(id); this.channels.delete(chId); }
      }
    }
    // serverInfo, status, etc.: nothing to do
  }

  onBinary(buf) {
    const md = messageData(buf);
    if (!md) return;
    const sub = this.subs.get(md.subscriptionId);
    if (!sub) return;
    let msg;
    try { msg = decode(sub.schema, md.reader); }
    catch (err) { console.warn("decode", sub.topic, err); return; }
    sub.cb(msg, md.timestampNs);
  }

  topics() { return [...this.subs.values()].map(s => s.topic); }
}
