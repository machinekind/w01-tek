// A strip chart for the HUD, no library: a few series over a sliding time
// window, one colour, identity by dash pattern, direct labels at the right
// edge. Colours come from CSS custom properties so the palette lives in
// one place.

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export class Strip {
  // opts: {series: [{name, dash}], window: seconds, min, max, fixed}
  constructor(canvas, opts) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.opts = { window: 10, fixed: 2, ...opts };
    this.series = this.opts.series.map(s => ({ ...s, t: [], v: [] }));
  }

  // values: one per series, in series order (NaN/undefined = gap)
  push(t, values) {
    const horizon = t - this.opts.window - 1;
    this.series.forEach((s, i) => {
      s.t.push(t); s.v.push(values[i]);
      while (s.t.length && s.t[0] < horizon) { s.t.shift(); s.v.shift(); }
    });
  }

  last(i) { const s = this.series[i]; return s.v.length ? s.v[s.v.length - 1] : undefined; }

  draw(now) {
    const c = this.canvas, ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    if (!W || !H) return;
    if (c.width !== W * dpr || c.height !== H * dpr) { c.width = W * dpr; c.height = H * dpr; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const hud = cssVar("--hud");

    const padR = 70, x0 = 0, x1 = W - padR, y0 = 3, y1 = H - 3;
    const t0 = now - this.opts.window, t1 = now;

    // y range: fixed if given, else from the data with a little headroom
    let lo = this.opts.min, hi = this.opts.max;
    if (lo === undefined || hi === undefined) {
      let mn = Infinity, mx = -Infinity;
      for (const s of this.series) for (const v of s.v)
        if (Number.isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; }
      if (!Number.isFinite(mn)) { mn = -1; mx = 1; }
      if (mx - mn < 1e-6) { mn -= 0.5; mx += 0.5; }
      const m = (mx - mn) * 0.1;
      if (lo === undefined) lo = mn - m;
      if (hi === undefined) hi = mx + m;
    }
    const X = t => x0 + (t - t0) / (t1 - t0) * (x1 - x0);
    const Y = v => y1 - (v - lo) / (hi - lo) * (y1 - y0);

    // one hairline baseline at zero (or the bottom when zero is off-range)
    ctx.strokeStyle = hud; ctx.globalAlpha = 0.5; ctx.lineWidth = 0.5;
    const zy = (0 >= lo && 0 <= hi) ? Y(0) : y1;
    ctx.beginPath(); ctx.moveTo(x0, Math.round(zy) + 0.5); ctx.lineTo(x1, Math.round(zy) + 0.5); ctx.stroke();
    ctx.globalAlpha = 1;

    ctx.font = `11px ${cssVar("--mono")}`;
    ctx.textBaseline = "middle"; ctx.fillStyle = hud;
    const labels = [];
    this.series.forEach((s, i) => {
      ctx.strokeStyle = hud; ctx.lineWidth = 1.25; ctx.lineJoin = "round";
      ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      let pen = false;
      for (let k = 0; k < s.t.length; k++) {
        const v = s.v[k];
        if (!Number.isFinite(v)) { pen = false; continue; }
        const x = X(s.t[k]), y = Y(Math.max(lo, Math.min(hi, v)));
        if (x < x0) { pen = false; continue; }
        if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
      }
      ctx.stroke();
      ctx.setLineDash([]);
      const last = this.last(i);
      if (Number.isFinite(last)) labels.push({ y: Y(Math.max(lo, Math.min(hi, last))), s, v: last });
    });

    // direct labels at the right edge, nudged apart so they never overlap
    labels.sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++)
      if (labels[i].y - labels[i - 1].y < 12) labels[i].y = labels[i - 1].y + 12;
    for (const l of labels) {
      const y = Math.max(y0 + 5, Math.min(y1 - 5, l.y));
      ctx.fillText(`${l.s.name} ${l.v.toFixed(this.opts.fixed)}`, x1 + 8, y);
    }
  }
}
