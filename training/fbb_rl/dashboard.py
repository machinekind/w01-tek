"""Minimal "wandb at home" live dashboard for an fbb_rl training run.

Stdlib-only HTTP server that reads a local mirror of a remote training run's
log, parses its progress lines, and serves a single self-contained HTML page
that auto-polls for updates and renders stat tiles + two inline-SVG charts.

No third-party dependencies (no wandb, no plotting libs) -- just
http.server / socketserver / json / re / argparse / html from the standard
library, so it runs anywhere Python 3.11 runs.

Run:
    python -m fbb_rl.dashboard [--log PATH] [--port 8765] [--budget 200000000]

Then open http://127.0.0.1:8765/ in a browser.
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_LOG_PATH = (
    "LOCAL_PATH
    "6ec0d1d1-3779-4e20-96f0-df8101febd93/scratchpad/fbb_v1_mirror.log"
)
DEFAULT_PORT = 8765
DEFAULT_BUDGET = 200_000_000

# A progress line looks like (field widths vary; numbers are comma-grouped):
#   steps      102,400  reward     14.12  1,326 steps/s
#   steps            0  reward      5.13  0 steps/s
#   steps   22,937,600  reward       nan  46,286 steps/s
# The log also contains lots of unrelated XLA compiler noise lines, which
# simply won't match this pattern.
PROGRESS_RE = re.compile(
    r"steps\s+([\d,]+)\s+reward\s+(nan|-?\d+(?:\.\d+)?)\s+([\d,]+)\s+steps/s",
    re.IGNORECASE,
)
DONE_RE = re.compile(r"done\s*->", re.IGNORECASE)
EXITED_MARKER = "PROCESS_EXITED"


def parse_log(path: Path) -> tuple[list[dict], bool, bool]:
    """Parse the mirror log into deduped, steps-sorted progress points.

    Only lines matching ``PROGRESS_RE`` are kept. Stream reconnects can
    duplicate lines, so points are deduped by ``steps``, keeping the *last*
    occurrence in the file (in case a re-logged line ever carries a revised
    reward/sps for the same step count). Missing file / unreadable file is
    treated as "no data yet" rather than an error.
    """
    by_steps: dict[int, dict] = {}
    done = False
    exited = False

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return [], False, False

    for line in text.splitlines():
        match = PROGRESS_RE.search(line)
        if match:
            steps = int(match.group(1).replace(",", ""))
            reward_raw = match.group(2)
            reward = None if reward_raw.lower() == "nan" else float(reward_raw)
            sps = int(match.group(3).replace(",", ""))
            by_steps[steps] = {"steps": steps, "reward": reward, "sps": sps}
            continue
        if DONE_RE.search(line):
            done = True
        elif EXITED_MARKER in line:
            exited = True

    points = [by_steps[key] for key in sorted(by_steps)]
    return points, done, exited


HTML_DOC = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>four_bar_bot — training dashboard</title>
<style>
  :root {
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --gridline: #e1e0d9;
    --baseline: #c3c2b7;
    --series-reward: #2a78d6;
    --series-sps: #008300;
    --good: #0ca30c;
    --critical: #d03b3b;
    --border: rgba(11, 11, 11, 0.10);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --gridline: #2c2c2a;
      --baseline: #383835;
      --series-reward: #3987e5;
      --series-sps: #008300;
      --good: #0ca30c;
      --critical: #d03b3b;
      --border: rgba(255, 255, 255, 0.10);
    }
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0;
    background: var(--page);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 960px; margin: 0 auto; padding: 24px 20px 48px; }
  header {
    display: flex; align-items: baseline; justify-content: space-between;
    gap: 16px; margin-bottom: 20px; flex-wrap: wrap;
  }
  h1 { font-size: 18px; font-weight: 600; margin: 0; }
  .subtitle { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
  .status {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 13px; font-weight: 600; padding: 4px 10px 4px 8px;
    border-radius: 999px; border: 1px solid var(--border); white-space: nowrap;
  }
  .status .dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
  .status.running .dot { background: var(--text-muted); }
  .status.running { color: var(--text-secondary); }
  .status.done .dot { background: var(--good); }
  .status.done { color: var(--good); }
  .status.exited .dot { background: var(--critical); }
  .status.exited { color: var(--critical); }
  .tiles {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 12px; margin-bottom: 24px;
  }
  @media (max-width: 720px) { .tiles { grid-template-columns: repeat(2, 1fr); } }
  .tile {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
  }
  .tile .label {
    font-size: 11px; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: .04em;
  }
  .tile .value { font-size: 24px; font-weight: 600; margin-top: 4px; line-height: 1.2; }
  .tile .sub { font-size: 12px; color: var(--text-secondary); margin-top: 2px; min-height: 1.4em; }
  .charts { display: flex; flex-direction: column; gap: 20px; }
  .card {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px; position: relative;
  }
  .card h2 { font-size: 13px; font-weight: 600; margin: 0 0 2px; }
  .card .card-sub { font-size: 11px; color: var(--text-muted); margin-bottom: 8px; }
  .chart-surface { position: relative; width: 100%; }
  svg.chart { display: block; width: 100%; height: 220px; overflow: visible; }
  .gridline { stroke: var(--gridline); stroke-width: 1; shape-rendering: crispEdges; }
  .axis-line { stroke: var(--baseline); stroke-width: 1; shape-rendering: crispEdges; }
  .tick-label {
    fill: var(--text-muted); font-size: 10px;
    font-family: system-ui, -apple-system, sans-serif;
  }
  .series-line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .series-dot { stroke-width: 2; }
  .crosshair-line { stroke: var(--baseline); stroke-width: 1; opacity: 0; pointer-events: none; }
  .hover-dot { opacity: 0; pointer-events: none; stroke-width: 2; }
  .empty-state { fill: var(--text-muted); font-size: 12px; }
  .overlay-rect { fill: transparent; cursor: crosshair; }
  .tooltip {
    position: absolute; pointer-events: none;
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 10px; font-size: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.16);
    opacity: 0; transform: translate(-50%, -100%);
    white-space: nowrap; z-index: 5; transition: opacity 80ms linear;
  }
  .tooltip .tt-value { font-weight: 600; font-size: 13px; color: var(--text-primary); }
  .tooltip .tt-label { color: var(--text-secondary); display: flex; align-items: center; gap: 5px; margin-top: 2px; }
  .tt-key { display: inline-block; width: 10px; height: 2px; flex: none; }
  footer { margin-top: 24px; font-size: 11px; color: var(--text-muted); }
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>four_bar_bot RL training</h1>
        <div class="subtitle" id="subtitle">connecting&hellip;</div>
      </div>
      <span class="status running" id="status"><span class="dot"></span><span id="status-text">RUNNING</span></span>
    </header>

    <div class="tiles">
      <div class="tile">
        <div class="label">Latest reward</div>
        <div class="value" id="tile-reward">&mdash;</div>
        <div class="sub" id="tile-reward-sub">&nbsp;</div>
      </div>
      <div class="tile">
        <div class="label">Steps / budget</div>
        <div class="value" id="tile-steps">&mdash;</div>
        <div class="sub" id="tile-steps-sub">&nbsp;</div>
      </div>
      <div class="tile">
        <div class="label">Throughput</div>
        <div class="value" id="tile-sps">&mdash;</div>
        <div class="sub">steps/s</div>
      </div>
      <div class="tile">
        <div class="label">ETA to budget</div>
        <div class="value" id="tile-eta">&mdash;</div>
        <div class="sub" id="tile-eta-sub">&nbsp;</div>
      </div>
    </div>

    <div class="charts">
      <div class="card">
        <h2>Eval reward</h2>
        <div class="card-sub">vs training steps (sparse, periodic eval)</div>
        <div class="chart-surface" id="reward-surface">
          <svg class="chart" id="reward-chart" role="img" aria-label="Eval reward over training steps"></svg>
          <div class="tooltip" id="reward-tooltip"></div>
        </div>
      </div>
      <div class="card">
        <h2>Throughput</h2>
        <div class="card-sub">steps/s vs training steps</div>
        <div class="chart-surface" id="sps-surface">
          <svg class="chart" id="sps-chart" role="img" aria-label="Throughput in steps per second over training steps"></svg>
          <div class="tooltip" id="sps-tooltip"></div>
        </div>
      </div>
    </div>

    <footer id="footer">Polling /data every 5s&hellip;</footer>
  </div>

<script>
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var POLL_MS = 5000;

  function svgEl(tag, attrs) {
    var e = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      for (var k in attrs) {
        if (Object.prototype.hasOwnProperty.call(attrs, k)) e.setAttribute(k, attrs[k]);
      }
    }
    return e;
  }

  function trimTrailingZero(s) { return s.replace(/\\.0$/, ""); }

  function fmtCompact(n) {
    var abs = Math.abs(n);
    if (abs >= 1e9) return trimTrailingZero((n / 1e9).toFixed(1)) + "B";
    if (abs >= 1e6) return trimTrailingZero((n / 1e6).toFixed(1)) + "M";
    if (abs >= 1e3) return trimTrailingZero((n / 1e3).toFixed(1)) + "K";
    return Math.round(n).toLocaleString();
  }

  function fmtFixed(decimals) {
    return function (n) {
      if (n === null || n === undefined || Number.isNaN(n)) return "—";
      return n.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
    };
  }

  function fmtInt(n) {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    return Math.round(n).toLocaleString();
  }

  function fmtDuration(seconds) {
    if (!isFinite(seconds) || seconds < 0) return "—";
    seconds = Math.round(seconds);
    var d = Math.floor(seconds / 86400); seconds -= d * 86400;
    var h = Math.floor(seconds / 3600); seconds -= h * 3600;
    var m = Math.floor(seconds / 60); seconds -= m * 60;
    if (d > 0) return d + "d " + h + "h";
    if (h > 0) return h + "h " + m + "m";
    if (m > 0) return m + "m " + seconds + "s";
    return seconds + "s";
  }

  // "Nice numbers" tick algorithm (Sparks/Heckbert), so axis ticks land on
  // clean round values rather than raw data boundaries.
  function niceNum(range, round) {
    if (range === 0) range = 1;
    var exponent = Math.floor(Math.log10(range));
    var fraction = range / Math.pow(10, exponent);
    var niceFraction;
    if (round) {
      if (fraction < 1.5) niceFraction = 1;
      else if (fraction < 3) niceFraction = 2;
      else if (fraction < 7) niceFraction = 5;
      else niceFraction = 10;
    } else {
      if (fraction <= 1) niceFraction = 1;
      else if (fraction <= 2) niceFraction = 2;
      else if (fraction <= 5) niceFraction = 5;
      else niceFraction = 10;
    }
    return niceFraction * Math.pow(10, exponent);
  }

  function niceDomain(min, max, maxTicks) {
    maxTicks = maxTicks || 5;
    if (min === max) {
      var pad = min === 0 ? 1 : Math.abs(min) * 0.1;
      min -= pad; max += pad;
    }
    var range = niceNum(max - min, false);
    var step = niceNum(range / (maxTicks - 1), true);
    var niceMin = Math.floor(min / step) * step;
    var niceMax = Math.ceil(max / step) * step;
    var ticks = [];
    for (var v = niceMin; v <= niceMax + step / 1000; v += step) {
      ticks.push(Math.round(v * 1e6) / 1e6);
    }
    return { min: niceMin, max: niceMax, ticks: ticks };
  }

  // One chart: an SVG line+point plot with a crosshair/tooltip hover layer.
  // Single series per chart -> no legend box (the card title names it).
  function makeChart(svg, tooltip, opts) {
    var state = { points: [], crosshair: null, hoverDot: null };

    function clear() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
    }

    function hide() {
      if (state.crosshair) state.crosshair.style.opacity = 0;
      if (state.hoverDot) state.hoverDot.style.opacity = 0;
      tooltip.style.opacity = 0;
    }

    function nearestByPixel(px) {
      var pts = state.points;
      var best = pts[0];
      var bestDist = Infinity;
      for (var i = 0; i < pts.length; i++) {
        var dist = Math.abs(pts[i].cx - px);
        if (dist < bestDist) { bestDist = dist; best = pts[i]; }
      }
      return best;
    }

    function showAt(px) {
      if (state.points.length === 0) return;
      var p = nearestByPixel(px);
      state.crosshair.setAttribute("x1", p.cx);
      state.crosshair.setAttribute("x2", p.cx);
      state.crosshair.style.opacity = 1;
      state.hoverDot.setAttribute("cx", p.cx);
      state.hoverDot.setAttribute("cy", p.cy);
      state.hoverDot.style.opacity = 1;

      tooltip.style.left = p.cx + "px";
      tooltip.style.top = Math.max(p.cy - 12, 0) + "px";
      tooltip.style.opacity = 1;
      while (tooltip.firstChild) tooltip.removeChild(tooltip.firstChild);
      var valueDiv = document.createElement("div");
      valueDiv.className = "tt-value";
      valueDiv.textContent = opts.yFmt(p.y) + (opts.unit ? " " + opts.unit : "");
      var labelDiv = document.createElement("div");
      labelDiv.className = "tt-label";
      var key = document.createElement("span");
      key.className = "tt-key";
      key.style.background = opts.color;
      labelDiv.appendChild(key);
      labelDiv.appendChild(document.createTextNode("steps " + p.x.toLocaleString()));
      tooltip.appendChild(valueDiv);
      tooltip.appendChild(labelDiv);
    }

    function onMove(evt) {
      var rect = svg.getBoundingClientRect();
      showAt(evt.clientX - rect.left);
    }
    function onTouchMove(evt) {
      if (!evt.touches || !evt.touches[0]) return;
      var rect = svg.getBoundingClientRect();
      showAt(evt.touches[0].clientX - rect.left);
    }

    function render(points) {
      clear();
      hide();
      var width = svg.clientWidth || 600;
      var height = 220;
      svg.setAttribute("width", width);
      svg.setAttribute("height", height);
      svg.setAttribute("viewBox", "0 0 " + width + " " + height);

      if (!points || points.length === 0) {
        var msg = svgEl("text", { x: width / 2, y: height / 2, "text-anchor": "middle", class: "empty-state" });
        msg.textContent = "Waiting for data…";
        svg.appendChild(msg);
        state.points = [];
        state.crosshair = null;
        state.hoverDot = null;
        return;
      }

      var padL = 56, padR = 12, padT = 12, padB = 26;
      var plotW = Math.max(width - padL - padR, 10);
      var plotH = Math.max(height - padT - padB, 10);

      var xs = points.map(function (p) { return p.x; });
      var ys = points.map(function (p) { return p.y; });
      var xd = niceDomain(Math.min.apply(null, xs), Math.max.apply(null, xs), 5);
      var yd = niceDomain(Math.min.apply(null, ys), Math.max.apply(null, ys), 4);
      var xSpan = (xd.max - xd.min) || 1;
      var ySpan = (yd.max - yd.min) || 1;

      function xScale(v) { return padL + ((v - xd.min) / xSpan) * plotW; }
      function yScale(v) { return padT + plotH - ((v - yd.min) / ySpan) * plotH; }

      yd.ticks.forEach(function (t) {
        var y = yScale(t);
        svg.appendChild(svgEl("line", { x1: padL, x2: padL + plotW, y1: y, y2: y, class: "gridline" }));
        var lbl = svgEl("text", { x: padL - 8, y: y + 3, "text-anchor": "end", class: "tick-label" });
        lbl.textContent = opts.yFmt(t);
        svg.appendChild(lbl);
      });

      xd.ticks.forEach(function (t) {
        var x = xScale(t);
        if (x < padL - 1 || x > padL + plotW + 1) return;
        var lbl = svgEl("text", { x: x, y: padT + plotH + 18, "text-anchor": "middle", class: "tick-label" });
        lbl.textContent = fmtCompact(t);
        svg.appendChild(lbl);
      });

      svg.appendChild(svgEl("line", { x1: padL, x2: padL + plotW, y1: padT + plotH, y2: padT + plotH, class: "axis-line" }));

      var plotted = points.map(function (p) { return { x: p.x, y: p.y, cx: xScale(p.x), cy: yScale(p.y) }; });

      if (plotted.length > 1) {
        var d = plotted.map(function (p, i) {
          return (i === 0 ? "M" : "L") + p.cx.toFixed(2) + "," + p.cy.toFixed(2);
        }).join(" ");
        var path = svgEl("path", { d: d, class: "series-line" });
        path.style.stroke = opts.color;
        svg.appendChild(path);
      }

      plotted.forEach(function (p) {
        var c = svgEl("circle", { cx: p.cx, cy: p.cy, r: 4, class: "series-dot" });
        c.style.fill = opts.color;
        c.style.stroke = "var(--surface-1)";
        svg.appendChild(c);
      });

      var crosshair = svgEl("line", { x1: 0, x2: 0, y1: padT, y2: padT + plotH, class: "crosshair-line" });
      svg.appendChild(crosshair);
      var hoverDot = svgEl("circle", { r: 6, class: "hover-dot" });
      hoverDot.style.fill = opts.color;
      hoverDot.style.stroke = "var(--surface-1)";
      svg.appendChild(hoverDot);

      var overlay = svgEl("rect", { x: padL, y: padT, width: plotW, height: plotH, class: "overlay-rect" });
      overlay.addEventListener("mousemove", onMove);
      overlay.addEventListener("mouseleave", hide);
      overlay.addEventListener("touchmove", onTouchMove, { passive: true });
      overlay.addEventListener("touchend", hide);
      svg.appendChild(overlay);

      state.points = plotted;
      state.crosshair = crosshair;
      state.hoverDot = hoverDot;
    }

    return { render: render };
  }

  var rewardChart = makeChart(
    document.getElementById("reward-chart"),
    document.getElementById("reward-tooltip"),
    { color: "var(--series-reward)", yFmt: fmtFixed(2), unit: "" }
  );
  var spsChart = makeChart(
    document.getElementById("sps-chart"),
    document.getElementById("sps-tooltip"),
    { color: "var(--series-sps)", yFmt: fmtInt, unit: "steps/s" }
  );

  function renderAll(data) {
    var points = data.points || [];
    var budget = data.budget || 0;
    var last = points.length ? points[points.length - 1] : null;

    var statusEl = document.getElementById("status");
    var statusText = document.getElementById("status-text");
    var cls = "running", label = "RUNNING";
    if (data.exited) { cls = "exited"; label = "EXITED"; }
    else if (data.done) { cls = "done"; label = "DONE"; }
    statusEl.className = "status " + cls;
    statusText.textContent = label;

    var subtitle = document.getElementById("subtitle");
    subtitle.textContent = points.length
      ? points.length.toLocaleString() + " logged point" + (points.length === 1 ? "" : "s") +
        " · last updated " + new Date().toLocaleTimeString()
      : "no progress lines parsed yet";

    var lastRewardPoint = null;
    for (var i = points.length - 1; i >= 0; i--) {
      if (points[i].reward !== null && points[i].reward !== undefined) { lastRewardPoint = points[i]; break; }
    }
    document.getElementById("tile-reward").textContent = lastRewardPoint ? fmtFixed(2)(lastRewardPoint.reward) : "—";
    document.getElementById("tile-reward-sub").textContent = lastRewardPoint
      ? ("at " + lastRewardPoint.steps.toLocaleString() + " steps")
      : " ";

    if (last && budget > 0) {
      var pct = (last.steps / budget) * 100;
      document.getElementById("tile-steps").textContent = (pct < 0.1 ? pct.toFixed(3) : pct.toFixed(1)) + "%";
      document.getElementById("tile-steps-sub").textContent =
        last.steps.toLocaleString() + " / " + budget.toLocaleString();
    } else {
      document.getElementById("tile-steps").textContent = "—";
      document.getElementById("tile-steps-sub").textContent = " ";
    }

    document.getElementById("tile-sps").textContent = last ? fmtInt(last.sps) : "—";

    var etaEl = document.getElementById("tile-eta");
    var etaSub = document.getElementById("tile-eta-sub");
    if (last && budget > last.steps && last.sps > 0) {
      var remaining = budget - last.steps;
      etaEl.textContent = fmtDuration(remaining / last.sps);
      etaSub.textContent = "at " + fmtInt(last.sps) + " steps/s";
    } else if (last && budget > 0 && last.steps >= budget) {
      etaEl.textContent = "0s";
      etaSub.textContent = "budget reached";
    } else {
      etaEl.textContent = "—";
      etaSub.textContent = " ";
    }

    var rewardPoints = points
      .filter(function (p) { return p.reward !== null && p.reward !== undefined; })
      .map(function (p) { return { x: p.steps, y: p.reward }; });
    var spsPoints = points.map(function (p) { return { x: p.steps, y: p.sps }; });

    rewardChart.render(rewardPoints);
    spsChart.render(spsPoints);
  }

  rewardChart.render([]);
  spsChart.render([]);

  var footer = document.getElementById("footer");
  var lastData = null;

  function poll() {
    fetch("/data", { cache: "no-store" })
      .then(function (res) {
        if (!res.ok) throw new Error("bad status " + res.status);
        return res.json();
      })
      .then(function (data) {
        lastData = data;
        renderAll(data);
        footer.textContent = "Polling /data every 5s · last fetch ok at " + new Date().toLocaleTimeString();
      })
      .catch(function () {
        footer.textContent = "Polling /data every 5s · last fetch failed, retrying…";
      });
  }

  poll();
  setInterval(poll, POLL_MS);

  var resizeTimer = null;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      if (lastData) renderAll(lastData);
    }, 150);
  });
})();
</script>
</body>
</html>
"""

HTML_PAGE = HTML_DOC.encode("utf-8")


class DashboardHandler(BaseHTTPRequestHandler):
    """Serves the dashboard page and its JSON data endpoint."""

    log_path: Path = Path(DEFAULT_LOG_PATH)
    budget: int = DEFAULT_BUDGET
    server_version = "FBBDashboard/1.0"

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", HTML_PAGE)
            return
        if self.path in ("/data", "/data/"):
            points, done, exited = parse_log(self.log_path)
            payload = {
                "points": points,
                "done": done,
                "exited": exited,
                "budget": self.budget,
            }
            body = json.dumps(payload).encode("utf-8")
            self._send(200, "application/json", body)
            return
        self._send(404, "text/plain; charset=utf-8", b"not found")

    def log_message(self, format: str, *args) -> None:  # noqa: A002 (stdlib signature)
        pass  # keep stdout/stderr quiet across hours of 5s polling


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal local live dashboard for an fbb_rl training run."
    )
    parser.add_argument(
        "--log", default=DEFAULT_LOG_PATH, help="Path to the local mirror log file."
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to serve the dashboard on."
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help="Total training step budget, used for the %% and ETA tiles.",
    )
    args = parser.parse_args()

    DashboardHandler.log_path = Path(args.log)
    DashboardHandler.budget = args.budget

    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    print(
        f"fbb_rl dashboard: http://127.0.0.1:{args.port}/ "
        f"(log={args.log}, budget={args.budget:,})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
