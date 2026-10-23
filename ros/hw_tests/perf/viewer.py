#!/usr/bin/env python3
"""Stand-in for the Steam Deck: pull the gateway's MJPEG stream and time it.

Reads http://<robot>:8090/stream.mjpg for --duration seconds and prints one
JSON object: frames, fps, kB/s and the frame-to-frame gaps as the PC saw
them -- what the panel's video looks like at the end of the network. Only
the standard library, so it runs on any PC.
"""

import argparse
import json
import statistics
import time
import urllib.request

SOI = b"\xff\xd8"  # JPEG start-of-image


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("url")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--connect-timeout", type=float, default=60.0)
    args = ap.parse_args()

    arrivals, total, error = [], 0, None
    # The gateway may still be starting: retry the connection for a while.
    resp, deadline = None, time.monotonic() + args.connect_timeout
    while resp is None:
        try:
            resp = urllib.request.urlopen(args.url, timeout=10)
        except OSError as e:
            if time.monotonic() >= deadline:
                print(json.dumps({"frames": 0, "fps": 0.0, "error": str(e)}))
                return
            time.sleep(1.0)
    # The window starts at the first frame, not at the connection: the
    # camera may still be warming up behind a gateway that already answers,
    # and a frameless head start would read as a slow stream. The wait for
    # that first frame is bounded by the connect timeout.
    t0 = None
    first_deadline = time.monotonic() + args.connect_timeout
    try:
        with resp:
            tail = b""
            while True:
                now = time.monotonic()
                if t0 is None:
                    if now >= first_deadline:
                        error = "no frame arrived"
                        break
                elif now - t0 >= args.duration:
                    break
                chunk = resp.read1(65536)
                if not chunk:
                    break
                buf = tail + chunk
                now = time.monotonic()
                new = buf.count(SOI) - tail.count(SOI)
                if t0 is None and new:
                    t0 = now
                    # Bytes before the first frame are not the stream's.
                    total = 0
                total += len(chunk)
                arrivals.extend(now for _ in range(new))
                tail = buf[-1:]
    except OSError as e:
        error = str(e)
    elapsed = max(time.monotonic() - t0, 1e-6) if t0 is not None else 1e-6
    gaps = [(b - a) * 1e3 for a, b in zip(arrivals, arrivals[1:])]
    out = {
        "frames": len(arrivals),
        "fps": round(len(arrivals) / elapsed, 2),
        "kBps": round(total / elapsed / 1024, 1),
        "frame_kb": round(total / len(arrivals) / 1024, 1) if arrivals else None,
    }
    if gaps:
        s = sorted(gaps)
        out["gap_ms"] = {
            "mean": round(statistics.fmean(gaps), 1),
            "p99": round(s[int(0.99 * (len(s) - 1))], 1),
            "max": round(s[-1], 1),
        }
    if error:
        out["error"] = error
    print(json.dumps(out))


if __name__ == "__main__":
    main()
