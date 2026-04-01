"""Probe the Wikimedia Commons API to find the minimum safe request interval.

Sends a series of lightweight API requests at progressively shorter intervals
and reports response times and any rate-limit responses (HTTP 429 / 503).
Use the results to pick a safe --rate-limit value for the download scripts.

Strategy
--------
Starting from --start-interval (default 1.0 s), the script steps down through
a configurable list of intervals. At each level it sends --probes requests and
records the outcome. Between levels it pauses for --cooldown seconds so the
server does not see the full test as a sustained burst.

If a 429 or 503 is received the test stops immediately and reports the last
safe interval. No retries are attempted on failures — the point is to detect
the threshold, not to work around it.

Endpoint used
-------------
action=query&prop=categoryinfo&titles=Category:Panthera_leo
This is the same lightweight call used by the category crawler. It returns a
small JSON payload (~200 bytes) and is representative of actual usage.

Usage
-----
    python scripts/probe_wikimedia_rate_limit.py
    python scripts/probe_wikimedia_rate_limit.py --start-interval 0.5 --min-interval 0.05
    python scripts/probe_wikimedia_rate_limit.py --probes 20 --cooldown 10

Output
------
A table of tested intervals with median / p95 response time and pass/fail,
followed by a recommendation.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_supplementary import USER_AGENT, WIKI_API

# Lightweight probe query — small response, representative of real usage
PROBE_PARAMS = {
    "action": "query",
    "prop": "categoryinfo",
    "titles": "Category:Panthera_leo",
    "format": "json",
}

# Default intervals to test (seconds), coarse-to-fine
DEFAULT_INTERVALS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05]


def _make_session():
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def probe_interval(session, interval: float, n_probes: int) -> dict:
    """Send n_probes requests at the given interval.

    Returns a result dict:
        interval      float
        n_ok          int   — HTTP 200 with valid JSON
        n_ratelimit   int   — HTTP 429 or 503
        n_error       int   — other failures
        latencies     list  — response times in ms for successful requests
        blocked       bool  — True if any 429/503 was seen
    """
    result = {
        "interval": interval,
        "n_ok": 0,
        "n_ratelimit": 0,
        "n_error": 0,
        "latencies": [],
        "blocked": False,
    }

    for i in range(n_probes):
        t0 = time.perf_counter()
        try:
            resp = session.get(WIKI_API, params=PROBE_PARAMS, timeout=15)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if resp.status_code in (429, 503):
                result["n_ratelimit"] += 1
                result["blocked"] = True
                print(f"  [{i+1:>3}/{n_probes}]  HTTP {resp.status_code}  *** RATE LIMITED ***",
                      flush=True)
                # Stop immediately — don't push further
                break

            resp.raise_for_status()
            result["n_ok"] += 1
            result["latencies"].append(elapsed_ms)
            print(f"  [{i+1:>3}/{n_probes}]  {elapsed_ms:6.0f} ms  OK", flush=True)

        except requests.exceptions.Timeout:
            result["n_error"] += 1
            print(f"  [{i+1:>3}/{n_probes}]  TIMEOUT", flush=True)
        except Exception as e:
            result["n_error"] += 1
            print(f"  [{i+1:>3}/{n_probes}]  ERROR: {e}", flush=True)

        if i < n_probes - 1:
            time.sleep(interval)

    return result


def summarise(result: dict) -> str:
    lats = result["latencies"]
    if not lats:
        return "no successful requests"
    med = statistics.median(lats)
    p95 = sorted(lats)[int(len(lats) * 0.95)] if len(lats) >= 2 else lats[-1]
    return f"median {med:.0f} ms  p95 {p95:.0f} ms  ({result['n_ok']} ok / {result['n_ratelimit']} blocked / {result['n_error']} err)"


def main():
    parser = argparse.ArgumentParser(
        description="Find the minimum safe Wikimedia API request interval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start-interval", type=float, default=1.0,
                        help="Start testing from this interval in seconds (default: 1.0)")
    parser.add_argument("--min-interval", type=float, default=0.05,
                        help="Do not test below this interval (default: 0.05)")
    parser.add_argument("--probes", type=int, default=15,
                        help="Number of requests to send per interval level (default: 15)")
    parser.add_argument("--cooldown", type=float, default=8.0,
                        help="Seconds to pause between interval levels (default: 8.0)")
    args = parser.parse_args()

    intervals = [x for x in DEFAULT_INTERVALS
                 if args.min_interval <= x <= args.start_interval]
    if not intervals:
        print(f"No intervals in range [{args.min_interval}, {args.start_interval}]")
        sys.exit(1)

    print("Wikimedia API rate-limit probe")
    print(f"  Endpoint  : {WIKI_API}")
    print(f"  Intervals : {intervals}")
    print(f"  Probes    : {args.probes} per level")
    print(f"  Cooldown  : {args.cooldown} s between levels")
    print()

    session = _make_session()
    results = []
    last_safe_interval = None

    for idx, interval in enumerate(intervals):
        print(f"── Interval {interval:.3f} s ──────────────────────────────", flush=True)
        result = probe_interval(session, interval, args.probes)
        results.append(result)

        if result["blocked"]:
            print(f"  Blocked at {interval:.3f} s — stopping.")
            break

        last_safe_interval = interval

        if idx < len(intervals) - 1:
            next_interval = intervals[idx + 1]
            if next_interval < args.min_interval:
                break
            print(f"  Cooling down {args.cooldown:.0f} s before next level…", flush=True)
            time.sleep(args.cooldown)

        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("─" * 72)
    print(f"{'Interval':>10}  {'Result':<10}  {'Stats'}")
    print("─" * 72)
    for r in results:
        status = "BLOCKED" if r["blocked"] else "ok"
        print(f"{r['interval']:>10.3f}  {status:<10}  {summarise(r)}")
    print("─" * 72)
    print()

    if last_safe_interval is None:
        print("All tested intervals were blocked. The server may have already")
        print("rate-limited this IP. Wait 15–30 minutes before retrying.")
    else:
        # Add a safety margin: recommend the level one step above the last safe one
        # (or the last safe one itself if no blocking occurred)
        blocked_any = any(r["blocked"] for r in results)
        if blocked_any:
            # Find the interval just above the last safe one in the tested list
            idx = intervals.index(last_safe_interval)
            recommended = intervals[max(0, idx - 1)] if idx > 0 else last_safe_interval
            note = "(one step above where blocking occurred)"
        else:
            recommended = last_safe_interval
            note = "(lowest tested without blocking)"

        print(f"Recommendation: --rate-limit {recommended:.3f}  {note}")
        print()
        print("Set this in the download scripts:")
        print(f"  python scripts/scrape_wikimedia_file_list.py  --rate-limit {recommended:.3f}")
        print(f"  python scripts/download_wikimedia_images.py   --rate-limit {recommended:.3f}")


if __name__ == "__main__":
    main()
