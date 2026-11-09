"""Simulate a room of participants hitting the gateway at once.

    python load_test.py --url http://localhost:8000 --key $KEY            # on the box
    python load_test.py --url https://<secure-link> --key $KEY            # through Brev's proxy

Phases (each fires all requests at once, then reports latency):
  1. main thread: N users x (/health, act, open_loop, both dreams, both mp4 downloads), all cached
  2. uncached GR00T: N users each ask for a chunk at a different frame (serialized by the GPU lock)
  3. uncached Cosmos: D dream jobs at once (single queue), polled like the notebook does
A GPU-memory sampler runs throughout (only when run on the box).
"""
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests


def stats(name, lat, errs):
    lat = np.array(lat) if lat else np.array([np.nan])
    print(f"  {name:28s} n={len(lat):3d}  p50 {np.nanpercentile(lat, 50):6.2f}s  "
          f"p95 {np.nanpercentile(lat, 95):6.2f}s  max {np.nanmax(lat):6.2f}s  errors {len(errs)}")
    for e in errs[:3]:
        print("     ", e)


def gpu_sampler(stop, peak):
    if not shutil.which("nvidia-smi"):
        return
    while not stop.is_set():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.split()
        if out:
            peak[0] = max(peak[0], int(out[0]))
        time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--key", required=True)
    ap.add_argument("--users", type=int, default=50)
    ap.add_argument("--dreams", type=int, default=3)
    ap.add_argument("--skip", default="", help="comma list of phases to skip, e.g. 3")
    a = ap.parse_args()
    H = {"X-API-Key": a.key}
    s = requests.Session()
    s.mount("http", requests.adapters.HTTPAdapter(pool_maxsize=a.users * 2))
    h = s.get(a.url + "/health", timeout=30).json()
    ep = h["episodes"][0]
    fr = int(h.get("start_frames", {}).get(str(ep), 0))
    print(f"server: episodes {h['episodes']}  demo clip {ep}@{fr}  users {a.users}")

    stop, peak = threading.Event(), [0]
    threading.Thread(target=gpu_sampler, args=(stop, peak), daemon=True).start()

    def timed(fn):
        t = time.time()
        try:
            r = fn()
            r.raise_for_status()
            return time.time() - t, None, r
        except Exception as e:  # noqa: BLE001
            return time.time() - t, repr(e)[:200], None

    def run_phase(title, calls):
        print(title)
        res = {}
        with ThreadPoolExecutor(len(calls)) as ex:
            futs = [(name, ex.submit(timed, fn)) for name, fn in calls]
            for name, f in futs:
                dt, err, _ = f.result()
                lat, errs = res.setdefault(name, ([], []))
                (errs.append(err) if err else lat.append(dt))
        for name, (lat, errs) in res.items():
            stats(name, lat, errs)

    skip = set(a.skip.split(","))
    dream_body = lambda src: {"episode": ep, "frame": fr, "action_source": src, "blocks": 6, "instruction": None}  # noqa: E731

    if "1" not in skip:
        vids = {}
        for src in ("gt", "groot"):  # warm lookups so we know the file URLs
            vids[src] = s.post(a.url + "/v1/cosmos/dream", json=dream_body(src), headers=H, timeout=60).json()
        calls = []
        for _ in range(a.users):
            calls += [
                ("health", lambda: s.get(a.url + "/health", timeout=60)),
                ("act (cached)", lambda: s.post(a.url + "/v1/groot/act", json={"episode": ep, "frame": fr},
                                                headers=H, timeout=120)),
                ("open_loop (cached)", lambda: s.post(a.url + "/v1/groot/open_loop",
                                                      json={"episode": ep, "start": 0, "steps": 10_000},
                                                      headers=H, timeout=600)),
            ]
            for src in ("gt", "groot"):
                calls.append((f"dream {src} (cached)", lambda src=src: s.post(
                    a.url + "/v1/cosmos/dream", json=dream_body(src), headers=H, timeout=120)))
                if vids[src].get("video"):
                    calls.append((f"mp4 {src} download", lambda src=src: s.get(
                        a.url + vids[src]["video"], headers=H, timeout=120)))
        random.shuffle(calls)
        run_phase(f"[1] main thread, {len(calls)} cached requests at once", calls)

    if "2" not in skip:
        frames = random.sample(range(0, 500), a.users)
        calls = [("act (uncached)", lambda f=f: s.post(a.url + "/v1/groot/act",
                                                      json={"episode": ep, "frame": f, "instruction": "needle pickup"},
                                                      headers=H, timeout=300)) for f in frames]
        run_phase(f"[2] {a.users} uncached GR00T-H chunks at once", calls)

    if "3" not in skip:
        print(f"[3] {a.dreams} uncached Cosmos dreams at once")
        t0, jobs = time.time(), []
        for i in range(a.dreams):
            # a unique (ignored) instruction makes every request a cache miss
            body = {"episode": ep, "frame": fr, "action_source": "gt", "blocks": 6,
                    "instruction": f"loadtest-{t0:.0f}-{i}"}
            jobs.append(s.post(a.url + "/v1/cosmos/dream", json=body, headers=H, timeout=60).json())
        done = {}
        while len(done) < len(jobs) and time.time() - t0 < 1800:
            for j in jobs:
                if j["job_id"] in done:
                    continue
                st = s.get(f"{a.url}/v1/jobs/{j['job_id']}", headers=H, timeout=60).json()
                if st.get("status") in ("done", "error"):
                    done[j["job_id"]] = (time.time() - t0, st.get("status"), st.get("error", ""))
            time.sleep(3)
        for jid, (dt, status, err) in done.items():
            print(f"  job {jid}: {status} after {dt:5.1f}s {err}")

    stop.set()
    if peak[0]:
        print(f"peak GPU memory during test: {peak[0] / 1024:.1f} GB")


if __name__ == "__main__":
    main()
