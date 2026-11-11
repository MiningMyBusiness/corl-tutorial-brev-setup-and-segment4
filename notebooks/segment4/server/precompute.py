"""Warm the gateway cache with every main-thread request and export an offline bundle.

    python precompute.py --url http://localhost:8000 --key $API_KEY --episode 1441 --frame 345

Writes ./bundle/ (JSON + mp4) that the notebook's CACHE backend downloads when the API is
unreachable. Upload it somewhere public (e.g. a HF dataset repo or GitHub release).
"""
from __future__ import annotations

import argparse
import json
import zipfile
import time
from pathlib import Path

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--key", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--alt_instructions", default="needle pickup,needle throw,knot tying")
    ap.add_argument("--out", default="bundle")
    ap.add_argument("--closed_loop_iters", type=int, default=20, help="capped by the server at the clip's end; 0 = skip")
    ap.add_argument("--act_url", default="https://huggingface.co/kdb6df/corl2026-suturebot-act/resolve/main/predictions/act_{episode}.npz",
                    help="SutureBot ACT predictions (act_precompute.py); '' = skip the ACT rollout")
    a = ap.parse_args()
    H = {"X-API-Key": a.key}
    out = Path(a.out); out.mkdir(exist_ok=True)
    post = lambda p, j: requests.post(a.url + p, json=j, headers=H, timeout=1800).json()  # noqa: E731

    def finish(j):  # poll a queued job until it ends
        while j.get("status") not in ("done", "error"):
            time.sleep(5)
            j = requests.get(f"{a.url}/v1/jobs/{j['job_id']}", headers=H).json() | {"job_id": j["job_id"]}
        return j

    def fetch(url, name):
        (out / name).write_bytes(requests.get(a.url + url, headers=H).content)
        return name

    b = {"episode": a.episode, "frame": a.frame, "blocks": a.blocks}
    b["act"] = post("/v1/groot/act", {"episode": a.episode, "frame": a.frame})
    b["open_loop"] = post("/v1/groot/open_loop", {"episode": a.episode, "start": 0, "steps": 10_000})
    b["act_alt"] = {i: post("/v1/groot/act", {"episode": a.episode, "frame": a.frame, "instruction": i})
                    for i in a.alt_instructions.split(",")}
    b["dreams"] = {}
    for src in ("gt", "groot"):
        j = post("/v1/cosmos/dream", {"episode": a.episode, "frame": a.frame,
                                       "action_source": src, "blocks": a.blocks})
        j = finish(j)
        print(src, j.get("status"), j.get("error", ""))
        if j.get("status") == "done":
            b["dreams"][src] = fetch(j["video"], f"dream_{src}.mp4")
    # Cosmos driven by ACT's actions (precomputed offline, act_precompute.py). Same request as the notebook
    # sends, so the notebook's live call is a cache hit.
    if a.act_url:
        r = requests.get(a.act_url.format(episode=a.episode), timeout=120)
        if r.ok:
            import io
            import numpy as np
            z = np.load(io.BytesIO(r.content))
            if int(z["start_frame"]) == a.frame:
                j = finish(post("/v1/cosmos/dream", {"episode": a.episode, "frame": a.frame, "action_source": "custom",
                                                     "blocks": a.blocks, "actions": z["dream"].tolist()}))
                print("act", j.get("status"), j.get("error", ""))
                if j.get("status") == "done":
                    b["dreams"]["act"] = fetch(j["video"], "dream_act.mp4")
            else:
                print(f"act: predictions start at {int(z['start_frame'])}, not {a.frame}; skipped")
        else:
            print("act: no predictions at", a.act_url.format(episode=a.episode), r.status_code)
    files = list(b["dreams"].values())
    # closed loop (CLOSED_LOOP.md): only if this server has it switched on
    if a.closed_loop_iters and requests.get(a.url + "/health").json().get("closed_loop"):
        b["open_loop_mono"] = post("/v1/groot/open_loop_mono", {"episode": a.episode, "start": 0, "steps": 10_000})
        b["closed_loop"] = {}
        for pol in ("groot", "recorded"):
            j = finish(post("/v1/closed_loop", {"episode": a.episode, "frame": a.frame,
                                                "iterations": a.closed_loop_iters, "policy": pol}))
            print("closed loop", pol, j.get("status"), j.get("error", ""))
            if j.get("status") == "done":
                b["closed_loop"][pol] = {"frame": j.get("frame", a.frame), "iterations": j.get("iterations"),
                                         "video": fetch(j["video"], f"closed_{pol}.mp4"),
                                         "actions": fetch(j["actions"], f"closed_{pol}_actions.npy")}
                files += [b["closed_loop"][pol]["video"], b["closed_loop"][pol]["actions"]]
    (out / "bundle.json").write_text(json.dumps(b))
    # the notebook extracts this zip into ./bundle/, so files sit at the zip root
    zpath = out.parent / f"seg4_bundle_{a.episode}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["bundle.json", *files]:
            z.write(out / f, arcname=f)
    print("bundle ->", out, "| zip ->", zpath, f"({zpath.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
