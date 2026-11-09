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
    a = ap.parse_args()
    H = {"X-API-Key": a.key}
    out = Path(a.out); out.mkdir(exist_ok=True)
    post = lambda p, j: requests.post(a.url + p, json=j, headers=H, timeout=1800).json()  # noqa: E731

    b = {"episode": a.episode, "frame": a.frame, "blocks": a.blocks}
    b["act"] = post("/v1/groot/act", {"episode": a.episode, "frame": a.frame})
    b["open_loop"] = post("/v1/groot/open_loop", {"episode": a.episode, "start": 0, "steps": 10_000})
    b["act_alt"] = {i: post("/v1/groot/act", {"episode": a.episode, "frame": a.frame, "instruction": i})
                    for i in a.alt_instructions.split(",")}
    b["dreams"] = {}
    for src in ("gt", "groot"):
        j = post("/v1/cosmos/dream", {"episode": a.episode, "frame": a.frame,
                                       "action_source": src, "blocks": a.blocks})
        while j.get("status") not in ("done", "error"):
            time.sleep(5)
            j = requests.get(f"{a.url}/v1/jobs/{j['job_id']}", headers=H).json() | {"job_id": j["job_id"]}
        print(src, j.get("status"), j.get("error", ""))
        if j.get("status") == "done":
            mp4 = requests.get(a.url + j["video"], headers=H).content
            (out / f"dream_{src}.mp4").write_bytes(mp4)
            b["dreams"][src] = f"dream_{src}.mp4"
    (out / "bundle.json").write_text(json.dumps(b))
    # the notebook extracts this zip into ./bundle/, so files sit at the zip root
    zpath = out.parent / f"seg4_bundle_{a.episode}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["bundle.json", *b["dreams"].values()]:
            z.write(out / f, arcname=f)
    print("bundle ->", out, "| zip ->", zpath, f"({zpath.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
