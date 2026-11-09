"""Check the Cosmos-H-Dreams action format on real data (replaces verify_cosmos_format.py).

Dreams the recorded actions of one episode plus sanity perturbations, scores each against
the real endoscope video (per-frame L1 / SSIM over 72 frames, Open-H report protocol),
and writes a frame grid. Needs the seg4-cosmos container running (start_services.sh).

    cd ~/work/GR00T-H && uv run --with scikit-image python ~/seg4/server/validate_cosmos.py \
        --episode 1379 --frame 0

Result on Nov 4 2026 (ep 1379, frame 0, L40S):
    context frame held   L1 0.043  SSIM 0.776    (trivial baseline)
    recorded actions     L1 0.020  SSIM 0.869    (Open-H report: ~0.01-0.02 / ~0.80-0.85)
    old per-block format L1 0.042  SSIM 0.765
    PSM1 frozen          L1 0.041  SSIM 0.777    (PSM1 stays still in the video)
    arms swapped         L1 0.049  SSIM 0.733
A format bug shows up as "recorded actions" scoring no better than the baseline.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import cosmos_actions as ca
import metrics

COSMOS_REPO = Path.home() / "work/Cosmos-H-Dreams"
STATS = COSMOS_REPO / "assets/example_data/offline/stats_tabletop.json"
DATA = Path("/data/suturebot")
BLOCKS = 6


def variants(A: np.ndarray, ref: np.ndarray) -> dict[str, np.ndarray]:
    st = str(STATS)
    out = {"recorded": ca.to_cosmos(A, ref, BLOCKS, st)}
    fz = A.copy(); fz[:, 0:8] = ref[0:8]
    out["freeze_psm1"] = ca.to_cosmos(fz, ref, BLOCKS, st)
    sw, rsw = A.copy(), ref.copy()
    sw[:, 0:8], sw[:, 8:16] = A[:, 8:16], A[:, 0:8]
    rsw[0:8], rsw[8:16] = ref[8:16], ref[0:8]
    out["swap_arms"] = ca.to_cosmos(sw, rsw, BLOCKS, st)
    out["freeze_both"] = ca.to_cosmos(np.repeat(ref[None], len(A), 0), ref, BLOCKS, st)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=1379)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default="/data/shared/val")
    a = ap.parse_args()
    V = Path(a.out); V.mkdir(parents=True, exist_ok=True)
    ep, f0 = a.episode, a.frame

    df = pd.read_parquet(DATA / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet")
    A = np.stack(df["action"].values).astype(np.float64)[f0:f0 + ca.n_raw(BLOCKS)]
    ref = np.stack(df["observation.state"].values).astype(np.float64)[f0]
    vid = DATA / f"videos/chunk-{ep // 1000:03d}/observation.images.endoscope.left/episode_{ep:06d}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(vid), "-vf",
                    f"select=eq(n\\,{f0}),scale=512:288", "-vframes", "1", str(V / "frame.png")], check=True)
    last = f0 + 3 * ca.n_frames(BLOCKS)
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(vid), "-vf",
                          f"select='between(n\\,{f0}\\,{last})*not(mod(n-{f0}\\,3))',scale=512:288",
                          "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    real = np.frombuffer(raw, np.uint8).reshape(-1, 288, 512, 3)

    man = []
    for k, x in variants(A, ref).items():
        np.save(V / f"{k}.npy", x)
        man.append({"input": str(V / "frame.png"), "input_action": str(V / f"{k}.npy"),
                    "output_video": str(V / f"out_{k}.mp4")})
    (V / "manifest.json").write_text(json.dumps(man, indent=1))
    cmd = ("uv run flashdreams-run cosmosHDreams-chunk3-vae-vae "
           f"--input-json {V}/manifest.json --cr1-embeddings-path checkpoints/cr1_empty_string_text_embeddings.pt "
           f"--root-dir . --total-blocks {ca.ar_steps(BLOCKS)} --resolution 288,512 "
           "--pipeline.diffusion-model.transformer.checkpoint-path checkpoints/model_ema_bf16_jhutabletop_288x512_h73.pt")
    with open(V / "run.log", "w") as log:
        subprocess.run(["docker", "exec", "-w", "/workspace/cosmos-h-dreams", "seg4-cosmos", "bash", "-lc", cmd],
                       check=True, stdout=log, stderr=subprocess.STDOUT)

    n = ca.n_frames(BLOCKS)
    res = {"context frame held": metrics.per_frame_l1_ssim(np.repeat(real[:1], n, 0), real[1:n + 1])}
    for m in man:
        k = Path(m["input_action"]).stem
        res[k] = metrics.per_frame_l1_ssim(np.load(V / f"out_{k}.npy")[1:n + 1], real[1:n + 1])
    for k, m in res.items():
        print(f"{k:20s} L1 {m['mean_l1']:.4f}  SSIM {m['mean_ssim']:.3f}")
    ok = res["recorded"]["mean_l1"] < 0.75 * res["context frame held"]["mean_l1"]
    print("PASS" if ok else "FAIL: recorded actions barely beat a frozen frame -> action format is wrong")

    from PIL import Image  # frame grid for a visual check
    rows = [real] + [np.load(V / f"out_{Path(m['input_action']).stem}.npy") for m in man]
    ts = [0, n // 3, 2 * n // 3, n]
    grid = np.concatenate([np.concatenate([r[t][::2, ::2] for t in ts], 1) for r in rows], 0)
    Image.fromarray(grid).save(V / "grid.jpg")
    print("rows: real,", ", ".join(Path(m["input_action"]).stem for m in man), "->", V / "grid.jpg")


if __name__ == "__main__":
    main()
