"""Pick a dream start frame per episode: the 214-frame window (72 generated frames at 10 Hz)
with the most two-arm motion plus jaw activity.

    python pick_start_frames.py --episodes 1441-1451        # prints a DEMO_FRAMES string

Score per window = min(psm1 path, psm2 path) in mm  +  20 * (psm1 jaw range + psm2 jaw range) in rad.
Taking the min of the two arms favours windows where BOTH arms move (more to see in the dream).
Start frames are snapped to multiples of 15 and kept >= 30 frames from the start of the episode.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import cosmos_actions  # noqa: E402

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data/suturebot"))


def parse_eps(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def score_windows(act: np.ndarray, n_raw: int, snap: int = 15, min_start: int = 30):
    xyz1, xyz2, jaw1, jaw2 = act[:, 0:3], act[:, 8:11], act[:, 7], act[:, 15]
    step1 = np.r_[0, np.linalg.norm(np.diff(xyz1, axis=0), axis=1)].cumsum() * 1000
    step2 = np.r_[0, np.linalg.norm(np.diff(xyz2, axis=0), axis=1)].cumsum() * 1000
    rows = []
    for f0 in range(min_start, len(act) - n_raw + 1, snap):
        f1 = f0 + n_raw - 1
        p1, p2 = step1[f1] - step1[f0], step2[f1] - step2[f0]
        j = np.ptp(jaw1[f0:f1 + 1]) + np.ptp(jaw2[f0:f1 + 1])
        rows.append((f0, min(p1, p2) + 20 * j, p1, p2, j))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="1441-1451")
    ap.add_argument("--blocks", type=int, default=6)
    a = ap.parse_args()
    n_raw = cosmos_actions.n_raw(a.blocks)
    pairs = []
    for ep in parse_eps(a.episodes):
        df = pd.read_parquet(DATA_ROOT / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet")
        act = np.stack(df["action"].to_numpy()).astype(np.float64)
        rows = score_windows(act, n_raw)
        f0, s, p1, p2, j = max(rows, key=lambda r: r[1])
        print(f"ep {ep} ({len(act)} frames): start {f0:4d}  score {s:6.1f}  "
              f"psm1 {p1:5.1f} mm  psm2 {p2:5.1f} mm  jaw range {j:.2f} rad")
        pairs.append(f"{ep}:{f0}")
    print("DEMO_FRAMES=" + ",".join(pairs))


if __name__ == "__main__":
    main()
