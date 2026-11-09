"""Rank SutureBot test-split episodes for the Segment 4 demo clip and build a contact sheet.

Runs anywhere with internet + ffmpeg (Brev box or Colab):
    pip install huggingface_hub pandas pyarrow pillow
    python rank_episodes.py --out ./ranking --top 8 [--task "needle throw"]

Selection rules (see plan):
  * test split only (episodes 1306-1451, never seen in training)
  * long enough for a 72-frame Cosmos rollout (6 blocks x 12 frames x stride 3 = 216
    raw frames) plus one 50-step GR00T chunk  -> >= 270 frames (~9 s)
  * not too long to show in class            -> <= 900 frames (~30 s)
  * score = motion of both arms (bimanual) + jaw activity, so something visibly happens
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "nvidia/PhysicalAI-Robotics-Open-H-Embodiment"
ROOT = "Surgical/jhu/imerse/suturebot"
VKEY = "observation.images.endoscope.left"
MIN_LEN, MAX_LEN = 270, 900


def get(path: str, local: Path) -> Path:
    return Path(hf_hub_download(REPO, f"{ROOT}/{path}", repo_type="dataset", local_dir=local))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ranking")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--task", default=None, help='e.g. "needle throw" (default: rank all tasks)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    local = out / "hf"

    info = json.loads(get("meta/info.json", local).read_text())
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
             for l in get("meta/tasks.jsonl", local).read_text().splitlines() if l.strip()}
    lo, hi = map(int, info["splits"]["test"].split(":"))
    cs = info["chunks_size"]

    rows = []
    for ep in range(lo, hi):
        try:
            df = pd.read_parquet(get(f"data/chunk-{ep // cs:03d}/episode_{ep:06d}.parquet", local))
        except Exception as e:  # noqa: BLE001
            print("skip", ep, e); continue
        act = np.stack(df["action"].values)
        n = len(act)
        path1 = np.linalg.norm(np.diff(act[:, 0:3], axis=0), axis=1).sum()
        path2 = np.linalg.norm(np.diff(act[:, 8:11], axis=0), axis=1).sum()
        jaw = np.abs(np.diff(act[:, 7])).sum() + np.abs(np.diff(act[:, 15])).sum()
        task = tasks.get(int(df["task_index"].iloc[0]), "?")
        instr = df["instruction.text"].iloc[0] if "instruction.text" in df else ""
        rows.append(dict(episode=ep, task=task, frames=n, seconds=n / info["fps"],
                         path_psm1=path1, path_psm2=path2, jaw_activity=jaw, instruction=instr))
    R = pd.DataFrame(rows)
    ok = R[(R.frames >= MIN_LEN) & (R.frames <= MAX_LEN)].copy()
    if a.task:
        ok = ok[ok.task == a.task]
    z = lambda s: (s - s.mean()) / (s.std() + 1e-9)  # noqa: E731
    ok["bimanual"] = np.minimum(ok.path_psm1, ok.path_psm2) / np.maximum(ok.path_psm1, ok.path_psm2)
    ok["score"] = z(ok.path_psm1 + ok.path_psm2) + z(ok.jaw_activity) + z(ok.bimanual)
    ok = ok.sort_values("score", ascending=False)
    R.to_csv(out / "all_test_episodes.csv", index=False)
    ok.to_csv(out / "ranked.csv", index=False)
    print(ok.head(a.top).to_string(index=False))

    # contact sheet: 6 evenly spaced endoscope frames per top episode
    from PIL import Image, ImageDraw
    tiles = []
    for ep in ok.episode.head(a.top):
        v = get(f"videos/chunk-{ep // cs:03d}/{VKEY}/episode_{ep:06d}.mp4", local)
        n = int(R.set_index("episode").loc[ep, "frames"])
        row = []
        for i, f in enumerate(np.linspace(0, n - 1, 6).astype(int)):
            png = out / f"ep{ep}_{i}.png"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(v), "-vf",
                            f"select=eq(n\\,{f}),scale=320:180", "-vframes", "1", str(png)], check=True)
            row.append(Image.open(png))
        strip = Image.new("RGB", (320 * 6, 200), "white")
        for i, im in enumerate(row):
            strip.paste(im, (320 * i, 20))
        t = R.set_index("episode").loc[ep]
        ImageDraw.Draw(strip).text((4, 4), f"episode {ep} | {t.task} | {t.seconds:.1f}s", fill="black")
        tiles.append(strip)
    sheet = Image.new("RGB", (320 * 6, 200 * len(tiles)), "white")
    for i, s in enumerate(tiles):
        sheet.paste(s, (0, 200 * i))
    sheet.save(out / "contact_sheet.jpg", quality=90)
    print("wrote", out / "contact_sheet.jpg")


if __name__ == "__main__":
    main()
