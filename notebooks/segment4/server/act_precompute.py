"""Precompute SutureBot ACT predictions for the Segment 4 notebook (one .npz per clip).

    ~/work/act-env/bin/python act_precompute.py --ckpt dot_best_6311.ckpt --stats checkpoint_normalization.json \
        --suturebot ~/work/SutureBot --out /data/act_release/predictions

Per clip (act_<episode>.npz, all (N, 16) absolute xyz + quat xyzw + jaw at 30 Hz, like GR00T-H's):
  open_loop   whole clip, GR00T-H's protocol: new chunk on the real frames every 50 steps, first 50 rows kept
  start_frame the clip's demo start frame (DEMO_FRAMES, same as the gateway)
  chunk       the full 60-step chunk at start_frame
  dream       the same receding-horizon rule from start_frame, n_raw(6 blocks) rows: drives Cosmos (section 4)
Runs in its own env (see act_policy.py); does not touch the live server.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import cosmos_actions  # noqa: E402
from act_policy import ACT, Episode, receding_horizon  # noqa: E402

DEMO = "1441:345,1442:255,1443:270,1444:330,1445:105,1446:120,1447:105,1448:105,1449:30,1450:165,1451:135"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--suturebot", required=True, help="SutureBot repo, ACT branch")
    ap.add_argument("--data_root", default="/data/suturebot")
    ap.add_argument("--demo_frames", default=DEMO)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    act = ACT(a.ckpt, a.stats, a.suturebot)
    n_dream = cosmos_actions.n_raw(a.blocks)
    summary = []
    for pair in a.demo_frames.split(","):
        ep, f0 = map(int, pair.split(":"))
        t0 = time.time()
        e = Episode(a.data_root, ep)
        ol = receding_horizon(act, e, 0, len(e.actions))
        chunk = act.chunk(e.frames([f0])[f0], e.states[f0], e.command)
        dream = receding_horizon(act, e, f0, n_dream)
        assert len(dream) == n_dream, (ep, len(dream))
        np.savez_compressed(out / f"act_{ep}.npz", open_loop=ol, chunk=chunk, dream=dream, start_frame=f0,
                            command=e.command, horizon=50, checkpoint_epoch=act.epoch)
        xyz = 1000 * np.abs(ol[:, [0, 1, 2, 8, 9, 10]] - e.actions[:len(ol), [0, 1, 2, 8, 9, 10]]).mean()
        jaw = np.abs(ol[:, [7, 15]] - e.actions[:len(ol), [7, 15]]).mean()
        summary.append({"episode": ep, "task": e.command, "start_frame": f0, "open_loop_xyz_mae_mm": round(float(xyz), 3),
                        "open_loop_jaw_mae_rad": round(float(jaw), 3)})
        print(summary[-1], f"{time.time() - t0:.0f}s", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
