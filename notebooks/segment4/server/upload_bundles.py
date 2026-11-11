"""Upload the offline bundles (seg4_bundle_<ep>.zip from precompute.py) to a public HF dataset,
with a clips.json index and a dataset card. The notebook's CACHE_URL points at this repo.

    HF_TOKEN=hf_... python upload_bundles.py [--repo kdb6df/corl2026-seg4-cache] [--src /data/shared]

The token comes from the environment only. It needs write access to the repo, plus permission
to create it if it doesn't exist yet (or create the empty public dataset on the website first).
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

CARD = """---
license: cc-by-4.0
task_categories:
- robotics
tags:
- surgical-robotics
- world-model
- vla
pretty_name: CoRL 2026 Segment 4 offline cache
---

# CoRL 2026 tutorial, Segment 4: offline cache

Precomputed results for the Segment 4 notebook (*Robot learning with a surgical VLA and a world model*), so it runs without the live GPU server.

Each `seg4_bundle_<episode>.zip` holds, for one held-out SutureBot episode:

- `bundle.json`: a GR00T-H-N1.7 action chunk at the start frame (dataset instruction plus three alternative instructions), and the full-episode open-loop predictions and metrics (official GR00T-H 50-step protocol).
- `dream_gt.mp4` / `dream_groot.mp4`: Cosmos-H-Dreams rollouts (72 frames at 10 fps, 288x512, plus the context frame), driven by the recorded actions and by GR00T-H's actions.
- `dream_act.mp4`: the same rollout driven by SutureBot ACT's actions (Segment 3's checkpoint, [kdb6df/corl2026-suturebot-act](https://huggingface.co/kdb6df/corl2026-suturebot-act), which also holds ACT's open-loop predictions).
- `bundle.json` also holds `open_loop_mono` (open-loop predictions of GR00T-H's endoscope-only embodiment, `jhu_imerse_dvrk_mono`) and the `closed_loop` index.
- `closed_groot.mp4` / `closed_groot_actions.npy`: closed loop. GR00T-H (endoscope only) sees a frame, its actions drive Cosmos-H-Dreams for 12 frames (1.2 s), the last generated frame goes back to GR00T-H, repeated to the end of the clip. The actions are the commanded 16-D poses at 30 Hz.
- `closed_recorded.mp4` / `closed_recorded_actions.npy`: the same loop driven by the recorded actions, for comparison.

| Episode | Task | Start frame | File |
|---|---|---|---|
{rows}

Episodes 1441–1451 are in the SutureBot test split and were not used to train either model.

**Sources.** Derived from the SutureBot subset of [nvidia/PhysicalAI-Robotics-Open-H-Embodiment](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment) (CC-BY-4.0). Outputs were generated with [nvidia/GR00T-H-N1.7](https://huggingface.co/nvidia/GR00T-H-N1.7) and [nvidia/Cosmos-H-Dreams](https://huggingface.co/nvidia/Cosmos-H-Dreams) (NVIDIA Open Model License).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="kdb6df/corl2026-seg4-cache")
    ap.add_argument("--src", default="/data/shared")
    a = ap.parse_args()
    from huggingface_hub import HfApi

    zips = sorted(Path(a.src).glob("seg4_bundle_*.zip"))
    assert zips, f"no seg4_bundle_*.zip in {a.src}; run start_services.sh first"
    with tempfile.TemporaryDirectory() as tmp:
        clips = []
        for z in zips:
            b = json.load(zipfile.ZipFile(z).open("bundle.json"))
            clips.append({"episode": b["episode"], "task": b["act"]["instruction"],
                          "start_frame": b["frame"], "file": z.name})
            shutil.copy(z, tmp)
        rows = "\n".join(f"| {c['episode']} | {c['task']} | {c['start_frame']} | `{c['file']}` |" for c in clips)
        Path(tmp, "clips.json").write_text(json.dumps(clips, indent=1))
        Path(tmp, "README.md").write_text(CARD.replace("{rows}", rows))
        api = HfApi()  # reads HF_TOKEN from the environment
        api.create_repo(a.repo, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=tmp,
                          commit_message=f"Segment 4 offline bundles ({len(clips)} clips)")
    print(f"uploaded {len(clips)} bundles -> https://huggingface.co/datasets/{a.repo}")


if __name__ == "__main__":
    main()
