# Plugging ACT (Segment 3) into the Segment 4 comparison

Segment 4 compares GR00T-H-N1.7 with ACT on the same clip, using the same open-loop protocol. If ACT produces one `.npy` in the format below, the Segment 4 notebook shows it next to GR00T-H and can also run it through Cosmos-H-Dreams. No other integration is needed.

## The clip

- SutureBot (`nvidia/PhysicalAI-Robotics-Open-H-Embodiment`, `Surgical/jhu/imerse/suturebot`), **episode 1441** (needle throw, 601 frames). The backup is episode 1443.
- Episodes 1441–1451 are in the test split (1306–1451) and were held out from GR00T-H *and* Cosmos-H-Dreams training. ACT shouldn't train on them either. (1440 is *not* fully held out from Cosmos.) Participants can switch the notebook to any of 1441–1451, so ACT files for other episodes (`act_open_loop_ep<N>.npy`, same format) are welcome but optional.

## What to produce

A file named `act_open_loop_ep1441.npy` containing a float32 array of shape **(601, 16)**: the episode's full length at 30 Hz.

- **Units:** the dataset's own absolute action space, identical to the `action` column in the parquet: `psm1 x y z (m), qx qy qz qw, jaw (rad), psm2 x y z, qx qy qz qw, jaw`.
- **Protocol** (GR00T-H's official `gr00t/eval/open_loop_eval.py`): at t = 0, 50, 100, …, give ACT the *recorded* observation at t, predict a chunk, and keep its first 50 steps. Concatenate the kept steps and truncate to 601 rows.
  - If ACT's chunk is shorter than 50 steps, keep the whole chunk and tell us the chunk length. We'll then rerun GR00T-H with the same stride, so the comparison stays fair.
- Store absolute poses, not deltas. If ACT predicts relative actions internally, convert back before saving.

## Metrics

Use `server/metrics.py`, the file Segment 4 uses:

```python
import numpy as np, metrics
gt = ...   # (601, 16) from the parquet's `action` column
act = np.load("act_open_loop_ep1441.npy")
act = metrics.align_quat_signs(gt, act)                           # q and -q are the same rotation: do this first
metrics.open_loop_action_metrics(gt, act)                         # official MSE / MAE over all 16 dims
metrics.per_group_mae(gt, act)                                    # MAE for xyz (mm), quaternion, jaw (rad)
metrics.per_group_mae(gt, metrics.hold_pose_baseline(gt, 50))     # no-model reference
```

GR00T-H on episode 1441, for reference (after sign alignment): MSE 0.0076, MAE 0.0296, xyz 1.37 mm, quat 0.021, jaw 0.149 rad. The hold-pose baseline: 1.16 mm, 0.017, 0.061 rad. The notebook applies `align_quat_signs` to every model, so skipping it only makes your own numbers look worse than what the notebook shows.

## Getting it into the notebook

Host the `.npy` anywhere with a public URL (a GitHub release or an HF repo) and send us the link. We'll put it in the notebook's `ACT_PRED_URL` field. An Explore cell then dreams ACT's actions through Cosmos-H-Dreams from the same start frame (1441 @ frame 345), next to the recorded and GR00T-H rollouts.
