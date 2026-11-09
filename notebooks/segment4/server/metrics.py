"""Evaluation metrics for Segment 4 (shared so Segment 3 / ACT can import the same file).

Everything here re-implements *published* protocols as literally as possible:

1. Open-loop action error  -> GR00T-H official open-loop eval
   (NVIDIA-Medtech/GR00T-H, gr00t/eval/open_loop_eval.py, `evaluate_single_trajectory`):
       mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
       mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))
   computed on the *unnormalized* 16-D SutureBot action vector
   (psm1 xyz+quat(4)+jaw, psm2 xyz+quat(4)+jaw), with a new chunk predicted every
   `action_horizon` steps from the ground-truth observation.

2. World-model video fidelity -> Open-H-Embodiment report, Methods,
   "Cosmos-H-Surgical-Simulator" evaluation (arXiv:2604.21017):
   per-frame L1 (mean absolute error in [0,1] pixel space) and SSIM, over
   72 autoregressively generated frames (6 chunks x 12 frames) at ~10 fps
   (SutureBot: every 3rd 30 Hz frame), 288x512 resolution.
   The report publishes no metric code, so this is a minimal implementation
   using scikit-image's SSIM (Wang et al., 2004).
"""
from __future__ import annotations

import numpy as np

ACTION_NAMES = (
    [f"psm1_{k}" for k in ("x", "y", "z", "qx", "qy", "qz", "qw", "jaw")]
    + [f"psm2_{k}" for k in ("x", "y", "z", "qx", "qy", "qz", "qw", "jaw")]
)


# ----------------------------------------------------------------- actions ---
def stitch_open_loop(chunks: list[np.ndarray], action_horizon: int, n_steps: int) -> np.ndarray:
    """Concatenate the first `action_horizon` steps of each predicted chunk
    (chunk i predicted at t = i * action_horizon), truncated to n_steps.
    Mirrors how open_loop_eval.py builds `pred_action_across_time`."""
    pred = np.concatenate([c[:action_horizon] for c in chunks], axis=0)
    return pred[:n_steps]


def open_loop_action_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """GR00T-H open_loop_eval.py metrics. gt/pred: (T, 16) unnormalized."""
    assert gt.shape == pred.shape, f"shape mismatch {gt.shape} vs {pred.shape}"
    return {
        "mse": float(np.mean((gt - pred) ** 2)),
        "mae": float(np.mean(np.abs(gt - pred))),
    }


QUAT_SLICES = (slice(3, 7), slice(11, 15))


def align_quat_signs(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """q and -q are the same rotation. Flip predicted quaternions into the ground truth's
    hemisphere so the raw-component MSE/MAE does not count a sign flip as error.
    Apply before open_loop_action_metrics (to every model, so comparisons stay fair).
    Without it, GR00T-H ep 1441 psm2 quat MAE went 0.016 -> 0.060 from 5.5% flipped rows."""
    pred = np.array(pred, copy=True)
    for s in QUAT_SLICES:
        flip = np.sum(gt[:, s] * pred[:, s], axis=1) < 0
        pred[flip, s] *= -1
    return pred


# Not part of the official protocol: added so the all-dims numbers (which mix meters,
# quaternions and radians) can be read against a no-model reference.
ACTION_GROUPS = {"xyz_mm": [0, 1, 2, 8, 9, 10], "quat": [3, 4, 5, 6, 11, 12, 13, 14], "jaw_rad": [7, 15]}


def hold_pose_baseline(gt: np.ndarray, action_horizon: int) -> np.ndarray:
    """'Do nothing' policy under the same protocol: every `action_horizon` steps, repeat
    the recorded action at the chunk start for the whole chunk."""
    starts = np.arange(len(gt)) // action_horizon * action_horizon
    return gt[starts]


def per_group_mae(gt: np.ndarray, pred: np.ndarray) -> dict:
    """MAE per physical group: xyz in mm, quaternion components, jaw in rad."""
    out = {}
    for name, ix in ACTION_GROUPS.items():
        scale = 1000.0 if name == "xyz_mm" else 1.0
        out[f"mae_{name}"] = float(np.mean(np.abs(gt[:, ix] - pred[:, ix])) * scale)
    return out


# ------------------------------------------------------------------- video ---
def _to_float01(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.dtype == np.uint8:
        return frames.astype(np.float32) / 255.0
    return frames.astype(np.float32)


def per_frame_l1_ssim(gen: np.ndarray, gt: np.ndarray) -> dict:
    """gen, gt: (T, H, W, 3), uint8 or float in [0,1], already aligned in time and
    resized to the same resolution (288x512 in the report). Returns per-frame arrays."""
    from skimage.metrics import structural_similarity

    gen, gt = _to_float01(gen), _to_float01(gt)
    n = min(len(gen), len(gt))
    gen, gt = gen[:n], gt[:n]
    l1 = np.abs(gen - gt).mean(axis=(1, 2, 3))
    ssim = np.array(
        [structural_similarity(a, b, channel_axis=-1, data_range=1.0) for a, b in zip(gen, gt)]
    )
    return {"l1": l1, "ssim": ssim, "mean_l1": float(l1.mean()), "mean_ssim": float(ssim.mean())}
