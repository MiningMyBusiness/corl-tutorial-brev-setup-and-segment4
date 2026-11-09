"""Convert SutureBot 16-D absolute actions (30 Hz) into Cosmos-H-Dreams action input.

Format = what the released tabletop student (model_ema_bf16_jhutabletop_288x512_h73.pt)
was trained on. Source: Cosmos-H-Surgical-Simulator (github.com/NVIDIA-Medtech/
Cosmos-H-Surgical-Simulator), the repo that built the student's training cache:
  * action/inference/inference_jhu_dvrk_warmup.py  (--num_frames 73 --chunk_size 72)
  * action/datasets/gr00t_dreams/groot_configs.py  ("jhu_dvrk_mono": rel_xyz_rot6d poses
    referenced to state.psm{1,2}_pose at delta_indices=[0]; absolute grippers; mean_std)
  * action/datasets/gr00t_dreams/data/transform/state_action.py::convert_to_hybrid_relative

Per sample (one context frame f0 + 72 frames at stride 3):
  row k (k = 0..71) = action at raw frame f0 + 3k   -> conditions generated frame k+1
  xyz   = action_xyz - state_xyz(f0)                   (base frame, meters)
  rot6d = first two COLUMNS of R_state(f0)^T @ R_action, flattened [r00 r10 r20 r01 r11 r21]
  jaw   = absolute
  layout psm1 xyz(0-2) rot6d(3-8) jaw(9) | psm2 xyz(10-12) rot6d(13-18) jaw(19)
  z-score with stats_tabletop.json, zero-pad to 44 (the runner also pads).
The reference is the state at the context frame for the WHOLE rollout (no per-block
re-anchoring), matching the cumulative position the repo's keyboard integrator keeps.

Block layout of the chunk3 recipe (cosmosHDreams/encoder/action.py): AR step 0 consumes
8 actions, every later step 12. One action per output frame.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation as R

STRIDE = 3          # 30 Hz -> 10 Hz (stats_tabletop.json: timestep_interval = 3)
CHUNK = 12          # frames per user-facing block (Open-H report protocol: 6 x 12 = 72)
FIRST_STEP = 8      # actions consumed by AR step 0 (chunk3: 2 latent frames x 4)
STEP = 12           # actions per later AR step (3 latent frames x 4)
FORMAT_VERSION = 2  # bump when the conversion changes (busts cached dreams)


def n_frames(blocks: int) -> int:
    """Generated frames we keep (excluding the context frame)."""
    return CHUNK * blocks


def ar_steps(blocks: int) -> int:
    """AR steps needed so the model produces at least n_frames(blocks) frames."""
    need = n_frames(blocks)
    return 1 + max(0, -(-(need - FIRST_STEP) // STEP))


def n_model_actions(blocks: int) -> int:
    return FIRST_STEP + STEP * (ar_steps(blocks) - 1)


def n_raw(blocks: int) -> int:
    """30 Hz action rows (from the context frame on) that drive the kept frames."""
    return STRIDE * (n_frames(blocks) - 1) + 1


def _arm(x16: np.ndarray, arm: int):
    o = 8 * arm
    # SutureBot orientation is x, y, z, w (meta/info.json names), scipy's order too.
    return x16[..., o:o + 3], x16[..., o + 3:o + 7], x16[..., o + 7:o + 8]


def absolute16_to_relative20(actions16: np.ndarray, ref_state16: np.ndarray,
                             n_actions: int | None = None) -> np.ndarray:
    """actions16: (T, 16) absolute at 30 Hz, row 0 = action at the context frame.
    ref_state16: (16,) observed state at the context frame.
    Returns (n_actions, 20) hybrid-relative actions at 10 Hz. If the input is shorter
    than needed, the last action is held (those rows only drive frames we trim)."""
    sub = np.asarray(actions16, np.float64)[::STRIDE]
    if n_actions is not None:
        if len(sub) < n_actions:
            sub = np.concatenate([sub, np.repeat(sub[-1:], n_actions - len(sub), 0)])
        sub = sub[:n_actions]
    ref = np.asarray(ref_state16, np.float64)
    parts = []
    for arm in (0, 1):
        xyz, quat, jaw = _arm(sub, arm)
        rxyz, rquat, _ = _arm(ref, arm)
        rel_R = R.from_quat(rquat).as_matrix().T @ R.from_quat(quat).as_matrix()
        rot6d = rel_R[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)  # column-major
        parts += [xyz - rxyz, rot6d, jaw]
    return np.concatenate(parts, axis=1).astype(np.float32)


def load_stats(stats_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-key mean/std in concat order (psm1_pose, psm1_gripper, psm2_pose, psm2_gripper)."""
    st = json.load(open(stats_path))
    keys = ["action.psm1_pose", "action.psm1_gripper", "action.psm2_pose", "action.psm2_gripper"]
    if all(k in st for k in keys):
        mean = np.concatenate([np.asarray(st[k]["mean"], np.float32).ravel() for k in keys])
        std = np.concatenate([np.asarray(st[k]["std"], np.float32).ravel() for k in keys])
    else:
        mean = np.asarray(st["action"]["mean"], np.float32)[:20]
        std = np.asarray(st["action"]["std"], np.float32)[:20]
    assert mean.shape == (20,) and std.shape == (20,), (mean.shape, std.shape)
    return mean, std


def normalize_and_pad(rel20: np.ndarray, stats_path: str, pad_to: int = 44) -> np.ndarray:
    mean, std = load_stats(stats_path)
    x = rel20.astype(np.float32).copy()
    m = std != 0  # same rule as the repo's Normalizer (mean_std)
    x[:, m] = (x[:, m] - mean[m]) / std[m]
    if pad_to and x.shape[1] < pad_to:
        x = np.pad(x, ((0, 0), (0, pad_to - x.shape[1])))
    return x


def to_cosmos(actions16: np.ndarray, ref_state16: np.ndarray, blocks: int,
              stats_path: str) -> np.ndarray:
    """(T,16) absolute @30 Hz + (16,) state -> (n_model_actions, 44) normalized."""
    rel = absolute16_to_relative20(actions16, ref_state16, n_model_actions(blocks))
    return normalize_and_pad(rel, stats_path)
