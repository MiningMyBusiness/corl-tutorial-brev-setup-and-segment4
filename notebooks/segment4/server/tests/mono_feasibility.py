"""Feasibility check for the closed loop (CLOSED_LOOP.md, build step 1). Runs next to the live
server; loads only the endoscope-only GR00T-H embodiment (~7 GB GPU).

1. open-loop accuracy of jhu_imerse_dvrk_mono on an episode (official 50-step protocol)
2. same, with the endoscope frame degraded to Cosmos' 288x512 and back
3. does the policy use the pose state? (shift the state 1 cm, see whether actions follow)
4. latency per call
Run from ~/work/GR00T-H:  uv run python mono_feasibility.py <seg4/server dir> [episode]
"""
import sys, time, zlib
import numpy as np, torch, cv2

sys.path.insert(0, sys.argv[1])
import metrics  # noqa: E402

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.eval.open_loop_eval import parse_action_gr00t, parse_observation_gr00t

EP = int(sys.argv[2]) if len(sys.argv) > 2 else 1441
TAG_S = "jhu_imerse_dvrk_mono"
policy = Gr00tPolicy(model_path="nvidia/GR00T-H-N1.7", embodiment_tag=TAG_S, device="cuda:0", strict=False)
MOD = policy.get_modality_config(); TAG = EmbodimentTag(TAG_S)
print("modality video keys:", MOD["video"].modality_keys, "| state:", MOD["state"].modality_keys)
loader = LeRobotEpisodeLoader(dataset_path="/data/suturebot", modality_configs=MOD, video_backend="torchcodec")
traj = loader[EP]
AK = list(MOD["action"].modality_keys)
gt = np.concatenate([np.stack(traj[f"action.{k}"].values) for k in AK], axis=1)
T = len(gt)


def obs_at(step, degrade=False, state_shift=None):
    dp = extract_step_data(traj, step, MOD, TAG, allow_padding=True)
    o = {f"state.{k}": np.array(v) for k, v in dp.states.items()}
    for k, v in dp.images.items():
        v = np.array(v)
        if degrade:  # what a Cosmos frame looks like: 512x288, then back up to the native size
            v = np.stack([cv2.resize(cv2.resize(f, (512, 288), interpolation=cv2.INTER_AREA),
                                     (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR) for f in v])
        o[f"video.{k}"] = v
    if state_shift is not None:
        for k, d in state_shift.items():
            o[f"state.{k}"] = o[f"state.{k}"] + d
    o[MOD["language"].modality_keys[0]] = dp.text
    return parse_observation_gr00t(o, MOD), o


def chunk(step, **kw):
    obs, _ = obs_at(step, **kw)
    torch.manual_seed(zlib.crc32(f"{EP}:{step}:mono".encode()))
    t0 = time.time()
    a, _ = policy.get_action(obs)
    dt = time.time() - t0
    a = parse_action_gr00t(a)
    return np.concatenate([a[f"action.{k}"] for k in AK], axis=1), dt


_, o0 = obs_at(0)
print("raw obs shapes:", {k: getattr(v, "shape", v) for k, v in o0.items()})

for label, kw in [("native frames", {}), ("Cosmos-like 288x512 frames", {"degrade": True})]:
    chunks, lat = [], []
    for s in range(0, T, 50):
        c, dt = chunk(s, **kw); chunks.append(c); lat.append(dt)
    pred = metrics.align_quat_signs(gt, metrics.stitch_open_loop(chunks, 50, T))
    print(f"[{label}] {metrics.open_loop_action_metrics(gt, pred)} {metrics.per_group_mae(gt, pred)} "
          f"| latency median {1000 * np.median(lat):.0f} ms")
print("[hold pose]", metrics.per_group_mae(gt, metrics.hold_pose_baseline(gt, 50)))

# state sensitivity: shift PSM1's position 1 cm in x, see how far the predicted PSM1 x moves
base, _ = chunk(300)
_, o = obs_at(300)
shift = np.zeros_like(o["state.psm1_pose"]); shift[..., 0] = 0.01
moved, _ = chunk(300, state_shift={"psm1_pose": shift})
print(f"state +1 cm in psm1 x -> predicted psm1 x moves by {1000 * np.mean(moved[:, 0] - base[:, 0]):.1f} mm "
      f"(psm2 x: {1000 * np.mean(moved[:, 8] - base[:, 8]):.1f} mm)")
print(f"GPU memory allocated by this process: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
