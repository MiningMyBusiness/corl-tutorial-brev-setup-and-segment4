# HANDOFF: CoRL 2026 Tutorial, Segment 4 (GR00T-H-N1.7 + Cosmos-H-Dreams)

You are picking this up on the Brev instance itself, which has GPU, internet, Hugging Face and GitHub access. The previous session could not reach the box and worked through the user pasting outputs back and forth. Read this file first, then `README.md`.

---

## 1. Context

- **Event:** CoRL 2026 tutorial, *Robot Learning on the Path to Surgical Autonomy*, **Nov 12, 2026, 2–6 pm CT**, Austin, in person. A 60-min hands-on block made of four 15-min segments, each its own Colab notebook.
- **Segment 4 (owner: Kiran Bhattacharyya, minutes 45–60):** robot learning and evaluation with a VLA plus a world model, on SutureBot data.
  1. Generate action chunks with **GR00T-H-N1.7** (`nvidia/GR00T-H-N1.7`, fine-tuned on Open-H incl. SutureBot).
  2. Compare with ACT (Segment 3, owner Noah Barnes) using offline metrics.
  3. Pipe actions into **Cosmos-H-Dreams** (`nvidia/Cosmos-H-Dreams`) to show how the scene would evolve, compared with the real video.
  4. Mention ORBIT-Surgical; show a recording if one exists.
- **Participants:** ~30–50 people on free CPU Colab. They call a **shared HTTP API hosted on this Brev instance**. The organizers' risk notes stress that cached or offline results are a must, because waiting on big-model inference live is boring and fragile.
- **Instance teardown loses ALL data.** Every setup step must live in a script in this folder (`brev/setup.sh`, `brev/start_services.sh`). If you fix something by hand, put the fix in the scripts too.

## 2. Decisions already made (don't re-litigate)

| Topic | Decision |
|---|---|
| Model | `nvidia/GR00T-H-N1.7`, embodiment tag `jhu_imerse_dvrk` (user confirmed it's the SutureBot fine-tune). |
| Demo clip | **Episode 1441 from frame 345** (main), **1443 from frame 270** (backup), both needle throws. Changed Nov 4 from 1379/1429, which were in Cosmos-H-Dreams' training data. 1441–1451 are held out for **both** models (1440 is not: frames 0–551 were Cosmos training starts). **All 11 clips (1441–1444 needle throw, 1445–1451 knot tying) are served and pre-computed**, so participants can switch `EPISODE`. Start frames from `server/pick_start_frames.py` (1443 kept at its validated 270). Set via `EPISODES` / `DEMO_FRAMES` in `start_services.sh`. |
| Access | Participants hit the shared API. Port **8000** is exposed via a Brev **Secure Link with authentication disabled**. The app-level `X-API-Key` header is the only auth (all endpoints except `/health`). |
| Metrics | Reuse published protocols, otherwise a minimal implementation with a citation. Noah hasn't settled Segment 3 metrics, so `server/metrics.py` is meant to be shared with him. **Actions:** GR00T-H official `gr00t/eval/open_loop_eval.py`: MSE/MAE over all 16 unnormalized dims, a new 50-step chunk every 50 steps from the real observation. **Video:** Open-H report (arXiv:2604.21017, Methods, Cosmos-H-Surgical-Simulator eval): per-frame L1 in [0,1] and SSIM over 72 generated frames (6 blocks × 12) at ~10 fps, 288×512. No code is published for that, so it's skimage SSIM. |
| Notebook pattern | The main thread uses `#@title` form cells. A separate "🧪 Explore on your own" section holds standalone cells; we chose this over commented-out code. `BACKEND = "api" \| "cache"`: cache replays a pre-computed bundle if the API is down. |
| Comparison design | Two Cosmos rollouts from the same start frame. **Recorded actions** = world-model error baseline. **GR00T-H actions** = policy error added on top. |

## 3. Current state (verified on this box, Nov 4)

- Driver **580.126.09** (Cosmos needs R580+/CUDA 13). ✅
- `brev/setup.sh` ran clean: GR00T-H repo at `~/work/GR00T-H` (uv, py3.10); Cosmos-H-Dreams at `~/work/Cosmos-H-Dreams` (Docker image `cosmos-h-dreams:latest`, checkpoints in `checkpoints/`); SutureBot test split at `/data/suturebot` (symlink into `/data/hf/...`) with GR00T's `modality.json` copied into `meta/`. ✅
- `brev/start_services.sh` starts both services, waits for both, and warms the cache + bundles for the demo clips (~3 min from an empty cache). ✅
  - Gateway on `0.0.0.0:8000`; `/health` OK locally and through the secure link.
  - Cosmos worker container `seg4-cosmos` on `127.0.0.1:8001`; `/health` OK, ckpt `model_ema_bf16_jhutabletop_288x512_h73.pt`.
- **`POST /v1/groot/act {"episode":1379,"frame":0}` works.** It returns a (50,16) chunk in dataset units (xyz in meters, unit quaternions xyzw per arm, jaw in rad). So the `VERIFY` items about `LeRobotEpisodeLoader` / `extract_step_data` / observation building in `gateway.py` evidently work. ✅
- API key lives in `/data/seg4.env` (it's regenerated on every new instance; don't hard-code it anywhere).
- **Not yet run:** `/v1/groot/open_loop` (quality sanity check), any Cosmos dream, `precompute.py`, notebook end-to-end, a load test.

### Earlier local edits: confirmed synced on this box (Nov 4); items 1–3 have since been superseded by §4
1. `gateway.py`: `/v1/cosmos/dream` now saves **absolute 16-D actions** (`{job_id}_abs16.npy`) and no longer converts.
2. `cosmos_worker.py`: converts abs16 → 20-D → normalized 44-D itself. `COSMOS_STATS` defaults to `assets/example_data/offline/stats_tabletop.json`.
3. `cosmos_actions.py`: rot6d changed to **column-major** (first two columns of ΔR), per the repo test `tests/test_controls.py::test_matrix_to_rot6d_layout_is_column_major`.
4. `inspect_cosmos_code.py` (new): a helper that dumps the relevant repo code. You can just read the repo directly instead.
5. `start_services.sh`: `EPISODES` defaulted to `1379,1429` (now 1441–1451).

After syncing, restart with `bash ~/seg4/brev/start_services.sh`.

## 4. Cosmos action format: RESOLVED (Nov 4, on the box)

Answers came from the code that built the student's training data, in **Cosmos-H-Surgical-Simulator** (cloned to `~/work/ref/`; not reproduced by setup.sh, since it's only a reference): `inference_jhu_dvrk_warmup.py` (`--num_frames 73 --chunk_size 72`), `groot_configs.py` (`jhu_dvrk_mono`), and `state_action.py::convert_to_hybrid_relative`. `cosmos_actions.py` now matches the teacher's function exactly (max diff 0.0 on ep 1379).

1. **Relative to what:** the *observed state* (`observation.state`) at the context frame, for the whole rollout. There is no per-block re-anchoring; the old per-block code was wrong. The gateway now sends that state to the worker (`{job}_ref16.npy`).
2. **rot6d:** raw `R_ref^T @ R_t`, first two **columns**, then plain z-score. There's no `− IDENTITY_ROT6D` (that's only in the keyboard control code).
3. **Jaw:** absolute, z-scored.
4. **Rows:** row k = action at raw frame `f0 + 3k` and drives generated frame k+1 (row 0 = the context frame's action). chunk3 recipe: AR step 0 uses **8** actions, later steps use 12. For 72 frames the worker runs 7 AR steps (80 frames) and trims to context + 72.
5. **Stride 3** is pre-applied by us; the runner passes the `.npy` through as-is (only zero-pads to 44).
6. **The mp4 includes the context frame** (73 frames at 10 fps after trimming).

Also:
- **Units:** SutureBot xyz is in **meters**, not cm. The stats are in meters too, so no conversion is needed.
- **Checkpoint path bug fixed:** `hf download` put the weights in `checkpoints/checkpoints/`. `setup.sh` now symlinks them.
- `trajectory.npy` in the repo is a synthetic keyboard-style ramp, not a format reference.

**Validation** (`server/validate_cosmos.py`, 72 frames, L1 ↓ / SSIM ↑):

| | ep 1379 f0 | ep 1429 f0 |
|---|---|---|
| context frame held (baseline) | 0.043 / 0.776 | 0.054 / 0.728 |
| **recorded actions** | **0.020 / 0.869** | **0.026 / 0.831** |
| old per-block format | 0.042 / 0.765 | n/a |
| PSM1 frozen | 0.041 / 0.777 | 0.045 / 0.768 |
| arms swapped | 0.049 / 0.733 | 0.051 / 0.715 |

This is in line with the Open-H report (≈0.01–0.02 / 0.80–0.85). Visually, PSM1 is the instrument on the image's right, and it stays still when frozen. `POST /v1/cosmos/dream` (gt, 1379, frame 0) end to end gives the same numbers.

**Timing:** ~35 s per job through the command-line tool (model load + compile), with only ~3.8 s of generation once loaded. That's fine for precompute. A resident pipeline would make live/explore dreams ~4 s.

**Held-out split:** our dataset is exactly `hf_suturebot` (516,334 frames) in the Cosmos student's training mix, which holds out the last 1% of *frames* (5,163). That boundary falls at frame 552 of episode 1440, so **1441–1451** are the clips unseen by both models. Hence the switch to 1441/1443.

## 5. Status of the Section 5 steps (Nov 4)

1. **GR00T quality check: done.** `/v1/groot/open_loop` was crashing on every episode (the last chunk's future GT actions ran past the episode end). Fixed with `allow_padding=True`, which doesn't change the policy input. Ep 1441: MSE 0.0076 / MAE 0.0296; xyz 1.37 mm, quat 0.021, jaw 0.149 rad. A "hold pose for 50 steps" baseline gets 1.16 mm / 0.017 / 0.061, because slow surgical motion makes holding still strong over 1.7 s. The overlay tracks well (no camera/instruction bug). The notebook now shows the baseline row. Two fixes along the way:
   - **Quaternion sign:** q ≡ −q. 5.5% of PSM2 rows came out sign-flipped in one run, quadrupling quat MAE. `metrics.align_quat_signs` (gateway output + notebook) aligns before scoring. Apply it to ACT too.
   - **Determinism:** GR00T-H is seeded per (episode, frame, instruction), so cache rebuilds reproduce the numbers exactly. The cache key has a version (`CACHE_VERSION` in `gateway.py`).
2. **Cosmos timing: done.** The worker now keeps the repo's runner resident (`COSMOS_RESIDENT=1`, the default; `0` = old per-job subprocess). Warm-up takes ~80 s at start, then **~5.5 s per dream** instead of ~31 s. Same quality (L1 0.0255 vs 0.0260 on 1441@345).
3. **Secure link: tested Nov 4** (`https://corl-seg4-1ydgfxbcn.gobrev.dev`, changes on relaunch). No Brev login wall; the API key is enforced (401 without it). `load_test.py` through the proxy, 50 users, 0 errors: 350 cached main-thread requests in one burst p50 0.9–5.9 s, max 9 s (vs <2 s on the box, so the proxy adds burst latency); 50 uncached GR00T chunks max 6.8 s; 10 uncached dreams all done in 55 s (~6 s apart). Peak GPU 23.3 GB. Full notebook run through the link: 13.5 s (1441), 25.7 s (1447 incl. two live dreams).
4. **Precompute + bundle: done for all 11 clips, automatic** (~6 min in `start_services.sh`). Writes `/data/shared/seg4_bundle_<ep>.zip` (0.6–0.8 MB each, incl. all three instruction variants). Hosting: HF dataset **`kdb6df/corl2026-seg4-cache`** (the notebook's `CACHE_URL` is a template with `{episode}`). **Uploaded Nov 4** (`server/upload_bundles.py`): 11 zips + `clips.json` + dataset card, public, downloads without a token. Notebook run end to end in cache mode from it (1445).
   Per-clip check (72 frames; open-loop xyz MAE mm vs hold-pose; dream L1/SSIM vs real for frozen frame / recorded actions / GR00T-H actions). Recorded beats frozen on every clip, so the format holds beyond 1441/1443:

   | Ep | Task | GR00T xyz | hold | frozen frame | recorded | GR00T-H |
   |---|---|---|---|---|---|---|
   | 1441 | throw | 1.37 | 1.16 | 0.041/0.776 | 0.025/0.851 | 0.042/0.771 |
   | 1442 | throw | 1.69 | 1.65 | 0.053/0.739 | 0.025/0.849 | 0.039/0.779 |
   | 1443 | throw | 1.53 | 1.51 | 0.049/0.759 | 0.028/0.848 | 0.042/0.774 |
   | 1444 | throw | 1.21 | 0.93 | 0.029/0.815 | 0.018/0.877 | 0.033/0.789 |
   | 1445 | knot | 1.98 | 1.58 | 0.037/0.794 | 0.022/0.850 | 0.030/0.805 |
   | 1446 | knot | 1.48 | 1.63 | 0.048/0.750 | 0.024/0.842 | 0.035/0.780 |
   | 1447 | knot | 1.33 | 1.33 | 0.038/0.760 | 0.023/0.844 | 0.032/0.794 |
   | 1448 | knot | 1.62 | 1.26 | 0.039/0.773 | 0.025/0.822 | 0.035/0.771 |
   | 1449 | knot | 1.48 | 1.28 | 0.040/0.766 | 0.025/0.831 | 0.039/0.753 |
   | 1450 | knot | 1.45 | 1.13 | 0.044/0.748 | 0.022/0.840 | 0.031/0.788 |
   | 1451 | knot | 1.60 | 1.22 | 0.032/0.801 | 0.019/0.867 | 0.028/0.810 |
5. **Notebook end-to-end: done locally, not yet on real Colab.** Run in a fresh Python 3.12 env with no HF token in `api`, `cache`, and API-unreachable (auto-fallback) modes, every cell. Fixes: `jinja2` added to the pip line; Explore dream cells print "needs the live server" in cache mode instead of crashing; Explore cell E accepts a URL and defaults to `ACT_PRED_URL`. **The dataset downloads without an HF token.** Also run (Nov 4) through the secure link in `api` mode on 1441 and 1447, and in fallback mode on 1447 with per-episode `CACHE_URL`. Still worth one pass on real Colab.
6. **Load test: done on the box** (`server/load_test.py`). 50 users × 7 cached main-thread calls at once: all <2 s, no errors. 50 uncached GR00T chunks at once: max 5.9 s. 20 uncached dreams at once: all done in 109 s. Peak GPU memory 22 GB of 46 GB.
7. **Automation: done.** `setup.sh` symlinks the Cosmos checkpoints. `start_services.sh` waits for the old gateway to exit (there was a restart race), waits for the worker, and warms the cache. A full teardown → `start_services.sh` restore was tested by moving the cache aside. Not done: making the Launchable run setup automatically.
8. **Segment 3 alignment: drafted.** `ACT_INTERFACE.md` describes the file format, protocol, metrics and hand-off for Noah. The ACT hook was tested end to end with a stand-in file.

**Before Nov 12** (dress rehearsal Nov 10): put the final secure link + API key in the notebook / slides (see `REBUILD.md`); one real-Colab pass; send `ACT_INTERFACE.md` to Noah; full dress rehearsal.

## 6. File map (`~/seg4` = `notebooks/segment4/` in the user's repo)

| File | Purpose |
|---|---|
| `segment4_vla_world_model.ipynb` | Participant Colab notebook, no outputs (built from a generator; edit the JSON directly or regenerate). |
| `segment4_vla_world_model_executed.ipynb` | Same notebook run end to end through the secure link (Nov 4, ep 1441), outputs saved, API key blanked. Reference / slide material; regenerate after changes. |
| `segment4_vla_world_model_v2.ipynb` | v2 (Nov 10), v1 kept as is: action plots in mm / rad, Explore A/B accept off-script or typed instructions, Explore C has 10 perturbations shown next to the real video and the recorded-actions rollout. Run end to end on the box in `api` and `cache` modes. |
| `server/gateway.py` | FastAPI on :8000. GR00T-H loaded in-process (`Gr00tPolicy`, official `LeRobotEpisodeLoader` / `extract_step_data` / `parse_*_gr00t` helpers). Disk cache in `/data/shared/cache`; single Cosmos job queue. Endpoints: `/health`, `/v1/episodes`, `/v1/groot/act`, `/v1/groot/open_loop`, `/v1/cosmos/dream` (async), `/v1/jobs/{id}`, `/v1/files/{name}`. |
| `server/cosmos_worker.py` | FastAPI on 127.0.0.1:8001 inside the `seg4-cosmos` container. Extracts the start frame (ffmpeg, 512×288), converts actions, and runs the repo's `CosmoshRunner` kept resident on one GPU thread (`COSMOS_RESIDENT=0` = old `flashdreams-run` subprocess per job). Runs `cosmos_actions.ar_steps(blocks)` AR steps and trims the output to context + 12·blocks frames. Env: `COSMOS_STATS`, `COSMOS_CKPT`, `COSMOS_RECIPE`. |
| `server/cosmos_actions.py` | abs16 (xyz + quat **xyzw** + jaw per arm, 30 Hz) + context state → hybrid-relative 20-D at stride 3 → z-score + pad to 44. Matches the teacher repo exactly. |
| `server/metrics.py` | Open-loop MSE/MAE (GR00T-H official) and per-frame L1/SSIM (Open-H report). Shared with Segment 3. |
| `server/rank_episodes.py` | Ranks test episodes (length 270–900 frames, bimanual motion, jaw activity) and makes a contact sheet. |
| `server/pick_start_frames.py` | Picks each clip's dream start frame (214-frame window with most two-arm motion + jaw activity); prints a `DEMO_FRAMES` string. |
| `server/precompute.py` | Warms the cache for the main thread and writes `bundle_<ep>/` + `seg4_bundle_<ep>.zip` for the notebook's cache backend. Run automatically by `start_services.sh`. |
| `server/load_test.py` | Simulates a room: N users of cached main-thread calls, N uncached GR00T calls, D uncached dreams; reports latency + peak GPU memory. Works against localhost or the secure link. |
| `ACT_INTERFACE.md` | For Noah: what ACT must output to appear in the Segment 4 comparison. |
| `CLOSED_LOOP.md` | Plan for running GR00T-H inside Cosmos-H-Dreams (endoscope-only `jhu_imerse_dvrk_mono`). Not implemented. |
| `server/validate_cosmos.py` | Format check: dreams recorded + perturbed actions and scores them vs the real video. Must print PASS. |
| `server/verify_cosmos_format.py`, `server/inspect_cosmos_code.py` | Obsolete diagnostics from before §4 was resolved. Left on the box, not in the repo copy. |
| `brev/setup.sh` | One-time, re-runnable install (driver check, uv, both repos, checkpoints, Docker build, SutureBot test split, API key). Needs `HF_TOKEN` env; optional `API_KEY` to keep the slide key. |
| `REBUILD.md` | What changes when the Launchable is rebuilt from scratch. |
| `brev/start_services.sh` | (Re)starts the Cosmos container and gateway, then waits for health. Logs: `/data/shared/gateway.log`, `docker logs seg4-cosmos`. |

## 7. Key technical facts (sourced)

- **GR00T-H-N1.7:** 3B params, Cosmos-Reason2-2B backbone + flow-matching DiT action head, bf16. `jhu_imerse_dvrk` uses video `endoscope_left`, `wrist_left`, `wrist_right`. State is 16-D (`psm{1,2}_pose` 7 + `psm{1,2}_gripper` 1) but has **100% state dropout** for this embodiment. Actions are 16-D, horizon 50 @ 30 Hz; poses are `REL_XYZ_ROT6D` internally, but the policy returns absolute dataset-space values. Language key `annotation.human.task_description` comes from `task_index`. API: `Gr00tPolicy(model_path, embodiment_tag, device, strict)`; `get_action(obs) -> (action, info)`; obs = `{"video": {k: (B,T,H,W,3) uint8}, "state": {k: (B,T,D)}, "language": {k: [[str]]}}`. Official ZMQ server: `gr00t/eval/run_gr00t_server.py` (we don't use it).
- **SutureBot (Open-H `Surgical/jhu/imerse/suturebot`):** LeRobot v2.1, 1452 episodes, 30 fps, AV1 video. Cameras: `observation.images.endoscope.{left,right}` 540×960, `observation.images.wrist.{left,right}` 480×640. Tasks: 0 knot tying, 1 needle pickup, 2 needle throw. Splits: train 0–1160, val 1161–1305, test 1306–1451.
- **Cosmos-H-Dreams:** distilled real-time student of Cosmos-H-Surgical-Simulator (Cosmos-Predict2.5-2B). One context frame (288×512) plus action chunks yields 12-frame blocks at ~10 Hz. Unified 44-D action space; the dVRK checkpoint uses 20-D dual-arm. Needs Python 3.12, CUDA 13, ≥12 GB VRAM, bf16 only. Offline mode: `flashdreams-run` with a JSON manifest `[{input, input_action, output_video}]`; `start_frame_idx` is allowed per entry. Streaming WebRTC servers exist (`cosmosHDreams.webrtc.server*`), but there's no REST API, so we wrap the offline CLI.
- **Open-H report results to quote:** GR00T-H matched ACT at 33% SutureBot data (≈47%), reached ≈73% vs ACT ≈50% with full data, and was the only model to complete end-to-end suturing (25%). The report recommends freezing the VLM and fine-tuning the action head for downstream tasks.

## 8. Rules and cautions

- **Never write tokens into files or commits.** The user's HF and GitHub tokens were pasted in earlier chats (HF again on Nov 4) and should be rotated. Use the `HF_TOKEN` env var only.
- Don't commit the user's untracked personal files (CV PDFs, LaTeX build files, `.DS_Store`) in the repo root. If pushing, push only `notebooks/segment4/` on a branch, and only when the user asks.
- Keep the notebook runnable on **free CPU Colab** with no NVIDIA packages; all GPU work stays on this box.
- Main-thread notebook cells must also work in `cache` mode. Explore cells must not depend on state created by other explore cells.
- The tutorial is **Nov 12, 2026, 2–6 pm CT**. The full dress rehearsal (~30 concurrent users) is **Nov 10**.

## 9. Sources

- GR00T-H: https://huggingface.co/nvidia/GR00T-H-N1.7 · https://github.com/NVIDIA-Medtech/GR00T-H (`getting_started/policy.md`, `gr00t/eval/open_loop_eval.py`, `open_h/embodiments/jhu_imerse_dvrk/`)
- Cosmos-H-Dreams: https://huggingface.co/nvidia/Cosmos-H-Dreams · https://github.com/isaac-for-healthcare/Cosmos-H-Dreams (`cosmosHDreams/GUIDE.md`, `cosmosHDreams/README.md`) · https://huggingface.co/blog/nvidia/cosmos-h-dreams
- Action conversion spec: https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/predict2_5/surgical_robotics/post_training.html
- Open-H-Embodiment report: https://arxiv.org/abs/2604.21017 (also `open-h-embodiment-report.pdf` in the repo root)
- Brev Launchables: https://docs.nvidia.com/brev/concepts/launchables
