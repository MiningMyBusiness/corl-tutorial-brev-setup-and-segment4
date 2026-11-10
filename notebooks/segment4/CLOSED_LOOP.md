# Closed loop inside the world model

Status, Nov 10: **server side built on branch `seg4-closed-loop`, tested with a stand-in Cosmos worker; not yet run with the real Cosmos-H-Dreams, not deployed, no notebook section yet.** See "Implementation status" at the end.

## The idea

Run GR00T-H-N1.7 "inside" Cosmos-H-Dreams instead of on recorded video:

1. GR00T-H looks at a **real** endoscope frame and predicts an action chunk.
2. Cosmos-H-Dreams renders what those actions would do.
3. The **last generated frame** goes back to GR00T-H, which predicts the next chunk.
4. Repeat for a whole clip (601 frames = 20 s for episode 1441).

The open-loop sections (2–4 in the notebook) always give the policy the real video, so errors never compound. In a closed loop the policy sees the consequences of its own actions, so drift builds up. Using a world model to evaluate a policy is how NVIDIA screened GR00T-H checkpoints (Cosmos-Surg-dVRK), and section F of the notebook already sets it as homework.

## Is it valid? Yes, with three caveats

1. **It needs a different GR00T-H embodiment.** The notebook uses `jhu_imerse_dvrk`, which takes three cameras: `endoscope_left`, `wrist_left`, `wrist_right`. Cosmos only renders the endoscope view, so there are no wrist frames to feed back. Don't fake them (blank or real wrist frames would be out of distribution, or would leak the real future).
   - The downloaded checkpoint also has **`jhu_imerse_dvrk_mono`** (endoscope only). Checked Nov 10: it's in `embodiment_id.json` (id 15), `statistics.json` and `processor_config.json`, and the repo has `open_h/embodiments/jhu_imerse_dvrk/jhu_imerse_dvrk_mono_config.py`. Same 16-D state/action, same 50-step horizon, same `REL_XYZ_ROT6D` pose actions.
   - **Measured Nov 10** (step 1 below): mono is *better* open loop than the 3-camera embodiment: xyz MAE 0.63 mm vs 1.37 mm on 1441, 0.62 vs 1.33 mm on 1447 (hold-pose baseline 1.16 / 1.33 mm). The model card doesn't say what mono was trained on, so we can't rule out that it saw these episodes; say so when presenting it.
   - For a fair comparison, show mono's **open-loop** score next to its closed-loop result. Don't compare it with the 3-camera numbers in Section 3.
2. **Some of the drift is the world model's, not the policy's.** Cosmos frames are 288×512 and softer than the real 540×960 video, and each iteration restarts Cosmos from a single frame, so it loses its own history. Keep the notebook's existing split: a **recorded-actions** rollout run the same way (re-anchored every block) shows the world-model-only drift, and the GR00T-H closed loop adds policy error on top.
3. **There's no success metric.** L1/SSIM against the real video only mean something for the first few seconds; after that the closed loop is allowed to diverge from what the surgeon did. Present it qualitatively (video side by side), plus L1/SSIM over time, plus the action trajectory against the recorded one.

## The loop, with exact numbers

Constants from `server/cosmos_actions.py`: actions at 30 Hz, Cosmos at 10 Hz (stride 3), 12 frames per block, AR step 0 uses 8 actions and later steps 12. `n_raw(1) = 34`, `ar_steps(1) = 2`; `n_raw(2) = 70`, which is more than GR00T-H's 50-step chunk, so **one block per iteration**.

```
img_0   = real endoscope frame at f0 (resized to 512x288, same as Cosmos input)
state_0 = observation.state at f0 (16-D, xyz m + quat xyzw + jaw rad, per arm)
for k in 0..K-1:                                    # K = ceil((T - f0) / 36), ~17 for a full clip from 0
    chunk = GR00T-H mono(img_k, state_k, instruction)       # (50, 16) absolute
    a16   = chunk[:34]                                      # rows 0..33 drive generated frames 1..12
    video = Cosmos(img_k, a16, ref_state=state_k, blocks=1) # context + 12 frames
    img_{k+1}   = video[-1]                                 # generated frame 12  (= raw time f0 + 36(k+1))
    state_{k+1} = chunk[36]                                 # commanded pose at that time (see below)
    keep video[1:], chunk[:36]
```

- **State has to come from the actions.** For `REL_XYZ_ROT6D`, GR00T-H predicts poses *relative to the state* and the policy adds the state back to return absolute poses, so the state is used even though the state embedding is dropped. In a generated scene there is no measured state, so use the commanded pose ("assume the robot tracks perfectly"). `chunk[36]` matches generated frame 12 in time; `chunk[33]` (the last row actually sent to Cosmos) is the alternative. Try both on one clip.
- **Cosmos reference state** for each block is `state_k`, the same convention as today (`ref_state_npy` = the state at the context frame).
- **Seeding:** seed GR00T-H per `(episode, f0, k, instruction)`, like `groot_chunk` does, so precompute is reproducible.
- **Image into GR00T-H:** pass the 288×512 generated frame. GR00T-H's processor resizes anyway. Give it the real frame at 288×512 too, on iteration 0 and in the mono open-loop baseline, so resolution doesn't differ between the conditions.

**Cost** (from Nov 4/10 timings, L40S): Cosmos ~0.8 s per AR step, so ~1.6 s plus overhead per 1-block iteration; GR00T-H ~0.2–0.5 s per chunk. A full clip is about **17 iterations ≈ 30–60 s of GPU time**, all on the single Cosmos queue. With 30–50 people running it live, people would wait for many minutes, so it has to be **precomputed** for every clip, with a live run only as an Explore cell that shows the queue position.

## Server changes

All additive: the current endpoints and cache stay as they are.

**`server/gateway.py`**
1. Load a second policy: `POLICY_MONO = Gr00tPolicy(model_path=GROOT_MODEL, embodiment_tag="jhu_imerse_dvrk_mono", device="cuda:0", strict=False)`. Same weights, about 7 GB more GPU memory (22 GB used at idle, 25 GB peak in the Nov 10 load test, 46 GB total). Check that `Gr00tPolicy` doesn't need a second copy of the weights for a different tag. If it can switch tags on one instance, do that instead. Share `GROOT_LOCK` between the two.
2. Build the mono observation from an image and a state instead of `extract_step_data`: `{"video": {"endoscope_left": (1,1,H,W,3) uint8}, "state": {psm1_pose: (1,1,7), psm1_gripper: (1,1,1), psm2_pose…, psm2_gripper…}, "language": {"annotation.human.task_description": [[instr]]}}`. Check the exact key layout against `parse_observation_gr00t` and the mono `modality.json`; the 3-camera path goes through `open_loop_eval.py` helpers, which assume a dataset step.
3. New endpoint `POST /v1/closed_loop {episode, frame, iterations, instruction}` → async job, like `/v1/cosmos/dream`. It runs the loop above, writes `{job}.mp4` (all generated frames, 10 fps) and `{job}_actions.npy` (the stitched (36·K, 16) commanded actions), and caches by request + `CACHE_VERSION`.
4. Optionally `POST /v1/groot/act_mono {episode, frame}` for the mono open-loop baseline (same 50-step protocol as `/v1/groot/open_loop`, but from the 288×512 endoscope frame only).

**`server/cosmos_worker.py`**
5. Accept an **input image** instead of `(episode, frame)`: add an optional `input_png` to `GenReq`, and skip `_extract_frame` when it's given. The rest (`to_cosmos`, `RES.run`, `_trim`) is unchanged with `blocks=1`.
6. Return the last frame directly (it's already in `{stem}.npy`), so the gateway doesn't have to decode the mp4.

**Keeping Cosmos's history (optional, better continuity, more work).** The offline runner takes all actions up front, which is why the loop above restarts Cosmos every block. The repo's interactive/streaming path (`cosmosHDreams.webrtc.server*`) keeps the model's state between action chunks. Driving it from the gateway would remove the single-frame restart and caveat 2's extra drift. Look at it only if the simple version drifts too fast to be useful.

**`server/precompute.py` / `start_services.sh`**
7. Precompute closed-loop results for all 11 clips (from each clip's `DEMO_FRAMES` start, or from frame 0 for a full clip), plus the re-anchored recorded-actions rollout. Add them to `seg4_bundle_<ep>.zip` (keep each bundle around 5 MB: 10 fps, 288×512) and re-upload with `server/upload_bundles.py`. Expect around 15 extra minutes in `start_services.sh`.

## Notebook (v3, or a new section in v2)

- **Main thread (precomputed, ~2 min):** a three-panel video: real | Cosmos + recorded actions (re-anchored the same way) | GR00T-H mono closed loop. Under it, L1/SSIM over time, and the commanded trajectory against the recorded one (mm / rad). Works in `cache` mode from the bundle.
- **Explore (live):** "Run your own closed loop" with `ITERATIONS` (default 5 ≈ 6 s of video) and the instruction dropdown from Explore A. Print the queue position, and say plainly that it needs the live server.
- Replace homework F with a pointer to the new section.
- Timing: Segment 4 is 15 min and the main thread is already ~13 min. Either take time from Section 4 or keep the closed loop in Explore only.

## Build order

1. **Feasibility, about an hour, off the live server.** A standalone script loads `jhu_imerse_dvrk_mono` next to the running gateway (24 GB free) and measures mono open-loop on 1441 (xyz/quat/jaw MAE against the 3-camera numbers and the hold-pose baseline). Then run 5 iterations by hand through the worker on 1441@345 and look at the video. **Stop here if mono is much worse than hold-pose or the video falls apart within 2 iterations.**
2. Worker `input_png` + gateway `/v1/closed_loop`, behind a flag, tested on localhost.
3. Precompute + bundles + re-upload; one restart of `start_services.sh` (~6 min downtime; the cache on disk survives).
4. Notebook section; run end to end in `api` and `cache` modes; one real-Colab pass.
5. Load test with ~10 simultaneous live closed loops; document the expected wait.

## Sources

- Mono config: `~/work/GR00T-H/open_h/embodiments/jhu_imerse_dvrk/jhu_imerse_dvrk_mono_config.py` and `README.md` ("Mono Configuration").
- Checkpoint: `nvidia/GR00T-H-N1.7` → `embodiment_id.json`, `statistics.json`, `processor_config.json`.
- Cosmos block math and action format: `server/cosmos_actions.py`, `HANDOFF.md` §4.
- Policy screening with a world model (Cosmos-Surg-dVRK): as stated in notebook section F; not re-checked against the Open-H report (https://arxiv.org/abs/2604.21017) for these notes.

## Implementation status (Nov 10, branch `seg4-closed-loop`)

**Feasibility (build step 1), done.** Endoscope-only GR00T-H next to the live server, official 50-step open-loop protocol:

| Episode | mono, real frames | mono, Cosmos-like 288×512 frames | 3-camera (live server) | hold pose |
|---|---|---|---|---|
| 1441 (throw) | 0.63 mm / 0.021 / 0.058 rad | 0.67 mm / 0.023 / 0.061 rad | 1.37 mm / 0.021 / 0.148 rad | 1.16 mm / 0.017 / 0.061 rad |
| 1447 (knot) | 0.62 mm / 0.025 / 0.071 rad | 0.64 mm / 0.027 / 0.064 rad | 1.33 mm / 0.030 / 0.297 rad | 1.33 mm / 0.020 / 0.046 rad |

(xyz MAE / quaternion MAE / jaw MAE.) Shifting the input pose 1 cm moves the predicted path by exactly 10.0 mm, so the pose input is used as the reference, as expected for `REL_XYZ_ROT6D`. 145 ms per chunk, 6 GB GPU.

**Built (build step 2):**
- `cosmos_worker.py`: optional `input_png` (must be in `SHARED/inputs`); without it, behaviour is unchanged.
- `gateway.py`:
  - Loads the mono policy as a second instance (`CLOSED_LOOP=1`, the default; `0` skips it). Shares `GROOT_LOCK`.
  - `POST /v1/groot/open_loop_mono` is the mono open-loop baseline, on real frames at Cosmos resolution.
  - `POST /v1/closed_loop {episode, frame=-1, iterations=6, policy="groot"|"recorded", instruction}` is an async job polled with `/v1/jobs/{id}`. It runs on its own thread, so its worker calls interleave with dream jobs at the worker. Outputs `{job}.mp4` (context + 12·iterations frames, 10 fps; generated frame j ↔ raw frame `frame + 3j`) and `{job}_actions.npy` (36·iterations × 16 commanded actions, 30 Hz). `iterations` is capped so the rollout stays inside the real clip (1441 from 345: 7).
  - The existing request models and cache keys are untouched.
  - New optional `HOST` env.
- `precompute.py`: when `/health` reports `closed_loop`, adds `open_loop_mono` and `closed_loop.{groot,recorded}` (video + actions) to the bundle. Adds about 0.1 MB per clip.

**Tests run** (test gateway on :8010 with its own `SHARED_DIR`, CPU stand-in worker that runs the real action conversion and shifts the frame): 29/29 pass.
- `/v1/groot/act` and `/v1/groot/open_loop` are identical to the live server, byte for byte.
- `open_loop_mono` gives 0.71 mm (1441) and 0.75 mm (1447) through the server.
- Closed-loop chaining: each block starts from the previous block's last frame; the reference pose is the real state for `recorded` and continues the commanded path for `groot` (max 1.3 mm from the last row sent); frame counts, action shapes and unit quaternions are right; temp files are cleaned up.
- Caching, the iteration cap, two concurrent jobs, and 400/401 errors behave as intended.
- `precompute.py` writes the closed-loop files into the zip.

**Not done yet:**
1. Run with the **real** Cosmos worker (needs the live worker swapped to this branch's `cosmos_worker.py`; backward compatible, about 2.5 min without new Cosmos videos while it warms up) and look at the videos. Then rerun the tests with `--real-cosmos`, and time one full 1441 run.
2. Deploy: merge, `start_services.sh` (gateway +6 GB GPU, ~28 GB idle), re-precompute, and decide whether to re-upload the bundles to HF.
3. Notebook section (v3 or v2 Explore), load test with ~10 concurrent closed loops.
