"""Segment 4 public API: GR00T-H-N1.7 (in-process) + Cosmos-H-Dreams (via local worker).

Run inside the GR00T-H uv environment on the Brev box (see brev/start_services.sh):
    uv run --with fastapi --with "uvicorn[standard]" --with scipy --with scikit-image \
        python server/gateway.py

Public endpoints (all except /health need header  X-API-Key: <API_KEY>):
    GET  /health
    GET  /v1/episodes                      -> episodes hosted on the server
    POST /v1/groot/act                     -> one 50-step action chunk
    POST /v1/groot/open_loop               -> official-style open-loop eval over a window
    POST /v1/cosmos/dream                  -> queue a world-model rollout, returns job_id
    GET  /v1/jobs/{job_id}                 -> job status (+ video URL when done)
    GET  /v1/files/{name}                  -> mp4 / npy outputs

Design notes
  * Every response is cached on disk, keyed by its request. The tutorial's main-thread
    requests are pre-computed with precompute.py, so 50 people pressing "run" at once
    hit the cache, not the GPU.
  * Cosmos jobs go through a single queue (one GPU); GR00T calls are serialized by a lock.
  * Actions are exchanged in the dataset's own 16-D absolute space at 30 Hz
    (what GR00T-H returns and what ACT predicts), so ACT outputs can be dreamed too.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import torch
import requests
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import cosmos_actions
import metrics

# ------------------------------------------------------------------ config ---
API_KEY = os.environ.get("API_KEY", "change-me")
PORT = int(os.environ.get("PORT", "8000"))
GROOT_MODEL = os.environ.get("GROOT_MODEL", "nvidia/GR00T-H-N1.7")
EMBODIMENT = os.environ.get("EMBODIMENT_TAG", "jhu_imerse_dvrk")
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data/suturebot"))  # LeRobot root of SutureBot subset
SHARED = Path(os.environ.get("SHARED_DIR", "/data/shared"))      # mounted into cosmos container
COSMOS_WORKER = os.environ.get("COSMOS_WORKER_URL", "http://127.0.0.1:8001")
EPISODES = [int(e) for e in os.environ.get("EPISODES", "").split(",") if e.strip()]
# demo start frame per episode, e.g. "1441:345,1443:270" (precompute.py warms exactly these)
DEMO_FRAMES = {int(k): int(v) for k, v in (x.split(":") for x in os.environ.get("DEMO_FRAMES", "").split(",") if ":" in x)}
ACTION_HORIZON = 50
FPS = 30

CACHE = SHARED / "cache"
OUT = SHARED / "outputs"
for d in (CACHE, OUT, SHARED / "inputs"):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------- GR00T-H ---
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402

try:  # reuse the official helpers so preprocessing matches NVIDIA's open-loop eval
    from gr00t.eval.open_loop_eval import parse_action_gr00t, parse_observation_gr00t
except Exception:  # pragma: no cover
    parse_action_gr00t = parse_observation_gr00t = None

print(f"[gateway] loading {GROOT_MODEL} ({EMBODIMENT}) ...", flush=True)
POLICY = Gr00tPolicy(model_path=GROOT_MODEL, embodiment_tag=EMBODIMENT, device="cuda:0", strict=False)
MODALITY = POLICY.get_modality_config()
TAG = EmbodimentTag(EMBODIMENT)
LOADER = LeRobotEpisodeLoader(  # VERIFY ctor kwargs against gr00t/data/dataset/lerobot_episode_loader.py
    dataset_path=str(DATA_ROOT), modality_configs=MODALITY, video_backend="torchcodec"
)
ACTION_KEYS = list(MODALITY["action"].modality_keys)
GROOT_LOCK = threading.Lock()
_TRAJ_CACHE: dict[int, object] = {}


def _traj(ep: int):
    if ep not in _TRAJ_CACHE:
        _TRAJ_CACHE[ep] = LOADER[ep]  # VERIFY: index == episode_index for the downloaded subset
    return _TRAJ_CACHE[ep]


def _gt_actions(ep: int) -> np.ndarray:
    traj = _traj(ep)
    return np.concatenate([np.stack(traj[f"action.{k}"].values) for k in ACTION_KEYS], axis=1)


def _gt_state(ep: int, frame: int) -> np.ndarray:
    """Observed 16-D state at `frame` (Cosmos' reference pose for hybrid-relative actions)."""
    import pandas as pd
    p = DATA_ROOT / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
    return np.asarray(pd.read_parquet(p, columns=["observation.state"])["observation.state"].iloc[frame],
                      np.float32)


def _observation(ep: int, step: int, instruction: str | None):
    """Build a GR00T observation exactly like gr00t/eval/open_loop_eval.py does."""
    # allow_padding: the last chunk's future GT actions run past the episode end; the policy
    # only reads video/state at delta 0, so clamping them does not change its input.
    dp = extract_step_data(_traj(ep), step, MODALITY, TAG, allow_padding=True)
    obs = {}
    for k, v in dp.states.items():          # VERIFY attribute names on StepData
        obs[f"state.{k}"] = v
    for k, v in dp.images.items():
        obs[f"video.{k}"] = np.array(v)
    lang_key = MODALITY["language"].modality_keys[0]
    obs[lang_key] = instruction or dp.text
    return parse_observation_gr00t(obs, MODALITY), obs[lang_key]


def groot_chunk(ep: int, step: int, instruction: str | None) -> tuple[np.ndarray, str, float]:
    obs, used_instruction = _observation(ep, step, instruction)
    t0 = time.time()
    with GROOT_LOCK:
        # GR00T-H samples noise (flow matching): seed per request so a cache rebuild
        # (e.g. on a new instance) reproduces the same chunks and numbers
        torch.manual_seed(zlib.crc32(f"{ep}:{step}:{used_instruction}".encode()))
        action, _ = POLICY.get_action(obs)
    dt = (time.time() - t0) * 1000
    act = parse_action_gr00t(action)
    chunk = np.concatenate([act[f"action.{k}"] for k in ACTION_KEYS], axis=1)  # (50, 16)
    return chunk.astype(np.float32), used_instruction, dt


# ------------------------------------------------------------------ cache ---
CACHE_VERSION = 2  # bump when outputs change for the same request (v2: seeded GR00T, aligned quats)


def _key(kind: str, payload: dict) -> str:
    payload = {**payload, "_v": CACHE_VERSION}
    return kind + "_" + hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def cache_get(k):
    p = CACHE / f"{k}.json"
    return json.loads(p.read_text()) if p.exists() else None


def cache_put(k, v):
    (CACHE / f"{k}.json").write_text(json.dumps(v))


# ---------------------------------------------------------------- cosmos ---
JOBS: dict[str, dict] = {}
COSMOS_Q: queue.Queue = queue.Queue()


def _cosmos_loop():
    while True:
        job_id = COSMOS_Q.get()
        job = JOBS[job_id]
        job["status"], job["started"] = "running", time.time()
        try:
            r = requests.post(f"{COSMOS_WORKER}/generate", json=job["worker_req"], timeout=1800)
            r.raise_for_status()
            job.update(status="done", video=f"/v1/files/{job_id}.mp4", finished=time.time())
            cache_put(job["cache_key"], {"job_id": job_id, "video": job["video"]})
        except Exception as e:  # noqa: BLE001
            job.update(status="error", error=str(e))
        finally:
            COSMOS_Q.task_done()


threading.Thread(target=_cosmos_loop, daemon=True).start()

# ------------------------------------------------------------------- API ---
app = FastAPI(title="CoRL 2026 Surgical Autonomy - Segment 4 API")


def auth(x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(401, "missing or wrong X-API-Key")


class ActReq(BaseModel):
    episode: int
    frame: int = 0
    instruction: str | None = Field(None, description="override the dataset task, e.g. 'knot tying'")


class OpenLoopReq(BaseModel):
    episode: int
    start: int = 0
    steps: int = 300
    instruction: str | None = None


class DreamReq(BaseModel):
    episode: int
    frame: int = 0
    action_source: str = Field("gt", description="'gt' | 'groot' | 'custom'")
    actions: list[list[float]] | None = Field(None, description="(T,16) absolute @30Hz if custom")
    blocks: int = Field(6, ge=1, le=12)
    instruction: str | None = None


@app.get("/health")
def health():
    return {"ok": True, "model": GROOT_MODEL, "embodiment": EMBODIMENT,
            "cosmos_queue": COSMOS_Q.qsize(), "episodes": EPISODES,
            "start_frames": {str(e): DEMO_FRAMES.get(e, 0) for e in EPISODES}}


@app.get("/v1/episodes", dependencies=[Depends(auth)])
def episodes():
    meta = json.loads((SHARED / "episodes.json").read_text()) if (SHARED / "episodes.json").exists() else {}
    return {"episodes": EPISODES, "meta": meta, "fps": FPS, "action_names": metrics.ACTION_NAMES}


@app.post("/v1/groot/act", dependencies=[Depends(auth)])
def act(req: ActReq):
    k = _key("act", req.model_dump())
    if (c := cache_get(k)):
        return {**c, "cached": True}
    chunk, instr, ms = groot_chunk(req.episode, req.frame, req.instruction)
    res = {"actions": chunk.tolist(), "instruction": instr, "horizon": ACTION_HORIZON,
           "fps": FPS, "action_names": metrics.ACTION_NAMES, "latency_ms": round(ms, 1)}
    cache_put(k, res)
    return {**res, "cached": False}


def _open_loop(ep, start, steps, instruction):
    gt = _gt_actions(ep)
    steps = min(steps, len(gt) - start)
    chunks = [groot_chunk(ep, t, instruction)[0] for t in range(start, start + steps, ACTION_HORIZON)]
    pred = metrics.stitch_open_loop(chunks, ACTION_HORIZON, steps)
    gt = gt[start : start + steps]
    return gt, metrics.align_quat_signs(gt, pred)  # q == -q; keeps MSE/MAE and plots honest


@app.post("/v1/groot/open_loop", dependencies=[Depends(auth)])
def open_loop(req: OpenLoopReq):
    k = _key("openloop", req.model_dump())
    if (c := cache_get(k)):
        return {**c, "cached": True}
    gt, pred = _open_loop(req.episode, req.start, req.steps, req.instruction)
    res = {"gt": gt.tolist(), "pred": pred.tolist(), "action_horizon": ACTION_HORIZON,
           "metrics": metrics.open_loop_action_metrics(gt, pred), "action_names": metrics.ACTION_NAMES}
    cache_put(k, res)
    return {**res, "cached": False}


@app.post("/v1/cosmos/dream", dependencies=[Depends(auth)])
def dream(req: DreamReq, request: Request):
    payload = req.model_dump()
    k = _key("dream", {**payload, "fmt": cosmos_actions.FORMAT_VERSION})
    if (c := cache_get(k)):
        return {"job_id": c["job_id"], "status": "done", "video": c["video"], "cached": True}

    n_raw = cosmos_actions.n_raw(req.blocks)  # 30 Hz rows driving the 12*blocks kept frames
    if req.action_source == "gt":
        a16 = _gt_actions(req.episode)[req.frame : req.frame + n_raw]
    elif req.action_source == "groot":  # receding-horizon GR00T chunks on real frames (open loop)
        _, a16 = _open_loop(req.episode, req.frame, n_raw, req.instruction)
    elif req.action_source == "custom" and req.actions:
        a16 = np.asarray(req.actions, np.float32)
    else:
        raise HTTPException(400, "action_source must be gt|groot|custom (custom needs actions)")
    if len(a16) < n_raw:
        raise HTTPException(400, f"need {n_raw} action steps for {req.blocks} blocks, got {len(a16)}")

    job_id = uuid.uuid4().hex[:12]
    act_path = SHARED / "inputs" / f"{job_id}_abs16.npy"
    np.save(act_path, np.asarray(a16[:n_raw], np.float32))  # worker converts (cosmos_actions.py)
    ref_path = SHARED / "inputs" / f"{job_id}_ref16.npy"
    np.save(ref_path, _gt_state(req.episode, req.frame))
    JOBS[job_id] = {
        "status": "queued", "created": time.time(), "cache_key": k,
        "worker_req": {"job_id": job_id, "episode": req.episode, "frame": req.frame,
                       "actions_npy": str(act_path), "ref_state_npy": str(ref_path),
                       "blocks": req.blocks},
    }
    COSMOS_Q.put(job_id)
    return {"job_id": job_id, "status": "queued", "position": COSMOS_Q.qsize(), "cached": False}


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(auth)])
def job(job_id: str):
    if job_id not in JOBS:
        if (OUT / f"{job_id}.mp4").exists():  # served from a warmed cache after a restart
            return {"status": "done", "video": f"/v1/files/{job_id}.mp4"}
        raise HTTPException(404, "unknown job")
    j = {k: v for k, v in JOBS[job_id].items() if k not in ("worker_req", "cache_key")}
    return j


@app.get("/v1/files/{name}", dependencies=[Depends(auth)])
def files(name: str):
    p = (OUT / name).resolve()
    if OUT.resolve() not in p.parents or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
