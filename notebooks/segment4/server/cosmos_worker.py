"""Cosmos-H-Dreams worker. Runs INSIDE the cosmos-h-dreams Docker container
(see brev/start_services.sh), listening on 127.0.0.1:8001 only.

Default (COSMOS_RESIDENT=1): builds the repo's own offline runner once
(cosmosHDreams.runner.CosmoshRunner, the code behind `flashdreams-run`), warms it up
(model load + torch.compile + CUDA graph capture, ~80 s), then serves each job with
the runner's per-entry method: ~5.5 s per 6-block dream instead of ~31 s.
All GPU work happens on ONE thread, because CUDA streams/graphs are per-thread.
COSMOS_RESIDENT=0 falls back to one `flashdreams-run` subprocess per job.

Output contract (what the gateway / notebook rely on):
    <job_id>.mp4 / .npy = context frame + 12*blocks generated frames at 10 fps,
    generated frame i (1-based) <-> raw 30 Hz frame  frame + 3*i  of the real video.
The chunk3 recipe yields 8 frames on AR step 0 and 12 per later step, so we run
cosmos_actions.ar_steps(blocks) steps and trim the tail.
"""
from __future__ import annotations

import copy
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
import cosmos_actions  # noqa: E402

REPO = Path(os.environ.get("COSMOS_REPO", "/workspace/cosmos-h-dreams"))
SHARED = Path(os.environ.get("SHARED_DIR", "/data/shared"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data/suturebot"))
CKPT = os.environ.get("COSMOS_CKPT", "checkpoints/model_ema_bf16_jhutabletop_288x512_h73.pt")
CR1 = os.environ.get("COSMOS_CR1", "checkpoints/cr1_empty_string_text_embeddings.pt")
STATS = os.environ.get("COSMOS_STATS") or "assets/example_data/offline/stats_tabletop.json"
RECIPE = os.environ.get("COSMOS_RECIPE", "cosmosHDreams-chunk3-vae-vae")
RESIDENT = os.environ.get("COSMOS_RESIDENT", "1") == "1"
VIDEO_KEY = "observation.images.endoscope.left"

app = FastAPI()


class GenReq(BaseModel):
    job_id: str
    episode: int
    frame: int
    actions_npy: str      # (T,16) absolute @30 Hz, row 0 = context frame
    ref_state_npy: str    # (16,) observed state at the context frame
    blocks: int = 6
    input_png: str | None = None  # context frame to start from (closed loop: the last generated
                                  # frame); must live in SHARED/inputs. None = cut it from the video


def _episode_video(ep: int) -> Path:
    chunk = ep // 1000
    return DATA_ROOT / f"videos/chunk-{chunk:03d}/{VIDEO_KEY}/episode_{ep:06d}.mp4"


def _extract_frame(ep: int, frame: int, out_png: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(_episode_video(ep)),
         "-vf", f"select=eq(n\\,{frame}),scale=512:288", "-vframes", "1", str(out_png)],
        check=True,
    )


def _trim(out_stem: Path, keep: int):
    """Keep the first `keep` frames of the runner's .npy and rewrite the mp4 from it."""
    import mediapy  # in the repo's `runners` extra

    frames = np.load(f"{out_stem}.npy")
    if len(frames) > keep:
        frames = frames[:keep]
        np.save(f"{out_stem}.npy", frames)
        mediapy.write_video(f"{out_stem}.mp4", frames, fps=10)
    Path(f"{out_stem}_annotated.mp4").unlink(missing_ok=True)
    return len(frames)


# ------------------------------------------------------- resident pipeline ---
class Resident:
    """Owns the GPU: one thread builds the runner and executes every entry."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self.ready = threading.Event()
        self.error: str | None = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _build(self):
        os.chdir(REPO)  # checkpoint / asset paths are repo-relative
        from cosmosHDreams.config import COSMOSHDREAMS_RUNNERS
        from cosmosHDreams.utils import load_cr1_text_embeddings

        cfg = copy.deepcopy(COSMOSHDREAMS_RUNNERS[RECIPE])
        cfg.cr1_embeddings_path = Path(CR1)
        cfg.root_dir = Path(".")
        cfg.resolution = "288,512"
        cfg.output_dir = SHARED / "outputs"
        cfg.pipeline.diffusion_model.transformer.checkpoint_path = CKPT
        self.runner = cfg.setup()
        self.emb = load_cr1_text_embeddings(CR1)

    def _entry(self, png: Path, act_path: Path, out_mp4: Path, ar_steps: int):
        self.runner.config.total_blocks = ar_steps
        stats = self.runner._run_entry(  # the per-entry body of CosmoshRunner.run()
            {"input": str(png), "input_action": str(act_path), "output_video": str(out_mp4)},
            input_path=str(png), text_embeddings_cpu=self.emb)
        if stats is None:
            raise RuntimeError("runner produced no frames (too few actions?)")

    def _loop(self):
        try:
            t0 = time.time()
            self._build()
            # warm-up: compile + graph capture happen on the first entry (~80 s)
            warm = SHARED / "inputs" / "_warmup"
            acts = np.zeros((cosmos_actions.n_model_actions(6), 44), np.float32)
            np.save(f"{warm}_actions.npy", acts)
            self._entry(REPO / "assets/example_data/offline/episode_001866_frame0.png",
                        Path(f"{warm}_actions.npy"), SHARED / "outputs" / "_warmup.mp4",
                        cosmos_actions.ar_steps(6))
            print(f"[cosmos_worker] resident pipeline ready in {time.time() - t0:.0f}s", flush=True)
            self.ready.set()
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
            self.ready.set()
            raise
        while True:
            args, fut = self.q.get()
            try:
                fut.set_result(self._entry(*args))
            except Exception as e:  # noqa: BLE001
                fut.set_exception(e)

    def run(self, *args):
        fut: Future = Future()
        self.q.put((args, fut))
        return fut.result()


RES = Resident() if RESIDENT else None


def _run_cli(png: Path, act_path: Path, out_mp4: Path, ar_steps: int, log_path: Path):
    manifest = act_path.with_suffix(".json")
    manifest.write_text(json.dumps([{"input": str(png), "input_action": str(act_path),
                                     "output_video": str(out_mp4)}]))
    cmd = ["uv", "run", "flashdreams-run", RECIPE,
           "--input-json", str(manifest), "--cr1-embeddings-path", CR1,
           "--root-dir", ".", "--total-blocks", str(ar_steps), "--resolution", "288,512",
           "--pipeline.diffusion-model.transformer.checkpoint-path", CKPT]
    with log_path.open("w") as log:
        subprocess.run(cmd, cwd=REPO, check=True, stdout=log, stderr=subprocess.STDOUT)


@app.get("/health")
def health():
    if RES is not None and not RES.ready.is_set():
        raise HTTPException(503, "warming up (model load + compile, ~80 s)")
    if RES is not None and RES.error:
        raise HTTPException(500, f"resident pipeline failed: {RES.error}")
    return {"ok": True, "ckpt": CKPT, "recipe": RECIPE, "resident": RESIDENT,
            "action_format": f"hybrid-relative to context state, norm44 (v{cosmos_actions.FORMAT_VERSION})"}


@app.post("/generate")
def generate(req: GenReq):
    t0 = time.time()
    inp, out = SHARED / "inputs", SHARED / "outputs"
    if req.input_png:
        png = Path(req.input_png).resolve()
        if inp.resolve() not in png.parents or not png.exists():
            raise HTTPException(400, "input_png must be an existing file in SHARED/inputs")
    else:
        png = inp / f"{req.job_id}_frame.png"
        _extract_frame(req.episode, req.frame, png)

    acts = cosmos_actions.to_cosmos(np.load(req.actions_npy), np.load(req.ref_state_npy),
                                    req.blocks, str(REPO / STATS))
    act_path = inp / f"{req.job_id}_cosmos_actions.npy"
    np.save(act_path, acts.astype(np.float32))

    stem = out / req.job_id
    steps = cosmos_actions.ar_steps(req.blocks)
    if RES is not None:
        RES.ready.wait()
        if RES.error:
            raise HTTPException(500, f"resident pipeline failed: {RES.error}")
        RES.run(png, act_path, Path(f"{stem}.mp4"), steps)
    else:
        _run_cli(png, act_path, Path(f"{stem}.mp4"), steps, out / f"{req.job_id}.log")
    n = _trim(stem, 1 + cosmos_actions.n_frames(req.blocks))
    return {"ok": True, "video": f"{stem}.mp4", "frames": n, "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
