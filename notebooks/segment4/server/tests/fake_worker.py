"""CPU stand-in for cosmos_worker.py, for testing the gateway's closed loop without a GPU.
Same request format. Runs the real action conversion (cosmos_actions.to_cosmos) so format
errors surface, then 'generates' 12 frames by shifting the context frame 2 px per frame.
Logs every request (plus hashes of the context image and the last generated frame) to log.jsonl.
Usage: python fake_worker.py <seg4/server dir> <shared dir> <cosmos stats json> <port>"""
import hashlib, json, subprocess, sys
from pathlib import Path
import cv2, numpy as np, uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, sys.argv[1])
import cosmos_actions  # noqa: E402

SHARED, STATS, PORT = Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
LOG = SHARED / "fake_worker_log.jsonl"
app = FastAPI()


class GenReq(BaseModel):
    job_id: str
    episode: int
    frame: int
    actions_npy: str
    ref_state_npy: str
    blocks: int = 6
    input_png: str | None = None


def h(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


@app.get("/health")
def health():
    return {"ok": True, "fake": True}


@app.post("/generate")
def generate(req: GenReq):
    if not req.input_png:
        raise HTTPException(400, "fake worker only serves closed-loop requests (input_png)")
    png = Path(req.input_png).resolve()
    if (SHARED / "inputs").resolve() not in png.parents or not png.exists():
        raise HTTPException(400, "input_png must be an existing file in SHARED/inputs")
    ctx = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
    a16, ref = np.load(req.actions_npy), np.load(req.ref_state_npy)
    acts = cosmos_actions.to_cosmos(a16, ref, req.blocks, STATS)   # real conversion, real shapes
    np.save(SHARED / "inputs" / f"{req.job_id}_cosmos_actions.npy", acts)   # like the real worker
    n = cosmos_actions.n_frames(req.blocks)
    frames = np.stack([ctx] + [np.roll(ctx, 2 * (i + 1), axis=1) for i in range(n)])
    out = SHARED / "outputs" / req.job_id
    np.save(f"{out}.npy", frames)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "512x288",
                    "-r", "10", "-i", "-", "-pix_fmt", "yuv420p", f"{out}.mp4"], input=frames.tobytes(), check=True)
    rec = {"job_id": req.job_id, "frame": req.frame, "blocks": req.blocks, "ctx_shape": list(ctx.shape),
           "ctx_hash": h(ctx), "last_hash": h(frames[-1]), "a16_shape": list(a16.shape), "ref": ref.tolist(),
           "a16_first": a16[0].tolist(), "a16_last": a16[-1].tolist(), "cosmos_shape": list(acts.shape),
           "cosmos_absmax": float(np.abs(acts).max())}
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return {"ok": True, "video": f"{out}.mp4", "frames": len(frames)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
