"""Tests for the closed-loop gateway changes. Needs: live gateway (:8000), test gateway (:8010)
with SHARED_DIR=/data/shared_dev, and a worker on :8012 (fake_worker.py, or the real one).
Usage: python test_closed_loop.py <seg4/server dir> [--real-cosmos]"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests, cv2

sys.path.insert(0, sys.argv[1])
import metrics  # noqa: E402

REAL = "--real-cosmos" in sys.argv
LIVE, DEV = "http://localhost:8000", "http://localhost:8010"
LIVE_KEY = open("/data/seg4.env").read().split("=", 1)[1].strip()
H_LIVE, H_DEV = {"X-API-Key": LIVE_KEY}, {"X-API-Key": "devkey"}
SHARED = Path("/data/shared_dev")
LOG = SHARED / "fake_worker_log.jsonl"
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    if not ok:
        fails.append(name)


def gt_state(ep, f):
    p = f"/data/suturebot/data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
    return np.asarray(pd.read_parquet(p, columns=["observation.state"])["observation.state"].iloc[f], np.float32)


def gt_actions(ep):
    p = f"/data/suturebot/data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
    return np.stack(pd.read_parquet(p, columns=["action"])["action"].values).astype(np.float32)


def wait(job_id, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = requests.get(f"{DEV}/v1/jobs/{job_id}", headers=H_DEV).json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(1)
    return {"status": "timeout"}


def n_video_frames(url):
    p = SHARED / "dl.mp4"
    p.write_bytes(requests.get(DEV + url, headers=H_DEV).content)
    cap, n = cv2.VideoCapture(str(p)), 0
    while cap.read()[0]:
        n += 1
    return n


# ---- 1. existing endpoints unchanged (deterministic GR00T-H -> identical to the live server)
for ep, f in [(1441, 345), (1447, 0)]:
    a = np.array(requests.post(f"{LIVE}/v1/groot/act", json={"episode": ep, "frame": f}, headers=H_LIVE).json()["actions"])
    b = np.array(requests.post(f"{DEV}/v1/groot/act", json={"episode": ep, "frame": f}, headers=H_DEV).json()["actions"])
    check(f"act {ep}@{f} identical to live", np.abs(a - b).max() < 1e-5, f"max diff {np.abs(a - b).max():.2e}")
ol = requests.post(f"{DEV}/v1/groot/open_loop", json={"episode": 1441, "start": 0, "steps": 10000}, headers=H_DEV).json()
check("open_loop 1441 unchanged (MSE 0.0075)", abs(ol["metrics"]["mse"] - 0.0075209) < 1e-4, f"{ol['metrics']}")
check("/health reports closed_loop", requests.get(f"{DEV}/health").json().get("closed_loop") is True)

# ---- 2. endoscope-only open-loop baseline
for ep in (1441, 1447):
    r = requests.post(f"{DEV}/v1/groot/open_loop_mono", json={"episode": ep, "start": 0, "steps": 10000}, headers=H_DEV, timeout=600).json()
    g = metrics.per_group_mae(np.array(r["gt"]), np.array(r["pred"]))
    check(f"open_loop_mono {ep} xyz < 1 mm (feasibility: ~0.65)", g["mae_xyz_mm"] < 1.0, f"{g}")
r2 = requests.post(f"{DEV}/v1/groot/open_loop_mono", json={"episode": 1441, "start": 0, "steps": 10000}, headers=H_DEV).json()
check("open_loop_mono cached on repeat", r2["cached"] is True)

# ---- 3. closed loop, recorded actions
LOG.unlink(missing_ok=True)
gt = gt_actions(1441)
r = requests.post(f"{DEV}/v1/closed_loop", json={"episode": 1441, "iterations": 3, "policy": "recorded"}, headers=H_DEV).json()
check("closed_loop recorded queued, frame -1 -> demo frame 345", r["frame"] == 345 and r["iterations"] == 3, f"{r}")
j = wait(r["job_id"])
check("closed_loop recorded done", j["status"] == "done", json.dumps(j)[:300])
if j["status"] == "done":
    nf = n_video_frames(j["video"])
    check("recorded video has 1 + 12*3 frames", nf == 37, f"{nf}")
    acts = np.load(SHARED / "outputs" / f"{r['job_id']}_actions.npy")
    check("recorded actions == GT rows 345..452", acts.shape == (108, 16) and np.abs(acts - gt[345:453]).max() < 1e-6, f"{acts.shape}")
    if not REAL:
        log = [json.loads(l) for l in LOG.read_text().splitlines()]
        check("worker got 3 one-block requests with 34 rows", len(log) == 3 and all(x["blocks"] == 1 and x["a16_shape"] == [34, 16] for x in log))
        check("each iteration starts from the previous one's last frame",
              all(log[k + 1]["ctx_hash"] == log[k]["last_hash"] for k in range(2)))
        check("recorded reference pose = real state at 345, 381, 417",
              all(np.abs(np.array(log[k]["ref"]) - gt_state(1441, 345 + 36 * k)).max() < 1e-5 for k in range(3)))
        check("Cosmos actions shape (20, 44) per block", all(x["cosmos_shape"] == [20, 44] for x in log), f"{log[0]['cosmos_shape']}")
    left = [p.name for p in (SHARED / "inputs").glob(f"{r['job_id']}_*")] + [p.name for p in (SHARED / "outputs").glob(f"{r['job_id']}_0*")]
    check("per-iteration temp files cleaned up", not left, f"{left[:4]}")
    c = requests.post(f"{DEV}/v1/closed_loop", json={"episode": 1441, "iterations": 3, "policy": "recorded"}, headers=H_DEV).json()
    check("repeat request is a cache hit with the same job", c.get("cached") is True and c["job_id"] == r["job_id"], f"{c}")
    j2 = requests.get(f"{DEV}/v1/jobs/{r['job_id']}", headers=H_DEV).json()
    check("job reports actions URL", "actions" in j2, f"{j2}")

# ---- 4. closed loop, GR00T-H; asking for 20 iterations is capped by the clip ((600-345)//36 = 7)
LOG.unlink(missing_ok=True)
t0 = time.time()
r = requests.post(f"{DEV}/v1/closed_loop", json={"episode": 1441, "iterations": 20, "policy": "groot"}, headers=H_DEV).json()
check("groot iterations capped at 7", r["iterations"] == 7, f"{r}")
j = wait(r["job_id"])
dt = time.time() - t0
check("closed_loop groot done", j["status"] == "done", json.dumps(j)[:300])
if j["status"] == "done":
    check("groot video has 1 + 12*7 frames", n_video_frames(j["video"]) == 85)
    acts = np.load(SHARED / "outputs" / f"{r['job_id']}_actions.npy")
    check("groot actions (252, 16), finite, unit quats", acts.shape == (252, 16) and np.isfinite(acts).all()
          and np.allclose(np.linalg.norm(acts[:, 3:7], axis=1), 1, atol=0.05), f"{acts.shape}")
    g = metrics.per_group_mae(gt[345:345 + 252], metrics.align_quat_signs(gt[345:345 + 252], acts))
    print(f"     info: commanded vs recorded over 8.4 s: {g} | wall time {dt:.1f} s", flush=True)
    if not REAL:
        log = [json.loads(l) for l in LOG.read_text().splitlines()]
        check("groot: 7 worker calls, chained frames", len(log) == 7 and all(log[k + 1]["ctx_hash"] == log[k]["last_hash"] for k in range(6)))
        check("groot: first reference pose = real state at 345", np.abs(np.array(log[0]["ref"]) - gt_state(1441, 345)).max() < 1e-5)
        jump = max(1000 * np.abs(np.array(log[k + 1]["ref"])[[0, 1, 2, 8, 9, 10]] - np.array(log[k]["a16_last"])[[0, 1, 2, 8, 9, 10]]).max() for k in range(6))
        check("groot: next pose continues the commanded path (< 5 mm from last row sent)", jump < 5, f"max {jump:.2f} mm")

# ---- 5. two closed loops at once (queue) + errors
ids = [requests.post(f"{DEV}/v1/closed_loop", json={"episode": ep, "iterations": 2, "policy": "groot"}, headers=H_DEV).json()["job_id"] for ep in (1445, 1449)]
check("two queued closed loops both finish", all(wait(i)["status"] == "done" for i in ids))
check("bad policy -> 400", requests.post(f"{DEV}/v1/closed_loop", json={"episode": 1441, "policy": "x"}, headers=H_DEV).status_code == 400)
check("start too close to the end -> 400", requests.post(f"{DEV}/v1/closed_loop", json={"episode": 1441, "frame": 590}, headers=H_DEV).status_code == 400)
check("no API key -> 401", requests.post(f"{DEV}/v1/closed_loop", json={"episode": 1441}).status_code == 401)

print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {fails}")
