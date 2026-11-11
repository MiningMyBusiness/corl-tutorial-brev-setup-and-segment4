# Closed-loop tests

Run on the Brev box from `~/work/GR00T-H` (`uv run --with requests --with fastapi --with uvicorn --with scipy python ...`). See `../../CLOSED_LOOP.md`.

- `mono_feasibility.py <server dir> [episode]`: endoscope-only GR00T-H open-loop accuracy, Cosmos-resolution frames, pose sensitivity, latency. About 7 GB GPU; safe next to the live server.
- `fake_worker.py <server dir> <shared dir> <cosmos stats json> <port>`: CPU stand-in for `cosmos_worker.py` that runs the real action conversion and logs every request.
- `test_closed_loop.py <server dir> [--real-cosmos]`: expects the live gateway on :8000, a test gateway on :8010 (`API_KEY=devkey SHARED_DIR=/data/shared_dev HOST=127.0.0.1 COSMOS_WORKER_URL=http://127.0.0.1:8012`) and a worker on :8012 (`fake_worker.py`, or the real worker for `--real-cosmos`).
