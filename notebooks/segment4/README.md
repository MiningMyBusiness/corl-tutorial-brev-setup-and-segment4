# Segment 4: GR00T-H-N1.7 + Cosmos-H-Dreams

```
segment4/
  segment4_vla_world_model.ipynb   participant notebook (Colab, CPU is enough)
  segment4_vla_world_model_executed.ipynb   same, with saved outputs from a full run (reference)
  server/gateway.py                public API: GR00T-H in-process, queues Cosmos jobs
  server/cosmos_worker.py          runs inside the Cosmos-H-Dreams container (localhost:8001)
  server/cosmos_actions.py         16-D absolute -> Cosmos hybrid-relative actions (teacher repo's training format)
  server/metrics.py                open-loop MSE/MAE (GR00T-H official) + L1/SSIM (Open-H report)
  server/rank_episodes.py          picks a test-split clip and makes a contact sheet
  server/pick_start_frames.py      dream start frame per clip (prints DEMO_FRAMES)
  server/precompute.py             warms the cache and exports the offline bundle (run by start_services.sh)
  server/load_test.py              N simultaneous users against the gateway (local or secure link)
  server/validate_cosmos.py        dreams recorded + perturbed actions, scores vs real video (format check)
  brev/setup.sh, start_services.sh one-time setup / start everything
  REBUILD.md                       what to change after rebuilding the Launchable
  ACT_INTERFACE.md                 what Segment 3 (ACT) must output to join the comparison
```

## Brev Launchable settings
- **Compute:** 1 GPU with at least 40 GB (L40S 48 GB or A100 80 GB). Both models together need about 25 GB, so there's headroom. **Driver R580+ is required** (Cosmos-H-Dreams needs CUDA 13). `setup.sh` stops if the driver is older.
- **Disk:** at least 200 GB (two repos, two Docker/uv environments, checkpoints, the SutureBot test split).
- **Software:** VM Mode with Jupyter. Setup script = `brev/setup.sh`. Environment variable: `HF_TOKEN`.
- **Ports:** expose **8000** (a Brev secure link works and needs no Brev login; tested Nov 4). That's the gateway, and the `X-API-Key` header is its only auth. Don't expose 8001 or 5555.

## Run order (from a Jupyter terminal on the box)
```bash
git clone <this repo> ~/seg4            # or upload the segment4 folder to ~/seg4
HF_TOKEN=hf_xxx API_KEY=<slide key> bash ~/seg4/brev/setup.sh
bash ~/seg4/brev/start_services.sh     # defaults: all held-out clips 1441-1451, start frames in DEMO_FRAMES
#   -> starts GR00T gateway + resident Cosmos worker, warms the cache, writes /data/shared/seg4_bundle_<ep>.zip
cd ~/work/GR00T-H && uv run --with scikit-image python ~/seg4/server/validate_cosmos.py --episode 1441 --frame 345
#   -> must print PASS (recorded actions clearly beat a frozen frame)
uv run --with requests python ~/seg4/server/load_test.py --url https://<secure-link> --key $(grep API_KEY /data/seg4.env | cut -d= -f2)
```

## Lines marked `VERIFY`
1. `LeRobotEpisodeLoader` constructor arguments, and the `StepData` attribute names (`gateway.py`). Confirmed working: `/v1/groot/act` returns sensible chunks.
2. Cosmos action format: **resolved Nov 4** (see `cosmos_actions.py` docstring and `HANDOFF.md` §4). Validated by `validate_cosmos.py`.
3. The generated mp4 includes the context frame: **yes**. The worker trims it to context + 12·blocks frames.
4. Cosmos time per 6-block job: ~5.5 s with the resident worker (default), after ~80 s warm-up at start.
