# Rebuilding the Brev Launchable from scratch

A fresh instance starts with no data. Everything is rebuilt by two scripts. A few values change on each rebuild; this note lists them.

## Steps

```bash
git clone <repo> ~/repo && ln -s ~/repo/notebooks/segment4 ~/seg4     # scripts expect ~/seg4 (or set SEG4=)
HF_TOKEN=hf_... API_KEY=<key on the slides> bash ~/seg4/brev/setup.sh  # ~30-40 min: repos, checkpoints, Docker image, data
bash ~/seg4/brev/start_services.sh                                     # ~6 min: both servers + cache for all 11 clips
cd ~/work/GR00T-H && uv run --with scikit-image python ~/seg4/server/validate_cosmos.py --episode 1441 --frame 345   # must print PASS
```

Launchable settings (see README): 1 GPU with 40 GB or more, **driver R580+**, 200 GB or more of disk, VM Mode + Jupyter, port **8000** exposed (a secure link works).

## What changes, and where to update it

| Value | Changes on rebuild? | Update |
|---|---|---|
| **Secure-link URL** (now `https://corl-seg4-5xptyrp9u.gobrev.dev`) | **Yes**, a new one per instance | Notebook cell 2 `API_URL` default, slides / handout. Re-run the load test against it (below). |
| **API key** (`/data/seg4.env`) | Yes, random, **unless** you pass `API_KEY=` to `setup.sh` | Slides / handout. Never commit it. Passing the old key avoids the change. |
| `HF_TOKEN` | Only needed by `setup.sh` (model downloads) and `upload_bundles.py` | Pass as env var. Never write it to a file. |
| HF cache dataset `kdb6df/corl2026-seg4-cache` | No. It lives on HF | Re-upload only if models, start frames or server code change: `HF_TOKEN=... uv run --with huggingface_hub python ~/seg4/server/upload_bundles.py` |
| Notebook `CACHE_URL` | No | — |
| Clips and start frames (`EPISODES`, `DEMO_FRAMES`) | No, they're defaults in `start_services.sh` | — |
| Cosmos / GR00T-H repo versions | Possibly: `setup.sh` clones the latest `main` | If something breaks, check `cosmos_worker.py` first: it calls the runner's private `_run_entry`. Pin a commit in `setup.sh` once things are final. |

## Checks after a rebuild

```bash
KEY=$(grep API_KEY /data/seg4.env | cut -d= -f2)
curl -s https://<new-secure-link>/health                       # lists 11 episodes + start_frames
cd ~/work/GR00T-H && uv run --with requests python ~/seg4/server/load_test.py --url https://<new-secure-link> --key $KEY --dreams 10
```

Expected (Nov 4): 0 errors; cached calls up to ~9 s when 350 land in the same second; 10 simultaneous dreams finish in ~55 s; peak GPU memory ~23 GB.

GR00T-H outputs are seeded, so the open-loop numbers should match `HANDOFF.md` §5 exactly. Cosmos dreams may differ slightly from the hosted bundles; that's fine.
