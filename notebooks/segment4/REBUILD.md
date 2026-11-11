# Rebuilding the Brev Launchable from scratch

A fresh instance starts with no data. Everything is rebuilt by two scripts. A few values change on each rebuild; this note lists them.

## Steps

```bash
git clone https://github.com/MiningMyBusiness/corl-tutorial-brev-setup-and-segment4.git ~/repo && ln -s ~/repo/notebooks/segment4 ~/seg4     # scripts expect ~/seg4 (or set SEG4=)
HF_TOKEN=hf_... API_KEY=<key on the slides> bash ~/seg4/brev/setup.sh  # ~30-40 min: repos, checkpoints, Docker image, data
bash ~/seg4/brev/start_services.sh                                     # ~15 min cold: both servers + cache for all 11 clips (open loop, dreams incl. ACT, closed loops)
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
| Cosmos / GR00T-H repo versions | No: `setup.sh` checks out the commits tested Nov 11 (`GROOT_REF`, `COSMOS_REF`) | Override with `GROOT_REF=main` etc. only on purpose. `cosmos_worker.py` calls the runner's private `_run_entry`, so newer Cosmos versions may break it. |
| SutureBot ACT | No: precomputed predictions live on HF (`kdb6df/corl2026-suturebot-act`) | Nothing to set up; `precompute.py` downloads them. Rebuild them only to change ACT (`HANDOFF.md`, "SutureBot ACT"). |

## Checks after a rebuild

```bash
KEY=$(grep API_KEY /data/seg4.env | cut -d= -f2)
curl -s https://<new-secure-link>/health                       # 11 episodes, start_frames, "closed_loop": true
unzip -l /data/shared/seg4_bundle_1441.zip | grep -E "dream_act|closed_groot"   # ACT rollout + closed loop made it into the bundle
cd ~/work/GR00T-H && uv run --with requests python ~/seg4/server/load_test.py --url https://<new-secure-link> --key $KEY --dreams 10
```

Expected (Nov 4): 0 errors; cached calls up to ~9 s when 350 land in the same second; 10 simultaneous dreams finish in ~55 s. GPU memory is ~24 GB at idle since the closed loop was added (second GR00T-H embodiment), ~29 GB under load.

Last, run the v4 notebook once in `api` mode with `API_URL` set to the new link and the key filled in.

GR00T-H outputs are seeded, so the open-loop numbers should match `HANDOFF.md` §5 exactly. Cosmos dreams may differ slightly from the hosted bundles; that's fine.

## Backup server

A second Brev instance running the same backend, in case the main one fails during the tutorial. The two servers share nothing: each builds its own cache, and both are independent of the HF bundles that the notebook's `cache` backend uses.

1. **Launch** with the same Launchable settings (README): GPU ≥ 40 GB, driver R580+, ≥ 200 GB disk, port 8000 exposed through a secure link with authentication disabled.
2. **Build it with the same API key as the main server**, so participants only have to change the URL. The key is on the slides and in the main box's `/data/seg4.env`. Pass it on the command line; never commit it.
   ```bash
   git clone https://github.com/MiningMyBusiness/corl-tutorial-brev-setup-and-segment4.git ~/repo && ln -s ~/repo/notebooks/segment4 ~/seg4
   HF_TOKEN=hf_... API_KEY=<same key as the main server> bash ~/seg4/brev/setup.sh   # ~30-40 min
   bash ~/seg4/brev/start_services.sh                                                # ~15 min cold
   ```
   A read-only HF token is enough. Everything is public, and the token only speeds up downloads.
3. **Check it** with the "Checks after a rebuild" above, against the backup's secure link.
4. **Don't run `upload_bundles.py` from the backup.** The HF bundles came from the main server. Cosmos isn't bit-for-bit repeatable, so re-uploading would change the cached numbers participants see. GR00T-H is seeded, so its outputs match the main server exactly; the backup's Cosmos dreams look the same but differ slightly in L1/SSIM.
5. **Switching over:** participants paste the backup URL into `API_URL` in the setup cell and re-run from there. Have the backup URL ready on a slide or in the chat. If neither server answers, the notebook falls back to the HF cache automatically: the main thread still works, but the live Explore cells (custom dreams, Explore F's closed loop) need a server.
6. **If the backup instance is stopped and restarted** (e.g. to save cost until the tutorial), rerun `start_services.sh`. The Cosmos container restarts on its own (`--restart unless-stopped`), but the gateway is a plain background process and does not survive a reboot. With the cache already on disk it's back in ~4 min, mostly Cosmos warm-up.
