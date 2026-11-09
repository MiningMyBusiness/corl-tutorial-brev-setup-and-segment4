#!/usr/bin/env bash
# Start (or restart) the Cosmos worker container and the public gateway.
# Usage:  [EPISODES=1441,1443 DEMO_FRAMES=1441:345,1443:270] bash ~/seg4/brev/start_services.sh
# Logs:   /data/shared/gateway.log   and   docker logs -f seg4-cosmos
set -euo pipefail
SEG4=${SEG4:-$HOME/seg4}; WORK=${WORK:-$HOME/work}; DATA=${DATA:-/data}
export PATH="$HOME/.local/bin:$PATH"
set -a; source "$DATA/seg4.env"; set +a
# All SutureBot clips held out for BOTH GR00T-H and Cosmos-H-Dreams. The first is the main-thread demo
# clip; 1443 is the backup. 1441-1444 needle throw, 1445-1451 knot tying.
EPISODES=${EPISODES:-1441,1442,1443,1444,1445,1446,1447,1448,1449,1450,1451}
# Dream start frame per episode: server/pick_start_frames.py (1443 kept at 270, validated before the picker existed)
DEMO_FRAMES=${DEMO_FRAMES:-1441:345,1442:255,1443:270,1444:330,1445:105,1446:120,1447:105,1448:105,1449:30,1450:165,1451:135}
PORT=${PORT:-8000}   # expose this port publicly in the Launchable (HTTP, all IPs)

echo "== Cosmos worker (container, localhost:8001) =="
docker rm -f seg4-cosmos >/dev/null 2>&1 || true
docker run -d --name seg4-cosmos --restart unless-stopped --network host --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -e SHARED_DIR=/data/shared -e DATA_ROOT=/data/suturebot \
  -e COSMOS_STATS="${COSMOS_STATS:-}" \
  -v "$WORK/Cosmos-H-Dreams:/workspace/cosmos-h-dreams" -v "$SEG4:/workspace/seg4" \
  -v "$DATA:/data" -w /workspace/cosmos-h-dreams cosmos-h-dreams:latest \
  bash -lc "uv sync --extra dev --extra runners && \
            uv run --with fastapi --with uvicorn --with scipy python /workspace/seg4/server/cosmos_worker.py"

echo "== Gateway (0.0.0.0:$PORT, GR00T-H in-process) =="
pkill -f server/gateway.py 2>/dev/null || true
# wait for the old gateway to exit, or the health check below can hit it while it shuts down
for i in $(seq 1 30); do pgrep -f "[s]erver/gateway.py" >/dev/null || break; sleep 1; done
cd "$WORK/GR00T-H"
API_KEY=$API_KEY PORT=$PORT EPISODES=$EPISODES DEMO_FRAMES=$DEMO_FRAMES DATA_ROOT=$DATA/suturebot SHARED_DIR=$DATA/shared \
PYTHONPATH="$SEG4/server" nohup uv run --with fastapi --with "uvicorn[standard]" --with scipy \
  --with scikit-image --with requests python "$SEG4/server/gateway.py" > "$DATA/shared/gateway.log" 2>&1 &

ok=0
for i in $(seq 1 60); do
  curl -sf "localhost:$PORT/health" && { echo; echo "Gateway up. API_KEY=$API_KEY"; ok=1; break; }
  sleep 5
done
[ $ok = 1 ] || { echo "Gateway not healthy after 5 min; see $DATA/shared/gateway.log"; exit 1; }

echo "== Waiting for the Cosmos worker (first start: uv sync + model load + compile, several minutes) =="
for i in $(seq 1 120); do
  curl -sf localhost:8001/health && { echo; echo "Cosmos worker ready."; ok=2; break; }
  sleep 5
done
[ $ok = 2 ] || { echo "Cosmos worker not ready after 10 min; see: docker logs seg4-cosmos"; exit 1; }

# Warm the cache for every demo clip so the notebook's main thread never waits on the GPU,
# and rebuild the offline bundles ($DATA/shared/seg4_bundle_<ep>.zip). Already-cached calls are instant.
if [ "${WARM_CACHE:-1}" = 1 ]; then
  echo "== Warming cache + bundles for $DEMO_FRAMES =="
  cd "$WORK/GR00T-H"
  for pair in ${DEMO_FRAMES//,/ }; do
    uv run --with requests python "$SEG4/server/precompute.py" --key "$API_KEY" --url "http://localhost:$PORT" \
      --episode "${pair%%:*}" --frame "${pair##*:}" --out "$DATA/shared/bundle_${pair%%:*}" | tail -1
  done
fi
echo "All services up. API_KEY=$API_KEY"
