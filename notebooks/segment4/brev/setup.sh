#!/usr/bin/env bash
# One-time setup for the Segment 4 model server on a Brev GPU instance (VM Mode + Jupyter).
# Usage (from a Jupyter terminal):   HF_TOKEN=hf_xxx [API_KEY=...] bash ~/seg4/brev/setup.sh
# Can also be pasted as the Launchable's setup script. Re-runnable.
set -euo pipefail

SEG4=${SEG4:-$HOME/seg4}
WORK=${WORK:-$HOME/work}
DATA=${DATA:-/data}
mkdir -p "$WORK"; sudo mkdir -p "$DATA"; sudo chown -R "$USER" "$DATA"
mkdir -p "$DATA/shared"

echo "== GPU / driver =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$DRV" -lt 580 ]; then
  echo "!! Driver $DRV < 580. Cosmos-H-Dreams needs R580+/CUDA 13. Pick another instance type."; exit 1
fi

# GR00T-H decodes the AV1 videos with torchcodec, which needs the system FFmpeg libraries.
# Some Brev images don't ship them ("libavutil.so.5x: cannot open shared object file").
command -v ffmpeg >/dev/null || { sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg; }

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv tool install -q "huggingface_hub[cli]" || true
[ -n "${HF_TOKEN:-}" ] && hf auth login --token "$HF_TOKEN" || true

echo "== GR00T-H =="
cd "$WORK"
[ -d GR00T-H ] || git clone --recurse-submodules https://github.com/NVIDIA-Medtech/GR00T-H.git
cd GR00T-H
uv sync --python 3.10
uv pip install -e .
uv run python -c "import flash_attn" 2>/dev/null || uv pip install flash-attn==2.7.4.post1 --no-build-isolation
hf download nvidia/GR00T-H-N1.7

echo "== Cosmos-H-Dreams =="
cd "$WORK"
[ -d Cosmos-H-Dreams ] || git clone https://github.com/isaac-for-healthcare/Cosmos-H-Dreams.git
cd Cosmos-H-Dreams
mkdir -p checkpoints
hf download nvidia/Cosmos-H-Dreams --local-dir checkpoints
# The HF repo keeps its weights under checkpoints/, so they land in checkpoints/checkpoints/.
# Link them to checkpoints/<file>, which is where cosmos_worker.py and the README commands look.
for f in checkpoints/checkpoints/*.pt; do
  [ -e "checkpoints/$(basename "$f")" ] || ln -s "checkpoints/$(basename "$f")" "checkpoints/$(basename "$f")"
done
docker build -t cosmos-h-dreams:latest docker/

echo "== SutureBot test split (data + 3 GR00T cameras) =="
hf download nvidia/PhysicalAI-Robotics-Open-H-Embodiment --repo-type dataset \
  --local-dir "$DATA/hf" \
  --include "Surgical/jhu/imerse/suturebot/meta/*" \
  --include "Surgical/jhu/imerse/suturebot/data/chunk-001/*" \
  --include "Surgical/jhu/imerse/suturebot/videos/chunk-001/observation.images.endoscope.left/*" \
  --include "Surgical/jhu/imerse/suturebot/videos/chunk-001/observation.images.wrist.left/*" \
  --include "Surgical/jhu/imerse/suturebot/videos/chunk-001/observation.images.wrist.right/*"
rm -rf "$DATA/suturebot" && ln -s "$DATA/hf/Surgical/jhu/imerse/suturebot" "$DATA/suturebot"
cp "$WORK/GR00T-H/open_h/embodiments/jhu_imerse_dvrk/modality.json" "$DATA/suturebot/meta/modality.json"

echo "== API key =="
# Pass API_KEY=... to reuse the key printed on the slides; otherwise a random one is generated.
[ -f "$DATA/seg4.env" ] || echo "API_KEY=${API_KEY:-$(openssl rand -hex 8)}" > "$DATA/seg4.env"
cat "$DATA/seg4.env"
echo "Setup done. Next: bash $SEG4/brev/start_services.sh   (all clips 1441-1451, ~6 min from scratch)"
