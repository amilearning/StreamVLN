#!/usr/bin/env bash
# Start the StreamVLN inference server (GPU, Docker).
#
#   ./deploy/run_server.sh                 # foreground, ctrl-C to stop
#   ./deploy/run_server.sh -d              # detached
#   CKPT=/other/ckpt ./deploy/run_server.sh
#
# Serves  POST http://0.0.0.0:5801/eval_vln
#         GET  http://<host>:5801/debug     <- live annotated view in a browser
#         GET  http://<host>:5801/status    <- JSON, same info for scripts
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/.." && pwd)}"

IMAGE="${IMAGE:-streamvln:thor}"
CKPT="${CKPT:-$HOME/streamvln_ckpt/streamvln_real_world}"
# The SigLIP vision tower is NOT bundled in the checkpoint: config.json references
# google/siglip-so400m-patch14-384 by repo id, so it must be pre-staged here or the
# server cannot start offline. See DEPLOYMENT_PLAN.md.
HF="${HF:-$HOME/streamvln_ckpt/hf_cache}"
NAME="${NAME:-streamvln_server}"
PORT="${PORT:-5801}"
ATTN="${STREAMVLN_ATTN:-sdpa}"   # sdpa measured 40% FASTER than flash_attention_2 on Thor

for path in "$CKPT" "$HF"; do
  [[ -d "$path" ]] || { echo "ERROR: missing $path -- see DEPLOYMENT_PLAN.md" >&2; exit 1; }
done
docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || { echo "ERROR: image '$IMAGE' not found. Build it: cd deploy/docker && docker build -t $IMAGE ." >&2; exit 1; }

# Only one client may talk to this server: it holds ONE global KV-cache session.
docker rm -f "$NAME" >/dev/null 2>&1 || true

MODE="${1:--it}"   # -d to detach
echo "starting $NAME  image=$IMAGE  attn=$ATTN  port=$PORT"
echo "  ckpt=$CKPT"
echo "  debug view -> http://localhost:$PORT/debug"

exec docker run --rm "$MODE" \
  --runtime nvidia --network host --ipc=host \
  --name "$NAME" \
  -e STREAMVLN_ATTN="$ATTN" \
  -e HF_HOME=/hf \
  -v "$REPO":/workspace/StreamVLN \
  -v "$CKPT":/ckpt:ro \
  -v "$HF":/hf:ro \
  "$IMAGE" \
  python3 http_realworld_server.py --model_path /ckpt --device cuda:0
