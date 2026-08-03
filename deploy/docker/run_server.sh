#!/usr/bin/env bash
# Launch the StreamVLN real-world inference server on the Jetson Thor.
# Code and checkpoint are mounted, not baked in, so both stay editable.
set -euo pipefail

REPO=${REPO:-/home/frlthor/StreamVLN}
CKPT=${CKPT:-/home/frlthor/streamvln_ckpt/streamvln_real_world}
# The SigLIP vision tower (google/siglip-so400m-patch14-384) is NOT bundled in the
# checkpoint -- config.json references it by repo id, so it must be pre-staged here
# or the server cannot start offline.
HF=${HF:-/home/frlthor/streamvln_ckpt/hf_cache}
NAME=${NAME:-streamvln_server}

exec docker run --rm -it \
  --runtime nvidia --network host --ipc=host \
  --name "$NAME" \
  -e STREAMVLN_ATTN="${STREAMVLN_ATTN:-sdpa}" \
  -e HF_HOME=/hf \
  -v "$REPO":/workspace/StreamVLN \
  -v "$CKPT":/ckpt:ro \
  -v "$HF":/hf:ro \
  streamvln:thor \
  python3 http_realworld_server.py --model_path /ckpt --device cuda:0
