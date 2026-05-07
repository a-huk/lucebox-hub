#!/usr/bin/env bash
# Start the DFlash OpenAI-compatible server inside the vllm container.
# Runs in the foreground — Ctrl+C to stop.
#
# Endpoints exposed on the host at http://localhost:8000/v1
#   POST /v1/chat/completions   (OpenAI + Anthropic format, streaming)
#   GET  /v1/models

set -euo pipefail

CONTAINER=vllm
PORT=${PORT:-8000}
KEEP_RATIO=${KEEP_RATIO:-0.15}
PREFILL_THRESHOLD=${PREFILL_THRESHOLD:-8000}   # tokens; compress prompts >= this
MAX_CTX=${MAX_CTX:-262144}

REPO=/home/hukad/specprefill/lucebox-hub/dflash
MODELS=/home/hukad/specprefill/models

if ! podman inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
  echo "Starting container $CONTAINER..."
  podman start "$CONTAINER"
  sleep 5
fi

podman exec -it "$CONTAINER" bash -c "
  cd $REPO/scripts

  # DFLASH_FP_USE_BSA / DFLASH_FP_ALPHA are set by server.py when
  # --prefill-compression != off; we only need the extras here.
  export DFLASH_FP_LOOKAHEAD=64

  python3 server_tools.py \
    --bin    $REPO/build-hip-phase2/test_dflash \
    --target $MODELS/Qwen3.6-27B-UD-Q4_K_XL.gguf \
    --draft  $MODELS/draft_3.6 \
    --tokenizer Qwen/Qwen3.6-27B \
    --port   $PORT \
    --max-ctx $MAX_CTX \
    --prefill-compression auto \
    --prefill-drafter          $MODELS/Qwen3-0.6B-BF16.gguf \
    --prefill-drafter-tokenizer Qwen/Qwen3-0.6B \
    --prefill-threshold  $PREFILL_THRESHOLD \
    --prefill-keep-ratio $KEEP_RATIO
"
