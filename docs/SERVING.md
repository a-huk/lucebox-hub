# Serving

OpenAI-compatible HTTP server wrapping Qwen3.6-27B with DFlash spec decode
and PFlash speculative prefill compression.

## Hardware

Ryzen AI Max+ 395 · Radeon 8060S (gfx1151) · 128 GB unified memory  
Effective GPU slice reported by ROCm: ~64 GB, but the full 128 GB is accessible.

## Start the server

```bash
~/specprefill/scripts/start_server.sh
```

Runs in the foreground inside the `vllm` container. Ctrl+C to stop.  
Default port: **8000**.

Override defaults with env vars before running:

```bash
PORT=9000 KEEP_RATIO=0.10 MAX_CTX=131072 ~/specprefill/scripts/start_server.sh
```

| Env var | Default | Meaning |
|---|---|---|
| `PORT` | `8000` | HTTP port |
| `KEEP_RATIO` | `0.15` | Fraction of prompt chunks kept after PFlash compression |
| `PREFILL_THRESHOLD` | `8000` | Min prompt tokens before compression fires (shorter → no drafter) |
| `MAX_CTX` | `262144` | KV cache size (input + output, model cap = 262144) |

## Connect

Set your client's OpenAI base URL and any non-empty API key:

```
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=sk-any
```

Works with: **Open WebUI**, **Cline**, **Cursor** (remote), **Continue**, any
OpenAI SDK or `curl`.

### curl example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "luce-dflash",
    "stream": true,
    "messages": [{"role":"user","content":"Write a binary search in Python."}]
  }'
```

### Python example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-any")
resp = client.chat.completions.create(
    model="luce-dflash",
    messages=[{"role": "user", "content": "Explain GQA in transformers."}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## How it works

For prompts **shorter than `PREFILL_THRESHOLD`** (default 8K tokens): requests go
straight to the spec-decode daemon — no drafter overhead.

For prompts **at or above the threshold**: PFlash kicks in before prefill:

1. Text is tokenized with the Qwen3-0.6B drafter tokenizer
2. The drafter scores each token chunk via tail-attention (BSA sparse FA, Phase 9 kernel)
3. Top `KEEP_RATIO` chunks are kept; the rest are dropped
4. Compressed text is re-tokenized with the Qwen3.6 tokenizer and prefilled

DFlash spec decode then runs at every generation step (draft tree width = 22 tokens),
giving ~2.5× decode speedup vs autoregressive on this hardware.

## Performance reference (Strix Halo, gfx1151)

| Metric | Value |
|---|---|
| Decode tok/s (AR baseline) | ~11.7 tok/s |
| Decode tok/s (DFlash spec) | ~29 tok/s (~2.5×) |
| PFlash compress — 32K input | ~9.8 s (Phase 9) |
| PFlash compress — 128K input | ~135–140 s |
| NIAH accuracy at 128K, keep=0.05 | 2/2 |
| Compression ratio at keep=0.15 | ~7× |
| Compression ratio at keep=0.05 | ~20× |

RTX 3090 reference: ~3.4× decode speedup, ~25s TTFT at 128K.  
Strix Halo is ~3.7× slower on memory bandwidth (256 vs 936 GB/s) — gap is hardware, not software.

## Tuning keep_ratio

| Use case | Recommended `KEEP_RATIO` |
|---|---|
| Coding / agentic (tool outputs, file context) | `0.15` |
| Long-doc Q&A (you need most of the doc) | `0.15`–`0.20` |
| Summarisation / extraction (structure matters less) | `0.05`–`0.10` |
| RAG / needle tasks at 64K+ | `0.05` |

Lower = faster TTFT, higher compression risk. 0.15 is the safe default for
workloads where losing any context chunk could break the answer.

## Troubleshooting

**Server prints `tokenizer_id = Qwen/Qwen3.5-27B` instead of 3.6**  
Auto-detection read the wrong metadata from the GGUF. The `--tokenizer Qwen/Qwen3.6-27B`
flag in the start script overrides this explicitly — it should not happen.

**`Prompt length (N) exceeds max_ctx`**  
The compressed prompt + a small buffer is larger than `MAX_CTX`. Either lower
`KEEP_RATIO` or raise `MAX_CTX`. At keep=0.15 and 128K input, compressed ≈ 19K tokens;
`MAX_CTX=262144` (the default) always covers this.

**Drafter load is slow on first request**  
Normal — the Qwen3-0.6B weights are loaded on the first request that hits the
compression threshold and freed after. Subsequent requests reload them each time
(drafter is not resident between requests to save VRAM for the KV cache).
