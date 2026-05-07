# Port Progress

## Current Phase: Phase 8 complete — 128K benchmark; NIAH 2/2 ✅, TTFT ~173s (7× vs RTX 3090)

## Status

| Task | Status | Notes |
|------|--------|-------|
| Repo cloned | ✅ done | `/home/hukad/specprefill/lucebox-hub` |
| Submodules init | ✅ done | llama.cpp (luce-dflash branch) + Block-Sparse-Attention |
| Source audit | ✅ done | see ARCHITECTURE.md and GOTCHAS.md |
| CMakeLists.txt HIP port | ✅ done | `DFLASH27B_USE_HIP=ON` flag, preserves original CUDA path |
| hip_compat/ shim headers | ✅ done | cuda_runtime.h, cuda_fp16.h, cuda_bf16.h, mma.h |
| f16_convert.hip.cu | ✅ done | properly hipified with hip_bfloat16 types |
| vendors/hip.h fixes | ✅ done | added missing cuBLAS + stream-capture mappings |
| Python client rocm-smi fix | ✅ done | `_read_vram_used_mib()` with rocm-smi + nvidia-smi fallback |
| Phase 1 build attempt | ✅ done | `test_generate` binary built 2026-05-05; all 7 compat errors fixed |
| Models downloaded | ✅ done | Qwen3.5-27B-Q4_K_M.gguf (16GB), DFlash draft (3.3GB), drafter GGUF (1.5GB) |
| AR baseline benchmark | ✅ done | 11.36 tok/s (post-warmup, 3 runs, coding prompt) |
| dflash spec decode working | ✅ done | 29.23 tok/s avg = **2.57× speedup** (3 runs) |
| PFlash pipeline smoke test | ✅ done | 8K→380 tokens, 21.5× compression, TTFT ~10s (Phase 1) |
| PFlash NIAH accuracy | ⚠️ partial | 2/3 at 8K keep=0.15 with NoPE fix (was 0/3); 0/3 at keep=0.05 |
| flashprefill_kernels.hip.cu (Phase 2) | ✅ done | rocWMMA port; FP 2.45s→1.02s (2.4×) at 8K; ~1.0× at 32K (kernel 2 overhead) |
| BSA port (Phase 3) | ✅ done | bsa_launcher_hip.cu wraps rocWMMA kernel; DFLASH27B_HAVE_BSA=1 enabled for HIP |
| Kernel 2 GEMM (Phase 4) | ✅ done | compute_block_score_gemm_kernel: score 3.48ms→0.67ms (8K), ~56ms→5.6ms (32K) |

## Files modified/created

| File | Change | Status |
|------|--------|--------|
| `dflash/CMakeLists.txt` | Added `DFLASH27B_USE_HIP` flag; dual CUDA/HIP support | ✅ |
| `dflash/src/f16_convert.hip.cu` | NEW — hipified version (hip_bfloat16 types, hipStream_t) | ✅ |
| `dflash/hip_compat/cuda_runtime.h` | NEW — cudaMemcpy* → hip, stream capture, peer access | ✅ |
| `dflash/hip_compat/cuda_fp16.h` | NEW — includes hip/hip_fp16.h | ✅ |
| `dflash/hip_compat/cuda_bf16.h` | host/device split: raw bit-cast on host (g++), constructors on device (hipcc) | ✅ |
| `dflash/hip_compat/cuda_runtime.h` | Added `__HIP_PLATFORM_AMD__` guard + `cudaEvent_t` alias | ✅ |
| `dflash/hip_compat/mma.h` | NEW — empty Phase 1 stub (rocWMMA Phase 2) | ✅ |
| `dflash/src/flashprefill_kernels.hip.cu` | NEW — rocWMMA port of all 4 kernels; AMD Wave32 acc layout | ✅ |
| `dflash/src/bsa_launcher_hip.cu` | NEW — Phase 3 HIP BSA launcher wrapping rocWMMA kernel 4 | ✅ |
| `dflash/src/flashprefill_kernels.hip.cu` | Phase 4: added compute_block_score_gemm_kernel + launcher | ✅ |
| `dflash/src/flashprefill.cpp` | Phase 4: mean_Q alloc + GEMM score call under DFLASH27B_USE_HIP | ✅ |
| `dflash/CMakeLists.txt` | Phase 4: DFLASH27B_USE_HIP=1 compile definition + bench_score target | ✅ |
| `dflash/deps/llama.cpp/ggml/src/ggml-cuda/vendors/hip.h` | Added missing cublasSgemmStridedBatched, cudaStreamCapture* mappings | ✅ |
| `pflash/pflash/dflash_client.py` | `_read_vram_used_mib()` with hipMemGetInfo delta + nvidia-smi fallback | ✅ |
| `dflash/src/qwen3_0p6b_graph.cpp` | Phase 6: NoPE tail-score (K_norope_v stores pre-RoPE K; used in tail attention) | ✅ |
| `pflash/tests/bench_niah_cpp.py` | Phase 6: enable_thinking=False + fixed default drafter GGUF path | ✅ |

## Build/runtime bugs fixed (in order)

1. **ROCm clang missing CRT** (`Scrt1.o`): fixed by using `CC=gcc CXX=g++`
2. **`/opt/rocm-7.0/include` missing**: switched build container from `llama-rocm7-nightlies` to `vllm`
3. **`hipcc` wrapper rejected by CMake**: use `/opt/rocm/llvm/bin/clang++` directly
4. **`cuda_fp16.h` not found in ggml HIP build** (`gated_delta_net.cu`): injected `hip_compat/` into ggml-hip include path via `target_include_directories(ggml-hip ...)`
5. **`cudaStreamCaptureStatusNone` undeclared** (`fattn-chunked.cu`): added to both `hip_compat/cuda_runtime.h` AND `vendors/hip.h`
6. **`cublasSgemmStridedBatched` undeclared** (`fattn-chunked.cu`): added to `vendors/hip.h`
7. **`__hip_bfloat16` / `__bfloat162float` wrong names** (`f16_convert.hip.cu`): fixed to `hip_bfloat16` + `static_cast<float>()` in source
8. **`__HIP_PLATFORM_AMD__` not set by g++** (Phase 2 CXX sources): added guard to `hip_compat/cuda_runtime.h`
9. **`cudaEvent_t` undeclared** (`flashprefill.cpp`): added `using cudaEvent_t = hipEvent_t` to compat header
10. **`hip_bfloat16` constructors device-only under g++** (`cuda_bf16.h`): rewrote with `#ifdef __HIPCC__` / raw bit-cast split
11. **`DFLASH27B_MIN_SM` not defined in HIP path**: added `DFLASH27B_MIN_SM=80` compile definition alongside `DFLASH27B_HAVE_FLASHPREFILL=1` so kernel dispatch activates
12. **`hip::host` missing from `test_flashprefill_kernels`**: added to get ROCm includes without device compile flags on g++

## Next actions (pick up here)

### Phase 3 build (complete — use this)
```bash
podman start vllm
podman exec -w /home/hukad/specprefill/lucebox-hub/dflash vllm bash -c "
cmake --build build-hip-phase2 --target test_dflash test_generate -j\$(nproc)
"
# Binary: build-hip-phase2/test_dflash
# Enable BSA path: DFLASH_FP_USE_BSA=1 DFLASH_FP_ALPHA=0.85
```

### NIAH root-cause: drafter token importance scoring
The needle (random 8-char key + 7-digit number) is not surviving the token selection step.
Verified at 5% AND 15% keep_ratio: model explicitly says "no numbers in text."
This is a DRAFTER SCORING quality issue, not our kernel:
- The mean-K block scoring (kernel 2) assigns similar scores to all filler/needle blocks
- After FA forward, the token-level importance scoring ranks needle tokens as filler
- Options to investigate:
  a) Use `--keep-ratio 0.30` or higher (brute force, degrades compression ratio)
  b) Add attention-sink + window parameters to block_select for positional coverage
  c) Swap to attention-score-based token importance (different from hidden-state norm)
  d) Test with a different NIAH prompt where the needle is semantically distinct

### Phase 4 build (complete — use this)
```bash
podman start vllm
ROCM_PATH=/opt/rocm CC=gcc CXX=g++ podman exec -w /home/hukad/specprefill/lucebox-hub/dflash vllm bash -c "
cmake -S . -B build-hip-phase2 -DDFLASH27B_USE_HIP=ON -DDFLASH27B_HIP_SM80_EQUIV=ON \
  -DCMAKE_HIP_COMPILER=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/llvm/bin/clang++
make -C build-hip-phase2 test_dflash bench_score -j\$(nproc)
"
```

### Phase 5 (complete — measured 2026-05-06)
Measured Phase 4 end-to-end compress timing. Key results (see BENCHMARKS.md Phase 4 section):
- 8K compress: 3.19s (Phase 2) → 1.9s (Phase 4) = 1.7× improvement
- 32K compress: 40.05s (Phase 2) → 14.5s (Phase 4) = 2.76× improvement
- FP at 32K: 34.78s → 9.5s = 3.66× (BSA dominant; GEMM adds ~1.4s savings)
- NIAH: 0/3 at 8K, 0/2 at 32K (root cause unchanged — needle scoring quality)

### Phase 6: NoPE tail-score fix (complete — 2026-05-06)

**Root cause**: RoPE distance decay in tail-score. Q_last × K dominated by positional distance,
not content. Chunks at Δ≈4000 always scored below recent/sink chunks even if content matches.

**Fix implemented**: `K_norope_v` persistent buffer stores pre-RoPE K for each layer.
Tail-score now uses K_norope_v instead of post-RoPE K_curr_v. Requires extra HBM:
  16 layers × 8 heads × (S_max/32) chunks × D=128 × BF16 ≈ +0.5GB at 8K, +4GB at 64K.

**Result (8K, BSA=1, alpha=0.85, n_lookahead=64, enable_thinking=False)**:
| keep_ratio | n_keep | NIAH | vs baseline |
|:----------:|:------:|:----:|:-----------:|
| 0.05       | 12     | 0/3  | same (needle is rank 19) |
| 0.15       | 38     | 2/3  | was 0/3 before fix |

Needle chunk (case 0, chunk 133 pos 4256-4287) rises to rank 19 after NoPE fix vs never
appearing in top 43 with RoPE. Benefit is real but random-key NIAH needles still score
similarly to uniform filler — no content advantage, only lack of positional disadvantage.

Also fixed in bench:
- `enable_thinking=False` in apply_chat_template (was spending all 64 tokens on CoT)
- Default `--drafter-gguf` path corrected to `/home/hukad/specprefill/models/Qwen3-0.6B-BF16.gguf`

### Phase 7: 64K benchmark (complete — measured 2026-05-06)
bench_niah_cpp.py, 2 cases, keep=0.05, BSA=1, alpha=0.85:
- FP step: ~41s, total compress: ~49.5s, prefill: ~15.5s, TTFT: ~73s
- RTX 3090 reference: 13.5s → 5.4× slower (bandwidth ratio 3.7× explains most)
- NIAH: 0/2 (same root cause as 8K/32K)
- Raising DFLASH_FP_LOOKAHEAD to 32 (question in Q_last) did NOT fix NIAH — attention sinks dominate.

### Phase 8: 128K benchmark (complete — 2026-05-06)
bench_niah_cpp.py --n 2, keep=0.05, BSA=1, alpha=0.85, n_lookahead=64, NoPE fix active.
- compress: ~135-141s (FP ~114-120s, tail ~8.5s)
- target prefill (6524 tok): ~33s
- TTFT: **~169-176s** (avg ~173s); RTX 3090 ref: 24.8s → **7× slower**
- Compression ratio: 20.1× (131K → 6524 tokens)
- NIAH: **2/2 ✅** (both cases correct with NoPE fix + keep=0.05)
- FP kernel scales ~O(S) from 64K→128K: 41s→114s ≈ 2.8×
- Tail-score scales ~O(S): 2.85s→8.5s ≈ 3×

### Extended benchmarks to run
- `bench_niah_cpp.py` at 8K, 32K, 64K, 128K (needs BSA for accuracy at 64K+)
- DFlash spec decode on HumanEval / Math500 tasks (beyond the quicksort coding prompt)
- AR prefill throughput (batched, not single-token decode)

## Known issues / open questions

1. **ggml-hip BF16 conflict**: `hip_compat/cuda_bf16.h` defines `__bfloat162float` as inline helper. May conflict with ROCm's own definition in some contexts. If redefinition errors appear, wrap with `#ifndef __bfloat162float`.
2. **boot_vram_mib**: on 128 GB unified, the 18 GB wait threshold needs tuning. Disable check or lower to 4000.
3. **RDNA3.5 rocBLAS BF16**: need to confirm rocBLAS has BF16 GEMM acceleration on gfx1151.
