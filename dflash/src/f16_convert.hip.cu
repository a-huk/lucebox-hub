// HIP port of f16_convert.cu. Manually hipified.
// Changes: cuda_runtime.h→hip_runtime.h, cuda_fp16.h→hip_fp16.h, cuda_bf16.h→hip_bfloat16.h,
//          __nv_bfloat16→hip_bfloat16 (ROCm uses no underscore prefix), cudaStream_t→hipStream_t,
//          __bfloat162float(x)→static_cast<float>(x) (hip_bfloat16 has operator float()).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

static __global__ void f16_to_f32_kernel(const __half * __restrict__ src,
                                         float * __restrict__ dst,
                                         size_t n_elems) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems) {
        dst[i] = __half2float(src[i]);
    }
}

static __global__ void bf16_to_f32_kernel(const hip_bfloat16 * __restrict__ src,
                                          float * __restrict__ dst,
                                          size_t n_elems) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems) {
        dst[i] = static_cast<float>(src[i]);  // hip_bfloat16 has implicit operator float()
    }
}

extern "C" void dflash27b_launch_f16_to_f32(const void * src,
                                            void * dst,
                                            size_t n_elems,
                                            hipStream_t stream) {
    const int threads = 256;
    const int blocks  = (int)((n_elems + threads - 1) / threads);
    f16_to_f32_kernel<<<blocks, threads, 0, stream>>>(
        (const __half *)src, (float *)dst, n_elems);
}

extern "C" void dflash27b_launch_bf16_to_f32(const void * src,
                                             void * dst,
                                             size_t n_elems,
                                             hipStream_t stream) {
    const int threads = 256;
    const int blocks  = (int)((n_elems + threads - 1) / threads);
    bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(
        (const hip_bfloat16 *)src, (float *)dst, n_elems);
}
