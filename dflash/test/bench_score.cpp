// Standalone timing of flash_prefill_forward_bf16 at multiple seq lengths.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

namespace dflash27b { namespace flashprefill {
struct FlashPrefillConfig {
    int block_size = 128;
    float alpha = 0.12f;
    int attention_sink = 2, window = 4, last_n_full = 2;
};
int flash_prefill_forward_bf16(
    const void*Q, const void*K, const void*V, void*O,
    int B, int S, int H, int Hk, int D, float scale,
    const FlashPrefillConfig& cfg);
}}

int main() {
    const int seqs[] = {8192, 16384, 32768, 65536};
    for (int S : seqs) {
        const int B=1, H=16, Hk=8, D=128;
        std::vector<hip_bfloat16> Q(B*S*H*D), K(B*S*Hk*D), V(B*S*Hk*D);
        uint16_t tiny = 0x2400u;  // 0.01f in bf16
        for (auto& x : Q) memcpy(&x, &tiny, 2);
        for (auto& x : K) memcpy(&x, &tiny, 2);
        for (auto& x : V) memcpy(&x, &tiny, 2);

        hip_bfloat16 *dQ, *dK, *dV, *dO;
        hipMalloc(&dQ, Q.size()*2); hipMalloc(&dK, K.size()*2);
        hipMalloc(&dV, V.size()*2); hipMalloc(&dO, Q.size()*2);
        hipMemcpy(dQ, Q.data(), Q.size()*2, hipMemcpyHostToDevice);
        hipMemcpy(dK, K.data(), K.size()*2, hipMemcpyHostToDevice);
        hipMemcpy(dV, V.data(), V.size()*2, hipMemcpyHostToDevice);

        dflash27b::flashprefill::FlashPrefillConfig cfg;
        float sc = 1.f/sqrtf((float)D);

        // warmup
        dflash27b::flashprefill::flash_prefill_forward_bf16(dQ,dK,dV,dO,B,S,H,Hk,D,sc,cfg);
        hipDeviceSynchronize();

        hipEvent_t e0,e1;
        hipEventCreate(&e0); hipEventCreate(&e1);
        hipEventRecord(e0);
        for (int i=0;i<3;i++)
            dflash27b::flashprefill::flash_prefill_forward_bf16(dQ,dK,dV,dO,B,S,H,Hk,D,sc,cfg);
        hipEventRecord(e1);
        hipEventSynchronize(e1);
        float ms; hipEventElapsedTime(&ms,e0,e1);
        std::printf("[bench] S=%5d  e2e=%.1f ms/iter\n", S, ms/3.f);

        hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO);
    }
}
