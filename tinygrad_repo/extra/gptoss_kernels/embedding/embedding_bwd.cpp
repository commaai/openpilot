#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#ifndef TOKENS
#define TOKENS 131072
#endif
#ifndef VOCAB
#define VOCAB 128256
#endif
#ifndef EMBED
#define EMBED 2880
#endif
#ifndef THREADS
#define THREADS 256
#endif

#ifdef INIT_HEADS
extern "C" __global__ __launch_bounds__(THREADS) void embedding_bwd_init_heads(int *__restrict__ head) {
  const int v=blockIdx.x*THREADS+threadIdx.x;
  if(v<VOCAB) head[v]=-1;
}
#elif defined(BUILD_LINKS)
// atomicExch is one operation per token, not one operation per embedding element.
extern "C" __global__ __launch_bounds__(THREADS) void embedding_bwd_build_links(
    int *__restrict__ next_idx, int *__restrict__ head, const int *__restrict__ idx) {
  const int i=blockIdx.x*THREADS+threadIdx.x;
  if(i<TOKENS) {
    const int token=max(0,min(VOCAB-1,idx[i]));
    next_idx[i]=atomicExch(&head[token],i);
  }
}
#else
// One workgroup owns a 256-column block of a vocabulary row. It traverses that token's occurrence list,
// accumulates in FP32, and rounds once to BF16. Empty rows naturally write zero, replacing the old fill.
extern "C" __global__ __launch_bounds__(THREADS) void embedding_bwd_owner_reduce(
    __hip_bfloat16 *__restrict__ out, const __hip_bfloat16 *__restrict__ grad_emb,
    const int *__restrict__ head, const int *__restrict__ next_idx) {
  constexpr int D_BLOCKS=(EMBED+THREADS-1)/THREADS;
  const int token=blockIdx.x/D_BLOCKS, d=(blockIdx.x%D_BLOCKS)*THREADS+threadIdx.x;
  if(d<EMBED) {
    float sum=0.0f;
    for (int i=head[token]; i>=0; i=next_idx[i]) sum+=(float)grad_emb[(long long)i*EMBED+d];
    out[(long long)token*EMBED+d]=(__hip_bfloat16)sum;
  }
}
#endif
