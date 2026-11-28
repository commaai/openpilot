#include <hip/hip_runtime.h>

// Valid inverse writes and padding writes are disjoint, so this needs no initialization pass or grid barrier.
extern "C" __global__ __launch_bounds__(256) void dispatch_inverse(
    int* __restrict__ out, const int* __restrict__ dest, const int* __restrict__ counts, const int* __restrict__ off) {
  const int group = blockIdx.x / ((ROWS + 255) / 256);
  const int i = (blockIdx.x % ((ROWS + 255) / 256)) * 256 + threadIdx.x;
  if (i < ENTRIES) out[(long long)group * ROWS + dest[(long long)group * ENTRIES + i]] = i;
  if (i < ROWS) {
    int expert = 0;
    #pragma unroll
    for (int e = 1; e < EXPERTS; e++) expert += i >= off[group * (EXPERTS + 1) + e];
    if (i >= off[group * (EXPERTS + 1) + expert] + counts[group * EXPERTS + expert])
      out[(long long)group * ROWS + i] = -1;
  }
}
