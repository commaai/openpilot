#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// One wave per token: gather its four expert gradients and write only their sum.
extern "C" __global__ __launch_bounds__(64) void gather_sum(
    __hip_bfloat16* __restrict__ out, const __hip_bfloat16* __restrict__ table, const int* __restrict__ indices) {
  #pragma clang fp reassociate(off)
  const int token = blockIdx.x;
  const long long group_base = (long long)(token / TOKENS) * ROWS * HIDDEN;
  const int4 rows = *reinterpret_cast<const int4*>(&indices[(long long)token * 4]);
  const long long row0 = group_base + (long long)rows.x * HIDDEN;
  const long long row1 = group_base + (long long)rows.y * HIDDEN;
  const long long row2 = group_base + (long long)rows.z * HIDDEN;
  const long long row3 = group_base + (long long)rows.w * HIDDEN;
  #pragma unroll 2
  for (int d = threadIdx.x; d < HIDDEN; d += 64) {
    float sum = (float)table[row0 + d];
    sum = sum + (float)table[row1 + d];
    sum = sum + (float)table[row2 + d];
    sum = sum + (float)table[row3 + d];
    out[(long long)token * HIDDEN + d] = (__hip_bfloat16)sum;
  }
}
