#include <hip/hip_runtime.h>

// Copy FP8/scale bytes unchanged; every output is written, including padded rows.
// dest is retained as a saved backward input, but forward uses only src_row.
extern "C" __global__ __launch_bounds__(64) void dispatch_gather(
    unsigned char* __restrict__ out, const unsigned char* __restrict__ x,
    const int* __restrict__ dest, const int* __restrict__ src_row) {
  const int row = blockIdx.x;
  const int source = src_row[row];
  const long long token = (long long)(row / ROWS) * TOKENS + (source >= 0 ? source / 4 : 0);
  for (int d = threadIdx.x; d < HIDDEN; d += 64)
    out[(long long)row * HIDDEN + d] = source >= 0 ? x[token * HIDDEN + d] : 0;
}
