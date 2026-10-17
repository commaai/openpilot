#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// GPT-OSS production keeps two 8,192-token sequences on each DP rank. One wave
// owns one token row, so its 16-byte weight reads and output writes are fully
// contiguous instead of interleaving four unrelated vocabulary rows per wave.
using uint4v = unsigned int __attribute__((ext_vector_type(4)));

constexpr int TOKENS = 16384;
constexpr int VOCAB = 128256;
constexpr int EMBED = 2880;
constexpr int VECS_PER_ROW = EMBED / 8;
constexpr int THREADS = 512;
constexpr int WAVE = 64;
constexpr int ROWS_PER_WG = THREADS / WAVE;

static_assert(EMBED % 8 == 0 && TOKENS % ROWS_PER_WG == 0);

extern "C" __global__ __launch_bounds__(THREADS) void gptoss_embedding_fwd(
    __hip_bfloat16 *__restrict__ out, const int *__restrict__ idx,
    const __hip_bfloat16 *__restrict__ weight) {
  const int lid = threadIdx.x;
  const int lane = lid & (WAVE - 1);
  const int row = blockIdx.x * ROWS_PER_WG + (lid / WAVE);
  const int token = idx[row];
  const uint4v zero = {0u, 0u, 0u, 0u};

#pragma unroll
  for (int j = lane; j < VECS_PER_ROW; j += WAVE) {
    // The embedding is a one-use, random-row source. Bypass temporal caching so
    // it does not displace the freshly written activation consumed by RMSNorm.
    const uint4v value = static_cast<unsigned>(token) < VOCAB ?
      __builtin_nontemporal_load(reinterpret_cast<const uint4v *>(weight) +
                                 static_cast<long long>(token) * VECS_PER_ROW + j) : zero;
    reinterpret_cast<uint4v *>(out)[static_cast<long long>(row) * VECS_PER_ROW + j] = value;
  }
}
