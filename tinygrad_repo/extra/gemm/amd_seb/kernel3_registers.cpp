typedef long unsigned int size_t;
extern "C" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);
struct Dim3 { size_t x, y, z; };
#define __shared__ __attribute__((shared, aligned(16)))
__attribute__((device)) inline void __syncthreads() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

#define BLOCK_SIZE 256
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, BLOCK_SIZE)))
kernel3_registers(float *a, float *b, float *c)
{
  constexpr int N = 4096;
  constexpr float alpha = 1.0;
  constexpr float beta = 0.0;

  const Dim3 blockIdx{ __ockl_get_group_id(0), __ockl_get_group_id(1), __ockl_get_group_id(2) };
  const Dim3 threadIdx{ __ockl_get_local_id(0), __ockl_get_local_id(1), __ockl_get_local_id(2) };

  // Block Tile size
  constexpr int BN = 128;
  constexpr int BM = 128;
  // Number of Row or column we read per batch
  constexpr int BK = 8;

  // Thread Tile size
  constexpr int TN = 4;
  constexpr int TM = 4;

  constexpr int nbWaves = BLOCK_SIZE / 32;
  // Wave Tile size
  constexpr int WN = 64;
  constexpr int WM = BN * BM / nbWaves / WN;

  // Number of wave on X & Y axis in the Block tile
  constexpr int nbWaveX = BN / WN;
  constexpr int nbWaveY = BM / WM;

  const int waveIndex = threadIdx.x / 32;
  const int waveIdx = waveIndex % nbWaveX;
  const int waveIdy = waveIndex / nbWaveX;
  const int indexInWave = threadIdx.x % 32;

  // A wave is a block of 8x4 of the output matrix
  constexpr int nbThreadXPerWave = 8;
  constexpr int nbThreadYPerWave = 4;

  // Thread coordinates in Wave
  const int idxInWave = indexInWave % nbThreadXPerWave;
  const int idyInWave = indexInWave / nbThreadXPerWave;

  constexpr int nbIterWaveN = WN / (nbThreadXPerWave * TN);
  constexpr int nbIterWaveM = WM / (nbThreadYPerWave * TM);

  // Wave Sub-tile size
  constexpr int SUBWN = WN / nbIterWaveN;
  constexpr int SUBWM = WM / nbIterWaveM;

  // Thread mapping to read BKxBN block from A
  int rAIdx = threadIdx.x % BK;
  int rAIdy = threadIdx.x / BK;
  // Thread mapping to read BNxBK block from B
  int rBIdx = threadIdx.x % BN;
  int rBIdy = threadIdx.x / BN;

  constexpr int strideReadB = BLOCK_SIZE / BN;
  constexpr int strideReadA = BLOCK_SIZE / BK;
  constexpr int nbReadsB = BN * BK / BLOCK_SIZE;
  constexpr int nbReadsA = BM * BK / BLOCK_SIZE;

  float A_col[nbIterWaveM * TM];
  float B_row[nbIterWaveN * TN];

  __shared__ float As[BK][BM];
  __shared__ float Bs[BK][BN];

  float c_regs[TM * nbIterWaveM * TN * nbIterWaveN] = {0.0f};

  // Iteration over BK blocks.
  for (int kId = 0; kId < N; kId += BK) {
    __syncthreads();

    // We populate the Shared Memory with Ks row and columns
    for (int i = 0; i < nbReadsB; i++) {
      int index_x = BN * blockIdx.x + rBIdx;
      int index_y = rBIdy + i * strideReadB + kId;
      Bs[index_y % BK][index_x % BN] = b[N * index_y + index_x];
    }

    for (int i = 0; i < nbReadsA; i++) {
      int index_x = rAIdx + kId;
      int index_y = BM * blockIdx.y + rAIdy + i * strideReadA;
      As[(index_x % BK)][(index_y % BM)] = a[N * index_y + index_x];
    }

    __syncthreads();
    for (int k = 0; k < BK; k++) {
      // we cache A & B for the entire Wave tile
      for (int iterWave = 0; iterWave < nbIterWaveN; iterWave++) {
        for (int i = 0; i < TN; i++) {
          int index = waveIdx * WN + iterWave * SUBWN + TN * idxInWave + i;
          B_row[iterWave * TN + i] = Bs[k][index];
        }
      }

      for (int iterWave = 0; iterWave < nbIterWaveM; iterWave++) {
        for (int i = 0; i < TM; i++) {
          int index = waveIdy * WM + iterWave * SUBWM + TM * idyInWave + i;
          A_col[iterWave * TM + i] = As[k][index];
        }
      }

      // we accumulate to C_regs
      for (int iterWaveM = 0; iterWaveM < nbIterWaveM; iterWaveM++) {
        for (int iterWaveN = 0; iterWaveN < nbIterWaveN; iterWaveN++) {
          for (int yt = 0; yt < TM; yt++) {
            for (int xt = 0; xt < TN; xt++) {
              const int x = iterWaveN * TN + xt;
              const int y = iterWaveM * TM + yt;
              c_regs[y * TN * nbIterWaveN + x] += A_col[y] * B_row[x];
            }
          }
        }
      }
    }
  }

  for (int iterWaveM = 0; iterWaveM < nbIterWaveM; iterWaveM++) {
    for (int iterWaveN = 0; iterWaveN < nbIterWaveN; iterWaveN++) {
      int xOut = blockIdx.x * BN + waveIdx * WN + iterWaveN * SUBWN + TN * idxInWave;
      int yOut = blockIdx.y * BM + waveIdy * WM + iterWaveM * SUBWM + TM * idyInWave;
      for (int yt = 0; yt < TM; yt++) {
        for (int xt = 0; xt < TN; xt++) {
          int indexC = N * (yOut + yt) + xOut + xt;
          c[indexC] = beta * c[indexC] + alpha * c_regs[TN * nbIterWaveN * (iterWaveM * TM + yt) + (iterWaveN * TN + xt)];
        }
      }
    }
  }
}
