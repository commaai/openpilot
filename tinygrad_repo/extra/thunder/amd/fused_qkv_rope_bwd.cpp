#include "kittens.cuh"

using namespace kittens;

#ifndef ATTN_B
#define ATTN_B 2
#endif
#ifndef ATTN_N
#define ATTN_N 8192
#endif
#ifndef ATTN_H
#define ATTN_H 32
#endif
#ifndef ATTN_H_KV
#define ATTN_H_KV 8
#endif
#ifndef ATTN_D
#define ATTN_D 128
#endif
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif
constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV;
constexpr int HALF_D = ATTN_D / 2;
constexpr int PACKED_H = ATTN_H_KV * (GROUP_SIZE + 2);
constexpr int HEADS_PER_WG = ATTN_D == 128 && GROUP_SIZE % 2 == 0 ? 2 : 1;
constexpr int KV_PARTIALS = GROUP_SIZE / HEADS_PER_WG;
constexpr int NUM_WARPS = 4;
constexpr int TILE_N = 16;

template<typename T> using grad_tile = rt<T, TILE_N, ATTN_D, row_l, rt_16x32_s>;

template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ __forceinline__ void load_fa_shuffled(RT &dst, const GL &src, const COORD &idx) {
  using U = typename GL::dtype;
  using U2 = base_types::packing<U>::packed_type;
  U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
  const int row_stride = src.template stride<axis>();
  const int lane = kittens::laneid();
  const int tile_row_stride = row_stride * dst.base_tile_rows;
  const int tile_stride = dst.base_tile_rows * dst.base_tile_cols;
  const uint32_t buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(U);
  const buffer_resource br = make_buffer_resource(reinterpret_cast<uintptr_t>(src_ptr), buffer_size, 0x00020000);

  #pragma unroll
  for (int i = 0; i < dst.height; i++) {
    #pragma unroll
    for (int j = 0; j < dst.width; j++) {
      const float4 loaded = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(
        std::bit_cast<i32x4>(br), (i * tile_row_stride + j * tile_stride + lane * 8) * sizeof(U), 0, 0));
      const U2 *packed = reinterpret_cast<const U2*>(&loaded);
      #pragma unroll
      for (int k = 0; k < dst.packed_per_base_tile; k++) dst.tiles[i][j].data[k] = packed[k];
    }
  }
}

template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ __forceinline__ void store_fa_shuffled(const GL &dst, const RT &src, const COORD &idx) {
  using U = typename GL::dtype;
  U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
  const int row_stride = dst.template stride<axis>();
  const int lane = kittens::laneid();
  const int row_offset = (lane % 4) * 4;
  const int col_offset = ((lane / 32) * 16) + (((lane % 32) / 16) * 2) + (((lane % 16) / 4) * 4);
  const uint32_t buffer_size = dst.batch() * dst.depth() * dst.rows() * dst.cols() * sizeof(U);
  const buffer_resource br = make_buffer_resource(reinterpret_cast<uintptr_t>(dst_ptr), buffer_size, 0x00020000);

  #pragma unroll
  for (int i = 0; i < src.height; i++) {
    const int row = src.base_tile_rows * i + row_offset;
    #pragma unroll
    for (int j = 0; j < src.width; j++) {
      const int col = src.base_tile_cols * j + col_offset;
      #pragma unroll
      for (int k = 0; k < src.packed_per_base_tile; k++) llvm_amdgcn_raw_buffer_store_b32(
        *reinterpret_cast<const uint32_t*>(&src.tiles[i][j].data[k]), std::bit_cast<i32x4>(br),
        ((row + k) * row_stride + col) * sizeof(U), 0, 0);
    }
  }
}

template<ducks::rt::row_layout RT>
__device__ __forceinline__ void inverse_rope_fa(RT &tile, const bf16_2 *freqs, const int n_base) {
  const int lane = kittens::laneid();
  const int row_offset = (lane % 4) * 4;
  const int col_offset = ((lane / 32) * 16) + (((lane % 32) / 16) * 2) + (((lane % 16) / 4) * 4);
  #pragma unroll
  for (int i = 0; i < tile.height; i++) {
    #pragma unroll
    for (int j = 0; j < tile.width; j++) {
      const int col = tile.base_tile_cols * j + col_offset;
      #pragma unroll
      for (int k = 0; k < tile.packed_per_base_tile; k++) {
        const int row = tile.base_tile_rows * i + row_offset + k;
        const float2 cs = __bfloat1622float2(freqs[(n_base + row) * HALF_D + col / 2]);
        const float2 g = __bfloat1622float2(tile.tiles[i][j].data[k]);
        tile.tiles[i][j].data[k] = __float22bfloat162_rn(make_float2(g.x * cs.x + g.y * cs.y, -g.x * cs.y + g.y * cs.x));
      }
    }
  }
}

template<ducks::rt::row_layout RT>
__device__ __forceinline__ void inverse_rope(RT &tile, const bf16_2 *freqs, const int n_base) {
  const int lane = kittens::laneid();
  #pragma unroll
  for (int i = 0; i < tile.height; i++) {
    const int row = tile.base_tile_rows * i + lane % tile.base_tile_rows;
    #pragma unroll
    for (int j = 0; j < tile.width; j++) {
      #pragma unroll
      for (int k = 0; k < tile.packed_per_base_tile; k++) {
        const int col = tile.base_tile_cols * j + tile.base_tile_stride * (lane / tile.base_tile_rows) + 2 * k;
        const float2 cs = __bfloat1622float2(freqs[(n_base + row) * HALF_D + col / 2]);
        const float2 g = __bfloat1622float2(tile.tiles[i][j].data[k]);
        tile.tiles[i][j].data[k] = __float22bfloat162_rn(make_float2(g.x * cs.x + g.y * cs.y, -g.x * cs.y + g.y * cs.x));
      }
    }
  }
}

extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK) void
fused_qkv_rope_backward(
    bf16*       __restrict__ dxqkv,
    const bf16* __restrict__ dq,
    const bf16* __restrict__ dk,
    const bf16* __restrict__ dv,
    const bf16* __restrict__ freqs_cis) {
  gl<bf16, -1, -1, -1, -1> out{dxqkv, ATTN_B, ATTN_N, PACKED_H, ATTN_D};
  gl<bf16, -1, -1, -1, -1> dqg{const_cast<bf16*>(dq), ATTN_B, ATTN_H, ATTN_N, ATTN_D};
  gl<bf16, -1, -1, -1, -1> dkg{const_cast<bf16*>(dk), ATTN_B * KV_PARTIALS, ATTN_N, ATTN_H_KV, ATTN_D};
  gl<bf16, -1, -1, -1, -1> dvg{const_cast<bf16*>(dv), ATTN_B * KV_PARTIALS, ATTN_N, ATTN_H_KV, ATTN_D};
  const int b = blockIdx.x, n_tile = blockIdx.y * NUM_WARPS + kittens::warpid(), n_base = n_tile * TILE_N;
  const int field = blockIdx.z;

  if (field < ATTN_H) {
    grad_tile<bf16> tile;
    load_fa_shuffled<2>(tile, dqg, {b, field, n_tile, 0});
    inverse_rope_fa(tile, reinterpret_cast<const bf16_2*>(freqs_cis), n_base);
    const int out_head = (field / GROUP_SIZE) * (GROUP_SIZE + 2) + field % GROUP_SIZE;
    store_fa_shuffled<1>(out, tile, {b, n_tile, out_head, 0});
  } else {
    const bool is_k = field < ATTN_H + ATTN_H_KV;
    const int kvh = field - ATTN_H - (is_k ? 0 : ATTN_H_KV);
    const auto &src = is_k ? dkg : dvg;
    grad_tile<bf16> partial, tile;
    grad_tile<float> partial_f, sum;
    zero(sum);
    #pragma unroll
    for (int p = 0; p < KV_PARTIALS; p++) {
      load<1>(partial, src, {b * KV_PARTIALS + p, n_tile, kvh, 0});
      copy(partial_f, partial);
      add(sum, sum, partial_f);
    }
    copy(tile, sum);
    if (is_k) inverse_rope(tile, reinterpret_cast<const bf16_2*>(freqs_cis), n_base);
    const int out_head = kvh * (GROUP_SIZE + 2) + GROUP_SIZE + !is_k;
    store<1>(out, tile, {b, n_tile, out_head, 0});
  }
}
