#pragma once

#include "../../../../../common/common.cuh"
#include "../../../../../types/types.cuh"

namespace kittens {
namespace detail {
namespace wgmma {

// templated wrapper for PTX
template<typename T_D, typename T_AB, int cols, int trans_a, int trans_b, int inv=1>
struct base {
    template<int scale_b=1> __device__ static inline void rt_st(
        rt<T_D, 16, cols, ducks::rt_layout::row> &dst,
        const rt<T_AB, 16, cols, ducks::rt_layout::row> & a_rt,
        const uint64_t b_st_desc,
        int scale_d = 1
    );
    template<int scale_b=1> __device__ static inline void st_st(
        rt<T_D, 16, cols, ducks::rt_layout::row> &dst,
        const uint64_t a_st_desc,
        const uint64_t b_st_desc,
        int scale_d = 1
    );
};

// all the ptx's
#include "64x16.impl"
#include "64x32.impl"
#include "64x48.impl"
#include "64x64.impl"
#include "64x80.impl"
#include "64x96.impl"
#include "64x112.impl"
#include "64x128.impl"
#include "64x144.impl"
#include "64x160.impl"
#include "64x176.impl"
#include "64x192.impl"
#include "64x208.impl"
#include "64x224.impl"
#include "64x240.impl"
#include "64x256.impl"

} // namespace wgmma
} // namespace detail
} // namespace kittens