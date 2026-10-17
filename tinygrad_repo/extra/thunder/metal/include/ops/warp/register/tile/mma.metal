#pragma once // doneington

#include <metal_stdlib>
#include "../../../../types/types.metal"
#include "../../../../common/common.metal"
namespace mittens {
        
template <typename R, typename T, typename U, typename V,
          typename l1, typename l2, typename l3, typename l4>
METAL_FUNC static void mma_base(thread rt_base<R, l1>& d,
                                thread rt_base<T, l2>& a,
                                thread rt_base<U, l3>& b,
                                thread rt_base<V, l4>& c) {
    metal::simdgroup_multiply_accumulate(d.data, a.data, b.data, c.data);
}
    
template <typename R, typename T, typename U,
          typename l1, typename l2, typename l3>
METAL_FUNC static void mm_base(thread rt_base<R, l1>& d,
                                thread rt_base<T, l2>& a,
                                thread rt_base<U, l3>& b) {
    metal::simdgroup_multiply(d.data, a.data, b.data);
}

namespace meta {
template<typename R, typename T, typename U, int N, int K, int M>
static METAL_FUNC typename metal::enable_if<ducks::base_types::isT1Type<R>() && ducks::base_types::isT1Type<T>() && ducks::base_types::isT1Type<U>(), void>::type
mma_AB_unroll_inner(int k, int n, int m,
                    thread rt<R, N, M, ducks::rt_layout::row>* d,
                    thread rt<T, N, K, ducks::rt_layout::row>* a,
                    thread rt<U, K, M, ducks::rt_layout::row>* b) {
    mma_base(
         d->tiles[n][m],
         a->tiles[n][k],
         b->tiles[k][m],
         d->tiles[n][m]
     );
}


template<typename R, typename T, typename U, typename V, int N, int K, int M>
static METAL_FUNC typename metal::enable_if<ducks::base_types::isT1Type<R>() && ducks::base_types::isT1Type<T>() && ducks::base_types::isT1Type<U>() && ducks::base_types::isT1Type<V>(), void>::type
mma_AB_unroll(int n, int m,
              thread rt<R, N, M, ducks::rt_layout::row>* d,
              thread rt<T, N, K, ducks::rt_layout::row>* a,
              thread rt<U, K, M, ducks::rt_layout::row>* b,
              thread rt<V, N, M, ducks::rt_layout::row>* c) {
    mma_base(
        d->tiles[n][m],
        a->tiles[n][0],
        b->tiles[0][m],
        c->tiles[n][m]
    );
    meta::unroll_i_in_range<1, K/TILE_DIM, 1>::run(meta::mma_AB_unroll_inner<R, T, U, N, K, M>, n, m, d, a, b);
}

template<typename R, typename T, typename U, int N, int K, int M>
static METAL_FUNC typename metal::enable_if<ducks::base_types::isT1Type<R>() && ducks::base_types::isT1Type<T>() && ducks::base_types::isT1Type<U>(), void>::type
mm_AB_unroll(int n, int m,
             thread rt<R, N, M, ducks::rt_layout::row>* d,
             thread rt<T, N, K, ducks::rt_layout::row>* a,
             thread rt<U, K, M, ducks::rt_layout::row>* b) {
    mm_base(
        d->tiles[n][m],
        a->tiles[n][0],
        b->tiles[0][m]
    );
    meta::unroll_i_in_range<1, K/TILE_DIM, 1>::run(meta::mma_AB_unroll_inner<R, T, U, N, K, M>, n, m, d, a, b);
}
}

template<typename R, typename T, typename U, typename V, int N, int K, int M>
static METAL_FUNC typename metal::enable_if<ducks::base_types::isT1Type<R>() && ducks::base_types::isT1Type<T>() && ducks::base_types::isT1Type<U>() && ducks::base_types::isT1Type<V>(), void>::type
mma_AB(thread rt<R, N, M, ducks::rt_layout::row>& d,
       thread rt<T, N, K, ducks::rt_layout::row>& a,
       thread rt<U, K, M, ducks::rt_layout::row>& b,
       thread rt<V, N, M, ducks::rt_layout::row>& c) {
    meta::unroll_i_j_in_range<0, N/TILE_DIM, 1, 0, M/TILE_DIM, 1>::run(meta::mma_AB_unroll<R, T, U, V, N, K, M>, &d, &a, &b, &c);
}

template<typename R, typename T, typename U, int N, int K, int M>
static METAL_FUNC typename metal::enable_if<ducks::base_types::isT1Type<R>() && ducks::base_types::isT1Type<T>() && ducks::base_types::isT1Type<U>(), void>::type
mm_AB(thread rt<R, N, M, ducks::rt_layout::row>& d,
      thread rt<T, N, K, ducks::rt_layout::row>& a,
      thread rt<U, K, M, ducks::rt_layout::row>& b) {
    meta::unroll_i_j_in_range<0, N/TILE_DIM, 1, 0, M/TILE_DIM, 1>::run(meta::mm_AB_unroll<R, T, U, N, K, M>, &d, &a, &b);
}

namespace meta {
template<typename R, typename T, typename U, int N, int K, int M>
static METAL_FUNC typename metal::enable_if<ducks::base_types::isT1Type<R>() && ducks::base_types::isT1Type<T>() && ducks::base_types::isT1Type<U>(), void>::type
mma_ABt_unroll_inner(int k, int n, int m,
               thread rt<R, N, M, ducks::rt_layout::row>* d,
               thread rt<T, N, K, ducks::rt_layout::row>* a,
               thread rt<U, M, K, ducks::rt_layout::col>* b) {
    mma_base(
         d->tiles[n][m],
         a->tiles[n][k],
         b->tiles[m][k],
         d->tiles[n][m]
     );
}


template<typename R, typename T, typename U, typename V, int N, int K, int M>
static METAL_FUNC typename metal::enable_if<ducks::base_types::isT1Type<R>() && ducks::base_types::isT1Type<T>() && ducks::base_types::isT1Type<U>() && ducks::base_types::isT1Type<V>(), void>::type
mma_ABt_unroll(int n, int m,
               thread rt<R, N, M, ducks::rt_layout::row>* d,
               thread rt<T, N, K, ducks::rt_layout::row>* a,
               thread rt<U, M, K, ducks::rt_layout::col>* b,
               thread rt<V, N, M, ducks::rt_layout::row>* c) {
    mma_base(
        d->tiles[n][m],
        a->tiles[n][0],
        b->tiles[m][0],
        c->tiles[n][m]
    );
    meta::unroll_i_in_range<1, K/TILE_DIM, 1>::run(meta::mma_ABt_unroll_inner<R, T, U, N, K, M>, n, m, d, a, b);
}

template<typename R, typename T, typename U, int N, int K, int M>
static METAL_FUNC typename metal::enable_if<ducks::base_types::isT1Type<R>() && ducks::base_types::isT1Type<T>() && ducks::base_types::isT1Type<U>(), void>::type
mm_ABt_unroll(int n, int m,
               thread rt<R, N, M, ducks::rt_layout::row>* d,
               thread rt<T, N, K, ducks::rt_layout::row>* a,
               thread rt<U, M, K, ducks::rt_layout::col>* b) {
    mm_base(
        d->tiles[n][m],
        a->tiles[n][0],
        b->tiles[m][0]
    );
    meta::unroll_i_in_range<1, K/TILE_DIM, 1>::run(meta::mma_ABt_unroll_inner<R, T, U, N, K, M>, n, m, d, a, b);
}
}
    
template<typename R, typename T, typename U, typename V, int N, int K, int M>
static METAL_FUNC typename metal::enable_if<ducks::base_types::isT1Type<R>() && ducks::base_types::isT1Type<T>() && ducks::base_types::isT1Type<U>() && ducks::base_types::isT1Type<V>(), void>::type
mma_ABt(thread rt<R, N, M, ducks::rt_layout::row>& d,
       thread rt<T, N, K, ducks::rt_layout::row>& a,
       thread rt<U, M, K, ducks::rt_layout::col>& b,
       thread rt<V, N, M, ducks::rt_layout::row>& c) {
    meta::unroll_i_j_in_range<0, N/TILE_DIM, 1, 0, M/TILE_DIM, 1>::run(meta::mma_ABt_unroll<R, T, U, V, N, K, M>, &d, &a, &b, &c);
}

template<typename R, typename T, typename U, int N, int K, int M>
static METAL_FUNC typename metal::enable_if<ducks::base_types::isT1Type<R>() && ducks::base_types::isT1Type<T>() && ducks::base_types::isT1Type<U>(), void>::type
mm_ABt(thread rt<R, N, M, ducks::rt_layout::row>& d,
       thread rt<T, N, K, ducks::rt_layout::row>& a,
       thread rt<U, M, K, ducks::rt_layout::col>& b) {
    meta::unroll_i_j_in_range<0, N/TILE_DIM, 1, 0, M/TILE_DIM, 1>::run(meta::mm_ABt_unroll<R, T, U, N, K, M>, &d, &a, &b);
}

template<typename R, typename T, typename U, typename V, int N, int K, int M>
static METAL_FUNC typename metal::enable_if<ducks::base_types::isT1Type<R>() && ducks::base_types::isT1Type<T>() && ducks::base_types::isT1Type<U>() && ducks::base_types::isT1Type<V>(), void>::type
mma_AtB(thread rt<R, N, M, ducks::rt_layout::row>& d,
        thread rt<T, K, N, ducks::rt_layout::col>& a,
        thread rt<U, K, M, ducks::rt_layout::row>& b,
        thread rt<V, N, M, ducks::rt_layout::row>& c) {
    #pragma clang loop unroll(full)
    for (int n = 0; n < N / TILE_DIM; n++) {
        #pragma clang loop unroll(full)
        for (int m = 0; m < M / TILE_DIM; m++) {
            mma_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma clang loop unroll(full)
            for (int k = 1; k < K / TILE_DIM; k++) {
                mma_base(
                     d.tiles[n][m],
                     a.tiles[k][n],
                     b.tiles[k][m],
                     d.tiles[n][m]
                 );
            }
        }
    }
}


template<typename R, typename T, typename U, typename V, int N, int K, int M>
static METAL_FUNC typename metal::enable_if<ducks::base_types::isT1Type<R>() && ducks::base_types::isT1Type<T>() && ducks::base_types::isT1Type<U>() && ducks::base_types::isT1Type<V>(), void>::type
mma_AtBt(thread rt<R, N, M, ducks::rt_layout::row>& d,
         thread rt<T, K, N, ducks::rt_layout::col>& a,
         thread rt<U, M, K, ducks::rt_layout::col>& b,
         thread rt<V, N, M, ducks::rt_layout::row>& c) {
    #pragma clang loop unroll(full)
    for (int n = 0; n < N / TILE_DIM; n++) {
        #pragma clang loop unroll(full)
        for (int m = 0; m < M / TILE_DIM; m++) {
            mma_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma clang loop unroll(full)
            for (int k = 1; k < K / TILE_DIM; k++) {
                mma_base(
                     d.tiles[n][m],
                     a.tiles[k][n],
                     b.tiles[m][k],
                     d.tiles[n][m]
                 );
            }
        }
    }
}


    
}
