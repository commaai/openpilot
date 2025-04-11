//==-------- joint_matrix_bfloat16.cpp  - DPC++ joint_matrix----------- ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %clangxx -fsycl %s -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bfloat16 = sycl::ext::oneapi::bfloat16;

//#define SG_SZ 16
#define SG_SZ 8

#define TM 8
#define TN SG_SZ
//#define TK 16
#define TK 16

#define BF16_EPSILON 0.00781250

template <typename T, size_t NUM_ROWS, size_t NUM_COLS> struct big_matrix {
private:
  T *mat;

public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

template <typename T1, typename T2, size_t M, size_t N, size_t K>
void matrix_multiply(big_matrix<T1, M, N> &C, big_matrix<T2, M, K> &A, big_matrix<T2, K / 2, N * 2> &B) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<bfloat16, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<bfloat16, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<float, 2> bufC((float *)C.get_data(), range<2>(M, N));

  auto program = [&](handler &cgh) {
    auto accC = bufC.get_access<access::mode::read_write>(cgh);
    auto accA = bufA.get_access<access::mode::read_write>(cgh);
    auto accB = bufB.get_access<access::mode::read_write>(cgh);

    cgh.parallel_for<class imatrix>(
      nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
      [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]]
      {
        // The submatrix API has to be accessed by all the workitems in a
        // subgroup these functions will be called once by the subgroup no
        // code divergence between the workitems
        const auto global_idx = spmd_item.get_global_id(0);
        const auto global_idy = spmd_item.get_global_id(1);
        const auto sg_startx = global_idx - spmd_item.get_local_id(0);
        const auto sg_starty = global_idy - spmd_item.get_local_id(1);

        sub_group sg = spmd_item.get_sub_group();
        joint_matrix<sub_group, bfloat16, use::a, TM, TK, layout::row_major> sub_a;
        // For B, we assume B has been already VNNIed.
        joint_matrix<sub_group, bfloat16, use::b, TK, TN, ext::intel::experimental::matrix::layout::packed> sub_b;
        joint_matrix<sub_group, float, use::accumulator, TM, TN> sub_c;
        joint_matrix_load(sg, sub_c, accC.get_pointer() + (sg_startx * TM) * N + sg_starty / SG_SZ * TN, N, layout::row_major);

        for (int k = 0; k < K / TK; k += 1) { //
          joint_matrix_load(sg, sub_a, accA.get_pointer() + (sg_startx * TM) * K + k * TK, K);
          joint_matrix_load(sg, sub_b, accB.get_pointer() + (k * TK / 2) * (N * 2) + sg_starty / SG_SZ * TN * 2, N * 2);
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
        }
        joint_matrix_store(sg, sub_c, accC.get_pointer() + (sg_startx * TM) * N + sg_starty / SG_SZ * TN, N, layout::row_major);
      }); // parallel for
  };

  queue q;
  auto start = std::chrono::steady_clock::now();
  auto e = q.submit(program);
  auto submit = std::chrono::steady_clock::now();
  e.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "submit:  " << std::chrono::duration_cast<std::chrono::milliseconds>(submit - start).count() << " ms" << std::endl;
  std::cout << "compute: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - submit).count() << " ms" << std::endl;

  // ahh, freeing is slow
}

//#define SCALE 1024
//#define SCALE 64
#define SCALE 256
static constexpr size_t MATRIX_M = TM * SCALE;
static constexpr size_t MATRIX_N = TN * SCALE;
static constexpr size_t MATRIX_K = TK * SCALE;
bfloat16 A[MATRIX_M][MATRIX_K];
bfloat16 B[MATRIX_K / 2][MATRIX_N * 2];
float C[MATRIX_M][MATRIX_N];
float D[MATRIX_M][MATRIX_N];

float make_fp32(bfloat16 x) {
  unsigned int y = *((int *)&x);
  y = y << 16;
  float *res = reinterpret_cast<float *>(&y);
  return *res;
}

void matrix_multiply_ref(int *A_mem, int *B_mem, int *C_mem, int M, int N,
                         int K) {
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        // Because B was assumed VNNIed
        bfloat16 *va = (bfloat16 *)(A_mem + m * K + k);
        bfloat16 *vb = (bfloat16 *)(B_mem + k * N + n);
        float acc = *((float *)(C_mem + m * N + n));
        for (int i = 0; i < 2; i++) {
          acc += (make_fp32(va[i]) * make_fp32(vb[i]));
        }
        *((float *)(C_mem + m * N + n)) = acc;
      }
    }
}

int main() {
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_K; j++) {
      A[i][j] = bfloat16(1.0f * (i + j));
    }
  }
  for (int i = 0; i < MATRIX_K / 2; i++) {
    for (int j = 0; j < MATRIX_N * 2; j++) {
      B[i][j] = bfloat16(2.0f * i + 3.0f * j);
    }
  }
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      C[i][j] = 1.0;
      D[i][j] = 1.0;
    }
  }

  std::cout << "M" << MATRIX_M << "N" << MATRIX_N << "K" << MATRIX_K << std::endl;

  big_matrix<float, MATRIX_M, MATRIX_N> MC((float *)&C);
  big_matrix<float, MATRIX_M, MATRIX_N> MD((float *)&D);
  big_matrix<bfloat16, MATRIX_M, MATRIX_K> MA((bfloat16 *)&A);
  big_matrix<bfloat16, MATRIX_K / 2, MATRIX_N * 2> MB((bfloat16 *)&B);

  matrix_multiply(MC, MA, MB);

  /*start = std::chrono::steady_clock::now();
  matrix_multiply_ref((int32_t *)A, (int32_t *)B, (int32_t *)D, MATRIX_M, MATRIX_N, MATRIX_K / 2);
  end = std::chrono::steady_clock::now();
  std::cout << "Elapsed time in milliseconds (reference): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

  bool res = true;
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      if ((fabs(C[i][j]) - fabs(D[i][j])) > BF16_EPSILON)
        res = false;
    }
  }
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;*/

  return 0;
}

