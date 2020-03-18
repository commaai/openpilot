// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_BLOCKING_H
#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_BLOCKING_H


namespace Eigen {
namespace internal {

enum {
  ShardByRow = 0,
  ShardByCol = 1
};


// Default Blocking Strategy
template <typename LhsMapper, typename RhsMapper, typename Index, int ShardingType=ShardByCol>
class TensorContractionBlocking {
 public:

  typedef typename LhsMapper::Scalar LhsScalar;
  typedef typename RhsMapper::Scalar RhsScalar;

  EIGEN_DEVICE_FUNC TensorContractionBlocking(Index k, Index m, Index n, Index num_threads = 1) :
      kc_(k), mc_(m), nc_(n)
  {
    if (ShardingType == ShardByCol) {
      computeProductBlockingSizes<LhsScalar, RhsScalar, 1>(kc_, mc_, nc_, num_threads);
    }
    else {
      computeProductBlockingSizes<LhsScalar, RhsScalar, 1>(kc_, nc_, mc_, num_threads);
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Index kc() const { return kc_; }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Index mc() const { return mc_; }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Index nc() const { return nc_; }

 private:
  Index kc_;
  Index mc_;
  Index nc_;
};


} // end namespace internal
} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_BLOCKING_H
