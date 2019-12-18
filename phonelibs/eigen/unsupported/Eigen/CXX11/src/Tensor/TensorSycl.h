// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: eigen@codeplay.com
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// General include header of SYCL target for Tensor Module
#ifndef UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_H
#define UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_H

#ifdef EIGEN_USE_SYCL

// global pointer to set different attribute state for a class
template <class T>
struct MakeGlobalPointer {
  typedef typename cl::sycl::global_ptr<T>::pointer_t Type;
};

// global pointer to set different attribute state for a class
template <class T>
struct MakeLocalPointer {
  typedef typename cl::sycl::local_ptr<T>::pointer_t Type;
};


namespace Eigen {
namespace TensorSycl {
namespace internal {

/// This struct is used for special expression nodes with no operations (for example assign and selectOP).
  struct NoOP;

template<bool IsConst, typename T> struct GetType{
  typedef const T Type;
};
template<typename T> struct GetType<false, T>{
  typedef T Type;
};

}
}
}

// tuple construction
#include "TensorSyclTuple.h"

// counting number of leaf at compile time
#include "TensorSyclLeafCount.h"

// The index PlaceHolder takes the actual expression and replaces the actual
// data on it with the place holder. It uses the same pre-order expression tree
// traverse as the leaf count in order to give the right access number to each
// node in the expression
#include "TensorSyclPlaceHolderExpr.h"

// creation of an accessor tuple from a tuple of SYCL buffers
#include "TensorSyclExtractAccessor.h"

// this is used to change the address space type in tensor map for GPU
#include "TensorSyclConvertToDeviceExpression.h"

// this is used to extract the functors
#include "TensorSyclExtractFunctors.h"

// this is used to create tensormap on the device
// this is used to construct the expression on the device
#include "TensorSyclExprConstructor.h"

/// this is used for extracting tensor reduction
#include "TensorReductionSycl.h"

// kernel execution using fusion
#include "TensorSyclRun.h"

#endif  // end of EIGEN_USE_SYCL
#endif  // UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_H
