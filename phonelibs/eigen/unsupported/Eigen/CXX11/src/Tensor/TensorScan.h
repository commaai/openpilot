// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Igor Babuschkin <igor@babuschk.in>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_SCAN_H
#define EIGEN_CXX11_TENSOR_TENSOR_SCAN_H

namespace Eigen {

namespace internal {

template <typename Op, typename XprType>
struct traits<TensorScanOp<Op, XprType> >
    : public traits<XprType> {
  typedef typename XprType::Scalar Scalar;
  typedef traits<XprType> XprTraits;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
  static const int NumDimensions = XprTraits::NumDimensions;
  static const int Layout = XprTraits::Layout;
};

template<typename Op, typename XprType>
struct eval<TensorScanOp<Op, XprType>, Eigen::Dense>
{
  typedef const TensorScanOp<Op, XprType>& type;
};

template<typename Op, typename XprType>
struct nested<TensorScanOp<Op, XprType>, 1,
            typename eval<TensorScanOp<Op, XprType> >::type>
{
  typedef TensorScanOp<Op, XprType> type;
};
} // end namespace internal

/** \class TensorScan
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor scan class.
  */
template <typename Op, typename XprType>
class TensorScanOp
    : public TensorBase<TensorScanOp<Op, XprType>, ReadOnlyAccessors> {
public:
  typedef typename Eigen::internal::traits<TensorScanOp>::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename Eigen::internal::nested<TensorScanOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorScanOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorScanOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorScanOp(
      const XprType& expr, const Index& axis, bool exclusive = false, const Op& op = Op())
      : m_expr(expr), m_axis(axis), m_accumulator(op), m_exclusive(exclusive) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  const Index axis() const { return m_axis; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  const XprType& expression() const { return m_expr; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  const Op accumulator() const { return m_accumulator; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  bool exclusive() const { return m_exclusive; }

protected:
  typename XprType::Nested m_expr;
  const Index m_axis;
  const Op m_accumulator;
  const bool m_exclusive;
};

template <typename Self, typename Reducer, typename Device>
struct ScanLauncher;

// Eval as rvalue
template <typename Op, typename ArgType, typename Device>
struct TensorEvaluator<const TensorScanOp<Op, ArgType>, Device> {

  typedef TensorScanOp<Op, ArgType> XprType;
  typedef typename XprType::Index Index;
  static const int NumDims = internal::array_size<typename TensorEvaluator<ArgType, Device>::Dimensions>::value;
  typedef DSizes<Index, NumDims> Dimensions;
  typedef typename internal::remove_const<typename XprType::Scalar>::type Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  typedef TensorEvaluator<const TensorScanOp<Op, ArgType>, Device> Self;

  enum {
    IsAligned = false,
    PacketAccess = (internal::unpacket_traits<PacketReturnType>::size > 1),
    BlockAccess = false,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = false,
    RawAccess = true
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op,
                                                        const Device& device)
      : m_impl(op.expression(), device),
        m_device(device),
        m_exclusive(op.exclusive()),
        m_accumulator(op.accumulator()),
        m_size(m_impl.dimensions()[op.axis()]),
        m_stride(1),
        m_output(NULL) {

    // Accumulating a scalar isn't supported.
    EIGEN_STATIC_ASSERT((NumDims > 0), YOU_MADE_A_PROGRAMMING_MISTAKE);
    eigen_assert(op.axis() >= 0 && op.axis() < NumDims);

    // Compute stride of scan axis
    const Dimensions& dims = m_impl.dimensions();
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int i = 0; i < op.axis(); ++i) {
        m_stride = m_stride * dims[i];
      }
    } else {
      for (int i = NumDims - 1; i > op.axis(); --i) {
        m_stride = m_stride * dims[i];
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const {
    return m_impl.dimensions();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Index& stride() const {
    return m_stride;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Index& size() const {
    return m_size;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Op& accumulator() const {
    return m_accumulator;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool exclusive() const {
    return m_exclusive;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const TensorEvaluator<ArgType, Device>& inner() const {
    return m_impl;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Device& device() const {
    return m_device;
  }

  EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar* data) {
    m_impl.evalSubExprsIfNeeded(NULL);
    ScanLauncher<Self, Op, Device> launcher;
    if (data) {
      launcher(*this, data);
      return false;
    }

    const Index total_size = internal::array_prod(dimensions());
    m_output = static_cast<CoeffReturnType*>(m_device.allocate(total_size * sizeof(Scalar)));
    launcher(*this, m_output);
    return true;
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC PacketReturnType packet(Index index) const {
    return internal::ploadt<PacketReturnType, LoadMode>(m_output + index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType* data() const
  {
    return m_output;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    return m_output[index];
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost costPerCoeff(bool) const {
    return TensorOpCost(sizeof(CoeffReturnType), 0, 0);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    if (m_output != NULL) {
      m_device.deallocate(m_output);
      m_output = NULL;
    }
    m_impl.cleanup();
  }

protected:
  TensorEvaluator<ArgType, Device> m_impl;
  const Device& m_device;
  const bool m_exclusive;
  Op m_accumulator;
  const Index m_size;
  Index m_stride;
  CoeffReturnType* m_output;
};

// CPU implementation of scan
// TODO(ibab) This single-threaded implementation should be parallelized,
// at least by running multiple scans at the same time.
template <typename Self, typename Reducer, typename Device>
struct ScanLauncher {
  void operator()(Self& self, typename Self::CoeffReturnType *data) {
    Index total_size = internal::array_prod(self.dimensions());

    // We fix the index along the scan axis to 0 and perform a
    // scan per remaining entry. The iteration is split into two nested
    // loops to avoid an integer division by keeping track of each idx1 and idx2.
    for (Index idx1 = 0; idx1 < total_size; idx1 += self.stride() * self.size()) {
      for (Index idx2 = 0; idx2 < self.stride(); idx2++) {
        // Calculate the starting offset for the scan
        Index offset = idx1 + idx2;

        // Compute the scan along the axis, starting at the calculated offset
        typename Self::CoeffReturnType accum = self.accumulator().initialize();
        for (Index idx3 = 0; idx3 < self.size(); idx3++) {
          Index curr = offset + idx3 * self.stride();

          if (self.exclusive()) {
            data[curr] = self.accumulator().finalize(accum);
            self.accumulator().reduce(self.inner().coeff(curr), &accum);
          } else {
            self.accumulator().reduce(self.inner().coeff(curr), &accum);
            data[curr] = self.accumulator().finalize(accum);
          }
        }
      }
    }
  }
};

#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)

// GPU implementation of scan
// TODO(ibab) This placeholder implementation performs multiple scans in
// parallel, but it would be better to use a parallel scan algorithm and
// optimize memory access.
template <typename Self, typename Reducer>
__global__ void ScanKernel(Self self, Index total_size, typename Self::CoeffReturnType* data) {
  // Compute offset as in the CPU version
  Index val = threadIdx.x + blockIdx.x * blockDim.x;
  Index offset = (val / self.stride()) * self.stride() * self.size() + val % self.stride();

  if (offset + (self.size() - 1) * self.stride() < total_size) {
    // Compute the scan along the axis, starting at the calculated offset
    typename Self::CoeffReturnType accum = self.accumulator().initialize();
    for (Index idx = 0; idx < self.size(); idx++) {
      Index curr = offset + idx * self.stride();
      if (self.exclusive()) {
        data[curr] = self.accumulator().finalize(accum);
        self.accumulator().reduce(self.inner().coeff(curr), &accum);
      } else {
        self.accumulator().reduce(self.inner().coeff(curr), &accum);
        data[curr] = self.accumulator().finalize(accum);
      }
    }
  }
  __syncthreads();

}

template <typename Self, typename Reducer>
struct ScanLauncher<Self, Reducer, GpuDevice> {
  void operator()(const Self& self, typename Self::CoeffReturnType* data) {
     Index total_size = internal::array_prod(self.dimensions());
     Index num_blocks = (total_size / self.size() + 63) / 64;
     Index block_size = 64;
     LAUNCH_CUDA_KERNEL((ScanKernel<Self, Reducer>), num_blocks, block_size, 0, self.device(), self, total_size, data);
  }
};
#endif  // EIGEN_USE_GPU && __CUDACC__

}  // end namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_SCAN_H
