// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_EXECUTOR_H
#define EIGEN_CXX11_TENSOR_TENSOR_EXECUTOR_H

namespace Eigen {

/** \class TensorExecutor
  * \ingroup CXX11_Tensor_Module
  *
  * \brief The tensor executor class.
  *
  * This class is responsible for launch the evaluation of the expression on
  * the specified computing device.
  */
namespace internal {

// Default strategy: the expression is evaluated with a single cpu thread.
template<typename Expression, typename Device, bool Vectorizable>
class TensorExecutor
{
 public:
  typedef typename Expression::Index Index;
  EIGEN_DEVICE_FUNC
  static inline void run(const Expression& expr, const Device& device = Device())
  {
    TensorEvaluator<Expression, Device> evaluator(expr, device);
    const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
    if (needs_assign)
    {
      const Index size = array_prod(evaluator.dimensions());
      for (Index i = 0; i < size; ++i) {
        evaluator.evalScalar(i);
      }
    }
    evaluator.cleanup();
  }
};


template<typename Expression>
class TensorExecutor<Expression, DefaultDevice, true>
{
 public:
  typedef typename Expression::Index Index;
  EIGEN_DEVICE_FUNC
  static inline void run(const Expression& expr, const DefaultDevice& device = DefaultDevice())
  {
    TensorEvaluator<Expression, DefaultDevice> evaluator(expr, device);
    const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
    if (needs_assign)
    {
      const Index size = array_prod(evaluator.dimensions());
      const int PacketSize = unpacket_traits<typename TensorEvaluator<Expression, DefaultDevice>::PacketReturnType>::size;
      // Give the compiler a strong hint to unroll the loop. But don't insist
      // on unrolling, because if the function is expensive the compiler should not
      // unroll the loop at the expense of inlining.
      const Index UnrolledSize = (size / (4 * PacketSize)) * 4 * PacketSize;
      for (Index i = 0; i < UnrolledSize; i += 4*PacketSize) {
        for (Index j = 0; j < 4; j++) {
          evaluator.evalPacket(i + j * PacketSize);
        }
      }
      const Index VectorizedSize = (size / PacketSize) * PacketSize;
      for (Index i = UnrolledSize; i < VectorizedSize; i += PacketSize) {
        evaluator.evalPacket(i);
      }
      for (Index i = VectorizedSize; i < size; ++i) {
        evaluator.evalScalar(i);
      }
    }
    evaluator.cleanup();
  }
};



// Multicore strategy: the index space is partitioned and each partition is executed on a single core
#ifdef EIGEN_USE_THREADS
template <typename Evaluator, typename Index, bool Vectorizable>
struct EvalRange {
  static void run(Evaluator* evaluator_in, const Index first, const Index last) {
    Evaluator evaluator = *evaluator_in;
    eigen_assert(last >= first);
    for (Index i = first; i < last; ++i) {
      evaluator.evalScalar(i);
    }
  }

  static Index alignBlockSize(Index size) {
    return size;
  }
};

template <typename Evaluator, typename Index>
struct EvalRange<Evaluator, Index, true> {
  static const int PacketSize = unpacket_traits<typename Evaluator::PacketReturnType>::size;

  static void run(Evaluator* evaluator_in, const Index first, const Index last) {
    Evaluator evaluator = *evaluator_in;
    eigen_assert(last >= first);
    Index i = first;
    if (last - first >= PacketSize) {
      eigen_assert(first % PacketSize == 0);
      Index last_chunk_offset = last - 4 * PacketSize;
      // Give the compiler a strong hint to unroll the loop. But don't insist
      // on unrolling, because if the function is expensive the compiler should not
      // unroll the loop at the expense of inlining.
      for (; i <= last_chunk_offset; i += 4*PacketSize) {
        for (Index j = 0; j < 4; j++) {
          evaluator.evalPacket(i + j * PacketSize);
        }
      }
      last_chunk_offset = last - PacketSize;
      for (; i <= last_chunk_offset; i += PacketSize) {
        evaluator.evalPacket(i);
      }
    }
    for (; i < last; ++i) {
      evaluator.evalScalar(i);
    }
  }

  static Index alignBlockSize(Index size) {
    // Align block size to packet size and account for unrolling in run above.
    if (size >= 16 * PacketSize) {
      return (size + 4 * PacketSize - 1) & ~(4 * PacketSize - 1);
    }
    // Aligning to 4 * PacketSize would increase block size by more than 25%.
    return (size + PacketSize - 1) & ~(PacketSize - 1);
  }
};

template <typename Expression, bool Vectorizable>
class TensorExecutor<Expression, ThreadPoolDevice, Vectorizable> {
 public:
  typedef typename Expression::Index Index;
  static inline void run(const Expression& expr, const ThreadPoolDevice& device)
  {
    typedef TensorEvaluator<Expression, ThreadPoolDevice> Evaluator;
    Evaluator evaluator(expr, device);
    const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
    if (needs_assign)
    {
      const Index size = array_prod(evaluator.dimensions());
#if !defined(EIGEN_USE_SIMPLE_THREAD_POOL)
      device.parallelFor(size, evaluator.costPerCoeff(Vectorizable),
                         EvalRange<Evaluator, Index, Vectorizable>::alignBlockSize,
                         [&evaluator](Index first, Index last) {
                           EvalRange<Evaluator, Index, Vectorizable>::run(&evaluator, first, last);
                         });
#else
      size_t num_threads = device.numThreads();
      if (num_threads > 1) {
        num_threads = TensorCostModel<ThreadPoolDevice>::numThreads(
            size, evaluator.costPerCoeff(Vectorizable), num_threads);
      }
      if (num_threads == 1) {
        EvalRange<Evaluator, Index, Vectorizable>::run(&evaluator, 0, size);
      } else {
        const Index PacketSize = Vectorizable ? unpacket_traits<typename Evaluator::PacketReturnType>::size : 1;
        Index blocksz = std::ceil<Index>(static_cast<float>(size)/num_threads) + PacketSize - 1;
        const Index blocksize = numext::maxi<Index>(PacketSize, (blocksz - (blocksz % PacketSize)));
        const Index numblocks = size / blocksize;

        Barrier barrier(numblocks);
        for (int i = 0; i < numblocks; ++i) {
          device.enqueue_with_barrier(
              &barrier, &EvalRange<Evaluator, Index, Vectorizable>::run,
              &evaluator, i * blocksize, (i + 1) * blocksize);
        }
        if (numblocks * blocksize < size) {
          EvalRange<Evaluator, Index, Vectorizable>::run(
              &evaluator, numblocks * blocksize, size);
        }
        barrier.Wait();
      }
#endif  // defined(!EIGEN_USE_SIMPLE_THREAD_POOL)
    }
    evaluator.cleanup();
  }
};
#endif  // EIGEN_USE_THREADS


// GPU: the evaluation of the expression is offloaded to a GPU.
#if defined(EIGEN_USE_GPU)

template <typename Expression, bool Vectorizable>
class TensorExecutor<Expression, GpuDevice, Vectorizable> {
 public:
  typedef typename Expression::Index Index;
  static void run(const Expression& expr, const GpuDevice& device);
};


#if defined(__CUDACC__)
template <typename Evaluator, typename Index, bool Vectorizable>
struct EigenMetaKernelEval {
  static __device__ EIGEN_ALWAYS_INLINE
  void run(Evaluator& eval, Index first, Index last, Index step_size) {
    for (Index i = first; i < last; i += step_size) {
      eval.evalScalar(i);
    }
  }
};

template <typename Evaluator, typename Index>
struct EigenMetaKernelEval<Evaluator, Index, true> {
  static __device__ EIGEN_ALWAYS_INLINE
  void run(Evaluator& eval, Index first, Index last, Index step_size) {
    const Index PacketSize = unpacket_traits<typename Evaluator::PacketReturnType>::size;
    const Index vectorized_size = (last / PacketSize) * PacketSize;
    const Index vectorized_step_size = step_size * PacketSize;

    // Use the vector path
    for (Index i = first * PacketSize; i < vectorized_size;
         i += vectorized_step_size) {
      eval.evalPacket(i);
    }
    for (Index i = vectorized_size + first; i < last; i += step_size) {
      eval.evalScalar(i);
    }
  }
};

template <typename Evaluator, typename Index>
__global__ void
__launch_bounds__(1024)
EigenMetaKernel(Evaluator eval, Index size) {

  const Index first_index = blockIdx.x * blockDim.x + threadIdx.x;
  const Index step_size = blockDim.x * gridDim.x;

  const bool vectorizable = Evaluator::PacketAccess & Evaluator::IsAligned;
  EigenMetaKernelEval<Evaluator, Index, vectorizable>::run(eval, first_index, size, step_size);
}

/*static*/
template <typename Expression, bool Vectorizable>
inline void TensorExecutor<Expression, GpuDevice, Vectorizable>::run(
    const Expression& expr, const GpuDevice& device) {
  TensorEvaluator<Expression, GpuDevice> evaluator(expr, device);
  const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
  if (needs_assign) {
    const int block_size = device.maxCudaThreadsPerBlock();
    const int max_blocks = device.getNumCudaMultiProcessors() *
                           device.maxCudaThreadsPerMultiProcessor() / block_size;
    const Index size = array_prod(evaluator.dimensions());
    // Create a least one block to ensure we won't crash when tensorflow calls with tensors of size 0.
    const int num_blocks = numext::maxi<int>(numext::mini<int>(max_blocks, divup<int>(size, block_size)), 1);

    LAUNCH_CUDA_KERNEL(
        (EigenMetaKernel<TensorEvaluator<Expression, GpuDevice>, Index>),
        num_blocks, block_size, 0, device, evaluator, size);
  }
  evaluator.cleanup();
}

#endif  // __CUDACC__
#endif  // EIGEN_USE_GPU

// SYCL Executor policy
#ifdef EIGEN_USE_SYCL

template <typename Expression, bool Vectorizable>
class TensorExecutor<Expression, SyclDevice, Vectorizable> {
public:
  static inline void run(const Expression &expr, const SyclDevice &device) {
    // call TensorSYCL module
    TensorSycl::run(expr, device);
  }
};

#endif

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_EXECUTOR_H
