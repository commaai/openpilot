// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_THREAD_POOL_H
#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_THREAD_POOL_H

// evaluator for thread pool device
#ifdef EIGEN_USE_THREADS

namespace Eigen {

#ifdef EIGEN_USE_SIMPLE_THREAD_POOL
namespace internal {

template<typename LhsScalar, typename LhsMapper, typename Index>
struct packLhsArg {
  LhsScalar* blockA;
  const LhsMapper& lhs;
  const Index m_start;
  const Index k_start;
  const Index mc;
  const Index kc;
};

template<typename LhsScalar, typename RhsScalar, typename RhsMapper, typename OutputMapper, typename Index>
struct packRhsAndKernelArg {
  const MaxSizeVector<LhsScalar*>* blockAs;
  RhsScalar* blockB;
  const RhsMapper& rhs;
  OutputMapper& output;
  const Index m;
  const Index k;
  const Index n;
  const Index mc;
  const Index kc;
  const Index nc;
  const Index num_threads;
  const Index num_blockAs;
  const Index max_m;
  const Index k_block_idx;
  const Index m_block_idx;
  const Index n_block_idx;
  const Index m_blocks;
  const Index n_blocks;
  MaxSizeVector<Notification*>* kernel_notifications;
  const MaxSizeVector<Notification*>* lhs_notifications;
  const bool need_to_pack;
};

}  // end namespace internal
#endif  // EIGEN_USE_SIMPLE_THREAD_POOL

template<typename Indices, typename LeftArgType, typename RightArgType>
struct TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, ThreadPoolDevice> :
    public TensorContractionEvaluatorBase<TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, ThreadPoolDevice> > {

  typedef ThreadPoolDevice Device;

  typedef TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, Device> Self;
  typedef TensorContractionEvaluatorBase<Self> Base;

  typedef TensorContractionOp<Indices, LeftArgType, RightArgType> XprType;
  typedef typename internal::remove_const<typename XprType::Scalar>::type Scalar;
  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;

  enum {
    Layout = TensorEvaluator<LeftArgType, Device>::Layout,
  };

  // Most of the code is assuming that both input tensors are ColMajor. If the
  // inputs are RowMajor, we will "cheat" by swapping the LHS and RHS:
  // If we want to compute A * B = C, where A is LHS and B is RHS, the code
  // will pretend B is LHS and A is RHS.
  typedef typename internal::conditional<
    static_cast<int>(Layout) == static_cast<int>(ColMajor), LeftArgType, RightArgType>::type EvalLeftArgType;
  typedef typename internal::conditional<
    static_cast<int>(Layout) == static_cast<int>(ColMajor), RightArgType, LeftArgType>::type EvalRightArgType;

  static const int LDims =
      internal::array_size<typename TensorEvaluator<EvalLeftArgType, Device>::Dimensions>::value;
  static const int RDims =
      internal::array_size<typename TensorEvaluator<EvalRightArgType, Device>::Dimensions>::value;
  static const int ContractDims = internal::array_size<Indices>::value;

  typedef array<Index, LDims> left_dim_mapper_t;
  typedef array<Index, RDims> right_dim_mapper_t;

  typedef array<Index, ContractDims> contract_t;
  typedef array<Index, LDims - ContractDims> left_nocontract_t;
  typedef array<Index, RDims - ContractDims> right_nocontract_t;

  static const int NumDims = LDims + RDims - 2 * ContractDims;

  typedef DSizes<Index, NumDims> Dimensions;

  // typedefs needed in evalTo
  typedef typename internal::remove_const<typename EvalLeftArgType::Scalar>::type LhsScalar;
  typedef typename internal::remove_const<typename EvalRightArgType::Scalar>::type RhsScalar;
  typedef typename internal::gebp_traits<LhsScalar, RhsScalar> Traits;

  typedef TensorEvaluator<EvalLeftArgType, Device> LeftEvaluator;
  typedef TensorEvaluator<EvalRightArgType, Device> RightEvaluator;

  TensorEvaluator(const XprType& op, const Device& device) :
      Base(op, device) {}

#ifndef EIGEN_USE_SIMPLE_THREAD_POOL
  template <bool lhs_inner_dim_contiguous, bool rhs_inner_dim_contiguous,
            bool rhs_inner_dim_reordered, int Alignment>
  void evalProduct(Scalar* buffer) const {
    typedef
        typename internal::remove_const<typename EvalLeftArgType::Scalar>::type
            LhsScalar;
    typedef
        typename internal::remove_const<typename EvalRightArgType::Scalar>::type
            RhsScalar;
    typedef typename internal::gebp_traits<LhsScalar, RhsScalar> Traits;
    typedef TensorEvaluator<EvalLeftArgType, Device> LeftEvaluator;
    typedef TensorEvaluator<EvalRightArgType, Device> RightEvaluator;
    typedef internal::TensorContractionInputMapper<
        LhsScalar, Index, internal::Lhs, LeftEvaluator, left_nocontract_t,
        contract_t, internal::packet_traits<LhsScalar>::size,
        lhs_inner_dim_contiguous, false, Unaligned>
        LhsMapper;
    typedef internal::TensorContractionInputMapper<
        RhsScalar, Index, internal::Rhs, RightEvaluator, right_nocontract_t,
        contract_t, internal::packet_traits<RhsScalar>::size,
        rhs_inner_dim_contiguous, rhs_inner_dim_reordered, Unaligned>
        RhsMapper;
    typedef internal::blas_data_mapper<Scalar, Index, ColMajor> OutputMapper;
    typedef internal::gemm_pack_lhs<LhsScalar, Index,
                                    typename LhsMapper::SubMapper, Traits::mr,
                                    Traits::LhsProgress, ColMajor>
        LhsPacker;
    typedef internal::gemm_pack_rhs<
        RhsScalar, Index, typename RhsMapper::SubMapper, Traits::nr, ColMajor>
        RhsPacker;
    typedef internal::gebp_kernel<LhsScalar, RhsScalar, Index, OutputMapper,
                                  Traits::mr, Traits::nr, false, false>
        GebpKernel;

    const Index m = this->m_i_size;
    const Index n = this->m_j_size;
    const Index k = this->m_k_size;
    if (m == 0 || n == 0 || k == 0) return;

    // Compute a set of algorithm parameters:
    // - kernel block sizes (bm, bn, bk)
    // - task grain sizes (number of kernels executed per task: gm, gn)
    // - number of threads
    // - sharding by row/column
    // - parallel packing or first lhs then rhs
    // and some derived parameters:
    // - number of tasks (nm, nn, nk)
    // - number of kernels (nm0, nn0)
    // Unfortunately, all these parameters are tightly interdependent.
    // So in some cases we first compute approximate values, then compute other
    // values based on these approximations and then refine the approximations.

    // There are lots of heuristics here. There is some reasoning behind them,
    // but ultimately they are just tuned on contraction benchmarks for
    // different input configurations, thread counts and instruction sets.
    // So feel free to question any of them.

    // Compute whether we want to shard by row or by column.
    // This is a first approximation, it will be refined later. Since we don't
    // know number of threads yet we use 2, because what's we are most
    // interested in at this point is whether it makes sense to use
    // parallelization at all or not.
    bool shard_by_col = shardByCol(m, n, 2);

    // First approximation of kernel blocking sizes.
    // Again, we don't know number of threads yet, so we use 2.
    Index bm, bn, bk;
    if (shard_by_col) {
      internal::TensorContractionBlocking<LhsMapper, RhsMapper, Index,
                                          internal::ShardByCol>
          blocking(k, m, n, 2);
      bm = blocking.mc();
      bn = blocking.nc();
      bk = blocking.kc();
    } else {
      internal::TensorContractionBlocking<LhsMapper, RhsMapper, Index,
                                          internal::ShardByRow>
          blocking(k, m, n, 2);
      bm = blocking.mc();
      bn = blocking.nc();
      bk = blocking.kc();
    }

    // Compute optimal number of threads.
    // Note: we use bk instead of k here because we are interested in amount of
    // _parallelizable_ computations, and computations are not parallelizable
    // across k dimension.
    const TensorOpCost cost =
        contractionCost(m, n, bm, bn, bk, shard_by_col, false);
    int num_threads = TensorCostModel<ThreadPoolDevice>::numThreads(
        static_cast<double>(n) * m, cost, this->m_device.numThreads());

    // TODO(dvyukov): this is a stop-gap to prevent regressions while the cost
    // model is not tuned. Remove this when the cost model is tuned.
    if (n == 1) num_threads = 1;

    if (num_threads == 1) {
      // The single-threaded algorithm should be faster in this case.
      if (n == 1)
        this->template evalGemv<lhs_inner_dim_contiguous,
                                rhs_inner_dim_contiguous,
                                rhs_inner_dim_reordered, Alignment>(buffer);
      else
        this->template evalGemm<lhs_inner_dim_contiguous,
                                rhs_inner_dim_contiguous,
                                rhs_inner_dim_reordered, Alignment>(buffer);
      return;
    }

    // Now that we know number of threads, recalculate sharding and blocking.
    shard_by_col = shardByCol(m, n, num_threads);
    if (shard_by_col) {
      internal::TensorContractionBlocking<LhsMapper, RhsMapper, Index,
                                          internal::ShardByCol>
          blocking(k, m, n, num_threads);
      bm = blocking.mc();
      bn = blocking.nc();
      bk = blocking.kc();
    } else {
      internal::TensorContractionBlocking<LhsMapper, RhsMapper, Index,
                                          internal::ShardByRow>
          blocking(k, m, n, num_threads);
      bm = blocking.mc();
      bn = blocking.nc();
      bk = blocking.kc();
    }

    // Number of kernels for each dimension.
    Index nm0 = divup(m, bm);
    Index nn0 = divup(n, bn);
    Index nk = divup(k, bk);

    // Calculate task grain size (number of kernels executed per task).
    // This task size coarsening serves two purposes:
    // 1. It reduces per-task overheads including synchronization overheads.
    // 2. It allows to use caches better (reuse the same packed rhs in several
    // consecutive kernels).
    Index gm = 1;
    Index gn = 1;
    // If we are sharding by column, then we prefer to reduce rows first.
    if (shard_by_col) {
      gm = coarsenM(m, n, bm, bn, bk, gn, num_threads, shard_by_col);
      gn = coarsenN(m, n, bm, bn, bk, gm, num_threads, shard_by_col);
    } else {
      gn = coarsenN(m, n, bm, bn, bk, gm, num_threads, shard_by_col);
      gm = coarsenM(m, n, bm, bn, bk, gn, num_threads, shard_by_col);
    }
    // Number of tasks in each dimension.
    Index nm = divup(nm0, gm);
    Index nn = divup(nn0, gn);

    // Last by not least, decide whether we want to issue both lhs and rhs
    // packing in parallel; or issue lhs packing first, and then issue rhs
    // packing when lhs packing completes (for !shard_by_col lhs and rhs are
    // swapped). Parallel packing allows more parallelism (for both packing and
    // kernels), while sequential packing provides better locality (once
    // a thread finishes rhs packing it proceed to kernels with that rhs).
    // First, we are interested in parallel packing if there are few tasks.
    bool parallel_pack = num_threads >= nm * nn;
    // Also do parallel packing if all data fits into L2$.
    if (m * bk * Index(sizeof(LhsScalar)) + n * bk * Index(sizeof(RhsScalar)) <=
        l2CacheSize() * num_threads)
      parallel_pack = true;
    // But don't do it if we will use each rhs only once. Locality seems to be
    // more important in this case.
    if ((shard_by_col ? nm : nn) == 1) parallel_pack = false;

    LhsMapper lhs(this->m_leftImpl, this->m_left_nocontract_strides,
                  this->m_i_strides, this->m_left_contracting_strides,
                  this->m_k_strides);

    RhsMapper rhs(this->m_rightImpl, this->m_right_nocontract_strides,
                  this->m_j_strides, this->m_right_contracting_strides,
                  this->m_k_strides);

    Context<LhsPacker, RhsPacker, GebpKernel, LhsMapper, RhsMapper,
            OutputMapper>(this->m_device, num_threads, lhs, rhs, buffer, m, n,
                          k, bm, bn, bk, nm, nn, nk, gm, gn, nm0, nn0,
                          shard_by_col, parallel_pack)
        .run();
  }

  // Context coordinates a single parallel gemm operation.
  template <typename LhsPacker, typename RhsPacker, typename GebpKernel,
            typename LhsMapper, typename RhsMapper, typename OutputMapper>
  class Context {
   public:
    Context(const Device& device, int num_threads, LhsMapper& lhs,
            RhsMapper& rhs, Scalar* buffer, Index tm, Index tn, Index tk, Index bm,
            Index bn, Index bk, Index nm, Index nn, Index nk, Index gm,
            Index gn, Index nm0, Index nn0, bool shard_by_col,
            bool parallel_pack)
        : device_(device),
          lhs_(lhs),
          rhs_(rhs),
          buffer_(buffer),
          output_(buffer, tm),
          num_threads_(num_threads),
          shard_by_col_(shard_by_col),
          parallel_pack_(parallel_pack),
          m_(tm),
          n_(tn),
          k_(tk),
          bm_(bm),
          bn_(bn),
          bk_(bk),
          nm_(nm),
          nn_(nn),
          nk_(nk),
          gm_(gm),
          gn_(gn),
          nm0_(nm0),
          nn0_(nn0)
  {
      for (Index x = 0; x < P; x++) {
        // Normal number of notifications for k slice switch is
        // nm_ + nn_ + nm_ * nn_. However, first P - 1 slices will receive only
        // nm_ + nn_ notifications, because they will not receive notifications
        // from preceeding kernels.
        state_switch_[x] =
            x == 0
                ? 1
                : (parallel_pack_ ? nn_ + nm_ : (shard_by_col_ ? nn_ : nm_)) +
                      (x == P - 1 ? nm_ * nn_ : 0);
        state_packing_ready_[x] =
            parallel_pack_ ? 0 : (shard_by_col_ ? nm_ : nn_);
        state_kernel_[x] = new std::atomic<uint8_t>*[nm_];
        for (Index m = 0; m < nm_; m++) {
          state_kernel_[x][m] = new std::atomic<uint8_t>[nn_];
          // Kernels generally receive 3 notifications (previous kernel + 2
          // packing), but the first slice won't get notifications from previous
          // kernels.
          for (Index n = 0; n < nn_; n++)
            state_kernel_[x][m][n].store(
                (x == 0 ? 0 : 1) + (parallel_pack_ ? 2 : 1),
                std::memory_order_relaxed);
        }
      }

      // Allocate memory for packed rhs/lhs matrices.
      size_t align = numext::maxi(EIGEN_MAX_ALIGN_BYTES, 1);
      size_t lhs_size =
          divup<size_t>(bm_ * bk_ * sizeof(LhsScalar), align) * align;
      size_t rhs_size =
          divup<size_t>(bn_ * bk_ * sizeof(RhsScalar), align) * align;
      packed_mem_ = static_cast<char*>(internal::aligned_malloc(
          (nm0_ * lhs_size + nn0_ * rhs_size) * std::min<size_t>(nk_, P - 1)));
      char* mem = static_cast<char*>(packed_mem_);
      for (Index x = 0; x < numext::mini<Index>(nk_, P - 1); x++) {
        packed_lhs_[x].resize(nm0_);
        for (Index m = 0; m < nm0_; m++) {
          packed_lhs_[x][m] = reinterpret_cast<LhsScalar*>(mem);
          mem += lhs_size;
        }
        packed_rhs_[x].resize(nn0_);
        for (Index n = 0; n < nn0_; n++) {
          packed_rhs_[x][n] = reinterpret_cast<RhsScalar*>(mem);
          mem += rhs_size;
        }
      }
    }

    ~Context() {
      for (Index x = 0; x < P; x++) {
        for (Index m = 0; m < nm_; m++) delete[] state_kernel_[x][m];
        delete[] state_kernel_[x];
      }
      internal::aligned_free(packed_mem_);
    }

    void run() {
      // Kick off packing of the first slice.
      signal_switch(0, 1);
      // Wait for overall completion.
      // TODO(dvyukov): this wait can lead to deadlock.
      // If nthreads contractions are concurrently submitted from worker
      // threads, this wait will block all worker threads and the system will
      // deadlock.
      done_.Wait();
    }

   private:
    Notification done_;
    const Device& device_;
    LhsMapper& lhs_;
    RhsMapper& rhs_;
    Scalar* const buffer_;
    OutputMapper output_;
    const int num_threads_;
    const bool shard_by_col_;
    const bool parallel_pack_;
    // Matrix sizes.
    const Index m_;
    const Index n_;
    const Index k_;
    // Block sizes.
    const Index bm_;
    const Index bn_;
    const Index bk_;
    // Number of tasks.
    const Index nm_;
    const Index nn_;
    const Index nk_;
    // Task grain sizes (number of kernels executed per task).
    const Index gm_;
    const Index gn_;
    // Number of blocks (this is different from ni_/nn_ because of task size
    // coarsening).
    const Index nm0_;
    const Index nn0_;

    // Parallelization strategy.
    //
    // Blocks related to the same k block can run in parallel because they write
    // to different output blocks. So we parallelize within k slices, this
    // gives us parallelism level of m x n. Before we can start any kernels
    // related to k-th slice, we need to issue m lhs packing tasks and n rhs
    // packing tasks.
    //
    // However, there is a bottleneck when we are finishing kernels for k-th
    // slice (at the very end there is only 1 runnable kernel). To mitigate this
    // bottleneck we allow kernels from k-th and k+1-th slices to run in
    // parallel. Note that (m, n, k) and (m, n, k+1) kernels write to the same
    // output block, so they must not run in parallel.
    //
    // This gives us the following dependency graph.
    // On each k slice we have m x n kernel tasks, m lhs paking tasks and n rhs
    // packing tasks.
    // Kernel (m, n, k) can start when:
    //  - kernel (m, n, k-1) has finished
    //  - lhs packing (m, k) has finished
    //  - rhs packing (n, k) has finished
    // Lhs/rhs packing can start when:
    //  - all k-1 packing has finished (artificially imposed to limit amount of
    //  parallel packing)
    //
    // On top of that we limit runnable tasks to two consecutive k slices.
    // This is done to limit amount of memory we need for packed lhs/rhs
    // (for each k slice we need m*bk + n*bk memory in packed_lhs_/packed_rhs_).
    //
    // state_switch_ tracks when we are ready to switch to the next k slice.
    // state_kernel_[m][n] tracks when we are ready to kick off kernel (m, n).
    // These variable are rolling over 3 consecutive k slices: first two we are
    // actively executing + one to track completion of kernels in the second
    // slice.
    static const Index P = 3;
    void* packed_mem_;
    std::vector<LhsScalar*> packed_lhs_[P - 1];
    std::vector<RhsScalar*> packed_rhs_[P - 1];
    std::atomic<uint8_t>** state_kernel_[P];
    // state_switch_ is frequently modified by worker threads, while other
    // fields are read-only after constructor. Let's move it to a separate cache
    // line to reduce cache-coherency traffic.
    char pad_[128];
    std::atomic<Index> state_packing_ready_[P];
    std::atomic<Index> state_switch_[P];

    void pack_lhs(Index m, Index k) {
      const Index mend = m * gm_ + gm(m);
      for (Index m1 = m * gm_; m1 < mend; m1++)
        LhsPacker()(packed_lhs_[k % (P - 1)][m1],
                    lhs_.getSubMapper(m1 * bm_, k * bk_), bk(k), bm(m1));

      if (!parallel_pack_ && shard_by_col_) {
        signal_packing(k);
      } else {
        signal_switch(k + 1);
        for (Index n = nn_ - 1; n >= 0; n--) signal_kernel(m, n, k, n == 0);
      }
    }

    void pack_rhs(Index n, Index k) {
      const Index nend = n * gn_ + gn(n);
      for (Index n1 = n * gn_; n1 < nend; n1++) {
        if (k == 0) {
          // Zero the output memory in parallel.
          // On 10000x2x10000 mm zeroing can easily take half of time.
          // Zero (bn x m) row. Safe to do here because all kernels that will
          // write to this memory depend on completion of this task.
          // Note: don't call device_.memset() here. device_.memset() blocks on
          // thread pool worker thread, which can lead to underutilization and
          // deadlocks.
          memset(buffer_ + n1 * bn_ * m_, 0, bn(n1) * m_ * sizeof(Scalar));
        }
        RhsPacker()(packed_rhs_[k % (P - 1)][n1],
                    rhs_.getSubMapper(k * bk_, n1 * bn_), bk(k), bn(n1));
      }

      if (parallel_pack_ || shard_by_col_) {
        signal_switch(k + 1);
        for (Index m = nm_ - 1; m >= 0; m--) signal_kernel(m, n, k, m == 0);
      } else {
        signal_packing(k);
      }
    }

    void kernel(Index m, Index n, Index k) {
      // Note: order of iteration matters here. Iteration over m is innermost
      // because we want to reuse the same packed rhs in consequetive tasks
      // (rhs fits into L2$ while lhs only into L3$).
      const Index nend = n * gn_ + gn(n);
      const Index mend = m * gm_ + gm(m);
      if (shard_by_col_) {
        for (Index n1 = n * gn_; n1 < nend; n1++) {
          for (Index m1 = m * gm_; m1 < mend; m1++)
            GebpKernel()(output_.getSubMapper(m1 * bm_, n1 * bn_),
                         packed_lhs_[k % (P - 1)][m1],
                         packed_rhs_[k % (P - 1)][n1], bm(m1), bk(k), bn(n1),
                         Scalar(1), -1, -1, 0, 0);
        }
      } else {
        for (Index m1 = m * gm_; m1 < mend; m1++)
          for (Index n1 = n * gn_; n1 < nend; n1++) {
            GebpKernel()(output_.getSubMapper(m1 * bm_, n1 * bn_),
                         packed_lhs_[k % (P - 1)][m1],
                         packed_rhs_[k % (P - 1)][n1], bm(m1), bk(k), bn(n1),
                         Scalar(1), -1, -1, 0, 0);
          }
      }
      signal_kernel(m, n, k + 1, false);
      signal_switch(k + 2);
    }

    void signal_packing(Index k) {
      eigen_assert(!parallel_pack_);
      Index s = state_packing_ready_[k % P].fetch_sub(1);
      eigen_assert(s > 0);
      if (s != 1) return;
      state_packing_ready_[k % P] = shard_by_col_ ? nm_ : nn_;
      enqueue_packing(k, shard_by_col_);
    }

    void signal_kernel(Index m, Index n, Index k, bool sync) {
      std::atomic<uint8_t>* state = &state_kernel_[k % P][m][n];
      Index s = state->load();
      eigen_assert(s > 0);
      if (s != 1 && state->fetch_sub(1) != 1) return;
      state->store(parallel_pack_ ? 3 : 2, std::memory_order_relaxed);
      if (sync)
        kernel(m, n, k);
      else
        device_.enqueueNoNotification([=]() { kernel(m, n, k); });
    }

    void signal_switch(Index k, Index v = 1) {
      Index s = state_switch_[k % P].fetch_sub(v);
      eigen_assert(s >= v);
      if (s != v) return;

      // Ready to switch to the next k slice.
      // Reset counter for the next iteration.
      state_switch_[k % P] =
          (parallel_pack_ ? nm_ + nn_ : (shard_by_col_ ? nn_ : nm_)) +
          nm_ * nn_;
      if (k < nk_) {
        // Issue lhs/rhs packing. Their completion will in turn kick off
        // kernels.
        if (parallel_pack_) {
          enqueue_packing(k, !shard_by_col_);
          enqueue_packing(k, shard_by_col_);
        } else if (shard_by_col_) {
          enqueue_packing(k, false);
        } else {
          enqueue_packing(k, true);
        }

        // Termination handling.
        // Because kernel completion signals k + 2 switch, we need to finish nk
        // + 2 slices without issuing any tasks on nk + 1 slice. So here we
        // pretend that all nk + 1 packing tasks just finish instantly; so that
        // nk + 2 switch only waits for completion of nk kernels.
      } else if (k == nk_) {
        signal_switch(k + 1,
                      parallel_pack_ ? nm_ + nn_ : (shard_by_col_ ? nn_ : nm_));
      } else {
        done_.Notify();
      }
    }

    // Enqueue all rhs/lhs packing for k-th slice.
    void enqueue_packing(Index k, bool rhs) {
      enqueue_packing_helper(0, rhs ? nn_ : nm_, k, rhs);
    }

    void enqueue_packing_helper(Index start, Index end, Index k, bool rhs) {
      if (end - start == 1) {
        if (rhs)
          pack_rhs(start, k);
        else
          pack_lhs(start, k);
      } else {
        Index mid = (start + end) / 2;
        device_.enqueueNoNotification(
            [=]() { enqueue_packing_helper(mid, end, k, rhs); });
        device_.enqueueNoNotification(
            [=]() { enqueue_packing_helper(start, mid, k, rhs); });
      }
    }

    // Block sizes with accounting for potentially incomplete last block.
    Index bm(Index m) const { return m + 1 < nm0_ ? bm_ : m_ + bm_ - bm_ * nm0_; }
    Index bn(Index n) const { return n + 1 < nn0_ ? bn_ : n_ + bn_ - bn_ * nn0_; }
    Index bk(Index k) const { return k + 1 < nk_ ? bk_ : k_ + bk_ - bk_ * nk_; }
    // Task grain sizes accounting for potentially incomplete last task.
    Index gm(Index m) const { return m + 1 < nm_ ? gm_ : nm0_ + gm_ - gm_ * nm_; }
    Index gn(Index n) const { return n + 1 < nn_ ? gn_ : nn0_ + gn_ - gn_ * nn_; }

    Context(const Context&) = delete;
    void operator=(const Context&) = delete;
  };

  // Decide whether we want to shard m x n contraction by columns or by rows.
  static bool shardByCol(Index m, Index n, Index num_threads) {
    // Note: we are comparing both n and m against Traits::nr, it is not
    // a mistake. We are trying to figure out how both n and m will fit into
    // the main sharding dimension.

    // Sharding by column is the default
    // ... unless there is enough data for vectorization over rows
    if (m / num_threads >= Traits::nr &&
        // and not enough data for vectorization over columns
        (n / num_threads < Traits::nr ||
         // ... or barely enough data for vectorization over columns,
         // but it is not evenly dividable across threads
         (n / num_threads < 4 * Traits::nr &&
          (n % (num_threads * Traits::nr)) != 0 &&
          // ... and it is evenly dividable across threads for rows
          ((m % (num_threads * Traits::nr)) == 0 ||
           // .. or it is not evenly dividable for both dimensions but
           // there is much more data over rows so that corner effects are
           // mitigated.
           (m / n >= 6)))))
      return false;
    // Wait, or if matrices are just substantially prolonged over the other
    // dimension.
    if (n / num_threads < 16 * Traits::nr && m > n * 32) return false;
    return true;
  }

  Index coarsenM(Index m, Index n, Index bm, Index bn, Index bk, Index gn,
                 int num_threads, bool shard_by_col) const {
    Index gm = 1;
    Index gm1 = 1;
    Index nm0 = divup(m, bm);
    Index nm1 = nm0;
    for (;;) {
      // Find the next candidate for m grain size. It needs to result in
      // different number of blocks. E.g. if we have 10 kernels, we want to try
      // 5 and 10, but not 6, 7, 8 and 9.
      while (gm1 <= nm0 && nm1 == divup(nm0, gm1)) gm1++;
      if (gm1 > nm0) break;
      // Check the candidate.
      int res = checkGrain(m, n, bm, bn, bk, gm1, gn, gm, gn, num_threads,
                           shard_by_col);
      if (res < 0) break;
      nm1 = divup(nm0, gm1);
      if (res == 0) continue;
      // Commit new grain size.
      gm = gm1;
    }
    return gm;
  }

  Index coarsenN(Index m, Index n, Index bm, Index bn, Index bk, Index gm,
                 int num_threads, bool shard_by_col) const {
    Index gn = 1;
    Index gn1 = 1;
    Index nn0 = divup(n, bn);
    Index nn1 = nn0;
    for (;;) {
      while (gn1 <= nn0 && nn1 == divup(nn0, gn1)) gn1++;
      if (gn1 > nn0) break;
      int res = checkGrain(m, n, bm, bn, bk, gm, gn1, gm, gn, num_threads,
                           shard_by_col);
      if (res < 0) break;
      nn1 = divup(nn0, gn1);
      if (res == 0) continue;
      gn = gn1;
    }
    return gn;
  }

  // checkGrain checks whether grain (gm, gn) is suitable and is better than
  // (oldgm, oldgn).
  int checkGrain(Index m, Index n, Index bm, Index bn, Index bk, Index gm,
                 Index gn, Index oldgm, Index oldgn, int num_threads,
                 bool shard_by_col) const {
    const TensorOpCost cost =
        contractionCost(bm * gm, bn * gn, bm, bn, bk, shard_by_col, true);
    double taskSize = TensorCostModel<ThreadPoolDevice>::taskSize(
        static_cast<double>(bm) * gm * bn * gn, cost);
    // If the task is too small, then we agree on it regardless of anything
    // else. Otherwise synchronization overheads will dominate.
    if (taskSize < 1) return 1;
    // If it is too large, then we reject it and all larger tasks.
    if (taskSize > 2) return -1;
    // Now we are in presumably good task size range.
    // The main deciding factor here is parallelism. Consider that we have 12
    // kernels and 4 threads. Grains of 2, 3 and 4 all yield good task sizes.
    // But 2/4 yield 6/3 tasks, which gives us parallelism of 0.75 (at most 3/4
    // of cores will be busy). While grain size 3 gives us 4 tasks, which gives
    // us parallelism of 1 (we can load all cores).
    Index nm0 = divup(m, bm);
    Index nn0 = divup(n, bn);
    Index new_tasks = divup(nm0, gm) * divup(nn0, gn);
    double new_parallelism = static_cast<double>(new_tasks) /
                             (divup<int>(new_tasks, num_threads) * num_threads);
    Index old_tasks = divup(nm0, oldgm) * divup(nn0, oldgn);
    double old_parallelism = static_cast<double>(old_tasks) /
                             (divup<int>(old_tasks, num_threads) * num_threads);
    if (new_parallelism > old_parallelism || new_parallelism == 1) return 1;
    return 0;
  }

#else  // EIGEN_USE_SIMPLE_THREAD_POOL

  template <bool lhs_inner_dim_contiguous, bool rhs_inner_dim_contiguous, bool rhs_inner_dim_reordered, int Alignment>
  void evalProduct(Scalar* buffer) const {
    if (this->m_j_size == 1) {
      this->template evalGemv<lhs_inner_dim_contiguous, rhs_inner_dim_contiguous, rhs_inner_dim_reordered, Alignment>(buffer);
      return;
    }

    evalGemm<lhs_inner_dim_contiguous, rhs_inner_dim_contiguous, rhs_inner_dim_reordered, Alignment>(buffer);
  }

  template <bool lhs_inner_dim_contiguous, bool rhs_inner_dim_contiguous, bool rhs_inner_dim_reordered, int Alignment>
  void evalGemm(Scalar* buffer) const {
    // columns in left side, rows in right side
    const Index k = this->m_k_size;

    // rows in left side
    const Index m = this->m_i_size;

    // columns in right side
    const Index n = this->m_j_size;

    // zero out the result buffer (which must be of size at least m * n * sizeof(Scalar)
    this->m_device.memset(buffer, 0, m * n * sizeof(Scalar));


    const int lhs_packet_size = internal::unpacket_traits<typename LeftEvaluator::PacketReturnType>::size;
    const int rhs_packet_size = internal::unpacket_traits<typename RightEvaluator::PacketReturnType>::size;

    typedef internal::TensorContractionInputMapper<LhsScalar, Index, internal::Lhs,
                                                   LeftEvaluator, left_nocontract_t,
                                                   contract_t, lhs_packet_size,
                                                   lhs_inner_dim_contiguous,
                                                   false, Unaligned> LhsMapper;

    typedef internal::TensorContractionInputMapper<RhsScalar, Index, internal::Rhs,
                                                   RightEvaluator, right_nocontract_t,
                                                   contract_t, rhs_packet_size,
                                                   rhs_inner_dim_contiguous,
                                                   rhs_inner_dim_reordered, Unaligned> RhsMapper;

    typedef internal::blas_data_mapper<Scalar, Index, ColMajor> OutputMapper;

    // TODO: packing could be faster sometimes if we supported row major tensor mappers
    typedef internal::gemm_pack_lhs<LhsScalar, Index, typename LhsMapper::SubMapper, Traits::mr,
                                    Traits::LhsProgress, ColMajor> LhsPacker;
    typedef internal::gemm_pack_rhs<RhsScalar, Index, typename RhsMapper::SubMapper, Traits::nr, ColMajor> RhsPacker;

    // TODO: replace false, false with conjugate values?
    typedef internal::gebp_kernel<LhsScalar, RhsScalar, Index, OutputMapper,
                                  Traits::mr, Traits::nr, false, false> GebpKernel;

    typedef internal::packLhsArg<LhsScalar, LhsMapper, Index> packLArg;
    typedef internal::packRhsAndKernelArg<LhsScalar, RhsScalar, RhsMapper, OutputMapper, Index> packRKArg;

    // initialize data mappers
    LhsMapper lhs(this->m_leftImpl, this->m_left_nocontract_strides, this->m_i_strides,
                  this->m_left_contracting_strides, this->m_k_strides);

    RhsMapper rhs(this->m_rightImpl, this->m_right_nocontract_strides, this->m_j_strides,
                  this->m_right_contracting_strides, this->m_k_strides);

    OutputMapper output(buffer, m);

    // compute block sizes (which depend on number of threads)
    const Index num_threads = this->m_device.numThreads();
    internal::TensorContractionBlocking<LhsMapper, RhsMapper, Index, internal::ShardByCol> blocking(k, m, n, num_threads);
    Index mc = blocking.mc();
    Index nc = blocking.nc();
    Index kc = blocking.kc();
    eigen_assert(mc <= m);
    eigen_assert(nc <= n);
    eigen_assert(kc <= k);

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
    const Index k_blocks = CEIL_DIV(k, kc);
    const Index n_blocks = CEIL_DIV(n, nc);
    const Index m_blocks = CEIL_DIV(m, mc);
    const Index sizeA = mc * kc;
    const Index sizeB = kc * nc;

    /*    cout << "m: " << m << " n: " << n << " k: " << k << endl;
    cout << "mc: " << mc << " nc: " << nc << " kc: " << kc << endl;
    cout << "m_blocks: " << m_blocks << " n_blocks: " << n_blocks << " k_blocks: " << k_blocks << endl;
    cout << "num threads: " << num_threads << endl;
    */

    // note: m_device.allocate should return 16 byte aligned pointers, but if blockA and blockB
    //       aren't 16 byte aligned segfaults will happen due to SIMD instructions
    // note: You can get away with allocating just a single blockA and offsets and meet the
    //       the alignment requirements with the assumption that
    //       (Traits::mr * sizeof(ResScalar)) % 16 == 0
    const Index numBlockAs = numext::mini(num_threads, m_blocks);
    MaxSizeVector<LhsScalar *> blockAs(num_threads);
    for (int i = 0; i < num_threads; i++) {
      blockAs.push_back(static_cast<LhsScalar *>(this->m_device.allocate(sizeA * sizeof(LhsScalar))));
    }

    // To circumvent alignment issues, I'm just going to separately allocate the memory for each thread
    // TODO: is this too much memory to allocate? This simplifies coding a lot, but is wasteful.
    //       Other options: (1) reuse memory when a thread finishes. con: tricky
    //                      (2) allocate block B memory in each thread. con: overhead
    MaxSizeVector<RhsScalar *> blockBs(n_blocks);
    for (int i = 0; i < n_blocks; i++) {
      blockBs.push_back(static_cast<RhsScalar *>(this->m_device.allocate(sizeB * sizeof(RhsScalar))));
    }

    // lhs_notifications starts with all null Notifications
    MaxSizeVector<Notification*> lhs_notifications(num_threads, nullptr);

    // this should really be numBlockAs * n_blocks;
    const Index num_kernel_notifications = num_threads * n_blocks;
    MaxSizeVector<Notification*> kernel_notifications(num_kernel_notifications,
                                                    nullptr);

    for (Index k_block_idx = 0; k_block_idx < k_blocks; k_block_idx++) {
      const Index k_start = k_block_idx * kc;
      // make sure we don't overshoot right edge of left matrix
      const Index actual_kc = numext::mini(k_start + kc, k) - k_start;

      for (Index m_block_idx = 0; m_block_idx < m_blocks; m_block_idx += numBlockAs) {
        const Index num_blocks = numext::mini(m_blocks-m_block_idx, numBlockAs);

        for (Index mt_block_idx = m_block_idx; mt_block_idx < m_block_idx+num_blocks; mt_block_idx++) {
          const Index m_start = mt_block_idx * mc;
          const Index actual_mc = numext::mini(m_start + mc, m) - m_start;
          eigen_assert(actual_mc > 0);

          Index blockAId = (k_block_idx * m_blocks + mt_block_idx) % num_threads;

          for (int i = 0; i < n_blocks; ++i) {
            Index notification_id = (blockAId * n_blocks + i);
            // Wait for any current kernels using this slot to complete
            // before using it.
            if (kernel_notifications[notification_id]) {
              wait_until_ready(kernel_notifications[notification_id]);
              delete kernel_notifications[notification_id];
            }
            kernel_notifications[notification_id] = new Notification();
          }
          const packLArg arg = {
            blockAs[blockAId], // blockA
            lhs,        // lhs
            m_start,    // m
            k_start,    // k
            actual_mc,  // mc
            actual_kc,  // kc
          };

          // Delete any existing notification since we may be
          // replacing it.  The algorithm should ensure that there are
          // no existing waiters on this notification.
          delete lhs_notifications[blockAId];
          lhs_notifications[blockAId] =
          this->m_device.enqueue(&Self::packLhs<packLArg, LhsPacker>, arg);
        }

        // now start kernels.
        const Index m_base_start = m_block_idx * mc;
        const bool need_to_pack = m_block_idx == 0;

        for (Index n_block_idx = 0; n_block_idx < n_blocks; n_block_idx++) {
          const Index n_start = n_block_idx * nc;
          const Index actual_nc = numext::mini(n_start + nc, n) - n_start;

          // first make sure the previous kernels are all done before overwriting rhs. Also wait if
          // we're going to start new k. In both cases need_to_pack is true.
          if (need_to_pack) {
            for (Index i = num_blocks; i < num_threads; ++i) {
              Index blockAId = (k_block_idx * m_blocks + i + m_block_idx) % num_threads;
              Index future_id = (blockAId * n_blocks + n_block_idx);
              wait_until_ready(kernel_notifications[future_id]);
            }
          }

          packRKArg arg = {
            &blockAs, // blockA
            blockBs[n_block_idx], // blockB
            rhs,          // rhs
            output,       // output
            m_base_start, // m
            k_start,      // k
            n_start,      // n
            mc,           // mc
            actual_kc,    // kc
            actual_nc,    // nc
            num_threads,
            numBlockAs,
            m,
            k_block_idx,
            m_block_idx,
            n_block_idx, // n_block_idx
            m_blocks, // m_blocks
            n_blocks, // n_blocks
            &kernel_notifications, // kernel notifications
            &lhs_notifications,    // lhs notifications
            need_to_pack, // need_to_pack
          };

          // We asynchronously kick off this function, which ends up
          // notifying the appropriate kernel_notifications objects,
          // which this thread waits on before exiting.
          this->m_device.enqueueNoNotification(&Self::packRhsAndKernel<packRKArg, RhsPacker, GebpKernel>, arg);
        }
      }
    }

    // Make sure all the kernels are done.
    for (size_t i = 0; i < kernel_notifications.size(); ++i) {
      wait_until_ready(kernel_notifications[i]);
      delete kernel_notifications[i];
    }

    // No need to wait for lhs notifications since they should have
    // already been waited on.  Just clean them up.
    for (size_t i = 0; i < lhs_notifications.size(); ++i) {
      delete lhs_notifications[i];
    }

    // deallocate all of the memory for both A and B's
    for (size_t i = 0; i < blockAs.size(); i++) {
      this->m_device.deallocate(blockAs[i]);
    }
    for (size_t i = 0; i < blockBs.size(); i++) {
      this->m_device.deallocate(blockBs[i]);
    }

#undef CEIL_DIV
  }

  /*
   * Packs a LHS block of size (mt, kc) starting at lhs(m, k). Before packing
   * the LHS block, check that all of the kernels that worked on the same
   * mt_block_idx in the previous m_block are done.
   */
  template <typename packLArg, typename LhsPacker>
  static void packLhs(const packLArg arg) {
    // perform actual packing
    LhsPacker pack_lhs;
    pack_lhs(arg.blockA, arg.lhs.getSubMapper(arg.m_start, arg.k_start), arg.kc, arg.mc);
  }

  /*
   * Packs a RHS block of size (kc, nc) starting at (k, n) after checking that
   * all kernels in the previous block are done.
   * Then for each LHS future, we wait on the future and then call GEBP
   * on the area packed by the future (which starts at
   * blockA + future_idx * mt * kc) on the LHS and with the full packed
   * RHS block.
   * The output of this GEBP is written to output(m + i * mt, n).
   */
  template <typename packRKArg, typename RhsPacker, typename GebpKernel>
  static void packRhsAndKernel(packRKArg arg) {
    if (arg.need_to_pack) {
      RhsPacker pack_rhs;
      pack_rhs(arg.blockB, arg.rhs.getSubMapper(arg.k, arg.n), arg.kc, arg.nc);
    }

    GebpKernel gebp;
    for (Index mt_block_idx = 0; mt_block_idx < arg.num_blockAs; mt_block_idx++) {
      const Index m_base_start = arg.m + arg.mc*mt_block_idx;
      if (m_base_start < arg.max_m) {
        Index blockAId = (arg.k_block_idx * arg.m_blocks + mt_block_idx + arg.m_block_idx) % arg.num_threads;
        wait_until_ready((*arg.lhs_notifications)[blockAId]);
        const Index actual_mc = numext::mini(m_base_start + arg.mc, arg.max_m) - m_base_start;
        gebp(arg.output.getSubMapper(m_base_start, arg.n),
             (*arg.blockAs)[blockAId], arg.blockB,
             actual_mc, arg.kc, arg.nc, Scalar(1), -1, -1, 0, 0);

        // Notify that the kernel is done.
        const Index set_idx = blockAId * arg.n_blocks + arg.n_block_idx;
        (*arg.kernel_notifications)[set_idx]->Notify();
      }
    }
  }
#endif  // EIGEN_USE_SIMPLE_THREAD_POOL

  TensorOpCost contractionCost(Index m, Index n, Index bm, Index bn, Index bk,
                               bool shard_by_col, bool prepacked) const {
    const int packed_size = std::min<int>(PacketType<LhsScalar, Device>::size,
                                          PacketType<RhsScalar, Device>::size);
    const int output_packet_size = internal::unpacket_traits<PacketReturnType>::size;
    const double kd = static_cast<double>(bk);
    // Peak VFMA bandwidth is 0.5. However if we have not enough data for
    // vectorization bandwidth drops. The 4.0 and 2.0 bandwidth is determined
    // experimentally.
    double computeBandwidth = bk == 1 ? 4.0 :
          (shard_by_col ? bn : bm) < Traits::nr ||
          (shard_by_col ? bm : bn) < Traits::mr ? 2.0 : 0.5;
#ifndef EIGEN_VECTORIZE_FMA
    // Bandwidth of all of VFMA/MULPS/ADDPS is 0.5 on latest Intel processors.
    // However for MULPS/ADDPS we have dependent sequence of 2 such instructions,
    // so overall bandwidth is 1.0.
    if (computeBandwidth == 0.5) computeBandwidth = 1.0;
#endif
    // Computations.
    TensorOpCost cost = TensorOpCost(0, 0, kd * computeBandwidth, true, packed_size);
    // Output stores.
    cost += TensorOpCost(0, sizeof(CoeffReturnType), 0, true, output_packet_size);
    if (prepacked) {
      // Packing and kernels are executed in different tasks. When we calculate
      // task grain size we look only at kernel cost assuming that kernel
      // is more expensive than packing.
      return cost;
    }
    // Lhs/rhs loads + computations.
    TensorOpCost lhsCost = this->m_leftImpl.costPerCoeff(true) * (kd / n);
    TensorOpCost rhsCost = this->m_rightImpl.costPerCoeff(true) * (kd / m);
    // Lhs packing memory cost does not contribute considerably to overall
    // execution time because lhs is prefetched early and accessed sequentially.
    if (shard_by_col)
      lhsCost.dropMemoryCost();
    else
      rhsCost.dropMemoryCost();
    return cost + lhsCost + rhsCost;
  }
};

} // end namespace Eigen

#endif  // EIGEN_USE_THREADS
#endif // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_THREAD_POOL_H
