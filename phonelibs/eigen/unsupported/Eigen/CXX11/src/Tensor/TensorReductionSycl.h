// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*****************************************************************
 * TensorSyclPlaceHolderExpr.h
 *
 * \brief:
 *  This is the specialisation of the placeholder expression based on the
 * operation type
 *
*****************************************************************/

#ifndef UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSOR_REDUCTION_SYCL_HPP
#define UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSOR_REDUCTION_SYCL_HPP

namespace Eigen {
namespace internal {

template<typename CoeffReturnType, typename KernelName> struct syclGenericBufferReducer{
template<typename BufferTOut, typename BufferTIn>
static void run(BufferTOut* bufOut, BufferTIn& bufI, const Eigen::SyclDevice& dev, size_t length, size_t local){
  do {
          auto f = [length, local, bufOut, &bufI](cl::sycl::handler& h) mutable {
            cl::sycl::nd_range<1> r{cl::sycl::range<1>{std::max(length, local)},
                                    cl::sycl::range<1>{std::min(length, local)}};
            /* Two accessors are used: one to the buffer that is being reduced,
             * and a second to local memory, used to store intermediate data. */
            auto aI =
                bufI.template get_access<cl::sycl::access::mode::read_write>(h);
            auto aOut =
                bufOut->template get_access<cl::sycl::access::mode::discard_write>(h);
            cl::sycl::accessor<CoeffReturnType, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::local>
                scratch(cl::sycl::range<1>(local), h);

            /* The parallel_for invocation chosen is the variant with an nd_item
             * parameter, since the code requires barriers for correctness. */
            h.parallel_for<KernelName>(
                r, [aOut, aI, scratch, local, length](cl::sycl::nd_item<1> id) {
                  size_t globalid = id.get_global(0);
                  size_t localid = id.get_local(0);
                  /* All threads collectively read from global memory into local.
                   * The barrier ensures all threads' IO is resolved before
                   * execution continues (strictly speaking, all threads within
                   * a single work-group - there is no co-ordination between
                   * work-groups, only work-items). */
                  if (globalid < length) {
                    scratch[localid] = aI[globalid];
                  }
                  id.barrier(cl::sycl::access::fence_space::local_space);

                  /* Apply the reduction operation between the current local
                   * id and the one on the other half of the vector. */
                  if (globalid < length) {
                    int min = (length < local) ? length : local;
                    for (size_t offset = min / 2; offset > 0; offset /= 2) {
                      if (localid < offset) {
                        scratch[localid] += scratch[localid + offset];
                      }
                      id.barrier(cl::sycl::access::fence_space::local_space);
                    }
                    /* The final result will be stored in local id 0. */
                    if (localid == 0) {
                      aI[id.get_group(0)] = scratch[localid];
                      if((length<=local) && globalid ==0){
                        aOut[globalid]=scratch[localid];
                      }
                    }
                  }
                });
          };
            dev.m_queue.submit(f);
            dev.m_queue.throw_asynchronous();

          /* At this point, you could queue::wait_and_throw() to ensure that
           * errors are caught quickly. However, this would likely impact
           * performance negatively. */
          length = length / local;

        } while (length > 1);



}

};

/// For now let's start with a full reducer
/// Self is useless here because in expression construction we are going to treat reduction as a leafnode.
/// we want to take reduction child and then build a construction and apply the full reducer function on it. Fullreducre applies the
/// reduction operation on the child of the reduction. once it is done the reduction is an empty shell and can be thrown away and treated as
// a leafNode.
template <typename Self, typename Op, bool Vectorizable>
struct FullReducer<Self, Op, const Eigen::SyclDevice, Vectorizable> {

  typedef typename Self::CoeffReturnType CoeffReturnType;
  static const bool HasOptimizedImplementation = false;

  static void run(const Self& self, Op& reducer, const Eigen::SyclDevice& dev, CoeffReturnType* output) {
    typedef const typename Self::ChildType HostExpr; /// this is the child of reduction
    typedef  typename TensorSycl::internal::createPlaceHolderExpression<HostExpr>::Type PlaceHolderExpr;
    auto functors = TensorSycl::internal::extractFunctors(self.impl());
    int red_factor =256; /// initial reduction. If the size is less than red_factor we only creates one thread.
    size_t inputSize =self.impl().dimensions().TotalSize();
    size_t rng = inputSize/red_factor; // the total number of thread initially is half the size of the input
    size_t remaining = inputSize% red_factor;
    if(rng ==0) {
      red_factor=1;
    };
    size_t tileSize =dev.m_queue.get_device(). template get_info<cl::sycl::info::device::max_work_group_size>()/2;
    size_t GRange=std::max((size_t )1, rng);

    // convert global range to power of 2 for redecution
    GRange--;
    GRange |= GRange >> 1;
    GRange |= GRange >> 2;
    GRange |= GRange >> 4;
    GRange |= GRange >> 8;
    GRange |= GRange >> 16;
#if __x86_64__ || __ppc64__ || _WIN64
    GRange |= GRange >> 32;
#endif
    GRange++;
    size_t  outTileSize = tileSize;
    /// if the shared memory is less than the GRange, we set shared_mem size to the TotalSize and in this case one kernel would be created for recursion to reduce all to one.
    if (GRange < outTileSize) outTileSize=GRange;
    // getting final out buffer at the moment the created buffer is true because there is no need for assign
    auto out_buffer =dev.template get_sycl_buffer<typename Eigen::internal::remove_all<CoeffReturnType>::type>(self.dimensions().TotalSize(), output);
    /// creating the shared memory for calculating reduction.
    /// This one is used to collect all the reduced value of shared memory as we dont have global barrier on GPU. Once it is saved we can
    /// recursively apply reduction on it in order to reduce the whole.
    auto temp_global_buffer =cl::sycl::buffer<CoeffReturnType, 1>(cl::sycl::range<1>(GRange));
    typedef typename Eigen::internal::remove_all<decltype(self.xprDims())>::type Dims;
    Dims dims= self.xprDims();
    Op functor = reducer;
    dev.m_queue.submit([&](cl::sycl::handler &cgh) {
      // create a tuple of accessors from Evaluator
      auto tuple_of_accessors =  TensorSycl::internal::createTupleOfAccessors(cgh, self.impl());
      auto tmp_global_accessor = temp_global_buffer. template get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>(cgh);

      cgh.parallel_for<PlaceHolderExpr>( cl::sycl::nd_range<1>(cl::sycl::range<1>(GRange), cl::sycl::range<1>(outTileSize)), [=](cl::sycl::nd_item<1> itemID) {
        typedef typename TensorSycl::internal::ConvertToDeviceExpression<const HostExpr>::Type DevExpr;
        auto device_expr = TensorSycl::internal::createDeviceExpression<DevExpr, PlaceHolderExpr>(functors, tuple_of_accessors);
        /// reduction cannot be captured automatically through our device conversion recursion. The reason is that reduction has two behaviour
        /// the first behaviour is when it is used as a root to lauch the sub-kernel. The second one is when it is treated as a leafnode to pass the
        /// calculated result to its parent kernel. While the latter is automatically detected through our device expression generator. The former is created here.
        const auto device_self_expr= TensorReductionOp<Op, Dims, decltype(device_expr.expr) ,MakeGlobalPointer>(device_expr.expr, dims, functor);
        /// This is the evaluator for device_self_expr. This is exactly similar to the self which has been passed to run function. The difference is
        /// the device_evaluator is detectable and recognisable on the device.
        auto device_self_evaluator = Eigen::TensorEvaluator<decltype(device_self_expr), Eigen::DefaultDevice>(device_self_expr, Eigen::DefaultDevice());
        /// const cast added as a naive solution to solve the qualifier drop error
        auto globalid=itemID.get_global_linear_id();

        if(globalid<rng)
          tmp_global_accessor.get_pointer()[globalid]=InnerMostDimReducer<decltype(device_self_evaluator), Op, false>::reduce(device_self_evaluator, red_factor*globalid, red_factor, const_cast<Op&>(functor));
        else
          tmp_global_accessor.get_pointer()[globalid]=static_cast<CoeffReturnType>(0);

        if(remaining!=0 && globalid==0 )
          // this will add the rest of input buffer when the input size is not devidable to red_factor.
          tmp_global_accessor.get_pointer()[globalid]+=InnerMostDimReducer<decltype(device_self_evaluator), Op, false>::reduce(device_self_evaluator, red_factor*(rng), remaining, const_cast<Op&>(functor));
      });
    });
  dev.m_queue.throw_asynchronous();

/// This is used to recursively reduce the tmp value to an element of 1;
  syclGenericBufferReducer<CoeffReturnType,HostExpr>::run(out_buffer, temp_global_buffer,dev, GRange,  outTileSize);
  }

};

template <typename Self, typename Op>
struct InnerReducer<Self, Op, const Eigen::SyclDevice> {

  typedef typename Self::CoeffReturnType CoeffReturnType;
  static const bool HasOptimizedImplementation = false;

  static bool run(const Self& self, Op& reducer, const Eigen::SyclDevice& dev, CoeffReturnType* output, typename Self::Index , typename Self::Index num_coeffs_to_preserve) {
    typedef const typename Self::ChildType HostExpr; /// this is the child of reduction
    typedef  typename TensorSycl::internal::createPlaceHolderExpression<HostExpr>::Type PlaceHolderExpr;
    auto functors = TensorSycl::internal::extractFunctors(self.impl());

    size_t tileSize =dev.m_queue.get_device(). template get_info<cl::sycl::info::device::max_work_group_size>()/2;

    size_t GRange=num_coeffs_to_preserve;
    if (tileSize>GRange) tileSize=GRange;
    else if(GRange>tileSize){
      size_t xMode = GRange % tileSize;
      if (xMode != 0) GRange += (tileSize - xMode);
    }
    // getting final out buffer at the moment the created buffer is true because there is no need for assign
    /// creating the shared memory for calculating reduction.
    /// This one is used to collect all the reduced value of shared memory as we dont have global barrier on GPU. Once it is saved we can
    /// recursively apply reduction on it in order to reduce the whole.
    typedef typename Eigen::internal::remove_all<decltype(self.xprDims())>::type Dims;
    Dims dims= self.xprDims();
    Op functor = reducer;

    dev.m_queue.submit([&](cl::sycl::handler &cgh) {
      // create a tuple of accessors from Evaluator
      auto tuple_of_accessors =  TensorSycl::internal::createTupleOfAccessors(cgh, self.impl());
      auto output_accessor = dev.template get_sycl_accessor<cl::sycl::access::mode::discard_write>(num_coeffs_to_preserve,cgh, output);

      cgh.parallel_for<Self>( cl::sycl::nd_range<1>(cl::sycl::range<1>(GRange), cl::sycl::range<1>(tileSize)), [=](cl::sycl::nd_item<1> itemID) {
        typedef typename TensorSycl::internal::ConvertToDeviceExpression<const HostExpr>::Type DevExpr;
        auto device_expr = TensorSycl::internal::createDeviceExpression<DevExpr, PlaceHolderExpr>(functors, tuple_of_accessors);
        /// reduction cannot be captured automatically through our device conversion recursion. The reason is that reduction has two behaviour
        /// the first behaviour is when it is used as a root to lauch the sub-kernel. The second one is when it is treated as a leafnode to pass the
        /// calculated result to its parent kernel. While the latter is automatically detected through our device expression generator. The former is created here.
        const auto device_self_expr= TensorReductionOp<Op, Dims, decltype(device_expr.expr) ,MakeGlobalPointer>(device_expr.expr, dims, functor);
        /// This is the evaluator for device_self_expr. This is exactly similar to the self which has been passed to run function. The difference is
        /// the device_evaluator is detectable and recognisable on the device.
        typedef Eigen::TensorEvaluator<decltype(device_self_expr), Eigen::DefaultDevice> DeiceSelf;
        auto device_self_evaluator = Eigen::TensorEvaluator<decltype(device_self_expr), Eigen::DefaultDevice>(device_self_expr, Eigen::DefaultDevice());
        /// const cast added as a naive solution to solve the qualifier drop error
        auto globalid=itemID.get_global_linear_id();
        if (globalid< static_cast<size_t>(num_coeffs_to_preserve)) {
          typename DeiceSelf::CoeffReturnType accum = functor.initialize();
          GenericDimReducer<DeiceSelf::NumReducedDims-1, DeiceSelf, Op>::reduce(device_self_evaluator, device_self_evaluator.firstInput(globalid),const_cast<Op&>(functor), &accum);
          functor.finalize(accum);
          output_accessor.get_pointer()[globalid]= accum;
        }
      });
    });
  dev.m_queue.throw_asynchronous();
    return false;
  }
};

}  // end namespace internal
}  // namespace Eigen

#endif  // UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSOR_REDUCTION_SYCL_HPP
