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
 * TensorSyclExtractAccessor.h
 *
 * \brief:
 * ExtractAccessor takes Expression placeHolder expression and the tuple of sycl
 * buffers as an input. Using pre-order tree traversal, ExtractAccessor
 * recursively calls itself for its children in the expression tree. The
 * leaf node in the PlaceHolder expression is nothing but a container preserving
 * the order of the actual data in the tuple of sycl buffer. By invoking the
 * extract accessor for the PlaceHolder<N>, an accessor is created for the Nth
 * buffer in the tuple of buffers. This accessor is then added as an Nth
 * element in the tuple of accessors. In this case we preserve the order of data
 * in the expression tree.
 *
 * This is the specialisation of extract accessor method for different operation
 * type in the PlaceHolder expression.
 *
*****************************************************************/

#ifndef UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_EXTRACT_ACCESSOR_HPP
#define UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_EXTRACT_ACCESSOR_HPP

namespace Eigen {
namespace TensorSycl {
namespace internal {
/// \struct ExtractAccessor: Extract Accessor Class is used to extract the
/// accessor from a buffer.
/// Depending on the type of the leaf node we can get a read accessor or a
/// read_write accessor
template <typename Evaluator>
struct ExtractAccessor;

struct AccessorConstructor{
  template<typename Arg> static inline auto getTuple(cl::sycl::handler& cgh, Arg eval)
  -> decltype(ExtractAccessor<Arg>::getTuple(cgh, eval)) {
  return ExtractAccessor<Arg>::getTuple(cgh, eval);
  }

  template<typename Arg1, typename Arg2> static inline auto getTuple(cl::sycl::handler& cgh, Arg1 eval1, Arg2 eval2)
  -> decltype(utility::tuple::append(ExtractAccessor<Arg1>::getTuple(cgh, eval1), ExtractAccessor<Arg2>::getTuple(cgh, eval2))) {
    return utility::tuple::append(ExtractAccessor<Arg1>::getTuple(cgh, eval1), ExtractAccessor<Arg2>::getTuple(cgh, eval2));
  }
  template<typename Arg1, typename Arg2, typename Arg3>	static inline auto getTuple(cl::sycl::handler& cgh, Arg1 eval1 , Arg2 eval2 , Arg3 eval3)
  -> decltype(utility::tuple::append(ExtractAccessor<Arg1>::getTuple(cgh, eval1),utility::tuple::append(ExtractAccessor<Arg2>::getTuple(cgh, eval2), ExtractAccessor<Arg3>::getTuple(cgh, eval3)))) {
    return utility::tuple::append(ExtractAccessor<Arg1>::getTuple(cgh, eval1),utility::tuple::append(ExtractAccessor<Arg2>::getTuple(cgh, eval2), ExtractAccessor<Arg3>::getTuple(cgh, eval3)));
  }
  template< cl::sycl::access::mode AcM, typename Arg> static inline auto getAccessor(cl::sycl::handler& cgh, Arg eval)
  -> decltype(utility::tuple::make_tuple( eval.device().template get_sycl_accessor<AcM,
  typename Eigen::internal::remove_all<typename Arg::CoeffReturnType>::type>(eval.dimensions().TotalSize(), cgh,eval.data()))){
    return utility::tuple::make_tuple(eval.device().template get_sycl_accessor<AcM, typename Eigen::internal::remove_all<typename Arg::CoeffReturnType>::type>(eval.dimensions().TotalSize(), cgh,eval.data()));
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// const TensorCwiseNullaryOp, const TensorCwiseUnaryOp and const TensorBroadcastingOp
template <template<class, class> class UnaryCategory, typename OP, typename RHSExpr, typename Dev>
struct ExtractAccessor<TensorEvaluator<const UnaryCategory<OP, RHSExpr>, Dev> > {
  static inline auto getTuple(cl::sycl::handler& cgh, const TensorEvaluator<const UnaryCategory<OP, RHSExpr>, Dev> eval)
  -> decltype(AccessorConstructor::getTuple(cgh, eval.impl())){
    return AccessorConstructor::getTuple(cgh, eval.impl());
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is TensorCwiseNullaryOp,  TensorCwiseUnaryOp and  TensorBroadcastingOp
template <template<class, class> class UnaryCategory, typename OP, typename RHSExpr, typename Dev>
struct ExtractAccessor<TensorEvaluator<UnaryCategory<OP, RHSExpr>, Dev> >
: ExtractAccessor<TensorEvaluator<const UnaryCategory<OP, RHSExpr>, Dev> > {};

/// specialisation of the \ref ExtractAccessor struct when the node type is const TensorCwiseBinaryOp
template <template<class, class, class> class BinaryCategory, typename OP,  typename LHSExpr, typename RHSExpr, typename Dev>
struct ExtractAccessor<TensorEvaluator<const BinaryCategory<OP, LHSExpr, RHSExpr>, Dev> > {
  static inline auto getTuple(cl::sycl::handler& cgh, const TensorEvaluator<const BinaryCategory<OP, LHSExpr, RHSExpr>, Dev> eval)
  -> decltype(AccessorConstructor::getTuple(cgh, eval.left_impl(), eval.right_impl())){
    return AccessorConstructor::getTuple(cgh, eval.left_impl(), eval.right_impl());
  }
};
/// specialisation of the \ref ExtractAccessor struct when the node type is TensorCwiseBinaryOp
template <template<class, class, class> class BinaryCategory, typename OP,  typename LHSExpr, typename RHSExpr, typename Dev>
struct ExtractAccessor<TensorEvaluator<BinaryCategory<OP, LHSExpr, RHSExpr>, Dev> >
: ExtractAccessor<TensorEvaluator<const BinaryCategory<OP, LHSExpr, RHSExpr>, Dev> >{};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// const TensorCwiseTernaryOp
template <template<class, class, class, class> class TernaryCategory, typename OP, typename Arg1Expr, typename Arg2Expr, typename Arg3Expr, typename Dev>
struct ExtractAccessor<TensorEvaluator<const TernaryCategory<OP, Arg1Expr, Arg2Expr, Arg3Expr>, Dev> > {
  static inline auto getTuple(cl::sycl::handler& cgh, const TensorEvaluator<const TernaryCategory<OP, Arg1Expr, Arg2Expr, Arg3Expr>, Dev> eval)
  -> decltype(AccessorConstructor::getTuple(cgh, eval.arg1Impl(), eval.arg2Impl(), eval.arg3Impl())){
    return AccessorConstructor::getTuple(cgh, eval.arg1Impl(), eval.arg2Impl(), eval.arg3Impl());
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is TensorCwiseTernaryOp
template <template<class, class, class, class> class TernaryCategory, typename OP, typename Arg1Expr, typename Arg2Expr, typename Arg3Expr, typename Dev>
struct ExtractAccessor<TensorEvaluator<TernaryCategory<OP, Arg1Expr, Arg2Expr, Arg3Expr>, Dev> >
: ExtractAccessor<TensorEvaluator<const TernaryCategory<OP, Arg1Expr, Arg2Expr, Arg3Expr>, Dev> >{};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// const TensorCwiseSelectOp. This is a special case where there is no OP
template <typename IfExpr, typename ThenExpr, typename ElseExpr, typename Dev>
struct ExtractAccessor<TensorEvaluator<const TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, Dev> > {
  static inline auto getTuple(cl::sycl::handler& cgh, const TensorEvaluator<const TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, Dev> eval)
  -> decltype(AccessorConstructor::getTuple(cgh, eval.cond_impl(), eval.then_impl(), eval.else_impl())){
    return AccessorConstructor::getTuple(cgh, eval.cond_impl(), eval.then_impl(), eval.else_impl());
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// TensorCwiseSelectOp. This is a special case where there is no OP
template <typename IfExpr, typename ThenExpr, typename ElseExpr, typename Dev>
struct ExtractAccessor<TensorEvaluator<TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, Dev> >
: ExtractAccessor<TensorEvaluator<const TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, Dev> >{};

/// specialisation of the \ref ExtractAccessor struct when the node type is const TensorAssignOp
template <typename LHSExpr, typename RHSExpr, typename Dev>
struct ExtractAccessor<TensorEvaluator<const TensorAssignOp<LHSExpr, RHSExpr>, Dev> > {
  static inline auto getTuple(cl::sycl::handler& cgh, const TensorEvaluator<const TensorAssignOp<LHSExpr, RHSExpr>, Dev> eval)
  -> decltype(AccessorConstructor::getTuple(cgh, eval.left_impl(), eval.right_impl())){
    return AccessorConstructor::getTuple(cgh, eval.left_impl(), eval.right_impl());
 }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is TensorAssignOp
template <typename LHSExpr, typename RHSExpr, typename Dev>
struct ExtractAccessor<TensorEvaluator<TensorAssignOp<LHSExpr, RHSExpr>, Dev> >
: ExtractAccessor<TensorEvaluator<const TensorAssignOp<LHSExpr, RHSExpr>, Dev> >{};

/// specialisation of the \ref ExtractAccessor struct when the node type is const TensorMap
#define TENSORMAPEXPR(CVQual, ACCType)\
template <typename PlainObjectType, int Options_, typename Dev>\
struct ExtractAccessor<TensorEvaluator<CVQual TensorMap<PlainObjectType, Options_>, Dev> > {\
  static inline auto getTuple(cl::sycl::handler& cgh,const TensorEvaluator<CVQual TensorMap<PlainObjectType, Options_>, Dev> eval)\
  -> decltype(AccessorConstructor::template getAccessor<ACCType>(cgh, eval)){\
    return AccessorConstructor::template getAccessor<ACCType>(cgh, eval);\
  }\
};
TENSORMAPEXPR(const, cl::sycl::access::mode::read)
TENSORMAPEXPR(, cl::sycl::access::mode::read_write)
#undef TENSORMAPEXPR

/// specialisation of the \ref ExtractAccessor struct when the node type is const TensorForcedEvalOp
template <typename Expr, typename Dev>
struct ExtractAccessor<TensorEvaluator<const TensorForcedEvalOp<Expr>, Dev> > {
  static inline auto getTuple(cl::sycl::handler& cgh, const TensorEvaluator<const TensorForcedEvalOp<Expr>, Dev> eval)
  -> decltype(AccessorConstructor::template getAccessor<cl::sycl::access::mode::read>(cgh, eval)){
    return AccessorConstructor::template getAccessor<cl::sycl::access::mode::read>(cgh, eval);
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is TensorForcedEvalOp
template <typename Expr, typename Dev>
struct ExtractAccessor<TensorEvaluator<TensorForcedEvalOp<Expr>, Dev> >
: ExtractAccessor<TensorEvaluator<const TensorForcedEvalOp<Expr>, Dev> >{};

/// specialisation of the \ref ExtractAccessor struct when the node type is const TensorEvalToOp
template <typename Expr, typename Dev>
struct ExtractAccessor<TensorEvaluator<const TensorEvalToOp<Expr>, Dev> > {
  static inline auto getTuple(cl::sycl::handler& cgh,const TensorEvaluator<const TensorEvalToOp<Expr>, Dev> eval)
  -> decltype(utility::tuple::append(AccessorConstructor::template getAccessor<cl::sycl::access::mode::write>(cgh, eval), AccessorConstructor::getTuple(cgh, eval.impl()))){
    return utility::tuple::append(AccessorConstructor::template getAccessor<cl::sycl::access::mode::write>(cgh, eval), AccessorConstructor::getTuple(cgh, eval.impl()));
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is TensorEvalToOp
template <typename Expr, typename Dev>
struct ExtractAccessor<TensorEvaluator<TensorEvalToOp<Expr>, Dev> >
: ExtractAccessor<TensorEvaluator<const TensorEvalToOp<Expr>, Dev> >{};

/// specialisation of the \ref ExtractAccessor struct when the node type is const TensorReductionOp
template <typename OP, typename Dim, typename Expr, typename Dev>
struct ExtractAccessor<TensorEvaluator<const TensorReductionOp<OP, Dim, Expr>, Dev> > {
  static inline auto getTuple(cl::sycl::handler& cgh, const TensorEvaluator<const TensorReductionOp<OP, Dim, Expr>, Dev> eval)
  -> decltype(AccessorConstructor::template getAccessor<cl::sycl::access::mode::read>(cgh, eval)){
    return AccessorConstructor::template getAccessor<cl::sycl::access::mode::read>(cgh, eval);
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is TensorReductionOp
template <typename OP, typename Dim, typename Expr, typename Dev>
struct ExtractAccessor<TensorEvaluator<TensorReductionOp<OP, Dim, Expr>, Dev> >
: ExtractAccessor<TensorEvaluator<const TensorReductionOp<OP, Dim, Expr>, Dev> >{};

/// template deduction for \ref ExtractAccessor
template <typename Evaluator>
auto createTupleOfAccessors(cl::sycl::handler& cgh, const Evaluator& expr)
-> decltype(ExtractAccessor<Evaluator>::getTuple(cgh, expr)) {
  return ExtractAccessor<Evaluator>::getTuple(cgh, expr);
}

} /// namespace TensorSycl
} /// namespace internal
} /// namespace Eigen
#endif  // UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_EXTRACT_ACCESSOR_HPP
