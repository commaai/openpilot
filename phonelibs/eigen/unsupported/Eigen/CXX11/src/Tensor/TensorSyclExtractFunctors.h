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
 * TensorSyclextractFunctors.h
 *
 * \brief:
 *  Used to extract all the functors allocated to each node of the expression
*tree.
 *
*****************************************************************/

#ifndef UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_EXTRACT_FUNCTORS_HPP
#define UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_EXTRACT_FUNCTORS_HPP

namespace Eigen {
namespace TensorSycl {
namespace internal {
/// \struct FunctorExtractor:  This struct is used to extract the functors
/// constructed on
/// the host-side, to pack them and reuse them in reconstruction of the
/// expression on the device.
/// We have to do that as in Eigen the functors are not stateless so we cannot
/// re-instantiate them on the device.
/// We have to pass instantiated functors to the device.
// This struct is used for leafNode (TensorMap) and nodes behaving like leafNode (TensorForcedEval).
template <typename Evaluator> struct FunctorExtractor{
  typedef typename Evaluator::Dimensions Dimensions;
  const Dimensions m_dimensions;
  const Dimensions& dimensions() const { return m_dimensions; }
  FunctorExtractor(const Evaluator& expr)
  : m_dimensions(expr.dimensions()) {}

};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// const TensorCwiseNullaryOp, const TensorCwiseUnaryOp, and const TensorBroadcastingOp
template <template <class, class> class UnaryCategory, typename OP, typename RHSExpr, typename Dev>
struct FunctorExtractor<TensorEvaluator<const UnaryCategory<OP, RHSExpr>, Dev> > {
  FunctorExtractor<TensorEvaluator<RHSExpr, Dev> > rhsExpr;
  OP func;
  FunctorExtractor(const TensorEvaluator<const UnaryCategory<OP, RHSExpr>, Dev>& expr)
  : rhsExpr(expr.impl()), func(expr.functor()) {}
};
/// specialisation of the \ref FunctorExtractor struct when the node type is
/// TensorCwiseNullaryOp, TensorCwiseUnaryOp, and TensorBroadcastingOp
template <template <class, class> class UnaryCategory, typename OP, typename RHSExpr, typename Dev>
struct FunctorExtractor<TensorEvaluator<UnaryCategory<OP, RHSExpr>, Dev> >
: FunctorExtractor<TensorEvaluator<const UnaryCategory<OP, RHSExpr>, Dev> >{};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// const TensorCwiseBinaryOp
template <template<class, class, class> class BinaryCategory, typename OP, typename LHSExpr, typename RHSExpr, typename Dev>
struct FunctorExtractor<TensorEvaluator<const BinaryCategory<OP, LHSExpr, RHSExpr>, Dev> > {
  FunctorExtractor<TensorEvaluator<LHSExpr, Dev> > lhsExpr;
  FunctorExtractor<TensorEvaluator<RHSExpr, Dev> > rhsExpr;
  OP func;
  FunctorExtractor(const TensorEvaluator<const BinaryCategory<OP, LHSExpr, RHSExpr>, Dev>& expr)
  : lhsExpr(expr.left_impl()),rhsExpr(expr.right_impl()),func(expr.functor()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// const TensorCwiseBinaryOp
template <template <class, class, class> class BinaryCategory, typename OP, typename LHSExpr, typename RHSExpr, typename Dev>
struct FunctorExtractor<TensorEvaluator<BinaryCategory<OP,  LHSExpr, RHSExpr>, Dev> >
: FunctorExtractor<TensorEvaluator<const BinaryCategory<OP,  LHSExpr, RHSExpr>, Dev> >{};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// const TensorCwiseTernaryOp
template <template <class, class, class, class> class TernaryCategory, typename OP, typename Arg1Expr, typename Arg2Expr, typename Arg3Expr,typename Dev>
struct FunctorExtractor<TensorEvaluator<const TernaryCategory<OP, Arg1Expr, Arg2Expr, Arg3Expr>, Dev> > {
  FunctorExtractor<TensorEvaluator<Arg1Expr, Dev> > arg1Expr;
  FunctorExtractor<TensorEvaluator<Arg2Expr, Dev> > arg2Expr;
  FunctorExtractor<TensorEvaluator<Arg3Expr, Dev> > arg3Expr;
  OP func;
  FunctorExtractor(const TensorEvaluator<const TernaryCategory<OP, Arg1Expr, Arg2Expr, Arg3Expr>, Dev>& expr)
  : arg1Expr(expr.arg1Impl()), arg2Expr(expr.arg2Impl()), arg3Expr(expr.arg3Impl()), func(expr.functor()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// TensorCwiseTernaryOp
template <template <class, class, class, class> class TernaryCategory, typename OP, typename Arg1Expr, typename Arg2Expr, typename Arg3Expr, typename Dev>
struct FunctorExtractor<TensorEvaluator< TernaryCategory<OP, Arg1Expr, Arg2Expr, Arg3Expr>, Dev> >
:FunctorExtractor<TensorEvaluator<const TernaryCategory<OP, Arg1Expr, Arg2Expr, Arg3Expr>, Dev> >{};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// const TensorCwiseSelectOp. This is an specialisation without OP so it has to be separated.
template <typename IfExpr, typename ThenExpr, typename ElseExpr, typename Dev>
struct FunctorExtractor< TensorEvaluator<const TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, Dev> > {
  FunctorExtractor<TensorEvaluator<IfExpr, Dev> > ifExpr;
  FunctorExtractor<TensorEvaluator<ThenExpr, Dev> > thenExpr;
  FunctorExtractor<TensorEvaluator<ElseExpr, Dev> > elseExpr;
  FunctorExtractor(const TensorEvaluator<const TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, Dev>& expr)
  : ifExpr(expr.cond_impl()), thenExpr(expr.then_impl()), elseExpr(expr.else_impl()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// TensorCwiseSelectOp. This is an specialisation without OP so it has to be separated
template <typename IfExpr, typename ThenExpr, typename ElseExpr, typename Dev>
struct FunctorExtractor<TensorEvaluator<TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, Dev> >
:FunctorExtractor< TensorEvaluator<const TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, Dev> > {};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// const TensorAssignOp. This is an specialisation without OP so it has to be separated.
template <typename LHSExpr, typename RHSExpr, typename Dev>
struct FunctorExtractor<TensorEvaluator<const TensorAssignOp<LHSExpr, RHSExpr>, Dev> > {
  FunctorExtractor<TensorEvaluator<LHSExpr, Dev> > lhsExpr;
  FunctorExtractor<TensorEvaluator<RHSExpr, Dev> > rhsExpr;
  FunctorExtractor(const TensorEvaluator<const TensorAssignOp<LHSExpr, RHSExpr>, Dev>& expr)
  : lhsExpr(expr.left_impl()), rhsExpr(expr.right_impl()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// TensorAssignOp. This is an specialisation without OP so it has to be separated.
template <typename LHSExpr, typename RHSExpr, typename Dev>
struct FunctorExtractor<TensorEvaluator<TensorAssignOp<LHSExpr, RHSExpr>, Dev> >
:FunctorExtractor<TensorEvaluator<const TensorAssignOp<LHSExpr, RHSExpr>, Dev> >{};


/// specialisation of the \ref FunctorExtractor struct when the node type is
/// const TensorEvalToOp, This is an specialisation without OP so it has to be separated.
template <typename RHSExpr, typename Dev>
struct FunctorExtractor<TensorEvaluator<const TensorEvalToOp<RHSExpr>, Dev> > {
  FunctorExtractor<TensorEvaluator<RHSExpr, Dev> > rhsExpr;
  FunctorExtractor(const TensorEvaluator<const TensorEvalToOp<RHSExpr>, Dev>& expr)
  : rhsExpr(expr.impl()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// TensorEvalToOp. This is a specialisation without OP so it has to be separated.
template <typename RHSExpr, typename Dev>
struct FunctorExtractor<TensorEvaluator<TensorEvalToOp<RHSExpr>, Dev> >
: FunctorExtractor<TensorEvaluator<const TensorEvalToOp<RHSExpr>, Dev> > {};

template<typename Dim, size_t NumOutputDim> struct DimConstr {
template<typename InDim>
  static inline Dim getDim(InDim dims ) {return dims;}
};

template<typename Dim> struct DimConstr<Dim, 0> {
  template<typename InDim>
    static inline Dim getDim(InDim dims ) {return Dim(dims.TotalSize());}
};

template<typename Op, typename Dims, typename ArgType, template <class> class MakePointer_, typename Device>
struct FunctorExtractor<TensorEvaluator<const TensorReductionOp<Op, Dims, ArgType, MakePointer_>, Device>>{
  typedef TensorEvaluator<const TensorReductionOp<Op, Dims, ArgType, MakePointer_>, Device> Evaluator;
  typedef typename Eigen::internal::conditional<Evaluator::NumOutputDims==0, DSizes<typename Evaluator::Index, 1>, typename Evaluator::Dimensions >::type Dimensions;
  const Dimensions m_dimensions;
  const Dimensions& dimensions() const { return m_dimensions; }
  FunctorExtractor(const TensorEvaluator<const TensorReductionOp<Op, Dims, ArgType, MakePointer_>, Device>& expr)
  : m_dimensions(DimConstr<Dimensions, Evaluator::NumOutputDims>::getDim(expr.dimensions())) {}
};


template<typename Op, typename Dims, typename ArgType, template <class> class MakePointer_, typename Device>
struct FunctorExtractor<TensorEvaluator<TensorReductionOp<Op, Dims, ArgType, MakePointer_>, Device>>
: FunctorExtractor<TensorEvaluator<const TensorReductionOp<Op, Dims, ArgType, MakePointer_>, Device>>{};
/// template deduction function for FunctorExtractor
template <typename Evaluator>
auto inline extractFunctors(const Evaluator& evaluator)-> FunctorExtractor<Evaluator> {
  return FunctorExtractor<Evaluator>(evaluator);
}
}  // namespace internal
}  // namespace TensorSycl
}  // namespace Eigen

#endif  // UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_EXTRACT_FUNCTORS_HPP
