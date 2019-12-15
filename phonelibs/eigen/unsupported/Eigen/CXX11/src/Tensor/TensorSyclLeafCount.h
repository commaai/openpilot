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
 * TensorSyclLeafCount.h
 *
 * \brief:
 *  The leaf count used the pre-order expression tree traverse in order to name
 *  count the number of leaf nodes in the expression
 *
*****************************************************************/

#ifndef UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_LEAF_COUNT_HPP
#define UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_LEAF_COUNT_HPP

namespace Eigen {
namespace TensorSycl {
namespace internal {
/// \brief LeafCount used to counting terminal nodes. The total number of
/// leaf nodes is used by MakePlaceHolderExprHelper to find the order
/// of the leaf node in a expression tree at compile time.
template <typename Expr>
struct LeafCount;

template<typename... Args> struct CategoryCount;

template<> struct CategoryCount<>
{
  static const size_t Count =0;
};

template<typename Arg, typename... Args>
struct CategoryCount<Arg,Args...>{
  static const size_t Count = LeafCount<Arg>::Count + CategoryCount<Args...>::Count;
};

/// specialisation of the \ref LeafCount struct when the node type is const TensorMap
template <typename PlainObjectType, int Options_, template <class> class MakePointer_>
struct LeafCount<const TensorMap<PlainObjectType, Options_, MakePointer_> > {
  static const size_t Count =1;
};

/// specialisation of the \ref LeafCount struct when the node type is TensorMap
template <typename PlainObjectType, int Options_, template <class> class MakePointer_>
struct LeafCount<TensorMap<PlainObjectType, Options_, MakePointer_> > :LeafCount<const TensorMap<PlainObjectType, Options_, MakePointer_> >{};

// const TensorCwiseUnaryOp, const TensorCwiseNullaryOp, const TensorCwiseBinaryOp, const TensorCwiseTernaryOp, and Const TensorBroadcastingOp
template <template <class, class...> class CategoryExpr, typename OP, typename... RHSExpr>
struct LeafCount<const CategoryExpr<OP, RHSExpr...> >: CategoryCount<RHSExpr...> {};
// TensorCwiseUnaryOp,  TensorCwiseNullaryOp,  TensorCwiseBinaryOp,  TensorCwiseTernaryOp, and  TensorBroadcastingOp
template <template <class, class...> class CategoryExpr, typename OP, typename... RHSExpr>
struct LeafCount<CategoryExpr<OP, RHSExpr...> > :LeafCount<const CategoryExpr<OP, RHSExpr...> >{};

/// specialisation of the \ref LeafCount struct when the node type is const TensorSelectOp is an exception
template <typename IfExpr, typename ThenExpr, typename ElseExpr>
struct LeafCount<const TensorSelectOp<IfExpr, ThenExpr, ElseExpr> > : CategoryCount<IfExpr, ThenExpr, ElseExpr> {};
/// specialisation of the \ref LeafCount struct when the node type is TensorSelectOp
template <typename IfExpr, typename ThenExpr, typename ElseExpr>
struct LeafCount<TensorSelectOp<IfExpr, ThenExpr, ElseExpr> >: LeafCount<const TensorSelectOp<IfExpr, ThenExpr, ElseExpr> > {};


/// specialisation of the \ref LeafCount struct when the node type is const TensorAssignOp
template <typename LHSExpr, typename RHSExpr>
struct LeafCount<const TensorAssignOp<LHSExpr, RHSExpr> >: CategoryCount<LHSExpr,RHSExpr> {};

/// specialisation of the \ref LeafCount struct when the node type is
/// TensorAssignOp is an exception. It is not the same as Unary
template <typename LHSExpr, typename RHSExpr>
struct LeafCount<TensorAssignOp<LHSExpr, RHSExpr> > :LeafCount<const TensorAssignOp<LHSExpr, RHSExpr> >{};

/// specialisation of the \ref LeafCount struct when the node type is const TensorForcedEvalOp
template <typename Expr>
struct LeafCount<const TensorForcedEvalOp<Expr> > {
    static const size_t Count =1;
};

/// specialisation of the \ref LeafCount struct when the node type is TensorForcedEvalOp
template <typename Expr>
struct LeafCount<TensorForcedEvalOp<Expr> >: LeafCount<const TensorForcedEvalOp<Expr> > {};

/// specialisation of the \ref LeafCount struct when the node type is const TensorEvalToOp
template <typename Expr>
struct LeafCount<const TensorEvalToOp<Expr> > {
  static const size_t Count = 1 + CategoryCount<Expr>::Count;
};

/// specialisation of the \ref LeafCount struct when the node type is const TensorReductionOp
template <typename OP, typename Dim, typename Expr>
struct LeafCount<const TensorReductionOp<OP, Dim, Expr> > {
    static const size_t Count =1;
};

/// specialisation of the \ref LeafCount struct when the node type is TensorReductionOp
template <typename OP, typename Dim, typename Expr>
struct LeafCount<TensorReductionOp<OP, Dim, Expr> >: LeafCount<const TensorReductionOp<OP, Dim, Expr> >{};

/// specialisation of the \ref LeafCount struct when the node type is TensorEvalToOp
template <typename Expr>
struct LeafCount<TensorEvalToOp<Expr> >: LeafCount<const TensorEvalToOp<Expr> >{};

} /// namespace TensorSycl
} /// namespace internal
} /// namespace Eigen

#endif  // UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_LEAF_COUNT_HPP
