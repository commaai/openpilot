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

#ifndef UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_PLACEHOLDER_EXPR_HPP
#define UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_PLACEHOLDER_EXPR_HPP

namespace Eigen {
namespace TensorSycl {
namespace internal {

/// \struct PlaceHolder
/// \brief PlaceHolder is used to replace the \ref TensorMap in the expression
/// tree.
/// PlaceHolder contains the order of the leaf node in the expression tree.
template <typename Scalar, size_t N>
struct PlaceHolder {
  static constexpr size_t I = N;
  typedef Scalar Type;
};

/// \sttruct PlaceHolderExpression
/// \brief it is used to create the PlaceHolder expression. The PlaceHolder
/// expression is a copy of expression type in which the TensorMap of the has
/// been replaced with PlaceHolder.
template <typename Expr, size_t N>
struct PlaceHolderExpression;

template<size_t N, typename... Args>
struct CalculateIndex;

template<size_t N, typename Arg>
struct CalculateIndex<N, Arg>{
  typedef typename PlaceHolderExpression<Arg, N>::Type ArgType;
  typedef utility::tuple::Tuple<ArgType> ArgsTuple;
};

template<size_t N, typename Arg1, typename Arg2>
struct CalculateIndex<N, Arg1, Arg2>{
  static const size_t Arg2LeafCount = LeafCount<Arg2>::Count;
  typedef typename PlaceHolderExpression<Arg1, N - Arg2LeafCount>::Type Arg1Type;
  typedef typename PlaceHolderExpression<Arg2, N>::Type Arg2Type;
  typedef utility::tuple::Tuple<Arg1Type, Arg2Type> ArgsTuple;
};

template<size_t N, typename Arg1, typename Arg2, typename Arg3>
struct CalculateIndex<N, Arg1, Arg2, Arg3> {
  static const size_t Arg3LeafCount = LeafCount<Arg3>::Count;
  static const size_t Arg2LeafCount = LeafCount<Arg2>::Count;
  typedef typename PlaceHolderExpression<Arg1, N - Arg3LeafCount - Arg2LeafCount>::Type Arg1Type;
  typedef typename PlaceHolderExpression<Arg2, N - Arg3LeafCount>::Type Arg2Type;
  typedef typename PlaceHolderExpression<Arg3, N>::Type Arg3Type;
  typedef utility::tuple::Tuple<Arg1Type, Arg2Type, Arg3Type> ArgsTuple;
};

template<template<class...> class Category , class OP, class TPL>
struct CategoryHelper;

template<template<class...> class Category , class OP, class ...T >
struct CategoryHelper<Category, OP, utility::tuple::Tuple<T...> > {
  typedef Category<OP, T... > Type;
};

template<template<class...> class Category , class ...T >
struct CategoryHelper<Category, NoOP, utility::tuple::Tuple<T...> > {
  typedef Category<T... > Type;
};

/// specialisation of the \ref PlaceHolderExpression when the node is
/// TensorCwiseNullaryOp, TensorCwiseUnaryOp, TensorBroadcastingOp, TensorCwiseBinaryOp,  TensorCwiseTernaryOp
#define OPEXPRCATEGORY(CVQual)\
template <template <class, class... > class Category, typename OP, typename... SubExpr, size_t N>\
struct PlaceHolderExpression<CVQual Category<OP, SubExpr...>, N>{\
  typedef CVQual typename CategoryHelper<Category, OP, typename CalculateIndex<N, SubExpr...>::ArgsTuple>::Type Type;\
};

OPEXPRCATEGORY(const)
OPEXPRCATEGORY()
#undef OPEXPRCATEGORY

/// specialisation of the \ref PlaceHolderExpression when the node is
/// TensorCwiseSelectOp
#define SELECTEXPR(CVQual)\
template <typename IfExpr, typename ThenExpr, typename ElseExpr, size_t N>\
struct PlaceHolderExpression<CVQual TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, N> {\
  typedef CVQual typename CategoryHelper<TensorSelectOp, NoOP, typename CalculateIndex<N, IfExpr, ThenExpr, ElseExpr>::ArgsTuple>::Type Type;\
};

SELECTEXPR(const)
SELECTEXPR()
#undef SELECTEXPR

/// specialisation of the \ref PlaceHolderExpression when the node is
/// TensorAssignOp
#define ASSIGNEXPR(CVQual)\
template <typename LHSExpr, typename RHSExpr, size_t N>\
struct PlaceHolderExpression<CVQual TensorAssignOp<LHSExpr, RHSExpr>, N> {\
  typedef CVQual typename CategoryHelper<TensorAssignOp, NoOP, typename CalculateIndex<N, LHSExpr, RHSExpr>::ArgsTuple>::Type Type;\
};

ASSIGNEXPR(const)
ASSIGNEXPR()
#undef ASSIGNEXPR

/// specialisation of the \ref PlaceHolderExpression when the node is
/// TensorMap
#define TENSORMAPEXPR(CVQual)\
template <typename Scalar_, int Options_, int Options2_, int NumIndices_, typename IndexType_, template <class> class MakePointer_, size_t N>\
struct PlaceHolderExpression< CVQual TensorMap< Tensor<Scalar_, NumIndices_, Options_, IndexType_>, Options2_, MakePointer_>, N> {\
  typedef CVQual PlaceHolder<CVQual TensorMap<Tensor<Scalar_, NumIndices_, Options_, IndexType_>, Options2_, MakePointer_>, N> Type;\
};

TENSORMAPEXPR(const)
TENSORMAPEXPR()
#undef TENSORMAPEXPR

/// specialisation of the \ref PlaceHolderExpression when the node is
/// TensorForcedEvalOp
#define FORCEDEVAL(CVQual)\
template <typename Expr, size_t N>\
struct PlaceHolderExpression<CVQual TensorForcedEvalOp<Expr>, N> {\
  typedef CVQual PlaceHolder<CVQual TensorForcedEvalOp<Expr>, N> Type;\
};

FORCEDEVAL(const)
FORCEDEVAL()
#undef FORCEDEVAL

/// specialisation of the \ref PlaceHolderExpression when the node is
/// TensorEvalToOp
#define EVALTO(CVQual)\
template <typename Expr, size_t N>\
struct PlaceHolderExpression<CVQual TensorEvalToOp<Expr>, N> {\
  typedef CVQual TensorEvalToOp<typename CalculateIndex <N, Expr>::ArgType> Type;\
};

EVALTO(const)
EVALTO()
#undef EVALTO


/// specialisation of the \ref PlaceHolderExpression when the node is
/// TensorReductionOp
#define SYCLREDUCTION(CVQual)\
template <typename OP, typename Dims, typename Expr, size_t N>\
struct PlaceHolderExpression<CVQual TensorReductionOp<OP, Dims, Expr>, N>{\
  typedef CVQual PlaceHolder<CVQual TensorReductionOp<OP, Dims,Expr>, N> Type;\
};
SYCLREDUCTION(const)
SYCLREDUCTION()
#undef SYCLREDUCTION

/// template deduction for \ref PlaceHolderExpression struct
template <typename Expr>
struct createPlaceHolderExpression {
  static const size_t TotalLeaves = LeafCount<Expr>::Count;
  typedef typename PlaceHolderExpression<Expr, TotalLeaves - 1>::Type Type;
};

}  // internal
}  // TensorSycl
}  // namespace Eigen

#endif  // UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_PLACEHOLDER_EXPR_HPP
