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
 * TensorSyclExprConstructor.h
 *
 * \brief:
 *  This file re-create an expression on the SYCL device in order
 *  to use the original tensor evaluator.
 *
*****************************************************************/

#ifndef UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_EXPR_CONSTRUCTOR_HPP
#define UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_EXPR_CONSTRUCTOR_HPP

namespace Eigen {
namespace TensorSycl {
namespace internal {
/// this class is used by EvalToOp in order to create an lhs expression which is
/// a pointer from an accessor on device-only buffer
template <typename PtrType, size_t N, typename... Params>
struct EvalToLHSConstructor {
  PtrType expr;
  EvalToLHSConstructor(const utility::tuple::Tuple<Params...> &t): expr((&(*(utility::tuple::get<N>(t).get_pointer())))) {}
};

/// \struct ExprConstructor is used to reconstruct the expression on the device and
/// recreate the expression with MakeGlobalPointer containing the device address
/// space for the TensorMap pointers used in eval function.
/// It receives the original expression type, the functor of the node, the tuple
/// of accessors, and the device expression type to re-instantiate the
/// expression tree for the device
template <typename OrigExpr, typename IndexExpr, typename... Params>
struct ExprConstructor;

/// specialisation of the \ref ExprConstructor struct when the node type is
/// TensorMap
#define TENSORMAP(CVQual)\
template <typename Scalar_, int Options_, int Options2_, int Options3_, int NumIndices_, typename IndexType_,\
template <class> class MakePointer_, size_t N, typename... Params>\
struct ExprConstructor< CVQual TensorMap<Tensor<Scalar_, NumIndices_, Options_, IndexType_>, Options2_, MakeGlobalPointer>,\
CVQual PlaceHolder<CVQual TensorMap<Tensor<Scalar_, NumIndices_, Options_, IndexType_>, Options3_, MakePointer_>, N>, Params...>{\
  typedef  CVQual TensorMap<Tensor<Scalar_, NumIndices_, Options_, IndexType_>, Options2_, MakeGlobalPointer>  Type;\
  Type expr;\
  template <typename FuncDetector>\
  ExprConstructor(FuncDetector &fd, const utility::tuple::Tuple<Params...> &t)\
  : expr(Type((&(*(utility::tuple::get<N>(t).get_pointer()))), fd.dimensions())) {}\
};

TENSORMAP(const)
TENSORMAP()
#undef TENSORMAP

#define UNARYCATEGORY(CVQual)\
template <template<class, class> class UnaryCategory, typename OP, typename OrigRHSExpr, typename RHSExpr, typename... Params>\
struct ExprConstructor<CVQual UnaryCategory<OP, OrigRHSExpr>, CVQual UnaryCategory<OP, RHSExpr>, Params...> {\
  typedef  ExprConstructor<OrigRHSExpr, RHSExpr, Params...> my_type;\
  my_type rhsExpr;\
  typedef CVQual UnaryCategory<OP, typename my_type::Type> Type;\
  Type expr;\
  template <typename FuncDetector>\
  ExprConstructor(FuncDetector &funcD, const utility::tuple::Tuple<Params...> &t)\
  : rhsExpr(funcD.rhsExpr, t), expr(rhsExpr.expr, funcD.func) {}\
};

UNARYCATEGORY(const)
UNARYCATEGORY()
#undef UNARYCATEGORY

/// specialisation of the \ref ExprConstructor struct when the node type is
/// TensorBinaryOp
#define BINARYCATEGORY(CVQual)\
template <template<class, class, class> class BinaryCategory, typename OP, typename OrigLHSExpr, typename OrigRHSExpr, typename LHSExpr,\
typename RHSExpr, typename... Params>\
struct ExprConstructor<CVQual BinaryCategory<OP, OrigLHSExpr, OrigRHSExpr>,  CVQual BinaryCategory<OP, LHSExpr, RHSExpr>, Params...> {\
  typedef  ExprConstructor<OrigLHSExpr, LHSExpr, Params...> my_left_type;\
  typedef  ExprConstructor<OrigRHSExpr, RHSExpr, Params...> my_right_type;\
  typedef  CVQual BinaryCategory<OP, typename my_left_type::Type, typename my_right_type::Type> Type;\
  my_left_type lhsExpr;\
  my_right_type rhsExpr;\
  Type expr;\
  template <typename FuncDetector>\
  ExprConstructor(FuncDetector &funcD, const utility::tuple::Tuple<Params...> &t)\
  : lhsExpr(funcD.lhsExpr, t),rhsExpr(funcD.rhsExpr, t), expr(lhsExpr.expr, rhsExpr.expr, funcD.func) {}\
};

BINARYCATEGORY(const)
BINARYCATEGORY()
#undef BINARYCATEGORY

/// specialisation of the \ref ExprConstructor struct when the node type is
/// TensorCwiseTernaryOp
#define TERNARYCATEGORY(CVQual)\
template <template <class, class, class, class> class TernaryCategory, typename OP, typename OrigArg1Expr, typename OrigArg2Expr,typename OrigArg3Expr,\
typename Arg1Expr, typename Arg2Expr, typename Arg3Expr, typename... Params>\
struct ExprConstructor<CVQual TernaryCategory<OP, OrigArg1Expr, OrigArg2Expr, OrigArg3Expr>, CVQual TernaryCategory<OP, Arg1Expr, Arg2Expr, Arg3Expr>, Params...> {\
  typedef ExprConstructor<OrigArg1Expr, Arg1Expr, Params...> my_arg1_type;\
  typedef ExprConstructor<OrigArg2Expr, Arg2Expr, Params...> my_arg2_type;\
  typedef ExprConstructor<OrigArg3Expr, Arg3Expr, Params...> my_arg3_type;\
  typedef  CVQual TernaryCategory<OP, typename my_arg1_type::Type, typename my_arg2_type::Type, typename my_arg3_type::Type> Type;\
  my_arg1_type arg1Expr;\
  my_arg2_type arg2Expr;\
  my_arg3_type arg3Expr;\
  Type expr;\
  template <typename FuncDetector>\
  ExprConstructor(FuncDetector &funcD,const utility::tuple::Tuple<Params...> &t)\
  : arg1Expr(funcD.arg1Expr, t), arg2Expr(funcD.arg2Expr, t), arg3Expr(funcD.arg3Expr, t), expr(arg1Expr.expr, arg2Expr.expr, arg3Expr.expr, funcD.func) {}\
};

TERNARYCATEGORY(const)
TERNARYCATEGORY()
#undef TERNARYCATEGORY

/// specialisation of the \ref ExprConstructor struct when the node type is
/// TensorCwiseSelectOp
#define SELECTOP(CVQual)\
template <typename OrigIfExpr, typename OrigThenExpr, typename OrigElseExpr, typename IfExpr, typename ThenExpr, typename ElseExpr, typename... Params>\
struct ExprConstructor< CVQual TensorSelectOp<OrigIfExpr, OrigThenExpr, OrigElseExpr>, CVQual TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, Params...> {\
  typedef  ExprConstructor<OrigIfExpr, IfExpr, Params...> my_if_type;\
  typedef  ExprConstructor<OrigThenExpr, ThenExpr, Params...> my_then_type;\
  typedef  ExprConstructor<OrigElseExpr, ElseExpr, Params...> my_else_type;\
  typedef CVQual TensorSelectOp<typename my_if_type::Type, typename my_then_type::Type, typename my_else_type::Type> Type;\
  my_if_type ifExpr;\
  my_then_type thenExpr;\
  my_else_type elseExpr;\
  Type expr;\
  template <typename FuncDetector>\
  ExprConstructor(FuncDetector &funcD, const utility::tuple::Tuple<Params...> &t)\
  : ifExpr(funcD.ifExpr, t), thenExpr(funcD.thenExpr, t), elseExpr(funcD.elseExpr, t), expr(ifExpr.expr, thenExpr.expr, elseExpr.expr) {}\
};

SELECTOP(const)
SELECTOP()
#undef SELECTOP

/// specialisation of the \ref ExprConstructor struct when the node type is
/// const TensorAssignOp
#define ASSIGN(CVQual)\
template <typename OrigLHSExpr, typename OrigRHSExpr, typename LHSExpr, typename RHSExpr, typename... Params>\
struct ExprConstructor<CVQual TensorAssignOp<OrigLHSExpr, OrigRHSExpr>,  CVQual TensorAssignOp<LHSExpr, RHSExpr>, Params...> {\
  typedef ExprConstructor<OrigLHSExpr, LHSExpr, Params...> my_left_type;\
  typedef ExprConstructor<OrigRHSExpr, RHSExpr, Params...> my_right_type;\
  typedef CVQual TensorAssignOp<typename my_left_type::Type, typename my_right_type::Type>  Type;\
  my_left_type lhsExpr;\
  my_right_type rhsExpr;\
  Type expr;\
  template <typename FuncDetector>\
  ExprConstructor(FuncDetector &funcD, const utility::tuple::Tuple<Params...> &t)\
  : lhsExpr(funcD.lhsExpr, t), rhsExpr(funcD.rhsExpr, t), expr(lhsExpr.expr, rhsExpr.expr) {}\
 };

 ASSIGN(const)
 ASSIGN()
 #undef ASSIGN
/// specialisation of the \ref ExprConstructor struct when the node type is
///  TensorEvalToOp
#define EVALTO(CVQual)\
template <typename OrigExpr, typename Expr, typename... Params>\
struct ExprConstructor<CVQual TensorEvalToOp<OrigExpr, MakeGlobalPointer>, CVQual TensorEvalToOp<Expr>, Params...> {\
  typedef ExprConstructor<OrigExpr, Expr, Params...> my_expr_type;\
  typedef typename TensorEvalToOp<OrigExpr, MakeGlobalPointer>::PointerType my_buffer_type;\
  typedef CVQual TensorEvalToOp<typename my_expr_type::Type, MakeGlobalPointer> Type;\
  my_expr_type nestedExpression;\
  EvalToLHSConstructor<my_buffer_type, 0, Params...> buffer;\
  Type expr;\
  template <typename FuncDetector>\
  ExprConstructor(FuncDetector &funcD, const utility::tuple::Tuple<Params...> &t)\
  : nestedExpression(funcD.rhsExpr, t), buffer(t), expr(buffer.expr, nestedExpression.expr) {}\
};

EVALTO(const)
EVALTO()
#undef EVALTO

/// specialisation of the \ref ExprConstructor struct when the node type is
/// TensorForcedEvalOp
#define FORCEDEVAL(CVQual)\
template <typename OrigExpr, typename DevExpr, size_t N, typename... Params>\
struct ExprConstructor<CVQual TensorForcedEvalOp<OrigExpr, MakeGlobalPointer>,\
CVQual PlaceHolder<CVQual TensorForcedEvalOp<DevExpr>, N>, Params...> {\
  typedef CVQual TensorMap<Tensor<typename TensorForcedEvalOp<DevExpr, MakeGlobalPointer>::Scalar,\
  TensorForcedEvalOp<DevExpr, MakeGlobalPointer>::NumDimensions, 0, typename TensorForcedEvalOp<DevExpr>::Index>, 0, MakeGlobalPointer> Type;\
  Type expr;\
  template <typename FuncDetector>\
  ExprConstructor(FuncDetector &fd, const utility::tuple::Tuple<Params...> &t)\
  : expr(Type((&(*(utility::tuple::get<N>(t).get_pointer()))), fd.dimensions())) {}\
};

FORCEDEVAL(const)
FORCEDEVAL()
#undef FORCEDEVAL

template <bool Conds,  size_t X , size_t Y > struct ValueCondition {
  static const size_t Res =X;
};
template<size_t X, size_t Y> struct ValueCondition<false, X , Y> {
  static const size_t Res =Y;
};

/// specialisation of the \ref ExprConstructor struct when the node type is TensorReductionOp
#define SYCLREDUCTIONEXPR(CVQual)\
template <typename OP, typename Dim, typename OrigExpr, typename DevExpr, size_t N, typename... Params>\
struct ExprConstructor<CVQual TensorReductionOp<OP, Dim, OrigExpr, MakeGlobalPointer>,\
CVQual PlaceHolder<CVQual TensorReductionOp<OP, Dim, DevExpr>, N>, Params...> {\
  static const size_t NumIndices= ValueCondition< TensorReductionOp<OP, Dim, DevExpr, MakeGlobalPointer>::NumDimensions==0,  1, TensorReductionOp<OP, Dim, DevExpr, MakeGlobalPointer>::NumDimensions >::Res;\
  typedef CVQual TensorMap<Tensor<typename TensorReductionOp<OP, Dim, DevExpr, MakeGlobalPointer>::Scalar,\
  NumIndices, 0, typename TensorReductionOp<OP, Dim, DevExpr>::Index>, 0, MakeGlobalPointer> Type;\
  Type expr;\
  template <typename FuncDetector>\
  ExprConstructor(FuncDetector &fd, const utility::tuple::Tuple<Params...> &t)\
  : expr(Type((&(*(utility::tuple::get<N>(t).get_pointer()))), fd.dimensions())) {}\
};

SYCLREDUCTIONEXPR(const)
SYCLREDUCTIONEXPR()
#undef SYCLREDUCTIONEXPR

/// template deduction for \ref ExprConstructor struct
template <typename OrigExpr, typename IndexExpr, typename FuncD, typename... Params>
auto createDeviceExpression(FuncD &funcD, const utility::tuple::Tuple<Params...> &t)
    -> decltype(ExprConstructor<OrigExpr, IndexExpr, Params...>(funcD, t)) {
  return ExprConstructor<OrigExpr, IndexExpr, Params...>(funcD, t);
}

} /// namespace TensorSycl
} /// namespace internal
} /// namespace Eigen


#endif  // UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_EXPR_CONSTRUCTOR_HPP
