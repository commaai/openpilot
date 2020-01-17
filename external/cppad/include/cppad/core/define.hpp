// $Id$
# ifndef CPPAD_CORE_DEFINE_HPP
# define CPPAD_CORE_DEFINE_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*!
\file define.hpp
Define processor symbols and macros that are used by CppAD.
*/

// ----------------------------------------------------------------------------
/*!
\def CPPAD_OP_CODE_TYPE
Is the type used to store enum OpCode values. If not the same as OpCode, then
<code>sizeof(CPPAD_OP_CODE_TYPE) <= sizeof( enum OpCode )</code>
to conserve memory.
This type must support \c std::numeric_limits,
the \c <= operator,
and conversion to \c size_t.
Make sure that the type chosen returns true for is_pod<CPPAD_OP_CODE_TYPE>
in pod_vector.hpp.
*/
# define CPPAD_OP_CODE_TYPE unsigned char


// ----------------------------------------------------------------------------
/*!
\def CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
A version of the inline command that works with MC compiler.

Microsoft Visual C++ version 9.0 generates a warning if a template
function is declared as a friend
(this was not a problem for version 7.0).
The warning identifier is
\verbatim
	warning C4396
\endverbatim
and it contains the text
\verbatim
	the inline specifier cannot be used when a friend declaration refers
	to a specialization of a function template
\endverbatim
This happens even if the function is not a specialization.
This macro is defined as empty for Microsoft compilers.
*/
# ifdef _MSC_VER
# define CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
# else
# define CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION inline
# endif

// ----------------------------------------------------------------------------
/*!
\def CPPAD_LIB_EXPORT
Special macro for exporting windows DLL symbols; see
https://cmake.org/Wiki/BuildingWinDLL
*/
# ifdef  _MSC_VER
# ifdef  cppad_lib_EXPORTS
# define CPPAD_LIB_EXPORT __declspec(dllexport)
# else
# define CPPAD_LIB_EXPORT __declspec(dllimport)
# endif  // cppad_lib_EXPORTS
# else   // _MSC_VER
# define CPPAD_LIB_EXPORT
# endif


// ============================================================================
/*!
\def CPPAD_FOLD_ASSIGNMENT_OPERATOR(Op)
Declares automatic coercion for certain AD assignment operations.

This macro assumes that the operator
\verbatim
	left Op right
\endverbatim
is defined for the case where left and right have type AD<Base>.
It uses this case to define the cases where
left has type AD<Base> and right has type
VecAD_reference<Base>,
Base, or
double.
The argument right is const and call by reference.
This macro converts the operands to AD<Base> and then
uses the definition of the same operation for that case.
*/

# define CPPAD_FOLD_ASSIGNMENT_OPERATOR(Op)                             \
/* ----------------------------------------------------------------*/   \
template <class Base>                                                   \
inline AD<Base>& operator Op                                            \
(AD<Base> &left, double right)                                          \
{	return left Op AD<Base>(right); }                                  \
                                                                        \
template <class Base>                                                   \
inline AD<Base>& operator Op                                            \
(AD<Base> &left, const Base &right)                                     \
{	return left Op AD<Base>(right); }                                  \
                                                                        \
inline AD<double>& operator Op                                          \
(AD<double> &left, const double &right)                                 \
{	return left Op AD<double>(right); }                                \
                                                                        \
template <class Base>                                                   \
inline AD<Base>& operator Op                                            \
(AD<Base> &left, const VecAD_reference<Base> &right)                    \
{	return left Op right.ADBase(); }

// =====================================================================
/*!
\def CPPAD_FOLD_AD_VALUED_BINARY_OPERATOR(Op)
Declares automatic coercion for certain binary operations with AD result.

This macro assumes that the operator
\verbatim
	left Op right
\endverbatim
is defined for the case where left and right
and the result of the operation all
have type AD<Base>.
It uses this case to define the cases either left
or right has type VecAD_reference<Base> or AD<Base>
and the type of the other operand is one of the following:
VecAD_reference<Base>, AD<Base>, Base, double.
All of the arguments are const and call by reference.
This macro converts the operands to AD<Base> and then
uses the definition of the same operation for that case.
*/
# define CPPAD_FOLD_AD_VALUED_BINARY_OPERATOR(Op)                      \
/* ----------------------------------------------------------------*/  \
/* Operations with VecAD_reference<Base> and AD<Base> only*/           \
                                                                       \
template <class Base>                                                  \
inline AD<Base> operator Op                                            \
(const AD<Base> &left, const VecAD_reference<Base> &right)             \
{	return left Op right.ADBase(); }                                  \
                                                                       \
template <class Base>                                                  \
inline AD<Base> operator Op                                            \
(const VecAD_reference<Base> &left, const VecAD_reference<Base> &right)\
{	return left.ADBase() Op right.ADBase(); }                         \
                                                                       \
template <class Base>                                                  \
inline AD<Base> operator Op                                            \
	(const VecAD_reference<Base> &left, const AD<Base> &right)        \
{	return left.ADBase() Op right; }                                  \
/* ----------------------------------------------------------------*/  \
/* Operations Base */                                                  \
                                                                       \
template <class Base>                                                  \
inline AD<Base> operator Op                                            \
	(const Base &left, const AD<Base> &right)                         \
{	return AD<Base>(left) Op right; }                                 \
                                                                       \
template <class Base>                                                  \
inline AD<Base> operator Op                                            \
	(const Base &left, const VecAD_reference<Base> &right)            \
{	return AD<Base>(left) Op right.ADBase(); }                        \
                                                                       \
template <class Base>                                                  \
inline AD<Base> operator Op                                            \
	(const AD<Base> &left, const Base &right)                         \
{	return left Op AD<Base>(right); }                                 \
                                                                       \
template <class Base>                                                  \
inline AD<Base> operator Op                                            \
	(const VecAD_reference<Base> &left, const Base &right)            \
{	return left.ADBase() Op AD<Base>(right); }                        \
                                                                       \
/* ----------------------------------------------------------------*/  \
/* Operations double */                                                \
                                                                       \
template <class Base>                                                  \
inline AD<Base> operator Op                                            \
	(const double &left, const AD<Base> &right)                       \
{	return AD<Base>(left) Op right; }                                 \
                                                                       \
template <class Base>                                                  \
inline AD<Base> operator Op                                            \
	(const double &left, const VecAD_reference<Base> &right)          \
{	return AD<Base>(left) Op right.ADBase(); }                        \
                                                                       \
template <class Base>                                                  \
inline AD<Base> operator Op                                            \
	(const AD<Base> &left, const double &right)                       \
{	return left Op AD<Base>(right); }                                 \
                                                                       \
template <class Base>                                                  \
inline AD<Base> operator Op                                            \
	(const VecAD_reference<Base> &left, const double &right)          \
{	return left.ADBase() Op AD<Base>(right); }                        \
/* ----------------------------------------------------------------*/  \
/* Special case to avoid ambuigity when Base is double */              \
                                                                       \
inline AD<double> operator Op                                          \
	(const double &left, const AD<double> &right)                     \
{	return AD<double>(left) Op right; }                               \
                                                                       \
inline AD<double> operator Op                                          \
	(const double &left, const VecAD_reference<double> &right)        \
{	return AD<double>(left) Op right.ADBase(); }                      \
                                                                       \
inline AD<double> operator Op                                          \
	(const AD<double> &left, const double &right)                     \
{	return left Op AD<double>(right); }                               \
                                                                       \
inline AD<double> operator Op                                          \
	(const VecAD_reference<double> &left, const double &right)        \
{	return left.ADBase() Op AD<double>(right); }

// =======================================================================

/*!
\def CPPAD_FOLD_BOOL_VALUED_BINARY_OPERATOR(Op)
Declares automatic coercion for certain binary operations with bool result.

This macro assumes that the operator
\verbatim
	left Op right
\endverbatim
is defined for the case where left and right
have type AD<Base> and the result has type bool.
It uses this case to define the cases either left
or right has type
VecAD_reference<Base> or AD<Base>
and the type of the other operand is one of the following:
VecAD_reference<Base>, AD<Base>, Base, double.
All of the arguments are const and call by reference.
This macro converts the operands to AD<Base> and then
uses the definition of the same operation for that case.
*/
# define CPPAD_FOLD_BOOL_VALUED_BINARY_OPERATOR(Op)                    \
/* ----------------------------------------------------------------*/  \
/* Operations with VecAD_reference<Base> and AD<Base> only*/           \
                                                                       \
template <class Base>                                                  \
inline bool operator Op                                                \
(const AD<Base> &left, const VecAD_reference<Base> &right)             \
{	return left Op right.ADBase(); }                                  \
                                                                       \
template <class Base>                                                  \
inline bool operator Op                                                \
(const VecAD_reference<Base> &left, const VecAD_reference<Base> &right)\
{	return left.ADBase() Op right.ADBase(); }                         \
                                                                       \
template <class Base>                                                  \
inline bool operator Op                                                \
	(const VecAD_reference<Base> &left, const AD<Base> &right)        \
{	return left.ADBase() Op right; }                                  \
/* ----------------------------------------------------------------*/  \
/* Operations Base */                                                  \
                                                                       \
template <class Base>                                                  \
inline bool operator Op                                                \
	(const Base &left, const AD<Base> &right)                         \
{	return AD<Base>(left) Op right; }                                 \
                                                                       \
template <class Base>                                                  \
inline bool operator Op                                                \
	(const Base &left, const VecAD_reference<Base> &right)            \
{	return AD<Base>(left) Op right.ADBase(); }                        \
                                                                       \
template <class Base>                                                  \
inline bool operator Op                                                \
	(const AD<Base> &left, const Base &right)                         \
{	return left Op AD<Base>(right); }                                 \
                                                                       \
template <class Base>                                                  \
inline bool operator Op                                                \
	(const VecAD_reference<Base> &left, const Base &right)            \
{	return left.ADBase() Op AD<Base>(right); }                        \
                                                                       \
/* ----------------------------------------------------------------*/  \
/* Operations double */                                                \
                                                                       \
template <class Base>                                                  \
inline bool operator Op                                                \
	(const double &left, const AD<Base> &right)                       \
{	return AD<Base>(left) Op right; }                                 \
                                                                       \
template <class Base>                                                  \
inline bool operator Op                                                \
	(const double &left, const VecAD_reference<Base> &right)          \
{	return AD<Base>(left) Op right.ADBase(); }                        \
                                                                       \
template <class Base>                                                  \
inline bool operator Op                                                \
	(const AD<Base> &left, const double &right)                       \
{	return left Op AD<Base>(right); }                                 \
                                                                       \
template <class Base>                                                  \
inline bool operator Op                                                \
	(const VecAD_reference<Base> &left, const double &right)          \
{	return left.ADBase() Op AD<Base>(right); }                        \
/* ----------------------------------------------------------------*/  \
/* Special case to avoid ambuigity when Base is double */              \
                                                                       \
inline bool operator Op                                                \
	(const double &left, const AD<double> &right)                     \
{	return AD<double>(left) Op right; }                               \
                                                                       \
inline bool operator Op                                                \
	(const double &left, const VecAD_reference<double> &right)        \
{	return AD<double>(left) Op right.ADBase(); }                      \
                                                                       \
inline bool operator Op                                                \
	(const AD<double> &left, const double &right)                     \
{	return left Op AD<double>(right); }                               \
                                                                       \
inline bool operator Op                                                \
	(const VecAD_reference<double> &left, const double &right)        \
{	return left.ADBase() Op AD<double>(right); }

# endif
