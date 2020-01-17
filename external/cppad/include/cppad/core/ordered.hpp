// $Id$
# ifndef CPPAD_CORE_ORDERED_HPP
# define CPPAD_CORE_ORDERED_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

# include <cppad/core/define.hpp>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE

/*!
\file ordered.hpp
Check and AD values ordering properties relative to zero.
*/

// GreaterThanZero ============================================================
/*!
Check if an AD<Base> is greater than zero.

\param x
value we are checking.

\return
returns true iff the \c x is greater than zero.
*/
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool GreaterThanZero(const AD<Base> &x)
{	return GreaterThanZero(x.value_); }
// GreaterThanOrZero =========================================================
/*!
Check if an AD<Base> is greater than or equal zero.

\param x
value we are checking.

\return
returns true iff the \c x is greater than or equal zero.
*/
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool GreaterThanOrZero(const AD<Base> &x)
{	return GreaterThanOrZero(x.value_); }
// LessThanZero ============================================================
/*!
Check if an AD<Base> is less than zero.

\param x
value we are checking.

\return
returns true iff the \c x is less than zero.
*/
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool LessThanZero(const AD<Base> &x)
{	return LessThanZero(x.value_); }
// LessThanOrZero =========================================================
/*!
Check if an AD<Base> is less than or equal zero.

\param x
value we are checking.

\return
returns true iff the \c x is less than or equal zero.
*/
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool LessThanOrZero(const AD<Base> &x)
{	return LessThanOrZero(x.value_); }
// abs_geq =========================================================
/*!
Check if absolute value of one AD<Base> is greater or equal another.

\param x
value we are checking if it is greater than or equal other.

\param y
value we are checking if it is less than other.

\return
returns true iff the absolute value of \c x is greater than or equal
absolute value of \c y.
*/
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool abs_geq(const AD<Base>& x, const AD<Base>& y)
{	return abs_geq(x.value_, y.value_); }
// ============================================================================
} // END_CPPAD_NAMESPACE
# endif

