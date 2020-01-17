// $Id$
# ifndef CPPAD_LOCAL_OPTIMIZE_CEXP_INFO_HPP
# define CPPAD_LOCAL_OPTIMIZE_CEXP_INFO_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
# include <cppad/local/declare_ad.hpp> // defines CompareOp
# include <cppad/utility/vector.hpp>

/*!
\file cexp_info.hpp
Information about one conditional expression.
*/

// BEGIN_CPPAD_LOCAL_OPTIMIZE_NAMESPACE
namespace CppAD { namespace local { namespace optimize  {
/*!
Information about one conditional expression.
*/
struct struct_cexp_info {
	/// The operator index for this conditional expression operation
	size_t i_op;

	/// (flag & 1) is true if and only if left is a variable
	/// (flag & 2) is true if and only if right is a variable
	size_t flag;

	/// variable or parameter index for left comparison operand
	size_t left;

	/// variable or parameter index for right comparison operand
	size_t right;

	/// maximum variable index between left and right (ignoring parameters).
	size_t max_left_right;

	/// set of operator that are not used when comparison result is true
	/// Note that UsrapOp, UsravOp, UsrrpOp, and UsrrvOp, are not in this
	/// vector and should be skipped when the corresponding UserOp are skipped.
	CppAD::vector<size_t> skip_op_true;

	/// set of variables that are not used when comparison result is false
	/// Note that UsrapOp, UsravOp, UsrrpOp, and UsrrvOp, are not in this
	/// vector and should be skipped when the corresponding UserOp are skipped.
	CppAD::vector<size_t> skip_op_false;

	/// comparision operator for this conditional expression
	CompareOp cop;
};

// Information about the conditional skip in the new operation sequence
struct struct_cskip_new {
	/// new variable or parameter index for left comparison operand
	size_t left;
	/// new variable or parameter index for right comparison operand
	size_t right;
	/// maximum variable index between left and right (ignoring parameters).
	size_t max_left_right;
	/// index where this conditional skips arguments start
	size_t i_arg;
};

} } } // END_CPPAD_LOCAL_OPTIMIZE_NAMESPACE

# endif
