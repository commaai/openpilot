// $Id$
# ifndef CPPAD_LOCAL_OPTIMIZE_CSUM_VARIABLE_HPP
# define CPPAD_LOCAL_OPTIMIZE_CSUM_VARIABLE_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
# include <cppad/local/op_code.hpp>
# include <cppad/local/declare_ad.hpp> // defines addr_t

/*!
\file csum_variable.hpp
Information about one old variable that is part of a new CSumOp operation.
*/

// BEGIN_CPPAD_LOCAL_OPTIMIZE_NAMESPACE
namespace CppAD { namespace local { namespace optimize  {
/*!
Information about one old variable that is part of a new CSumOp operation.
*/
struct struct_csum_variable {
	/// Pointer to first argument (child) for this old operator.
	/// Set by the reverse sweep at beginning of optimization.
	const addr_t*       arg;

	/// Was this old variable added to the summation
	/// (if not it was subtracted)
	bool                add;

	/// Operator for which this old variable is the result, NumRes(op) > 0.
	OpCode              op;
};

} } } // END_CPPAD_LOCAL_OPTIMIZE_NAMESPACE

# endif
