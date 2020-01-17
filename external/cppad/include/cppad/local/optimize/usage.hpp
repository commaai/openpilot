// $Id$
# ifndef CPPAD_LOCAL_OPTIMIZE_USAGE_HPP
# define CPPAD_LOCAL_OPTIMIZE_USAGE_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

// BEGIN_CPPAD_LOCAL_OPTIMIZE_NAMESPACE
namespace CppAD { namespace local { namespace optimize {

enum enum_usage {
	/// This operator is not used.
	no_usage,

	/// This operator is used one or more times.
	yes_usage,

	/*!
	This operator is only used once, it is a summation operator,
	and its parrent is a summation operator. Furthermore, its result is not
	a dependent variable. Hence case it can be removed as part of a
	cumulative summation starting at its parent or above.
	*/
	csum_usage
};

} } } // END_CPPAD_LOCAL_OPTIMIZE_NAMESPACE
# endif
