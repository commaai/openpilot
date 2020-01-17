// $Id$
# ifndef CPPAD_LOCAL_OPTIMIZE_SIZE_PAIR_HPP
# define CPPAD_LOCAL_OPTIMIZE_SIZE_PAIR_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*!
\file size_pair.hpp
Information for one variable and one operation sequence.
*/
// BEGIN_CPPAD_LOCAL_OPTIMIZE_NAMESPACE
namespace CppAD { namespace local { namespace optimize  {

/*!
\file size_pair.hpp
Information for one variable in one operation sequence.
*/
struct struct_size_pair {
	size_t i_op;  /// operator index for this variable
	size_t i_var; /// variable index for this variable
};

} } } // END_CPPAD_LOCAL_OPTIMIZE_NAMESPACE

# endif
