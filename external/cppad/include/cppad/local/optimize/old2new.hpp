// $Id$
# ifndef CPPAD_LOCAL_OPTIMIZE_OLD2NEW_HPP
# define CPPAD_LOCAL_OPTIMIZE_OLD2NEW_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*!
\file old2new.hpp
Information that maps old an old operator to a new opeator and new variable.
*/

// BEGIN_CPPAD_LOCAL_OPTIMIZE_NAMESPACE
namespace CppAD { namespace local { namespace optimize  {
/*!
Information that maps old an old operator to a new opeator and new variable.
*/
struct struct_old2new {
	/// New operator index for this old operator.
	addr_t  new_op;

	/// New varaible index for this old operator.
	addr_t  new_var;
};

} } } // END_CPPAD_LOCAL_OPTIMIZE_NAMESPACE

# endif
