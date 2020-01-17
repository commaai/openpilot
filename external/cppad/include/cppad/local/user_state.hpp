// $Id$
# ifndef CPPAD_LOCAL_USER_STATE_HPP
# define CPPAD_LOCAL_USER_STATE_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

namespace CppAD { namespace local { // BEGIN_CPPAD_LOCAL_NAMESPACE

enum enum_user_state {
	/// next UserOp marks beginning of a user atomic call
	start_user,

	/// next UsrapOp (UsravOp) is a parameter (variable) argument
	arg_user,

	/// next UsrrpOp (UsrrvOp) is a parameter (variable) result
	ret_user,

	/// next UserOp marks end of a user atomic call
	end_user
};

} } // END_CPPAD_LOCAL_NAMESPACE
# endif
