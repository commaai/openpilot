// $Id$
# ifndef CPPAD_LOCAL_SET_GET_IN_PARALLEL_HPP
# define CPPAD_LOCAL_SET_GET_IN_PARALLEL_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

# include <cassert>
# include <cppad/configure.hpp>
namespace CppAD { namespace local { // BEGIN_CPPAD_LOCAL_NAMESPACE

/*!
\file set_get_in_parallel.hpp
File used to set and get user in_parallel routine.
*/
/*!
Set and call the routine that determine if we are in parallel execution mode.

\return
value retuned by most recent setting for in_parallel_new.
If set is true,
or the most recent setting is CPPAD_NULL (its initial value),
the return value is false.
Otherwise the function corresponding to the most recent setting
is called and its value returned by set_get_in_parallel.

\param in_parallel_new [in]
If set is false, in_parallel_new it is not used.
Otherwise, the current value of in_parallel_new becomes the
most recent setting for in_parallel_user.

\param set
If set is true, then parallel_new is becomes the most
recent setting for this set_get_in_parallel.
In this case, it is assumed that we are currently in sequential execution mode.
*/
static bool set_get_in_parallel(
	bool (*in_parallel_new)(void) ,
	bool set = false           )
{	static bool (*in_parallel_user)(void) = CPPAD_NULL;

	if( set )
	{	in_parallel_user = in_parallel_new;
		// Doing a raw assert in this case because set_get_in_parallel is used
		// by ErrorHandler and hence cannot use ErrorHandler.
		// CPPAD_ASSERT_UNKNOWN( in_parallel_user() == false )
		assert(in_parallel_user == CPPAD_NULL || in_parallel_user() == false);
		return false;
	}
	//
	if( in_parallel_user == CPPAD_NULL )
		return false;
	//
	return in_parallel_user();
}

} } // END_CPPAD_LOCAL_NAMESPACE

# endif
