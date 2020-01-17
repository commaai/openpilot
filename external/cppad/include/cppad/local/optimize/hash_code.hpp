// $Id$
# ifndef CPPAD_LOCAL_OPTIMIZE_HASH_CODE_HPP
# define CPPAD_LOCAL_OPTIMIZE_HASH_CODE_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*!
\file local/optimize/hash_code.hpp
CppAD hashing utility.
*/


// BEGIN_CPPAD_LOCAL_OPTIMIZE_NAMESPACE
namespace CppAD { namespace local { namespace optimize {
/*!
Specialized hash code for a CppAD operator and its arguments
(used during optimization).

\param op
is the operator that we are computing a hash code for.

\param num_arg
number of elements of arg to include in the hash code
(num_arg <= 2).

\param arg
is a vector of length num_arg
containing the corresponding argument indices for this operator.

\return
is a hash code that is between zero and CPPAD_HASH_TABLE_SIZE - 1.
*/

inline size_t optimize_hash_code(
	OpCode        op      ,
	size_t        num_arg ,
	const addr_t* arg     )
{
	//
	CPPAD_ASSERT_UNKNOWN(num_arg <= 2 );
	size_t sum = size_t(arg[0]) + size_t(op);
	if( 1 < num_arg )
		sum += size_t(arg[1]);
	//
	return sum % CPPAD_HASH_TABLE_SIZE;
}

} } } // END_CPPAD_LOCAL_OPTIMIZE_NAMESPACE

# endif
