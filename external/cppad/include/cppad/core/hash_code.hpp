// $Id$
# ifndef CPPAD_CORE_HASH_CODE_HPP
# define CPPAD_CORE_HASH_CODE_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*!
\file core/hash_code.hpp
CppAD hashing utility.
*/
# include <cppad/local/hash_code.hpp>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
/*!
General purpose hash code for an arbitrary value.

\tparam Value
is the type of the argument being hash coded.
It should be a plain old data class; i.e.,
the values included in the equality operator in the object and
not pointed to by the object.

\param value
the value that we are generating a hash code for.
All of the fields in value should have been set before the hash code
is computed (otherwise undefined values are used).

\return
is a hash code that is between zero and CPPAD_HASH_TABLE_SIZE - 1.

\par Checked Assertions
\li \c std::numeric_limits<unsigned short>::max() >= CPPAD_HASH_TABLE_SIZE
\li \c sizeof(value) is even
\li \c sizeof(unsigned short)  == 2
*/
template <class Value>
unsigned short hash_code(const Value& value)
{	return local::local_hash_code(value); }

} // END_CPPAD_NAMESPACE


# endif
