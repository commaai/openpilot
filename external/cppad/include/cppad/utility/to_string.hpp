# ifndef CPPAD_UTILITY_TO_STRING_HPP
# define CPPAD_UTILITY_TO_STRING_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin to_string$$
$spell
	cppad.hpp
	long long
	std
	const
	ostringstream
$$

$section Convert Certain Types to a String$$

$head Syntax$$
$codei%# include <cppad/utility/to_string.hpp>
%$$
$icode%s% = to_string(%value%)%$$.

$head See Also$$
$cref base_to_string$$, $cref ad_to_string$$

$head Purpose$$
This routine is similar to the C++11 routine $code std::to_string$$
with the following differences:
$list number$$
It works with C++98.
$lnext
It has been extended to the fundamental floating point types.
$lnext
It has specifications for extending to an arbitrary type; see
$cref base_to_string$$.
$lnext
If $code <cppad/cppad.hpp>$$ is included,
and it has been extended to a $icode Base$$ type,
it automatically extends to the
$cref/AD types above Base/glossary/AD Type Above Base/$$.
$lnext
For integer types, conversion to a string is exact.
For floating point types, conversion to a string yields a value
that has relative error within machine epsilon.
$lend

$head value$$

$subhead Integer$$
The argument $icode value$$ can have the following prototype
$codei%
	const %Integer%&  %value%
%$$
where $icode Integer$$ is any of the fundamental integer types; e.g.,
$code short int$$ and $code unsigned long$$.
Note that if C++11 is supported by this compilation,
$code unsigned long long$$ is also a fundamental integer type.

$subhead Float$$
The argument $icode value$$ can have the following prototype
$codei%
	const %Float%&  %value%
%$$
where $icode Float$$ is any of the fundamental floating point types; i.e.,
$code float$$, $code double$$, and $code long double$$.

$head s$$
The return value has prototype
$codei%
	std::string %s%
%$$
and contains a representation of the specified $icode value$$.

$subhead Integer$$
If $icode value$$ is an $codei Integer$$,
the representation is equivalent to $codei%os% << %value%$$
where $icode os$$ is an $code std::ostringstream$$.

$subhead Float$$
If $icode value$$ is a $codei Float$$,
enough digits are used in the representation so that
the result is accurate to withing round off error.

$children%
	example/utility/to_string.cpp
%$$
$head Example$$
The file $cref to_string.cpp$$
contains an example and test of this routine.
It returns true if it succeeds and false otherwise.

$end
*/
# include <limits>
# include <cmath>
# include <iomanip>
# include <sstream>
# include <cppad/core/cppad_assert.hpp>

# define CPPAD_SPECIALIZE_TO_STRING_INTEGER(Type) \
template <> struct to_string_struct<Type>\
{	std::string operator()(const Type& value) \
	{	std::stringstream os;\
		os << value;\
		return os.str();\
	}\
};

# define CPPAD_SPECIALIZE_TO_STRING_FLOAT(Float) \
template <> struct to_string_struct<Float>\
{	std::string operator()(const Float& value) \
	{	std::stringstream os;\
		int n_digits = 1 + std::numeric_limits<Float>::digits10;\
		os << std::setprecision(n_digits);\
		os << value;\
		return os.str();\
	}\
};

namespace CppAD {

	// Default implementation,
	// each type must define its own specilization.
	template <class Type>
	struct to_string_struct
	{	std::string operator()(const Type& value)
		{	CPPAD_ASSERT_KNOWN(
				false,
				"to_string is not implemented for this type"
			);
			// return empty string
			return std::string("");
		}
	};

	// specialization for the fundamental integer types
	CPPAD_SPECIALIZE_TO_STRING_INTEGER(signed short)
	CPPAD_SPECIALIZE_TO_STRING_INTEGER(unsigned short)
	//
	CPPAD_SPECIALIZE_TO_STRING_INTEGER(signed int)
	CPPAD_SPECIALIZE_TO_STRING_INTEGER(unsigned int)
	//
	CPPAD_SPECIALIZE_TO_STRING_INTEGER(signed long)
	CPPAD_SPECIALIZE_TO_STRING_INTEGER(unsigned long)
	//
# if CPPAD_USE_CPLUSPLUS_2011
	CPPAD_SPECIALIZE_TO_STRING_INTEGER(signed long long)
	CPPAD_SPECIALIZE_TO_STRING_INTEGER(unsigned long long)
# endif

	// specialization for the fundamental floating point types
	CPPAD_SPECIALIZE_TO_STRING_FLOAT(float)
	CPPAD_SPECIALIZE_TO_STRING_FLOAT(double)
	CPPAD_SPECIALIZE_TO_STRING_FLOAT(long double)

	// link from function to function object in structure
	template<class Type>
	std::string to_string(const Type& value)
	{	to_string_struct<Type> to_str;
		return to_str(value);
	}
}

# undef CPPAD_SPECIALIZE_TO_STRING_FLOAT
# undef CPPAD_SPECIALIZE_TO_STRING_INTEGER
# endif
