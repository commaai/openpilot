# ifndef CPPAD_CORE_BASE_TO_STRING_HPP
# define CPPAD_CORE_BASE_TO_STRING_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin base_to_string$$
$spell
	std
	namespace
	CppAD
	struct
	const
	stringstream
	setprecision
	str
$$

$section Extending to_string To Another Floating Point Type$$

$head Base Requirement$$
If the function $cref to_string$$ is used by an
$cref/AD type above Base/glossary/AD Type Above Base/$$,
A specialization for the template structure
$code CppAD::to_string_struct$$ must be defined.

$head CPPAD_TO_STRING$$
For most $icode Base$$ types,
the following can be used to define the specialization:
$codei%
	namespace CppAD {
		CPPAD_TO_STRING(%Base%)
	}
%$$
Note that the $code CPPAD_TO_STRING$$ macro assumes that the
$cref base_limits$$ and $cref base_std_math$$ have already been defined
for this type.
This macro is defined as follows:
$srccode%cpp% */
# define CPPAD_TO_STRING(Base) \
template <> struct to_string_struct<Base>\
{	std::string operator()(const Base& value) \
	{	std::stringstream os;\
		int n_digits = 1 + CppAD::numeric_limits<Base>::digits10; \
		os << std::setprecision(n_digits);\
		os << value;\
		return os.str();\
	}\
};
/* %$$
$end
------------------------------------------------------------------------------
*/
// make sure to_string has been included
# include <cppad/utility/to_string.hpp>

# endif
