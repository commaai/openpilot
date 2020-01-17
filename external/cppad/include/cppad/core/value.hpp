# ifndef CPPAD_CORE_VALUE_HPP
# define CPPAD_CORE_VALUE_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin Value$$
$spell
	const
$$



$section Convert From an AD Type to its Base Type$$
$mindex Value$$

$head Syntax$$
$icode%b% = Value(%x%)%$$

$head See Also$$
$cref var2par$$


$head Purpose$$
Converts from an AD type to the corresponding
$cref/base type/glossary/Base Type/$$.

$head x$$
The argument $icode x$$ has prototype
$codei%
	const AD<%Base%> &%x%
%$$

$head b$$
The return value $icode b$$ has prototype
$codei%
	%Base% %b%
%$$

$head Operation Sequence$$
The result of this operation is not an
$cref/AD of Base/glossary/AD of Base/$$ object.
Thus it will not be recorded as part of an
AD of $icode Base$$
$cref/operation sequence/glossary/Operation/Sequence/$$.

$head Restriction$$
If the argument $icode x$$ is a
$cref/variable/glossary/Variable/$$ its dependency information
would not be included in the $code Value$$ result (see above).
For this reason,
the argument $icode x$$ must be a $cref/parameter/glossary/Parameter/$$; i.e.,
it cannot depend on the current
$cref/independent variables/glossary/Tape/Independent Variable/$$.

$head Example$$
$children%
	example/general/value.cpp
%$$
The file
$cref value.cpp$$
contains an example and test of this operation.

$end
-------------------------------------------------------------------------------
*/

//  BEGIN CppAD namespace
namespace CppAD {

template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
Base Value(const AD<Base> &x)
{	Base result;

	CPPAD_ASSERT_KNOWN(
		Parameter(x) ,
		"Value: argument is a variable (not a parameter)"
	);


	result = x.value_;

	return result;
}

}
//  END CppAD namespace


# endif
