# ifndef CPPAD_CORE_INTEGER_HPP
# define CPPAD_CORE_INTEGER_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
------------------------------------------------------------------------------
$begin Integer$$
$spell
	std
	VecAD
	CppAD
	namespace
	const
	bool
$$



$section Convert From AD to Integer$$

$head Syntax$$
$icode%i% = Integer(%x%)%$$


$head Purpose$$
Converts from an AD type to the corresponding integer value.

$head i$$
The result $icode i$$ has prototype
$codei%
	int %i%
%$$

$head x$$

$subhead Real Types$$
If the argument $icode x$$ has either of the following prototypes:
$codei%
	const float                %%  &%x%
	const double               %%  &%x%
%$$
the fractional part is dropped to form the integer value.
For example, if $icode x$$ is 1.5, $icode i$$ is 1.
In general, if $latex x \geq 0$$, $icode i$$ is the
greatest integer less than or equal $icode x$$.
If $latex x \leq 0$$, $icode i$$ is the
smallest integer greater than or equal $icode x$$.

$subhead Complex Types$$
If the argument $icode x$$ has either of the following prototypes:
$codei%
	const std::complex<float>  %%  &%x%
	const std::complex<double> %%  &%x%
%$$
The result $icode i$$ is given by
$codei%
	%i% = Integer(%x%.real())
%$$

$subhead AD Types$$
If the argument $icode x$$ has either of the following prototypes:
$codei%
	const AD<%Base%>               &%x%
	const VecAD<%Base%>::reference &%x%
%$$
$icode Base$$ must support the $code Integer$$ function and
the conversion has the same meaning as for $icode Base$$.

$head Operation Sequence$$
The result of this operation is not an
$cref/AD of Base/glossary/AD of Base/$$ object.
Thus it will not be recorded as part of an
AD of $icode Base$$
$cref/operation sequence/glossary/Operation/Sequence/$$.

$head Example$$
$children%
	example/general/integer.cpp
%$$
The file
$cref integer.cpp$$
contains an example and test of this operation.

$end
------------------------------------------------------------------------------
*/


namespace CppAD {

	template <class Base>
	CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
	int Integer(const AD<Base> &x)
	{	return Integer(x.value_); }

	template <class Base>
	CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
	int Integer(const VecAD_reference<Base> &x)
	{	return Integer( x.ADBase() ); }
}
# endif

