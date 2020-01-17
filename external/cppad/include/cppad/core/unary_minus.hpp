# ifndef CPPAD_CORE_UNARY_MINUS_HPP
# define CPPAD_CORE_UNARY_MINUS_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin UnaryMinus$$
$spell
	Vec
	const
	inline
$$


$section AD Unary Minus Operator$$
$mindex -$$

$head Syntax$$

$icode%y% = - %x%$$


$head Purpose$$
Computes the negative of $icode x$$.

$head Base$$
The operation in the syntax above must be supported for the case where
the operand is a $code const$$ $icode Base$$ object.

$head x$$
The operand $icode x$$ has one of the following prototypes
$codei%
	const AD<%Base%>               &%x%
	const VecAD<%Base%>::reference &%x%
%$$

$head y$$
The result $icode y$$ has type
$codei%
	AD<%Base%> %y%
%$$
It is equal to the negative of the operand $icode x$$.

$head Operation Sequence$$
This is an AD of $icode Base$$
$cref/atomic operation/glossary/Operation/Atomic/$$
and hence is part of the current
AD of $icode Base$$
$cref/operation sequence/glossary/Operation/Sequence/$$.

$head Derivative$$
If $latex f$$ is a
$cref/Base function/glossary/Base Function/$$,
$latex \[
	\D{[ - f(x) ]}{x} = - \D{f(x)}{x}
\] $$

$head Example$$
$children%
	example/general/unary_minus.cpp
%$$
The file
$cref unary_minus.cpp$$
contains an example and test of this operation.

$end
-------------------------------------------------------------------------------
*/

//  BEGIN CppAD namespace
namespace CppAD {

// Broken g++ compiler inhibits declaring unary minus a member or friend
template <class Base>
inline AD<Base> AD<Base>::operator - (void) const
{	// should make a more efficient version by adding unary minus to
	// Operator.h (some day)
	AD<Base> result(0);

	result  -= *this;

	return result;
}


template <class Base>
inline AD<Base> operator - (const VecAD_reference<Base> &right)
{	return - right.ADBase(); }

}
//  END CppAD namespace


# endif
