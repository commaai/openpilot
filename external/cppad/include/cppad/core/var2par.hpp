# ifndef CPPAD_CORE_VAR2PAR_HPP
# define CPPAD_CORE_VAR2PAR_HPP

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

$begin Var2Par$$
$spell
	var
	const
$$


$section Convert an AD Variable to a Parameter$$
$mindex Var2Par from value_ obtain during taping$$

$head Syntax$$
$icode%y% = Var2Par(%x%)%$$

$head See Also$$
$cref value$$


$head Purpose$$
Returns a
$cref/parameter/glossary/Parameter/$$ $icode y$$
with the same value as the
$cref/variable/glossary/Variable/$$ $icode x$$.

$head x$$
The argument $icode x$$ has prototype
$codei%
	const AD<%Base%> &x
%$$
The argument $icode x$$ may be a variable or parameter.


$head y$$
The result $icode y$$ has prototype
$codei%
	AD<%Base%> &y
%$$
The return value $icode y$$ will be a parameter.


$head Example$$
$children%
	example/general/var2par.cpp
%$$
The file
$cref var2par.cpp$$
contains an example and test of this operation.
It returns true if it succeeds and false otherwise.

$end
------------------------------------------------------------------------------
*/

//  BEGIN CppAD namespace
namespace CppAD {

template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
AD<Base> Var2Par(const AD<Base> &x)
{	AD<Base> y(x.value_);
	return y;
}


template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
AD<Base> Var2Par(const VecAD_reference<Base> &x)
{	AD<Base> y(x.ADBase());
	y.id_ = 0;
}


} // END CppAD namespace

# endif
