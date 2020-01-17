// $Id$
# ifndef CPPAD_CORE_EPSILON_HPP
# define CPPAD_CORE_EPSILON_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
------------------------------------------------------------------------------
$begin epsilon$$
$spell
	std
	eps
	CppAD
	namespace
	const
$$

$section Machine Epsilon For AD Types$$

$head Deprecated 2012-06-17$$
This routine has been deprecated.
You should use the $cref numeric_limits$$ $code epsilon$$ instead.

$head Syntax$$
$icode%eps% = epsilon<%Float%>()%$$

$head Purpose$$
Obtain the value of machine epsilon corresponding
to the type $icode%Float%$$.

$head Float$$
this type can either be $codei%AD<%Base%>%$$,
or it can be $icode Base$$ for any $codei%AD<%Base%>%$$ type.

$head eps$$
The result $icode eps$$ has prototype
$codei%
	%Float% eps
%$$

$end
------------------------------------------------------------------------------
*/

namespace CppAD {

	template <class Type>
	inline Type epsilon(void)
	{	return Type ( numeric_limits<Type>::epsilon() ); }

}
# endif
