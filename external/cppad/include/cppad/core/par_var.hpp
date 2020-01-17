# ifndef CPPAD_CORE_PAR_VAR_HPP
# define CPPAD_CORE_PAR_VAR_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
---------------------------------------------------------------------------

$begin ParVar$$
$spell
	VecAD
	const
	bool
$$

$section Is an AD Object a Parameter or Variable$$

$head Syntax$$
$icode%b% = Parameter(%x%)%$$
$pre
$$
$icode%b% = Variable(%x%)%$$


$head Purpose$$
Determine if $icode x$$ is a
$cref/parameter/glossary/Parameter/$$ or
$cref/variable/glossary/Variable/$$.

$head x$$
The argument $icode x$$ has prototype
$codei%
	const AD<%Base%>    &%x%
	const VecAD<%Base%> &%x%
%$$

$head b$$
The return value $icode b$$ has prototype
$codei%
	bool %b%
%$$
The return value for $code Parameter$$ ($code Variable$$)
is true if and only if $icode x$$ is a parameter (variable).
Note that a $cref/VecAD<Base>/VecAD/$$ object
is a variable if any element of the vector depends on the independent
variables.

$head Operation Sequence$$
The result of this operation is not an
$cref/AD of Base/glossary/AD of Base/$$ object.
Thus it will not be recorded as part of an
AD of $icode Base$$
$cref/operation sequence/glossary/Operation/Sequence/$$.

$head Example$$
$children%
	example/general/par_var.cpp
%$$
The file
$cref par_var.cpp$$
contains an example and test of these functions.
It returns true if it succeeds and false otherwise.

$end
-----------------------------------------------------------------------------
*/

namespace CppAD {
	// Parameter
	template <class Base>
	CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
	bool Parameter(const AD<Base> &x)
	{	if( x.tape_id_ == 0 )
			return true;
		size_t thread = size_t(x.tape_id_ % CPPAD_MAX_NUM_THREADS);
		return x.tape_id_ != *AD<Base>::tape_id_ptr(thread);
	}

	template <class Base>
	CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
	bool Parameter(const VecAD<Base> &x)
	{	if( x.tape_id_ == 0 )
			return true;
		size_t thread = size_t(x.tape_id_ % CPPAD_MAX_NUM_THREADS);
		return x.tape_id_ != *AD<Base>::tape_id_ptr(thread);
	}

	// Variable
	template <class Base>
	CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
	bool Variable(const AD<Base> &x)
	{	if( x.tape_id_ == 0 )
			return false;
		size_t thread = size_t(x.tape_id_ % CPPAD_MAX_NUM_THREADS);
		return x.tape_id_ == *AD<Base>::tape_id_ptr(thread);
	}

	template <class Base>
	CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
	bool Variable(const VecAD<Base> &x)
	{	if( x.tape_id_ == 0 )
			return false;
		size_t thread = size_t(x.tape_id_ % CPPAD_MAX_NUM_THREADS);
		return x.tape_id_ == *AD<Base>::tape_id_ptr(thread);
	}
}
// END CppAD namespace


# endif
