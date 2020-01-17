# ifndef CPPAD_CORE_AZMUL_HPP
# define CPPAD_CORE_AZMUL_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin azmul$$
$spell
	azmul
	const
	namespace
	Vec
$$

$section Absolute Zero Multiplication$$

$head Syntax$$
$icode%z% = azmul(%x%, %y%)%$$

$head Purpose$$
Evaluates multiplication with an absolute zero
for any of the possible types listed below.
The result is given by
$latex \[
z = \left\{ \begin{array}{ll}
	0          & {\rm if} \; x = 0 \\
	x \cdot y  & {\rm otherwise}
\end{array} \right.
\] $$
Note if $icode x$$ is zero and $icode y$$ is infinity,
ieee multiplication would result in not a number whereas
$icode z$$ would be zero.

$head Base$$
If $icode Base$$ satisfies the
$cref/base type requirements/base_require/$$
and arguments $icode x$$, $icode y$$ have prototypes
$codei%
	const %Base%& %x%
	const %Base%& %y%
%$$
then the result $icode z$$ has prototype
$codei%
	%Base% %z%
%$$

$head AD<Base>$$
If the arguments $icode x$$, $icode y$$ have prototype
$codei%
	const AD<%Base%>& %x%
	const AD<%Base%>& %y%
%$$
then the result $icode z$$ has prototype
$codei%
	AD<%Base%> %z%
%$$

$head VecAD<Base>$$
If the arguments $icode x$$, $icode y$$ have prototype
$codei%
	const VecAD<%Base%>::reference& %x%
	const VecAD<%Base%>::reference& %y%
%$$
then the result $icode z$$ has prototype
$codei%
	AD<%Base%> %z%
%$$

$head Example$$
$children%
	example/general/azmul.cpp
%$$
The file
$cref azmul.cpp$$
is an examples and tests of this function.
It returns true if it succeeds and false otherwise.

$end
*/

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
// ==========================================================================

// case where x and y are AD<Base> -------------------------------------------
template <class Base> AD<Base>
azmul(const AD<Base>& x, const AD<Base>& y)
{
	// compute the Base part
	AD<Base> result;
	result.value_ = azmul(x.value_, y.value_);

	// check if there is a recording in progress
	local::ADTape<Base>* tape = AD<Base>::tape_ptr();
	if( tape == CPPAD_NULL )
		return result;
	tape_id_t tape_id = tape->id_;

	// tape_id cannot match the default value for tape_id_; i.e., 0
	CPPAD_ASSERT_UNKNOWN( tape_id > 0 );
	bool var_x = x.tape_id_ == tape_id;
	bool var_y = y.tape_id_ == tape_id;

	if( var_x )
	{	if( var_y )
		{	// result = azmul(variable, variable)
			CPPAD_ASSERT_UNKNOWN( local::NumRes(local::ZmulvvOp) == 1 );
			CPPAD_ASSERT_UNKNOWN( local::NumArg(local::ZmulvvOp) == 2 );

			// put operand addresses in tape
			tape->Rec_.PutArg(x.taddr_, y.taddr_);

			// put operator in the tape
			result.taddr_ = tape->Rec_.PutOp(local::ZmulvvOp);

			// make result a variable
			result.tape_id_ = tape_id;
		}
		else if( IdenticalZero( y.value_ ) )
		{	// result = variable * 0
		}
		else if( IdenticalOne( y.value_ ) )
		{	// result = variable * 1
			result.make_variable(x.tape_id_, x.taddr_);
		}
		else
		{	// result = zmul(variable, parameter)
			CPPAD_ASSERT_UNKNOWN( local::NumRes(local::ZmulvpOp) == 1 );
			CPPAD_ASSERT_UNKNOWN( local::NumArg(local::ZmulvpOp) == 2 );

			// put operand addresses in tape
			addr_t p = tape->Rec_.PutPar(y.value_);
			tape->Rec_.PutArg(x.taddr_, p);

			// put operator in the tape
			result.taddr_ = tape->Rec_.PutOp(local::ZmulvpOp);

			// make result a variable
			result.tape_id_ = tape_id;
		}
	}
	else if( var_y )
	{	if( IdenticalZero(x.value_) )
		{	// result = 0 * variable
		}
		else if( IdenticalOne( x.value_ ) )
		{	// result = 1 * variable
			result.make_variable(y.tape_id_, y.taddr_);
		}
		else
		{	// result = zmul(parameter, variable)
			CPPAD_ASSERT_UNKNOWN( local::NumRes(local::ZmulpvOp) == 1 );
			CPPAD_ASSERT_UNKNOWN( local::NumArg(local::ZmulpvOp) == 2 );

			// put operand addresses in tape
			addr_t p = tape->Rec_.PutPar(x.value_);
			tape->Rec_.PutArg(p, y.taddr_);

			// put operator in the tape
			result.taddr_ = tape->Rec_.PutOp(local::ZmulpvOp);

			// make result a variable
			result.tape_id_ = tape_id;
		}
	}
	return result;
}
// =========================================================================
// Fold operations into case above
// -------------------------------------------------------------------------
// Operations with VecAD_reference<Base> and AD<Base> only

template <class Base> AD<Base>
azmul(const AD<Base>& x, const VecAD_reference<Base>& y)
{	return azmul(x, y.ADBase()); }

template <class Base> AD<Base>
azmul(const VecAD_reference<Base>& x, const VecAD_reference<Base>& y)
{	return azmul(x.ADBase(), y.ADBase()); }

template <class Base> AD<Base>
azmul(const VecAD_reference<Base>& x, const AD<Base>& y)
{	return azmul(x.ADBase(), y); }
// -------------------------------------------------------------------------
// Operations with Base

template <class Base> AD<Base>
azmul(const Base& x, const AD<Base>& y)
{	return azmul(AD<Base>(x), y); }

template <class Base> AD<Base>
azmul(const Base& x, const VecAD_reference<Base>& y)
{	return azmul(AD<Base>(x), y.ADBase()); }

template <class Base> AD<Base>
azmul(const AD<Base>& x, const Base& y)
{	return azmul(x, AD<Base>(y)); }

template <class Base> AD<Base>
azmul(const VecAD_reference<Base>& x, const Base& y)
{	return azmul(x.ADBase(), AD<Base>(y)); }

// ==========================================================================
} // END_CPPAD_NAMESPACE

# endif
