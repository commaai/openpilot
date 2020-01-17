# ifndef CPPAD_CORE_ABS_HPP
# define CPPAD_CORE_ABS_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
-------------------------------------------------------------------------------
$begin abs$$
$spell
	fabs
	Vec
	std
	faq
	Taylor
	Cpp
	namespace
	const
	abs
$$

$section AD Absolute Value Functions: abs, fabs$$

$head Syntax$$
$icode%y% = abs(%x%)
%$$
$icode%y% = fabs(%x%)%$$

$head x, y$$
See the $cref/possible types/unary_standard_math/Possible Types/$$
for a unary standard math function.

$head Atomic$$
In the case where $icode x$$ is an AD type,
this is an $cref/atomic operation/glossary/Operation/Atomic/$$.

$head Complex Types$$
The functions $code abs$$ and $icode fabs$$
are not defined for the base types
$code std::complex<float>$$ or $code std::complex<double>$$
because the complex $code abs$$ function is not complex differentiable
(see $cref/complex types faq/Faq/Complex Types/$$).

$head Derivative$$
CppAD defines the derivative of the $code abs$$ function is
the $cref sign$$ function; i.e.,
$latex \[
{\rm abs}^{(1)} ( x ) = {\rm sign} (x ) =
\left\{ \begin{array}{rl}
	+1 & {\rm if} \; x > 0 \\
	0  & {\rm if} \; x = 0 \\
	-1 & {\rm if} \; x < 0
\end{array} \right.
\] $$
The result for $icode%x% == 0%$$ used to be a directional derivative.

$head Example$$
$children%
	example/general/fabs.cpp
%$$
The file
$cref fabs.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$end
-------------------------------------------------------------------------------
*/

//  BEGIN CppAD namespace
namespace CppAD {

template <class Base>
AD<Base> AD<Base>::abs_me (void) const
{
	AD<Base> result;
	result.value_ = abs(value_);
	CPPAD_ASSERT_UNKNOWN( Parameter(result) );

	if( Variable(*this) )
	{	// add this operation to the tape
		CPPAD_ASSERT_UNKNOWN( local::NumRes(local::AbsOp) == 1 );
		CPPAD_ASSERT_UNKNOWN( local::NumArg(local::AbsOp) == 1 );
		local::ADTape<Base> *tape = tape_this();

		// corresponding operand address
		tape->Rec_.PutArg(taddr_);
		// put operator in the tape
		result.taddr_ = tape->Rec_.PutOp(local::AbsOp);
		// make result a variable
		result.tape_id_    = tape->id_;
	}
	return result;
}

template <class Base>
inline AD<Base> abs(const AD<Base> &x)
{	return x.abs_me(); }

template <class Base>
inline AD<Base> abs(const VecAD_reference<Base> &x)
{	return x.ADBase().abs_me(); }

} // END CppAD namespace

# endif
