# ifndef CPPAD_CORE_REV_ONE_HPP
# define CPPAD_CORE_REV_ONE_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin RevOne$$
$spell
	dw
	Taylor
	const
$$




$section First Order Derivative: Driver Routine$$
$mindex derivative easy$$

$head Syntax$$
$icode%dw% = %f%.RevOne(%x%, %i%)%$$


$head Purpose$$
We use $latex F : B^n \rightarrow B^m$$ to denote the
$cref/AD function/glossary/AD Function/$$ corresponding to $icode f$$.
The syntax above sets $icode dw$$ to the
derivative of $latex F_i$$ with respect to $latex x$$; i.e.,
$latex \[
dw =
F_i^{(1)} (x)
= \left[
	\D{ F_i }{ x_0 } (x) , \cdots , \D{ F_i }{ x_{n-1} } (x)
\right]
\] $$

$head f$$
The object $icode f$$ has prototype
$codei%
	ADFun<%Base%> %f%
%$$
Note that the $cref ADFun$$ object $icode f$$ is not $code const$$
(see $cref/RevOne Uses Forward/RevOne/RevOne Uses Forward/$$ below).

$head x$$
The argument $icode x$$ has prototype
$codei%
	const %Vector% &%x%
%$$
(see $cref/Vector/RevOne/Vector/$$ below)
and its size
must be equal to $icode n$$, the dimension of the
$cref/domain/seq_property/Domain/$$ space for $icode f$$.
It specifies
that point at which to evaluate the derivative.

$head i$$
The index $icode i$$ has prototype
$codei%
	size_t %i%
%$$
and is less than $latex m$$, the dimension of the
$cref/range/seq_property/Range/$$ space for $icode f$$.
It specifies the
component of $latex F$$ that we are computing the derivative of.

$head dw$$
The result $icode dw$$ has prototype
$codei%
	%Vector% %dw%
%$$
(see $cref/Vector/RevOne/Vector/$$ below)
and its size is $icode n$$, the dimension of the
$cref/domain/seq_property/Domain/$$ space for $icode f$$.
The value of $icode dw$$ is the derivative of $latex F_i$$
evaluated at $icode x$$; i.e.,
for $latex j = 0 , \ldots , n - 1 $$
$latex \[.
	dw[ j ] = \D{ F_i }{ x_j } ( x )
\] $$

$head Vector$$
The type $icode Vector$$ must be a $cref SimpleVector$$ class with
$cref/elements of type/SimpleVector/Elements of Specified Type/$$
$icode Base$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$head RevOne Uses Forward$$
After each call to $cref Forward$$,
the object $icode f$$ contains the corresponding
$cref/Taylor coefficients/glossary/Taylor Coefficient/$$.
After a call to $code RevOne$$,
the zero order Taylor coefficients correspond to
$icode%f%.Forward(0, %x%)%$$
and the other coefficients are unspecified.

$head Example$$
$children%
	example/general/rev_one.cpp
%$$
The routine
$cref/RevOne/rev_one.cpp/$$ is both an example and test.
It returns $code true$$, if it succeeds and $code false$$ otherwise.

$end
-----------------------------------------------------------------------------
*/

//  BEGIN CppAD namespace
namespace CppAD {

template <typename Base>
template <typename Vector>
Vector ADFun<Base>::RevOne(const Vector  &x, size_t i)
{	size_t i1;

	size_t n = Domain();
	size_t m = Range();

	// check Vector is Simple Vector class with Base type elements
	CheckSimpleVector<Base, Vector>();

	CPPAD_ASSERT_KNOWN(
		x.size() == n,
		"RevOne: Length of x not equal domain dimension for f"
	);
	CPPAD_ASSERT_KNOWN(
		i < m,
		"RevOne: the index i is not less than range dimension for f"
	);

	// point at which we are evaluating the derivative
	Forward(0, x);

	// component which are are taking the derivative of
	Vector w(m);
	for(i1 = 0; i1 < m; i1++)
		w[i1] = 0.;
	w[i] = Base(1.0);

	// dimension the return value
	Vector dw(n);

	// compute the return value
	dw = Reverse(1, w);

	return dw;
}

} // END CppAD namespace

# endif
