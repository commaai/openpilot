# ifndef CPPAD_CORE_FOR_ONE_HPP
# define CPPAD_CORE_FOR_ONE_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin ForOne$$
$spell
	dy
	typename
	Taylor
	const
$$




$section First Order Partial Derivative: Driver Routine$$
$mindex easy$$

$head Syntax$$
$icode%dy% = %f%.ForOne(%x%, %j%)%$$


$head Purpose$$
We use $latex F : B^n \rightarrow B^m$$ to denote the
$cref/AD function/glossary/AD Function/$$ corresponding to $icode f$$.
The syntax above sets $icode dy$$ to the
partial of $latex F$$ with respect to $latex x_j$$; i.e.,
$latex \[
dy
= \D{F}{ x_j } (x)
= \left[
	\D{ F_0 }{ x_j } (x) , \cdots , \D{ F_{m-1} }{ x_j } (x)
\right]
\] $$

$head f$$
The object $icode f$$ has prototype
$codei%
	ADFun<%Base%> %f%
%$$
Note that the $cref ADFun$$ object $icode f$$ is not $code const$$
(see $cref/ForOne Uses Forward/ForOne/ForOne Uses Forward/$$ below).

$head x$$
The argument $icode x$$ has prototype
$codei%
	const %Vector% &%x%
%$$
(see $cref/Vector/ForOne/Vector/$$ below)
and its size
must be equal to $icode n$$, the dimension of the
$cref/domain/seq_property/Domain/$$ space for $icode f$$.
It specifies
that point at which to evaluate the partial derivative.

$head j$$
The argument $icode j$$ has prototype
$codei%
	size_t %j%
%$$
an is less than $icode n$$,
$cref/domain/seq_property/Domain/$$ space for $icode f$$.
It specifies the component of $icode F$$
for which we are computing the partial derivative.

$head dy$$
The result $icode dy$$ has prototype
$codei%
	%Vector% %dy%
%$$
(see $cref/Vector/ForOne/Vector/$$ below)
and its size is $latex m$$, the dimension of the
$cref/range/seq_property/Range/$$ space for $icode f$$.
The value of $icode dy$$ is the partial of $latex F$$ with respect to
$latex x_j$$ evaluated at $icode x$$; i.e.,
for $latex i = 0 , \ldots , m - 1$$
$latex \[.
	dy[i] = \D{ F_i }{ x_j } ( x )
\] $$


$head Vector$$
The type $icode Vector$$ must be a $cref SimpleVector$$ class with
$cref/elements of type/SimpleVector/Elements of Specified Type/$$
$icode Base$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$head ForOne Uses Forward$$
After each call to $cref Forward$$,
the object $icode f$$ contains the corresponding
$cref/Taylor coefficients/glossary/Taylor Coefficient/$$.
After a call to $code ForOne$$,
the zero order Taylor coefficients correspond to
$icode%f%.Forward(0,%x%)%$$
and the other coefficients are unspecified.

$head Example$$
$children%
	example/general/for_one.cpp
%$$
The routine
$cref/ForOne/for_one.cpp/$$ is both an example and test.
It returns $code true$$, if it succeeds and $code false$$ otherwise.

$end
-----------------------------------------------------------------------------
*/

//  BEGIN CppAD namespace
namespace CppAD {

template <typename Base>
template <typename Vector>
Vector ADFun<Base>::ForOne(const Vector &x, size_t j)
{	size_t j1;

	size_t n = Domain();
	size_t m = Range();

	// check Vector is Simple Vector class with Base type elements
	CheckSimpleVector<Base, Vector>();

	CPPAD_ASSERT_KNOWN(
		x.size() == n,
		"ForOne: Length of x not equal domain dimension for f"
	);
	CPPAD_ASSERT_KNOWN(
		j < n,
		"ForOne: the index j is not less than domain dimension for f"
	);

	// point at which we are evaluating the second partials
	Forward(0, x);

	// direction in which are are taking the derivative
	Vector dx(n);
	for(j1 = 0; j1 < n; j1++)
		dx[j1] = Base(0.0);
	dx[j] = Base(1.0);

	// dimension the return value
	Vector dy(m);

	// compute the return value
	dy = Forward(1, dx);

	return dy;
}

} // END CppAD namespace

# endif
