# ifndef CPPAD_CORE_HESSIAN_HPP
# define CPPAD_CORE_HESSIAN_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin Hessian$$
$spell
	hes
	typename
	Taylor
	HesLuDet
	const
$$


$section Hessian: Easy Driver$$
$mindex second derivative$$

$head Syntax$$
$icode%hes% = %f%.Hessian(%x%, %w%)
%$$
$icode%hes% = %f%.Hessian(%x%, %l%)
%$$


$head Purpose$$
We use $latex F : B^n \rightarrow B^m$$ to denote the
$cref/AD function/glossary/AD Function/$$ corresponding to $icode f$$.
The syntax above sets $icode hes$$ to the Hessian
The syntax above sets $icode h$$ to the Hessian
$latex \[
	hes = \dpow{2}{x} \sum_{i=1}^m w_i F_i (x)
\] $$
The routine $cref sparse_hessian$$ may be faster in the case
where the Hessian is sparse.

$head f$$
The object $icode f$$ has prototype
$codei%
	ADFun<%Base%> %f%
%$$
Note that the $cref ADFun$$ object $icode f$$ is not $code const$$
(see $cref/Hessian Uses Forward/Hessian/Hessian Uses Forward/$$ below).

$head x$$
The argument $icode x$$ has prototype
$codei%
	const %Vector% &%x%
%$$
(see $cref/Vector/Hessian/Vector/$$ below)
and its size
must be equal to $icode n$$, the dimension of the
$cref/domain/seq_property/Domain/$$ space for $icode f$$.
It specifies
that point at which to evaluate the Hessian.

$head l$$
If the argument $icode l$$ is present, it has prototype
$codei%
	size_t %l%
%$$
and is less than $icode m$$, the dimension of the
$cref/range/seq_property/Range/$$ space for $icode f$$.
It specifies the component of $icode F$$
for which we are evaluating the Hessian.
To be specific, in the case where the argument $icode l$$ is present,
$latex \[
	w_i = \left\{ \begin{array}{ll}
		1 & i = l \\
		0 & {\rm otherwise}
	\end{array} \right.
\] $$

$head w$$
If the argument $icode w$$ is present, it has prototype
$codei%
	const %Vector% &%w%
%$$
and size $latex m$$.
It specifies the value of $latex w_i$$ in the expression
for $icode h$$.

$head hes$$
The result $icode hes$$ has prototype
$codei%
	%Vector% %hes%
%$$
(see $cref/Vector/Hessian/Vector/$$ below)
and its size is $latex n * n$$.
For $latex j = 0 , \ldots , n - 1 $$
and $latex \ell = 0 , \ldots , n - 1$$
$latex \[
	hes [ j * n + \ell ] = \DD{ w^{\rm T} F }{ x_j }{ x_\ell } ( x )
\] $$

$head Vector$$
The type $icode Vector$$ must be a $cref SimpleVector$$ class with
$cref/elements of type/SimpleVector/Elements of Specified Type/$$
$icode Base$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$head Hessian Uses Forward$$
After each call to $cref Forward$$,
the object $icode f$$ contains the corresponding
$cref/Taylor coefficients/glossary/Taylor Coefficient/$$.
After a call to $code Hessian$$,
the zero order Taylor coefficients correspond to
$icode%f%.Forward(0, %x%)%$$
and the other coefficients are unspecified.

$head Example$$
$children%
	example/general/hessian.cpp%
	example/general/hes_lagrangian.cpp
%$$
The routines
$cref hessian.cpp$$ and
$cref hes_lagrangian.cpp$$
are examples and tests of $code Hessian$$.
They return $code true$$, if they succeed and $code false$$ otherwise.


$end
-----------------------------------------------------------------------------
*/

//  BEGIN CppAD namespace
namespace CppAD {

template <typename Base>
template <typename Vector>
Vector ADFun<Base>::Hessian(const Vector &x, size_t l)
{	size_t i, m = Range();
	CPPAD_ASSERT_KNOWN(
		l < m,
		"Hessian: index i is not less than range dimension for f"
	);

	Vector w(m);
	for(i = 0; i < m; i++)
		w[i] = Base(0.0);
	w[l] = Base(1.0);

	return Hessian(x, w);
}


template <typename Base>
template <typename Vector>
Vector ADFun<Base>::Hessian(const Vector &x, const Vector &w)
{	size_t j;
	size_t k;

	size_t n = Domain();

	// check Vector is Simple Vector class with Base type elements
	CheckSimpleVector<Base, Vector>();

	CPPAD_ASSERT_KNOWN(
		size_t(x.size()) == n,
		"Hessian: length of x not equal domain dimension for f"
	);
	CPPAD_ASSERT_KNOWN(
		size_t(w.size()) == Range(),
		"Hessian: length of w not equal range dimension for f"
	);

	// point at which we are evaluating the Hessian
	Forward(0, x);

	// define the return value
	Vector hes(n * n);

	// direction vector for calls to forward
	Vector u(n);
	for(j = 0; j < n; j++)
		u[j] = Base(0.0);


	// location for return values from Reverse
	Vector ddw(n * 2);

	// loop over forward directions
	for(j = 0; j < n; j++)
	{	// evaluate partials of entire function w.r.t. j-th coordinate
		u[j] = Base(1.0);
		Forward(1, u);
		u[j] = Base(0.0);

		// evaluate derivative of partial corresponding to F_i
		ddw = Reverse(2, w);

		// return desired components
		for(k = 0; k < n; k++)
			hes[k * n + j] = ddw[k * 2 + 1];
	}

	return hes;
}

} // END CppAD namespace

# endif
