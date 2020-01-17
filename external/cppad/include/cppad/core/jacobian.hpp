# ifndef CPPAD_CORE_JACOBIAN_HPP
# define CPPAD_CORE_JACOBIAN_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin Jacobian$$
$spell
	jac
	typename
	Taylor
	Jacobian
	DetLu
	const
$$


$section Jacobian: Driver Routine$$
$mindex Jacobian first derivative$$

$head Syntax$$
$icode%jac% = %f%.Jacobian(%x%)%$$


$head Purpose$$
We use $latex F : B^n \rightarrow B^m$$ to denote the
$cref/AD function/glossary/AD Function/$$ corresponding to $icode f$$.
The syntax above sets $icode jac$$ to the
Jacobian of $icode F$$ evaluated at $icode x$$; i.e.,
$latex \[
	jac = F^{(1)} (x)
\] $$

$head f$$
The object $icode f$$ has prototype
$codei%
	ADFun<%Base%> %f%
%$$
Note that the $cref ADFun$$ object $icode f$$ is not $code const$$
(see $cref/Forward or Reverse/Jacobian/Forward or Reverse/$$ below).

$head x$$
The argument $icode x$$ has prototype
$codei%
	const %Vector% &%x%
%$$
(see $cref/Vector/Jacobian/Vector/$$ below)
and its size
must be equal to $icode n$$, the dimension of the
$cref/domain/seq_property/Domain/$$ space for $icode f$$.
It specifies
that point at which to evaluate the Jacobian.

$head jac$$
The result $icode jac$$ has prototype
$codei%
	%Vector% %jac%
%$$
(see $cref/Vector/Jacobian/Vector/$$ below)
and its size is $latex m * n$$; i.e., the product of the
$cref/domain/seq_property/Domain/$$
and
$cref/range/seq_property/Range/$$
dimensions for $icode f$$.
For $latex i = 0 , \ldots , m - 1 $$
and $latex j = 0 , \ldots , n - 1$$
$latex \[.
	jac[ i * n + j ] = \D{ F_i }{ x_j } ( x )
\] $$


$head Vector$$
The type $icode Vector$$ must be a $cref SimpleVector$$ class with
$cref/elements of type/SimpleVector/Elements of Specified Type/$$
$icode Base$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$head Forward or Reverse$$
This will use order zero Forward mode and either
order one Forward or order one Reverse to compute the Jacobian
(depending on which it estimates will require less work).
After each call to $cref Forward$$,
the object $icode f$$ contains the corresponding
$cref/Taylor coefficients/glossary/Taylor Coefficient/$$.
After a call to $code Jacobian$$,
the zero order Taylor coefficients correspond to
$icode%f%.Forward(0, %x%)%$$
and the other coefficients are unspecified.

$head Example$$
$children%
	example/general/jacobian.cpp
%$$
The routine
$cref/Jacobian/jacobian.cpp/$$ is both an example and test.
It returns $code true$$, if it succeeds and $code false$$ otherwise.

$end
-----------------------------------------------------------------------------
*/

//  BEGIN CppAD namespace
namespace CppAD {

template <typename Base, typename Vector>
void JacobianFor(ADFun<Base> &f, const Vector &x, Vector &jac)
{	size_t i;
	size_t j;

	size_t n = f.Domain();
	size_t m = f.Range();

	// check Vector is Simple Vector class with Base type elements
	CheckSimpleVector<Base, Vector>();

	CPPAD_ASSERT_UNKNOWN( size_t(x.size())   == f.Domain() );
	CPPAD_ASSERT_UNKNOWN( size_t(jac.size()) == f.Range() * f.Domain() );

	// argument and result for forward mode calculations
	Vector u(n);
	Vector v(m);

	// initialize all the components
	for(j = 0; j < n; j++)
		u[j] = Base(0.0);

	// loop through the different coordinate directions
	for(j = 0; j < n; j++)
	{	// set u to the j-th coordinate direction
		u[j] = Base(1.0);

		// compute the partial of f w.r.t. this coordinate direction
		v = f.Forward(1, u);

		// reset u to vector of all zeros
		u[j] = Base(0.0);

		// return the result
		for(i = 0; i < m; i++)
			jac[ i * n + j ] = v[i];
	}
}
template <typename Base, typename Vector>
void JacobianRev(ADFun<Base> &f, const Vector &x, Vector &jac)
{	size_t i;
	size_t j;

	size_t n = f.Domain();
	size_t m = f.Range();

	CPPAD_ASSERT_UNKNOWN( size_t(x.size())   == f.Domain() );
	CPPAD_ASSERT_UNKNOWN( size_t(jac.size()) == f.Range() * f.Domain() );

	// argument and result for reverse mode calculations
	Vector u(n);
	Vector v(m);

	// initialize all the components
	for(i = 0; i < m; i++)
		v[i] = Base(0.0);

	// loop through the different coordinate directions
	for(i = 0; i < m; i++)
	{	if( f.Parameter(i) )
		{	// return zero for this component of f
			for(j = 0; j < n; j++)
				jac[ i * n + j ] = Base(0.0);
		}
		else
		{
			// set v to the i-th coordinate direction
			v[i] = Base(1.0);

			// compute the derivative of this component of f
			u = f.Reverse(1, v);

			// reset v to vector of all zeros
			v[i] = Base(0.0);

			// return the result
			for(j = 0; j < n; j++)
				jac[ i * n + j ] = u[j];
		}
	}
}

template <typename Base>
template <typename Vector>
Vector ADFun<Base>::Jacobian(const Vector &x)
{	size_t i;
	size_t n = Domain();
	size_t m = Range();

	CPPAD_ASSERT_KNOWN(
		size_t(x.size()) == n,
		"Jacobian: length of x not equal domain dimension for F"
	);

	// point at which we are evaluating the Jacobian
	Forward(0, x);

	// work factor for forward mode
	size_t workForward = n;

	// work factor for reverse mode
	size_t workReverse = 0;
	for(i = 0; i < m; i++)
	{	if( ! Parameter(i) )
			++workReverse;
	}

	// choose the method with the least work
	Vector jac( n * m );
	if( workForward <= workReverse )
		JacobianFor(*this, x, jac);
	else	JacobianRev(*this, x, jac);

	return jac;
}

} // END CppAD namespace

# endif
