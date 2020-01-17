# ifndef CPPAD_UTILITY_ROMBERG_MUL_HPP
# define CPPAD_UTILITY_ROMBERG_MUL_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin RombergMul$$
$spell
	cppad.hpp
	bool
	const
	Cpp
	RombergMulMul
$$

$section Multi-dimensional Romberg Integration$$
$mindex integrate multi dimensional dimension$$


$head Syntax$$
$codei%# include <cppad/utility/romberg_mul.hpp>
%$$
$codei%RombergMul<%Fun%, %SizeVector%, %FloatVector%, %m%> %R%$$
$pre
$$
$icode%r% = %R%(%F%, %a%, %b%, %n%, %p%, %e%)%$$


$head Description$$
Returns the Romberg integration estimate
$latex r$$ for the multi-dimensional integral
$latex \[
r =
\int_{a[0]}^{b[0]} \cdots \int_{a[m-1]}^{b[m-1]}
\; F(x) \;
{\bf d} x_0 \cdots {\bf d} x_{m-1}
\; + \;
\sum_{i=0}^{m-1}
O \left[ ( b[i] - a[i] ) / 2^{n[i]-1} \right]^{2(p[i]+1)}
\] $$

$head Include$$
The file $code cppad/romberg_mul.hpp$$ is included by $code cppad/cppad.hpp$$
but it can also be included separately with out the rest of
the $code CppAD$$ routines.

$head m$$
The template parameter $icode m$$ must be convertible to a $code size_t$$
object with a value that can be determined at compile time; for example
$code 2$$.
It determines the dimension of the domain space for the integration.

$head r$$
The return value $icode r$$ has prototype
$codei%
	%Float% %r%
%$$
It is the estimate computed by $code RombergMul$$ for the integral above
(see description of $cref/Float/RombergMul/Float/$$ below).

$head F$$
The object $icode F$$ has the prototype
$codei%
	%Fun% &%F%
%$$
It must support the operation
$codei%
	%F%(%x%)
%$$
The argument $icode x$$ to $icode F$$ has prototype
$codei%
	const %Float% &%x%
%$$
The return value of $icode F$$ is a $icode Float$$ object

$head a$$
The argument $icode a$$ has prototype
$codei%
	const %FloatVector% &%a%
%$$
It specifies the lower limit for the integration
(see description of $cref/FloatVector/RombergMul/FloatVector/$$ below).

$head b$$
The argument $icode b$$ has prototype
$codei%
	const %FloatVector% &%b%
%$$
It specifies the upper limit for the integration.

$head n$$
The argument $icode n$$ has prototype
$codei%
	const %SizeVector% &%n%
%$$
A total number of $latex 2^{n[i]-1} + 1$$
evaluations of $icode%F%(%x%)%$$ are used to estimate the integral
with respect to $latex {\bf d} x_i$$.

$head p$$
The argument $icode p$$ has prototype
$codei%
	const %SizeVector% &%p%
%$$
For $latex i = 0 , \ldots , m-1$$,
$latex n[i]$$ determines the accuracy order in the
approximation for the integral
that is returned by $code RombergMul$$.
The values in $icode p$$ must be less than or equal $icode n$$; i.e.,
$icode%p%[%i%] <= %n%[%i%]%$$.

$head e$$
The argument $icode e$$ has prototype
$codei%
	%Float% &%e%
%$$
The input value of $icode e$$ does not matter
and its output value is an approximation for the absolute error in
the integral estimate.

$head Float$$
The type $icode Float$$ is defined as the type of the elements of
$cref/FloatVector/RombergMul/FloatVector/$$.
The type $icode Float$$ must satisfy the conditions
for a $cref NumericType$$ type.
The routine $cref CheckNumericType$$ will generate an error message
if this is not the case.
In addition, if $icode x$$ and $icode y$$ are $icode Float$$ objects,
$codei%
	%x% < %y%
%$$
returns the $code bool$$ value true if $icode x$$ is less than
$icode y$$ and false otherwise.

$head FloatVector$$
The type $icode FloatVector$$ must be a $cref SimpleVector$$ class.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.


$children%
	example/utility/romberg_mul.cpp
%$$
$head Example$$
$comment%
	example/utility/romberg_mul.cpp
%$$
The file
$cref Rombergmul.cpp$$
contains an example and test a test of using this routine.
It returns true if it succeeds and false otherwise.

$head Source Code$$
The source code for this routine is in the file
$code cppad/romberg_mul.hpp$$.

$end
*/

# include <cppad/utility/romberg_one.hpp>
# include <cppad/utility/check_numeric_type.hpp>
# include <cppad/utility/check_simple_vector.hpp>

namespace CppAD { // BEGIN CppAD namespace

template <class Fun, class FloatVector>
class SliceLast {
	typedef typename FloatVector::value_type Float;
private:
	Fun        *F;
	size_t      last;
	FloatVector x;
public:
	SliceLast( Fun *F_, size_t last_, const FloatVector &x_ )
	: F(F_) , last(last_), x(last + 1)
	{	size_t i;
		for(i = 0; i < last; i++)
			x[i] = x_[i];
	}
	double operator()(const Float &xlast)
	{	x[last] = xlast;
		return (*F)(x);
	}
};

template <class Fun, class SizeVector, class FloatVector, class Float>
class IntegrateLast {
private:
	Fun                 *F;
	const size_t        last;
	const FloatVector   a;
	const FloatVector   b;
	const SizeVector    n;
	const SizeVector    p;
	Float               esum;
	size_t              ecount;

public:
	IntegrateLast(
		Fun                *F_    ,
		size_t              last_ ,
		const FloatVector  &a_    ,
		const FloatVector  &b_    ,
		const SizeVector   &n_    ,
		const SizeVector   &p_    )
	: F(F_) , last(last_), a(a_) , b(b_) , n(n_) , p(p_)
	{ }
	Float operator()(const FloatVector           &x)
	{	Float r, e;
		SliceLast<Fun, FloatVector           > S(F, last, x);
		r     = CppAD::RombergOne(
			S, a[last], b[last], n[last], p[last], e
		);
		esum = esum + e;
		ecount++;
		return r;
	}
	void ClearEsum(void)
	{	esum   = 0.; }
	Float GetEsum(void)
	{	return esum; }

	void ClearEcount(void)
	{	ecount   = 0; }
	size_t GetEcount(void)
	{	return ecount; }
};

template <class Fun, class SizeVector, class FloatVector, size_t m>
class RombergMul {
	typedef typename FloatVector::value_type Float;
public:
	RombergMul(void)
	{	}
	Float operator() (
		Fun                 &F  ,
		const FloatVector   &a  ,
		const FloatVector   &b  ,
		const SizeVector    &n  ,
		const SizeVector    &p  ,
		Float               &e  )
	{	Float r;

		typedef IntegrateLast<
			Fun         ,
			SizeVector  ,
			FloatVector ,
			Float       > IntegrateOne;

		IntegrateOne Fm1(&F, m-1, a, b, n, p);
		RombergMul<
			IntegrateOne,
			SizeVector  ,
			FloatVector ,
			m-1         > RombergMulM1;

		Fm1.ClearEsum();
		Fm1.ClearEcount();

		r  = RombergMulM1(Fm1, a, b, n, p, e);

		size_t i, j;
		Float prod = 1;
		size_t pow2 = 1;
		for(i = 0; i < m-1; i++)
		{	prod *= (b[i] - a[i]);
			for(j = 0; j < (n[i] - 1); j++)
				pow2 *= 2;
		}
		assert( Fm1.GetEcount() == (pow2+1) );

		e = e + Fm1.GetEsum() * prod / Float( double(Fm1.GetEcount()) );

		return r;
	}
};

template <class Fun, class SizeVector, class FloatVector>
class RombergMul <Fun, SizeVector, FloatVector, 1> {
	typedef typename FloatVector::value_type Float;
public:
	Float operator() (
		Fun                 &F  ,
		const FloatVector   &a  ,
		const FloatVector   &b  ,
		const SizeVector    &n  ,
		const SizeVector    &p  ,
		Float               &e  )
	{	Float r;
		typedef IntegrateLast<
			Fun         ,
			SizeVector  ,
			FloatVector ,
			Float       > IntegrateOne;

		// check simple vector class specifications
		CheckSimpleVector<Float, FloatVector>();

		// check numeric type specifications
		CheckNumericType<Float>();

		IntegrateOne F0(&F, 0, a, b, n, p);

		F0.ClearEsum();
		F0.ClearEcount();

		r  = F0(a);

		assert( F0.GetEcount() == 1 );
		e = F0.GetEsum();

		return r;
	}
};

} // END CppAD namespace

# endif
