# ifndef CPPAD_CORE_BENDER_QUAD_HPP
# define CPPAD_CORE_BENDER_QUAD_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin BenderQuad$$
$spell
	cppad.hpp
	BAvector
	gx
	gxx
	CppAD
	Fy
	dy
	Jacobian
	ADvector
	const
	dg
	ddg
$$


$section Computing Jacobian and Hessian of Bender's Reduced Objective$$
$mindex BenderQuad$$

$head Syntax$$
$codei%
# include <cppad/cppad.hpp>
BenderQuad(%x%, %y%, %fun%, %g%, %gx%, %gxx%)%$$

$head See Also$$
$cref opt_val_hes$$

$head Problem$$
The type $cref/ADvector/BenderQuad/ADvector/$$ cannot be determined
form the arguments above
(currently the type $icode ADvector$$ must be
$codei%CPPAD_TESTVECTOR(%Base%)%$$.)
This will be corrected in the future by requiring $icode Fun$$
to define $icode%Fun%::vector_type%$$ which will specify the
type $icode ADvector$$.

$head Purpose$$
We are given the optimization problem
$latex \[
\begin{array}{rcl}
{\rm minimize} & F(x, y) & {\rm w.r.t.} \; (x, y) \in \B{R}^n \times \B{R}^m
\end{array}
\] $$
that is convex with respect to $latex y$$.
In addition, we are given a set of equations $latex H(x, y)$$
such that
$latex \[
	H[ x , Y(x) ] = 0 \;\; \Rightarrow \;\; F_y [ x , Y(x) ] = 0
\] $$
(In fact, it is often the case that $latex H(x, y) = F_y (x, y)$$.)
Furthermore, it is easy to calculate a Newton step for these equations; i.e.,
$latex \[
	dy = - [ \partial_y H(x, y)]^{-1} H(x, y)
\] $$
The purpose of this routine is to compute the
value, Jacobian, and Hessian of the reduced objective function
$latex \[
	G(x) = F[ x , Y(x) ]
\] $$
Note that if only the value and Jacobian are needed, they can be
computed more quickly using the relations
$latex \[
	G^{(1)} (x) = \partial_x F [x, Y(x) ]
\] $$

$head x$$
The $code BenderQuad$$ argument $icode x$$ has prototype
$codei%
	const %BAvector% &%x%
%$$
(see $cref/BAvector/BenderQuad/BAvector/$$ below)
and its size must be equal to $icode n$$.
It specifies the point at which we evaluating
the reduced objective function and its derivatives.


$head y$$
The $code BenderQuad$$ argument $icode y$$ has prototype
$codei%
	const %BAvector% &%y%
%$$
and its size must be equal to $icode m$$.
It must be equal to $latex Y(x)$$; i.e.,
it must solve the problem in $latex y$$ for this given value of $latex x$$
$latex \[
\begin{array}{rcl}
	{\rm minimize} & F(x, y) & {\rm w.r.t.} \; y \in \B{R}^m
\end{array}
\] $$

$head fun$$
The $code BenderQuad$$ object $icode fun$$
must support the member functions listed below.
The $codei%AD<%Base%>%$$ arguments will be variables for
a tape created by a call to $cref Independent$$ from $code BenderQuad$$
(hence they can not be combined with variables corresponding to a
different tape).

$subhead fun.f$$
The $code BenderQuad$$ argument $icode fun$$ supports the syntax
$codei%
	%f% = %fun%.f(%x%, %y%)
%$$
The $icode%fun%.f%$$ argument $icode x$$ has prototype
$codei%
	const %ADvector% &%x%
%$$
(see $cref/ADvector/BenderQuad/ADvector/$$ below)
and its size must be equal to $icode n$$.
The $icode%fun%.f%$$ argument $icode y$$ has prototype
$codei%
	const %ADvector% &%y%
%$$
and its size must be equal to $icode m$$.
The $icode%fun%.f%$$ result $icode f$$ has prototype
$codei%
	%ADvector% %f%
%$$
and its size must be equal to one.
The value of $icode f$$ is
$latex \[
	f = F(x, y)
\] $$.

$subhead fun.h$$
The $code BenderQuad$$ argument $icode fun$$ supports the syntax
$codei%
	%h% = %fun%.h(%x%, %y%)
%$$
The $icode%fun%.h%$$ argument $icode x$$ has prototype
$codei%
	const %ADvector% &%x%
%$$
and its size must be equal to $icode n$$.
The $icode%fun%.h%$$ argument $icode y$$ has prototype
$codei%
	const %BAvector% &%y%
%$$
and its size must be equal to $icode m$$.
The $icode%fun%.h%$$ result $icode h$$ has prototype
$codei%
	%ADvector% %h%
%$$
and its size must be equal to $icode m$$.
The value of $icode h$$ is
$latex \[
	h = H(x, y)
\] $$.

$subhead fun.dy$$
The $code BenderQuad$$ argument $icode fun$$ supports the syntax
$codei%
	%dy% = %fun%.dy(%x%, %y%, %h%)

%x%
%$$
The $icode%fun%.dy%$$ argument $icode x$$ has prototype
$codei%
	const %BAvector% &%x%
%$$
and its size must be equal to $icode n$$.
Its value will be exactly equal to the $code BenderQuad$$ argument
$icode x$$ and values depending on it can be stored as private objects
in $icode f$$ and need not be recalculated.
$codei%

%y%
%$$
The $icode%fun%.dy%$$ argument $icode y$$ has prototype
$codei%
	const %BAvector% &%y%
%$$
and its size must be equal to $icode m$$.
Its value will be exactly equal to the $code BenderQuad$$ argument
$icode y$$ and values depending on it can be stored as private objects
in $icode f$$ and need not be recalculated.
$codei%

%h%
%$$
The $icode%fun%.dy%$$ argument $icode h$$ has prototype
$codei%
	const %ADvector% &%h%
%$$
and its size must be equal to $icode m$$.
$codei%

%dy%
%$$
The $icode%fun%.dy%$$ result $icode dy$$ has prototype
$codei%
	%ADvector% %dy%
%$$
and its size must be equal to $icode m$$.
The return value $icode dy$$ is given by
$latex \[
	dy = - [ \partial_y H (x , y) ]^{-1} h
\] $$
Note that if $icode h$$ is equal to $latex H(x, y)$$,
$latex dy$$ is the Newton step for finding a zero
of $latex H(x, y)$$ with respect to $latex y$$;
i.e.,
$latex y + dy$$ is an approximate solution for the equation
$latex H (x, y + dy) = 0$$.

$head g$$
The argument $icode g$$ has prototype
$codei%
	%BAvector% &%g%
%$$
and has size one.
The input value of its element does not matter.
On output,
it contains the value of $latex G (x)$$; i.e.,
$latex \[
	g[0] = G (x)
\] $$

$head gx$$
The argument $icode gx$$ has prototype
$codei%
	%BAvector% &%gx%
%$$
and has size $latex n $$.
The input values of its elements do not matter.
On output,
it contains the Jacobian of $latex G (x)$$; i.e.,
for $latex j = 0 , \ldots , n-1$$,
$latex \[
	gx[ j ] = G^{(1)} (x)_j
\] $$

$head gxx$$
The argument $icode gx$$ has prototype
$codei%
	%BAvector% &%gxx%
%$$
and has size $latex n \times n$$.
The input values of its elements do not matter.
On output,
it contains the Hessian of $latex G (x)$$; i.e.,
for $latex i = 0 , \ldots , n-1$$, and
$latex j = 0 , \ldots , n-1$$,
$latex \[
	gxx[ i * n + j ] = G^{(2)} (x)_{i,j}
\] $$

$head BAvector$$
The type $icode BAvector$$ must be a
$cref SimpleVector$$ class.
We use $icode Base$$ to refer to the type of the elements of
$icode BAvector$$; i.e.,
$codei%
	%BAvector%::value_type
%$$

$head ADvector$$
The type $icode ADvector$$ must be a
$cref SimpleVector$$ class with elements of type
$codei%AD<%Base%>%$$; i.e.,
$codei%
	%ADvector%::value_type
%$$
must be the same type as
$codei%
	AD< %BAvector%::value_type >
%$$.


$head Example$$
$children%
	example/general/bender_quad.cpp
%$$
The file
$cref bender_quad.cpp$$
contains an example and test of this operation.
It returns true if it succeeds and false otherwise.


$end
-----------------------------------------------------------------------------
*/

namespace CppAD { // BEGIN CppAD namespace

template <class BAvector, class Fun>
void BenderQuad(
	const BAvector   &x     ,
	const BAvector   &y     ,
	Fun               fun   ,
	BAvector         &g     ,
	BAvector         &gx    ,
	BAvector         &gxx   )
{	// determine the base type
	typedef typename BAvector::value_type Base;

	// check that BAvector is a SimpleVector class
	CheckSimpleVector<Base, BAvector>();

	// declare the ADvector type
	typedef CPPAD_TESTVECTOR(AD<Base>) ADvector;

	// size of the x and y spaces
	size_t n = size_t(x.size());
	size_t m = size_t(y.size());

	// check the size of gx and gxx
	CPPAD_ASSERT_KNOWN(
		g.size() == 1,
		"BenderQuad: size of the vector g is not equal to 1"
	);
	CPPAD_ASSERT_KNOWN(
		size_t(gx.size()) == n,
		"BenderQuad: size of the vector gx is not equal to n"
	);
	CPPAD_ASSERT_KNOWN(
		size_t(gxx.size()) == n * n,
		"BenderQuad: size of the vector gxx is not equal to n * n"
	);

	// some temporary indices
	size_t i, j;

	// variable versions x
	ADvector vx(n);
	for(j = 0; j < n; j++)
		vx[j] = x[j];

	// declare the independent variables
	Independent(vx);

	// evaluate h = H(x, y)
	ADvector h(m);
	h = fun.h(vx, y);

	// evaluate dy (x) = Newton step as a function of x through h only
	ADvector dy(m);
	dy = fun.dy(x, y, h);

	// variable version of y
	ADvector vy(m);
	for(j = 0; j < m; j++)
		vy[j] = y[j] + dy[j];

	// evaluate G~ (x) = F [ x , y + dy(x) ]
	ADvector gtilde(1);
	gtilde = fun.f(vx, vy);

	// AD function object that corresponds to G~ (x)
	// We will make heavy use of this tape, so optimize it
	ADFun<Base> Gtilde;
	Gtilde.Dependent(vx, gtilde);
	Gtilde.optimize();

	// value of G(x)
	g = Gtilde.Forward(0, x);

	// initial forward direction vector as zero
	BAvector dx(n);
	for(j = 0; j < n; j++)
		dx[j] = Base(0.0);

	// weight, first and second order derivative values
	BAvector dg(1), w(1), ddw(2 * n);
	w[0] = 1.;


	// Jacobian and Hessian of G(x) is equal Jacobian and Hessian of Gtilde
	for(j = 0; j < n; j++)
	{	// compute partials in x[j] direction
		dx[j] = Base(1.0);
		dg    = Gtilde.Forward(1, dx);
		gx[j] = dg[0];

		// restore the dx vector to zero
		dx[j] = Base(0.0);

		// compute second partials w.r.t x[j] and x[l]  for l = 1, n
		ddw = Gtilde.Reverse(2, w);
		for(i = 0; i < n; i++)
			gxx[ i * n + j ] = ddw[ i * 2 + 1 ];
	}

	return;
}

} // END CppAD namespace

# endif
