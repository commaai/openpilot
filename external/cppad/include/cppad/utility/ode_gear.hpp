# ifndef CPPAD_UTILITY_ODE_GEAR_HPP
# define CPPAD_UTILITY_ODE_GEAR_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin OdeGear$$
$spell
	cppad.hpp
	Jan
	bool
	const
	CppAD
	dep
$$


$section An Arbitrary Order Gear Method$$
$mindex OdeGear Ode stiff differential equation$$

$head Syntax$$
$codei%# include <cppad/utility/ode_gear.hpp>
%$$
$codei%OdeGear(%F%, %m%, %n%, %T%, %X%, %e%)%$$


$head Purpose$$
This routine applies
$cref/Gear's Method/OdeGear/Gear's Method/$$
to solve an explicit set of ordinary differential equations.
We are given
$latex f : \B{R} \times \B{R}^n \rightarrow \B{R}^n$$ be a smooth function.
This routine solves the following initial value problem
$latex \[
\begin{array}{rcl}
	x( t_{m-1} )  & = & x^0    \\
	x^\prime (t)  & = & f[t , x(t)]
\end{array}
\] $$
for the value of $latex x( t_m )$$.
If your set of  ordinary differential equations are not stiff
an explicit method may be better (perhaps $cref Runge45$$.)

$head Include$$
The file $code cppad/ode_gear.hpp$$ is included by $code cppad/cppad.hpp$$
but it can also be included separately with out the rest of
the $code CppAD$$ routines.

$head Fun$$
The class $icode Fun$$
and the object $icode F$$ satisfy the prototype
$codei%
	%Fun% &%F%
%$$
This must support the following set of calls
$codei%
	%F%.Ode(%t%, %x%, %f%)
	%F%.Ode_dep(%t%, %x%, %f_x%)
%$$

$subhead t$$
The argument $icode t$$ has prototype
$codei%
	const %Scalar% &%t%
%$$
(see description of $cref/Scalar/OdeGear/Scalar/$$ below).

$subhead x$$
The argument $icode x$$ has prototype
$codei%
	const %Vector% &%x%
%$$
and has size $icode n$$
(see description of $cref/Vector/OdeGear/Vector/$$ below).

$subhead f$$
The argument $icode f$$ to $icode%F%.Ode%$$ has prototype
$codei%
	%Vector% &%f%
%$$
On input and output, $icode f$$ is a vector of size $icode n$$
and the input values of the elements of $icode f$$ do not matter.
On output,
$icode f$$ is set equal to $latex f(t, x)$$
(see $icode f(t, x)$$ in $cref/Purpose/OdeGear/Purpose/$$).

$subhead f_x$$
The argument $icode f_x$$ has prototype
$codei%
	%Vector% &%f_x%
%$$
On input and output, $icode f_x$$ is a vector of size $latex n * n$$
and the input values of the elements of $icode f_x$$ do not matter.
On output,
$latex \[
	f\_x [i * n + j] = \partial_{x(j)} f_i ( t , x )
\] $$

$subhead Warning$$
The arguments $icode f$$, and $icode f_x$$
must have a call by reference in their prototypes; i.e.,
do not forget the $code &$$ in the prototype for
$icode f$$ and $icode f_x$$.

$head m$$
The argument $icode m$$ has prototype
$codei%
	size_t %m%
%$$
It specifies the order (highest power of $latex t$$)
used to represent the function $latex x(t)$$ in the multi-step method.
Upon return from $code OdeGear$$,
the $th i$$ component of the polynomial is defined by
$latex \[
	p_i ( t_j ) = X[ j * n + i ]
\] $$
for $latex j = 0 , \ldots , m$$ (where $latex 0 \leq i < n$$).
The value of $latex m$$ must be greater than or equal one.

$head n$$
The argument $icode n$$ has prototype
$codei%
	size_t %n%
%$$
It specifies the range space dimension of the
vector valued function $latex x(t)$$.

$head T$$
The argument $icode T$$ has prototype
$codei%
	const %Vector% &%T%
%$$
and size greater than or equal to $latex m+1$$.
For $latex j = 0 , \ldots m$$, $latex T[j]$$ is the time
corresponding to time corresponding
to a previous point in the multi-step method.
The value $latex T[m]$$ is the time
of the next point in the multi-step method.
The array $latex T$$ must be monotone increasing; i.e.,
$latex T[j] < T[j+1]$$.
Above and below we often use the shorthand $latex t_j$$ for $latex T[j]$$.


$head X$$
The argument $icode X$$ has the prototype
$codei%
	%Vector% &%X%
%$$
and size greater than or equal to $latex (m+1) * n$$.
On input to $code OdeGear$$,
for $latex j = 0 , \ldots , m-1$$, and
$latex i = 0 , \ldots , n-1$$
$latex \[
	X[ j * n + i ] = x_i ( t_j )
\] $$
Upon return from $code OdeGear$$,
for $latex i = 0 , \ldots , n-1$$
$latex \[
	X[ m * n + i ] \approx x_i ( t_m )
\] $$

$head e$$
The vector $icode e$$ is an approximate error bound for the result; i.e.,
$latex \[
	e[i] \geq | X[ m * n + i ] - x_i ( t_m ) |
\] $$
The order of this approximation is one less than the order of
the solution; i.e.,
$latex \[
	e = O ( h^m )
\] $$
where $latex h$$ is the maximum of $latex t_{j+1} - t_j$$.

$head Scalar$$
The type $icode Scalar$$ must satisfy the conditions
for a $cref NumericType$$ type.
The routine $cref CheckNumericType$$ will generate an error message
if this is not the case.
In addition, the following operations must be defined for
$icode Scalar$$ objects $icode a$$ and $icode b$$:

$table
$bold Operation$$ $cnext $bold Description$$  $rnext
$icode%a% < %b%$$ $cnext
	less than operator (returns a $code bool$$ object)
$tend

$head Vector$$
The type $icode Vector$$ must be a $cref SimpleVector$$ class with
$cref/elements of type Scalar/SimpleVector/Elements of Specified Type/$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$head Example$$
$children%
	example/utility/ode_gear.cpp
%$$
The file
$cref ode_gear.cpp$$
contains an example and test a test of using this routine.
It returns true if it succeeds and false otherwise.

$head Source Code$$
The source code for this routine is in the file
$code cppad/ode_gear.hpp$$.

$head Theory$$
For this discussion we use the shorthand $latex x_j$$
for the value $latex x ( t_j ) \in \B{R}^n$$ which is not to be confused
with $latex x_i (t) \in \B{R}$$ in the notation above.
The interpolating polynomial $latex p(t)$$ is given by
$latex \[
p(t) =
\sum_{j=0}^m
x_j
\frac{
	\prod_{i \neq j} ( t - t_i )
}{
	\prod_{i \neq j} ( t_j - t_i )
}
\] $$
The derivative $latex p^\prime (t)$$ is given by
$latex \[
p^\prime (t) =
\sum_{j=0}^m
x_j
\frac{
	\sum_{i \neq j} \prod_{k \neq i,j} ( t - t_k )
}{
	\prod_{k \neq j} ( t_j - t_k )
}
\] $$
Evaluating the derivative at the point $latex t_\ell$$ we have
$latex \[
\begin{array}{rcl}
p^\prime ( t_\ell ) & = &
x_\ell
\frac{
	\sum_{i \neq \ell} \prod_{k \neq i,\ell} ( t_\ell - t_k )
}{
	\prod_{k \neq \ell} ( t_\ell - t_k )
}
+
\sum_{j \neq \ell}
x_j
\frac{
	\sum_{i \neq j} \prod_{k \neq i,j} ( t_\ell - t_k )
}{
	\prod_{k \neq j} ( t_j - t_k )
}
\\
& = &
x_\ell
\sum_{i \neq \ell}
\frac{ 1 }{ t_\ell - t_i }
+
\sum_{j \neq \ell}
x_j
\frac{
	\prod_{k \neq \ell,j} ( t_\ell - t_k )
}{
	\prod_{k \neq j} ( t_j - t_k )
}
\\
& = &
x_\ell
\sum_{k \neq \ell} ( t_\ell - t_k )^{-1}
+
\sum_{j \neq \ell}
x_j
( t_j - t_\ell )^{-1}
\prod_{k \neq \ell ,j} ( t_\ell - t_k ) / ( t_j - t_k )
\end{array}
\] $$
We define the vector $latex \alpha \in \B{R}^{m+1}$$ by
$latex \[
\alpha_j = \left\{ \begin{array}{ll}
\sum_{k \neq m} ( t_m - t_k )^{-1}
	& {\rm if} \; j = m
\\
( t_j - t_m )^{-1}
\prod_{k \neq m,j} ( t_m - t_k ) / ( t_j - t_k )
	& {\rm otherwise}
\end{array} \right.
\] $$
It follows that
$latex \[
	p^\prime ( t_m ) = \alpha_0 x_0 + \cdots + \alpha_m x_m
\] $$
Gear's method determines $latex x_m$$ by solving the following
nonlinear equation
$latex \[
	f( t_m , x_m ) = \alpha_0 x_0 + \cdots + \alpha_m x_m
\] $$
Newton's method for solving this equation determines iterates,
which we denote by $latex x_m^k$$, by solving the following affine
approximation of the equation above
$latex \[
\begin{array}{rcl}
f( t_m , x_m^{k-1} ) + \partial_x f( t_m , x_m^{k-1} ) ( x_m^k - x_m^{k-1} )
& = &
\alpha_0 x_0^k + \alpha_1 x_1 + \cdots + \alpha_m x_m
\\
\left[ \alpha_m I - \partial_x f( t_m , x_m^{k-1} ) \right]  x_m
& = &
\left[
f( t_m , x_m^{k-1} ) - \partial_x f( t_m , x_m^{k-1} ) x_m^{k-1}
- \alpha_0 x_0 - \cdots - \alpha_{m-1} x_{m-1}
\right]
\end{array}
\] $$
In order to initialize Newton's method; i.e. choose $latex x_m^0$$
we define the vector $latex \beta \in \B{R}^{m+1}$$ by
$latex \[
\beta_j = \left\{ \begin{array}{ll}
\sum_{k \neq m-1} ( t_{m-1} - t_k )^{-1}
	& {\rm if} \; j = m-1
\\
( t_j - t_{m-1} )^{-1}
\prod_{k \neq m-1,j} ( t_{m-1} - t_k ) / ( t_j - t_k )
	& {\rm otherwise}
\end{array} \right.
\] $$
It follows that
$latex \[
	p^\prime ( t_{m-1} ) = \beta_0 x_0 + \cdots + \beta_m x_m
\] $$
We solve the following approximation of the equation above to determine
$latex x_m^0$$:
$latex \[
	f( t_{m-1} , x_{m-1} ) =
	\beta_0 x_0 + \cdots + \beta_{m-1} x_{m-1} + \beta_m x_m^0
\] $$


$head Gear's Method$$
C. W. Gear,
``Simultaneous Numerical Solution of Differential-Algebraic Equations,''
IEEE Transactions on Circuit Theory,
vol. 18, no. 1, pp. 89-95, Jan. 1971.


$end
--------------------------------------------------------------------------
*/

# include <cstddef>
# include <cppad/core/cppad_assert.hpp>
# include <cppad/utility/check_simple_vector.hpp>
# include <cppad/utility/check_numeric_type.hpp>
# include <cppad/utility/vector.hpp>
# include <cppad/utility/lu_factor.hpp>
# include <cppad/utility/lu_invert.hpp>

namespace CppAD { // BEGIN CppAD namespace

template <typename Vector, typename Fun>
void OdeGear(
	Fun          &F  ,
	size_t        m  ,
	size_t        n  ,
	const Vector &T  ,
	Vector       &X  ,
	Vector       &e  )
{
	// temporary indices
	size_t i, j, k;

	typedef typename Vector::value_type Scalar;

	// check numeric type specifications
	CheckNumericType<Scalar>();

	// check simple vector class specifications
	CheckSimpleVector<Scalar, Vector>();

	CPPAD_ASSERT_KNOWN(
		m >= 1,
		"OdeGear: m is less than one"
	);
	CPPAD_ASSERT_KNOWN(
		n > 0,
		"OdeGear: n is equal to zero"
	);
	CPPAD_ASSERT_KNOWN(
		size_t(T.size()) >= (m+1),
		"OdeGear: size of T is not greater than or equal (m+1)"
	);
	CPPAD_ASSERT_KNOWN(
		size_t(X.size()) >= (m+1) * n,
		"OdeGear: size of X is not greater than or equal (m+1) * n"
	);
	for(j = 0; j < m; j++) CPPAD_ASSERT_KNOWN(
		T[j] < T[j+1],
		"OdeGear: the array T is not monotone increasing"
	);

	// some constants
	Scalar zero(0);
	Scalar one(1);

	// vectors required by method
	Vector alpha(m + 1);
	Vector beta(m + 1);
	Vector f(n);
	Vector f_x(n * n);
	Vector x_m0(n);
	Vector x_m(n);
	Vector b(n);
	Vector A(n * n);

	// compute alpha[m]
	alpha[m] = zero;
	for(k = 0; k < m; k++)
		alpha[m] += one / (T[m] - T[k]);

	// compute beta[m-1]
	beta[m-1] = one / (T[m-1] - T[m]);
	for(k = 0; k < m-1; k++)
		beta[m-1] += one / (T[m-1] - T[k]);


	// compute other components of alpha
	for(j = 0; j < m; j++)
	{	// compute alpha[j]
		alpha[j] = one / (T[j] - T[m]);
		for(k = 0; k < m; k++)
		{	if( k != j )
			{	alpha[j] *= (T[m] - T[k]);
				alpha[j] /= (T[j] - T[k]);
			}
		}
	}

	// compute other components of beta
	for(j = 0; j <= m; j++)
	{	if( j != m-1 )
		{	// compute beta[j]
			beta[j] = one / (T[j] - T[m-1]);
			for(k = 0; k <= m; k++)
			{	if( k != j && k != m-1 )
				{	beta[j] *= (T[m-1] - T[k]);
					beta[j] /= (T[j] - T[k]);
				}
			}
		}
	}

	// evaluate f(T[m-1], x_{m-1} )
	for(i = 0; i < n; i++)
		x_m[i] = X[(m-1) * n + i];
	F.Ode(T[m-1], x_m, f);

	// solve for x_m^0
	for(i = 0; i < n; i++)
	{	x_m[i] =  f[i];
		for(j = 0; j < m; j++)
			x_m[i] -= beta[j] * X[j * n + i];
		x_m[i] /= beta[m];
	}
	x_m0 = x_m;

	// evaluate partial w.r.t x of f(T[m], x_m^0)
	F.Ode_dep(T[m], x_m, f_x);

	// compute the matrix A = ( alpha[m] * I - f_x )
	for(i = 0; i < n; i++)
	{	for(j = 0; j < n; j++)
			A[i * n + j]  = - f_x[i * n + j];
		A[i * n + i] += alpha[m];
	}

	// LU factor (and overwrite) the matrix A
	CppAD::vector<size_t> ip(n) , jp(n);
# ifndef NDEBUG
	int sign =
# endif
	LuFactor(ip, jp, A);
	CPPAD_ASSERT_KNOWN(
		sign != 0,
		"OdeGear: step size is to large"
	);

	// Iterations of Newton's method
	for(k = 0; k < 3; k++)
	{
		// only evaluate f( T[m] , x_m ) keep f_x during iteration
		F.Ode(T[m], x_m, f);

		// b = f + f_x x_m - alpha[0] x_0 - ... - alpha[m-1] x_{m-1}
		for(i = 0; i < n; i++)
		{	b[i]         = f[i];
			for(j = 0; j < n; j++)
				b[i]         -= f_x[i * n + j] * x_m[j];
			for(j = 0; j < m; j++)
				b[i] -= alpha[j] * X[ j * n + i ];
		}
		LuInvert(ip, jp, A, b);
		x_m = b;
	}

	// return estimate for x( t[k] ) and the estimated error bound
	for(i = 0; i < n; i++)
	{	X[m * n + i] = x_m[i];
		e[i]         = x_m[i] - x_m0[i];
		if( e[i] < zero )
			e[i] = - e[i];
	}
}

} // End CppAD namespace

# endif
