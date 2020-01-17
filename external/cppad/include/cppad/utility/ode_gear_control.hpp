# ifndef CPPAD_UTILITY_ODE_GEAR_CONTROL_HPP
# define CPPAD_UTILITY_ODE_GEAR_CONTROL_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin OdeGearControl$$
$spell
	cppad.hpp
	CppAD
	xf
	xi
	smin
	smax
	eabs
	ef
	maxabs
	nstep
	tf
	sini
	erel
	dep
	const
	tb
	ta
	exp
$$



$section An Error Controller for Gear's Ode Solvers$$
$mindex OdeGearControl Gear differential equation$$

$head Syntax$$
$codei%# include <cppad/utility/ode_gear_control.hpp>
%$$
$icode%xf% = OdeGearControl(%F%, %M%, %ti%, %tf%, %xi%,
	%smin%, %smax%, %sini%, %eabs%, %erel%, %ef% , %maxabs%, %nstep% )%$$


$head Purpose$$
Let $latex \B{R}$$ denote the real numbers
and let $latex f : \B{R} \times \B{R}^n \rightarrow \B{R}^n$$ be a smooth function.
We define $latex X : [ti , tf] \rightarrow \B{R}^n$$ by
the following initial value problem:
$latex \[
\begin{array}{rcl}
	X(ti)  & = & xi    \\
	X'(t)  & = & f[t , X(t)]
\end{array}
\] $$
The routine $cref OdeGear$$ is a stiff multi-step method that
can be used to approximate the solution to this equation.
The routine $code OdeGearControl$$ sets up this multi-step method
and controls the error during such an approximation.

$head Include$$
The file $code cppad/ode_gear_control.hpp$$
is included by $code cppad/cppad.hpp$$
but it can also be included separately with out the rest of
the $code CppAD$$ routines.

$head Notation$$
The template parameter types $cref/Scalar/OdeGearControl/Scalar/$$ and
$cref/Vector/OdeGearControl/Vector/$$ are documented below.

$head xf$$
The return value $icode xf$$ has the prototype
$codei%
	%Vector% %xf%
%$$
and the size of $icode xf$$ is equal to $icode n$$
(see description of $cref/Vector/OdeGear/Vector/$$ below).
It is the approximation for $latex X(tf)$$.

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
and has size $icode N$$
(see description of $cref/Vector/OdeGear/Vector/$$ below).

$subhead f$$
The argument $icode f$$ to $icode%F%.Ode%$$ has prototype
$codei%
	%Vector% &%f%
%$$
On input and output, $icode f$$ is a vector of size $icode N$$
and the input values of the elements of $icode f$$ do not matter.
On output,
$icode f$$ is set equal to $latex f(t, x)$$
(see $icode f(t, x)$$ in $cref/Purpose/OdeGear/Purpose/$$).

$subhead f_x$$
The argument $icode f_x$$ has prototype
$codei%
	%Vector% &%f_x%
%$$
On input and output, $icode f_x$$ is a vector of size $latex N * N$$
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

$head M$$
The argument $icode M$$ has prototype
$codei%
	size_t %M%
%$$
It specifies the order of the multi-step method; i.e.,
the order of the approximating polynomial
(after the initialization process).
The argument $icode M$$ must greater than or equal one.

$head ti$$
The argument $icode ti$$ has prototype
$codei%
	const %Scalar% &%ti%
%$$
It specifies the initial time for the integration of
the differential equation.

$head tf$$
The argument $icode tf$$ has prototype
$codei%
	const %Scalar% &%tf%
%$$
It specifies the final time for the integration of
the differential equation.

$head xi$$
The argument $icode xi$$ has prototype
$codei%
	const %Vector% &%xi%
%$$
and size $icode n$$.
It specifies value of $latex X(ti)$$.

$head smin$$
The argument $icode smin$$ has prototype
$codei%
	const %Scalar% &%smin%
%$$
The minimum value of $latex T[M] -  T[M-1]$$ in a call to $code OdeGear$$
will be $latex smin$$ except for the last two calls where it may be
as small as $latex smin / 2$$.
The value of $icode smin$$ must be less than or equal $icode smax$$.

$head smax$$
The argument $icode smax$$ has prototype
$codei%
	const %Scalar% &%smax%
%$$
It specifies the maximum step size to use during the integration;
i.e., the maximum value for $latex T[M] - T[M-1]$$
in a call to $code OdeGear$$.

$head sini$$
The argument $icode sini$$ has prototype
$codei%
	%Scalar% &%sini%
%$$
The value of $icode sini$$ is the minimum
step size to use during initialization of the multi-step method; i.e.,
for calls to $code OdeGear$$ where $latex m < M$$.
The value of $icode sini$$ must be less than or equal $icode smax$$
(and can also be less than $icode smin$$).

$head eabs$$
The argument $icode eabs$$ has prototype
$codei%
	const %Vector% &%eabs%
%$$
and size $icode n$$.
Each of the elements of $icode eabs$$ must be
greater than or equal zero.
It specifies a bound for the absolute
error in the return value $icode xf$$ as an approximation for $latex X(tf)$$.
(see the
$cref/error criteria discussion/OdeGearControl/Error Criteria Discussion/$$
below).

$head erel$$
The argument $icode erel$$ has prototype
$codei%
	const %Scalar% &%erel%
%$$
and is greater than or equal zero.
It specifies a bound for the relative
error in the return value $icode xf$$ as an approximation for $latex X(tf)$$
(see the
$cref/error criteria discussion/OdeGearControl/Error Criteria Discussion/$$
below).

$head ef$$
The argument value $icode ef$$ has prototype
$codei%
	%Vector% &%ef%
%$$
and size $icode n$$.
The input value of its elements does not matter.
On output,
it contains an estimated bound for the
absolute error in the approximation $icode xf$$; i.e.,
$latex \[
	ef_i > | X( tf )_i - xf_i |
\] $$

$head maxabs$$
The argument $icode maxabs$$ is optional in the call to $code OdeGearControl$$.
If it is present, it has the prototype
$codei%
	%Vector% &%maxabs%
%$$
and size $icode n$$.
The input value of its elements does not matter.
On output,
it contains an estimate for the
maximum absolute value of $latex X(t)$$; i.e.,
$latex \[
	maxabs[i] \approx \max \left\{
		| X( t )_i | \; : \;  t \in [ti, tf]
	\right\}
\] $$

$head nstep$$
The argument $icode nstep$$ has the prototype
$codei%
	%size_t% &%nstep%
%$$
Its input value does not matter and its output value
is the number of calls to $cref OdeGear$$
used by $code OdeGearControl$$.

$head Error Criteria Discussion$$
The relative error criteria $icode erel$$ and
absolute error criteria $icode eabs$$ are enforced during each step of the
integration of the ordinary differential equations.
In addition, they are inversely scaled by the step size so that
the total error bound is less than the sum of the error bounds.
To be specific, if $latex \tilde{X} (t)$$ is the approximate solution
at time $latex t$$,
$icode ta$$ is the initial step time,
and $icode tb$$ is the final step time,
$latex \[
\left| \tilde{X} (tb)_j  - X (tb)_j \right|
\leq
\frac{tf - ti}{tb - ta}
\left[ eabs[j] + erel \;  | \tilde{X} (tb)_j | \right]
\] $$
If $latex X(tb)_j$$ is near zero for some $latex tb \in [ti , tf]$$,
and one uses an absolute error criteria $latex eabs[j]$$ of zero,
the error criteria above will force $code OdeGearControl$$
to use step sizes equal to
$cref/smin/OdeGearControl/smin/$$
for steps ending near $latex tb$$.
In this case, the error relative to $icode maxabs$$ can be judged after
$code OdeGearControl$$ returns.
If $icode ef$$ is to large relative to $icode maxabs$$,
$code OdeGearControl$$ can be called again
with a smaller value of $icode smin$$.

$head Scalar$$
The type $icode Scalar$$ must satisfy the conditions
for a $cref NumericType$$ type.
The routine $cref CheckNumericType$$ will generate an error message
if this is not the case.
In addition, the following operations must be defined for
$icode Scalar$$ objects $icode a$$ and $icode b$$:

$table
$bold Operation$$ $cnext $bold Description$$  $rnext
$icode%a% <= %b%$$ $cnext
	returns true (false) if $icode a$$ is less than or equal
	(greater than) $icode b$$.
$rnext
$icode%a% == %b%$$ $cnext
	returns true (false) if $icode a$$ is equal to $icode b$$.
$rnext
$codei%log(%a%)%$$ $cnext
	returns a $icode Scalar$$ equal to the logarithm of $icode a$$
$rnext
$codei%exp(%a%)%$$ $cnext
	returns a $icode Scalar$$ equal to the exponential of $icode a$$
$tend


$head Vector$$
The type $icode Vector$$ must be a $cref SimpleVector$$ class with
$cref/elements of type Scalar/SimpleVector/Elements of Specified Type/$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$head Example$$
$children%
	example/utility/ode_gear_control.cpp
%$$
The file
$cref ode_gear_control.cpp$$
contains an example and test a test of using this routine.
It returns true if it succeeds and false otherwise.

$head Theory$$
Let $latex e(s)$$ be the error as a function of the
step size $latex s$$ and suppose that there is a constant
$latex K$$ such that $latex e(s) = K s^m$$.
Let $latex a$$ be our error bound.
Given the value of $latex e(s)$$, a step of size $latex \lambda s$$
would be ok provided that
$latex \[
\begin{array}{rcl}
	a  & \geq & e( \lambda s ) (tf - ti) / ( \lambda s ) \\
	a  & \geq & K \lambda^m s^m (tf - ti) / ( \lambda s ) \\
	a  & \geq & \lambda^{m-1} s^{m-1} (tf - ti) e(s) / s^m \\
	a  & \geq & \lambda^{m-1} (tf - ti) e(s) / s           \\
	\lambda^{m-1} & \leq & \frac{a}{e(s)} \frac{s}{tf - ti}
\end{array}
\] $$
Thus if the right hand side of the last inequality is greater
than or equal to one, the step of size $latex s$$ is ok.

$head Source Code$$
The source code for this routine is in the file
$code cppad/ode_gear_control.hpp$$.

$end
--------------------------------------------------------------------------
*/

// link exp and log for float and double
# include <cppad/base_require.hpp>

# include <cppad/utility/ode_gear.hpp>

namespace CppAD { // Begin CppAD namespace

template <class Scalar, class Vector, class Fun>
Vector OdeGearControl(
	Fun             &F     ,
	size_t           M     ,
	const Scalar    &ti    ,
	const Scalar    &tf    ,
	const Vector    &xi    ,
	const Scalar    &smin  ,
	const Scalar    &smax  ,
	Scalar          &sini  ,
	const Vector    &eabs  ,
	const Scalar    &erel  ,
	Vector          &ef    ,
	Vector          &maxabs,
	size_t          &nstep )
{
	// check simple vector class specifications
	CheckSimpleVector<Scalar, Vector>();

	// dimension of the state space
	size_t n = size_t(xi.size());

	CPPAD_ASSERT_KNOWN(
		M >= 1,
		"Error in OdeGearControl: M is less than one"
	);
	CPPAD_ASSERT_KNOWN(
		smin <= smax,
		"Error in OdeGearControl: smin is greater than smax"
	);
	CPPAD_ASSERT_KNOWN(
		sini <= smax,
		"Error in OdeGearControl: sini is greater than smax"
	);
	CPPAD_ASSERT_KNOWN(
		size_t(eabs.size()) == n,
		"Error in OdeGearControl: size of eabs is not equal to n"
	);
	CPPAD_ASSERT_KNOWN(
		size_t(maxabs.size()) == n,
		"Error in OdeGearControl: size of maxabs is not equal to n"
	);

	// some constants
	const Scalar zero(0);
	const Scalar one(1);
	const Scalar one_plus( Scalar(3) / Scalar(2) );
	const Scalar two(2);
	const Scalar ten(10);

	// temporary indices
	size_t i, k;

	// temporary Scalars
	Scalar step, sprevious, lambda, axi, a, root, r;

	// vectors of Scalars
	Vector T  (M + 1);
	Vector X( (M + 1) * n );
	Vector e(n);
	Vector xf(n);

	// initial integer values
	size_t m = 1;
	nstep    = 0;

	// initialize T
	T[0] = ti;

	// initialize X, ef, maxabs
	for(i = 0; i < n; i++)
	for(i = 0; i < n; i++)
	{	X[i] = xi[i];
		ef[i] = zero;
		X[i]  = xi[i];
		if( zero <= xi[i] )
			maxabs[i] = xi[i];
		else	maxabs[i] = - xi[i];

	}

	// initial step size
	step = smin;

	while( T[m-1] < tf )
	{	sprevious = step;

		// check maximum
		if( smax <= step )
			step = smax;

		// check minimum
		if( m < M )
		{	if( step <= sini )
				step = sini;
		}
		else	if( step <= smin )
				step = smin;

		// check if near the end
		if( tf <= T[m-1] + one_plus * step )
			T[m] = tf;
		else	T[m] = T[m-1] + step;

		// try using this step size
		nstep++;
		OdeGear(F, m, n, T, X, e);
		step = T[m] - T[m-1];

		// compute value of lambda for this step
		lambda = Scalar(10) *  sprevious / step;
		for(i = 0; i < n; i++)
		{	axi = X[m * n + i];
			if( axi <= zero )
				axi = - axi;
			a  = eabs[i] + erel * axi;
			if( e[i] > zero )
			{	if( m == 1 )
					root = (a / e[i]) / ten;
				else
				{	r = ( a / e[i] ) * step / (tf - ti);
					root = exp( log(r) / Scalar(m-1) );
				}
				if( root <= lambda )
					lambda = root;
			}
		}

		bool advance;
		if( m == M )
			advance = one <= lambda || step <= one_plus * smin;
		else	advance = one <= lambda || step <= one_plus * sini;


		if( advance )
		{	// accept the results of this time step
			CPPAD_ASSERT_UNKNOWN( m <= M );
			if( m == M )
			{	// shift for next step
				for(k = 0; k < m; k++)
				{	T[k] = T[k+1];
					for(i = 0; i < n; i++)
						X[k*n + i] = X[(k+1)*n + i];
				}
			}
			// update ef and maxabs
			for(i = 0; i < n; i++)
			{	ef[i] = ef[i] + e[i];
				axi = X[m * n + i];
				if( axi <= zero )
					axi = - axi;
				if( axi > maxabs[i] )
					maxabs[i] = axi;
			}
			if( m != M )
				m++;  // all we need do in this case
		}

		// new step suggested by error criteria
		step = std::min(lambda , ten) * step / two;
	}
	for(i = 0; i < n; i++)
		xf[i] = X[(m-1) * n + i];

	return xf;
}

} // End CppAD namespace

# endif
