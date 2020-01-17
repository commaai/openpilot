# ifndef CPPAD_UTILITY_ODE_ERR_CONTROL_HPP
# define CPPAD_UTILITY_ODE_ERR_CONTROL_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin OdeErrControl$$
$spell
	cppad.hpp
	nstep
	maxabs
	exp
	scur
	CppAD
	xf
	tf
	xi
	smin
	smax
	eabs
	erel
	ef
	ta
	tb
	xa
	xb
	const
	eb
$$



$section An Error Controller for ODE Solvers$$
$mindex OdeErrControl differential equation$$

$head Syntax$$
$codei%# include <cppad/utility/ode_err_control.hpp>
%$$
$icode%xf% = OdeErrControl(%method%, %ti%, %tf%, %xi%,
	%smin%, %smax%, %scur%, %eabs%, %erel%, %ef% , %maxabs%, %nstep% )%$$


$head Description$$
Let $latex \B{R}$$ denote the real numbers
and let $latex F : \B{R} \times \B{R}^n \rightarrow \B{R}^n$$ be a smooth function.
We define $latex X : [ti , tf] \rightarrow \B{R}^n$$ by
the following initial value problem:
$latex \[
\begin{array}{rcl}
	X(ti)  & = & xi    \\
	X'(t)  & = & F[t , X(t)]
\end{array}
\] $$
The routine $code OdeErrControl$$ can be used to adjust the step size
used an arbitrary integration methods in order to be as fast as possible
and still with in a requested error bound.

$head Include$$
The file $code cppad/ode_err_control.hpp$$ is included by
$code cppad/cppad.hpp$$
but it can also be included separately with out the rest of
the $code CppAD$$ routines.

$head Notation$$
The template parameter types $cref/Scalar/OdeErrControl/Scalar/$$ and
$cref/Vector/OdeErrControl/Vector/$$ are documented below.

$head xf$$
The return value $icode xf$$ has the prototype
$codei%
	%Vector% %xf%
%$$
(see description of $cref/Vector/OdeErrControl/Vector/$$ below).
and the size of $icode xf$$ is equal to $icode n$$.
If $icode xf$$ contains not a number $cref nan$$,
see the discussion of $cref/step/OdeErrControl/Method/Nan/$$.

$head Method$$
The class $icode Method$$
and the object $icode method$$ satisfy the following syntax
$codei%
	%Method% &%method%
%$$
The object $icode method$$ must support $code step$$ and
$code order$$ member functions defined below:

$subhead step$$
The syntax
$codei%
	%method%.step(%ta%, %tb%, %xa%, %xb%, %eb%)
%$$
executes one step of the integration method.
$codei%

%ta%
%$$
The argument $icode ta$$ has prototype
$codei%
	const %Scalar% &%ta%
%$$
It specifies the initial time for this step in the
ODE integration.
(see description of $cref/Scalar/OdeErrControl/Scalar/$$ below).
$codei%

%tb%
%$$
The argument $icode tb$$ has prototype
$codei%
	const %Scalar% &%tb%
%$$
It specifies the final time for this step in the
ODE integration.
$codei%

%xa%
%$$
The argument $icode xa$$ has prototype
$codei%
	const %Vector% &%xa%
%$$
and size $icode n$$.
It specifies the value of $latex X(ta)$$.
(see description of $cref/Vector/OdeErrControl/Vector/$$ below).
$codei%

%xb%
%$$
The argument value $icode xb$$ has prototype
$codei%
	%Vector% &%xb%
%$$
and size $icode n$$.
The input value of its elements does not matter.
On output,
it contains the approximation for $latex X(tb)$$ that the method obtains.
$codei%

%eb%
%$$
The argument value $icode eb$$ has prototype
$codei%
	%Vector% &%eb%
%$$
and size $icode n$$.
The input value of its elements does not matter.
On output,
it contains an estimate for the error in the approximation $icode xb$$.
It is assumed (locally) that the error bound in this approximation
nearly equal to $latex K (tb - ta)^m$$
where $icode K$$ is a fixed constant and $icode m$$
is the corresponding argument to $code CodeControl$$.

$subhead Nan$$
If any element of the vector $icode eb$$ or $icode xb$$ are
not a number $code nan$$,
the current step is considered to large.
If this happens with the current step size equal to $icode smin$$,
$code OdeErrControl$$ returns with $icode xf$$ and $icode ef$$ as vectors
of $code nan$$.

$subhead order$$
If $icode m$$ is $code size_t$$,
the object $icode method$$ must also support the following syntax
$codei%
	%m% = %method%.order()
%$$
The return value $icode m$$ is the order of the error estimate;
i.e., there is a constant K such that if $latex ti \leq ta \leq tb \leq tf$$,
$latex \[
	| eb(tb) | \leq K | tb - ta |^m
\] $$
where $icode ta$$, $icode tb$$, and $icode eb$$ are as in
$icode%method%.step(%ta%, %tb%, %xa%, %xb%, %eb%)%$$


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
The step size during a call to $icode method$$ is defined as
the corresponding value of $latex tb - ta$$.
If $latex tf - ti \leq smin$$,
the integration will be done in one step of size $icode tf - ti$$.
Otherwise,
the minimum value of $icode tb - ta$$ will be $latex smin$$
except for the last two calls to $icode method$$ where it may be
as small as $latex smin / 2$$.

$head smax$$
The argument $icode smax$$ has prototype
$codei%
	const %Scalar% &%smax%
%$$
It specifies the maximum step size to use during the integration;
i.e., the maximum value for $latex tb - ta$$ in a call to $icode method$$.
The value of $icode smax$$ must be greater than or equal $icode smin$$.

$head scur$$
The argument $icode scur$$ has prototype
$codei%
	%Scalar% &%scur%
%$$
The value of $icode scur$$ is the suggested next step size,
based on error criteria, to try in the next call to $icode method$$.
On input it corresponds to the first call to $icode method$$,
in this call to $code OdeErrControl$$ (where $latex ta = ti$$).
On output it corresponds to the next call to $icode method$$,
in a subsequent call to $code OdeErrControl$$ (where $icode ta = tf$$).

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
$cref/error criteria discussion/OdeErrControl/Error Criteria Discussion/$$
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
$cref/error criteria discussion/OdeErrControl/Error Criteria Discussion/$$
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
If on output $icode ef$$ contains not a number $code nan$$,
see the discussion of $cref/step/OdeErrControl/Method/Nan/$$.

$head maxabs$$
The argument $icode maxabs$$ is optional in the call to $code OdeErrControl$$.
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
The argument $icode nstep$$ is optional in the call to $code OdeErrControl$$.
If it is present, it has the prototype
$codei%
	%size_t% &%nstep%
%$$
Its input value does not matter and its output value
is the number of calls to $icode%method%.step%$$
used by $code OdeErrControl$$.

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
the error criteria above will force $code OdeErrControl$$
to use step sizes equal to
$cref/smin/OdeErrControl/smin/$$
for steps ending near $latex tb$$.
In this case, the error relative to $icode maxabs$$ can be judged after
$code OdeErrControl$$ returns.
If $icode ef$$ is to large relative to $icode maxabs$$,
$code OdeErrControl$$ can be called again
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
	example/utility/ode_err_control.cpp%
	example/utility/ode_err_maxabs.cpp
%$$
The files
$cref ode_err_control.cpp$$
and
$cref ode_err_maxabs.cpp$$
contain examples and tests of using this routine.
They return true if they succeed and false otherwise.

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
$code cppad/ode_err_control.hpp$$.

$end
--------------------------------------------------------------------------
*/

// link exp and log for float and double
# include <cppad/base_require.hpp>

# include <cppad/core/cppad_assert.hpp>
# include <cppad/utility/check_simple_vector.hpp>
# include <cppad/utility/nan.hpp>

namespace CppAD { // Begin CppAD namespace

template <typename Scalar, typename Vector, typename Method>
Vector OdeErrControl(
	Method          &method,
	const Scalar    &ti    ,
	const Scalar    &tf    ,
	const Vector    &xi    ,
	const Scalar    &smin  ,
	const Scalar    &smax  ,
	Scalar          &scur  ,
	const Vector    &eabs  ,
	const Scalar    &erel  ,
	Vector          &ef    ,
	Vector          &maxabs,
	size_t          &nstep )
{
	// check simple vector class specifications
	CheckSimpleVector<Scalar, Vector>();

	size_t n = size_t(xi.size());

	CPPAD_ASSERT_KNOWN(
		smin <= smax,
		"Error in OdeErrControl: smin > smax"
	);
	CPPAD_ASSERT_KNOWN(
		size_t(eabs.size()) == n,
		"Error in OdeErrControl: size of eabs is not equal to n"
	);
	CPPAD_ASSERT_KNOWN(
		size_t(maxabs.size()) == n,
		"Error in OdeErrControl: size of maxabs is not equal to n"
	);
	size_t m = method.order();
	CPPAD_ASSERT_KNOWN(
		m > 1,
		"Error in OdeErrControl: m is less than or equal one"
	);

	bool    ok;
	bool    minimum_step;
	size_t  i;
	Vector xa(n), xb(n), eb(n), nan_vec(n);

	// initialization
	Scalar zero(0.0);
	Scalar one(1.0);
	Scalar two(2.0);
	Scalar three(3.0);
	Scalar m1(double(m-1));
	Scalar ta = ti;
	for(i = 0; i < n; i++)
	{	nan_vec[i] = nan(zero);
		ef[i]      = zero;
		xa[i]      = xi[i];
		if( zero <= xi[i] )
			maxabs[i] = xi[i];
		else	maxabs[i] = - xi[i];

	}
	nstep = 0;

	Scalar tb, step, lambda, axbi, a, r, root;
	while( ! (ta == tf) )
	{	// start with value suggested by error criteria
		step = scur;

		// check maximum
		if( smax <= step )
			step = smax;

		// check minimum
		minimum_step = step <= smin;
		if( minimum_step )
			step = smin;

		// check if near the end
		if( tf <= ta + step * three / two )
			tb = tf;
		else	tb = ta + step;

		// try using this step size
		nstep++;
		method.step(ta, tb, xa, xb, eb);
		step = tb - ta;

		// check if this steps error estimate is ok
		ok = ! (hasnan(xb) || hasnan(eb));
		if( (! ok) && minimum_step )
		{	ef = nan_vec;
			return nan_vec;
		}

		// compute value of lambda for this step
		lambda = Scalar(10) * scur / step;
		for(i = 0; i < n; i++)
		{	if( zero <= xb[i] )
				axbi = xb[i];
			else	axbi = - xb[i];
			a    = eabs[i] + erel * axbi;
			if( ! (eb[i] == zero) )
			{	r = ( a / eb[i] ) * step / (tf - ti);
				root = exp( log(r) / m1 );
				if( root <= lambda )
					lambda = root;
			}
		}
		if( ok && ( one <= lambda || step <= smin * three / two) )
		{	// this step is within error limits or
			// close to the minimum size
			ta = tb;
			for(i = 0; i < n; i++)
			{	xa[i] = xb[i];
				ef[i] = ef[i] + eb[i];
				if( zero <= xb[i] )
					axbi = xb[i];
				else	axbi = - xb[i];
				if( axbi > maxabs[i] )
					maxabs[i] = axbi;
			}
		}
		if( ! ok )
		{	// decrease step an see if method will work this time
			scur = step / two;
		}
		else if( ! (ta == tf) )
		{	// step suggested by the error criteria is not used
			// on the last step because it may be very small.
			scur = lambda * step / two;
		}
	}
	return xa;
}

template <typename Scalar, typename Vector, typename Method>
Vector OdeErrControl(
	Method          &method,
	const Scalar    &ti    ,
	const Scalar    &tf    ,
	const Vector    &xi    ,
	const Scalar    &smin  ,
	const Scalar    &smax  ,
	Scalar          &scur  ,
	const Vector    &eabs  ,
	const Scalar    &erel  ,
	Vector          &ef    )
{	Vector maxabs(xi.size());
	size_t nstep;
	return OdeErrControl(
	method, ti, tf, xi, smin, smax, scur, eabs, erel, ef, maxabs, nstep
	);
}

template <typename Scalar, typename Vector, typename Method>
Vector OdeErrControl(
	Method          &method,
	const Scalar    &ti    ,
	const Scalar    &tf    ,
	const Vector    &xi    ,
	const Scalar    &smin  ,
	const Scalar    &smax  ,
	Scalar          &scur  ,
	const Vector    &eabs  ,
	const Scalar    &erel  ,
	Vector          &ef    ,
	Vector          &maxabs)
{	size_t nstep;
	return OdeErrControl(
	method, ti, tf, xi, smin, smax, scur, eabs, erel, ef, maxabs, nstep
	);
}

} // End CppAD namespace

# endif
