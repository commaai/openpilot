# ifndef CPPAD_UTILITY_RUNGE_45_HPP
# define CPPAD_UTILITY_RUNGE_45_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin Runge45$$
$spell
	std
	fabs
	cppad.hpp
	bool
	xf
	templated
	const
	Runge-Kutta
	CppAD
	xi
	ti
	tf
	Karp
$$


$section An Embedded 4th and 5th Order Runge-Kutta ODE Solver$$
$mindex Runge45 Runge Kutta solve differential equation$$

$head Syntax$$
$codei%# include <cppad/utility/runge_45.hpp>
%$$
$icode%xf% = Runge45(%F%, %M%, %ti%, %tf%, %xi%)
%$$
$icode%xf% = Runge45(%F%, %M%, %ti%, %tf%, %xi%, %e%)
%$$


$head Purpose$$
This is an implementation of the
Cash-Karp embedded 4th and 5th order Runge-Kutta ODE solver
described in Section 16.2 of $cref/Numerical Recipes/Bib/Numerical Recipes/$$.
We use $latex n$$ for the size of the vector $icode xi$$.
Let $latex \B{R}$$ denote the real numbers
and let $latex F : \B{R} \times \B{R}^n \rightarrow \B{R}^n$$
be a smooth function.
The return value $icode xf$$ contains a 5th order
approximation for the value $latex X(tf)$$ where
$latex X : [ti , tf] \rightarrow \B{R}^n$$ is defined by
the following initial value problem:
$latex \[
\begin{array}{rcl}
	X(ti)  & = & xi    \\
	X'(t)  & = & F[t , X(t)]
\end{array}
\] $$
If your set of ordinary differential equations
are stiff, an implicit method may be better
(perhaps $cref Rosen34$$.)

$head Operation Sequence$$
The $cref/operation sequence/glossary/Operation/Sequence/$$ for $icode Runge$$
does not depend on any of its $icode Scalar$$ input values provided that
the operation sequence for
$codei%
	%F%.Ode(%t%, %x%, %f%)
%$$
does not on any of its $icode Scalar$$ inputs (see below).

$head Include$$
The file $code cppad/runge_45.hpp$$ is included by $code cppad/cppad.hpp$$
but it can also be included separately with out the rest of
the $code CppAD$$ routines.

$head xf$$
The return value $icode xf$$ has the prototype
$codei%
	%Vector% %xf%
%$$
and the size of $icode xf$$ is equal to $icode n$$
(see description of $cref/Vector/Runge45/Vector/$$ below).
$latex \[
	X(tf) = xf + O( h^6 )
\] $$
where $latex h = (tf - ti) / M$$ is the step size.
If $icode xf$$ contains not a number $cref nan$$,
see the discussion for $cref/f/Runge45/Fun/f/$$.

$head Fun$$
The class $icode Fun$$
and the object $icode F$$ satisfy the prototype
$codei%
	%Fun% &%F%
%$$
The object $icode F$$ (and the class $icode Fun$$)
must have a member function named $code Ode$$
that supports the syntax
$codei%
	%F%.Ode(%t%, %x%, %f%)
%$$

$subhead t$$
The argument $icode t$$ to $icode%F%.Ode%$$ has prototype
$codei%
	const %Scalar% &%t%
%$$
(see description of $cref/Scalar/Runge45/Scalar/$$ below).

$subhead x$$
The argument $icode x$$ to $icode%F%.Ode%$$ has prototype
$codei%
	const %Vector% &%x%
%$$
and has size $icode n$$
(see description of $cref/Vector/Runge45/Vector/$$ below).

$subhead f$$
The argument $icode f$$ to $icode%F%.Ode%$$ has prototype
$codei%
	%Vector% &%f%
%$$
On input and output, $icode f$$ is a vector of size $icode n$$
and the input values of the elements of $icode f$$ do not matter.
On output,
$icode f$$ is set equal to $latex F(t, x)$$ in the differential equation.
If any of the elements of $icode f$$ have the value not a number $code nan$$
the routine $code Runge45$$ returns with all the
elements of $icode xf$$ and $icode e$$ equal to $code nan$$.

$subhead Warning$$
The argument $icode f$$ to $icode%F%.Ode%$$
must have a call by reference in its prototype; i.e.,
do not forget the $code &$$ in the prototype for $icode f$$.

$head M$$
The argument $icode M$$ has prototype
$codei%
	size_t %M%
%$$
It specifies the number of steps
to use when solving the differential equation.
This must be greater than or equal one.
The step size is given by $latex h = (tf - ti) / M$$, thus
the larger $icode M$$, the more accurate the
return value $icode xf$$ is as an approximation
for $latex X(tf)$$.

$head ti$$
The argument $icode ti$$ has prototype
$codei%
	const %Scalar% &%ti%
%$$
(see description of $cref/Scalar/Runge45/Scalar/$$ below).
It specifies the initial time for $icode t$$ in the
differential equation; i.e.,
the time corresponding to the value $icode xi$$.

$head tf$$
The argument $icode tf$$ has prototype
$codei%
	const %Scalar% &%tf%
%$$
It specifies the final time for $icode t$$ in the
differential equation; i.e.,
the time corresponding to the value $icode xf$$.

$head xi$$
The argument $icode xi$$ has the prototype
$codei%
	const %Vector% &%xi%
%$$
and the size of $icode xi$$ is equal to $icode n$$.
It specifies the value of $latex X(ti)$$

$head e$$
The argument $icode e$$ is optional and has the prototype
$codei%
	%Vector% &%e%
%$$
If $icode e$$ is present,
the size of $icode e$$ must be equal to $icode n$$.
The input value of the elements of $icode e$$ does not matter.
On output
it contains an element by element
estimated bound for the absolute value of the error in $icode xf$$
$latex \[
	e = O( h^5 )
\] $$
where $latex h = (tf - ti) / M$$ is the step size.
If on output, $icode e$$ contains not a number $code nan$$,
see the discussion for $cref/f/Runge45/Fun/f/$$.

$head Scalar$$
The type $icode Scalar$$ must satisfy the conditions
for a $cref NumericType$$ type.
The routine $cref CheckNumericType$$ will generate an error message
if this is not the case.

$subhead fabs$$
In addition, the following function must be defined for
$icode Scalar$$ objects $icode a$$ and $icode b$$
$codei%
	%a% = fabs(%b%)
%$$
Note that this operation is only used for computing $icode e$$; hence
the operation sequence for $icode xf$$ can still be independent of
the arguments to $code Runge45$$ even if
$codei%
	fabs(%b%) = std::max(-%b%, %b%)
%$$.

$head Vector$$
The type $icode Vector$$ must be a $cref SimpleVector$$ class with
$cref/elements of type Scalar/SimpleVector/Elements of Specified Type/$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$head Parallel Mode$$
For each set of types
$cref/Scalar/Runge45/Scalar/$$,
$cref/Vector/Runge45/Vector/$$, and
$cref/Fun/Runge45/Fun/$$,
the first call to $code Runge45$$
must not be $cref/parallel/ta_in_parallel/$$ execution mode.


$head Example$$
$children%
	example/utility/runge45_1.cpp%
	example/general/runge45_2.cpp
%$$
The file
$cref runge45_1.cpp$$
contains a simple example and test of $code Runge45$$.
It returns true if it succeeds and false otherwise.
$pre

$$
The file
$cref runge45_2.cpp$$ contains an example using $code Runge45$$
in the context of algorithmic differentiation.
It also returns true if it succeeds and false otherwise.

$head Source Code$$
The source code for this routine is in the file
$code cppad/runge_45.hpp$$.

$end
--------------------------------------------------------------------------
*/
# include <cstddef>
# include <cppad/core/cppad_assert.hpp>
# include <cppad/utility/check_simple_vector.hpp>
# include <cppad/utility/check_numeric_type.hpp>
# include <cppad/utility/nan.hpp>

// needed before one can use CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL
# include <cppad/utility/thread_alloc.hpp>

namespace CppAD { // BEGIN CppAD namespace

template <typename Scalar, typename Vector, typename Fun>
Vector Runge45(
	Fun           &F ,
	size_t         M ,
	const Scalar &ti ,
	const Scalar &tf ,
	const Vector &xi )
{	Vector e( xi.size() );
	return Runge45(F, M, ti, tf, xi, e);
}

template <typename Scalar, typename Vector, typename Fun>
Vector Runge45(
	Fun           &F ,
	size_t         M ,
	const Scalar &ti ,
	const Scalar &tf ,
	const Vector &xi ,
	Vector       &e )
{
	CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL;

	// check numeric type specifications
	CheckNumericType<Scalar>();

	// check simple vector class specifications
	CheckSimpleVector<Scalar, Vector>();

	// Cash-Karp parameters for embedded Runge-Kutta method
	// are static to avoid recalculation on each call and
	// do not use Vector to avoid possible memory leak
	static Scalar a[6] = {
		Scalar(0),
		Scalar(1) / Scalar(5),
		Scalar(3) / Scalar(10),
		Scalar(3) / Scalar(5),
		Scalar(1),
		Scalar(7) / Scalar(8)
	};
	static Scalar b[5 * 5] = {
		Scalar(1) / Scalar(5),
		Scalar(0),
		Scalar(0),
		Scalar(0),
		Scalar(0),

		Scalar(3) / Scalar(40),
		Scalar(9) / Scalar(40),
		Scalar(0),
		Scalar(0),
		Scalar(0),

		Scalar(3) / Scalar(10),
		-Scalar(9) / Scalar(10),
		Scalar(6) / Scalar(5),
		Scalar(0),
		Scalar(0),

		-Scalar(11) / Scalar(54),
		Scalar(5) / Scalar(2),
		-Scalar(70) / Scalar(27),
		Scalar(35) / Scalar(27),
		Scalar(0),

		Scalar(1631) / Scalar(55296),
		Scalar(175) / Scalar(512),
		Scalar(575) / Scalar(13824),
		Scalar(44275) / Scalar(110592),
		Scalar(253) / Scalar(4096)
	};
	static Scalar c4[6] = {
		Scalar(2825) / Scalar(27648),
		Scalar(0),
		Scalar(18575) / Scalar(48384),
		Scalar(13525) / Scalar(55296),
		Scalar(277) / Scalar(14336),
		Scalar(1) / Scalar(4),
	};
	static Scalar c5[6] = {
		Scalar(37) / Scalar(378),
		Scalar(0),
		Scalar(250) / Scalar(621),
		Scalar(125) / Scalar(594),
		Scalar(0),
		Scalar(512) / Scalar(1771)
	};

	CPPAD_ASSERT_KNOWN(
		M >= 1,
		"Error in Runge45: the number of steps is less than one"
	);
	CPPAD_ASSERT_KNOWN(
		e.size() == xi.size(),
		"Error in Runge45: size of e not equal to size of xi"
	);
	size_t i, j, k, m;              // indices

	size_t  n = xi.size();           // number of components in X(t)
	Scalar  ns = Scalar(int(M));     // number of steps as Scalar object
	Scalar  h = (tf - ti) / ns;      // step size
	Scalar  zero_or_nan = Scalar(0); // zero (nan if Ode returns has a nan)
	for(i = 0; i < n; i++)           // initialize e = 0
		e[i] = zero_or_nan;

	// vectors used to store values returned by F
	Vector fh(6 * n), xtmp(n), ftmp(n), x4(n), x5(n), xf(n);

	xf = xi;           // initialize solution
	for(m = 0; m < M; m++)
	{	// time at beginning of this interval
		// (convert to int to avoid MS compiler warning)
		Scalar t = ti * (Scalar(int(M - m)) / ns)
		         + tf * (Scalar(int(m)) / ns);

		// loop over integration steps
		x4 = x5 = xf;   // start x4 and x5 at same point for each step
		for(j = 0; j < 6; j++)
		{	// loop over function evaluations for this step
			xtmp = xf;  // location for next function evaluation
			for(k = 0; k < j; k++)
			{	// loop over previous function evaluations
				Scalar bjk = b[ (j-1) * 5 + k ];
				for(i = 0; i < n; i++)
				{	// loop over elements of x
					xtmp[i] += bjk * fh[i * 6 + k];
				}
			}
			// ftmp = F(t + a[j] * h, xtmp)
			F.Ode(t + a[j] * h, xtmp, ftmp);

			// if ftmp has a nan, set zero_or_nan to nan
			for(i = 0; i < n; i++)
				zero_or_nan *= ftmp[i];

			for(i = 0; i < n; i++)
			{	// loop over elements of x
				Scalar fhi    = ftmp[i] * h;
				fh[i * 6 + j] = fhi;
				x4[i]        += c4[j] * fhi;
				x5[i]        += c5[j] * fhi;
				x5[i]        += zero_or_nan;
			}
		}
		// accumulate error bound
		for(i = 0; i < n; i++)
		{	// cant use abs because cppad.hpp may not be included
			Scalar diff = x5[i] - x4[i];
			e[i] += fabs(diff);
			e[i] += zero_or_nan;
		}

		// advance xf for this step using x5
		xf = x5;
	}
	return xf;
}

} // End CppAD namespace

# endif
