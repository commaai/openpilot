# ifndef CPPAD_UTILITY_ROSEN_34_HPP
# define CPPAD_UTILITY_ROSEN_34_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin Rosen34$$
$spell
	cppad.hpp
	bool
	xf
	templated
	const
	Rosenbrock
	CppAD
	xi
	ti
	tf
	Karp
	Rosen
	Shampine
	ind
	dep
$$


$section A 3rd and 4th Order Rosenbrock ODE Solver$$
$mindex Rosen34 solve stiff differential equation$$

$head Syntax$$
$codei%# include <cppad/utility/rosen_34.hpp>
%$$
$icode%xf% = Rosen34(%F%, %M%, %ti%, %tf%, %xi%)
%$$
$icode%xf% = Rosen34(%F%, %M%, %ti%, %tf%, %xi%, %e%)
%$$


$head Description$$
This is an embedded 3rd and 4th order Rosenbrock ODE solver
(see Section 16.6 of $cref/Numerical Recipes/Bib/Numerical Recipes/$$
for a description of Rosenbrock ODE solvers).
In particular, we use the formulas taken from page 100 of
$cref/Shampine, L.F./Bib/Shampine, L.F./$$
(except that the fraction 98/108 has been correction to be 97/108).
$pre

$$
We use $latex n$$ for the size of the vector $icode xi$$.
Let $latex \B{R}$$ denote the real numbers
and let $latex F : \B{R} \times \B{R}^n \rightarrow \B{R}^n$$ be a smooth function.
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
If your set of  ordinary differential equations are not stiff
an explicit method may be better (perhaps $cref Runge45$$.)

$head Include$$
The file $code cppad/rosen_34.hpp$$ is included by $code cppad/cppad.hpp$$
but it can also be included separately with out the rest of
the $code CppAD$$ routines.

$head xf$$
The return value $icode xf$$ has the prototype
$codei%
	%Vector% %xf%
%$$
and the size of $icode xf$$ is equal to $icode n$$
(see description of $cref/Vector/Rosen34/Vector/$$ below).
$latex \[
	X(tf) = xf + O( h^5 )
\] $$
where $latex h = (tf - ti) / M$$ is the step size.
If $icode xf$$ contains not a number $cref nan$$,
see the discussion of $cref/f/Rosen34/Fun/Nan/$$.

$head Fun$$
The class $icode Fun$$
and the object $icode F$$ satisfy the prototype
$codei%
	%Fun% &%F%
%$$
This must support the following set of calls
$codei%
	%F%.Ode(%t%, %x%, %f%)
	%F%.Ode_ind(%t%, %x%, %f_t%)
	%F%.Ode_dep(%t%, %x%, %f_x%)
%$$

$subhead t$$
In all three cases,
the argument $icode t$$ has prototype
$codei%
	const %Scalar% &%t%
%$$
(see description of $cref/Scalar/Rosen34/Scalar/$$ below).

$subhead x$$
In all three cases,
the argument $icode x$$ has prototype
$codei%
	const %Vector% &%x%
%$$
and has size $icode n$$
(see description of $cref/Vector/Rosen34/Vector/$$ below).

$subhead f$$
The argument $icode f$$ to $icode%F%.Ode%$$ has prototype
$codei%
	%Vector% &%f%
%$$
On input and output, $icode f$$ is a vector of size $icode n$$
and the input values of the elements of $icode f$$ do not matter.
On output,
$icode f$$ is set equal to $latex F(t, x)$$
(see $icode F(t, x)$$ in $cref/Description/Rosen34/Description/$$).

$subhead f_t$$
The argument $icode f_t$$ to $icode%F%.Ode_ind%$$ has prototype
$codei%
	%Vector% &%f_t%
%$$
On input and output, $icode f_t$$ is a vector of size $icode n$$
and the input values of the elements of $icode f_t$$ do not matter.
On output, the $th i$$ element of
$icode f_t$$ is set equal to $latex \partial_t F_i (t, x)$$
(see $icode F(t, x)$$ in $cref/Description/Rosen34/Description/$$).

$subhead f_x$$
The argument $icode f_x$$ to $icode%F%.Ode_dep%$$ has prototype
$codei%
	%Vector% &%f_x%
%$$
On input and output, $icode f_x$$ is a vector of size $icode%n%*%n%$$
and the input values of the elements of $icode f_x$$ do not matter.
On output, the [$icode%i%*%n%+%j%$$] element of
$icode f_x$$ is set equal to $latex \partial_{x(j)} F_i (t, x)$$
(see $icode F(t, x)$$ in $cref/Description/Rosen34/Description/$$).

$subhead Nan$$
If any of the elements of $icode f$$, $icode f_t$$, or $icode f_x$$
have the value not a number $code nan$$,
the routine $code Rosen34$$ returns with all the
elements of $icode xf$$ and $icode e$$ equal to $code nan$$.

$subhead Warning$$
The arguments $icode f$$, $icode f_t$$, and $icode f_x$$
must have a call by reference in their prototypes; i.e.,
do not forget the $code &$$ in the prototype for
$icode f$$, $icode f_t$$ and $icode f_x$$.

$subhead Optimization$$
Every call of the form
$codei%
	%F%.Ode_ind(%t%, %x%, %f_t%)
%$$
is directly followed by a call of the form
$codei%
	%F%.Ode_dep(%t%, %x%, %f_x%)
%$$
where the arguments $icode t$$ and $icode x$$ have not changed between calls.
In many cases it is faster to compute the values of $icode f_t$$
and $icode f_x$$ together and then pass them back one at a time.

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
(see description of $cref/Scalar/Rosen34/Scalar/$$ below).
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
	e = O( h^4 )
\] $$
where $latex h = (tf - ti) / M$$ is the step size.

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

$head Parallel Mode$$
For each set of types
$cref/Scalar/Rosen34/Scalar/$$,
$cref/Vector/Rosen34/Vector/$$, and
$cref/Fun/Rosen34/Fun/$$,
the first call to $code Rosen34$$
must not be $cref/parallel/ta_in_parallel/$$ execution mode.

$head Example$$
$children%
	example/general/rosen_34.cpp
%$$
The file
$cref rosen_34.cpp$$
contains an example and test a test of using this routine.
It returns true if it succeeds and false otherwise.

$head Source Code$$
The source code for this routine is in the file
$code cppad/rosen_34.hpp$$.

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

// needed before one can use CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL
# include <cppad/utility/thread_alloc.hpp>

namespace CppAD { // BEGIN CppAD namespace

template <typename Scalar, typename Vector, typename Fun>
Vector Rosen34(
	Fun           &F ,
	size_t         M ,
	const Scalar &ti ,
	const Scalar &tf ,
	const Vector &xi )
{	Vector e( xi.size() );
	return Rosen34(F, M, ti, tf, xi, e);
}

template <typename Scalar, typename Vector, typename Fun>
Vector Rosen34(
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

	// Parameters for Shampine's Rosenbrock method
	// are static to avoid recalculation on each call and
	// do not use Vector to avoid possible memory leak
	static Scalar a[3] = {
		Scalar(0),
		Scalar(1),
		Scalar(3)   / Scalar(5)
	};
	static Scalar b[2 * 2] = {
		Scalar(1),
		Scalar(0),
		Scalar(24)  / Scalar(25),
		Scalar(3)   / Scalar(25)
	};
	static Scalar ct[4] = {
		Scalar(1)   / Scalar(2),
		- Scalar(3) / Scalar(2),
		Scalar(121) / Scalar(50),
		Scalar(29)  / Scalar(250)
	};
	static Scalar cg[3 * 3] = {
		- Scalar(4),
		Scalar(0),
		Scalar(0),
		Scalar(186) / Scalar(25),
		Scalar(6)   / Scalar(5),
		Scalar(0),
		- Scalar(56) / Scalar(125),
		- Scalar(27) / Scalar(125),
		- Scalar(1)  / Scalar(5)
	};
	static Scalar d3[3] = {
		Scalar(97) / Scalar(108),
		Scalar(11) / Scalar(72),
		Scalar(25) / Scalar(216)
	};
	static Scalar d4[4] = {
		Scalar(19)  / Scalar(18),
		Scalar(1)   / Scalar(4),
		Scalar(25)  / Scalar(216),
		Scalar(125) / Scalar(216)
	};
	CPPAD_ASSERT_KNOWN(
		M >= 1,
		"Error in Rosen34: the number of steps is less than one"
	);
	CPPAD_ASSERT_KNOWN(
		e.size() == xi.size(),
		"Error in Rosen34: size of e not equal to size of xi"
	);
	size_t i, j, k, l, m;             // indices

	size_t  n    = xi.size();         // number of components in X(t)
	Scalar  ns   = Scalar(double(M)); // number of steps as Scalar object
	Scalar  h    = (tf - ti) / ns;    // step size
	Scalar  zero = Scalar(0);         // some constants
	Scalar  one  = Scalar(1);
	Scalar  two  = Scalar(2);

	// permutation vectors needed for LU factorization routine
	CppAD::vector<size_t> ip(n), jp(n);

	// vectors used to store values returned by F
	Vector E(n * n), Eg(n), f_t(n);
	Vector g(n * 3), x3(n), x4(n), xf(n), ftmp(n), xtmp(n), nan_vec(n);

	// initialize e = 0, nan_vec = nan
	for(i = 0; i < n; i++)
	{	e[i]       = zero;
		nan_vec[i] = nan(zero);
	}

	xf = xi;           // initialize solution
	for(m = 0; m < M; m++)
	{	// time at beginning of this interval
		Scalar t = ti * (Scalar(int(M - m)) / ns)
		         + tf * (Scalar(int(m)) / ns);

		// value of x at beginning of this interval
		x3 = x4 = xf;

		// evaluate partial derivatives at beginning of this interval
		F.Ode_ind(t, xf, f_t);
		F.Ode_dep(t, xf, E);    // E = f_x
		if( hasnan(f_t) || hasnan(E) )
		{	e = nan_vec;
			return nan_vec;
		}

		// E = I - f_x * h / 2
		for(i = 0; i < n; i++)
		{	for(j = 0; j < n; j++)
				E[i * n + j] = - E[i * n + j] * h / two;
			E[i * n + i] += one;
		}

		// LU factor the matrix E
# ifndef NDEBUG
		int sign = LuFactor(ip, jp, E);
# else
		LuFactor(ip, jp, E);
# endif
		CPPAD_ASSERT_KNOWN(
			sign != 0,
			"Error in Rosen34: I - f_x * h / 2 not invertible"
		);

		// loop over integration steps
		for(k = 0; k < 3; k++)
		{	// set location for next function evaluation
			xtmp = xf;
			for(l = 0; l < k; l++)
			{	// loop over previous function evaluations
				Scalar bkl = b[(k-1)*2 + l];
				for(i = 0; i < n; i++)
				{	// loop over elements of x
					xtmp[i] += bkl * g[i*3 + l] * h;
				}
			}
			// ftmp = F(t + a[k] * h, xtmp)
			F.Ode(t + a[k] * h, xtmp, ftmp);
			if( hasnan(ftmp) )
			{	e = nan_vec;
				return nan_vec;
			}

			// Form Eg for this integration step
			for(i = 0; i < n; i++)
				Eg[i] = ftmp[i] + ct[k] * f_t[i] * h;
			for(l = 0; l < k; l++)
			{	for(i = 0; i < n; i++)
					Eg[i] += cg[(k-1)*3 + l] * g[i*3 + l];
			}

			// Solve the equation E * g = Eg
			LuInvert(ip, jp, E, Eg);

			// save solution and advance x3, x4
			for(i = 0; i < n; i++)
			{	g[i*3 + k]  = Eg[i];
				x3[i]      += h * d3[k] * Eg[i];
				x4[i]      += h * d4[k] * Eg[i];
			}
		}
		// Form Eg for last update to x4 only
		for(i = 0; i < n; i++)
			Eg[i] = ftmp[i] + ct[3] * f_t[i] * h;
		for(l = 0; l < 3; l++)
		{	for(i = 0; i < n; i++)
				Eg[i] += cg[2*3 + l] * g[i*3 + l];
		}

		// Solve the equation E * g = Eg
		LuInvert(ip, jp, E, Eg);

		// advance x4 and accumulate error bound
		for(i = 0; i < n; i++)
		{	x4[i] += h * d4[3] * Eg[i];

			// cant use abs because cppad.hpp may not be included
			Scalar diff = x4[i] - x3[i];
			if( diff < zero )
				e[i] -= diff;
			else	e[i] += diff;
		}

		// advance xf for this step using x4
		xf = x4;
	}
	return xf;
}

} // End CppAD namespace

# endif
