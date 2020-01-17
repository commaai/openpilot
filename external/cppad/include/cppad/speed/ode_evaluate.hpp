# ifndef CPPAD_SPEED_ODE_EVALUATE_HPP
# define CPPAD_SPEED_ODE_EVALUATE_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin ode_evaluate$$
$spell
	Runge
	fabs
	retaped
	Jacobian
	const
	Cpp
	cppad
	hpp
	fp
	namespace
	exp
$$

$section Evaluate a Function Defined in Terms of an ODE$$
$mindex ode_evaluate$$


$head Syntax$$
$codei%# include <cppad/speed/ode_evaluate.hpp>
%$$
$codei%ode_evaluate(%x%, %p%, %fp%)%$$

$head Purpose$$
This routine evaluates a function $latex f : \B{R}^n \rightarrow \B{R}^n$$
defined by
$latex \[
	f(x) = y(x, 1)
\] $$
where $latex y(x, t)$$ solves the ordinary differential equation
$latex \[
\begin{array}{rcl}
	y(x, 0)              & = & x
	\\
	\partial_t y (x, t ) & = & g[ y(x,t) , t ]
\end{array}
\] $$
where $latex g : \B{R}^n \times \B{R} \rightarrow \B{R}^n$$
is an unspecified function.

$head Inclusion$$
The template function $code ode_evaluate$$
is defined in the $code CppAD$$ namespace by including
the file $code cppad/speed/ode_evaluate.hpp$$
(relative to the CppAD distribution directory).

$head Float$$

$subhead Operation Sequence$$
The type $icode Float$$ must be a $cref NumericType$$.
The $icode Float$$
$cref/operation sequence/glossary/Operation/Sequence/$$
for this routine does not depend on the value of the argument $icode x$$,
hence it does not need to be retaped for each value of $latex x$$.

$subhead fabs$$
If $icode y$$ and $icode z$$ are $icode Float$$ objects, the syntax
$codei%
	%y% = fabs(%z%)
%$$
must be supported. Note that it does not matter if the operation
sequence for $code fabs$$ depends on $icode z$$ because the
corresponding results are not actually used by $code ode_evaluate$$;
see $code fabs$$ in $cref/Runge45/Runge45/Scalar/fabs/$$.

$head x$$
The argument $icode x$$ has prototype
$codei%
	const CppAD::vector<%Float%>& %x%
%$$
It contains he argument value for which the function,
or its derivative, is being evaluated.
The value $latex n$$ is determined by the size of the vector $icode x$$.

$head p$$
The argument $icode p$$ has prototype
$codei%
	size_t %p%
%$$

$subhead p == 0$$
In this case a numerical method is used to solve the ode
and obtain an accurate approximation for $latex y(x, 1)$$.
This numerical method has a fixed
that does not depend on $icode x$$.

$subhead p = 1$$
In this case an analytic solution for the partial derivative
$latex \partial_x y(x, 1)$$ is returned.

$head fp$$
The argument $icode fp$$ has prototype
$codei%
	CppAD::vector<%Float%>& %fp%
%$$
The input value of the elements of $icode fp$$ does not matter.

$subhead Function$$
If $icode p$$ is zero, $icode fp$$ has size equal to $latex n$$
and contains the value of $latex y(x, 1)$$.

$subhead Gradient$$
If $icode p$$ is one, $icode fp$$ has size equal to $icode n^2$$
and for $latex i = 0 , \ldots 1$$, $latex j = 0 , \ldots , n-1$$
$latex \[
	\D{y[i]}{x[j]} (x, 1) = fp [ i \cdot n + j ]
\] $$

$children%
	speed/example/ode_evaluate.cpp%
	omh/ode_evaluate.omh
%$$

$head Example$$
The file
$cref ode_evaluate.cpp$$
contains an example and test  of $code ode_evaluate.hpp$$.
It returns true if it succeeds and false otherwise.


$head Source Code$$
The file
$cref ode_evaluate.hpp$$
contains the source code for this template function.

$end
*/
// BEGIN C++
# include <cppad/utility/vector.hpp>
# include <cppad/utility/ode_err_control.hpp>
# include <cppad/utility/runge_45.hpp>

namespace CppAD {

	template <class Float>
	class ode_evaluate_fun {
	public:
		// Given that y_i (0) = x_i,
		// the following y_i (t) satisfy the ODE below:
		// y_0 (t) = x[0]
		// y_1 (t) = x[1] + x[0] * t
		// y_2 (t) = x[2] + x[1] * t + x[0] * t^2/2
		// y_3 (t) = x[3] + x[2] * t + x[1] * t^2/2 + x[0] * t^3 / 3!
		// ...
		void Ode(
			const Float&                    t,
			const CppAD::vector<Float>&     y,
			CppAD::vector<Float>&           f)
		{	size_t n  = y.size();
			f[0]      = 0.;
			for(size_t k = 1; k < n; k++)
				f[k] = y[k-1];
		}
	};
	//
	template <class Float>
	void ode_evaluate(
		const CppAD::vector<Float>& x  ,
		size_t                      p  ,
		CppAD::vector<Float>&       fp )
	{	using CppAD::vector;
		typedef vector<Float> VectorFloat;

		size_t n = x.size();
		CPPAD_ASSERT_KNOWN( p == 0 || p == 1,
			"ode_evaluate: p is not zero or one"
		);
		CPPAD_ASSERT_KNOWN(
			((p==0) & (fp.size()==n)) || ((p==1) & (fp.size()==n*n)),
			"ode_evaluate: the size of fp is not correct"
		);
		if( p == 0 )
		{	// function that defines the ode
			ode_evaluate_fun<Float> F;

			// number of Runge45 steps to use
			size_t M = 10;

			// initial and final time
			Float ti = 0.0;
			Float tf = 1.0;

			// initial value for y(x, t); i.e. y(x, 0)
			// (is a reference to x)
			const VectorFloat& yi = x;

			// final value for y(x, t); i.e., y(x, 1)
			// (is a reference to fp)
			VectorFloat& yf = fp;

			// Use fourth order Runge-Kutta to solve ODE
			yf = CppAD::Runge45(F, M, ti, tf, yi);

			return;
		}
		/* Compute derivaitve of y(x, 1) w.r.t x
		y_0 (x, t) = x[0]
		y_1 (x, t) = x[1] + x[0] * t
		y_2 (x, t) = x[2] + x[1] * t + x[0] * t^2/2
		y_3 (x, t) = x[3] + x[2] * t + x[1] * t^2/2 + x[0] * t^3 / 3!
		...
		*/
		size_t i, j, k;
		for(i = 0; i < n; i++)
		{	for(j = 0; j < n; j++)
				fp[ i * n + j ] = 0.0;
		}
		size_t factorial = 1;
		for(k = 0; k < n; k++)
		{	if( k > 1 )
				factorial *= k;
			for(i = k; i < n; i++)
			{	// partial w.r.t x[i-k] of x[i-k] * t^k / k!
				j = i - k;
				fp[ i * n + j ] += 1.0 / Float(factorial);
			}
		}
	}
}
// END C++

# endif
