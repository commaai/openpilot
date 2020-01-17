# ifndef CPPAD_UTILITY_LU_SOLVE_HPP
# define CPPAD_UTILITY_LU_SOLVE_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin LuSolve$$
$escape #$$
$spell
	cppad.hpp
	det
	exp
	Leq
	typename
	bool
	const
	namespace
	std
	Geq
	Lu
	CppAD
	signdet
	logdet
$$


$section Compute Determinant and Solve Linear Equations$$
$mindex LuSolve Lu$$

$pre
$$

$head Syntax$$ $codei%# include <cppad/utility/lu_solve.hpp>
%$$
$icode%signdet% = LuSolve(%n%, %m%, %A%, %B%, %X%, %logdet%)%$$


$head Description$$
Use an LU factorization of the matrix $icode A$$ to
compute its determinant
and solve for $icode X$$ in the linear of equation
$latex \[
	A * X = B
\] $$
where $icode A$$ is an
$icode n$$ by $icode n$$ matrix,
$icode X$$ is an
$icode n$$ by $icode m$$ matrix, and
$icode B$$ is an $latex n x m$$ matrix.

$head Include$$
The file $code cppad/lu_solve.hpp$$ is included by $code cppad/cppad.hpp$$
but it can also be included separately with out the rest of
the $code CppAD$$ routines.

$head Factor and Invert$$
This routine is an easy to user interface to
$cref LuFactor$$ and $cref LuInvert$$ for computing determinants and
solutions of linear equations.
These separate routines should be used if
one right hand side $icode B$$
depends on the solution corresponding to another
right hand side (with the same value of $icode A$$).
In this case only one call to $code LuFactor$$ is required
but there will be multiple calls to $code LuInvert$$.


$head Matrix Storage$$
All matrices are stored in row major order.
To be specific, if $latex Y$$ is a vector
that contains a $latex p$$ by $latex q$$ matrix,
the size of $latex Y$$ must be equal to $latex  p * q $$ and for
$latex i = 0 , \ldots , p-1$$,
$latex j = 0 , \ldots , q-1$$,
$latex \[
	Y_{i,j} = Y[ i * q + j ]
\] $$

$head signdet$$
The return value $icode signdet$$ is a $code int$$ value
that specifies the sign factor for the determinant of $icode A$$.
This determinant of $icode A$$ is zero if and only if $icode signdet$$
is zero.

$head n$$
The argument $icode n$$ has type $code size_t$$
and specifies the number of rows in the matrices
$icode A$$,
$icode X$$,
and $icode B$$.
The number of columns in $icode A$$ is also equal to $icode n$$.

$head m$$
The argument $icode m$$ has type $code size_t$$
and specifies the number of columns in the matrices
$icode X$$
and $icode B$$.
If $icode m$$ is zero,
only the determinant of $icode A$$ is computed and
the matrices $icode X$$ and $icode B$$ are not used.

$head A$$
The argument $icode A$$ has the prototype
$codei%
	const %FloatVector% &%A%
%$$
and the size of $icode A$$ must equal $latex n * n$$
(see description of $cref/FloatVector/LuSolve/FloatVector/$$ below).
This is the $latex n$$ by $icode n$$ matrix that
we are computing the determinant of
and that defines the linear equation.

$head B$$
The argument $icode B$$ has the prototype
$codei%
	const %FloatVector% &%B%
%$$
and the size of $icode B$$ must equal $latex n * m$$
(see description of $cref/FloatVector/LuSolve/FloatVector/$$ below).
This is the $latex n$$ by $icode m$$ matrix that
defines the right hand side of the linear equations.
If $icode m$$ is zero, $icode B$$ is not used.

$head X$$
The argument $icode X$$ has the prototype
$codei%
	%FloatVector% &%X%
%$$
and the size of $icode X$$ must equal $latex n * m$$
(see description of $cref/FloatVector/LuSolve/FloatVector/$$ below).
The input value of $icode X$$ does not matter.
On output, the elements of $icode X$$ contain the solution
of the equation we wish to solve
(unless $icode signdet$$ is equal to zero).
If $icode m$$ is zero, $icode X$$ is not used.

$head logdet$$
The argument $icode logdet$$ has prototype
$codei%
	%Float% &%logdet%
%$$
On input, the value of $icode logdet$$ does not matter.
On output, it has been set to the
log of the determinant of $icode A$$
(but not quite).
To be more specific,
the determinant of $icode A$$ is given by the formula
$codei%
	%det% = %signdet% * exp( %logdet% )
%$$
This enables $code LuSolve$$ to use logs of absolute values
in the case where $icode Float$$ corresponds to a real number.

$head Float$$
The type $icode Float$$ must satisfy the conditions
for a $cref NumericType$$ type.
The routine $cref CheckNumericType$$ will generate an error message
if this is not the case.
In addition, the following operations must be defined for any pair
of $icode Float$$ objects $icode x$$ and $icode y$$:

$table
$bold Operation$$ $cnext $bold Description$$  $rnext
$codei%log(%x%)%$$ $cnext
	returns the logarithm of $icode x$$ as a $icode Float$$ object
$tend

$head FloatVector$$
The type $icode FloatVector$$ must be a $cref SimpleVector$$ class with
$cref/elements of type Float/SimpleVector/Elements of Specified Type/$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$head LeqZero$$
Including the file $code lu_solve.hpp$$ defines the template function
$codei%
	template <typename %Float%>
	bool LeqZero<%Float%>(const %Float% &%x%)
%$$
in the $code CppAD$$ namespace.
This function returns true if $icode x$$ is less than or equal to zero
and false otherwise.
It is used by $code LuSolve$$ to avoid taking the log of
zero (or a negative number if $icode Float$$ corresponds to real numbers).
This template function definition assumes that the operator
$code <=$$ is defined for $icode Float$$ objects.
If this operator is not defined for your use of $icode Float$$,
you will need to specialize this template so that it works for your
use of $code LuSolve$$.
$pre

$$
Complex numbers do not have the operation or $code <=$$ defined.
In addition, in the complex case,
one can take the log of a negative number.
The specializations
$codei%
	bool LeqZero< std::complex<float> > (const std::complex<float> &%x%)
	bool LeqZero< std::complex<double> >(const std::complex<double> &%x%)
%$$
are defined by including $code lu_solve.hpp$$.
These return true if $icode x$$ is zero and false otherwise.

$head AbsGeq$$
Including the file $code lu_solve.hpp$$ defines the template function
$codei%
	template <typename %Float%>
	bool AbsGeq<%Float%>(const %Float% &%x%, const %Float% &%y%)
%$$
If the type $icode Float$$ does not support the $code <=$$ operation
and it is not $code std::complex<float>$$ or $code std::complex<double>$$,
see the documentation for $code AbsGeq$$ in $cref/LuFactor/LuFactor/AbsGeq/$$.

$children%
	example/utility/lu_solve.cpp%
	omh/lu_solve_hpp.omh
%$$
$head Example$$
The file
$cref lu_solve.cpp$$
contains an example and test of using this routine.
It returns true if it succeeds and false otherwise.

$head Source$$
The file $cref lu_solve.hpp$$ contains the
current source code that implements these specifications.

$end
--------------------------------------------------------------------------
*/
// BEGIN C++
# include <complex>
# include <vector>

// link exp for float and double cases
# include <cppad/base_require.hpp>

# include <cppad/core/cppad_assert.hpp>
# include <cppad/utility/check_simple_vector.hpp>
# include <cppad/utility/check_numeric_type.hpp>
# include <cppad/utility/lu_factor.hpp>
# include <cppad/utility/lu_invert.hpp>

namespace CppAD { // BEGIN CppAD namespace

// LeqZero
template <typename Float>
inline bool LeqZero(const Float &x)
{	return x <= Float(0); }
inline bool LeqZero( const std::complex<double> &x )
{	return x == std::complex<double>(0); }
inline bool LeqZero( const std::complex<float> &x )
{	return x == std::complex<float>(0); }

// LuSolve
template <typename Float, typename FloatVector>
int LuSolve(
	size_t             n      ,
	size_t             m      ,
	const FloatVector &A      ,
	const FloatVector &B      ,
	FloatVector       &X      ,
	Float        &logdet      )
{
	// check numeric type specifications
	CheckNumericType<Float>();

	// check simple vector class specifications
	CheckSimpleVector<Float, FloatVector>();

	size_t        p;       // index of pivot element (diagonal of L)
	int     signdet;       // sign of the determinant
	Float     pivot;       // pivot element

	// the value zero
	const Float zero(0);

	// pivot row and column order in the matrix
	std::vector<size_t> ip(n);
	std::vector<size_t> jp(n);

	// -------------------------------------------------------
	CPPAD_ASSERT_KNOWN(
		size_t(A.size()) == n * n,
		"Error in LuSolve: A must have size equal to n * n"
	);
	CPPAD_ASSERT_KNOWN(
		size_t(B.size()) == n * m,
		"Error in LuSolve: B must have size equal to n * m"
	);
	CPPAD_ASSERT_KNOWN(
		size_t(X.size()) == n * m,
		"Error in LuSolve: X must have size equal to n * m"
	);
	// -------------------------------------------------------

	// copy A so that it does not change
	FloatVector Lu(A);

	// copy B so that it does not change
	X = B;

	// Lu factor the matrix A
	signdet = LuFactor(ip, jp, Lu);

	// compute the log of the determinant
	logdet  = Float(0);
	for(p = 0; p < n; p++)
	{	// pivot using the max absolute element
		pivot   = Lu[ ip[p] * n + jp[p] ];

		// check for determinant equal to zero
		if( pivot == zero )
		{	// abort the mission
			logdet = Float(0);
			return   0;
		}

		// update the determinant
		if( LeqZero ( pivot ) )
		{	logdet += log( - pivot );
			signdet = - signdet;
		}
		else	logdet += log( pivot );

	}

	// solve the linear equations
	LuInvert(ip, jp, Lu, X);

	// return the sign factor for the determinant
	return signdet;
}
} // END CppAD namespace
// END C++
# endif
