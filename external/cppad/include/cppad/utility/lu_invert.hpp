# ifndef CPPAD_UTILITY_LU_INVERT_HPP
# define CPPAD_UTILITY_LU_INVERT_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin LuInvert$$
$escape #$$
$spell
	cppad.hpp
	Lu
	Cpp
	jp
	ip
	const
	namespace
	typename
	etmp
$$


$section Invert an LU Factored Equation$$
$mindex LuInvert linear$$

$pre
$$

$head Syntax$$ $codei%# include <cppad/utility/lu_invert.hpp>
%$$
$codei%LuInvert(%ip%, %jp%, %LU%, %X%)%$$


$head Description$$
Solves the matrix equation $icode%A% * %X% = %B%$$
using an LU factorization computed by $cref LuFactor$$.

$head Include$$
The file $code cppad/lu_invert.hpp$$ is included by $code cppad/cppad.hpp$$
but it can also be included separately with out the rest of
the $code CppAD$$ routines.

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

$head ip$$
The argument $icode ip$$ has prototype
$codei%
	const %SizeVector% &%ip%
%$$
(see description for $icode SizeVector$$ in
$cref/LuFactor/LuFactor/SizeVector/$$ specifications).
The size of $icode ip$$ is referred to as $icode n$$ in the
specifications below.
The elements of $icode ip$$ determine
the order of the rows in the permuted matrix.

$head jp$$
The argument $icode jp$$ has prototype
$codei%
	const %SizeVector% &%jp%
%$$
(see description for $icode SizeVector$$ in
$cref/LuFactor/LuFactor/SizeVector/$$ specifications).
The size of $icode jp$$ must be equal to $icode n$$.
The elements of $icode jp$$ determine
the order of the columns in the permuted matrix.

$head LU$$
The argument $icode LU$$ has the prototype
$codei%
	const %FloatVector% &%LU%
%$$
and the size of $icode LU$$ must equal $latex n * n$$
(see description for $icode FloatVector$$ in
$cref/LuFactor/LuFactor/FloatVector/$$ specifications).

$subhead L$$
We define the lower triangular matrix $icode L$$ in terms of $icode LU$$.
The matrix $icode L$$ is zero above the diagonal
and the rest of the elements are defined by
$codei%
	%L%(%i%, %j%) = %LU%[ %ip%[%i%] * %n% + %jp%[%j%] ]
%$$
for $latex i = 0 , \ldots , n-1$$ and $latex j = 0 , \ldots , i$$.

$subhead U$$
We define the upper triangular matrix $icode U$$ in terms of $icode LU$$.
The matrix $icode U$$ is zero below the diagonal,
one on the diagonal,
and the rest of the elements are defined by
$codei%
	%U%(%i%, %j%) = %LU%[ %ip%[%i%] * %n% + %jp%[%j%] ]
%$$
for $latex i = 0 , \ldots , n-2$$ and $latex j = i+1 , \ldots , n-1$$.

$subhead P$$
We define the permuted matrix $icode P$$ in terms of
the matrix $icode L$$ and the matrix $icode U$$
by $icode%P% = %L% * %U%$$.

$subhead A$$
The matrix $icode A$$,
which defines the linear equations that we are solving, is given by
$codei%
	%P%(%i%, %j%) = %A%[ %ip%[%i%] * %n% + %jp%[%j%] ]
%$$
(Hence
$icode LU$$ contains a permuted factorization of the matrix $icode A$$.)


$head X$$
The argument $icode X$$ has prototype
$codei%
	%FloatVector% &%X%
%$$
(see description for $icode FloatVector$$ in
$cref/LuFactor/LuFactor/FloatVector/$$ specifications).
The matrix $icode X$$
must have the same number of rows as the matrix $icode A$$.
The input value of $icode X$$ is the matrix $icode B$$ and the
output value solves the matrix equation $icode%A% * %X% = %B%$$.


$children%
	example/utility/lu_invert.cpp%
	omh/lu_invert_hpp.omh
%$$
$head Example$$
The file $cref lu_solve.hpp$$ is a good example usage of
$code LuFactor$$ with $code LuInvert$$.
The file
$cref lu_invert.cpp$$
contains an example and test of using $code LuInvert$$ by itself.
It returns true if it succeeds and false otherwise.

$head Source$$
The file $cref lu_invert.hpp$$ contains the
current source code that implements these specifications.

$end
--------------------------------------------------------------------------
*/
// BEGIN C++
# include <cppad/core/cppad_assert.hpp>
# include <cppad/utility/check_simple_vector.hpp>
# include <cppad/utility/check_numeric_type.hpp>

namespace CppAD { // BEGIN CppAD namespace

// LuInvert
template <typename SizeVector, typename FloatVector>
void LuInvert(
	const SizeVector  &ip,
	const SizeVector  &jp,
	const FloatVector &LU,
	FloatVector       &B )
{	size_t k; // column index in X
	size_t p; // index along diagonal in LU
	size_t i; // row index in LU and X

	typedef typename FloatVector::value_type Float;

	// check numeric type specifications
	CheckNumericType<Float>();

	// check simple vector class specifications
	CheckSimpleVector<Float, FloatVector>();
	CheckSimpleVector<size_t, SizeVector>();

	Float etmp;

	size_t n = ip.size();
	CPPAD_ASSERT_KNOWN(
		size_t(jp.size()) == n,
		"Error in LuInvert: jp must have size equal to n * n"
	);
	CPPAD_ASSERT_KNOWN(
		size_t(LU.size()) == n * n,
		"Error in LuInvert: Lu must have size equal to n * m"
	);
	size_t m = size_t(B.size()) / n;
	CPPAD_ASSERT_KNOWN(
		size_t(B.size()) == n * m,
		"Error in LuSolve: B must have size equal to a multiple of n"
	);

	// temporary storage for reordered solution
	FloatVector x(n);

	// loop over equations
	for(k = 0; k < m; k++)
	{	// invert the equation c = L * b
		for(p = 0; p < n; p++)
		{	// solve for c[p]
			etmp = B[ ip[p] * m + k ] / LU[ ip[p] * n + jp[p] ];
			B[ ip[p] * m + k ] = etmp;
			// subtract off effect on other variables
			for(i = p+1; i < n; i++)
				B[ ip[i] * m + k ] -=
					etmp * LU[ ip[i] * n + jp[p] ];
		}

		// invert the equation x = U * c
		p = n;
		while( p > 0 )
		{	--p;
			etmp       = B[ ip[p] * m + k ];
			x[ jp[p] ] = etmp;
			for(i = 0; i < p; i++ )
				B[ ip[i] * m + k ] -=
					etmp * LU[ ip[i] * n + jp[p] ];
		}

		// copy reordered solution into B
		for(i = 0; i < n; i++)
			B[i * m + k] = x[i];
	}
	return;
}
} // END CppAD namespace
// END C++
# endif
