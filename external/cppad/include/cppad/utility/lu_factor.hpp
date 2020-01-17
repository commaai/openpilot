# ifndef CPPAD_UTILITY_LU_FACTOR_HPP
# define CPPAD_UTILITY_LU_FACTOR_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin LuFactor$$
$escape #$$
$spell
	cppad.hpp
	Cpp
	Geq
	Lu
	bool
	const
	ip
	jp
	namespace
	std
	typename
$$


$section LU Factorization of A Square Matrix$$
$mindex LuFactor linear equation solve$$

$pre
$$

$head Syntax$$ $codei%# include <cppad/utility/lu_factor.hpp>
%$$
$icode%sign% = LuFactor(%ip%, %jp%, %LU%)%$$


$head Description$$
Computes an LU factorization of the matrix $icode A$$
where $icode A$$ is a square matrix.

$head Include$$
The file $code cppad/lu_factor.hpp$$ is included by $code cppad/cppad.hpp$$
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

$head sign$$
The return value $icode sign$$ has prototype
$codei%
	int %sign%
%$$
If $icode A$$ is invertible, $icode sign$$ is plus or minus one
and is the sign of the permutation corresponding to the row ordering
$icode ip$$ and column ordering $icode jp$$.
If $icode A$$ is not invertible, $icode sign$$ is zero.

$head ip$$
The argument $icode ip$$ has prototype
$codei%
	%SizeVector% &%ip%
%$$
(see description of $cref/SizeVector/LuFactor/SizeVector/$$ below).
The size of $icode ip$$ is referred to as $icode n$$ in the
specifications below.
The input value of the elements of $icode ip$$ does not matter.
The output value of the elements of $icode ip$$ determine
the order of the rows in the permuted matrix.

$head jp$$
The argument $icode jp$$ has prototype
$codei%
	%SizeVector% &%jp%
%$$
(see description of $cref/SizeVector/LuFactor/SizeVector/$$ below).
The size of $icode jp$$ must be equal to $icode n$$.
The input value of the elements of $icode jp$$ does not matter.
The output value of the elements of $icode jp$$ determine
the order of the columns in the permuted matrix.

$head LU$$
The argument $icode LU$$ has the prototype
$codei%
	%FloatVector% &%LU%
%$$
and the size of $icode LU$$ must equal $latex n * n$$
(see description of $cref/FloatVector/LuFactor/FloatVector/$$ below).

$subhead A$$
We define $icode A$$ as the matrix corresponding to the input
value of $icode LU$$.

$subhead P$$
We define the permuted matrix $icode P$$ in terms of $icode A$$ by
$codei%
	%P%(%i%, %j%) = %A%[ %ip%[%i%] * %n% + %jp%[%j%] ]
%$$

$subhead L$$
We define the lower triangular matrix $icode L$$ in terms of the
output value of $icode LU$$.
The matrix $icode L$$ is zero above the diagonal
and the rest of the elements are defined by
$codei%
	%L%(%i%, %j%) = %LU%[ %ip%[%i%] * %n% + %jp%[%j%] ]
%$$
for $latex i = 0 , \ldots , n-1$$ and $latex j = 0 , \ldots , i$$.

$subhead U$$
We define the upper triangular matrix $icode U$$ in terms of the
output value of $icode LU$$.
The matrix $icode U$$ is zero below the diagonal,
one on the diagonal,
and the rest of the elements are defined by
$codei%
	%U%(%i%, %j%) = %LU%[ %ip%[%i%] * %n% + %jp%[%j%] ]
%$$
for $latex i = 0 , \ldots , n-2$$ and $latex j = i+1 , \ldots , n-1$$.

$subhead Factor$$
If the return value $icode sign$$ is non-zero,
$codei%
	%L% * %U% = %P%
%$$
If the return value of $icode sign$$ is zero,
the contents of $icode L$$ and $icode U$$ are not defined.

$subhead Determinant$$
If the return value $icode sign$$ is zero,
the determinant of $icode A$$ is zero.
If $icode sign$$ is non-zero,
using the output value of $icode LU$$
the determinant of the matrix $icode A$$ is equal to
$codei%
%sign% * %LU%[%ip%[0], %jp%[0]] * %...% * %LU%[%ip%[%n%-1], %jp%[%n%-1]]
%$$

$head SizeVector$$
The type $icode SizeVector$$ must be a $cref SimpleVector$$ class with
$cref/elements of type size_t/SimpleVector/Elements of Specified Type/$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$head FloatVector$$
The type $icode FloatVector$$ must be a
$cref/simple vector class/SimpleVector/$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$head Float$$
This notation is used to denote the type corresponding
to the elements of a $icode FloatVector$$.
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

$head AbsGeq$$
Including the file $code lu_factor.hpp$$ defines the template function
$codei%
	template <typename %Float%>
	bool AbsGeq<%Float%>(const %Float% &%x%, const %Float% &%y%)
%$$
in the $code CppAD$$ namespace.
This function returns true if the absolute value of
$icode x$$ is greater than or equal the absolute value of $icode y$$.
It is used by $code LuFactor$$ to choose the pivot elements.
This template function definition uses the operator
$code <=$$ to obtain the absolute value for $icode Float$$ objects.
If this operator is not defined for your use of $icode Float$$,
you will need to specialize this template so that it works for your
use of $code LuFactor$$.
$pre

$$
Complex numbers do not have the operation $code <=$$ defined.
The specializations
$codei%
bool AbsGeq< std::complex<float> >
	(const std::complex<float> &%x%, const std::complex<float> &%y%)
bool AbsGeq< std::complex<double> >
	(const std::complex<double> &%x%, const std::complex<double> &%y%)
%$$
are define by including $code lu_factor.hpp$$
These return true if the sum of the square of the real and imaginary parts
of $icode x$$ is greater than or equal the
sum of the square of the real and imaginary parts of $icode y$$.

$children%
	example/utility/lu_factor.cpp%
	omh/lu_factor_hpp.omh
%$$
$head Example$$
The file
$cref lu_factor.cpp$$
contains an example and test of using $code LuFactor$$ by itself.
It returns true if it succeeds and false otherwise.
$pre

$$
The file $cref lu_solve.hpp$$ provides a useful example usage of
$code LuFactor$$ with $code LuInvert$$.

$head Source$$
The file $cref lu_factor.hpp$$ contains the
current source code that implements these specifications.

$end
--------------------------------------------------------------------------
*/
// BEGIN C++

# include <complex>
# include <vector>

# include <cppad/core/cppad_assert.hpp>
# include <cppad/utility/check_simple_vector.hpp>
# include <cppad/utility/check_numeric_type.hpp>

namespace CppAD { // BEGIN CppAD namespace

// AbsGeq
template <typename Float>
inline bool AbsGeq(const Float &x, const Float &y)
{	Float xabs = x;
	if( xabs <= Float(0) )
		xabs = - xabs;
	Float yabs = y;
	if( yabs <= Float(0) )
		yabs = - yabs;
	return xabs >= yabs;
}
inline bool AbsGeq(
	const std::complex<double> &x,
	const std::complex<double> &y)
{	double xsq = x.real() * x.real() + x.imag() * x.imag();
	double ysq = y.real() * y.real() + y.imag() * y.imag();

	return xsq >= ysq;
}
inline bool AbsGeq(
	const std::complex<float> &x,
	const std::complex<float> &y)
{	float xsq = x.real() * x.real() + x.imag() * x.imag();
	float ysq = y.real() * y.real() + y.imag() * y.imag();

	return xsq >= ysq;
}

// Lines that are different from code in cppad/core/lu_ratio.hpp end with //
template <class SizeVector, class FloatVector>                          //
int LuFactor(SizeVector &ip, SizeVector &jp, FloatVector &LU)           //
{
	// type of the elements of LU                                   //
	typedef typename FloatVector::value_type Float;                 //

	// check numeric type specifications
	CheckNumericType<Float>();

	// check simple vector class specifications
	CheckSimpleVector<Float, FloatVector>();
	CheckSimpleVector<size_t, SizeVector>();

	size_t  i, j;          // some temporary indices
	const Float zero( 0 ); // the value zero as a Float object
	size_t  imax;          // row index of maximum element
	size_t  jmax;          // column indx of maximum element
	Float    emax;         // maximum absolute value
	size_t  p;             // count pivots
	int     sign;          // sign of the permutation
	Float   etmp;          // temporary element
	Float   pivot;         // pivot element

	// -------------------------------------------------------
	size_t n = ip.size();
	CPPAD_ASSERT_KNOWN(
		size_t(jp.size()) == n,
		"Error in LuFactor: jp must have size equal to n"
	);
	CPPAD_ASSERT_KNOWN(
		size_t(LU.size()) == n * n,
		"Error in LuFactor: LU must have size equal to n * m"
	);
	// -------------------------------------------------------

	// initialize row and column order in matrix not yet pivoted
	for(i = 0; i < n; i++)
	{	ip[i] = i;
		jp[i] = i;
	}
	// initialize the sign of the permutation
	sign = 1;
	// ---------------------------------------------------------

	// Reduce the matrix P to L * U using n pivots
	for(p = 0; p < n; p++)
	{	// determine row and column corresponding to element of
		// maximum absolute value in remaining part of P
		imax = jmax = n;
		emax = zero;
		for(i = p; i < n; i++)
		{	for(j = p; j < n; j++)
			{	CPPAD_ASSERT_UNKNOWN(
					(ip[i] < n) & (jp[j] < n)
				);
				etmp = LU[ ip[i] * n + jp[j] ];

				// check if maximum absolute value so far
				if( AbsGeq (etmp, emax) )
				{	imax = i;
					jmax = j;
					emax = etmp;
				}
			}
		}
		CPPAD_ASSERT_KNOWN(
		(imax < n) & (jmax < n) ,
		"LuFactor can't determine an element with "
		"maximum absolute value.\n"
		"Perhaps original matrix contains not a number or infinity.\n"
		"Perhaps your specialization of AbsGeq is not correct."
		);
		if( imax != p )
		{	// switch rows so max absolute element is in row p
			i        = ip[p];
			ip[p]    = ip[imax];
			ip[imax] = i;
			sign     = -sign;
		}
		if( jmax != p )
		{	// switch columns so max absolute element is in column p
			j        = jp[p];
			jp[p]    = jp[jmax];
			jp[jmax] = j;
			sign     = -sign;
		}
		// pivot using the max absolute element
		pivot   = LU[ ip[p] * n + jp[p] ];

		// check for determinant equal to zero
		if( pivot == zero )
		{	// abort the mission
			return   0;
		}

		// Reduce U by the elementary transformations that maps
		// LU( ip[p], jp[p] ) to one.  Only need transform elements
		// above the diagonal in U and LU( ip[p] , jp[p] ) is
		// corresponding value below diagonal in L.
		for(j = p+1; j < n; j++)
			LU[ ip[p] * n + jp[j] ] /= pivot;

		// Reduce U by the elementary transformations that maps
		// LU( ip[i], jp[p] ) to zero. Only need transform elements
		// above the diagonal in U and LU( ip[i], jp[p] ) is
		// corresponding value below diagonal in L.
		for(i = p+1; i < n; i++ )
		{	etmp = LU[ ip[i] * n + jp[p] ];
			for(j = p+1; j < n; j++)
			{	LU[ ip[i] * n + jp[j] ] -=
					etmp * LU[ ip[p] * n + jp[j] ];
			}
		}
	}
	return sign;
}
} // END CppAD namespace
// END C++
# endif
