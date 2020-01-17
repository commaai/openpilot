# ifndef CPPAD_CORE_LU_RATIO_HPP
# define CPPAD_CORE_LU_RATIO_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin LuRatio$$
$spell
	cppad.hpp
	xk
	Cpp
	Lu
	bool
	const
	ip
	jp
	std
	ADvector
$$


$section LU Factorization of A Square Matrix and Stability Calculation$$
$mindex LuRatio linear equation solve$$

$head Syntax$$
$code# include <cppad/cppad.hpp>$$
$pre
$$
$icode%sign% = LuRatio(%ip%, %jp%, %LU%, %ratio%)%$$


$head Description$$
Computes an LU factorization of the matrix $icode A$$
where $icode A$$ is a square matrix.
A measure of the numerical stability called $icode ratio$$ is calculated.
This ratio is useful when the results of $code LuRatio$$ are
used as part of an $cref ADFun$$ object.

$head Include$$
This routine is designed to be used with AD objects and
requires the $code cppad/cppad.hpp$$ file to be included.

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
	%ADvector% &%LU%
%$$
and the size of $icode LU$$ must equal $latex n * n$$
(see description of $cref/ADvector/LuRatio/ADvector/$$ below).

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

$head ratio$$
The argument $icode ratio$$ has prototype
$codei%
        AD<%Base%> &%ratio%
%$$
On input, the value of $icode ratio$$ does not matter.
On output it is a measure of how good the choice of pivots is.
For $latex p = 0 , \ldots , n-1$$,
the $th p$$ pivot element is the element of maximum absolute value of a
$latex (n-p) \times (n-p)$$ sub-matrix.
The ratio of each element of sub-matrix divided by the pivot element
is computed.
The return value of $icode ratio$$ is the maximum absolute value of
such ratios over with respect to all elements and all the pivots.

$subhead Purpose$$
Suppose that the execution of a call to $code LuRatio$$
is recorded in the $codei%ADFun<%Base%>%$$ object $icode F$$.
Then a call to $cref Forward$$ of the form
$codei%
	%F%.Forward(%k%, %xk%)
%$$
with $icode k$$ equal to zero will revaluate this Lu factorization
with the same pivots and a new value for $icode A$$.
In this case, the resulting $icode ratio$$ may not be one.
If $icode ratio$$ is too large (the meaning of too large is up to you),
the current pivots do not yield a stable LU factorization of $icode A$$.
A better choice for the pivots (for this value of $icode A$$)
will be made if you recreate the $code ADFun$$ object
starting with the $cref Independent$$ variable values
that correspond to the vector $icode xk$$.

$head SizeVector$$
The type $icode SizeVector$$ must be a $cref SimpleVector$$ class with
$cref/elements of type size_t/SimpleVector/Elements of Specified Type/$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$head ADvector$$
The type $icode ADvector$$ must be a
$cref/simple vector class/SimpleVector/$$ with elements of type
$codei%AD<%Base%>%$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.


$head Example$$
$children%
	example/general/lu_ratio.cpp
%$$
The file $cref lu_ratio.cpp$$
contains an example and test of using $code LuRatio$$.
It returns true if it succeeds and false otherwise.

$end
--------------------------------------------------------------------------
*/
namespace CppAD { // BEGIN CppAD namespace

// Lines different from the code in cppad/lu_factor.hpp end with           //
template <class SizeVector, class ADvector, class Base>                    //
int LuRatio(SizeVector &ip, SizeVector &jp, ADvector &LU, AD<Base> &ratio) //
{
	typedef ADvector FloatVector;                                       //
	typedef AD<Base>       Float;                                       //

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
	size_t n = size_t(ip.size());
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
	// initialize the ratio                                             //
	ratio = Float(1);                                                   //
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
		for(i = p; i < n; i++)                                       //
		{	for(j = p; j < n; j++)                               //
			{	etmp  = fabs(LU[ ip[i] * n + jp[j] ] / emax); //
				ratio =                                      //
				CondExpGt(etmp, ratio, etmp, ratio);         //
			}                                                    //
		}                                                            //
		CPPAD_ASSERT_KNOWN(
			(imax < n) & (jmax < n) ,
			"AbsGeq must return true when second argument is zero"
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

# endif
