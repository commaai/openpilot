# ifndef CPPAD_SPEED_DET_OF_MINOR_HPP
# define CPPAD_SPEED_DET_OF_MINOR_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin det_of_minor$$
$spell
	CppAD
	hpp
	std
	Det
	const
	namespace
	cppad
$$


$section Determinant of a Minor$$
$mindex det_of_minor matrix$$

$head Syntax$$
$codei%# include <cppad/speed/det_of_minor.hpp>
%$$
$icode%d% = det_of_minor(%a%, %m%, %n%, %r%, %c%)%$$


$head Inclusion$$
The template function $code det_of_minor$$ is defined in the $code CppAD$$
namespace by including
the file $code cppad/speed/det_of_minor.hpp$$
(relative to the CppAD distribution directory).

$head Purpose$$
This template function
returns the determinant of a minor of the matrix $latex A$$
using expansion by minors.
The elements of the $latex n \times n$$ minor $latex M$$
of the matrix $latex A$$ are defined,
for $latex i = 0 , \ldots , n-1$$ and $latex j = 0 , \ldots , n-1$$, by
$latex \[
	M_{i,j} = A_{R(i), C(j)}
\]$$
where the functions
$latex R(i)$$ is defined by the $cref/argument r/det_of_minor/r/$$ and
$latex C(j)$$ is defined by the $cref/argument c/det_of_minor/c/$$.
$pre

$$
This template function
is for example and testing purposes only.
Expansion by minors is chosen as an example because it uses
a lot of floating point operations yet does not require much source code
(on the order of $icode m$$ factorial floating point operations and
about 70 lines of source code including comments).
This is not an efficient method for computing a determinant;
for example, using an LU factorization would be better.

$head Determinant of A$$
If the following conditions hold, the minor is the
entire matrix $latex A$$ and hence $code det_of_minor$$
will return the determinant of $latex A$$:

$list number$$
$latex n = m$$.
$lnext
for $latex i = 0 , \ldots , m-1$$, $latex r[i] = i+1$$,
and $latex r[m] = 0$$.
$lnext
for $latex j = 0 , \ldots , m-1$$, $latex c[j] = j+1$$,
and $latex c[m] = 0$$.
$lend

$head a$$
The argument $icode a$$ has prototype
$codei%
	const std::vector<%Scalar%>& %a%
%$$
and is a vector with size $latex m * m$$
(see description of $cref/Scalar/det_of_minor/Scalar/$$ below).
The elements of the $latex m \times m$$ matrix $latex A$$ are defined,
for $latex i = 0 , \ldots , m-1$$ and $latex j = 0 , \ldots , m-1$$, by
$latex \[
	A_{i,j} = a[ i * m + j]
\] $$

$head m$$
The argument $icode m$$ has prototype
$codei%
	size_t %m%
%$$
and is the number of rows (and columns) in the square matrix $latex A$$.

$head n$$
The argument $icode n$$ has prototype
$codei%
	size_t %n%
%$$
and is the number of rows (and columns) in the square minor $latex M$$.

$head r$$
The argument $icode r$$ has prototype
$codei%
	std::vector<size_t>& %r%
%$$
and is a vector with $latex m + 1$$ elements.
This vector defines the function $latex R(i)$$
which specifies the rows of the minor $latex M$$.
To be specific, the function $latex R(i)$$
for $latex i = 0, \ldots , n-1$$ is defined by
$latex \[
\begin{array}{rcl}
	R(0)   & = & r[m]
	\\
	R(i+1) & = & r[ R(i) ]
\end{array}
\] $$
All the elements of $icode r$$ must have value
less than or equal $icode m$$.
The elements of vector $icode r$$ are modified during the computation,
and restored to their original value before the return from
$code det_of_minor$$.

$head c$$
The argument $icode c$$ has prototype
$codei%
	std::vector<size_t>& %c%
%$$
and is a vector with $latex m + 1$$ elements
This vector defines the function $latex C(i)$$
which specifies the rows of the minor $latex M$$.
To be specific, the function $latex C(i)$$
for $latex j = 0, \ldots , n-1$$ is defined by
$latex \[
\begin{array}{rcl}
	C(0)   & = & c[m]
	\\
	C(j+1) & = & c[ C(j) ]
\end{array}
\] $$
All the elements of $icode c$$ must have value
less than or equal $icode m$$.
The elements of vector $icode c$$ are modified during the computation,
and restored to their original value before the return from
$code det_of_minor$$.

$head d$$
The result $icode d$$ has prototype
$codei%
	%Scalar% %d%
%$$
and is equal to the determinant of the minor $latex M$$.

$head Scalar$$
If $icode x$$ and $icode y$$ are objects of type $icode Scalar$$
and $icode i$$ is an object of type $code int$$,
the $icode Scalar$$ must support the following operations:
$table
$bold Syntax$$
	$cnext $bold Description$$
	$cnext $bold Result Type$$
$rnext
$icode%Scalar% %x%$$
	$cnext default constructor for $icode Scalar$$ object.
$rnext
$icode%x% = %i%$$
	$cnext set value of $icode x$$ to current value of $icode i$$
$rnext
$icode%x% = %y%$$
	$cnext set value of $icode x$$ to current value of $icode y$$
$rnext
$icode%x% + %y%$$
	$cnext value of $icode x$$ plus $icode y$$
	$cnext $icode Scalar$$
$rnext
$icode%x% - %y%$$
	$cnext value of $icode x$$ minus $icode y$$
	$cnext $icode Scalar$$
$rnext
$icode%x% * %y%$$
	$cnext value of $icode x$$ times value of $icode y$$
	$cnext $icode Scalar$$
$tend

$children%
	speed/example/det_of_minor.cpp%
	omh/det_of_minor_hpp.omh
%$$

$head Example$$
The file
$cref det_of_minor.cpp$$
contains an example and test of $code det_of_minor.hpp$$.
It returns true if it succeeds and false otherwise.

$head Source Code$$
The file
$cref det_of_minor.hpp$$
contains the source for this template function.


$end
---------------------------------------------------------------------------
*/
// BEGIN C++
# include <vector>
# include <cstddef>

namespace CppAD { // BEGIN CppAD namespace
template <class Scalar>
Scalar det_of_minor(
	const std::vector<Scalar>& a  ,
	size_t                     m  ,
	size_t                     n  ,
	std::vector<size_t>&       r  ,
	std::vector<size_t>&       c  )
{
	const size_t R0 = r[m]; // R(0)
	size_t       Cj = c[m]; // C(j)    (case j = 0)
	size_t       Cj1 = m;   // C(j-1)  (case j = 0)

	// check for 1 by 1 case
	if( n == 1 ) return a[ R0 * m + Cj ];

	// initialize determinant of the minor M
	Scalar detM = Scalar(0);

	// initialize sign of factor for next sub-minor
	int s = 1;

	// remove row with index 0 in M from all the sub-minors of M
	r[m] = r[R0];

	// for each column of M
	for(size_t j = 0; j < n; j++)
	{	// element with index (0,j) in the minor M
		Scalar M0j = a[ R0 * m + Cj ];

		// remove column with index j in M to form next sub-minor S of M
		c[Cj1] = c[Cj];

		// compute determinant of the current sub-minor S
		Scalar detS = det_of_minor(a, m, n - 1, r, c);

		// restore column Cj to represenation of M as a minor of A
		c[Cj1] = Cj;

		// include this sub-minor term in the summation
		if( s > 0 )
			detM = detM + M0j * detS;
		else	detM = detM - M0j * detS;

		// advance to next column of M
		Cj1 = Cj;
		Cj  = c[Cj];
		s   = - s;
	}

	// restore row zero to the minor representation for M
	r[m] = R0;

	// return the determinant of the minor M
	return detM;
}
} // END CppAD namespace
// END C++
# endif
