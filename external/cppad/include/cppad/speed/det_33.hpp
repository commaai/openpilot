# ifndef CPPAD_SPEED_DET_33_HPP
# define CPPAD_SPEED_DET_33_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin det_33$$
$spell
	cppad
	CppAD
	det
	namespace
	const
	bool
	hpp
$$

$section Check Determinant of 3 by 3 matrix$$
$mindex det_33 correct$$


$head Syntax$$
$codei%# include <cppad/speed/det_33.hpp>
%$$
$icode%ok% = det_33(%x%, %d%)%$$

$head Purpose$$
This routine can be used to check a method for computing
the determinant of a matrix.

$head Inclusion$$
The template function $code det_33$$ is defined in the $code CppAD$$
namespace by including
the file $code cppad/speed/det_33.hpp$$
(relative to the CppAD distribution directory).

$head x$$
The argument $icode x$$ has prototype
$codei%
	const %Vector% &%x%
%$$.
It contains the elements of the matrix $latex X$$ in row major order; i.e.,
$latex \[
	X_{i,j} = x [ i * 3 + j ]
\] $$

$head d$$
The argument $icode d$$ has prototype
$codei%
	const %Vector% &%d%
%$$.
It is tested to see if $icode%d%[0]%$$ it is equal to $latex \det ( X )$$.

$head Vector$$
If $icode y$$ is a $icode Vector$$ object,
it must support the syntax
$codei%
	%y%[%i%]
%$$
where $icode i$$ has type $code size_t$$ with value less than 9.
This must return a $code double$$ value corresponding to the $th i$$
element of the vector $icode y$$.
This is the only requirement of the type $icode Vector$$.
(Note that only the first element of the vector $icode d$$ is used.)

$head ok$$
The return value $icode ok$$ has prototype
$codei%
	bool %ok%
%$$
It is true, if the determinant $icode%d%[0]%$$
passes the test and false otherwise.

$children%
	omh/det_33_hpp.omh
%$$

$head Source Code$$
The file
$cref det_33.hpp$$
contains the source code for this template function.

$end
------------------------------------------------------------------------------
*/
// BEGIN C++
# include <cppad/utility/near_equal.hpp>
namespace CppAD {
template <class Vector>
	bool det_33(const Vector &x, const Vector &d)
	{	bool ok = true;
		double eps99 = 99.0 * std::numeric_limits<double>::epsilon();

		// use expansion by minors to compute the determinant by hand
		double check = 0.;
		check += x[0] * ( x[4] * x[8] - x[5] * x[7] );
		check -= x[1] * ( x[3] * x[8] - x[5] * x[6] );
		check += x[2] * ( x[3] * x[7] - x[4] * x[6] );

		ok &= CppAD::NearEqual(check, d[0], eps99, eps99);

		return ok;
	}
}
// END C++
# endif
