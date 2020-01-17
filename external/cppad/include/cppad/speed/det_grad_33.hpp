# ifndef CPPAD_SPEED_DET_GRAD_33_HPP
# define CPPAD_SPEED_DET_GRAD_33_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin det_grad_33$$
$spell
	cppad
	CppAD
	det
	namespace
	const
	bool
	hpp
$$

$section Check Gradient of Determinant of 3 by 3 matrix$$
$mindex det_grad_33 correct$$


$head Syntax$$
$codei%# include <cppad/speed/det_grad_33.hpp>
%$$
$icode%ok% = det_grad_33(%x%, %g%)%$$

$head Purpose$$
This routine can be used to check a method for computing the
gradient of the determinant of a matrix.

$head Inclusion$$
The template function $code det_grad_33$$ is defined in the $code CppAD$$
namespace by including
the file $code cppad/speed/det_grad_33.hpp$$
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

$head g$$
The argument $icode g$$ has prototype
$codei%
	const %Vector% &%g%
%$$.
It contains the elements of the gradient of
$latex \det ( X )$$ in row major order; i.e.,
$latex \[
	\D{\det (X)}{X(i,j)} = g [ i * 3 + j ]
\] $$

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

$head ok$$
The return value $icode ok$$ has prototype
$codei%
	bool %ok%
%$$
It is true, if the gradient $icode g$$
passes the test and false otherwise.

$children%
	omh/det_grad_33_hpp.omh
%$$

$head Source Code$$
The file
$cref det_grad_33.hpp$$
contains the source code for this template function.

$end
------------------------------------------------------------------------------
*/
// BEGIN C++
# include <limits>
# include <cppad/utility/near_equal.hpp>
namespace CppAD {
template <class Vector>
	bool det_grad_33(const Vector &x, const Vector &g)
	{	bool ok = true;
		typedef typename Vector::value_type Float;
		Float eps = 10. * Float( std::numeric_limits<double>::epsilon() );

		// use expansion by minors to compute the derivative by hand
		double check[9];
		check[0] = + ( x[4] * x[8] - x[5] * x[7] );
		check[1] = - ( x[3] * x[8] - x[5] * x[6] );
		check[2] = + ( x[3] * x[7] - x[4] * x[6] );
		//
		check[3] = - ( x[1] * x[8] - x[2] * x[7] );
		check[4] = + ( x[0] * x[8] - x[2] * x[6] );
		check[5] = - ( x[0] * x[7] - x[1] * x[6] );
		//
		check[6] = + ( x[1] * x[5] - x[2] * x[4] );
		check[7] = - ( x[0] * x[5] - x[2] * x[3] );
		check[8] = + ( x[0] * x[4] - x[1] * x[3] );
		//
		for(size_t i = 0; i < 3 * 3; i++)
			ok &= CppAD::NearEqual(check[i], g[i], eps, eps);

		return ok;
	}
}
// END C++
# endif
