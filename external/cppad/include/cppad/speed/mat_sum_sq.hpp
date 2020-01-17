# ifndef CPPAD_SPEED_MAT_SUM_SQ_HPP
# define CPPAD_SPEED_MAT_SUM_SQ_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin mat_sum_sq$$
$spell
	sq
	namespace
	const
	CppAD
	sq
	cppad
	hpp
$$

$section Sum Elements of a Matrix Times Itself$$
$mindex mat_sum_sq multiply speed test$$

$head Syntax$$
$codei%# include <cppad/speed/mat_sum_sq.hpp>
%$$
$icode%mat_sum_sq(%n%, %x%, %y%, %z%)%$$

$head Purpose$$
This routine is intended for use with the matrix multiply speed tests;
to be specific, it computes
$latex \[
\begin{array}{rcl}
	y_{i,j} & = & \sum_{k=0}^{n-1} x_{i,k} x_{k,j}
	\\
	z_0     & = & \sum_{i=0}^{n-1} \sum_{j=0}^{n-1} y_{i,j}
\end{array}
\] $$
see $cref link_mat_mul$$.

$head Inclusion$$
The template function $code mat_sum_sq$$ is defined in the $code CppAD$$
namespace by including
the file $code cppad/speed/mat_sum_sq.hpp$$
(relative to the CppAD distribution directory).

$head n$$
This argument has prototype
$codei%
	size_t %n%
%$$
It specifies the size of the matrices.

$head x$$
The argument $icode x$$ has prototype
$codei%
	const %Vector% &%x%
%$$
and $icode%x%.size() == %n% * %n%$$.
It contains the elements of $latex x$$ in row major order; i.e.,
$latex \[
	x_{i,j} = x [ i * n + j ]
\] $$

$head y$$
The argument $icode y$$ has prototype
$codei%
	%Vector%& %y%
%$$
and $icode%y%.size() == %n% * %n%$$.
The input value of its elements does not matter.
Upon return,
$latex \[
\begin{array}{rcl}
	y_{i,j}        & = & \sum_{k=0}^{n-1} x_{i,k} x_{k,j}
	\\
	y[ i * n + j ] & = & y_{i,j}
\end{array}
\] $$


$head z$$
The argument $icode d$$ has prototype
$codei%
	%Vector%& %z%
%$$.
The input value of its element does not matter.
Upon return
$latex \[
\begin{array}{rcl}
	z_0 & = & \sum_{i=0}^{n-1} \sum_{j=0}^n y_{i,j}
	\\
	z[0] & = & z_0
\end{array}
\] $$

$head Vector$$
The type $icode Vector$$ is any
$cref SimpleVector$$, or it can be a raw pointer to the vector elements.
The element type must support
addition, multiplication, and assignment to both its own type
and to a double value.

$children%
	speed/example/mat_sum_sq.cpp%
	omh/mat_sum_sq_hpp.omh
%$$


$head Example$$
The file
$cref mat_sum_sq.cpp$$
contains an example and test of $code mat_sum_sq.hpp$$.
It returns true if it succeeds and false otherwise.

$head Source Code$$
The file
$cref mat_sum_sq.hpp$$
contains the source for this template function.

$end
------------------------------------------------------------------------------
*/
// BEGIN C++
# include <cstddef>
//
namespace CppAD {
	template <class Vector>
	void mat_sum_sq(size_t n, Vector& x , Vector& y , Vector& z)
	{	size_t i, j, k;
		// Very simple computation of y = x * x for speed comparison
		for(i = 0; i < n; i++)
		{	for(j = 0; j < n; j++)
			{	y[i * n + j] = 0.;
				for(k = 0; k < n; k++)
					y[i * n + j] += x[i * n + k] * x[k * n + j];
			}
		}
		z[0] = 0.;
		for(i = 0; i < n; i++)
		{	for(j = 0; j < n; j++)
				z[0] += y[i * n + j];
		}
		return;
	}

}
// END C++
# endif
