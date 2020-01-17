# ifndef CPPAD_SPEED_UNIFORM_01_HPP
# define CPPAD_SPEED_UNIFORM_01_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin uniform_01$$
$spell
	CppAD
	namespace
	cppad
	hpp
$$

$section Simulate a [0,1] Uniform Random Variate$$
$mindex uniform_01$$


$head Syntax$$
$codei%# include <cppad/speed/uniform_01.hpp>
%$$
$codei%uniform_01(%seed%)
%$$
$codei%uniform_01(%n%, %x%)%$$

$head Purpose$$
This routine is used to create random values for speed testing purposes.

$head Inclusion$$
The template function $code uniform_01$$ is defined in the $code CppAD$$
namespace by including
the file $code cppad/speed/uniform_01.hpp$$
(relative to the CppAD distribution directory).

$head seed$$
The argument $icode seed$$ has prototype
$codei%
	size_t %seed%
%$$
It specifies a seed
for the uniform random number generator.

$head n$$
The argument $icode n$$ has prototype
$codei%
	size_t %n%
%$$
It specifies the number of elements in the random vector $icode x$$.

$head x$$
The argument $icode x$$ has prototype
$codei%
	%Vector% &%x%
%$$.
The input value of the elements of $icode x$$ does not matter.
Upon return, the elements of $icode x$$ are set to values
randomly sampled over the interval [0,1].

$head Vector$$
If $icode y$$ is a $code double$$ value,
the object $icode x$$ must support the syntax
$codei%
	%x%[%i%] = %y%
%$$
where $icode i$$ has type $code size_t$$ with value less than
or equal $latex n-1$$.
This is the only requirement of the type $icode Vector$$.

$children%
	omh/uniform_01_hpp.omh
%$$

$head Source Code$$
The file
$cref uniform_01.hpp$$
constraints the source code for this template function.

$end
------------------------------------------------------------------------------
*/
// BEGIN C++
# include <cstdlib>

namespace CppAD {
	inline void uniform_01(size_t seed)
	{	std::srand( (unsigned int) seed); }

	template <class Vector>
	void uniform_01(size_t n, Vector &x)
	{	static double factor = 1. / double(RAND_MAX);
		while(n--)
			x[n] = std::rand() * factor;
	}
}
// END C++
# endif
