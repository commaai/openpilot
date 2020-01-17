# ifndef CPPAD_WNO_CONVERSION_HPP
# define CPPAD_WNO_CONVERSION_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin wno_conversion$$
$spell
	cppad
	wno
	cpp
	hpp
$$

$section Suppress Suspect Implicit Conversion Warnings$$

$head Syntax$$
$codei%# include <cppad/wno_conversion.hpp>%$$

$head Purpose$$
In many cases it is good to have warnings for implicit conversions
that may loose range or precision.
The include command above, before any other includes, suppresses
these warning for a particular compilation unit (which usually corresponds
to a $icode%*%.cpp%$$ file).

$end
*/

# include <cppad/configure.hpp>
# if CPPAD_COMPILER_IS_GNUCXX
# pragma GCC diagnostic ignored "-Wfloat-conversion"
# pragma GCC diagnostic ignored "-Wconversion"
# endif

# endif
