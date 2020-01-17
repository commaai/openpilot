// $Id$
# ifndef CPPAD_CORE_BASE_STD_MATH_HPP
# define CPPAD_CORE_BASE_STD_MATH_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin base_std_math$$
$spell
	expm1
	atanh
	acosh
	asinh
	inline
	fabs
	isnan
	alloc
	std
	acos
	asin
	atan
	cos
	exp
	sqrt
	const
	CppAD
	namespace
	erf
$$

$section Base Type Requirements for Standard Math Functions$$

$head Purpose$$
These definitions are required for the user's code to use the type
$codei%AD<%Base%>%$$:

$head Unary Standard Math$$
The type $icode Base$$ must support the following functions
unary standard math functions (in the CppAD namespace):
$table
$bold Syntax$$ $cnext $bold Result$$
$rnext
$icode%y% = abs(%x%)%$$  $cnext absolute value     $rnext
$icode%y% = acos(%x%)%$$ $cnext inverse cosine     $rnext
$icode%y% = asin(%x%)%$$ $cnext inverse sine       $rnext
$icode%y% = atan(%x%)%$$ $cnext inverse tangent    $rnext
$icode%y% = cos(%x%)%$$  $cnext cosine             $rnext
$icode%y% = cosh(%x%)%$$ $cnext hyperbolic cosine  $rnext
$icode%y% = exp(%x%)%$$  $cnext exponential        $rnext
$icode%y% = fabs(%x%)%$$ $cnext absolute value     $rnext
$icode%y% = log(%x%)%$$  $cnext natural logarithm  $rnext
$icode%y% = sin(%x%)%$$  $cnext sine               $rnext
$icode%y% = sinh(%x%)%$$ $cnext hyperbolic sine    $rnext
$icode%y% = sqrt(%x%)%$$ $cnext square root        $rnext
$icode%y% = tan(%x%)%$$  $cnext tangent
$tend
where the arguments and return value have the prototypes
$codei%
	const %Base%& %x%
	%Base%        %y%
%$$
For example,
$cref/base_alloc/base_alloc.hpp/Unary Standard Math/$$,


$head CPPAD_STANDARD_MATH_UNARY$$
The macro invocation, within the CppAD namespace,
$codei%
	CPPAD_STANDARD_MATH_UNARY(%Base%, %Fun%)
%$$
defines the syntax
$codei%
	%y% = CppAD::%Fun%(%x%)
%$$
This macro uses the functions $codei%std::%Fun%$$ which
must be defined and have the same prototype as $codei%CppAD::%Fun%$$.
For example,
$cref/float/base_float.hpp/Unary Standard Math/$$.

$head erf, asinh, acosh, atanh, expm1, log1p$$
If this preprocessor symbol
$code CPPAD_USE_CPLUSPLUS_2011$$ is true ($code 1$$),
when compiling for c++11, the type
$code double$$ is supported for the functions listed below.
In this case, the type $icode Base$$ must also support these functions:
$table
$bold Syntax$$ $cnext $bold Result$$
$rnext
$icode%y% = erf(%x%)%$$    $cnext error function                $rnext
$icode%y% = asinh(%x%)%$$  $cnext inverse hyperbolic sin        $rnext
$icode%y% = acosh(%x%)%$$  $cnext inverse hyperbolic cosine     $rnext
$icode%y% = atanh(%x%)%$$  $cnext inverse hyperbolic tangent    $rnext
$icode%y% = expm1(%x%)%$$  $cnext exponential of x minus one    $rnext
$icode%y% = log1p(%x%)%$$  $cnext logarithm of one plus x
$tend
where the arguments and return value have the prototypes
$codei%
	const %Base%& %x%
	%Base%        %y%
%$$

$head sign$$
The type $icode Base$$ must support the syntax
$codei%
	%y% = CppAD::sign(%x%)
%$$
which computes
$latex \[
y = \left\{ \begin{array}{ll}
	+1 & {\rm if} \; x > 0 \\
	 0 & {\rm if} \; x = 0 \\
	-1 & {\rm if} \; x < 0
\end{array} \right.
\] $$
where $icode x$$ and $icode y$$ have the same prototype as above.
For example, see
$cref/base_alloc/base_alloc.hpp/sign/$$.
Note that, if ordered comparisons are not defined for the type $icode Base$$,
the $code code sign$$ function should generate an assert if it is used; see
$cref/complex invalid unary math/base_complex.hpp/Invalid Unary Math/$$.

$head pow$$
The type $icode Base$$ must support the syntax
$codei%
	%z% = CppAD::pow(%x%, %y%)
%$$
which computes $latex z = x^y$$.
The arguments $icode x$$ and $icode y$$ have prototypes
$codei%
	const %Base%& %x%
	const %Base%& %y%
%$$
and the return value $icode z$$ has prototype
$codei%
	%Base% %z%
%$$
For example, see
$cref/base_alloc/base_alloc.hpp/pow/$$.


$head isnan$$
If $icode Base$$ defines the $code isnan$$ function,
you may also have to provide a definition in the CppAD namespace
(to avoid a function ambiguity).
For example, see
$cref/base_complex/base_complex.hpp/isnan/$$.


$end
-------------------------------------------------------------------------------
*/

# include <cmath>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE

/*!
\file base_std_math.hpp
Defintions that aid meeting Base type requirements for standard math functions.
*/

/*!
\def CPPAD_STANDARD_MATH_UNARY(Type, Fun)
This macro defines the function
\verbatim
	y = CppAD:Fun(x)
\endverbatim
where the argument \c x and return value \c y have type \c Type
using the corresponding function <code>std::Fun</code>.
*/
# define CPPAD_STANDARD_MATH_UNARY(Type, Fun) \
	inline Type Fun(const Type& x)            \
	{	return std::Fun(x); }

} // END_CPPAD_NAMESPACE

# endif
