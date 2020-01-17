# ifndef CPPAD_CORE_EXPM1_HPP
# define CPPAD_CORE_EXPM1_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
-------------------------------------------------------------------------------
$begin expm1$$
$spell
	exp
	expm1
	const
	Vec
	std
	cmath
	CppAD
$$
$section The Exponential Function Minus One: expm1$$

$head Syntax$$
$icode%y% = expm1(%x%)%$$

$head Description$$
Returns the value of the exponential function minus one which is defined
by $icode%y% == exp(%x%) - 1%$$.

$head x, y$$
See the $cref/possible types/unary_standard_math/Possible Types/$$
for a unary standard math function.

$head CPPAD_USE_CPLUSPLUS_2011$$

$subhead true$$
If this preprocessor symbol is true ($code 1$$),
and $icode x$$ is an AD type,
this is an $cref/atomic operation/glossary/Operation/Atomic/$$.

$subhead false$$
If this preprocessor symbol is false ($code 0$$),
CppAD uses the representation
$latex \[
\R{expm1} (x) = \exp(x) - 1
\] $$
to compute this function.

$head Example$$
$children%
	example/general/expm1.cpp
%$$
The file
$cref expm1.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$end
-------------------------------------------------------------------------------
*/
# include <cppad/configure.hpp>
# if ! CPPAD_USE_CPLUSPLUS_2011

// BEGIN CppAD namespace
namespace CppAD {

template <class Type>
Type expm1_template(const Type &x)
{	return CppAD::exp(x) - Type(1);
}

inline float expm1(const float &x)
{	return expm1_template(x); }

inline double expm1(const double &x)
{	return expm1_template(x); }

template <class Base>
inline AD<Base> expm1(const AD<Base> &x)
{	return expm1_template(x); }

template <class Base>
inline AD<Base> expm1(const VecAD_reference<Base> &x)
{	return expm1_template( x.ADBase() ); }


} // END CppAD namespace

# endif // CPPAD_USE_CPLUSPLUS_2011
# endif // CPPAD_EXPM1_INCLUDED
