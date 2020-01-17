# ifndef CPPAD_CORE_LOG1P_HPP
# define CPPAD_CORE_LOG1P_HPP

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
$begin log1p$$
$spell
	CppAD
$$

$section The Logarithm of One Plus Argument: log1p$$

$head Syntax$$
$icode%y% = log1p(%x%)%$$

$head Description$$
Returns the value of the logarithm of one plus argument which is defined
by $icode%y% == log(1 + %x%)%$$.

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
\R{log1p} (x) = \log(1 + x)
\] $$
to compute this function.

$head Example$$
$children%
	example/general/log1p.cpp
%$$
The file
$cref log1p.cpp$$
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
Type log1p_template(const Type &x)
{	return CppAD::log(Type(1) + x);
}

inline float log1p(const float &x)
{	return log1p_template(x); }

inline double log1p(const double &x)
{	return log1p_template(x); }

template <class Base>
inline AD<Base> log1p(const AD<Base> &x)
{	return log1p_template(x); }

template <class Base>
inline AD<Base> log1p(const VecAD_reference<Base> &x)
{	return log1p_template( x.ADBase() ); }


} // END CppAD namespace

# endif // CPPAD_USE_CPLUSPLUS_2011
# endif // CPPAD_LOG1P_INCLUDED
