# ifndef CPPAD_CORE_ERF_HPP
# define CPPAD_CORE_ERF_HPP

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
$begin erf$$
$spell
	erf
	const
	Vec
	std
	cmath
	CppAD
	Vedder
$$
$section The Error Function$$

$head Syntax$$
$icode%y% = erf(%x%)%$$

$head Description$$
Returns the value of the error function which is defined by
$latex \[
{\rm erf} (x) = \frac{2}{ \sqrt{\pi} } \int_0^x \exp( - t * t ) \; {\bf d} t
\] $$

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
CppAD uses a fast approximation (few numerical operations)
with relative error bound $latex 4 \times 10^{-4}$$; see
Vedder, J.D.,
$icode Simple approximations for the error function and its inverse$$,
American Journal of Physics,
v 55,
n 8,
1987,
p 762-3.

$head Example$$
$children%
	example/general/erf.cpp
%$$
The file
$cref erf.cpp$$
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
Type erf_template(const Type &x)
{	using CppAD::exp;
	const Type a = static_cast<Type>(993./880.);
	const Type b = static_cast<Type>(89./880.);

	return tanh( (a + b * x * x) * x );
}

inline float erf(const float &x)
{	return erf_template(x); }

inline double erf(const double &x)
{	return erf_template(x); }

template <class Base>
inline AD<Base> erf(const AD<Base> &x)
{	return erf_template(x); }

template <class Base>
inline AD<Base> erf(const VecAD_reference<Base> &x)
{	return erf_template( x.ADBase() ); }


} // END CppAD namespace

# endif // CPPAD_USE_CPLUSPLUS_2011
# endif // CPPAD_ERF_INCLUDED
