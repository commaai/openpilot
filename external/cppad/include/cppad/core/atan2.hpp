# ifndef CPPAD_CORE_ATAN2_HPP
# define CPPAD_CORE_ATAN2_HPP

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
$begin atan2$$
$spell
	Vec
	CppAD
	namespace
	std
	atan
	const
$$


$section AD Two Argument Inverse Tangent Function$$
$mindex tan atan2$$

$head Syntax$$
$icode%theta% = atan2(%y%, %x%)%$$


$head Purpose$$
Determines an angle $latex \theta \in [ - \pi , + \pi ]$$
such that
$latex \[
\begin{array}{rcl}
	\sin ( \theta )  & = & y / \sqrt{ x^2 + y^2 }  \\
	\cos ( \theta )  & = & x / \sqrt{ x^2 + y^2 }
\end{array}
\] $$

$head y$$
The argument $icode y$$ has one of the following prototypes
$codei%
	const AD<%Base%>               &%y%
	const VecAD<%Base%>::reference &%y%
%$$

$head x$$
The argument $icode x$$ has one of the following prototypes
$codei%
	const AD<%Base%>               &%x%
	const VecAD<%Base%>::reference &%x%
%$$

$head theta$$
The result $icode theta$$ has prototype
$codei%
	AD<%Base%> %theta%
%$$

$head Operation Sequence$$
The AD of $icode Base$$
operation sequence used to calculate $icode theta$$ is
$cref/independent/glossary/Operation/Independent/$$
of $icode x$$ and $icode y$$.

$head Example$$
$children%
	example/general/atan2.cpp
%$$
The file
$cref atan2.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$end
-------------------------------------------------------------------------------
*/

namespace CppAD { // BEGIN CppAD namespace

inline float atan2(float x, float y)
{	return std::atan2(x, y); }

inline double atan2(double x, double y)
{	return std::atan2(x, y); }

// The code below is used as an example by the CondExp documentation.
// BEGIN CondExp
template <class Base>
AD<Base> atan2 (const AD<Base> &y, const AD<Base> &x)
{	AD<Base> alpha;
	AD<Base> beta;
	AD<Base> theta;

	AD<Base> zero(0.);
	AD<Base> pi2(2. * atan(1.));
	AD<Base> pi(2. * pi2);

	AD<Base> ax = fabs(x);
	AD<Base> ay = fabs(y);

	// if( ax > ay )
	//	theta = atan(ay / ax);
	// else	theta = pi2 - atan(ax / ay);
	alpha = atan(ay / ax);
	beta  = pi2 - atan(ax / ay);
	theta = CondExpGt(ax, ay, alpha, beta);         // use of CondExp

	// if( x <= 0 )
	//	theta = pi - theta;
	theta = CondExpLe(x, zero, pi - theta, theta);  // use of CondExp

	// if( y <= 0 )
	//	theta = - theta;
	theta = CondExpLe(y, zero, -theta, theta);      // use of CondExp

	return theta;
}
// END CondExp

template <class Base>
inline AD<Base> atan2 (const VecAD_reference<Base> &y, const AD<Base> &x)
{	return atan2( y.ADBase() , x ); }

template <class Base>
inline AD<Base> atan2 (const AD<Base> &y, const VecAD_reference<Base> &x)
{	return atan2( y , x.ADBase() ); }

template <class Base>
inline AD<Base> atan2
(const VecAD_reference<Base> &y, const VecAD_reference<Base> &x)
{	return atan2( y.ADBase() , x.ADBase() ); }

} // END CppAD namespace

# endif
