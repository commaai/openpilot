# ifndef CPPAD_CORE_NEAR_EQUAL_EXT_HPP
# define CPPAD_CORE_NEAR_EQUAL_EXT_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin NearEqualExt$$
$spell
	cout
	endl
	Microsoft
	std
	Cpp
	namespace
	const
	bool
$$

$section Compare AD and Base Objects for Nearly Equal$$
$mindex NearEqual with$$


$head Syntax$$
$icode%b% = NearEqual(%x%, %y%, %r%, %a%)%$$


$head Purpose$$
The routine $cref NearEqual$$ determines if two objects of
the same type are nearly.
This routine is extended to the case where one object can have type
$icode Type$$ while the other can have type
$codei%AD<%Type%>%$$ or
$codei%AD< std::complex<%Type%> >%$$.

$head x$$
The arguments $icode x$$
has one of the following possible prototypes:
$codei%
	const %Type%                     &%x%
	const AD<%Type%>                 &%x%
	const AD< std::complex<%Type%> > &%x%
%$$

$head y$$
The arguments $icode y$$
has one of the following possible prototypes:
$codei%
	const %Type%                     &%y%
	const AD<%Type%>                 &%y%
	const AD< std::complex<%Type%> > &%x%
%$$


$head r$$
The relative error criteria $icode r$$ has prototype
$codei%
	const %Type% &%r%
%$$
It must be greater than or equal to zero.
The relative error condition is defined as:
$latex \[
	\frac{ | x - y | } { |x| + |y| } \leq r
\] $$

$head a$$
The absolute error criteria $icode a$$ has prototype
$codei%
	const %Type% &%a%
%$$
It must be greater than or equal to zero.
The absolute error condition is defined as:
$latex \[
	| x - y | \leq a
\] $$

$head b$$
The return value $icode b$$ has prototype
$codei%
	bool %b%
%$$
If either $icode x$$ or $icode y$$ is infinite or not a number,
the return value is false.
Otherwise, if either the relative or absolute error
condition (defined above) is satisfied, the return value is true.
Otherwise, the return value is false.

$head Type$$
The type $icode Type$$ must be a
$cref NumericType$$.
The routine $cref CheckNumericType$$ will generate
an error message if this is not the case.
If $icode a$$ and $icode b$$ have type $icode Type$$,
the following operation must be defined
$table
$bold Operation$$     $cnext
	$bold Description$$ $rnext
$icode%a% <= %b%$$  $cnext
	less that or equal operator (returns a $code bool$$ object)
$tend

$head Operation Sequence$$
The result of this operation is not an
$cref/AD of Base/glossary/AD of Base/$$ object.
Thus it will not be recorded as part of an
AD of $icode Base$$
$cref/operation sequence/glossary/Operation/Sequence/$$.

$head Example$$
$children%
	example/general/near_equal_ext.cpp
%$$
The file $cref near_equal_ext.cpp$$ contains an example
and test of this extension of $cref NearEqual$$.
It return true if it succeeds and false otherwise.

$end

*/
// BEGIN CppAD namespace
namespace CppAD {
// ------------------------------------------------------------------------

// fold into base type and then use <cppad/near_equal.hpp>
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool NearEqual(
const AD<Base> &x, const AD<Base> &y, const Base &r, const Base &a)
{	return NearEqual(x.value_, y.value_, r, a);
}

template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool NearEqual(
const Base &x, const AD<Base> &y, const Base &r, const Base &a)
{	return NearEqual(x, y.value_, r, a);
}

template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool NearEqual(
const AD<Base> &x, const Base &y, const Base &r, const Base &a)
{	return NearEqual(x.value_, y, r, a);
}

// fold into AD type and then use cases above
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool NearEqual(
	const VecAD_reference<Base> &x, const VecAD_reference<Base> &y,
	const Base &r, const Base &a)
{	return NearEqual(x.ADBase(), y.ADBase(), r, a);
}
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool NearEqual(const VecAD_reference<Base> &x, const AD<Base> &y,
	const Base &r, const Base &a)
{	return NearEqual(x.ADBase(), y, r, a);
}
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool NearEqual(const VecAD_reference<Base> &x, const Base &y,
	const Base &r, const Base &a)
{	return NearEqual(x.ADBase(), y, r, a);
}
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool NearEqual(const AD<Base> &x, const VecAD_reference<Base> &y,
	const Base &r, const Base &a)
{	return NearEqual(x, y.ADBase(), r, a);
}
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool NearEqual(const Base &x, const VecAD_reference<Base> &y,
	const Base &r, const Base &a)
{	return NearEqual(x, y.ADBase(), r, a);
}

} // END CppAD namespace

# endif
