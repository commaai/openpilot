# ifndef CPPAD_CORE_AD_IO_HPP
# define CPPAD_CORE_AD_IO_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin ad_input$$
$spell
	VecAD
	std
	istream
	const
$$


$section AD Output Stream Operator$$
$mindex >> input write$$

$head Syntax$$
$icode%is% >> %x%$$


$head Purpose$$
Sets $icode x$$ to a $cref/parameter/glossary/Parameter/$$
with value $icode b$$ corresponding to
$codei%
	%is% >> %b%
%$$
where $icode b$$ is a $icode Base$$ object.
It is assumed that this $icode Base$$ input operation returns
a reference to $icode is$$.

$head is$$
The operand $icode is$$ has prototype
$codei%
	std::istream& %is%
%$$

$head x$$
The operand $icode x$$ has one of the following prototypes
$codei%
	AD<%Base%>&               %x%
%$$

$head Result$$
The result of this operation can be used as a reference to $icode is$$.
For example, if the operand $icode y$$ has prototype
$codei%
	AD<%Base%> %y%
%$$
then the syntax
$codei%
	%is% >> %x% >> %y%
%$$
will first read the $icode Base$$ value of $icode x$$ from $icode is$$,
and then read the $icode Base$$ value to $icode y$$.

$head Operation Sequence$$
The result of this operation is not an
$cref/AD of Base/glossary/AD of Base/$$ object.
Thus it will not be recorded as part of an
AD of $icode Base$$
$cref/operation sequence/glossary/Operation/Sequence/$$.

$head Example$$
$children%
	example/general/ad_input.cpp
%$$
The file
$cref ad_input.cpp$$
contains an example and test of this operation.
It returns true if it succeeds and false otherwise.

$end
------------------------------------------------------------------------------
$begin ad_output$$
$spell
	VecAD
	std
	ostream
	const
$$


$section AD Output Stream Operator$$
$mindex <<$$

$head Syntax$$
$icode%os% << %x%$$


$head Purpose$$
Writes the $icode Base$$ value, corresponding to $icode x$$,
to the output stream $icode os$$.

$head Assumption$$
If $icode b$$ is a $icode Base$$ object,
$codei%
	%os% << %b%
%$$
returns a reference to $icode os$$.

$head os$$
The operand $icode os$$ has prototype
$codei%
	std::ostream& %os%
%$$

$head x$$
The operand $icode x$$ has one of the following prototypes
$codei%
	const AD<%Base%>&               %x%
	const VecAD<%Base%>::reference& %x%
%$$

$head Result$$
The result of this operation can be used as a reference to $icode os$$.
For example, if the operand $icode y$$ has prototype
$codei%
	AD<%Base%> %y%
%$$
then the syntax
$codei%
	%os% << %x% << %y%
%$$
will output the value corresponding to $icode x$$
followed by the value corresponding to $icode y$$.

$head Operation Sequence$$
The result of this operation is not an
$cref/AD of Base/glossary/AD of Base/$$ object.
Thus it will not be recorded as part of an
AD of $icode Base$$
$cref/operation sequence/glossary/Operation/Sequence/$$.

$head Example$$
$children%
	example/general/ad_output.cpp
%$$
The file
$cref ad_output.cpp$$
contains an example and test of this operation.
It returns true if it succeeds and false otherwise.

$end
------------------------------------------------------------------------------
*/
namespace CppAD { // BEGIN_CPPAD_NAMESPACE
/*!
\file ad_io.hpp
AD<Base> input and ouput stream operators.
*/
// ---------------------------------------------------------------------------
/*!
Read an AD<Base> object from an input stream.

\tparam Base
Base type for the AD object.

\param is [in,out]
Is the input stream from which that value is read.

\param x [out]
is the object that is being set to a value.
Upone return, x.value_ is read from the input stream
and x.tape_is_ is zero; i.e., x is a parameter.
*/
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
std::istream& operator >> (std::istream& is, AD<Base>& x)
{	// like assignment to a base type value
	x.tape_id_ = 0;
	CPPAD_ASSERT_UNKNOWN( Parameter(x) );
	return (is >> x.value_);
}
// ---------------------------------------------------------------------------
/*!
Write an AD<Base> object to an output stream.

\tparam Base
Base type for the AD object.

\param os [in,out]
Is the output stream to which that value is written.

\param x
is the object that is being written to the output stream.
This is equivalent to writing x.value_ to the output stream.
*/
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
std::ostream& operator << (std::ostream &os, const AD<Base> &x)
{	return (os << x.value_); }
// ---------------------------------------------------------------------------
/*!
Write a VecAD_reference<Base> object to an output stream.

\tparam Base
Base type for the VecAD_reference object.

\param os [in,out]
Is the output stream to which that value is written.

\param x
is the element of the VecAD object that is being written to the output stream.
This is equivalent to writing the corresponing Base value to the stream.
*/
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
std::ostream& operator << (std::ostream &os, const VecAD_reference<Base> &x)
{	return (os << x.ADBase()); }

} // END_CPPAD_NAMESPACE
# endif
