# ifndef CPPAD_CORE_BOOL_FUN_HPP
# define CPPAD_CORE_BOOL_FUN_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin BoolFun$$
$spell
	namespace
	bool
	CppAD
	const
$$


$section AD Boolean Functions$$
$mindex bool CPPAD_BOOL_UNARY CPPAD_BOOL_BINARY$$

$head Syntax$$
$codei%CPPAD_BOOL_UNARY(%Base%, %unary_name%)
%$$
$icode%b% = %unary_name%(%u%)
%$$
$icode%b% = %unary_name%(%x%)
%$$
$codei%CPPAD_BOOL_BINARY(%Base%, %binary_name%)
%$$
$icode%b% = %binary_name%(%u%, %v%)
%$$
$icode%b% = %binary_name%(%x%, %y%)%$$


$head Purpose$$
Create a $code bool$$ valued function that has $codei%AD<%Base%>%$$ arguments.

$head unary_name$$
This is the name of the $code bool$$ valued function with one argument
(as it is used in the source code).
The user must provide a version of $icode unary_name$$ where
the argument has type $icode Base$$.
CppAD uses this to create a version of $icode unary_name$$ where the
argument has type $codei%AD<%Base%>%$$.

$head u$$
The argument $icode u$$ has prototype
$codei%
	const %Base% &%u%
%$$
It is the value at which the user provided version of $icode unary_name$$
is to be evaluated.
It is also used for the first argument to the
user provided version of $icode binary_name$$.

$head x$$
The argument $icode x$$ has prototype
$codei%
	const AD<%Base%> &%x%
%$$
It is the value at which the CppAD provided version of $icode unary_name$$
is to be evaluated.
It is also used for the first argument to the
CppAD provided version of $icode binary_name$$.

$head b$$
The result $icode b$$ has prototype
$codei%
	bool %b%
%$$

$head Create Unary$$
The preprocessor macro invocation
$codei%
	CPPAD_BOOL_UNARY(%Base%, %unary_name%)
%$$
defines the version of $icode unary_name$$ with a $codei%AD<%Base%>%$$
argument.
This can with in a namespace
(not the $code CppAD$$ namespace)
but must be outside of any routine.

$head binary_name$$
This is the name of the $code bool$$ valued function with two arguments
(as it is used in the source code).
The user must provide a version of $icode binary_name$$ where
the arguments have type $icode Base$$.
CppAD uses this to create a version of $icode binary_name$$ where the
arguments have type $codei%AD<%Base%>%$$.

$head v$$
The argument $icode v$$ has prototype
$codei%
	const %Base% &%v%
%$$
It is the second argument to
the user provided version of $icode binary_name$$.

$head y$$
The argument $icode x$$ has prototype
$codei%
	const AD<%Base%> &%y%
%$$
It is the second argument to
the CppAD provided version of $icode binary_name$$.

$head Create Binary$$
The preprocessor macro invocation
$codei%
	CPPAD_BOOL_BINARY(%Base%, %binary_name%)
%$$
defines the version of $icode binary_name$$ with $codei%AD<%Base%>%$$
arguments.
This can with in a namespace
(not the $code CppAD$$ namespace)
but must be outside of any routine.


$head Operation Sequence$$
The result of this operation is not an
$cref/AD of Base/glossary/AD of Base/$$ object.
Thus it will not be recorded as part of an
AD of $icode Base$$
$cref/operation sequence/glossary/Operation/Sequence/$$.

$head Example$$
$children%
	example/general/bool_fun.cpp
%$$
The file
$cref bool_fun.cpp$$
contains an example and test of these operations.
It returns true if it succeeds and false otherwise.

$head Deprecated 2007-07-31$$
The preprocessor symbols $code CppADCreateUnaryBool$$
and $code CppADCreateBinaryBool$$ are defined to be the same as
$code CPPAD_BOOL_UNARY$$ and $code CPPAD_BOOL_BINARY$$ respectively
(but their use is deprecated).

$end
*/

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
/*!
\file bool_fun.hpp
Routines and macros that implement functions from AD<Base> to bool.
*/

/*!
Macro that defines a unary function <tt>bool F(AD<Base> x)</tt>
using <tt>bool F(Base x)</tt>.

\param Base
base for the AD type of arguments to this unary bool valued function.

\param unary_name
name of this unary function; i.e., \c F.
*/
# define CPPAD_BOOL_UNARY(Base, unary_name)                        \
     inline bool unary_name (const CppAD::AD<Base> &x)             \
     {                                                             \
          return CppAD::AD<Base>::UnaryBool(unary_name, x);        \
     }

/*!
Deprecated name for CPPAD_BOOL_UNARY
*/
# define CppADCreateUnaryBool  CPPAD_BOOL_UNARY

/*!
Link a function name, and AD value pair to function call with base argument
and bool retrun value.

\param FunName
is the name of the function that we are linking.

\param x
is the argument where we are evaluating the function.
*/
template <class Base>
inline bool AD<Base>::UnaryBool(
	bool FunName(const Base &x),
	const AD<Base> &x
)
{
	return FunName(x.value_);
}

/*!
Macro that defines a binary function <tt>bool F(AD<Base> x, AD<Base> y)</tt>
using <tt>bool F(Base x, Base y)</tt>.

\param Base
base for the AD type of arguments to this binary bool valued function.

\param binary_name
name of this binary function; i.e., \c F.
*/

# define CPPAD_BOOL_BINARY(Base, binary_name)                      \
     inline bool binary_name (                                     \
          const CppAD::AD<Base> &x, const CppAD::AD<Base> &y)      \
     {                                                             \
          return CppAD::AD<Base>::BinaryBool(binary_name, x, y);   \
     }
/*!
Deprecated name for CPPAD_BOOL_BINARY
*/
# define CppADCreateBinaryBool CPPAD_BOOL_BINARY


/*!
Link a function name, and two AD values to function call with base arguments
and bool retrun value.

\param FunName
is the name of the function that we are linking.

\param x
is the first argument where we are evaluating the function at.

\param y
is the second argument where we are evaluating the function at.
*/
template <class Base>
inline bool AD<Base>::BinaryBool(
	bool FunName(const Base &x, const Base &y),
	const AD<Base> &x, const AD<Base> &y
)
{
	return FunName(x.value_, y.value_);
}

} // END_CPPAD_NAMESPACE
# endif
