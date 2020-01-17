// $Id$
# ifndef CPPAD_CORE_BASE_COND_EXP_HPP
# define CPPAD_CORE_BASE_COND_EXP_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin base_cond_exp$$
$spell
	alloc
	Rel
	hpp
	enum
	namespace
	Op
	Lt
	Le
	Eq
	Ge
	Gt
	Ne
	cond
	exp
	const
	adolc
	CppAD
	inline
$$

$section Base Type Requirements for Conditional Expressions$$
$mindex CondExp require CPPAD_COND_EXP_REL$$

$head Purpose$$
These definitions are required by the user's code to support the
$codei%AD<%Base%>%$$ type for $cref CondExp$$ operations:

$head CompareOp$$
The following $code enum$$ type is used in the specifications below:
$codep
namespace CppAD {
	// The conditional expression operator enum type
	enum CompareOp
	{	CompareLt, // less than
		CompareLe, // less than or equal
		CompareEq, // equal
		CompareGe, // greater than or equal
		CompareGt, // greater than
		CompareNe  // not equal
	};
}
$$

$head CondExpTemplate$$
The type $icode Base$$ must support the syntax
$codei%
	%result% = CppAD::CondExpOp(
		%cop%, %left%, %right%, %exp_if_true%, %exp_if_false%
	)
%$$
which computes implements the corresponding $cref CondExp$$
function when the result has prototype
$codei%
	%Base% %result%
%$$
The argument $icode cop$$ has prototype
$codei%
	enum CppAD::CompareOp %cop%
%$$
The other arguments have the prototype
$codei%
	const %Base%&  %left%
	const %Base%&  %right%
	const %Base%&  %exp_if_true%
	const %Base%&  %exp_if_false%
%$$

$subhead Ordered Type$$
If $icode Base$$ is a relatively simple type
that supports
$code <$$, $code <=$$, $code ==$$, $code >=$$, and $code >$$ operators
its $code CondExpOp$$ function can be defined by
$codei%
namespace CppAD {
	inline %Base% CondExpOp(
	enum CppAD::CompareOp  cop            ,
	const %Base%           &left          ,
	const %Base%           &right         ,
	const %Base%           &exp_if_true   ,
	const %Base%           &exp_if_false  )
	{	return CondExpTemplate(
			cop, left, right, trueCase, falseCase);
	}
}
%$$
For example, see
$cref/double CondExpOp/base_alloc.hpp/CondExpOp/$$.
For an example of and implementation of $code CondExpOp$$ with
a more involved $icode Base$$ type see
$cref/adolc CondExpOp/base_adolc.hpp/CondExpOp/$$.


$subhead Not Ordered$$
If the type $icode Base$$ does not support ordering,
the $code CondExpOp$$ function does not make sense.
In this case one might (but need not) define $code CondExpOp$$ as follows:
$codei%
namespace CppAD {
	inline %Base% CondExpOp(
	enum CompareOp cop           ,
	const %Base%   &left         ,
	const %Base%   &right        ,
	const %Base%   &exp_if_true  ,
	const %Base%   &exp_if_false )
	{	// attempt to use CondExp with a %Base% argument
		assert(0);
		return %Base%(0);
	}
}
%$$
For example, see
$cref/complex CondExpOp/base_complex.hpp/CondExpOp/$$.

$head CondExpRel$$
The macro invocation
$codei%
	CPPAD_COND_EXP_REL(%Base%)
%$$
uses $code CondExpOp$$ above to define the following functions
$codei%
	CondExpLt(%left%, %right%, %exp_if_true%, %exp_if_false%)
	CondExpLe(%left%, %right%, %exp_if_true%, %exp_if_false%)
	CondExpEq(%left%, %right%, %exp_if_true%, %exp_if_false%)
	CondExpGe(%left%, %right%, %exp_if_true%, %exp_if_false%)
	CondExpGt(%left%, %right%, %exp_if_true%, %exp_if_false%)
%$$
where the arguments have type $icode Base$$.
This should be done inside of the CppAD namespace.
For example, see
$cref/base_alloc/base_alloc.hpp/CondExpRel/$$.

$end
*/

namespace CppAD { // BEGIN_CPPAD_NAMESPACE

/*!
\file base_cond_exp.hpp
CondExp operations that aid in meeting Base type requirements.
*/

/*!
\def CPPAD_COND_EXP_BASE_REL(Type, Rel, Op)
This macro defines the operation
\verbatim
	CondExpRel(left, right, exp_if_true, exp_if_false)
\endverbatim
The argument \c Type is the \c Base type for this base require operation.
The argument \c Rel is one of \c Lt, \c Le, \c Eq, \c Ge, \c Gt.
The argument \c Op is the corresponding \c CompareOp value.
*/
# define CPPAD_COND_EXP_BASE_REL(Type, Rel, Op)       \
	inline Type CondExp##Rel(                        \
		const Type& left      ,                     \
		const Type& right     ,                     \
		const Type& exp_if_true  ,                  \
		const Type& exp_if_false )                  \
	{	return CondExpOp(Op, left, right, exp_if_true, exp_if_false); \
	}

/*!
\def CPPAD_COND_EXP_REL(Type)
The macro defines the operations
\verbatim
	CondExpLt(left, right, exp_if_true, exp_if_false)
	CondExpLe(left, right, exp_if_true, exp_if_false)
	CondExpEq(left, right, exp_if_true, exp_if_false)
	CondExpGe(left, right, exp_if_true, exp_if_false)
	CondExpGt(left, right, exp_if_true, exp_if_false)
\endverbatim
The argument \c Type is the \c Base type for this base require operation.
*/
# define CPPAD_COND_EXP_REL(Type)                     \
	CPPAD_COND_EXP_BASE_REL(Type, Lt, CompareLt)     \
	CPPAD_COND_EXP_BASE_REL(Type, Le, CompareLe)     \
	CPPAD_COND_EXP_BASE_REL(Type, Eq, CompareEq)     \
	CPPAD_COND_EXP_BASE_REL(Type, Ge, CompareGe)     \
	CPPAD_COND_EXP_BASE_REL(Type, Gt, CompareGt)

/*!
Template function to implement Conditional Expressions for simple types
that have comparision operators.

\tparam CompareType
is the type of the left and right operands to the comparision operator.

\tparam ResultType
is the type of the result, which is the same as \c CompareType except
during forward and reverse mode sparese calculations.

\param cop
specifices which comparision to use; i.e.,
$code <$$,
$code <=$$,
$code ==$$,
$code >=$$,
$code >$$, or
$code !=$$.

\param left
is the left operand to the comparision operator.

\param right
is the right operand to the comparision operator.

\param exp_if_true
is the return value is the comparision results in true.

\param exp_if_false
is the return value is the comparision results in false.

\return
see \c exp_if_true and \c exp_if_false above.
*/
template <class CompareType, class ResultType>
ResultType CondExpTemplate(
	enum  CompareOp            cop          ,
	const CompareType&         left         ,
	const CompareType&         right        ,
	const ResultType&          exp_if_true  ,
	const ResultType&          exp_if_false )
{	ResultType returnValue;
	switch( cop )
	{
		case CompareLt:
		if( left < right )
			returnValue = exp_if_true;
		else	returnValue = exp_if_false;
		break;

		case CompareLe:
		if( left <= right )
			returnValue = exp_if_true;
		else	returnValue = exp_if_false;
		break;

		case CompareEq:
		if( left == right )
			returnValue = exp_if_true;
		else	returnValue = exp_if_false;
		break;

		case CompareGe:
		if( left >= right )
			returnValue = exp_if_true;
		else	returnValue = exp_if_false;
		break;

		case CompareGt:
		if( left > right )
			returnValue = exp_if_true;
		else	returnValue = exp_if_false;
		break;

		default:
		CPPAD_ASSERT_UNKNOWN(0);
		returnValue = exp_if_true;
	}
	return returnValue;
}

} // END_CPPAD_NAMESPACE
# endif
