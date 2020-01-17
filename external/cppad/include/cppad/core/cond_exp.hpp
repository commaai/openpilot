# ifndef CPPAD_CORE_COND_EXP_HPP
# define CPPAD_CORE_COND_EXP_HPP

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
$begin CondExp$$
$spell
	Atan2
	CondExp
	Taylor
	std
	Cpp
	namespace
	inline
	const
	abs
	Rel
	bool
	Lt
	Le
	Eq
	Ge
	Gt
$$


$section AD Conditional Expressions$$
$mindex assign$$

$head Syntax$$
$icode%result% = CondExp%Rel%(%left%, %right%, %if_true%, %if_false%)%$$


$head Purpose$$
Record,
as part of an AD of $icode Base$$
$cref/operation sequence/glossary/Operation/Sequence/$$,
the conditional result
$codei%
	if( %left% %Cop% %right% )
		%result% = %if_true%
	else	%result% = %if_false%
%$$
The relational $icode Rel$$ and comparison operator $icode Cop$$
above have the following correspondence:
$codei%
	%Rel%   Lt   Le   Eq   Ge   Gt
	%Cop%    <   <=   ==   >=   >
%$$
If $icode f$$ is the $cref ADFun$$ object corresponding to the
AD operation sequence,
the assignment choice for $icode result$$
in an AD conditional expression is made each time
$cref/f.Forward/Forward/$$ is used to evaluate the zero order Taylor
coefficients with new values for the
$cref/independent variables/glossary/Tape/Independent Variable/$$.
This is in contrast to the $cref/AD comparison operators/Compare/$$
which are boolean valued and not included in the AD operation sequence.

$head Rel$$
In the syntax above, the relation $icode Rel$$ represents one of the following
two characters: $code Lt$$, $code Le$$, $code Eq$$, $code Ge$$, $code Gt$$.
As in the table above,
$icode Rel$$ determines which comparison operator $icode Cop$$ is used
when comparing $icode left$$ and $icode right$$.

$head Type$$
These functions are defined in the CppAD namespace for arguments of
$icode Type$$ is $code float$$ , $code double$$, or any type of the form
$codei%AD<%Base%>%$$.
(Note that all four arguments must have the same type.)

$head left$$
The argument $icode left$$ has prototype
$codei%
	const %Type%& %left%
%$$
It specifies the value for the left side of the comparison operator.

$head right$$
The argument $icode right$$ has prototype
$codei%
	const %Type%& %right%
%$$
It specifies the value for the right side of the comparison operator.

$head if_true$$
The argument $icode if_true$$ has prototype
$codei%
	const %Type%& %if_true%
%$$
It specifies the return value if the result of the comparison is true.

$head if_false$$
The argument $icode if_false$$ has prototype
$codei%
	const %Type%& %if_false%
%$$
It specifies the return value if the result of the comparison is false.

$head result$$
The $icode result$$ has prototype
$codei%
	%Type%& %if_false%
%$$

$head Optimize$$
The $cref optimize$$ method will optimize conditional expressions
in the following way:
During $cref/zero order forward mode/forward_zero/$$,
once the value of the $icode left$$ and $icode right$$ have been determined,
it is known if the true or false case is required.
From this point on, values corresponding to the case that is not required
are not computed.
This optimization is done for the rest of zero order forward mode
as well as forward and reverse derivatives calculations.

$head Deprecate 2005-08-07$$
Previous versions of CppAD used
$codei%
	CondExp(%flag%, %if_true%, %if_false%)
%$$
for the same meaning as
$codei%
	CondExpGt(%flag%, %Type%(0), %if_true%, %if_false%)
%$$
Use of $code CondExp$$ is deprecated, but continues to be supported.

$head Operation Sequence$$
This is an AD of $icode Base$$
$cref/atomic operation/glossary/Operation/Atomic/$$
and hence is part of the current
AD of $icode Base$$
$cref/operation sequence/glossary/Operation/Sequence/$$.


$head Example$$

$head Test$$
$children%
	example/general/cond_exp.cpp
%$$
The file
$cref cond_exp.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$head Atan2$$
The following implementation of the
AD $cref atan2$$ function is a more complex
example of using conditional expressions:
$code
$srcfile%cppad/core/atan2.hpp%0%BEGIN CondExp%// END CondExp%$$
$$


$end
-------------------------------------------------------------------------------
*/
//  BEGIN CppAD namespace
namespace CppAD {

template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
AD<Base> CondExpOp(
	enum  CompareOp cop       ,
	const AD<Base> &left      ,
	const AD<Base> &right     ,
	const AD<Base> &if_true   ,
	const AD<Base> &if_false  )
{
	AD<Base> returnValue;
	CPPAD_ASSERT_UNKNOWN( Parameter(returnValue) );

	// check first case where do not need to tape
	if( IdenticalPar(left) & IdenticalPar(right) )
	{	switch( cop )
		{
			case CompareLt:
			if( left.value_ < right.value_ )
				returnValue = if_true;
			else	returnValue = if_false;
			break;

			case CompareLe:
			if( left.value_ <= right.value_ )
				returnValue = if_true;
			else	returnValue = if_false;
			break;

			case CompareEq:
			if( left.value_ == right.value_ )
				returnValue = if_true;
			else	returnValue = if_false;
			break;

			case CompareGe:
			if( left.value_ >= right.value_ )
				returnValue = if_true;
			else	returnValue = if_false;
			break;

			case CompareGt:
			if( left.value_ > right.value_ )
				returnValue = if_true;
			else	returnValue = if_false;
			break;

			default:
			CPPAD_ASSERT_UNKNOWN(0);
			returnValue = if_true;
		}
		return returnValue;
	}

	// must use CondExp incase Base is an AD type and recording
	returnValue.value_ = CondExpOp(cop,
		left.value_, right.value_, if_true.value_, if_false.value_);

	local::ADTape<Base> *tape = CPPAD_NULL;
	if( Variable(left) )
		tape = left.tape_this();
	if( Variable(right) )
		tape = right.tape_this();
	if( Variable(if_true) )
		tape = if_true.tape_this();
	if( Variable(if_false) )
		tape = if_false.tape_this();

	// add this operation to the tape
	if( tape != CPPAD_NULL )
		tape->RecordCondExp(cop,
			returnValue, left, right, if_true, if_false);

	return returnValue;
}

// --- RecordCondExp(cop, returnValue, left, right, if_true, if_false) -----

/// All these operations are done in \c Rec_, so we should move this
/// routine to <tt>recorder<Base></tt>.
template <class Base>
void local::ADTape<Base>::RecordCondExp(
	enum CompareOp  cop         ,
	AD<Base>       &returnValue ,
	const AD<Base> &left        ,
	const AD<Base> &right       ,
	const AD<Base> &if_true     ,
	const AD<Base> &if_false    )
{	addr_t   ind0, ind1, ind2, ind3, ind4, ind5;
	addr_t   returnValue_taddr;

	// taddr_ of this variable
	CPPAD_ASSERT_UNKNOWN( NumRes(CExpOp) == 1 );
	returnValue_taddr = Rec_.PutOp(CExpOp);

	// ind[0] = cop
	ind0 = addr_t( cop );

	// ind[1] = base 2 represenation of the value
	// [Var(left), Var(right), Var(if_true), Var(if_false)]
	ind1 = 0;

	// Make sure returnValue is in the list of variables and set its taddr
	if( Parameter(returnValue) )
		returnValue.make_variable(id_, returnValue_taddr );
	else	returnValue.taddr_ = returnValue_taddr;

	// ind[2] = left address
	if( Parameter(left) )
		ind2 = Rec_.PutPar(left.value_);
	else
	{	ind1 += 1;
		ind2 = left.taddr_;
	}

	// ind[3] = right address
	if( Parameter(right) )
		ind3 = Rec_.PutPar(right.value_);
	else
	{	ind1 += 2;
		ind3 = right.taddr_;
	}

	// ind[4] = if_true address
	if( Parameter(if_true) )
		ind4 = Rec_.PutPar(if_true.value_);
	else
	{	ind1 += 4;
		ind4 = if_true.taddr_;
	}

	// ind[5] =  if_false address
	if( Parameter(if_false) )
		ind5 = Rec_.PutPar(if_false.value_);
	else
	{	ind1 += 8;
		ind5 = if_false.taddr_;
	}

	CPPAD_ASSERT_UNKNOWN( NumArg(CExpOp) == 6 );
	CPPAD_ASSERT_UNKNOWN( ind1 > 0 );
	Rec_.PutArg(ind0, ind1, ind2, ind3, ind4, ind5);

	// check that returnValue is a dependent variable
	CPPAD_ASSERT_UNKNOWN( Variable(returnValue) );
}

// ------------ CondExpOp(left, right, if_true, if_false) ----------------

# define CPPAD_COND_EXP(Name)                                        \
	template <class Base>                                           \
	CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION                           \
	AD<Base> CondExp##Name(                                         \
		const AD<Base> &left      ,                                \
		const AD<Base> &right     ,                                \
		const AD<Base> &if_true   ,                                \
		const AD<Base> &if_false  )                                \
	{                                                               \
		return CondExpOp(Compare##Name,                            \
			left, right, if_true, if_false);                      \
	}

// AD<Base>
CPPAD_COND_EXP(Lt)
CPPAD_COND_EXP(Le)
CPPAD_COND_EXP(Eq)
CPPAD_COND_EXP(Ge)
CPPAD_COND_EXP(Gt)
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
AD<Base> CondExp(
	const AD<Base> &flag      ,
	const AD<Base> &if_true   ,
	const AD<Base> &if_false  )
{
	return CondExpOp(CompareGt, flag, AD<Base>(0), if_true, if_false);
}

# undef CPPAD_COND_EXP
} // END CppAD namespace

# endif
