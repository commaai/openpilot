# ifndef CPPAD_CORE_COMPARE_HPP
# define CPPAD_CORE_COMPARE_HPP

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
$begin Compare$$
$spell
	cos
	Op
	bool
	const
$$



$section AD Binary Comparison Operators$$
$mindex compare < <= > >= == !=$$


$head Syntax$$

$icode%b% = %x% %Op% %y%$$


$head Purpose$$
Compares two operands where one of the operands is an
$codei%AD<%Base%>%$$ object.
The comparison has the same interpretation as for
the $icode Base$$ type.


$head Op$$
The operator $icode Op$$ is one of the following:
$table
$bold Op$$ $pre $$  $cnext $bold Meaning$$                           $rnext
$code <$$   $cnext is $icode x$$ less than $icode y$$              $rnext
$code <=$$  $cnext is $icode x$$ less than or equal $icode y$$     $rnext
$code >$$   $cnext is $icode x$$ greater than $icode y$$           $rnext
$code >=$$  $cnext is $icode x$$ greater than or equal $icode y$$  $rnext
$code ==$$  $cnext is $icode x$$ equal to $icode y$$               $rnext
$code !=$$  $cnext is $icode x$$ not equal to $icode y$$
$tend

$head x$$
The operand $icode x$$ has prototype
$codei%
	const %Type% &%x%
%$$
where $icode Type$$ is $codei%AD<%Base%>%$$, $icode Base$$, or $code int$$.

$head y$$
The operand $icode y$$ has prototype
$codei%
	const %Type% &%y%
%$$
where $icode Type$$ is $codei%AD<%Base%>%$$, $icode Base$$, or $code int$$.

$head b$$
The result $icode b$$ has type
$codei%
	bool %b%
%$$

$head Operation Sequence$$
The result of this operation is a $code bool$$ value
(not an $cref/AD of Base/glossary/AD of Base/$$ object).
Thus it will not be recorded as part of an
AD of $icode Base$$
$cref/operation sequence/glossary/Operation/Sequence/$$.
$pre

$$
For example, suppose
$icode x$$ and $icode y$$ are $codei%AD<%Base%>%$$ objects,
the tape corresponding to $codei%AD<%Base%>%$$ is recording,
$icode b$$ is true,
and the subsequent code is
$codei%
	if( %b% )
		%y% = cos(%x%);
	else	%y% = sin(%x%);
%$$
only the assignment $icode%y% = cos(%x%)%$$ is recorded on the tape
(if $icode x$$ is a $cref/parameter/glossary/Parameter/$$,
nothing is recorded).
The $cref CompareChange$$ function can yield
some information about changes in comparison operation results.
You can use $cref CondExp$$ to obtain comparison operations
that depends on the
$cref/independent variable/glossary/Tape/Independent Variable/$$
values with out re-taping the AD sequence of operations.

$head Assumptions$$
If one of the $icode Op$$ operators listed above
is used with an $codei%AD<%Base%>%$$ object,
it is assumed that the same operator is supported by the base type
$icode Base$$.

$head Example$$
$children%
	example/general/compare.cpp
%$$
The file
$cref compare.cpp$$
contains an example and test of these operations.
It returns true if it succeeds and false otherwise.

$end
-------------------------------------------------------------------------------
*/
//  BEGIN CppAD namespace
namespace CppAD {

// -------------------------------- < --------------------------
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool operator < (const AD<Base> &left , const AD<Base> &right)
{	bool result    =  (left.value_ < right.value_);
	bool var_left  = Variable(left);
	bool var_right = Variable(right);

	local::ADTape<Base> *tape = CPPAD_NULL;
	if( var_left )
	{	tape = left.tape_this();
		if( var_right )
		{	if( result )
			{	tape->Rec_.PutOp(local::LtvvOp);
				tape->Rec_.PutArg(left.taddr_, right.taddr_);
			}
			else
			{	tape->Rec_.PutOp(local::LevvOp);
				tape->Rec_.PutArg(right.taddr_, left.taddr_);
			}
		}
		else
		{	addr_t arg1 = tape->Rec_.PutPar(right.value_);
			if( result )
			{	tape->Rec_.PutOp(local::LtvpOp);
				tape->Rec_.PutArg(left.taddr_, arg1);
			}
			else
			{	tape->Rec_.PutOp(local::LepvOp);
				tape->Rec_.PutArg(arg1, left.taddr_);
			}
		}
	}
	else if ( var_right )
	{	tape = right.tape_this();
		addr_t arg0 = tape->Rec_.PutPar(left.value_);
		if( result )
		{	tape->Rec_.PutOp(local::LtpvOp);
			tape->Rec_.PutArg(arg0, right.taddr_);
		}
		else
		{	tape->Rec_.PutOp(local::LevpOp);
			tape->Rec_.PutArg(right.taddr_, arg0);
		}
	}

	return result;
}
// convert other cases into the case above
CPPAD_FOLD_BOOL_VALUED_BINARY_OPERATOR(<)

// -------------------------------- <= -------------------------
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool operator <= (const AD<Base> &left , const AD<Base> &right)
{	bool result    =  (left.value_ <= right.value_);
	bool var_left  = Variable(left);
	bool var_right = Variable(right);

	local::ADTape<Base> *tape = CPPAD_NULL;
	if( var_left )
	{	tape = left.tape_this();
		if( var_right )
		{	if( result )
			{	tape->Rec_.PutOp(local::LevvOp);
				tape->Rec_.PutArg(left.taddr_, right.taddr_);
			}
			else
			{	tape->Rec_.PutOp(local::LtvvOp);
				tape->Rec_.PutArg(right.taddr_, left.taddr_);
			}
		}
		else
		{	addr_t arg1 = tape->Rec_.PutPar(right.value_);
			if( result )
			{	tape->Rec_.PutOp(local::LevpOp);
				tape->Rec_.PutArg(left.taddr_, arg1);
			}
			else
			{	tape->Rec_.PutOp(local::LtpvOp);
				tape->Rec_.PutArg(arg1, left.taddr_);
			}
		}
	}
	else if ( var_right )
	{	tape = right.tape_this();
		addr_t arg0 = tape->Rec_.PutPar(left.value_);
		if( result )
		{	tape->Rec_.PutOp(local::LepvOp);
			tape->Rec_.PutArg(arg0, right.taddr_);
		}
		else
		{	tape->Rec_.PutOp(local::LtvpOp);
			tape->Rec_.PutArg(right.taddr_, arg0);
		}
	}

	return result;
}
// convert other cases into the case above
CPPAD_FOLD_BOOL_VALUED_BINARY_OPERATOR(<=)

// -------------------------------- > --------------------------
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool operator > (const AD<Base> &left , const AD<Base> &right)
{	bool result    =  (left.value_ > right.value_);
	bool var_left  = Variable(left);
	bool var_right = Variable(right);

	local::ADTape<Base> *tape = CPPAD_NULL;
	if( var_left )
	{	tape = left.tape_this();
		if( var_right )
		{	if( result )
			{	tape->Rec_.PutOp(local::LtvvOp);
				tape->Rec_.PutArg(right.taddr_, left.taddr_);
			}
			else
			{	tape->Rec_.PutOp(local::LevvOp);
				tape->Rec_.PutArg(left.taddr_, right.taddr_);
			}
		}
		else
		{	addr_t arg1 = tape->Rec_.PutPar(right.value_);
			if( result )
			{	tape->Rec_.PutOp(local::LtpvOp);
				tape->Rec_.PutArg(arg1, left.taddr_);
			}
			else
			{	tape->Rec_.PutOp(local::LevpOp);
				tape->Rec_.PutArg(left.taddr_, arg1);
			}
		}
	}
	else if ( var_right )
	{	tape = right.tape_this();
		addr_t arg0 = tape->Rec_.PutPar(left.value_);
		if( result )
		{	tape->Rec_.PutOp(local::LtvpOp);
			tape->Rec_.PutArg(right.taddr_, arg0);
		}
		else
		{	tape->Rec_.PutOp(local::LepvOp);
			tape->Rec_.PutArg(arg0, right.taddr_);
		}
	}

	return result;
}
// convert other cases into the case above
CPPAD_FOLD_BOOL_VALUED_BINARY_OPERATOR(>)

// -------------------------------- >= -------------------------
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool operator >= (const AD<Base> &left , const AD<Base> &right)
{	bool result    =  (left.value_ >= right.value_);
	bool var_left  = Variable(left);
	bool var_right = Variable(right);

	local::ADTape<Base> *tape = CPPAD_NULL;
	if( var_left )
	{	tape = left.tape_this();
		if( var_right )
		{	if( result )
			{	tape->Rec_.PutOp(local::LevvOp);
				tape->Rec_.PutArg(right.taddr_, left.taddr_);
			}
			else
			{	tape->Rec_.PutOp(local::LtvvOp);
				tape->Rec_.PutArg(left.taddr_, right.taddr_);
			}
		}
		else
		{	addr_t arg1 = tape->Rec_.PutPar(right.value_);
			if( result )
			{	tape->Rec_.PutOp(local::LepvOp);
				tape->Rec_.PutArg(arg1, left.taddr_);
			}
			else
			{	tape->Rec_.PutOp(local::LtvpOp);
				tape->Rec_.PutArg(left.taddr_, arg1);
			}
		}
	}
	else if ( var_right )
	{	tape = right.tape_this();
		addr_t arg0 = tape->Rec_.PutPar(left.value_);
		if( result )
		{	tape->Rec_.PutOp(local::LevpOp);
			tape->Rec_.PutArg(right.taddr_, arg0);
		}
		else
		{	tape->Rec_.PutOp(local::LtpvOp);
			tape->Rec_.PutArg(arg0, right.taddr_);
		}
	}

	return result;
}
// convert other cases into the case above
CPPAD_FOLD_BOOL_VALUED_BINARY_OPERATOR(>=)

// -------------------------------- == -------------------------
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool operator == (const AD<Base> &left , const AD<Base> &right)
{	bool result    =  (left.value_ == right.value_);
	bool var_left  = Variable(left);
	bool var_right = Variable(right);

	local::ADTape<Base> *tape = CPPAD_NULL;
	if( var_left )
	{	tape = left.tape_this();
		if( var_right )
		{	tape->Rec_.PutArg(left.taddr_, right.taddr_);
			if( result )
				tape->Rec_.PutOp(local::EqvvOp);
			else
				tape->Rec_.PutOp(local::NevvOp);
		}
		else
		{	addr_t arg1 = tape->Rec_.PutPar(right.value_);
			tape->Rec_.PutArg(arg1, left.taddr_);
			if( result )
				tape->Rec_.PutOp(local::EqpvOp);
			else
				tape->Rec_.PutOp(local::NepvOp);
		}
	}
	else if ( var_right )
	{	tape = right.tape_this();
		addr_t arg0 = tape->Rec_.PutPar(left.value_);
		tape->Rec_.PutArg(arg0, right.taddr_);
		if( result )
			tape->Rec_.PutOp(local::EqpvOp);
		else
			tape->Rec_.PutOp(local::NepvOp);
	}

	return result;
}
// convert other cases into the case above
CPPAD_FOLD_BOOL_VALUED_BINARY_OPERATOR(==)

// -------------------------------- != -------------------------
template <class Base>
CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
bool operator != (const AD<Base> &left , const AD<Base> &right)
{	bool result    =  (left.value_ != right.value_);
	bool var_left  = Variable(left);
	bool var_right = Variable(right);

	local::ADTape<Base> *tape = CPPAD_NULL;
	if( var_left )
	{	tape = left.tape_this();
		if( var_right )
		{	tape->Rec_.PutArg(left.taddr_, right.taddr_);
			if( result )
				tape->Rec_.PutOp(local::NevvOp);
			else
				tape->Rec_.PutOp(local::EqvvOp);
		}
		else
		{	addr_t arg1 = tape->Rec_.PutPar(right.value_);
			tape->Rec_.PutArg(arg1, left.taddr_);
			if( result )
				tape->Rec_.PutOp(local::NepvOp);
			else
				tape->Rec_.PutOp(local::EqpvOp);
		}
	}
	else if ( var_right )
	{	tape = right.tape_this();
		addr_t arg0 = tape->Rec_.PutPar(left.value_);
		tape->Rec_.PutArg(arg0, right.taddr_);
		if( result )
			tape->Rec_.PutOp(local::NepvOp);
		else
			tape->Rec_.PutOp(local::EqpvOp);
	}

	return result;
}
// convert other cases into the case above
CPPAD_FOLD_BOOL_VALUED_BINARY_OPERATOR(!=)

} // END CppAD namespace

# endif
