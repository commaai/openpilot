// $Id$
# ifndef CPPAD_CORE_ADD_HPP
# define CPPAD_CORE_ADD_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

//  BEGIN CppAD namespace
namespace CppAD {

template <class Base>
AD<Base> operator + (const AD<Base> &left , const AD<Base> &right)
{
	// compute the Base part of this AD object
	AD<Base> result;
	result.value_  = left.value_ + right.value_;
	CPPAD_ASSERT_UNKNOWN( Parameter(result) );

	// check if there is a recording in progress
	local::ADTape<Base>* tape = AD<Base>::tape_ptr();
	if( tape == CPPAD_NULL )
		return result;
	tape_id_t tape_id = tape->id_;

	// tape_id cannot match the default value for tape_id_; i.e., 0
	CPPAD_ASSERT_UNKNOWN( tape_id > 0 );
	bool var_left  = left.tape_id_  == tape_id;
	bool var_right = right.tape_id_ == tape_id;

	if( var_left )
	{	if( var_right )
		{	// result = variable + variable
			CPPAD_ASSERT_UNKNOWN( local::NumRes(local::AddvvOp) == 1 );
			CPPAD_ASSERT_UNKNOWN( local::NumArg(local::AddvvOp) == 2 );

			// put operand addresses in tape
			tape->Rec_.PutArg(left.taddr_, right.taddr_);
			// put operator in the tape
			result.taddr_ = tape->Rec_.PutOp(local::AddvvOp);
			// make result a variable
			result.tape_id_ = tape_id;
		}
		else if( IdenticalZero(right.value_) )
		{	// result = variable + 0
			result.make_variable(left.tape_id_, left.taddr_);
		}
		else
		{	// result = variable  + parameter
			//        = parameter + variable
			CPPAD_ASSERT_UNKNOWN( local::NumRes(local::AddpvOp) == 1 );
			CPPAD_ASSERT_UNKNOWN( local::NumArg(local::AddpvOp) == 2 );

			// put operand addresses in tape
			addr_t p = tape->Rec_.PutPar(right.value_);
			tape->Rec_.PutArg(p, left.taddr_);
			// put operator in the tape
			result.taddr_ = tape->Rec_.PutOp(local::AddpvOp);
			// make result a variable
			result.tape_id_ = tape_id;
		}
	}
	else if( var_right )
	{	if( IdenticalZero(left.value_) )
		{	// result = 0 + variable
			result.make_variable(right.tape_id_, right.taddr_);
		}
		else
		{	// result = parameter + variable
			CPPAD_ASSERT_UNKNOWN( local::NumRes(local::AddpvOp) == 1 );
			CPPAD_ASSERT_UNKNOWN( local::NumArg(local::AddpvOp) == 2 );

			// put operand addresses in tape
			addr_t p = tape->Rec_.PutPar(left.value_);
			tape->Rec_.PutArg(p, right.taddr_);
			// put operator in the tape
			result.taddr_ = tape->Rec_.PutOp(local::AddpvOp);
			// make result a variable
			result.tape_id_ = tape_id;
		}
	}
	return result;
}

// convert other cases into the case above
CPPAD_FOLD_AD_VALUED_BINARY_OPERATOR(+)

} // END CppAD namespace

# endif
