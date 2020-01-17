# ifndef CPPAD_LOCAL_OPTIMIZE_MATCH_OP_HPP
# define CPPAD_LOCAL_OPTIMIZE_MATCH_OP_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
# include <cppad/local/optimize/hash_code.hpp>
/*!
\file match_op.hpp
Check if current operator matches a previous operator.
*/
// BEGIN_CPPAD_LOCAL_OPTIMIZE_NAMESPACE
namespace CppAD { namespace local { namespace optimize  {
/*!
Search for a previous operator that matches the current one.

If an argument for the current operator is a variable,
and the argument has previous match,
the previous match for the argument is used when checking for a match
for the current operator.

\param var2op
mapping from variable index to operator index.

\param op_info
Mapping from operator index to operator information.
The input value of op_info[current].previous is assumed to be zero.
If a match if found,
the output value of op_info[current].previous is set to the
matching operator index, otherwise it is left as is.
Note that op_info[current].previous < current.

\param current
is the index of the current operator which must be an unary
or binary operator. Note that NumArg(ErfOp) == 3 but it is effectivey
a unary operator and is allowed otherwise NumArg( op_info[current].op) < 3.
It is assumed that hash_table_op is initialized as a vector of emtpy
sets. After this initialization, the value of current inceases with
each call to match_op.

\li
This must be a unary or binary
operator; hence, NumArg( op_info[current].op ) is one or two.
There is one exception, NumRes( ErfOp ) == 3, but arg[0]
is the only true arguments (the others are always the same).

\li
This must not be a VecAD load or store operation; i.e.,
LtpvOp, LtvpOp, LtvvOp, StppOp, StpvOp, StvpOp, StvvOp.
It also must not be an independent variable operator InvOp.

\param hash_table_op
is a vector of sets,
hash_table_op.n_set() == CPPAD_HASH_TABLE_SIZE and
hash_table_op.end() == op_info.size().
If i_op is an element of set[j],
then the operation op_info[i_op] has hash code j,
and op_info[i_op] does not match any other element of set[j].
An entry will be added each time match_op is called
and a match for the current operator is not found.
*/

inline void match_op(
	const vector<addr_t>&          var2op         ,
	vector<struct_op_info>&        op_info        ,
	size_t                         current        ,
	sparse_list&                   hash_table_op  )
{	size_t num_op = op_info.size();
	//
	CPPAD_ASSERT_UNKNOWN( op_info[current].previous == 0 );
	CPPAD_ASSERT_UNKNOWN(
		hash_table_op.n_set() == CPPAD_HASH_TABLE_SIZE
	);
	CPPAD_ASSERT_UNKNOWN( hash_table_op.end() == num_op );
	CPPAD_ASSERT_UNKNOWN( current < num_op );
	//
	// current operator
	OpCode        op         = op_info[current].op;
	const addr_t* arg        = op_info[current].arg;
	//
	// which arguments are variable
	size_t num_arg = NumArg(op);
	//
	bool   variable[2];
	variable[0] = false;
	variable[1] = false;
	switch(op)
	{	//
		case ErfOp:
		num_arg = 1; // other arugments are always the same
		//
		case AbsOp:
		case AcosOp:
		case AcoshOp:
		case AsinOp:
		case AsinhOp:
		case AtanOp:
		case AtanhOp:
		case CosOp:
		case CoshOp:
		case ExpOp:
		case Expm1Op:
		case LogOp:
		case Log1pOp:
		case SignOp:
		case SinOp:
		case SinhOp:
		case SqrtOp:
		case TanOp:
		case TanhOp:
		CPPAD_ASSERT_UNKNOWN( num_arg == 1 );
		variable[0] = true;
		break;


		case AddpvOp:
		case DisOp:
		case DivpvOp:
		case EqpvOp:
		case LepvOp:
		case LtpvOp:
		case MulpvOp:
		case NepvOp:
		case PowpvOp:
		case SubpvOp:
		case ZmulpvOp:
		CPPAD_ASSERT_UNKNOWN( num_arg == 2 );
		variable[1] = true;
		break;

		case DivvpOp:
		case LevpOp:
		case LtvpOp:
		case PowvpOp:
		case SubvpOp:
		case ZmulvpOp:
		CPPAD_ASSERT_UNKNOWN( num_arg == 2 );
		variable[0] = true;
		break;

		case AddvvOp:
		case DivvvOp:
		case EqvvOp:
		case LevvOp:
		case LtvvOp:
		case MulvvOp:
		case NevvOp:
		case PowvvOp:
		case SubvvOp:
		case ZmulvvOp:
		CPPAD_ASSERT_UNKNOWN( num_arg == 2 );
		variable[0] = true;
		variable[1] = true;
		break;

		default:
		CPPAD_ASSERT_UNKNOWN(false);
	}
	//
	// If i-th argument to current operator has a previous operator,
	// this is the i-th argument for previous operator.
	// Otherwise, it is the i-th argument for the current operator
	// (if a previous variable exists)
	addr_t arg_match[2];
	for(size_t j = 0; j < num_arg; ++j)
	{	arg_match[j] = arg[j];
		if( variable[j] )
		{	size_t previous = op_info[ var2op[arg[j]] ].previous;
			if( previous != 0 )
			{	CPPAD_ASSERT_UNKNOWN( op_info[previous].previous == 0 );
				//
				arg_match[j] = op_info[previous].i_var;
			}
		}
	}
	size_t code = optimize_hash_code(op, num_arg, arg_match);
	//
	// iterator for the set with this hash code
	sparse_list_const_iterator itr(hash_table_op, code);
	//
	// check for a match
	size_t count = 0;
	while( *itr != num_op )
	{	++count;
		//
		// candidate previous for current operator
		size_t  candidate  = *itr;
		CPPAD_ASSERT_UNKNOWN( candidate < current );
		CPPAD_ASSERT_UNKNOWN( op_info[candidate].previous == 0 );
		//
		// check for a match
		bool match = op == op_info[candidate].op;
		if( match )
		{	for(size_t j = 0; j < num_arg; j++)
			{	if( variable[j] )
				{	size_t previous =
						op_info[ var2op[op_info[candidate].arg[j]] ].previous;
					if( previous != 0 )
					{	CPPAD_ASSERT_UNKNOWN(op_info[previous].previous == 0);
						//
						match &=
							arg_match[j] == addr_t( op_info[previous].i_var );
					}
					else
						match &= arg_match[j] == op_info[candidate].arg[j];
				}
			}
		}
		if( match )
		{	op_info[current].previous = static_cast<addr_t>( candidate );
			return;
		}
		++itr;
	}

	// special case where operator is commutative
	if( (op == AddvvOp) | (op == MulvvOp ) )
	{	CPPAD_ASSERT_UNKNOWN( NumArg(op) == 2 );
		std::swap( arg_match[0], arg_match[1] );
		//
		code      = optimize_hash_code(op, num_arg, arg_match);
		sparse_list_const_iterator itr_swap(hash_table_op, code);
		while( *itr_swap != num_op )
		{
			size_t candidate  = *itr_swap;
			CPPAD_ASSERT_UNKNOWN( candidate < current );
			CPPAD_ASSERT_UNKNOWN( op_info[candidate].previous == 0 );
			//
			bool match = op == op_info[candidate].op;
			if( match )
			{	for(size_t j = 0; j < num_arg; j++)
				{	CPPAD_ASSERT_UNKNOWN( variable[j] )
					size_t previous =
						op_info[ var2op[op_info[candidate].arg[j]] ].previous;
					if( previous != 0 )
					{	CPPAD_ASSERT_UNKNOWN(op_info[previous].previous == 0);
						//
						match &=
							arg_match[j] == addr_t( op_info[previous].i_var );
					}
					else
						match &= arg_match[j] == op_info[candidate].arg[j];
				}
			}
			if( match )
			{	op_info[current].previous = static_cast<addr_t>( candidate );
				return;
			}
			++itr_swap;
		}
	}
	CPPAD_ASSERT_UNKNOWN( count < 11 );
	if( count == 10 )
	{	// restart the list
		hash_table_op.clear(code);
	}
	// no match was found, add this operator the the set for this hash code
	hash_table_op.add_element(code, current);
}

} } } // END_CPPAD_LOCAL_OPTIMIZE_NAMESPACE

# endif
