# ifndef CPPAD_LOCAL_OPTIMIZE_RECORD_CSUM_HPP
# define CPPAD_LOCAL_OPTIMIZE_RECORD_CSUM_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*!
\file record_csum.hpp
Recording a cummulative cummulative summation.
*/
# include <cppad/local/optimize/old2new.hpp>

// BEGIN_CPPAD_LOCAL_OPTIMIZE_NAMESPACE
namespace CppAD { namespace local { namespace optimize  {
/*!
Recording a cummulative cummulative summation.

\param var2op
mapping from old variable index to old operator index.

\param op_info
mapping from old index to operator index to operator information

\param old2new
mapping from old operator index to information about the new recording.

\param current
is the index in the old operation sequence for
the variable corresponding to the result for the current operator.
We use the notation i_op = var2op[current].
It follows that  NumRes( op_info[i_op].op ) > 0.
If 0 < j_op < i_op, either op_info[j_op].usage == csum_usage,
op_info[j_op].usage = no_usage, or old2new[j_op].new_var != 0.

\param npar
is the number of parameters corresponding to the old operation sequence.

\param par
is a vector of length npar containing the parameters
the old operation sequence; i.e.,
given a parameter index i < npar, the corresponding parameter value is par[i].

\param rec
is the object that will record the new operations.

\return
is the operator and variable indices in the new operation sequence.

\param work
Is temporary work space. On input and output,
work.op_stack, work.add_stack, and work.sub_stack, are all empty.
These stacks are passed in so that they are created once
and then be reused with calls to record_csum.

\par Assumptions
op_info[i_o].op
must be one of AddpvOp, AddvvOp, SubpvOp, SubvpOp, SubvvOp.
op_info[i_op].usage != no_usage and ! op_info[i_op].usage == csum_usage.
Furthermore op_info[j_op].usage == csum_usage is true from some
j_op that corresponds to a variable that is an argument to
op_info[i_op].
*/

template <class Base>
struct_size_pair record_csum(
	const vector<addr_t>&                              var2op         ,
	const vector<struct_op_info>&                      op_info        ,
	const CppAD::vector<struct struct_old2new>&        old2new        ,
	size_t                                             current        ,
	size_t                                             npar           ,
	const Base*                                        par            ,
	recorder<Base>*                                    rec            ,
	// local information passed so stacks need not be allocated for every call
	struct_csum_stacks&                                work           )
{
	// check assumption about work space
	CPPAD_ASSERT_UNKNOWN( work.op_stack.empty() );
	CPPAD_ASSERT_UNKNOWN( work.add_stack.empty() );
	CPPAD_ASSERT_UNKNOWN( work.sub_stack.empty() );
	//
	size_t i_op = var2op[current];
	CPPAD_ASSERT_UNKNOWN( ! ( op_info[i_op].usage == csum_usage ) );
	//
	size_t                        i;
	OpCode                        op;
	const addr_t*                 arg;
	bool                          add;
	struct struct_csum_variable var;
	//
	// information corresponding to the root node in the cummulative summation
	var.op  = op_info[i_op].op;   // this operator
	var.arg = op_info[i_op].arg;  // arguments for this operator
	var.add = true;               // was parrent operator positive or negative
	//
	// initialize stack as containing this one operator
	work.op_stack.push( var );
	//
	// initialize sum of parameter values as zero
	Base sum_par(0);
	//
# ifndef NDEBUG
	bool ok = false;
	struct_op_info info = op_info[i_op];
	if( var.op == SubvpOp )
		ok = op_info[ var2op[info.arg[0]] ].usage == csum_usage;
	if( var.op == AddpvOp || var.op == SubpvOp )
		ok = op_info[ var2op[info.arg[1]] ].usage == csum_usage;
	if( var.op == AddvvOp || var.op == SubvvOp )
	{	ok  = op_info[ var2op[info.arg[0]] ].usage == csum_usage;
		ok |= op_info[ var2op[info.arg[1]] ].usage == csum_usage;
	}
	CPPAD_ASSERT_UNKNOWN( ok );
# endif
	//
	// while there are operators left on the stack
	while( ! work.op_stack.empty() )
	{	// get this summation operator
		var     = work.op_stack.top();
		work.op_stack.pop();
		op      = var.op;
		arg     = var.arg;
		add     = var.add;
		//
		// process first argument to this operator
		switch(op)
		{	// cases where first argument is a parameter
			case AddpvOp:
			case SubpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < npar );
			// first argument has same sign as parent node
			if( add )
				sum_par += par[arg[0]];
			else	sum_par -= par[arg[0]];
			break;

			// cases where first argument is a variable
			case AddvvOp:
			case SubvpOp:
			case SubvvOp:
			//
			// check if the first argument has csum usage
			if( op_info[var2op[arg[0]]].usage == csum_usage )
			{	CPPAD_ASSERT_UNKNOWN(
					size_t( old2new[ var2op[arg[0]] ].new_var) == 0
				);
				// push the operator corresponding to the first argument
				var.op  = op_info[ var2op[arg[0]] ].op;
				var.arg = op_info[ var2op[arg[0]] ].arg;
				// first argument has same sign as parent node
				var.add = add;
				work.op_stack.push( var );
			}
			else
			{	// there are no nodes below this one
				CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < current );
				if( add )
					work.add_stack.push(arg[0]);
				else	work.sub_stack.push(arg[0]);
			}
			break;

			default:
			CPPAD_ASSERT_UNKNOWN(false);
		}
		// process second argument to this operator
		switch(op)
		{	// cases where second argument is a parameter
			case SubvpOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < npar );
			// second argument has opposite sign of parent node
			if( add )
				sum_par -= par[arg[1]];
			else	sum_par += par[arg[1]];
			break;

			// cases where second argument is a variable and has opposite sign
			case SubvvOp:
			case SubpvOp:
			add = ! add;

			// cases where second argument is a variable and has same sign
			case AddvvOp:
			case AddpvOp:
			// check if the second argument has csum usage
			if( op_info[var2op[arg[1]]].usage == csum_usage )
			{	CPPAD_ASSERT_UNKNOWN(
					size_t( old2new[ var2op[arg[1]] ].new_var) == 0
				);
				// push the operator corresoponding to the second arugment
				var.op   = op_info[ var2op[arg[1]] ].op;
				var.arg  = op_info[ var2op[arg[1]] ].arg;
				var.add  = add;
				work.op_stack.push( var );
			}
			else
			{	// there are no nodes below this one
				CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < current );
				if( add )
					work.add_stack.push(arg[1]);
				else	work.sub_stack.push(arg[1]);
			}
			break;

			default:
			CPPAD_ASSERT_UNKNOWN(false);
		}
	}
	// number of variables to add in this cummulative sum operator
	size_t n_add = work.add_stack.size();
	// number of variables to subtract in this cummulative sum operator
	size_t n_sub = work.sub_stack.size();
	//
	CPPAD_ASSERT_UNKNOWN(
		std::numeric_limits<addr_t>::max() >= n_add + n_sub
	);
	//
	rec->PutArg( addr_t(n_add) );                // arg[0]
	rec->PutArg( addr_t(n_sub) );                // arg[1]
	addr_t new_arg = rec->PutPar(sum_par);
	rec->PutArg(new_arg);              // arg[2]
	// addition arguments
	for(i = 0; i < n_add; i++)
	{	CPPAD_ASSERT_UNKNOWN( ! work.add_stack.empty() );
		size_t old_arg = work.add_stack.top();
		new_arg        = old2new[ var2op[old_arg] ].new_var;
		CPPAD_ASSERT_UNKNOWN( 0 < new_arg && size_t(new_arg) < current );
		rec->PutArg(new_arg);         // arg[3+i]
		work.add_stack.pop();
	}
	// subtraction arguments
	for(i = 0; i < n_sub; i++)
	{	CPPAD_ASSERT_UNKNOWN( ! work.sub_stack.empty() );
		size_t old_arg = work.sub_stack.top();
		new_arg        = old2new[ var2op[old_arg] ].new_var;
		CPPAD_ASSERT_UNKNOWN( 0 < new_arg && size_t(new_arg) < current );
		rec->PutArg(new_arg);      // arg[3 + arg[0] + i]
		work.sub_stack.pop();
	}
	// number of additions plus number of subtractions
	rec->PutArg( addr_t(n_add + n_sub) );      // arg[3 + arg[0] + arg[1]]
	//
	// return value
	struct_size_pair ret;
	ret.i_op  = rec->num_op_rec();
	ret.i_var = rec->PutOp(CSumOp);
	//
	return ret;
}

} } } // END_CPPAD_LOCAL_OPTIMIZE_NAMESPACE


# endif
