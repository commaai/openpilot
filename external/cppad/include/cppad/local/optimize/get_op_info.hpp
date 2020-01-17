# ifndef CPPAD_LOCAL_OPTIMIZE_GET_OP_INFO_HPP
# define CPPAD_LOCAL_OPTIMIZE_GET_OP_INFO_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*!
\file op_info.hpp
Create operator information tables
*/

# include <cppad/local/optimize/op_info.hpp>
# include <cppad/local/optimize/match_op.hpp>
# include <cppad/local/optimize/cexp_info.hpp>
# include <cppad/local/optimize/usage.hpp>

// BEGIN_CPPAD_LOCAL_OPTIMIZE_NAMESPACE
namespace CppAD { namespace local { namespace optimize {

/// Is this an addition or subtraction operator
inline bool add_or_subtract(OpCode op)
{	bool result;
	switch(op)
	{
		case AddpvOp:
		case AddvvOp:
		case SubpvOp:
		case SubvpOp:
		case SubvvOp:
		result = true;
		break;

		default:
		result = false;
		break;
	}
	return result;
}


/*!
Increarse argument usage and propagate cexp_set from result to argument.

\param sum_result
is result an addition or subtraction operator (passed for speed so
do not need to call add_or_subtract for result).

\param i_result
is the operator index for the result operator.

\param i_arg
is the operator index for the argument to the result operator.

\param op_info
structure that holds the information for each of the operators.
The output value of op_info[i_arg].usage is increased; to be specific,
If sum_result is true and the input value of op_info[i_arg].usage
is no_usage, its output value is csum_usage.
Otherwise, the output value of op_info[i_arg].usage is yes_usage.

\param cexp_set
This is a vector of sets with one set for each operator. We denote
the i-th set by set[i].

\li
In the special case where cexp_set.n_set() is zero,
cexp_set is not changed.

\li
If cexp_set.n_set() != 0 and op_info[i_arg].usage == no_usage,
the input value of set[i_arg] must be empty.
In this case the output value if set[i_arg] is equal to set[i_result]
(which may also be empty).

\li
If cexp_set.n_set() != 0 and op_info[i_arg].usage != no_usage,
the output value of set[i_arg] is the intersection of
its input value and set[i_result].
*/
inline void usage_cexp_result2arg(
	bool                    sum_result ,
	size_t                  i_result   ,
	size_t                  i_arg      ,
	vector<struct_op_info>& op_info    ,
	sparse_list&            cexp_set   )
{
	// cexp_set
	if( cexp_set.n_set() > 0 )
	{	if( op_info[i_arg].usage == no_usage )
		{	// set[i_arg] = set[i_result]
			cexp_set.assignment(i_arg, i_result, cexp_set);
		}
		else
		{	// set[i_arg] = set[i_arg] intersect set[i_result]
			cexp_set.binary_intersection(i_arg, i_arg, i_result, cexp_set);
		}
	}
	// usage
	bool csum = sum_result && op_info[i_arg].usage == no_usage;
	if( csum )
		csum = add_or_subtract( op_info[i_arg].op );
	if( csum )
		op_info[i_arg].usage = csum_usage;
	else
		op_info[i_arg].usage = yes_usage;
	//
	return;
}

/*!
Get variable to operator map and operator basic operator information

\tparam Base
base type for the operator; i.e., this operation was recorded
using AD< \a Base > and computations by this routine are done using type
\a Base.

\param conditional_skip
If conditional_skip this is true, the conditional expression information
cexp_info will be calculated.
This may be time intensive and may not have much benefit in the optimized
recording.

\param compare_op
if this is true, arguments are considered used if they appear in compare
operators. This is a side effect because compare operators have boolean
results (and the result is not in the tape; i.e. NumRes(op) is zero
for these operators. (This is an example of a side effect.)

\param print_for_op
if this is true, arguments are considered used if they appear in
print forward operators; i.e., PriOp.
This is also a side effect; i.e. NumRes(PriOp) is zero.

\param play
This is the operation sequence.
It is essentially const, except for play back state which
changes while it plays back the operation seqeunce.

\param dep_taddr
is a vector of variable indices for the dependent variables.

\param var2op
The input size of this vector must be zero.
Upone return it has size equal to the number of variables
in the operation sequence; i.e., num_var = play->nun_var_rec().
It maps each variable index to the operator that created the variable.
This is only true for the primary variables.
If the index i_var corresponds to an auxillary variable, var2op[i_var]
is equalt to num_op (which is not a valid operator index).

\param cexp_info
The input size of this vector must be zero.
If conditional_skip is false, cexp_info is not changed.
Otherwise,
upon return cexp_info has size equal to the number of conditional expressions
in the operation sequence; i.e., the number of CExpOp operators.
The value cexp_info[j] is the information corresponding to the j-th
conditional expression in the operation sequence.
This vector is in the same order as the operation sequence; i.e.
if j1 > j2, cexp_info[j1].i_op > cexp_info[j2].i_op.
Note that skip_op_true and skip_op_false could be part of this structure,
but then we would allocate and deallocate two vectors for each conditonal
expression in the operation sequence.

\param skip_op_true
This vector of sets is empty on input.
Upon return, the j-th set is the operators that are not used when
comparison result for cexp_info[j] is true.
Note that UsrapOp, UsravOp, UsrrpOp, and UsrrvOp, are not in this
set and should be skipped when the corresponding UserOp are skipped.

\param skip_op_false
This vector of sets is empty on input.
Upon return, the j-th set is the operators that are not used when
comparison result for cexp_info[j] is false.
Note that UsrapOp, UsravOp, UsrrpOp, and UsrrvOp, are not in this
set and should be skipped when the corresponding UserOp are skipped.

\param vecad_used
The input size of this vector must be zero.
Upon retun it has size equal to the number of VecAD vectors
in the operations sequences; i.e., play->num_vecad_vec_rec().
The VecAD vectors are indexed in the order that thier indices apprear
in the one large play->GetVecInd that holds all the VecAD vectors.

\param op_info
The input size of this vector must be zero.
Upon return it has size equal to the number of operators
in the operation sequence; i.e., num_op = play->nun_var_rec().
The value op_info[i]
have been set to the values corresponding to the i-th operator
in the operation sequence.
*/

template <class Base>
void get_op_info(
	bool                          conditional_skip    ,
	bool                          compare_op          ,
	bool                          print_for_op        ,
	player<Base>*                 play                ,
	const vector<size_t>&         dep_taddr           ,
	vector<addr_t>&               var2op              ,
	vector<struct_cexp_info>&     cexp_info           ,
	sparse_list&                  skip_op_true        ,
	sparse_list&                  skip_op_false       ,
	vector<bool>&                 vecad_used          ,
	vector<struct_op_info>&       op_info             )
{
	CPPAD_ASSERT_UNKNOWN( var2op.size()  == 0 );
	CPPAD_ASSERT_UNKNOWN( cexp_info.size() == 0 );
	CPPAD_ASSERT_UNKNOWN( vecad_used.size() == 0 );
	CPPAD_ASSERT_UNKNOWN( op_info.size() == 0 );

	// number of operators in the tape
	const size_t num_op = play->num_op_rec();
	op_info.resize( num_op );
	//
	// number of variables in the tape
	const size_t num_var = play->num_var_rec();
	var2op.resize( num_var );
	//
	// initialize mapping from variable index to operator index
	CPPAD_ASSERT_UNKNOWN(
		std::numeric_limits<addr_t>::max() >= num_op
	);
	for(size_t i = 0; i < num_var; i++)
		var2op[i] = addr_t( num_op ); // invalid (used for auxillary variables)
	//
	// information set by forward_user
	size_t user_old=0, user_m=0, user_n=0, user_i=0, user_j=0;
	enum_user_state user_state;
	//
	// information set by forward_next
	OpCode        op;     // operator
	const addr_t* arg;    // arguments
	size_t        i_op;   // operator index
	size_t        i_var;  // variable index of first result
	//
	// ----------------------------------------------------------------------
	// Forward pass to compute op, arg, i_var for each operator and var2op
	// ----------------------------------------------------------------------
	play->forward_start(op, arg, i_op, i_var);
	CPPAD_ASSERT_UNKNOWN( op              == BeginOp );
	CPPAD_ASSERT_UNKNOWN( NumRes(BeginOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( i_op            == 0 );
	CPPAD_ASSERT_UNKNOWN( i_var           == 0 );
	op_info[i_op].op    = op;
	op_info[i_op].arg   = arg;
	op_info[i_op].i_var = addr_t( i_var );
	//
	// This variaible index, 0, is automatically created, but it should
	// not used because variable index 0 represents a paraemeter during
	// the recording process. So we set
	var2op[i_var] = addr_t( num_op );
	//
	size_t num_cexp_op = 0;
	user_state = start_user;
	while(op != EndOp)
	{	// next operator
		play->forward_next(op, arg, i_op, i_var);
		CPPAD_ASSERT_UNKNOWN(
			size_t( std::numeric_limits<addr_t>::max() ) > i_var
		);
		CPPAD_ASSERT_UNKNOWN(
			size_t( std::numeric_limits<addr_t>::max() ) > i_op
		);
		//
		// information for this operator
		op_info[i_op].op    = op;
		op_info[i_op].arg   = arg;
		op_info[i_op].i_var = addr_t( i_var );
		//
		// mapping from variable index to operator index
		if( NumRes(op) > 0 )
			var2op[i_var] = addr_t( i_op );
		//
		switch( op )
		{	case CSumOp:
			// must correct arg before next operator
			play->forward_csum(op, arg, i_op, i_var);
			break;

			case CSkipOp:
			// must correct arg before next operator
			play->forward_csum(op, arg, i_op, i_var);
			break;

			case UserOp:
			case UsrapOp:
			case UsravOp:
			case UsrrpOp:
			case UsrrvOp:
			play->forward_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			break;

			case CExpOp:
			// Set the operator index for this conditional expression and
			// count the number of conditional expressions.
			++num_cexp_op;
			break;

			default:
			break;
		}
	}
	// vector that maps conditional expression index to operator index
	vector<size_t> cexp2op( num_cexp_op );
	// ----------------------------------------------------------------------
	// Reverse pass to compute usage and cexp_set for each operator
	// ----------------------------------------------------------------------
	// work space used by user defined atomic functions
	typedef std::set<size_t> size_set;
	vector<Base>     user_x;       // parameters in x as integers
	vector<size_t>   user_ix;      // variables indices for argument vector
	vector<size_set> user_r_set;   // set sparsity pattern for result
	vector<size_set> user_s_set;   // set sparisty pattern for argument
	vector<bool>     user_r_bool;  // bool sparsity pattern for result
	vector<bool>     user_s_bool;  // bool sparisty pattern for argument
	vectorBool       user_r_pack;  // pack sparsity pattern for result
	vectorBool       user_s_pack;  // pack sparisty pattern for argument
	//
	atomic_base<Base>* user_atom = CPPAD_NULL; // current user atomic function
	bool               user_pack = false;      // sparsity pattern type is pack
	bool               user_bool = false;      // sparsity pattern type is bool
	bool               user_set  = false;      // sparsity pattern type is set
	// -----------------------------------------------------------------------
	// vecad information
	size_t num_vecad      = play->num_vecad_vec_rec();
	size_t num_vecad_ind  = play->num_vec_ind_rec();
	//
	vecad_used.resize(num_vecad);
	for(size_t i = 0; i < num_vecad; i++)
		vecad_used[i] = false;
	//
	vector<size_t> arg2vecad(num_vecad_ind);
	for(size_t i = 0; i < num_vecad_ind; i++)
		arg2vecad[i] = num_vecad; // invalid value
	size_t arg_0 = 1; // value of arg[0] for theh first vecad
	for(size_t i = 0; i < num_vecad; i++)
	{
		// mapping from arg[0] value to index for this vecad object.
		arg2vecad[arg_0] = i;
		//
		// length of this vecad object
		size_t length = play->GetVecInd(arg_0 - 1);
		//
		// set to proper index in GetVecInd for next VecAD arg[0] value
		arg_0        += length + 1;
	}
	CPPAD_ASSERT_UNKNOWN( arg_0 == num_vecad_ind + 1 );
	// -----------------------------------------------------------------------
	// parameter information (used by atomic function calls)
	size_t num_par = play->num_par_rec();
	const Base* parameter = CPPAD_NULL;
	if( num_par > 0 )
		parameter = play->GetPar();
	// -----------------------------------------------------------------------
	// Set of conditional expressions comparisons that usage of each
	/// operator depends on. The operator can be skipped if any of the
	// comparisons results in the set holds. A set for operator i_op is
	// not defined and left empty when op_info[i_op].usage = no_usage.
	/// It is also left empty for the result of any VecAD operations.
	sparse_list cexp_set;
	//
	// number of sets
	size_t num_set = 0;
	if( conditional_skip && num_cexp_op > 0)
		num_set = num_op;
	//
	// conditional expression index   = element / 2
	// conditional expression compare = bool ( element % 2)
	size_t end_set = 2 * num_cexp_op;
	//
	cexp_set.resize(num_set, end_set);
	// -----------------------------------------------------------------------
	//
	// initialize operator usage
	for(size_t i = 0; i < num_op; i++)
		op_info[i].usage = no_usage;
	for(size_t i = 0; i < dep_taddr.size(); i++)
	{	i_op                = var2op[ dep_taddr[i] ];
		op_info[i_op].usage = yes_usage;    // dependent variables
	}
	//
	// Initialize reverse pass
	size_t last_user_i_op = 0;
	size_t cexp_index     = num_cexp_op;
	user_state            = end_user;
	i_op = num_op;
	while(i_op != 0 )
	{	--i_op;
		//
		// this operator information
		op    =  op_info[i_op].op;
		arg   =  op_info[i_op].arg;
		i_var =  op_info[i_op].i_var;
		//
		// Is the result of this operation used.
		// (This only makes sense when NumRes(op) > 0.)
		enum_usage use_result = op_info[i_op].usage;
		//
		bool sum_op = false;
		switch( op )
		{
			// =============================================================
			// normal operators
			// =============================================================

			// Only one variable with index arg[0]
			case SubvpOp:
			sum_op = true;
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
			case DivvpOp:
			case ErfOp:
			case ExpOp:
			case Expm1Op:
			case LogOp:
			case Log1pOp:
			case PowvpOp:
			case SignOp:
			case SinOp:
			case SinhOp:
			case SqrtOp:
			case TanOp:
			case TanhOp:
			case ZmulvpOp:
			CPPAD_ASSERT_UNKNOWN( NumRes(op) > 0 );
			if( use_result != no_usage )
			{	size_t j_op = var2op[ arg[0] ];
				usage_cexp_result2arg(sum_op, i_op, j_op, op_info, cexp_set);
			}
			break; // --------------------------------------------

			// Only one variable with index arg[1]
			case AddpvOp:
			case SubpvOp:
			sum_op = true;
			//
			case DisOp:
			case DivpvOp:
			case MulpvOp:
			case PowpvOp:
			case ZmulpvOp:
			CPPAD_ASSERT_UNKNOWN( NumRes(op) > 0 );
			if( use_result != no_usage )
			{	size_t j_op = var2op[ arg[1] ];
				usage_cexp_result2arg(sum_op, i_op, j_op, op_info, cexp_set);
			}
			break; // --------------------------------------------

			// arg[0] and arg[1] are the only variables
			case AddvvOp:
			case SubvvOp:
			sum_op = true;
			//
			case DivvvOp:
			case MulvvOp:
			case PowvvOp:
			case ZmulvvOp:
			CPPAD_ASSERT_UNKNOWN( NumRes(op) > 0 );
			if( use_result != no_usage )
			{	for(size_t i = 0; i < 2; i++)
				{	size_t j_op = var2op[ arg[i] ];
					usage_cexp_result2arg(
						sum_op, i_op, j_op, op_info, cexp_set
					);
				}
			}
			break; // --------------------------------------------

			// Conditional expression operators
			// arg[2], arg[3], arg[4], arg[5] are parameters or variables
			case CExpOp:
			--cexp_index;
			cexp2op[ cexp_index ] = i_op;
			CPPAD_ASSERT_UNKNOWN( NumRes(op) > 0 );
			if( use_result != no_usage )
			{	CPPAD_ASSERT_UNKNOWN( NumArg(CExpOp) == 6 );
				// propgate from result to left argument
				if( arg[1] & 1 )
				{	size_t j_op = var2op[ arg[2] ];
					usage_cexp_result2arg(
						sum_op, i_op, j_op, op_info, cexp_set
					);
				}
				// propgate from result to right argument
				if( arg[1] & 2 )
				{	size_t j_op = var2op[ arg[3] ];
					usage_cexp_result2arg(
							sum_op, i_op, j_op, op_info, cexp_set
					);
				}
				// are if_true and if_false cases the same variable
				bool same_variable = bool(arg[1] & 4) && bool(arg[1] & 8);
				same_variable     &= arg[4] == arg[5];
				//
				// if_true
				if( arg[1] & 4 )
				{	size_t j_op = var2op[ arg[4] ];
					bool can_skip = conditional_skip & (! same_variable);
					can_skip     &= op_info[j_op].usage == no_usage;
					usage_cexp_result2arg(
						sum_op, i_op, j_op, op_info, cexp_set
					);
					if( can_skip )
					{	// j_op corresponds to the value used when the
						// comparison result is true. It can be skipped when
						// the comparison is false (0).
						size_t element = 2 * cexp_index + 0;
						cexp_set.add_element(j_op, element);
						//
						op_info[j_op].usage = yes_usage;
					}
				}
				//
				// if_false
				if( arg[1] & 8 )
				{	size_t j_op = var2op[ arg[5] ];
					bool can_skip = conditional_skip & (! same_variable);
					can_skip     &= op_info[j_op].usage == no_usage;
					usage_cexp_result2arg(
						sum_op, i_op, j_op, op_info, cexp_set
					);
					if( can_skip )
					{	// j_op corresponds to the value used when the
						// comparison result is false. It can be skipped when
						// the comparison is true (0).
						size_t element = 2 * cexp_index + 1;
						cexp_set.add_element(j_op, element);
						//
						op_info[j_op].usage = yes_usage;
					}
				}
			}
			break;  // --------------------------------------------

			// Operations that are never used
			// (new CSkip options are generated if conditional_skip is true)
			case CSkipOp:
			case ParOp:
			break;

			// Operators that are always used
			case InvOp:
			case BeginOp:
			case EndOp:
			op_info[i_op].usage = yes_usage;
			break;  // -----------------------------------------------

			// The print forward operator
			case PriOp:
			CPPAD_ASSERT_NARG_NRES(op, 5, 0);
			if( print_for_op )
			{	op_info[i_op].usage = yes_usage;
				if( arg[0] & 1 )
				{	// arg[1] is a variable
					size_t j_op = var2op[ arg[1] ];
					usage_cexp_result2arg(
						sum_op, i_op, j_op, op_info, cexp_set
					);
				}
				if( arg[0] & 2 )
				{	// arg[3] is a variable
					size_t j_op = var2op[ arg[3] ];
					usage_cexp_result2arg(
						sum_op, i_op, j_op, op_info, cexp_set
					);
				}
			}
			break; // -----------------------------------------------------

			// =============================================================
			// Comparison operators
			// =============================================================

			// Compare operators where arg[1] is only variable
			case LepvOp:
			case LtpvOp:
			case EqpvOp:
			case NepvOp:
			CPPAD_ASSERT_UNKNOWN( NumRes(op) == 0 );
			if( compare_op )
			{	op_info[i_op].usage = yes_usage;
				//
				size_t j_op = var2op[ arg[1] ];
				usage_cexp_result2arg(sum_op, i_op, j_op, op_info, cexp_set);
			}
			break; // ----------------------------------------------

			// Compare operators where arg[0] is only variable
			case LevpOp:
			case LtvpOp:
			CPPAD_ASSERT_UNKNOWN( NumRes(op) == 0 );
			if( compare_op )
			{	op_info[i_op].usage = yes_usage;
				//
				size_t j_op = var2op[ arg[0] ];
				usage_cexp_result2arg(sum_op, i_op, j_op, op_info, cexp_set);
			}
			break; // ----------------------------------------------

			// Compare operators where arg[0] and arg[1] are variables
			case LevvOp:
			case LtvvOp:
			case EqvvOp:
			case NevvOp:
			if( compare_op )
			CPPAD_ASSERT_UNKNOWN( NumRes(op) == 0 );
			if( compare_op )
			{	op_info[i_op].usage = yes_usage;
				//
				for(size_t i = 0; i < 2; i++)
				{	size_t j_op = var2op[ arg[i] ];
					usage_cexp_result2arg(
						sum_op, i_op, j_op, op_info, cexp_set
					);
				}
			}
			break; // ----------------------------------------------

			// =============================================================
			// VecAD operators
			// =============================================================

			// load operator using a parameter index
			case LdpOp:
			CPPAD_ASSERT_UNKNOWN( NumRes(op) > 0 );
			if( use_result != no_usage )
			{	size_t i_vec = arg2vecad[ arg[0] ];
				vecad_used[i_vec] = true;
			}
			break; // --------------------------------------------

			// load operator using a variable index
			case LdvOp:
			CPPAD_ASSERT_UNKNOWN( NumRes(op) > 0 );
			if( use_result != no_usage )
			{	size_t i_vec = arg2vecad[ arg[0] ];
				vecad_used[i_vec] = true;
				//
				size_t j_op = var2op[ arg[1] ];
				op_info[j_op].usage = yes_usage;
			}
			break; // --------------------------------------------

			// Store a variable using a parameter index
			case StpvOp:
			CPPAD_ASSERT_UNKNOWN( NumRes(op) == 0 );
			if( vecad_used[ arg2vecad[ arg[0] ] ] )
			{	op_info[i_op].usage = yes_usage;
				//
				size_t j_op = var2op[ arg[2] ];
				op_info[j_op].usage = yes_usage;
			}
			break; // --------------------------------------------

			// Store a variable using a variable index
			case StvvOp:
			CPPAD_ASSERT_UNKNOWN( NumRes(op) == 0 );
			if( vecad_used[ arg2vecad[ arg[0] ] ] )
			{	op_info[i_op].usage = yes_usage;
				//
				size_t j_op = var2op[ arg[1] ];
				op_info[j_op].usage = yes_usage;
				size_t k_op = var2op[ arg[2] ];
				op_info[k_op].usage = yes_usage;
			}
			break; // -----------------------------------------------------

			// =============================================================
			// cumulative summation operator
			// ============================================================
			case CSumOp:
			CPPAD_ASSERT_UNKNOWN( NumRes(op) == 1 );
			{
				size_t num_add = size_t( arg[0] );
				size_t num_sub = size_t( arg[1] );
				for(size_t i = 0; i < num_add + num_sub; i++)
				{	size_t j_op = var2op[ arg[3 + i] ];
					usage_cexp_result2arg(
						sum_op, i_op, j_op, op_info, cexp_set
					);
				}
			}
			// =============================================================
			// user defined atomic operators
			// ============================================================

			case UserOp:
			// start or end atomic operation sequence
			if( user_state == end_user )
			{	// revese_user using op_info instead of play
				size_t user_index = arg[0];
				user_old          = arg[1];
				user_n            = arg[2];
				user_m            = arg[3];
				user_j            = user_n;
				user_i            = user_m;
				user_state        = ret_user;
				user_atom  = atomic_base<Base>::class_object(user_index);
				// -------------------------------------------------------
				last_user_i_op = i_op;
				CPPAD_ASSERT_UNKNOWN( i_op > user_n + user_m + 1 );
				CPPAD_ASSERT_UNKNOWN(op_info[last_user_i_op].usage==no_usage);
# ifndef NDEBUG
				if( cexp_set.n_set() > 0 )
				{	sparse_list_const_iterator itr(cexp_set, last_user_i_op);
					CPPAD_ASSERT_UNKNOWN( *itr == cexp_set.end() );
				}
# endif
				//
				user_x.resize(  user_n );
				user_ix.resize( user_n );
				//
				user_pack  = user_atom->sparsity() ==
							atomic_base<Base>::pack_sparsity_enum;
				user_bool  = user_atom->sparsity() ==
							atomic_base<Base>::bool_sparsity_enum;
				user_set   = user_atom->sparsity() ==
							atomic_base<Base>::set_sparsity_enum;
				CPPAD_ASSERT_UNKNOWN( user_pack || user_bool || user_set );
				//
				// Note that q is one for this call the sparsity calculation
				if( user_pack )
				{	user_r_pack.resize( user_m );
					user_s_pack.resize( user_n );
					for(size_t i = 0; i < user_m; i++)
						user_r_pack[ i ] = false;
				}
				if( user_bool )
				{	user_r_bool.resize( user_m );
					user_s_bool.resize( user_n );
					for(size_t i = 0; i < user_m; i++)
						user_r_bool[ i ] = false;
				}
				if( user_set )
				{	user_s_set.resize(user_n);
					user_r_set.resize(user_m);
					for(size_t i = 0; i < user_m; i++)
						user_r_set[i].clear();
				}
			}
			else
			{	// reverse_user using op_info instead of play
				CPPAD_ASSERT_UNKNOWN( user_state == start_user );
				CPPAD_ASSERT_UNKNOWN( user_n == size_t(arg[2]) );
				CPPAD_ASSERT_UNKNOWN( user_m == size_t(arg[3]) );
				CPPAD_ASSERT_UNKNOWN( user_j == 0 );
				CPPAD_ASSERT_UNKNOWN( user_i == 0 );
				user_state = end_user;
				// -------------------------------------------------------
				CPPAD_ASSERT_UNKNOWN(
					i_op + user_n + user_m + 1 == last_user_i_op
				);
				// call users function for this operation
				user_atom->set_old(user_old);
				bool user_ok  = false;
				size_t user_q = 1; // as if sum of dependent variables
				if( user_pack )
				{	user_ok = user_atom->rev_sparse_jac(
						user_q, user_r_pack, user_s_pack, user_x
					);
					if( ! user_ok ) user_ok = user_atom->rev_sparse_jac(
						user_q, user_r_pack, user_s_pack
					);
				}
				if( user_bool )
				{	user_ok = user_atom->rev_sparse_jac(
						user_q, user_r_bool, user_s_bool, user_x
					);
					if( ! user_ok ) user_ok = user_atom->rev_sparse_jac(
						user_q, user_r_bool, user_s_bool
					);
				}
				if( user_set )
				{	user_ok = user_atom->rev_sparse_jac(
						user_q, user_r_set, user_s_set, user_x
					);
					if( ! user_ok ) user_ok = user_atom->rev_sparse_jac(
						user_q, user_r_set, user_s_set
					);
				}
				if( ! user_ok )
				{	std::string s =
						"Optimizing an ADFun object"
						" that contains the atomic function\n\t";
					s += user_atom->afun_name();
					s += "\nCurrent atomic_sparsity is set to ";
					//
					if( user_set )
						s += "set_sparsity_enum.\n";
					if( user_bool )
						s += "bool_sparsity_enum.\n";
					if( user_pack )
						s += "pack_sparsity_enum.\n";
					//
					s += "This version of rev_sparse_jac returned false";
					CPPAD_ASSERT_KNOWN(false, s.c_str() );
				}

				if( op_info[last_user_i_op].usage != no_usage )
				for(size_t j = 0; j < user_n; j++)
				if( user_ix[j] > 0 )
				{	// This user argument is a variable
					bool use_arg_j = false;
					if( user_set )
					{	if( ! user_s_set[j].empty() )
							use_arg_j = true;
					}
					if( user_bool )
					{	if( user_s_bool[j] )
							use_arg_j = true;
					}
					if( user_pack )
					{	if( user_s_pack[j] )
							use_arg_j = true;
					}
					if( use_arg_j )
					{	size_t j_op = var2op[ user_ix[j] ];
						usage_cexp_result2arg(
							sum_op, last_user_i_op, j_op, op_info, cexp_set
						);
					}
				}
				// copy set infomation from last to first
				if( cexp_set.n_set() > 0 )
					cexp_set.assignment(i_op, last_user_i_op, cexp_set);
				// copy user information from last to all the user operators
				// for this call
				for(size_t j = 0; j < user_n + user_m + 1; ++j)
					op_info[i_op + j].usage = op_info[last_user_i_op].usage;
			}
			break; // -------------------------------------------------------

			case UsrapOp:
			// parameter argument in an atomic operation sequence
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			//
			// reverse_user using op_info instead of play
			CPPAD_ASSERT_NARG_NRES(op, 1, 0);
			CPPAD_ASSERT_UNKNOWN( 0 < user_j && user_j < user_n );
			--user_j;
			if( user_j == 0 )
				user_state = start_user;
			// -------------------------------------------------------------
			user_ix[user_j] = 0;
			//
			// parameter arguments
			user_x[user_j] = parameter[arg[0]];
			//
			break;

			case UsravOp:
			// variable argument in an atomic operation sequence
			CPPAD_ASSERT_UNKNOWN( arg[0] <= op_info[i_op].i_var );
			CPPAD_ASSERT_UNKNOWN( 0 < arg[0] );
			//
			// reverse_user using op_info instead of play
			CPPAD_ASSERT_NARG_NRES(op, 1, 0);
			CPPAD_ASSERT_UNKNOWN( 0 < user_j && user_j <= user_n );
			--user_j;
			if( user_j == 0 )
				user_state = start_user;
			// -------------------------------------------------------------
			user_ix[user_j] = arg[0];
			//
			// variable arguments as parameters
			user_x[user_j] = CppAD::numeric_limits<Base>::quiet_NaN();
			//
			break;

			case UsrrvOp:
			// variable result in an atomic operation sequence
			//
			// reverse_user using op_info instead of play
			CPPAD_ASSERT_NARG_NRES(op, 0, 1);
			CPPAD_ASSERT_UNKNOWN( 0 < user_i && user_i <= user_m );
			--user_i;
			if( user_i == 0 )
				user_state = arg_user;
			// -------------------------------------------------------------
			if( use_result )
			{	if( user_set )
					user_r_set[user_i].insert(0);
				if( user_bool )
					user_r_bool[user_i] = true;
				if( user_pack )
					user_r_pack[user_i] = true;
				//
				usage_cexp_result2arg(
					sum_op, i_op, last_user_i_op, op_info, cexp_set
				);
			}
			break; // --------------------------------------------------------

			case UsrrpOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			//
			// reverse_user using op_info instead of play
			CPPAD_ASSERT_NARG_NRES(op, 0, 1);
			CPPAD_ASSERT_UNKNOWN( 0 < user_i && user_i < user_m );
			--user_i;
			if( user_i == 0 )
				user_state = arg_user;
			break;
			// ============================================================

			// all cases should be handled above
			default:
			CPPAD_ASSERT_UNKNOWN(0);
		}
	}
	// ----------------------------------------------------------------------
	// compute previous in op_info
	// ----------------------------------------------------------------------
	sparse_list  hash_table_op;
	hash_table_op.resize(CPPAD_HASH_TABLE_SIZE, num_op);
	//
	user_state = start_user;
	for(i_op = 0; i_op < num_op; ++i_op)
	{	op_info[i_op].previous = 0;

		if( op_info[i_op].usage == yes_usage ) switch( op_info[i_op].op )
		{
			case NumberOp:
			CPPAD_ASSERT_UNKNOWN(false);
			break;

			case BeginOp:
			case CExpOp:
			case CSkipOp:
			case CSumOp:
			case EndOp:
			case InvOp:
			case LdpOp:
			case LdvOp:
			case ParOp:
			case PriOp:
			case StppOp:
			case StpvOp:
			case StvpOp:
			case StvvOp:
			case UserOp:
			case UsrapOp:
			case UsravOp:
			case UsrrpOp:
			case UsrrvOp:
			// these operators never match pevious operators
			break;

			case AbsOp:
			case AcosOp:
			case AcoshOp:
			case AddpvOp:
			case AddvvOp:
			case AsinOp:
			case AsinhOp:
			case AtanOp:
			case AtanhOp:
			case CosOp:
			case CoshOp:
			case DisOp:
			case DivpvOp:
			case DivvpOp:
			case DivvvOp:
			case EqpvOp:
			case EqvvOp:
			case ErfOp:
			case ExpOp:
			case Expm1Op:
			case LepvOp:
			case LevpOp:
			case LevvOp:
			case LogOp:
			case Log1pOp:
			case LtpvOp:
			case LtvpOp:
			case LtvvOp:
			case MulpvOp:
			case MulvvOp:
			case NepvOp:
			case NevvOp:
			case PowpvOp:
			case PowvpOp:
			case PowvvOp:
			case SignOp:
			case SinOp:
			case SinhOp:
			case SqrtOp:
			case SubpvOp:
			case SubvpOp:
			case SubvvOp:
			case TanOp:
			case TanhOp:
			case ZmulpvOp:
			case ZmulvpOp:
			case ZmulvvOp:
			// check for a previous match
			match_op( var2op, op_info, i_op, hash_table_op );
			if( op_info[i_op].previous != 0 )
			{	// like a unary operator that assigns i_op equal to previous.
				size_t previous = op_info[i_op].previous;
				bool sum_op = false;
				CPPAD_ASSERT_UNKNOWN( previous < i_op );
				usage_cexp_result2arg(
					sum_op, i_op, previous, op_info, cexp_set
				);
			}
			break;
		}
	}
	// ----------------------------------------------------------------------
	// compute cexp_info
	// ----------------------------------------------------------------------
	if( cexp_set.n_set() == 0 )
		return;
	//
	// initialize information for each conditional expression
	cexp_info.resize(num_cexp_op);
	skip_op_true.resize(num_cexp_op, num_op);
	skip_op_false.resize(num_cexp_op, num_op);
	//
	for(size_t i = 0; i < num_cexp_op; i++)
	{	CPPAD_ASSERT_UNKNOWN(
			op_info[i].previous == 0 || op_info[i].usage == yes_usage
		);
		i_op            = cexp2op[i];
		arg             = op_info[i_op].arg;
		CPPAD_ASSERT_UNKNOWN( op_info[i_op].op == CExpOp );
		//
		struct_cexp_info info;
		info.i_op       = i_op;
		info.cop        = CompareOp( arg[0] );
		info.flag       = arg[1];
		info.left       = arg[2];
		info.right      = arg[3];
		//
		// max_left_right
		size_t index    = 0;
		if( arg[1] & 1 )
			index = std::max(index, info.left);
		if( arg[1] & 2 )
			index = std::max(index, info.right);
		CPPAD_ASSERT_UNKNOWN( index > 0 );
		info.max_left_right = index;
		//
		cexp_info[i] = info;
	};
	// Determine which operators can be conditionally skipped
	i_op = 0;
	while(i_op < num_op)
	{	size_t j_op = i_op;
		bool keep = op_info[i_op].usage != no_usage;
		keep     &= op_info[i_op].usage != csum_usage;
		keep     &= op_info[i_op].previous == 0;
		if( keep )
		{	sparse_list_const_iterator itr(cexp_set, i_op);
			if( *itr != cexp_set.end() )
			{	if( op_info[i_op].op == UserOp )
				{	// i_op is the first operations in this user atomic call.
					// Find the last operation in this call.
					++j_op;
					while( op_info[j_op].op != UserOp )
					{	switch( op_info[j_op].op )
						{	case UsrapOp:
							case UsravOp:
							case UsrrpOp:
							case UsrrvOp:
							break;

							default:
							CPPAD_ASSERT_UNKNOWN(false);
						}
						++j_op;
					}
				}
			}
			while( *itr != cexp_set.end() )
			{	size_t element = *itr;
				size_t index   = element / 2;
				bool   compare = bool( element % 2 );
				if( compare == false )
				{	// cexp_info[index].skip_op_false.push_back(i_op);
					skip_op_false.add_element(index, i_op);
					if( j_op != i_op )
					{	// cexp_info[index].skip_op_false.push_back(j_op);
						skip_op_false.add_element(index, j_op);
					}
				}
				else
				{	// cexp_info[index].skip_op_true.push_back(i_op);
					skip_op_true.add_element(index, i_op);
					if( j_op != i_op )
					{	// cexp_info[index].skip_op_true.push_back(j_op);
						skip_op_true.add_element(index, j_op);
					}
				}
				++itr;
			}
		}
		CPPAD_ASSERT_UNKNOWN( i_op <= j_op );
		i_op += (1 + j_op) - i_op;
	}
	return;
}

} } } // END_CPPAD_LOCAL_OPTIMIZE_NAMESPACE

# endif
