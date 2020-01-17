# ifndef CPPAD_LOCAL_FOR_HES_SWEEP_HPP
# define CPPAD_LOCAL_FOR_HES_SWEEP_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

namespace CppAD { namespace local { // BEGIN_CPPAD_LOCAL_NAMESPACE
/*!
\file for_hes_sweep.hpp
Compute Forward mode Hessian sparsity patterns.
*/

/*!
\def CPPAD_FOR_HES_SWEEP_TRACE
This value is either zero or one.
Zero is the normal operational value.
If it is one, a trace of every rev_hes_sweep computation is printed.
*/
# define CPPAD_FOR_HES_SWEEP_TRACE 0

/*!
Given the forward Jacobian sparsity pattern for all the variables,
and the reverse Jacobian sparsity pattern for the dependent variables,
ForHesSweep computes the Hessian sparsity pattern for all the independent
variables.

\tparam Base
base type for the operator; i.e., this operation sequence was recorded
using AD< \a Base > and computations by this routine are done using type
\a Base.

\tparam Vector_set
is the type used for vectors of sets. It can be either
sparse_pack or sparse_list.

\param n
is the number of independent variables on the tape.

\param numvar
is the total number of variables on the tape; i.e.,
\a play->num_var_rec().
This is also the number of rows in the entire sparsity pattern
\a for_hes_sparse.

\param play
The information stored in \a play
is a recording of the operations corresponding to a function
\f[
	F : {\bf R}^n \rightarrow {\bf R}^m
\f]
where \f$ n \f$ is the number of independent variables
and \f$ m \f$ is the number of dependent variables.
The object \a play is effectly constant.
It is not declared const because while playing back the tape
the object \a play holds information about the current location
with in the tape and this changes during playback.

\param for_jac_sparse
For i = 0 , ... , \a numvar - 1,
(for all the variables on the tape),
the forward Jacobian sparsity pattern for the variable with index i
corresponds to the set with index i in \a for_jac_sparse.

\param rev_jac_sparse
\b Input:
For i = 0, ... , \a numvar - 1
the if the function we are computing the Hessian for has a non-zero
derivative w.r.t. variable with index i,
the set with index i has element zero.
Otherwise it has no elements.

\param for_hes_sparse
The forward Hessian sparsity pattern for the variable with index i
corresponds to the set with index i in \a for_hes_sparse.
The number of rows in this sparsity patter is n+1 and the row
with index zero is not used.
\n
\n
\b Input: For i = 1 , ... , \a n
the forward Hessian sparsity pattern for the variable with index i is empty.
\n
\n
\b Output: For j = 1 , ... , \a n,
the forward Hessian sparsity pattern for the independent dependent variable
with index (j-1) is given by the set with index j
in \a for_hes_sparse.
*/

template <class Base, class Vector_set>
void ForHesSweep(
	size_t                n,
	size_t                numvar,
	local::player<Base>*  play,
	const Vector_set&     for_jac_sparse,
	const Vector_set&     rev_jac_sparse,
	Vector_set&           for_hes_sparse
)
{
	OpCode           op;
	size_t         i_op;
	size_t        i_var;

	const addr_t*   arg = CPPAD_NULL;

	// length of the parameter vector (used by CppAD assert macros)
	const size_t num_par = play->num_par_rec();

	size_t             i, j, k;

	// check numvar argument
	size_t limit = n+1;
	CPPAD_ASSERT_UNKNOWN( play->num_var_rec()    == numvar );
	CPPAD_ASSERT_UNKNOWN( for_jac_sparse.n_set() == numvar );
	CPPAD_ASSERT_UNKNOWN( for_hes_sparse.n_set() == limit );
	CPPAD_ASSERT_UNKNOWN( numvar > 0 );

	// upper limit exclusive for set elements
	CPPAD_ASSERT_UNKNOWN( for_jac_sparse.end() == limit );
	CPPAD_ASSERT_UNKNOWN( for_hes_sparse.end() == limit );

	// vecad_sparsity contains a sparsity pattern for each VecAD object.
	// vecad_ind maps a VecAD index (beginning of the VecAD object)
	// to the index for the corresponding set in vecad_sparsity.
	size_t num_vecad_ind   = play->num_vec_ind_rec();
	size_t num_vecad_vec   = play->num_vecad_vec_rec();
	Vector_set vecad_sparse;
	vecad_sparse.resize(num_vecad_vec, limit);
	pod_vector<size_t> vecad_ind;
	pod_vector<bool>   vecad_jac;
	if( num_vecad_vec > 0 )
	{	size_t length;
		vecad_ind.extend(num_vecad_ind);
		vecad_jac.extend(num_vecad_vec);
		j             = 0;
		for(i = 0; i < num_vecad_vec; i++)
		{	// length of this VecAD
			length   = play->GetVecInd(j);
			// set vecad_ind to proper index for this VecAD
			vecad_ind[j] = i;
			// make all other values for this vector invalid
			for(k = 1; k <= length; k++)
				vecad_ind[j+k] = num_vecad_vec;
			// start of next VecAD
			j       += length + 1;
			// initialize this vector's reverse jacobian value
			vecad_jac[i] = false;
		}
		CPPAD_ASSERT_UNKNOWN( j == play->num_vec_ind_rec() );
	}
	// ------------------------------------------------------------------------
	// user's atomic op calculator
	atomic_base<Base>* user_atom = CPPAD_NULL; // user's atomic op calculator
	//
	// work space used by UserOp.
	vector<Base>       user_x;   // value of parameter arguments to function
	vector<size_t>     user_ix;  // variable index (on tape) for each argument
	vector<size_t>     user_iy;  // variable index (on tape) for each result
	//
	// information set by forward_user (initialization to avoid warnings)
	size_t user_old=0, user_m=0, user_n=0, user_i=0, user_j=0;
	// information set by forward_user (necessary initialization)
	enum_user_state user_state = start_user;
	// -------------------------------------------------------------------------
	//
	// pointer to the beginning of the parameter vector
	// (used by user atomic functions)
	const Base* parameter = CPPAD_NULL;
	if( num_par > 0 )
		parameter = play->GetPar();

	// Initialize
	play->forward_start(op, arg, i_op, i_var);
	CPPAD_ASSERT_UNKNOWN( op == BeginOp );
	bool more_operators = true;
# if CPPAD_FOR_HES_SWEEP_TRACE
	vector<size_t> user_usrrp; // parameter index for UsrrpOp operators
	std::cout << std::endl;
	CppAD::vectorBool zf_value(limit);
	CppAD::vectorBool zh_value(limit * limit);
# endif
	bool flag; // temporary for use in switch cases below
	while(more_operators)
	{
		// next op
		play->forward_next(op, arg, i_op, i_var);
# ifndef NDEBUG
		if( i_op <= n )
		{	CPPAD_ASSERT_UNKNOWN((op == InvOp) | (op == BeginOp));
		}
		else	CPPAD_ASSERT_UNKNOWN((op != InvOp) & (op != BeginOp));
# endif

		// does the Hessian in question have a non-zero derivative
		// with respect to this variable
		bool include = rev_jac_sparse.is_element(i_var, 0);
		//
		// operators to include even if derivative is zero
		include |= op == EndOp;
		include |= op == CSkipOp;
		include |= op == CSumOp;
		include |= op == UserOp;
		include |= op == UsrapOp;
		include |= op == UsravOp;
		include |= op == UsrrpOp;
		include |= op == UsrrvOp;
		//
		if( include ) switch( op )
		{	// operators that should not occurr
			// case BeginOp
			// -------------------------------------------------

			// operators that do not affect hessian
			case AbsOp:
			case AddvvOp:
			case AddpvOp:
			case CExpOp:
			case DisOp:
			case DivvpOp:
			case InvOp:
			case LdpOp:
			case LdvOp:
			case MulpvOp:
			case ParOp:
			case PriOp:
			case SignOp:
			case StppOp:
			case StpvOp:
			case StvpOp:
			case StvvOp:
			case SubvvOp:
			case SubpvOp:
			case SubvpOp:
			case ZmulpvOp:
			case ZmulvpOp:
			break;
			// -------------------------------------------------

			// nonlinear unary operators
			case AcosOp:
			case AsinOp:
			case AtanOp:
			case CosOp:
			case CoshOp:
			case ExpOp:
			case LogOp:
			case SinOp:
			case SinhOp:
			case SqrtOp:
			case TanOp:
			case TanhOp:
# if CPPAD_USE_CPLUSPLUS_2011
			case AcoshOp:
			case AsinhOp:
			case AtanhOp:
			case Expm1Op:
			case Log1pOp:
# endif
			CPPAD_ASSERT_UNKNOWN( NumArg(op) == 1 )
			forward_sparse_hessian_nonlinear_unary_op(
				arg[0], for_jac_sparse, for_hes_sparse
			);
			break;
			// -------------------------------------------------

			case CSkipOp:
			// CSkipOp has a variable number of arguments and
			// reverse_next thinks it one has one argument.
			// We must inform reverse_next of this special case.
			play->forward_cskip(op, arg, i_op, i_var);
			break;
			// -------------------------------------------------

			case CSumOp:
			// CSumOp has a variable number of arguments and
			// reverse_next thinks it one has one argument.
			// We must inform reverse_next of this special case.
			play->forward_csum(op, arg, i_op, i_var);
			break;
			// -------------------------------------------------

			case DivvvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1)
			forward_sparse_hessian_div_op(
				arg, for_jac_sparse, for_hes_sparse
			);
			break;
			// -------------------------------------------------

			case DivpvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1)
			forward_sparse_hessian_nonlinear_unary_op(
				arg[1], for_jac_sparse, for_hes_sparse
			);
			break;
			// -------------------------------------------------

			case EndOp:
			CPPAD_ASSERT_NARG_NRES(op, 0, 0);
			more_operators = false;
			break;
			// -------------------------------------------------

			case ErfOp:
			// arg[1] is always the parameter 0
			// arg[2] is always the parameter 2 / sqrt(pi)
			CPPAD_ASSERT_NARG_NRES(op, 3, 5);
			forward_sparse_hessian_nonlinear_unary_op(
				arg[0], for_jac_sparse, for_hes_sparse
			);
			break;
			// -------------------------------------------------

			// -------------------------------------------------
			// logical comparision operators
			case EqpvOp:
			case EqvvOp:
			case LtpvOp:
			case LtvpOp:
			case LtvvOp:
			case LepvOp:
			case LevpOp:
			case LevvOp:
			case NepvOp:
			case NevvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 0);
			break;
			// -------------------------------------------------

			case MulvvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1)
			forward_sparse_hessian_mul_op(
				arg, for_jac_sparse, for_hes_sparse
			);
			break;
			// -------------------------------------------------

			case PowpvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 3)
			forward_sparse_hessian_nonlinear_unary_op(
				arg[1], for_jac_sparse, for_hes_sparse
			);
			break;
			// -------------------------------------------------

			case PowvpOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 3)
			forward_sparse_hessian_nonlinear_unary_op(
				arg[0], for_jac_sparse, for_hes_sparse
			);
			break;
			// -------------------------------------------------

			case PowvvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 3)
			forward_sparse_hessian_pow_op(
				arg, for_jac_sparse, for_hes_sparse
			);
			break;
			// -------------------------------------------------

			case UserOp:
			CPPAD_ASSERT_UNKNOWN(
				user_state == start_user || user_state == end_user
			);
			flag = user_state == start_user;
			user_atom = play->forward_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			if( flag )
			{	// start of user atomic operation sequence
				user_x.resize( user_n );
				user_ix.resize( user_n );
				user_iy.resize( user_m );
# if CPPAD_FOR_HES_SWEEP_TRACE
				user_usrrp.resize( user_m );
# endif
			}
			else
			{	// end of user atomic operation sequence
				user_atom->set_old(user_old);
				user_atom->for_sparse_hes(
					user_x, user_ix, user_iy,
					for_jac_sparse, rev_jac_sparse, for_hes_sparse
				);
			}
			break;

			case UsrapOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			// argument parameter value
			user_x[user_j] = parameter[arg[0]];
			// special variable user for parameters
			user_ix[user_j] = 0;
			//
			play->forward_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			break;

			case UsravOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) <= i_var );
			CPPAD_ASSERT_UNKNOWN( 0 < arg[0] );
			// arguemnt varialbes not avaialbe during sparisty calculations
			user_x[user_j] = CppAD::numeric_limits<Base>::quiet_NaN();
			// varialbe index for this argument
			user_ix[user_j] = arg[0];
			//
			play->forward_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			break;

			case UsrrpOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			// special variable index user for parameters
			user_iy[user_i] = 0;
# if CPPAD_FOR_HES_SWEEP_TRACE
			// remember argument for delayed tracing
			user_usrrp[user_i] = arg[0];
# endif
			//
			play->forward_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			break;

			case UsrrvOp:
			// variable index for this result
			user_iy[user_i] = i_var;
			//
			play->forward_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			break;
			// -------------------------------------------------

			case ZmulvvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1)
			forward_sparse_hessian_mul_op(
				arg, for_jac_sparse, for_hes_sparse
			);
			break;

			// -------------------------------------------------

			default:
			CPPAD_ASSERT_UNKNOWN(0);
		}
# if CPPAD_FOR_HES_SWEEP_TRACE
		typedef typename Vector_set::const_iterator const_iterator;
		if( op == UserOp && user_state == start_user )
		{	// print operators that have been delayed
			CPPAD_ASSERT_UNKNOWN( user_m == user_iy.size() );
			CPPAD_ASSERT_UNKNOWN( i_op > user_m );
			CPPAD_ASSERT_NARG_NRES(UsrrpOp, 1, 0);
			CPPAD_ASSERT_NARG_NRES(UsrrvOp, 0, 1);
			addr_t arg_tmp[1];
			for(k = 0; k < user_m; k++)
			{	size_t k_var = user_iy[k];
				// value for this variable
				for(i = 0; i < limit; i++)
				{	zf_value[i] = false;
					for(j = 0; j < limit; j++)
						zh_value[i * limit + j] = false;
				}
				const_iterator itr_1(for_jac_sparse, i_var);
				j = *itr_1;
				while( j < limit )
				{	zf_value[j] = true;
					j = *(++itr_1);
				}
				for(i = 0; i < limit; i++)
				{	const_iterator itr_2(for_hes_sparse, i);
					j = *itr_2;
					while( j < limit )
					{	zh_value[i * limit + j] = true;
						j = *(++itr_2);
					}
				}
				OpCode op_tmp = UsrrvOp;
				if( k_var == 0 )
				{	op_tmp     = UsrrpOp;
					arg_tmp[0] = user_usrrp[k];
				}
				// k_var is zero when there is no result
				printOp(
					std::cout,
					play,
					i_op - user_m + k,
					k_var,
					op_tmp,
					arg_tmp
				);
				if( k_var > 0 ) printOpResult(
					std::cout,
					1,
					&zf_value,
					1,
					&zh_value
				);
				std::cout << std::endl;
			}
		}
		const addr_t*   arg_tmp = arg;
		if( op == CSumOp )
			arg_tmp = arg - arg[-1] - 4;
		if( op == CSkipOp )
			arg_tmp = arg - arg[-1] - 7;
		for(i = 0; i < limit; i++)
		{	zf_value[i] = false;
			for(j = 0; j < limit; j++)
				zh_value[i * limit + j] = false;
		}
		const_iterator itr_1(for_jac_sparse, i_var);
		j = *itr_1;
		while( j < limit )
		{	zf_value[j] = true;
			j = *(++itr_1);
		}
		for(i = 0; i < limit; i++)
		{	const_iterator itr_2(for_hes_sparse, i);
			j = *itr_2;
			while( j < limit )
			{	zh_value[i * limit + j] = true;
				j = *(++itr_2);
			}
		}
		// must delay print for these cases till after atomic user call
		bool delay_print = op == UsrrpOp;
		delay_print     |= op == UsrrvOp;
		if( ! delay_print )
		{	 printOp(
				std::cout,
				play,
				i_op,
				i_var,
				op,
				arg_tmp
			);
			if( NumRes(op) > 0 && (! delay_print) ) printOpResult(
				std::cout,
				1,
				&zf_value,
				1,
				&zh_value
			);
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
# else
	}
# endif
	// value corresponding to EndOp
	CPPAD_ASSERT_UNKNOWN( i_var + 1 == play->num_var_rec() );

	return;
}
} } // END_CPPAD_LOCAL_NAMESPACE

// preprocessor symbols that are local to this file
# undef CPPAD_FOR_HES_SWEEP_TRACE

# endif
