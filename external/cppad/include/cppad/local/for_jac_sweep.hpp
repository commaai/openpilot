# ifndef CPPAD_LOCAL_FOR_JAC_SWEEP_HPP
# define CPPAD_LOCAL_FOR_JAC_SWEEP_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

# include <set>
# include <cppad/local/pod_vector.hpp>

namespace CppAD { namespace local { // BEGIN_CPPAD_LOCAL_NAMESPACE
/*!
\file for_jac_sweep.hpp
Compute Forward mode Jacobian sparsity patterns.
*/

/*!
\def CPPAD_FOR_JAC_SWEEP_TRACE
This value is either zero or one.
Zero is the normal operational value.
If it is one, a trace of every for_jac_sweep computation is printed.
*/
# define CPPAD_FOR_JAC_SWEEP_TRACE 0

/*!
Given the sparsity pattern for the independent variables,
ForJacSweep computes the sparsity pattern for all the other variables.

\tparam Base
base type for the operator; i.e., this operation sequence was recorded
using AD< \a Base > and computations by this routine are done using type
\a Base.

\tparam Vector_set
is the type used for vectors of sets. It can be either
sparse_pack or sparse_list.

\param dependency
Are the derivatives with respect to left and right of the expression below
considered to be non-zero:
\code
	CondExpRel(left, right, if_true, if_false)
\endcode
This is used by the optimizer to obtain the correct dependency relations.

\param n
is the number of independent variables on the tape.

\param numvar
is the total number of variables on the tape; i.e.,
\a play->num_var_rec().

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

\param var_sparsity
\b Input: For j = 1 , ... , \a n,
the sparsity pattern for the independent variable with index (j-1)
corresponds to the set with index j in \a var_sparsity.
\n
\n
\b Output: For i = \a n + 1 , ... , \a numvar - 1,
the sparsity pattern for the variable with index i on the tape
corresponds to the set with index i in \a var_sparsity.

\par Checked Assertions:
\li numvar == var_sparsity.n_set()
\li numvar == play->num_var_rec()
*/

template <class Base, class Vector_set>
void ForJacSweep(
	bool                  dependency   ,
	size_t                n            ,
	size_t                numvar       ,
	local::player<Base>*  play         ,
	Vector_set&           var_sparsity )
{
	OpCode           op;
	size_t         i_op;
	size_t        i_var;

	const addr_t*   arg = CPPAD_NULL;

	size_t            i, j, k;

	// check numvar argument
	CPPAD_ASSERT_UNKNOWN( play->num_var_rec()  == numvar );
	CPPAD_ASSERT_UNKNOWN( var_sparsity.n_set() == numvar );

	// length of the parameter vector (used by CppAD assert macros)
	const size_t num_par = play->num_par_rec();

	// cum_sparsity accumulates sparsity pattern a cummulative sum
	size_t limit = var_sparsity.end();

	// vecad_sparsity contains a sparsity pattern from each VecAD object
	// to all the other variables.
	// vecad_ind maps a VecAD index (the beginning of the
	// VecAD object) to its from index in vecad_sparsity
	size_t num_vecad_ind   = play->num_vec_ind_rec();
	size_t num_vecad_vec   = play->num_vecad_vec_rec();
	Vector_set  vecad_sparsity;
	vecad_sparsity.resize(num_vecad_vec, limit);
	pod_vector<size_t> vecad_ind;
	if( num_vecad_vec > 0 )
	{	size_t length;
		vecad_ind.extend(num_vecad_ind);
		j             = 0;
		for(i = 0; i < num_vecad_vec; i++)
		{	// length of this VecAD
			length   = play->GetVecInd(j);
			// set to proper index for this VecAD
			vecad_ind[j] = i;
			for(k = 1; k <= length; k++)
				vecad_ind[j+k] = num_vecad_vec; // invalid index
			// start of next VecAD
			j       += length + 1;
		}
		CPPAD_ASSERT_UNKNOWN( j == play->num_vec_ind_rec() );
	}

	// --------------------------------------------------------------
	// user's atomic op calculator
	atomic_base<Base>* user_atom = CPPAD_NULL;
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
	// --------------------------------------------------------------
	//
	// pointer to the beginning of the parameter vector
	// (used by user atomic functions)
	const Base* parameter = CPPAD_NULL;
	if( num_par > 0 )
		parameter = play->GetPar();

# if CPPAD_FOR_JAC_SWEEP_TRACE
	vector<size_t>    user_usrrp; // parameter index for UsrrpOp operators
	std::cout << std::endl;
	CppAD::vectorBool z_value(limit);
# endif

	// skip the BeginOp at the beginning of the recording
	play->forward_start(op, arg, i_op, i_var);
	CPPAD_ASSERT_UNKNOWN( op == BeginOp );
	bool more_operators = true;
	while(more_operators)
	{	bool flag; // temporary for use in switch cases.

		// this op
		play->forward_next(op, arg, i_op, i_var);
		CPPAD_ASSERT_UNKNOWN( (i_op > n)  | (op == InvOp) );
		CPPAD_ASSERT_UNKNOWN( (i_op <= n) | (op != InvOp) );
		CPPAD_ASSERT_ARG_BEFORE_RESULT(op, arg, i_var);

		// rest of information depends on the case
		switch( op )
		{
			case AbsOp:
			CPPAD_ASSERT_NARG_NRES(op, 1, 1);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
			// -------------------------------------------------

			case AddvvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1);
			forward_sparse_jacobian_binary_op(
				i_var, arg, var_sparsity
			);
			break;
			// -------------------------------------------------

			case AddpvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1);
			forward_sparse_jacobian_unary_op(
				i_var, arg[1], var_sparsity
			);
			break;
			// -------------------------------------------------

			case AcosOp:
			// sqrt(1 - x * x), acos(x)
			CPPAD_ASSERT_NARG_NRES(op, 1, 2);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
			// -------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case AcoshOp:
			// sqrt(x * x - 1), acosh(x)
			CPPAD_ASSERT_NARG_NRES(op, 1, 2);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
# endif
			// -------------------------------------------------

			case AsinOp:
			// sqrt(1 - x * x), asin(x)
			CPPAD_ASSERT_NARG_NRES(op, 1, 2);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
			// -------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case AsinhOp:
			// sqrt(1 + x * x), asinh(x)
			CPPAD_ASSERT_NARG_NRES(op, 1, 2);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
# endif
			// -------------------------------------------------

			case AtanOp:
			// 1 + x * x, atan(x)
			CPPAD_ASSERT_NARG_NRES(op, 1, 2);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
			// -------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case AtanhOp:
			// 1 - x * x, atanh(x)
			CPPAD_ASSERT_NARG_NRES(op, 1, 2);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
# endif
			// -------------------------------------------------

			case CSkipOp:
			// CSipOp has a variable number of arguments and
			// forward_next thinks it has no arguments.
			// we must inform forward_next of this special case.
			play->forward_cskip(op, arg, i_op, i_var);
			break;
			// -------------------------------------------------

			case CSumOp:
			// CSumOp has a variable number of arguments and
			// forward_next thinks it has no arguments.
			// we must inform forward_next of this special case.
			forward_sparse_jacobian_csum_op(
				i_var, arg, var_sparsity
			);
			play->forward_csum(op, arg, i_op, i_var);
			break;
			// -------------------------------------------------

			case CExpOp:
			forward_sparse_jacobian_cond_op(
				dependency, i_var, arg, num_par, var_sparsity
			);
			break;
			// --------------------------------------------------

			case CosOp:
			// sin(x), cos(x)
			CPPAD_ASSERT_NARG_NRES(op, 1, 2);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
			// ---------------------------------------------------

			case CoshOp:
			// sinh(x), cosh(x)
			CPPAD_ASSERT_NARG_NRES(op, 1, 2);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
			// -------------------------------------------------

			case DisOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1);
			// derivative is identically zero but dependency is not
			if( dependency ) forward_sparse_jacobian_unary_op(
				i_var, arg[1], var_sparsity
			);
			else
				var_sparsity.clear(i_var);
			break;
			// -------------------------------------------------

			case DivvvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1);
			forward_sparse_jacobian_binary_op(
				i_var, arg, var_sparsity
			);
			break;
			// -------------------------------------------------

			case DivpvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1);
			forward_sparse_jacobian_unary_op(
				i_var, arg[1], var_sparsity
			);
			break;
			// -------------------------------------------------

			case DivvpOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
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
			// arg[0] is always the parameter 2 / sqrt(pi)
			CPPAD_ASSERT_NARG_NRES(op, 3, 5);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
			// -------------------------------------------------

			case ExpOp:
			CPPAD_ASSERT_NARG_NRES(op, 1, 1);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
			// -------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case Expm1Op:
			CPPAD_ASSERT_NARG_NRES(op, 1, 1);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
# endif
			// -------------------------------------------------

			case InvOp:
			CPPAD_ASSERT_NARG_NRES(op, 0, 1);
			// sparsity pattern is already defined
			break;
			// -------------------------------------------------

			case LdpOp:
			forward_sparse_load_op(
				dependency,
				op,
				i_var,
				arg,
				num_vecad_ind,
				vecad_ind.data(),
				var_sparsity,
				vecad_sparsity
			);
			break;
			// -------------------------------------------------

			case LdvOp:
			forward_sparse_load_op(
				dependency,
				op,
				i_var,
				arg,
				num_vecad_ind,
				vecad_ind.data(),
				var_sparsity,
				vecad_sparsity
			);
			break;
			// -------------------------------------------------

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

			case LogOp:
			CPPAD_ASSERT_NARG_NRES(op, 1, 1);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
			// -------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case Log1pOp:
			CPPAD_ASSERT_NARG_NRES(op, 1, 1);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
# endif
			// -------------------------------------------------

			case MulpvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1);
			forward_sparse_jacobian_unary_op(
				i_var, arg[1], var_sparsity
			);
			break;
			// -------------------------------------------------

			case MulvvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1);
			forward_sparse_jacobian_binary_op(
				i_var, arg, var_sparsity
			);
			break;
			// -------------------------------------------------

			case ParOp:
			CPPAD_ASSERT_NARG_NRES(op, 1, 1);
			var_sparsity.clear(i_var);
			break;
			// -------------------------------------------------

			case PowvpOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 3);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
			// -------------------------------------------------

			case PowpvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 3);
			forward_sparse_jacobian_unary_op(
				i_var, arg[1], var_sparsity
			);
			break;
			// -------------------------------------------------

			case PowvvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 3);
			forward_sparse_jacobian_binary_op(
				i_var, arg, var_sparsity
			);
			break;
			// -------------------------------------------------

			case PriOp:
			CPPAD_ASSERT_NARG_NRES(op, 5, 0);
			break;
			// -------------------------------------------------

			case SignOp:
			CPPAD_ASSERT_NARG_NRES(op, 1, 1);
			// derivative is identically zero but dependency is not
			if( dependency ) forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			else
				var_sparsity.clear(i_var);
			break;
			// -------------------------------------------------

			case SinOp:
			// cos(x), sin(x)
			CPPAD_ASSERT_NARG_NRES(op, 1, 2);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
			// -------------------------------------------------

			case SinhOp:
			// cosh(x), sinh(x)
			CPPAD_ASSERT_NARG_NRES(op, 1, 2);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
			// -------------------------------------------------

			case SqrtOp:
			CPPAD_ASSERT_NARG_NRES(op, 1, 1);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
			// -------------------------------------------------

			case StppOp:
			CPPAD_ASSERT_NARG_NRES(op, 3, 0);
			// if both arguments are parameters does not affect sparsity
			// or dependency
			break;
			// -------------------------------------------------

			case StpvOp:
			forward_sparse_store_op(
				dependency,
				op,
				arg,
				num_vecad_ind,
				vecad_ind.data(),
				var_sparsity,
				vecad_sparsity
			);
			break;
			// -------------------------------------------------

			case StvpOp:
			CPPAD_ASSERT_NARG_NRES(op, 3, 0);
			forward_sparse_store_op(
				dependency,
				op,
				arg,
				num_vecad_ind,
				vecad_ind.data(),
				var_sparsity,
				vecad_sparsity
			);
			break;
			// -------------------------------------------------

			case StvvOp:
			forward_sparse_store_op(
				dependency,
				op,
				arg,
				num_vecad_ind,
				vecad_ind.data(),
				var_sparsity,
				vecad_sparsity
			);
			break;
			// -------------------------------------------------

			case SubvvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1);
			forward_sparse_jacobian_binary_op(
				i_var, arg, var_sparsity
			);
			break;
			// -------------------------------------------------

			case SubpvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1);
			forward_sparse_jacobian_unary_op(
				i_var, arg[1], var_sparsity
			);
			break;
			// -------------------------------------------------

			case SubvpOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
			// -------------------------------------------------

			case TanOp:
			// tan(x)^2, tan(x)
			CPPAD_ASSERT_NARG_NRES(op, 1, 2);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
			// -------------------------------------------------

			case TanhOp:
			// tanh(x)^2, tanh(x)
			CPPAD_ASSERT_NARG_NRES(op, 1, 2);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
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
# if CPPAD_FOR_JAC_SWEEP_TRACE
				user_usrrp.resize( user_m );
# endif
			}
			else
			{	// end of user atomic operation sequence
				user_atom->set_old(user_old);
				user_atom->for_sparse_jac(
					user_x, user_ix, user_iy, var_sparsity
				);
			}
			break;

			case UsrapOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			// argument parameter value
			user_x[user_j]  = parameter[arg[0]];
			// special variable index used for parameters
			user_ix[user_j] = 0;
			//
			play->forward_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			break;

			case UsravOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) <= i_var );
			CPPAD_ASSERT_UNKNOWN( 0 < arg[0] );
			// argument variables not avaiable during sparsity calculations
			user_x[user_j]  = CppAD::numeric_limits<Base>::quiet_NaN();
			// variable index for this argument
			user_ix[user_j] = arg[0];
			//
			play->forward_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			break;

			case UsrrpOp:
			// special variable index used for parameters
			user_iy[user_i] = 0;
# if CPPAD_FOR_JAC_SWEEP_TRACE
			// remember argument for delayed tracing
			user_usrrp[user_i] = arg[0];
# endif
			play->forward_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			break;

			case UsrrvOp:
			// variable index for this result
			user_iy[user_i] = i_var;
			play->forward_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			break;
			// -------------------------------------------------

			case ZmulpvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1);
			forward_sparse_jacobian_unary_op(
				i_var, arg[1], var_sparsity
			);
			break;
			// -------------------------------------------------

			case ZmulvpOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1);
			forward_sparse_jacobian_unary_op(
				i_var, arg[0], var_sparsity
			);
			break;
			// -------------------------------------------------

			case ZmulvvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1);
			forward_sparse_jacobian_binary_op(
				i_var, arg, var_sparsity
			);
			break;
			// -------------------------------------------------

			default:
			CPPAD_ASSERT_UNKNOWN(0);
		}
# if CPPAD_FOR_JAC_SWEEP_TRACE
		if( op == UserOp && user_state == start_user )
		{	// print operators that have been delayed
			CPPAD_ASSERT_UNKNOWN( user_m == user_iy.size() );
			CPPAD_ASSERT_UNKNOWN( i_op > user_m );
			CPPAD_ASSERT_NARG_NRES(UsrrpOp, 1, 0);
			CPPAD_ASSERT_NARG_NRES(UsrrvOp, 0, 1);
			addr_t arg_tmp[1];
			for(i = 0; i < user_m; i++)
			{	size_t j_var = user_iy[i];
				// value for this variable
				for(j = 0; j < limit; j++)
					z_value[j] = false;
				typename Vector_set::const_iterator itr(var_sparsity, j_var);
				j = *itr;
				while( j < limit )
				{	z_value[j] = true;
					j = *(++itr);
				}
				OpCode op_tmp = UsrrvOp;
				if( j_var == 0 )
				{	op_tmp     = UsrrpOp;
					arg_tmp[0] = user_usrrp[i];
				}
				// j_var is zero when there is no result.
				printOp(
					std::cout,
					play,
					i_op - user_m + i,
					j_var,
					op_tmp,
					arg_tmp
				);
				if( j_var > 0 ) printOpResult(
					std::cout,
					1,
					&z_value,
					0,
					(CppAD::vectorBool *) CPPAD_NULL
				);
				std::cout << std::endl;
			}
		}
		const addr_t*   arg_tmp = arg;
		if( op == CSumOp )
			arg_tmp = arg - arg[-1] - 4;
		if( op == CSkipOp )
			arg_tmp = arg - arg[-1] - 7;
		//
		// value for this variable
		for(j = 0; j < limit; j++)
			z_value[j] = false;
		typename Vector_set::const_iterator itr(var_sparsity, i_var);
		j = *itr;
		while( j < limit )
		{	z_value[j] = true;
			j = *(++itr);
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
				&z_value,
				0,
				(CppAD::vectorBool *) CPPAD_NULL
			);
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
# else
	}
# endif
	CPPAD_ASSERT_UNKNOWN( i_var + 1 == play->num_var_rec() );

	return;
}

} } // END_CPPAD_LOCAL_NAMESPACE

// preprocessor symbols that are local to this file
# undef CPPAD_FOR_JAC_SWEEP_TRACE

# endif
