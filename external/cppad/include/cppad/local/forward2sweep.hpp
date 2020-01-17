# ifndef CPPAD_LOCAL_FORWARD2SWEEP_HPP
# define CPPAD_LOCAL_FORWARD2SWEEP_HPP

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
\file forward2sweep.hpp
Compute one Taylor coefficient for each direction requested.
*/

/*
\def CPPAD_ATOMIC_CALL
This avoids warnings when NDEBUG is defined and user_ok is not used.
If NDEBUG is defined, this resolves to
\code
	user_atom->forward
\endcode
otherwise, it respolves to
\code
	user_ok = user_atom->forward
\endcode
This macro is undefined at the end of this file to facillitate its
use with a different definition in other files.
*/
# ifdef NDEBUG
# define CPPAD_ATOMIC_CALL user_atom->forward
# else
# define CPPAD_ATOMIC_CALL user_ok = user_atom->forward
# endif

/*!
\def CPPAD_FORWARD2SWEEP_TRACE
This value is either zero or one.
Zero is the normal operational value.
If it is one, a trace of every forward2sweep computation is printed.
*/
# define CPPAD_FORWARD2SWEEP_TRACE 0

/*!
Compute multiple directions forward mode Taylor coefficients.

\tparam Base
The type used during the forward mode computations; i.e., the corresponding
recording of operations used the type AD<Base>.

\param q
is the order of the Taylor coefficients
that are computed during this call;
<code>q > 0</code>.

\param r
is the number of Taylor coefficients
that are computed during this call.

\param n
is the number of independent variables on the tape.

\param numvar
is the total number of variables on the tape.
This is also equal to the number of rows in the matrix taylor; i.e.,
play->num_var_rec().

\param play
The information stored in play
is a recording of the operations corresponding to the function
\f[
	F : {\bf R}^n \rightarrow {\bf R}^m
\f]
where \f$ n \f$ is the number of independent variables and
\f$ m \f$ is the number of dependent variables.
\n
\n
The object play is effectly constant.
The exception to this is that while palying back the tape
the object play holds information about the current location
with in the tape and this changes during palyback.

\param J
Is the number of columns in the coefficient matrix taylor.
This must be greater than or equal one.

\param taylor
\n
\b Input:
For <code>i = 1 , ... , numvar-1</code>,
<code>taylor[ (J-1)*r*i + i + 0 ]</code>
is the zero order Taylor coefficient corresponding to
the i-th variable and all directions.
For <code>i = 1 , ... , numvar-1</code>,
For <code>k = 1 , ... , q-1</code>,
<code>ell = 0 , ... , r-1</code>,
<code>taylor[ (J-1)*r*i + i + (k-1)*r + ell + 1 ]</code>
is the k-th order Taylor coefficient corresponding to
the i-th variabel and ell-th direction.
\n
\n
\b Input:
For <code>i = 1 , ... , n</code>,
<code>ell = 0 , ... , r-1</code>,
<code>taylor[ (J-1)*r*i + i + (q-1)*r + ell + 1 ]</code>
is the q-th order Taylor coefficient corresponding to
the i-th variable and ell-th direction
(these are the independent varaibles).
\n
\n
\b Output:
For <code>i = n+1 , ... , numvar-1</code>,
<code>ell = 0 , ... , r-1</code>,
<code>taylor[ (J-1)*r*i + i + (q-1)*r + ell + 1 ]</code>
is the q-th order Taylor coefficient corresponding to
the i-th variable and ell-th direction.

\param cskip_op
Is a vector with size play->num_op_rec().
If cskip_op[i] is true, the operator with index i
does not affect any of the dependent variable (given the value
of the independent variables).

\param var_by_load_op
is a vector with size play->num_load_op_rec().
It is the variable index corresponding to each the
load instruction.
In the case where the index is zero,
the instruction corresponds to a parameter (not variable).

*/

template <class Base>
void forward2sweep(
	const size_t                q,
	const size_t                r,
	const size_t                n,
	const size_t                numvar,
	      local::player<Base>*  play,
	const size_t                J,
	      Base*                 taylor,
	const bool*                 cskip_op,
	const pod_vector<addr_t>&   var_by_load_op
)
{
	CPPAD_ASSERT_UNKNOWN( q > 0 );
	CPPAD_ASSERT_UNKNOWN( J >= q + 1 );
	CPPAD_ASSERT_UNKNOWN( play->num_var_rec() == numvar );

	// used to avoid compiler errors until all operators are implemented
	size_t p = q;

	// op code for current instruction
	OpCode op;

	// index for current instruction
	size_t i_op;

	// next variables
	size_t i_var;

	// operation argument indices
	const addr_t*   arg = CPPAD_NULL;

	// work space used by UserOp.
	vector<bool> user_vx;        // empty vecotor
	vector<bool> user_vy;        // empty vecotor
	vector<Base> user_tx_one;    // argument vector Taylor coefficients
	vector<Base> user_tx_all;
	vector<Base> user_ty_one;    // result vector Taylor coefficients
	vector<Base> user_ty_all;
	//
	// information defined by forward_user
	size_t user_old=0, user_m=0, user_n=0, user_i=0, user_j=0;
	enum_user_state user_state = start_user; // proper initialization
	//
	atomic_base<Base>* user_atom = CPPAD_NULL; // user's atomic op calculator
# ifndef NDEBUG
	bool               user_ok   = false;      // atomic op return value
# endif

	// length of the parameter vector (used by CppAD assert macros)
	const size_t num_par = play->num_par_rec();

	// pointer to the beginning of the parameter vector
	const Base* parameter = CPPAD_NULL;
	if( num_par > 0 )
		parameter = play->GetPar();

	// temporary indices
	size_t i, j, k, ell;

	// number of orders for this user calculation
	// (not needed for order zero)
	const size_t user_q1 = q+1;

	// variable indices for results vector
	// (done differently for order zero).
	vector<size_t> user_iy;

	// skip the BeginOp at the beginning of the recording
	play->forward_start(op, arg, i_op, i_var);
	CPPAD_ASSERT_UNKNOWN( op == BeginOp );
# if CPPAD_FORWARD2SWEEP_TRACE
	bool user_trace  = false;
	std::cout << std::endl;
	CppAD::vector<Base> Z_vec(q+1);
# endif
	bool flag; // a temporary flag to use in switch cases
	bool more_operators = true;
	while(more_operators)
	{
		// this op
		play->forward_next(op, arg, i_op, i_var);
		CPPAD_ASSERT_UNKNOWN( (i_op > n)  | (op == InvOp) );
		CPPAD_ASSERT_UNKNOWN( (i_op <= n) | (op != InvOp) );
		CPPAD_ASSERT_UNKNOWN( i_op < play->num_op_rec() );
		CPPAD_ASSERT_ARG_BEFORE_RESULT(op, arg, i_var);

		// check if we are skipping this operation
		while( cskip_op[i_op] )
		{	switch(op)
			{	case CSumOp:
				// CSumOp has a variable number of arguments
				play->forward_csum(op, arg, i_op, i_var);
				break;

				case CSkipOp:
				// CSkip has a variable number of arguments
				play->forward_cskip(op, arg, i_op, i_var);
				break;

				case UserOp:
				{	// skip all operations in this user atomic call
					CPPAD_ASSERT_UNKNOWN( user_state == start_user );
					play->forward_user(op, user_state,
						user_old, user_m, user_n, user_i, user_j
					);
					size_t n_skip = user_m + user_n + 1;
					for(i = 0; i < n_skip; i++)
					{	play->forward_next(op, arg, i_op, i_var);
						play->forward_user(op, user_state,
							user_old, user_m, user_n, user_i, user_j
						);
					}
					CPPAD_ASSERT_UNKNOWN( user_state == start_user );
				}
				break;

				default:
				break;
			}
			play->forward_next(op, arg, i_op, i_var);
			CPPAD_ASSERT_UNKNOWN( i_op < play->num_op_rec() );
		}

		// action depends on the operator
		switch( op )
		{
			case AbsOp:
			forward_abs_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

			case AddvvOp:
			forward_addvv_op_dir(q, r, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case AddpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			forward_addpv_op_dir(q, r, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case AcosOp:
			// sqrt(1 - x * x), acos(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_acos_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case AcoshOp:
			// sqrt(x * x - 1), acosh(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_acosh_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
# endif
			// -------------------------------------------------

			case AsinOp:
			// sqrt(1 - x * x), asin(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_asin_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case AsinhOp:
			// sqrt(1 + x * x), asinh(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_asinh_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
# endif
			// -------------------------------------------------

			case AtanOp:
			// 1 + x * x, atan(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_atan_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case AtanhOp:
			// 1 - x * x, atanh(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_atanh_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
# endif
			// -------------------------------------------------

			case CExpOp:
			forward_cond_op_dir(
				q, r, i_var, arg, num_par, parameter, J, taylor
			);
			break;
			// ---------------------------------------------------

			case CosOp:
			// sin(x), cos(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_cos_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
			// ---------------------------------------------------

			case CoshOp:
			// sinh(x), cosh(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_cosh_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

			case CSkipOp:
			// CSkipOp has a variable number of arguments and
			// forward_next thinks it has no arguments.
			// we must inform forward_next of this special case.
			play->forward_cskip(op, arg, i_op, i_var);
			break;
			// -------------------------------------------------

			case CSumOp:
			// CSumOp has a variable number of arguments and
			// forward_next thinks it has no arguments.
			// we must inform forward_next of this special case.
			forward_csum_op_dir(
				q, r, i_var, arg, num_par, parameter, J, taylor
			);
			play->forward_csum(op, arg, i_op, i_var);
			break;
			// -------------------------------------------------

			case DisOp:
			forward_dis_op(p, q, r, i_var, arg, J, taylor);
			break;
			// -------------------------------------------------

			case DivvvOp:
			forward_divvv_op_dir(q, r, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case DivpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			forward_divpv_op_dir(q, r, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case DivvpOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < num_par );
			forward_divvp_op_dir(q, r, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case EndOp:
			// needed for sparse_jacobian test
			CPPAD_ASSERT_NARG_NRES(op, 0, 0);
			more_operators = false;
			break;
			// -------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case ErfOp:
			forward_erf_op_dir(q, r, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------
# endif

			case ExpOp:
			forward_exp_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case Expm1Op:
			forward_expm1_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
# endif
			// -------------------------------------------------

			case InvOp:
			CPPAD_ASSERT_NARG_NRES(op, 0, 1);
			break;
			// -------------------------------------------------

			case LdpOp:
			case LdvOp:
			forward_load_op(
				play,
				op,
				p,
				q,
				r,
				J,
				i_var,
				arg,
				var_by_load_op.data(),
				taylor
			);
			break;
			// ---------------------------------------------------

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
			CPPAD_ASSERT_UNKNOWN(q > 0 );
			break;
			// -------------------------------------------------

			case LogOp:
			forward_log_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
			// ---------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case Log1pOp:
			forward_log1p_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
# endif
			// ---------------------------------------------------

			case MulpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			forward_mulpv_op_dir(q, r, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case MulvvOp:
			forward_mulvv_op_dir(q, r, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case ParOp:
			k = i_var*(J-1)*r + i_var + (q-1)*r + 1;
			for(ell = 0; ell < r; ell++)
				taylor[k + ell] = Base(0.0);
			break;
			// -------------------------------------------------

			case PowpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			forward_powpv_op_dir(q, r, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case PowvpOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < num_par );
			forward_powvp_op_dir(q, r, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case PowvvOp:
			forward_powvv_op_dir(q, r, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case PriOp:
			CPPAD_ASSERT_UNKNOWN(q > 0);
			break;
			// -------------------------------------------------

			case SignOp:
			// sign(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_sign_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

			case SinOp:
			// cos(x), sin(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_sin_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

			case SinhOp:
			// cosh(x), sinh(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_sinh_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

			case SqrtOp:
			forward_sqrt_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

			case StppOp:
			case StpvOp:
			case StvpOp:
			case StvvOp:
			CPPAD_ASSERT_UNKNOWN(q > 0 );
			break;
			// -------------------------------------------------

			case SubvvOp:
			forward_subvv_op_dir(q, r, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case SubpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			forward_subpv_op_dir(q, r, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case SubvpOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < num_par );
			forward_subvp_op_dir(q, r, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case TanOp:
			// tan(x)^2, tan(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_tan_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

			case TanhOp:
			// tanh(x)^2, tanh(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar  );
			forward_tanh_op_dir(q, r, i_var, arg[0], J, taylor);
			break;
			// -------------------------------------------------

			case UserOp:
			// start or end an atomic operation sequence
			flag = user_state == start_user;
			user_atom = play->forward_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			if( flag )
			{	user_tx_one.resize(user_n * user_q1);
				user_tx_all.resize(user_n * (q * r + 1));
				//
				user_ty_one.resize(user_m * user_q1);
				user_ty_all.resize(user_m * (q * r + 1));
				//
				user_iy.resize(user_m);
			}
			else
			{	// call users function for this operation
				user_atom->set_old(user_old);
				for(ell = 0; ell < r; ell++)
				{	// set user_tx
					for(j = 0; j < user_n; j++)
					{	size_t j_all     = j * (q * r + 1);
						size_t j_one     = j * user_q1;
						user_tx_one[j_one+0] = user_tx_all[j_all+0];
						for(k = 1; k < user_q1; k++)
						{	size_t k_all       = j_all + (k-1)*r+1+ell;
							size_t k_one       = j_one + k;
							user_tx_one[k_one] = user_tx_all[k_all];
						}
					}
					// set user_ty
					for(i = 0; i < user_m; i++)
					{	size_t i_all     = i * (q * r + 1);
						size_t i_one     = i * user_q1;
						user_ty_one[i_one+0] = user_ty_all[i_all+0];
						for(k = 1; k < q; k++)
						{	size_t k_all       = i_all + (k-1)*r+1+ell;
							size_t k_one       = i_one + k;
							user_ty_one[k_one] = user_ty_all[k_all];
						}
					}
					CPPAD_ATOMIC_CALL(
					q, q, user_vx, user_vy, user_tx_one, user_ty_one
					);
# ifndef NDEBUG
					if( ! user_ok )
					{	std::string msg =
							user_atom->afun_name()
							+ ": atomic_base.forward: returned false";
						CPPAD_ASSERT_KNOWN(false, msg.c_str() );
					}
# endif
					for(i = 0; i < user_m; i++)
					{	if( user_iy[i] > 0 )
						{	size_t i_taylor = user_iy[i]*((J-1)*r+1);
							size_t q_taylor = i_taylor + (q-1)*r+1+ell;
							size_t q_one    = i * user_q1 + q;
							taylor[q_taylor] = user_ty_one[q_one];
						}
					}
				}
# if CPPAD_FORWARD2SWEEP_TRACE
				user_trace = true;
# endif
			}
			break;

			case UsrapOp:
			// parameter argument in an atomic operation sequence
			user_tx_all[user_j*(q*r+1) + 0] = parameter[ arg[0]];
			for(ell = 0; ell < r; ell++)
				for(k = 1; k < user_q1; k++)
					user_tx_all[user_j*(q*r+1) + (k-1)*r+1+ell] = Base(0.0);
			play->forward_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			break;

			case UsravOp:
			// variable argument in an atomic operation sequence
			user_tx_all[user_j*(q*r+1)+0] = taylor[arg[0]*((J-1)*r+1)+0];
			for(ell = 0; ell < r; ell++)
			{	for(k = 1; k < user_q1; k++)
				{	user_tx_all[user_j*(q*r+1) + (k-1)*r+1+ell] =
						taylor[arg[0]*((J-1)*r+1) + (k-1)*r+1+ell];
				}
			}
			play->forward_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			break;

			case UsrrpOp:
			// parameter result in an atomic operation sequence
			user_iy[user_i] = 0;
			user_ty_all[user_i*(q*r+1) + 0] = parameter[ arg[0]];
			for(ell = 0; ell < r; ell++)
				for(k = 1; k < user_q1; k++)
					user_ty_all[user_i*(q*r+1) + (k-1)*r+1+ell] = Base(0.0);
			play->forward_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			break;

			case UsrrvOp:
			// variable result in an atomic operation sequence
			user_iy[user_i] = i_var;
			user_ty_all[user_i*(q*r+1)+0] = taylor[i_var*((J-1)*r+1)+0];
			for(ell = 0; ell < r; ell++)
			{	for(k = 1; k < user_q1; k++)
				{	user_ty_all[user_i*(q*r+1) + (k-1)*r+1+ell] =
						taylor[i_var*((J-1)*r+1) + (k-1)*r+1+ell];
				}
			}
			play->forward_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			break;
			// -------------------------------------------------

			case ZmulpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			forward_zmulpv_op_dir(q, r, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case ZmulvpOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < num_par );
			forward_zmulvp_op_dir(q, r, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			case ZmulvvOp:
			forward_zmulvv_op_dir(q, r, i_var, arg, parameter, J, taylor);
			break;
			// -------------------------------------------------

			default:
			CPPAD_ASSERT_UNKNOWN(0);
		}
# if CPPAD_FORWARD2SWEEP_TRACE
		if( user_trace )
		{	user_trace = false;
			CPPAD_ASSERT_UNKNOWN( op == UserOp );
			CPPAD_ASSERT_UNKNOWN( NumArg(UsrrvOp) == 0 );
			for(i = 0; i < user_m; i++) if( user_iy[i] > 0 )
			{	size_t i_tmp   = (i_op + i) - user_m;
				printOp(
					std::cout,
					play,
					i_tmp,
					user_iy[i],
					UsrrvOp,
					CPPAD_NULL
				);
				Base* Z_tmp = taylor + user_iy[i]*((J-1) * r + 1);
				{	Z_vec[0]    = Z_tmp[0];
					for(ell = 0; ell < r; ell++)
					{	std::cout << std::endl << "     ";
						for(size_t p_tmp = 1; p_tmp <= q; p_tmp++)
							Z_vec[p_tmp] = Z_tmp[(p_tmp-1)*r+ell+1];
						printOpResult(
							std::cout,
							q + 1,
							Z_vec.data(),
							0,
							(Base *) CPPAD_NULL
						);
					}
				}
				std::cout << std::endl;
			}
		}
		const addr_t*   arg_tmp = arg;
		if( op == CSumOp )
			arg_tmp = arg - arg[-1] - 4;
		if( op == CSkipOp )
			arg_tmp = arg - arg[-1] - 7;
		if( op != UsrrvOp )
		{	printOp(
				std::cout,
				play,
				i_op,
				i_var,
				op,
				arg_tmp
			);
			Base* Z_tmp = CPPAD_NULL;
			if( op == UsravOp )
				Z_tmp = taylor + arg[0]*((J-1) * r + 1);
			else if( NumRes(op) > 0 )
				Z_tmp = taylor + i_var*((J-1)*r + 1);
			if( Z_tmp != CPPAD_NULL )
			{	Z_vec[0]    = Z_tmp[0];
				for(ell = 0; ell < r; ell++)
				{	std::cout << std::endl << "     ";
					for(size_t p_tmp = 1; p_tmp <= q; p_tmp++)
						Z_vec[p_tmp] = Z_tmp[ (p_tmp-1)*r + ell + 1];
					printOpResult(
						std::cout,
						q + 1,
						Z_vec.data(),
						0,
						(Base *) CPPAD_NULL
					);
				}
			}
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
# else
	}
# endif
	CPPAD_ASSERT_UNKNOWN( user_state == start_user );
	CPPAD_ASSERT_UNKNOWN( i_var + 1 == play->num_var_rec() );

	return;
}

// preprocessor symbols that are local to this file
# undef CPPAD_FORWARD2SWEEP_TRACE
# undef CPPAD_ATOMIC_CALL

} } // END_CPPAD_LOCAL_NAMESPACE
# endif
