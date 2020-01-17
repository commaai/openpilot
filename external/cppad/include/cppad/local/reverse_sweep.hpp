// $Id: reverse_sweep.hpp 3853 2016-12-14 14:40:11Z bradbell $
# ifndef CPPAD_LOCAL_REVERSE_SWEEP_HPP
# define CPPAD_LOCAL_REVERSE_SWEEP_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */


namespace CppAD { namespace local { // BEGIN_CPPAD_LOCAL_NAMESPACE
/*!
\file reverse_sweep.hpp
Compute derivatives of arbitrary order Taylor coefficients.
*/

/*
\def CPPAD_ATOMIC_CALL
This avoids warnings when NDEBUG is defined and user_ok is not used.
If \c NDEBUG is defined, this resolves to
\code
	user_atom->reverse
\endcode
otherwise, it respolves to
\code
	user_ok = user_atom->reverse
\endcode
This maco is undefined at the end of this file to facillitate is
use with a different definition in other files.
*/
# ifdef NDEBUG
# define CPPAD_ATOMIC_CALL user_atom->reverse
# else
# define CPPAD_ATOMIC_CALL user_ok = user_atom->reverse
# endif

/*!
\def CPPAD_REVERSE_SWEEP_TRACE
This value is either zero or one.
Zero is the normal operational value.
If it is one, a trace of every reverse_sweep computation is printed.
*/
# define CPPAD_REVERSE_SWEEP_TRACE 0

/*!
Compute derivative of arbitrary order forward mode Taylor coefficients.

\tparam Base
base type for the operator; i.e., this operation sequence was recorded
using AD< \a Base > and computations by this routine are done using type
\a Base.

\param d
is the highest order Taylor coefficients that
we are computing the derivative of.

\param n
is the number of independent variables on the tape.

\param numvar
is the total number of variables on the tape.
This is also equal to the number of rows in the matrix \a Taylor; i.e.,
play->num_var_rec().

\param play
The information stored in \a play
is a recording of the operations corresponding to the function
\f[
	F : {\bf R}^n \rightarrow {\bf R}^m
\f]
where \f$ n \f$ is the number of independent variables and
\f$ m \f$ is the number of dependent variables.
We define \f$ u^{(k)} \f$ as the value of <code>x_k</code> in the previous call
of the form
<code>
	f.Forward(k, x_k)
</code>
We define
\f$ X : {\bf R}^{n \times d} \rightarrow {\bf R}^n \f$ by
\f[
	X(t, u) =  u^{(0)} + u^{(1)} t + \cdots + u^{(d)} t^d
\f]
We define
\f$ Y : {\bf R}^{n \times d} \rightarrow {\bf R}^m \f$ by
\f[
	Y(t, u) =  F[ X(t, u) ]
\f]
We define the function
\f$ W : {\bf R}^{n \times d} \rightarrow {\bf R} \f$ by
\f[
W(u)
=
\sum_{k=0}^{d} ( w^{(k)} )^{\rm T}
	\frac{1}{k !} \frac{\partial^k}{\partial t^k} Y(0, u)
\f]
(The matrix \f$ w \in {\bf R}^m \f$,
is defined below under the heading Partial.)
Note that the scale factor  1 / k  converts
the k-th partial derivative to the k-th order Taylor coefficient.
This routine computes the derivative of \f$ W(u) \f$
with respect to all the Taylor coefficients
\f$ u^{(k)} \f$ for \f$ k = 0 , ... , d \f$.
\n
\n
The object \a play is effectly constant.
There is an exception to this,
while palying back the tape
the object \a play holds information about the current location
with in the tape and this changes during palyback.

\param J
Is the number of columns in the coefficient matrix \a Taylor.
This must be greater than or equal \a d + 1.

\param Taylor
For i = 1 , ... , \a numvar, and for k = 0 , ... , \a d,
\a Taylor [ i * J + k ]
is the k-th order Taylor coefficient corresponding to
variable with index i on the tape.
The value \f$ u \in {\bf R}^{n \times d} \f$,
at which the derivative is computed,
is defined by
\f$ u_j^{(k)} \f$ = \a Taylor [ j * J + k ]
for j = 1 , ... , \a n, and for k = 0 , ... , \a d.

\param K
Is the number of columns in the partial derivative matrix \a Partial.
It must be greater than or equal \a d + 1.

\param Partial
\b Input:
The last \f$ m \f$ rows of \a Partial are inputs.
The matrix \f$ w \f$, used to define \f$ W(u) \f$,
is specified by these rows.
For i = 0 , ... , m - 1,
for k = 0 , ... , d,
<code>Partial [ (numvar - m + i ) * K + k ] = w[i,k]</code>.
\n
\n
\b Temporary:
For i = n+1 , ... , \a numvar - 1 and for k = 0 , ... , d,
the value of \a Partial [ i * K + k ] is used for temporary work space
and its output value is not defined.
\n
\n
\b Output:
For j = 1 , ... , n and for k = 0 , ... , d,
\a Partial [ j * K + k ]
is the partial derivative of \f$ W( u ) \f$ with
respect to \f$ u_j^{(k)} \f$.

\param cskip_op
Is a vector with size play->num_op_rec().
If cskip_op[i] is true, the operator index i in the recording
does not affect any of the dependent variable (given the value
of the independent variables).
Note that all the operators in an atomic function call are skipped as a block,
so only the last UserOp fore each call needs to have cskip_op[i] true.

\param var_by_load_op
is a vector with size play->num_load_op_rec().
Is the variable index corresponding to each load instruction.
In the case where the index is zero,
the instruction corresponds to a parameter (not variable).

\par Assumptions
The first operator on the tape is a BeginOp,
and the next \a n operators are InvOp operations for the
corresponding independent variables.
*/
template <class Base>
void ReverseSweep(
	size_t                      d,
	size_t                      n,
	size_t                      numvar,
	local::player<Base>*               play,
	size_t                      J,
	const Base*                 Taylor,
	size_t                      K,
	Base*                       Partial,
	bool*                       cskip_op,
	const pod_vector<addr_t>&   var_by_load_op
)
{
	OpCode           op;
	size_t         i_op;
	size_t        i_var;

	const addr_t*   arg = CPPAD_NULL;

	// check numvar argument
	CPPAD_ASSERT_UNKNOWN( play->num_var_rec() == numvar );
	CPPAD_ASSERT_UNKNOWN( numvar > 0 );

	// length of the parameter vector (used by CppAD assert macros)
	const size_t num_par = play->num_par_rec();

	// pointer to the beginning of the parameter vector
	const Base* parameter = CPPAD_NULL;
	if( num_par > 0 )
		parameter = play->GetPar();

	// work space used by UserOp.
	const size_t user_k  = d;    // highest order we are differentiating
	const size_t user_k1 = d+1;  // number of orders for this calculation
	vector<size_t> user_ix;      // variable indices for argument vector
	vector<Base> user_tx;        // argument vector Taylor coefficients
	vector<Base> user_ty;        // result vector Taylor coefficients
	vector<Base> user_px;        // partials w.r.t argument vector
	vector<Base> user_py;        // partials w.r.t. result vector
	//
	atomic_base<Base>* user_atom = CPPAD_NULL; // user's atomic op calculator
# ifndef NDEBUG
	bool               user_ok   = false;      // atomic op return value
# endif
	//
	// information defined by forward_user
	size_t user_old=0, user_m=0, user_n=0, user_i=0, user_j=0;
	enum_user_state user_state = end_user; // proper initialization

	// temporary indices
	size_t j, ell;

	// Initialize
	play->reverse_start(op, arg, i_op, i_var);
	CPPAD_ASSERT_UNKNOWN( op == EndOp );
# if CPPAD_REVERSE_SWEEP_TRACE
	std::cout << std::endl;
# endif
	bool more_operators = true;
	while(more_operators)
	{	bool flag; // temporary for use in switch cases
		//
		// next op
		play->reverse_next(op, arg, i_op, i_var);
		CPPAD_ASSERT_UNKNOWN((i_op >  n) | (op == InvOp) | (op == BeginOp));
		CPPAD_ASSERT_UNKNOWN((i_op <= n) | (op != InvOp) | (op != BeginOp));
		CPPAD_ASSERT_UNKNOWN( i_op < play->num_op_rec() );

		// check if we are skipping this operation
		while( cskip_op[i_op] )
		{	switch(op)
			{	case CSumOp:
				// CSumOp has a variable number of arguments
				play->reverse_csum(op, arg, i_op, i_var);
				break;

				case CSkipOp:
				// CSkip has a variable number of arguments
				play->reverse_cskip(op, arg, i_op, i_var);
				break;

				case UserOp:
				{	// skip all operations in this user atomic call
					CPPAD_ASSERT_UNKNOWN( user_state == end_user );
					play->reverse_user(op, user_state,
						user_old, user_m, user_n, user_i, user_j
					);
					size_t n_skip = user_m + user_n + 1;
					for(size_t i = 0; i < n_skip; i++)
					{	play->reverse_next(op, arg, i_op, i_var);
						play->reverse_user(op, user_state,
							user_old, user_m, user_n, user_i, user_j
						);
					}
					CPPAD_ASSERT_UNKNOWN( user_state == end_user );
				}
				break;

				default:
				break;
			}
			play->reverse_next(op, arg, i_op, i_var);
			CPPAD_ASSERT_UNKNOWN( i_op < play->num_op_rec() );
		}

		// rest of informaiton depends on the case
# if CPPAD_REVERSE_SWEEP_TRACE
		if( op == CSumOp )
		{	// CSumOp has a variable number of arguments
			play->reverse_csum(op, arg, i_op, i_var);
		}
		if( op == CSkipOp )
		{	// CSkip has a variable number of arguments
			play->reverse_cskip(op, arg, i_op, i_var);
		}
		size_t       i_tmp  = i_var;
		const Base*  Z_tmp  = Taylor + i_var * J;
		const Base*  pZ_tmp = Partial + i_var * K;
		printOp(
			std::cout,
			play,
			i_op,
			i_tmp,
			op,
			arg
		);
		if( NumRes(op) > 0 && op != BeginOp ) printOpResult(
			std::cout,
			d + 1,
			Z_tmp,
			d + 1,
			pZ_tmp
		);
		std::cout << std::endl;
# endif
		switch( op )
		{

			case AbsOp:
			reverse_abs_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

			case AcosOp:
			// sqrt(1 - x * x), acos(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar );
			reverse_acos_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case AcoshOp:
			// sqrt(x * x - 1), acosh(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar );
			reverse_acosh_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
# endif
			// --------------------------------------------------

			case AddvvOp:
			reverse_addvv_op(
				d, i_var, arg, parameter, J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

			case AddpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			reverse_addpv_op(
				d, i_var, arg, parameter, J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

			case AsinOp:
			// sqrt(1 - x * x), asin(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar );
			reverse_asin_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case AsinhOp:
			// sqrt(1 + x * x), asinh(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar );
			reverse_asinh_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
# endif
			// --------------------------------------------------

			case AtanOp:
			// 1 + x * x, atan(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar );
			reverse_atan_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
			// -------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case AtanhOp:
			// 1 - x * x, atanh(x)
			CPPAD_ASSERT_UNKNOWN( i_var < numvar );
			reverse_atanh_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
# endif
			// -------------------------------------------------

			case BeginOp:
			CPPAD_ASSERT_NARG_NRES(op, 1, 1);
			more_operators = false;
			break;
			// --------------------------------------------------

			case CSkipOp:
			// CSkipOp has a variable number of arguments and
			// forward_next thinks it one has one argument.
			// we must inform reverse_next of this special case.
# if ! CPPAD_REVERSE_SWEEP_TRACE
			play->reverse_cskip(op, arg, i_op, i_var);
# endif
			break;
			// -------------------------------------------------

			case CSumOp:
			// CSumOp has a variable number of arguments and
			// reverse_next thinks it one has one argument.
			// We must inform reverse_next of this special case.
# if ! CPPAD_REVERSE_SWEEP_TRACE
			play->reverse_csum(op, arg, i_op, i_var);
# endif
			reverse_csum_op(
				d, i_var, arg, K, Partial
			);
			// end of a cummulative summation
			break;
			// -------------------------------------------------

			case CExpOp:
			reverse_cond_op(
				d,
				i_var,
				arg,
				num_par,
				parameter,
				J,
				Taylor,
				K,
				Partial
			);
			break;
			// --------------------------------------------------

			case CosOp:
			CPPAD_ASSERT_UNKNOWN( i_var < numvar );
			reverse_cos_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

			case CoshOp:
			CPPAD_ASSERT_UNKNOWN( i_var < numvar );
			reverse_cosh_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

			case DisOp:
			// Derivative of discrete operation is zero so no
			// contribution passes through this operation.
			break;
			// --------------------------------------------------

			case DivvvOp:
			reverse_divvv_op(
				d, i_var, arg, parameter, J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

			case DivpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			reverse_divpv_op(
				d, i_var, arg, parameter, J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

			case DivvpOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < num_par );
			reverse_divvp_op(
				d, i_var, arg, parameter, J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case ErfOp:
			reverse_erf_op(
				d, i_var, arg, parameter, J, Taylor, K, Partial
			);
			break;
# endif
			// --------------------------------------------------

			case ExpOp:
			reverse_exp_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case Expm1Op:
			reverse_expm1_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
# endif
			// --------------------------------------------------

			case InvOp:
			break;
			// --------------------------------------------------

			case LdpOp:
			reverse_load_op(
			op, d, i_var, arg, J, Taylor, K, Partial, var_by_load_op.data()
			);
			break;
			// -------------------------------------------------

			case LdvOp:
			reverse_load_op(
			op, d, i_var, arg, J, Taylor, K, Partial, var_by_load_op.data()
			);
			break;
			// --------------------------------------------------

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
			break;
			// -------------------------------------------------

			case LogOp:
			reverse_log_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

# if CPPAD_USE_CPLUSPLUS_2011
			case Log1pOp:
			reverse_log1p_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
# endif
			// --------------------------------------------------

			case MulpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			reverse_mulpv_op(
				d, i_var, arg, parameter, J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

			case MulvvOp:
			reverse_mulvv_op(
				d, i_var, arg, parameter, J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

			case ParOp:
			break;
			// --------------------------------------------------

			case PowvpOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < num_par );
			reverse_powvp_op(
				d, i_var, arg, parameter, J, Taylor, K, Partial
			);
			break;
			// -------------------------------------------------

			case PowpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			reverse_powpv_op(
				d, i_var, arg, parameter, J, Taylor, K, Partial
			);
			break;
			// -------------------------------------------------

			case PowvvOp:
			reverse_powvv_op(
				d, i_var, arg, parameter, J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

			case PriOp:
			// no result so nothing to do
			break;
			// --------------------------------------------------

			case SignOp:
			CPPAD_ASSERT_UNKNOWN( i_var < numvar );
			reverse_sign_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
			// -------------------------------------------------

			case SinOp:
			CPPAD_ASSERT_UNKNOWN( i_var < numvar );
			reverse_sin_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
			// -------------------------------------------------

			case SinhOp:
			CPPAD_ASSERT_UNKNOWN( i_var < numvar );
			reverse_sinh_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

			case SqrtOp:
			reverse_sqrt_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

			case StppOp:
			break;
			// --------------------------------------------------

			case StpvOp:
			break;
			// -------------------------------------------------

			case StvpOp:
			break;
			// -------------------------------------------------

			case StvvOp:
			break;
			// --------------------------------------------------

			case SubvvOp:
			reverse_subvv_op(
				d, i_var, arg, parameter, J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

			case SubpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			reverse_subpv_op(
				d, i_var, arg, parameter, J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

			case SubvpOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < num_par );
			reverse_subvp_op(
				d, i_var, arg, parameter, J, Taylor, K, Partial
			);
			break;
			// -------------------------------------------------

			case TanOp:
			CPPAD_ASSERT_UNKNOWN( i_var < numvar );
			reverse_tan_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
			// -------------------------------------------------

			case TanhOp:
			CPPAD_ASSERT_UNKNOWN( i_var < numvar );
			reverse_tanh_op(
				d, i_var, arg[0], J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

			case UserOp:
			// start or end an atomic operation sequence
			flag = user_state == end_user;
			user_atom = play->reverse_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			if( flag )
			{	user_ix.resize(user_n);
				if(user_tx.size() != user_n * user_k1)
				{	user_tx.resize(user_n * user_k1);
					user_px.resize(user_n * user_k1);
				}
				if(user_ty.size() != user_m * user_k1)
				{	user_ty.resize(user_m * user_k1);
					user_py.resize(user_m * user_k1);
				}
			}
			else
			{	// call users function for this operation
				user_atom->set_old(user_old);
				CPPAD_ATOMIC_CALL(
					user_k, user_tx, user_ty, user_px, user_py
				);
# ifndef NDEBUG
				if( ! user_ok )
				{	std::string msg =
						user_atom->afun_name()
						+ ": atomic_base.reverse: returned false";
					CPPAD_ASSERT_KNOWN(false, msg.c_str() );
				}
# endif
				for(j = 0; j < user_n; j++) if( user_ix[j] > 0 )
				{	for(ell = 0; ell < user_k1; ell++)
						Partial[user_ix[j] * K + ell] +=
							user_px[j * user_k1 + ell];
				}
			}
			break;

			case UsrapOp:
			// parameter argument in an atomic operation sequence
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			play->reverse_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			user_ix[user_j] = 0;
			user_tx[user_j * user_k1 + 0] = parameter[ arg[0]];
			for(ell = 1; ell < user_k1; ell++)
				user_tx[user_j * user_k1 + ell] = Base(0.);
			break;

			case UsravOp:
			// variable argument in an atomic operation sequence
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) <= i_var );
			CPPAD_ASSERT_UNKNOWN( 0 < arg[0] );
			play->reverse_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			user_ix[user_j] = arg[0];
			for(ell = 0; ell < user_k1; ell++)
				user_tx[user_j*user_k1 + ell] = Taylor[ arg[0] * J + ell];
			break;

			case UsrrpOp:
			// parameter result in an atomic operation sequence
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			play->reverse_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			for(ell = 0; ell < user_k1; ell++)
			{	user_py[user_i * user_k1 + ell] = Base(0.);
				user_ty[user_i * user_k1 + ell] = Base(0.);
			}
			user_ty[user_i * user_k1 + 0] = parameter[ arg[0] ];
			break;

			case UsrrvOp:
			// variable result in an atomic operation sequence
			play->reverse_user(op, user_state,
				user_old, user_m, user_n, user_i, user_j
			);
			for(ell = 0; ell < user_k1; ell++)
			{	user_py[user_i * user_k1 + ell] =
						Partial[i_var * K + ell];
				user_ty[user_i * user_k1 + ell] =
						Taylor[i_var * J + ell];
			}
			break;
			// ------------------------------------------------------------

			case ZmulpvOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < num_par );
			reverse_zmulpv_op(
				d, i_var, arg, parameter, J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

			case ZmulvpOp:
			CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < num_par );
			reverse_zmulvp_op(
				d, i_var, arg, parameter, J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

			case ZmulvvOp:
			reverse_zmulvv_op(
				d, i_var, arg, parameter, J, Taylor, K, Partial
			);
			break;
			// --------------------------------------------------

			default:
			CPPAD_ASSERT_UNKNOWN(false);
		}
	}
# if CPPAD_REVERSE_SWEEP_TRACE
	std::cout << std::endl;
# endif
	// values corresponding to BeginOp
	CPPAD_ASSERT_UNKNOWN( i_op == 0 );
	CPPAD_ASSERT_UNKNOWN( i_var == 0 );
}

} } // END_CPPAD_LOCAL_NAMESPACE

// preprocessor symbols that are local to this file
# undef CPPAD_REVERSE_SWEEP_TRACE
# undef CPPAD_ATOMIC_CALL

# endif
