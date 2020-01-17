# ifndef CPPAD_LOCAL_ABS_OP_HPP
# define CPPAD_LOCAL_ABS_OP_HPP

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
\file abs_op.hpp
Forward and reverse mode calculations for z = fabs(x).
*/

/*!
Compute forward mode Taylor coefficient for result of op = AbsOp.

The C++ source code corresponding to this operation is
\verbatim
	z = fabs(x)
\endverbatim

\copydetails CppAD::local::forward_unary1_op
*/
template <class Base>
inline void forward_abs_op(
	size_t p           ,
	size_t q           ,
	size_t i_z         ,
	size_t i_x         ,
	size_t cap_order   ,
	Base*  taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(AbsOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(AbsOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( p <= q );

	// Taylor coefficients corresponding to argument and result
	Base* x = taylor + i_x * cap_order;
	Base* z = taylor + i_z * cap_order;

	for(size_t j = p; j <= q; j++)
		z[j] = sign(x[0]) * x[j];
}

/*!
Multiple directions forward mode Taylor coefficient for op = AbsOp.

The C++ source code corresponding to this operation is
\verbatim
	z = fabs(x)
\endverbatim

\copydetails CppAD::local::forward_unary1_op_dir
*/
template <class Base>
inline void forward_abs_op_dir(
	size_t q           ,
	size_t r           ,
	size_t i_z         ,
	size_t i_x         ,
	size_t cap_order   ,
	Base*  taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(AbsOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(AbsOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( 0 < q );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );

	// Taylor coefficients corresponding to argument and result
	size_t num_taylor_per_var = (cap_order-1) * r + 1;
	Base* x = taylor + i_x * num_taylor_per_var;
	Base* z = taylor + i_z * num_taylor_per_var;

	size_t m = (q-1) * r + 1;
	for(size_t ell = 0; ell < r; ell++)
		z[m + ell] = sign(x[0]) * x[m + ell];
}

/*!
Compute zero order forward mode Taylor coefficient for result of op = AbsOp.

The C++ source code corresponding to this operation is
\verbatim
	z = fabs(x)
\endverbatim

\copydetails CppAD::local::forward_unary1_op_0
*/
template <class Base>
inline void forward_abs_op_0(
	size_t i_z         ,
	size_t i_x         ,
	size_t cap_order   ,
	Base*  taylor      )
{

	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(AbsOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(AbsOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( 0 < cap_order );

	// Taylor coefficients corresponding to argument and result
	Base x0 = *(taylor + i_x * cap_order);
	Base* z = taylor + i_z * cap_order;

	z[0] = fabs(x0);
}
/*!
Compute reverse mode partial derivatives for result of op = AbsOp.

The C++ source code corresponding to this operation is
\verbatim
	z = fabs(x)
\endverbatim

\copydetails CppAD::local::reverse_unary1_op
*/

template <class Base>
inline void reverse_abs_op(
	size_t      d            ,
	size_t      i_z          ,
	size_t      i_x          ,
	size_t      cap_order    ,
	const Base* taylor       ,
	size_t      nc_partial   ,
	Base*       partial      )
{	size_t j;

	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(AbsOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(AbsOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( d < cap_order );
	CPPAD_ASSERT_UNKNOWN( d < nc_partial );

	// Taylor coefficients and partials corresponding to argument
	const Base* x  = taylor  + i_x * cap_order;
	Base* px       = partial + i_x * nc_partial;

	// Taylor coefficients and partials corresponding to result
	Base* pz       = partial +    i_z * nc_partial;

	// do not need azmul because sign is either +1, -1, or zero
	for(j = 0; j <= d; j++)
		px[j] += sign(x[0]) * pz[j];
}

} } // END_CPPAD_LOCAL_NAMESPACE
# endif
