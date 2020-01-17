// $Id: sign_op.hpp 3865 2017-01-19 01:57:55Z bradbell $
# ifndef CPPAD_LOCAL_SIGN_OP_HPP
# define CPPAD_LOCAL_SIGN_OP_HPP

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
\file sign_op.hpp
Forward and reverse mode calculations for z = sign(x).
*/

/*!
Compute forward mode Taylor coefficient for result of op = SignOp.

The C++ source code corresponding to this operation is
\verbatim
	z = sign(x)
\endverbatim

\copydetails CppAD::local::forward_unary1_op
*/
template <class Base>
inline void forward_sign_op(
	size_t p           ,
	size_t q           ,
	size_t i_z         ,
	size_t i_x         ,
	size_t cap_order   ,
	Base*  taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SignOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SignOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( p <= q );

	// Taylor coefficients corresponding to argument and result
	Base* x = taylor + i_x * cap_order;
	Base* z = taylor + i_z * cap_order;

	if( p == 0 )
	{	z[0] = sign(x[0]);
		p++;
	}
	for(size_t j = p; j <= q; j++)
		z[j] = Base(0.);
}
/*!
Multiple direction forward mode Taylor coefficient for op = SignOp.

The C++ source code corresponding to this operation is
\verbatim
	z = sign(x)
\endverbatim

\copydetails CppAD::local::forward_unary1_op_dir
*/
template <class Base>
inline void forward_sign_op_dir(
	size_t q           ,
	size_t r           ,
	size_t i_z         ,
	size_t i_x         ,
	size_t cap_order   ,
	Base*  taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SignOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SignOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( 0 < q );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );

	// Taylor coefficients corresponding to argument and result
	size_t num_taylor_per_var = (cap_order-1) * r + 1;
	size_t m = (q - 1) * r + 1;
	Base* z = taylor + i_z * num_taylor_per_var;

	for(size_t ell = 0; ell < r; ell++)
		z[m+ell] = Base(0.);
}

/*!
Compute zero order forward mode Taylor coefficient for result of op = SignOp.

The C++ source code corresponding to this operation is
\verbatim
	z = sign(x)
\endverbatim

\copydetails CppAD::local::forward_unary1_op_0
*/
template <class Base>
inline void forward_sign_op_0(
	size_t i_z         ,
	size_t i_x         ,
	size_t cap_order   ,
	Base*  taylor      )
{

	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SignOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SignOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( 0 < cap_order );

	// Taylor coefficients corresponding to argument and result
	Base x0 = *(taylor + i_x * cap_order);
	Base* z = taylor + i_z * cap_order;

	z[0] = sign(x0);
}
/*!
Compute reverse mode partial derivatives for result of op = SignOp.

The C++ source code corresponding to this operation is
\verbatim
	z = sign(x)
\endverbatim

\copydetails CppAD::local::reverse_unary1_op
*/

template <class Base>
inline void reverse_sign_op(
	size_t      d            ,
	size_t      i_z          ,
	size_t      i_x          ,
	size_t      cap_order    ,
	const Base* taylor       ,
	size_t      nc_partial   ,
	Base*       partial      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SignOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SignOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( d < cap_order );
	CPPAD_ASSERT_UNKNOWN( d < nc_partial );

	// nothing to do because partials of sign are zero
	return;
}

} } // END_CPPAD_LOCAL_NAMESPACE
# endif
