# ifndef CPPAD_LOCAL_SQRT_OP_HPP
# define CPPAD_LOCAL_SQRT_OP_HPP

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
\file sqrt_op.hpp
Forward and reverse mode calculations for z = sqrt(x).
*/


/*!
Compute forward mode Taylor coefficient for result of op = SqrtOp.

The C++ source code corresponding to this operation is
\verbatim
	z = sqrt(x)
\endverbatim

\copydetails CppAD::local::forward_unary1_op
*/
template <class Base>
inline void forward_sqrt_op(
	size_t p           ,
	size_t q           ,
	size_t i_z         ,
	size_t i_x         ,
	size_t cap_order   ,
	Base*  taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SqrtOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SqrtOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( p <= q );

	// Taylor coefficients corresponding to argument and result
	Base* x = taylor + i_x * cap_order;
	Base* z = taylor + i_z * cap_order;

	size_t k;
	if( p == 0 )
	{	z[0] = sqrt( x[0] );
		p++;
	}
	for(size_t j = p; j <= q; j++)
	{
		z[j] = Base(0.0);
		for(k = 1; k < j; k++)
			z[j] -= Base(double(k)) * z[k] * z[j-k];
		z[j] /= Base(double(j));
		z[j] += x[j] / Base(2.0);
		z[j] /= z[0];
	}
}

/*!
Multiple direction forward mode Taylor coefficient for op = SqrtOp.

The C++ source code corresponding to this operation is
\verbatim
	z = sqrt(x)
\endverbatim

\copydetails CppAD::local::forward_unary1_op_dir
*/
template <class Base>
inline void forward_sqrt_op_dir(
	size_t q           ,
	size_t r           ,
	size_t i_z         ,
	size_t i_x         ,
	size_t cap_order   ,
	Base*  taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SqrtOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SqrtOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( 0 < q );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );

	// Taylor coefficients corresponding to argument and result
	size_t num_taylor_per_var = (cap_order-1) * r + 1;
	Base* z = taylor + i_z * num_taylor_per_var;
	Base* x = taylor + i_x * num_taylor_per_var;

	size_t m = (q-1) * r + 1;
	for(size_t ell = 0; ell < r; ell++)
	{	z[m+ell] = Base(0.0);
		for(size_t k = 1; k < q; k++)
			z[m+ell] -= Base(double(k)) * z[(k-1)*r+1+ell] * z[(q-k-1)*r+1+ell];
		z[m+ell] /= Base(double(q));
		z[m+ell] += x[m+ell] / Base(2.0);
		z[m+ell] /= z[0];
	}
}

/*!
Compute zero order forward mode Taylor coefficient for result of op = SqrtOp.

The C++ source code corresponding to this operation is
\verbatim
	z = sqrt(x)
\endverbatim

\copydetails CppAD::local::forward_unary1_op_0
*/
template <class Base>
inline void forward_sqrt_op_0(
	size_t i_z         ,
	size_t i_x         ,
	size_t cap_order   ,
	Base*  taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SqrtOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SqrtOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( 0 < cap_order );

	// Taylor coefficients corresponding to argument and result
	Base* x = taylor + i_x * cap_order;
	Base* z = taylor + i_z * cap_order;

	z[0] = sqrt( x[0] );
}
/*!
Compute reverse mode partial derivatives for result of op = SqrtOp.

The C++ source code corresponding to this operation is
\verbatim
	z = sqrt(x)
\endverbatim

\copydetails CppAD::local::reverse_unary1_op
*/

template <class Base>
inline void reverse_sqrt_op(
	size_t      d            ,
	size_t      i_z          ,
	size_t      i_x          ,
	size_t      cap_order    ,
	const Base* taylor       ,
	size_t      nc_partial   ,
	Base*       partial      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SqrtOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SqrtOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( d < cap_order );
	CPPAD_ASSERT_UNKNOWN( d < nc_partial );

	// Taylor coefficients and partials corresponding to argument
	Base* px       = partial + i_x * nc_partial;

	// Taylor coefficients and partials corresponding to result
	const Base* z  = taylor  + i_z * cap_order;
	Base* pz       = partial + i_z * nc_partial;


	Base inv_z0 = Base(1.0) / z[0];

	// number of indices to access
	size_t j = d;
	size_t k;
	while(j)
	{

		// scale partial w.r.t. z[j]
		pz[j]    = azmul(pz[j], inv_z0);

		pz[0]   -= azmul(pz[j], z[j]);
		px[j]   += pz[j] / Base(2.0);
		for(k = 1; k < j; k++)
			pz[k]   -= azmul(pz[j], z[j-k]);
		--j;
	}
	px[0] += azmul(pz[0], inv_z0) / Base(2.0);
}

} } // END_CPPAD_LOCAL_NAMESPACE
# endif
