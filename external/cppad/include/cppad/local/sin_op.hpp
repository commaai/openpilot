# ifndef CPPAD_LOCAL_SIN_OP_HPP
# define CPPAD_LOCAL_SIN_OP_HPP

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
\file sin_op.hpp
Forward and reverse mode calculations for z = sin(x).
*/


/*!
Compute forward mode Taylor coefficient for result of op = SinOp.

The C++ source code corresponding to this operation is
\verbatim
	z = sin(x)
\endverbatim
The auxillary result is
\verbatim
	y = cos(x)
\endverbatim
The value of y, and its derivatives, are computed along with the value
and derivatives of z.

\copydetails CppAD::local::forward_unary2_op
*/
template <class Base>
inline void forward_sin_op(
	size_t p           ,
	size_t q           ,
	size_t i_z         ,
	size_t i_x         ,
	size_t cap_order   ,
	Base*  taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SinOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SinOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( p <= q );

	// Taylor coefficients corresponding to argument and result
	Base* x = taylor + i_x * cap_order;
	Base* s = taylor + i_z * cap_order;
	Base* c = s      -       cap_order;

	// rest of this routine is identical for the following cases:
	// forward_sin_op, forward_cos_op, forward_sinh_op, forward_cosh_op.
	// (except that there is a sign difference for the hyperbolic case).
	size_t k;
	if( p == 0 )
	{	s[0] = sin( x[0] );
		c[0] = cos( x[0] );
		p++;
	}
	for(size_t j = p; j <= q; j++)
	{
		s[j] = Base(0.0);
		c[j] = Base(0.0);
		for(k = 1; k <= j; k++)
		{	s[j] += Base(double(k)) * x[k] * c[j-k];
			c[j] -= Base(double(k)) * x[k] * s[j-k];
		}
		s[j] /= Base(double(j));
		c[j] /= Base(double(j));
	}
}
/*!
Compute forward mode Taylor coefficient for result of op = SinOp.

The C++ source code corresponding to this operation is
\verbatim
	z = sin(x)
\endverbatim
The auxillary result is
\verbatim
	y = cos(x)
\endverbatim
The value of y, and its derivatives, are computed along with the value
and derivatives of z.

\copydetails CppAD::local::forward_unary2_op_dir
*/
template <class Base>
inline void forward_sin_op_dir(
	size_t q           ,
	size_t r           ,
	size_t i_z         ,
	size_t i_x         ,
	size_t cap_order   ,
	Base*  taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SinOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SinOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( 0 < q );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );

	// Taylor coefficients corresponding to argument and result
	size_t num_taylor_per_var = (cap_order-1) * r + 1;
	Base* x = taylor + i_x * num_taylor_per_var;
	Base* s = taylor + i_z * num_taylor_per_var;
	Base* c = s      -       num_taylor_per_var;


	// rest of this routine is identical for the following cases:
	// forward_sin_op, forward_cos_op, forward_sinh_op, forward_cosh_op
	// (except that there is a sign difference for the hyperbolic case).
	size_t m = (q-1) * r + 1;
	for(size_t ell = 0; ell < r; ell++)
	{	s[m+ell] =   Base(double(q)) * x[m + ell] * c[0];
		c[m+ell] = - Base(double(q)) * x[m + ell] * s[0];
		for(size_t k = 1; k < q; k++)
		{	s[m+ell] += Base(double(k)) * x[(k-1)*r+1+ell] * c[(q-k-1)*r+1+ell];
			c[m+ell] -= Base(double(k)) * x[(k-1)*r+1+ell] * s[(q-k-1)*r+1+ell];
		}
		s[m+ell] /= Base(double(q));
		c[m+ell] /= Base(double(q));
	}
}


/*!
Compute zero order forward mode Taylor coefficient for result of op = SinOp.

The C++ source code corresponding to this operation is
\verbatim
	z = sin(x)
\endverbatim
The auxillary result is
\verbatim
	y = cos(x)
\endverbatim
The value of y is computed along with the value of z.

\copydetails CppAD::local::forward_unary2_op_0
*/
template <class Base>
inline void forward_sin_op_0(
	size_t i_z         ,
	size_t i_x         ,
	size_t cap_order   ,
	Base*  taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SinOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SinOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( 0 < cap_order );

	// Taylor coefficients corresponding to argument and result
	Base* x = taylor + i_x * cap_order;
	Base* s = taylor + i_z * cap_order;  // called z in documentation
	Base* c = s      -       cap_order;  // called y in documentation

	s[0] = sin( x[0] );
	c[0] = cos( x[0] );
}

/*!
Compute reverse mode partial derivatives for result of op = SinOp.

The C++ source code corresponding to this operation is
\verbatim
	z = sin(x)
\endverbatim
The auxillary result is
\verbatim
	y = cos(x)
\endverbatim
The value of y is computed along with the value of z.

\copydetails CppAD::local::reverse_unary2_op
*/

template <class Base>
inline void reverse_sin_op(
	size_t      d            ,
	size_t      i_z          ,
	size_t      i_x          ,
	size_t      cap_order    ,
	const Base* taylor       ,
	size_t      nc_partial   ,
	Base*       partial      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SinOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SinOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( d < cap_order );
	CPPAD_ASSERT_UNKNOWN( d < nc_partial );

	// Taylor coefficients and partials corresponding to argument
	const Base* x  = taylor  + i_x * cap_order;
	Base* px       = partial + i_x * nc_partial;

	// Taylor coefficients and partials corresponding to first result
	const Base* s  = taylor  + i_z * cap_order; // called z in doc
	Base* ps       = partial + i_z * nc_partial;

	// Taylor coefficients and partials corresponding to auxillary result
	const Base* c  = s  - cap_order; // called y in documentation
	Base* pc       = ps - nc_partial;


	// rest of this routine is identical for the following cases:
	// reverse_sin_op, reverse_cos_op, reverse_sinh_op, reverse_cosh_op.
	size_t j = d;
	size_t k;
	while(j)
	{
		ps[j]   /= Base(double(j));
		pc[j]   /= Base(double(j));
		for(k = 1; k <= j; k++)
		{
			px[k]   += Base(double(k)) * azmul(ps[j], c[j-k]);
			px[k]   -= Base(double(k)) * azmul(pc[j], s[j-k]);

			ps[j-k] -= Base(double(k)) * azmul(pc[j], x[k]);
			pc[j-k] += Base(double(k)) * azmul(ps[j], x[k]);

		}
		--j;
	}
	px[0] += azmul(ps[0], c[0]);
	px[0] -= azmul(pc[0], s[0]);
}

} } // END_CPPAD_LOCAL_NAMESPACE
# endif
