# ifndef CPPAD_LOCAL_ACOS_OP_HPP
# define CPPAD_LOCAL_ACOS_OP_HPP

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
\file acos_op.hpp
Forward and reverse mode calculations for z = acos(x).
*/


/*!
Compute forward mode Taylor coefficient for result of op = AcosOp.

The C++ source code corresponding to this operation is
\verbatim
	z = acos(x)
\endverbatim
The auxillary result is
\verbatim
	y = sqrt(1 - x * x)
\endverbatim
The value of y, and its derivatives, are computed along with the value
and derivatives of z.

\copydetails CppAD::local::forward_unary2_op
*/
template <class Base>
inline void forward_acos_op(
	size_t p           ,
	size_t q           ,
	size_t i_z         ,
	size_t i_x         ,
	size_t cap_order   ,
	Base*  taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(AcosOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(AcosOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( p <= q );

	// Taylor coefficients corresponding to argument and result
	Base* x = taylor + i_x * cap_order;
	Base* z = taylor + i_z * cap_order;
	Base* b = z      -       cap_order;  // called y in documentation

	size_t k;
	Base uj;
	if( p == 0 )
	{	z[0] = acos( x[0] );
		uj   = Base(1.0) - x[0] * x[0];
		b[0] = sqrt( uj );
		p++;
	}
	for(size_t j = p; j <= q; j++)
	{	uj = Base(0.0);
		for(k = 0; k <= j; k++)
			uj -= x[k] * x[j-k];
		b[j] = Base(0.0);
		z[j] = Base(0.0);
		for(k = 1; k < j; k++)
		{	b[j] -= Base(double(k)) * b[k] * b[j-k];
			z[j] -= Base(double(k)) * z[k] * b[j-k];
		}
		b[j] /= Base(double(j));
		z[j] /= Base(double(j));
		//
		b[j] += uj / Base(2.0);
		z[j] -= x[j];
		//
		b[j] /= b[0];
		z[j] /= b[0];
	}
}
/*!
Multiple directions forward mode Taylor coefficient for op = AcosOp.

The C++ source code corresponding to this operation is
\verbatim
	z = acos(x)
\endverbatim
The auxillary result is
\verbatim
	y = sqrt(1 - x * x)
\endverbatim
The value of y, and its derivatives, are computed along with the value
and derivatives of z.

\copydetails CppAD::local::forward_unary2_op_dir
*/
template <class Base>
inline void forward_acos_op_dir(
	size_t q           ,
	size_t r           ,
	size_t i_z         ,
	size_t i_x         ,
	size_t cap_order   ,
	Base*  taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(AcosOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(AcosOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( 0 < q );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );

	// Taylor coefficients corresponding to argument and result
	size_t num_taylor_per_var = (cap_order-1) * r + 1;
	Base* x = taylor + i_x * num_taylor_per_var;
	Base* z = taylor + i_z * num_taylor_per_var;
	Base* b = z - num_taylor_per_var;  // called y in documentation

	size_t k, ell;
	size_t m = (q-1) * r + 1;
	for(ell = 0; ell < r; ell ++)
	{	Base uq = - 2.0 * x[m + ell] * x[0];
		for(k = 1; k < q; k++)
			uq -= x[(k-1)*r+1+ell] * x[(q-k-1)*r+1+ell];
		b[m+ell] = Base(0.0);
		z[m+ell] = Base(0.0);
		for(k = 1; k < q; k++)
		{	b[m+ell] += Base(double(k)) * b[(k-1)*r+1+ell] * b[(q-k-1)*r+1+ell];
			z[m+ell] += Base(double(k)) * z[(k-1)*r+1+ell] * b[(q-k-1)*r+1+ell];
		}
		b[m+ell] =  ( uq / Base(2.0) - b[m+ell] / Base(double(q)) ) / b[0];
		z[m+ell] = -( x[m+ell]     + z[m+ell] / Base(double(q)) ) / b[0];
	}
}

/*!
Compute zero order forward mode Taylor coefficient for result of op = AcosOp.

The C++ source code corresponding to this operation is
\verbatim
	z = acos(x)
\endverbatim
The auxillary result is
\verbatim
	y = sqrt( 1 - x * x )
\endverbatim
The value of y is computed along with the value of z.

\copydetails CppAD::local::forward_unary2_op_0
*/
template <class Base>
inline void forward_acos_op_0(
	size_t i_z         ,
	size_t i_x         ,
	size_t cap_order   ,
	Base*  taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(AcosOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(AcosOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( 0 < cap_order );

	// Taylor coefficients corresponding to argument and result
	Base* x = taylor + i_x * cap_order;
	Base* z = taylor + i_z * cap_order;
	Base* b = z      -       cap_order; // called y in documentation

	z[0] = acos( x[0] );
	b[0] = sqrt( Base(1.0) - x[0] * x[0] );
}
/*!
Compute reverse mode partial derivatives for result of op = AcosOp.

The C++ source code corresponding to this operation is
\verbatim
	z = acos(x)
\endverbatim
The auxillary result is
\verbatim
	y = sqrt( 1 - x * x )
\endverbatim
The value of y is computed along with the value of z.

\copydetails CppAD::local::reverse_unary2_op
*/

template <class Base>
inline void reverse_acos_op(
	size_t      d            ,
	size_t      i_z          ,
	size_t      i_x          ,
	size_t      cap_order    ,
	const Base* taylor       ,
	size_t      nc_partial   ,
	Base*       partial      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(AcosOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumRes(AcosOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( d < cap_order );
	CPPAD_ASSERT_UNKNOWN( d < nc_partial );

	// Taylor coefficients and partials corresponding to argument
	const Base* x  = taylor  + i_x * cap_order;
	Base* px       = partial + i_x * nc_partial;

	// Taylor coefficients and partials corresponding to first result
	const Base* z  = taylor  + i_z * cap_order;
	Base* pz       = partial + i_z * nc_partial;

	// Taylor coefficients and partials corresponding to auxillary result
	const Base* b  = z  - cap_order; // called y in documentation
	Base* pb       = pz - nc_partial;

	Base inv_b0 = Base(1.0) / b[0];

	// number of indices to access
	size_t j = d;
	size_t k;
	while(j)
	{
		// scale partials w.r.t b[j] by 1 / b[0]
		pb[j]  = azmul(pb[j], inv_b0);

		// scale partials w.r.t z[j] by 1 / b[0]
		pz[j]  = azmul(pz[j], inv_b0);

		// update partials w.r.t b^0
		pb[0] -= azmul(pz[j], z[j]) + azmul(pb[j], b[j]);

		// update partial w.r.t. x^0
		px[0] -= azmul(pb[j], x[j]);

		// update partial w.r.t. x^j
		px[j] -= pz[j] + azmul(pb[j], x[0]);

		// further scale partial w.r.t. z[j] by 1 / j
		pz[j] /= Base(double(j));

		for(k = 1; k < j; k++)
		{	// update partials w.r.t b^(j-k)
			pb[j-k] -= Base(double(k)) * azmul(pz[j], z[k]) + azmul(pb[j], b[k]);

			// update partials w.r.t. x^k
			px[k]   -= azmul(pb[j], x[j-k]);

			// update partials w.r.t. z^k
			pz[k]   -= Base(double(k)) * azmul(pz[j], b[j-k]);
		}
		--j;
	}

	// j == 0 case
	px[0] -= azmul( pz[0] + azmul(pb[0], x[0]), inv_b0);
}

} } // END_CPPAD_LOCAL_NAMESPACE
# endif
