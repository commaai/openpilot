# ifndef CPPAD_LOCAL_ZMUL_OP_HPP
# define CPPAD_LOCAL_ZMUL_OP_HPP

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
\file mul_op.hpp
Forward and reverse mode calculations for z = azmul(x, y).
*/

// --------------------------- Zmulvv -----------------------------------------
/*!
Compute forward mode Taylor coefficients for result of op = ZmulvvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = azmul(x, y)
\endverbatim
In the documentation below,
this operations is for the case where both x and y are variables
and the argument \a parameter is not used.

\copydetails CppAD::local::forward_binary_op
*/

template <class Base>
inline void forward_zmulvv_op(
	size_t        p           ,
	size_t        q           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(ZmulvvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(ZmulvvOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( p <= q );

	// Taylor coefficients corresponding to arguments and result
	Base* x = taylor + arg[0] * cap_order;
	Base* y = taylor + arg[1] * cap_order;
	Base* z = taylor + i_z    * cap_order;

	size_t k;
	for(size_t d = p; d <= q; d++)
	{	z[d] = Base(0.0);
		for(k = 0; k <= d; k++)
			z[d] += azmul(x[d-k], y[k]);
	}
}
/*!
Multiple directions forward mode Taylor coefficients for op = ZmulvvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = azmul(x, y)
\endverbatim
In the documentation below,
this operations is for the case where both x and y are variables
and the argument \a parameter is not used.

\copydetails CppAD::local::forward_binary_op_dir
*/

template <class Base>
inline void forward_zmulvv_op_dir(
	size_t        q           ,
	size_t        r           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(ZmulvvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(ZmulvvOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( 0 < q );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );

	// Taylor coefficients corresponding to arguments and result
	size_t num_taylor_per_var = (cap_order-1) * r + 1;
	Base* x = taylor + arg[0] * num_taylor_per_var;
	Base* y = taylor + arg[1] * num_taylor_per_var;
	Base* z = taylor +    i_z * num_taylor_per_var;

	size_t k, ell, m;
	for(ell = 0; ell < r; ell++)
	{	m = (q-1)*r + ell + 1;
		z[m] = azmul(x[0], y[m]) + azmul(x[m],  y[0]);
		for(k = 1; k < q; k++)
			z[m] += azmul(x[(q-k-1)*r + ell + 1], y[(k-1)*r + ell + 1]);
	}
}

/*!
Compute zero order forward mode Taylor coefficients for result of op = ZmulvvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = azmul(x, y)
\endverbatim
In the documentation below,
this operations is for the case where both x and y are variables
and the argument \a parameter is not used.

\copydetails CppAD::local::forward_binary_op_0
*/

template <class Base>
inline void forward_zmulvv_op_0(
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(ZmulvvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(ZmulvvOp) == 1 );

	// Taylor coefficients corresponding to arguments and result
	Base* x = taylor + arg[0] * cap_order;
	Base* y = taylor + arg[1] * cap_order;
	Base* z = taylor + i_z    * cap_order;

	z[0] = azmul(x[0], y[0]);
}

/*!
Compute reverse mode partial derivatives for result of op = ZmulvvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = azmul(x, y)
\endverbatim
In the documentation below,
this operations is for the case where both x and y are variables
and the argument \a parameter is not used.

\copydetails CppAD::local::reverse_binary_op
*/

template <class Base>
inline void reverse_zmulvv_op(
	size_t        d           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	const Base*   taylor      ,
	size_t        nc_partial  ,
	Base*         partial     )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(ZmulvvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(ZmulvvOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( d < cap_order );
	CPPAD_ASSERT_UNKNOWN( d < nc_partial );

	// Arguments
	const Base* x  = taylor + arg[0] * cap_order;
	const Base* y  = taylor + arg[1] * cap_order;

	// Partial derivatives corresponding to arguments and result
	Base* px = partial + arg[0] * nc_partial;
	Base* py = partial + arg[1] * nc_partial;
	Base* pz = partial + i_z    * nc_partial;

	// number of indices to access
	size_t j = d + 1;
	size_t k;
	while(j)
	{	--j;
		for(k = 0; k <= j; k++)
		{
			px[j-k] += azmul(pz[j], y[k]);
			py[k]   += azmul(pz[j], x[j-k]);
		}
	}
}
// --------------------------- Zmulpv -----------------------------------------
/*!
Compute forward mode Taylor coefficients for result of op = ZmulpvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = azmul(x, y)
\endverbatim
In the documentation below,
this operations is for the case where x is a parameter and y is a variable.

\copydetails CppAD::local::forward_binary_op
*/

template <class Base>
inline void forward_zmulpv_op(
	size_t        p           ,
	size_t        q           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(ZmulpvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(ZmulpvOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( p <= q );

	// Taylor coefficients corresponding to arguments and result
	Base* y = taylor + arg[1] * cap_order;
	Base* z = taylor + i_z    * cap_order;

	// Paraemter value
	Base x = parameter[ arg[0] ];

	for(size_t d = p; d <= q; d++)
		z[d] = azmul(x, y[d]);
}
/*!
Multiple directions forward mode Taylor coefficients for op = ZmulpvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = azmul(x, y)
\endverbatim
In the documentation below,
this operations is for the case where x is a parameter and y is a variable.

\copydetails CppAD::local::forward_binary_op_dir
*/

template <class Base>
inline void forward_zmulpv_op_dir(
	size_t        q           ,
	size_t        r           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(ZmulpvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(ZmulpvOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( 0 < q );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );

	// Taylor coefficients corresponding to arguments and result
	size_t num_taylor_per_var = (cap_order-1) * r + 1;
	size_t m                  = (q-1) * r + 1;
	Base* y = taylor + arg[1] * num_taylor_per_var + m;
	Base* z = taylor + i_z    * num_taylor_per_var + m;

	// Paraemter value
	Base x = parameter[ arg[0] ];

	for(size_t ell = 0; ell < r; ell++)
		z[ell] = azmul(x, y[ell]);
}
/*!
Compute zero order forward mode Taylor coefficient for result of op = ZmulpvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = azmul(x, y)
\endverbatim
In the documentation below,
this operations is for the case where x is a parameter and y is a variable.

\copydetails CppAD::local::forward_binary_op_0
*/

template <class Base>
inline void forward_zmulpv_op_0(
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(ZmulpvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(ZmulpvOp) == 1 );

	// Paraemter value
	Base x = parameter[ arg[0] ];

	// Taylor coefficients corresponding to arguments and result
	Base* y = taylor + arg[1] * cap_order;
	Base* z = taylor + i_z    * cap_order;

	z[0] = azmul(x, y[0]);
}

/*!
Compute reverse mode partial derivative for result of op = ZmulpvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = azmul(x, y)
\endverbatim
In the documentation below,
this operations is for the case where x is a parameter and y is a variable.

\copydetails CppAD::local::reverse_binary_op
*/

template <class Base>
inline void reverse_zmulpv_op(
	size_t        d           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	const Base*   taylor      ,
	size_t        nc_partial  ,
	Base*         partial     )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(ZmulpvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(ZmulpvOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( d < cap_order );
	CPPAD_ASSERT_UNKNOWN( d < nc_partial );

	// Arguments
	Base x  = parameter[ arg[0] ];

	// Partial derivatives corresponding to arguments and result
	Base* py = partial + arg[1] * nc_partial;
	Base* pz = partial + i_z    * nc_partial;

	// number of indices to access
	size_t j = d + 1;
	while(j)
	{	--j;
		py[j] += azmul(pz[j], x);
	}
}
// --------------------------- Zmulvp -----------------------------------------
/*!
Compute forward mode Taylor coefficients for result of op = ZmulvpOp.

The C++ source code corresponding to this operation is
\verbatim
	z = azmul(x, y)
\endverbatim
In the documentation below,
this operations is for the case where x is a parameter and y is a variable.

\copydetails CppAD::local::forward_binary_op
*/

template <class Base>
inline void forward_zmulvp_op(
	size_t        p           ,
	size_t        q           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(ZmulvpOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(ZmulvpOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( p <= q );

	// Taylor coefficients corresponding to arguments and result
	Base* x = taylor + arg[0] * cap_order;
	Base* z = taylor + i_z    * cap_order;

	// Paraemter value
	Base y = parameter[ arg[1] ];

	for(size_t d = p; d <= q; d++)
		z[d] = azmul(x[d], y);
}
/*!
Multiple directions forward mode Taylor coefficients for op = ZmulvpOp.

The C++ source code corresponding to this operation is
\verbatim
	z = azmul(x, y)
\endverbatim
In the documentation below,
this operations is for the case where x is a parameter and y is a variable.

\copydetails CppAD::local::forward_binary_op_dir
*/

template <class Base>
inline void forward_zmulvp_op_dir(
	size_t        q           ,
	size_t        r           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(ZmulvpOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(ZmulvpOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( 0 < q );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );

	// Taylor coefficients corresponding to arguments and result
	size_t num_taylor_per_var = (cap_order-1) * r + 1;
	size_t m                  = (q-1) * r + 1;
	Base* x = taylor + arg[0] * num_taylor_per_var + m;
	Base* z = taylor + i_z    * num_taylor_per_var + m;

	// Paraemter value
	Base y = parameter[ arg[1] ];

	for(size_t ell = 0; ell < r; ell++)
		z[ell] = azmul(x[ell], y);
}
/*!
Compute zero order forward mode Taylor coefficient for result of op = ZmulvpOp.

The C++ source code corresponding to this operation is
\verbatim
	z = azmul(x, y)
\endverbatim
In the documentation below,
this operations is for the case where x is a parameter and y is a variable.

\copydetails CppAD::local::forward_binary_op_0
*/

template <class Base>
inline void forward_zmulvp_op_0(
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(ZmulvpOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(ZmulvpOp) == 1 );

	// Paraemter value
	Base y = parameter[ arg[1] ];

	// Taylor coefficients corresponding to arguments and result
	Base* x = taylor + arg[0] * cap_order;
	Base* z = taylor + i_z    * cap_order;

	z[0] = azmul(x[0], y);
}

/*!
Compute reverse mode partial derivative for result of op = ZmulvpOp.

The C++ source code corresponding to this operation is
\verbatim
	z = azmul(x, y)
\endverbatim
In the documentation below,
this operations is for the case where x is a parameter and y is a variable.

\copydetails CppAD::local::reverse_binary_op
*/

template <class Base>
inline void reverse_zmulvp_op(
	size_t        d           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	const Base*   taylor      ,
	size_t        nc_partial  ,
	Base*         partial     )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(ZmulvpOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(ZmulvpOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( d < cap_order );
	CPPAD_ASSERT_UNKNOWN( d < nc_partial );

	// Arguments
	Base y  = parameter[ arg[1] ];

	// Partial derivatives corresponding to arguments and result
	Base* px = partial + arg[0] * nc_partial;
	Base* pz = partial + i_z    * nc_partial;

	// number of indices to access
	size_t j = d + 1;
	while(j)
	{	--j;
		px[j] += azmul(pz[j], y);
	}
}

} } // END_CPPAD_LOCAL_NAMESPACE
# endif
