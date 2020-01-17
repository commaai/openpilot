// $Id: sub_op.hpp 3865 2017-01-19 01:57:55Z bradbell $
# ifndef CPPAD_LOCAL_SUB_OP_HPP
# define CPPAD_LOCAL_SUB_OP_HPP

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
\file sub_op.hpp
Forward and reverse mode calculations for z = x - y.
*/

// --------------------------- Subvv -----------------------------------------
/*!
Compute forward mode Taylor coefficients for result of op = SubvvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x - y
\endverbatim
In the documentation below,
this operations is for the case where both x and y are variables
and the argument \a parameter is not used.

\copydetails CppAD::local::forward_binary_op
*/

template <class Base>
inline void forward_subvv_op(
	size_t        p           ,
	size_t        q           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SubvvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SubvvOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( p <= q );

	// Taylor coefficients corresponding to arguments and result
	Base* x = taylor + arg[0] * cap_order;
	Base* y = taylor + arg[1] * cap_order;
	Base* z = taylor + i_z    * cap_order;

	for(size_t d = p; d <= q; d++)
		z[d] = x[d] - y[d];
}
/*!
Multiple directions forward mode Taylor coefficients for op = SubvvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x - y
\endverbatim
In the documentation below,
this operations is for the case where both x and y are variables
and the argument \a parameter is not used.

\copydetails CppAD::local::forward_binary_op_dir
*/

template <class Base>
inline void forward_subvv_op_dir(
	size_t        q           ,
	size_t        r           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SubvvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SubvvOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( 0 < q );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );

	// Taylor coefficients corresponding to arguments and result
	size_t num_taylor_per_var = (cap_order-1) * r + 1;
	size_t m                  = (q-1) * r + 1;
	Base* x = taylor + arg[0] * num_taylor_per_var + m;
	Base* y = taylor + arg[1] * num_taylor_per_var + m;
	Base* z = taylor + i_z    * num_taylor_per_var + m;

	for(size_t ell = 0; ell < r; ell++)
		z[ell] = x[ell] - y[ell];
}

/*!
Compute zero order forward mode Taylor coefficients for result of op = SubvvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x - y
\endverbatim
In the documentation below,
this operations is for the case where both x and y are variables
and the argument \a parameter is not used.

\copydetails CppAD::local::forward_binary_op_0
*/

template <class Base>
inline void forward_subvv_op_0(
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SubvvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SubvvOp) == 1 );

	// Taylor coefficients corresponding to arguments and result
	Base* x = taylor + arg[0] * cap_order;
	Base* y = taylor + arg[1] * cap_order;
	Base* z = taylor + i_z    * cap_order;

	z[0] = x[0] - y[0];
}

/*!
Compute reverse mode partial derivatives for result of op = SubvvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x - y
\endverbatim
In the documentation below,
this operations is for the case where both x and y are variables
and the argument \a parameter is not used.

\copydetails CppAD::local::reverse_binary_op
*/

template <class Base>
inline void reverse_subvv_op(
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
	CPPAD_ASSERT_UNKNOWN( NumArg(SubvvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SubvvOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( d < cap_order );
	CPPAD_ASSERT_UNKNOWN( d < nc_partial );

	// Partial derivatives corresponding to arguments and result
	Base* px = partial + arg[0] * nc_partial;
	Base* py = partial + arg[1] * nc_partial;
	Base* pz = partial + i_z    * nc_partial;

	// number of indices to access
	size_t i = d + 1;
	while(i)
	{	--i;
		px[i] += pz[i];
		py[i] -= pz[i];
	}
}

// --------------------------- Subpv -----------------------------------------
/*!
Compute forward mode Taylor coefficients for result of op = SubpvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x - y
\endverbatim
In the documentation below,
this operations is for the case where x is a parameter and y is a variable.

\copydetails CppAD::local::forward_binary_op
*/

template <class Base>
inline void forward_subpv_op(
	size_t        p           ,
	size_t        q           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SubpvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SubpvOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( p <= q );

	// Taylor coefficients corresponding to arguments and result
	Base* y = taylor + arg[1] * cap_order;
	Base* z = taylor + i_z    * cap_order;

	// Paraemter value
	Base x = parameter[ arg[0] ];
	if( p == 0 )
	{	z[0] = x - y[0];
		p++;
	}
	for(size_t d = p; d <= q; d++)
		z[d] = - y[d];
}
/*!
Multiple directions forward mode Taylor coefficients for op = SubpvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x - y
\endverbatim
In the documentation below,
this operations is for the case where x is a parameter and y is a variable.

\copydetails CppAD::local::forward_binary_op_dir
*/

template <class Base>
inline void forward_subpv_op_dir(
	size_t        q           ,
	size_t        r           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SubpvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SubpvOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( 0 < q );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );

	// Taylor coefficients corresponding to arguments and result
	size_t num_taylor_per_var = (cap_order-1) * r + 1;
	size_t m                  = (q-1) * r + 1;
	Base* y = taylor + arg[1] * num_taylor_per_var + m;
	Base* z = taylor + i_z    * num_taylor_per_var + m;

	// Paraemter value
	for(size_t ell = 0; ell < r; ell++)
		z[ell] = - y[ell];
}
/*!
Compute zero order forward mode Taylor coefficient for result of op = SubpvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x - y
\endverbatim
In the documentation below,
this operations is for the case where x is a parameter and y is a variable.

\copydetails CppAD::local::forward_binary_op_0
*/

template <class Base>
inline void forward_subpv_op_0(
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SubpvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SubpvOp) == 1 );

	// Paraemter value
	Base x = parameter[ arg[0] ];

	// Taylor coefficients corresponding to arguments and result
	Base* y = taylor + arg[1] * cap_order;
	Base* z = taylor + i_z    * cap_order;

	z[0] = x - y[0];
}

/*!
Compute reverse mode partial derivative for result of op = SubpvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x - y
\endverbatim
In the documentation below,
this operations is for the case where x is a parameter and y is a variable.

\copydetails CppAD::local::reverse_binary_op
*/

template <class Base>
inline void reverse_subpv_op(
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
	CPPAD_ASSERT_UNKNOWN( NumArg(SubvvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SubvvOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( d < cap_order );
	CPPAD_ASSERT_UNKNOWN( d < nc_partial );

	// Partial derivatives corresponding to arguments and result
	Base* py = partial + arg[1] * nc_partial;
	Base* pz = partial + i_z    * nc_partial;

	// number of indices to access
	size_t i = d + 1;
	while(i)
	{	--i;
		py[i] -= pz[i];
	}
}

// --------------------------- Subvp -----------------------------------------
/*!
Compute forward mode Taylor coefficients for result of op = SubvvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x - y
\endverbatim
In the documentation below,
this operations is for the case where x is a variable and y is a parameter.

\copydetails CppAD::local::forward_binary_op
*/

template <class Base>
inline void forward_subvp_op(
	size_t        p           ,
	size_t        q           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SubvpOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SubvpOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( p <= q );

	// Taylor coefficients corresponding to arguments and result
	Base* x = taylor + arg[0] * cap_order;
	Base* z = taylor + i_z    * cap_order;

	// Parameter value
	Base y = parameter[ arg[1] ];
	if( p == 0 )
	{	z[0] = x[0] - y;
		p++;
	}
	for(size_t d = p; d <= q; d++)
		z[d] = x[d];
}
/*!
Multiple directions forward mode Taylor coefficients for op = SubvvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x - y
\endverbatim
In the documentation below,
this operations is for the case where x is a variable and y is a parameter.

\copydetails CppAD::local::forward_binary_op_dir
*/

template <class Base>
inline void forward_subvp_op_dir(
	size_t        q           ,
	size_t        r           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SubvpOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SubvpOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( 0 < q );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );

	// Taylor coefficients corresponding to arguments and result
	size_t num_taylor_per_var = (cap_order-1) * r + 1;
	Base* x = taylor + arg[0] * num_taylor_per_var;
	Base* z = taylor + i_z    * num_taylor_per_var;

	// Parameter value
	size_t m = (q-1) * r + 1;
	for(size_t ell = 0; ell < r; ell++)
		z[m+ell] = x[m+ell];
}

/*!
Compute zero order forward mode Taylor coefficients for result of op = SubvvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x - y
\endverbatim
In the documentation below,
this operations is for the case where x is a variable and y is a parameter.

\copydetails CppAD::local::forward_binary_op_0
*/

template <class Base>
inline void forward_subvp_op_0(
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(SubvpOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SubvpOp) == 1 );

	// Parameter value
	Base y = parameter[ arg[1] ];

	// Taylor coefficients corresponding to arguments and result
	Base* x = taylor + arg[0] * cap_order;
	Base* z = taylor + i_z    * cap_order;

	z[0] = x[0] - y;
}

/*!
Compute reverse mode partial derivative for result of op = SubvpOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x - y
\endverbatim
In the documentation below,
this operations is for the case where x is a variable and y is a parameter.

\copydetails CppAD::local::reverse_binary_op
*/

template <class Base>
inline void reverse_subvp_op(
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
	CPPAD_ASSERT_UNKNOWN( NumArg(SubvpOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(SubvpOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( d < cap_order );
	CPPAD_ASSERT_UNKNOWN( d < nc_partial );

	// Partial derivatives corresponding to arguments and result
	Base* px = partial + arg[0] * nc_partial;
	Base* pz = partial + i_z    * nc_partial;

	// number of indices to access
	size_t i = d + 1;
	while(i)
	{	--i;
		px[i] += pz[i];
	}
}

} } // END_CPPAD_LOCAL_NAMESPACE
# endif
