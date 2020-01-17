# ifndef CPPAD_LOCAL_DIV_OP_HPP
# define CPPAD_LOCAL_DIV_OP_HPP

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
\file div_op.hpp
Forward and reverse mode calculations for z = x / y.
*/

// --------------------------- Divvv -----------------------------------------
/*!
Compute forward mode Taylor coefficients for result of op = DivvvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x / y
\endverbatim
In the documentation below,
this operations is for the case where both x and y are variables
and the argument \a parameter is not used.

\copydetails CppAD::local::forward_binary_op
*/

template <class Base>
inline void forward_divvv_op(
	size_t        p           ,
	size_t        q           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(DivvvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(DivvvOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( p <= q );

	// Taylor coefficients corresponding to arguments and result
	Base* x = taylor + arg[0] * cap_order;
	Base* y = taylor + arg[1] * cap_order;
	Base* z = taylor + i_z    * cap_order;


	// Using CondExp, it can make sense to divide by zero,
	// so do not make it an error.
	size_t k;
	for(size_t d = p; d <= q; d++)
	{	z[d] = x[d];
		for(k = 1; k <= d; k++)
			z[d] -= z[d-k] * y[k];
		z[d] /= y[0];
	}
}
/*!
Multiple directions forward mode Taylor coefficients for op = DivvvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x / y
\endverbatim
In the documentation below,
this operations is for the case where both x and y are variables
and the argument \a parameter is not used.

\copydetails CppAD::local::forward_binary_op_dir
*/

template <class Base>
inline void forward_divvv_op_dir(
	size_t        q           ,
	size_t        r           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(DivvvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(DivvvOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( 0 < q );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );

	// Taylor coefficients corresponding to arguments and result
	size_t num_taylor_per_var = (cap_order-1) * r + 1;
	Base* x = taylor + arg[0] * num_taylor_per_var;
	Base* y = taylor + arg[1] * num_taylor_per_var;
	Base* z = taylor + i_z    * num_taylor_per_var;


	// Using CondExp, it can make sense to divide by zero,
	// so do not make it an error.
	size_t m = (q-1) * r + 1;
	for(size_t ell = 0; ell < r; ell++)
	{	z[m+ell] = x[m+ell] - z[0] * y[m+ell];
		for(size_t k = 1; k < q; k++)
			z[m+ell] -= z[(q-k-1)*r+1+ell] * y[(k-1)*r+1+ell];
		z[m+ell] /= y[0];
	}
}


/*!
Compute zero order forward mode Taylor coefficients for result of op = DivvvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x / y
\endverbatim
In the documentation below,
this operations is for the case where both x and y are variables
and the argument \a parameter is not used.

\copydetails CppAD::local::forward_binary_op_0
*/

template <class Base>
inline void forward_divvv_op_0(
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(DivvvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(DivvvOp) == 1 );

	// Taylor coefficients corresponding to arguments and result
	Base* x = taylor + arg[0] * cap_order;
	Base* y = taylor + arg[1] * cap_order;
	Base* z = taylor + i_z    * cap_order;

	z[0] = x[0] / y[0];
}

/*!
Compute reverse mode partial derivatives for result of op = DivvvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x / y
\endverbatim
In the documentation below,
this operations is for the case where both x and y are variables
and the argument \a parameter is not used.

\copydetails CppAD::local::reverse_binary_op
*/

template <class Base>
inline void reverse_divvv_op(
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
	CPPAD_ASSERT_UNKNOWN( NumArg(DivvvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(DivvvOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( d < cap_order );
	CPPAD_ASSERT_UNKNOWN( d < nc_partial );

	// Arguments
	const Base* y  = taylor + arg[1] * cap_order;
	const Base* z  = taylor + i_z    * cap_order;

	// Partial derivatives corresponding to arguments and result
	Base* px = partial + arg[0] * nc_partial;
	Base* py = partial + arg[1] * nc_partial;
	Base* pz = partial + i_z    * nc_partial;

	// Using CondExp, it can make sense to divide by zero
	// so do not make it an error.
	Base inv_y0 = Base(1.0) / y[0];

	size_t k;
	// number of indices to access
	size_t j = d + 1;
	while(j)
	{	--j;
		// scale partial w.r.t. z[j]
		pz[j] = azmul(pz[j], inv_y0);

		px[j] += pz[j];
		for(k = 1; k <= j; k++)
		{	pz[j-k] -= azmul(pz[j], y[k]  );
			py[k]   -= azmul(pz[j], z[j-k]);
		}
		py[0] -= azmul(pz[j], z[j]);
	}
}

// --------------------------- Divpv -----------------------------------------
/*!
Compute forward mode Taylor coefficients for result of op = DivpvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x / y
\endverbatim
In the documentation below,
this operations is for the case where x is a parameter and y is a variable.

\copydetails CppAD::local::forward_binary_op
*/

template <class Base>
inline void forward_divpv_op(
	size_t        p           ,
	size_t        q           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(DivpvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(DivpvOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( p <= q );

	// Taylor coefficients corresponding to arguments and result
	Base* y = taylor + arg[1] * cap_order;
	Base* z = taylor + i_z    * cap_order;

	// Paraemter value
	Base x = parameter[ arg[0] ];

	// Using CondExp, it can make sense to divide by zero,
	// so do not make it an error.
	size_t k;
	if( p == 0 )
	{	z[0] = x / y[0];
		p++;
	}
	for(size_t d = p; d <= q; d++)
	{	z[d] = Base(0.0);
		for(k = 1; k <= d; k++)
			z[d] -= z[d-k] * y[k];
		z[d] /= y[0];
	}
}
/*!
Multiple directions forward mode Taylor coefficients for op = DivpvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x / y
\endverbatim
In the documentation below,
this operations is for the case where x is a parameter and y is a variable.

\copydetails CppAD::local::forward_binary_op_dir
*/

template <class Base>
inline void forward_divpv_op_dir(
	size_t        q           ,
	size_t        r           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(DivpvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(DivpvOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( 0 < q );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );

	// Taylor coefficients corresponding to arguments and result
	size_t num_taylor_per_var = (cap_order-1) * r + 1;
	Base* y = taylor + arg[1] * num_taylor_per_var;
	Base* z = taylor + i_z    * num_taylor_per_var;

	// Using CondExp, it can make sense to divide by zero,
	// so do not make it an error.
	size_t m = (q-1) * r + 1;
	for(size_t ell = 0; ell < r; ell++)
	{	z[m+ell] = - z[0] * y[m+ell];
		for(size_t k = 1; k < q; k++)
			z[m+ell] -= z[(q-k-1)*r+1+ell] * y[(k-1)*r+1+ell];
		z[m+ell] /= y[0];
	}
}

/*!
Compute zero order forward mode Taylor coefficient for result of op = DivpvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x / y
\endverbatim
In the documentation below,
this operations is for the case where x is a parameter and y is a variable.

\copydetails CppAD::local::forward_binary_op_0
*/

template <class Base>
inline void forward_divpv_op_0(
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(DivpvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(DivpvOp) == 1 );

	// Paraemter value
	Base x = parameter[ arg[0] ];

	// Taylor coefficients corresponding to arguments and result
	Base* y = taylor + arg[1] * cap_order;
	Base* z = taylor + i_z    * cap_order;

	z[0] = x / y[0];
}

/*!
Compute reverse mode partial derivative for result of op = DivpvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x / y
\endverbatim
In the documentation below,
this operations is for the case where x is a parameter and y is a variable.

\copydetails CppAD::local::reverse_binary_op
*/

template <class Base>
inline void reverse_divpv_op(
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
	CPPAD_ASSERT_UNKNOWN( NumArg(DivvvOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(DivvvOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( d < cap_order );
	CPPAD_ASSERT_UNKNOWN( d < nc_partial );

	// Arguments
	const Base* y = taylor + arg[1] * cap_order;
	const Base* z = taylor + i_z    * cap_order;

	// Partial derivatives corresponding to arguments and result
	Base* py = partial + arg[1] * nc_partial;
	Base* pz = partial + i_z    * nc_partial;

	// Using CondExp, it can make sense to divide by zero so do not
	// make it an error.
	Base inv_y0 = Base(1.0) / y[0];

	size_t k;
	// number of indices to access
	size_t j = d + 1;
	while(j)
	{	--j;
		// scale partial w.r.t z[j]
		pz[j] = azmul(pz[j], inv_y0);

		for(k = 1; k <= j; k++)
		{	pz[j-k] -= azmul(pz[j], y[k]  );
			py[k]   -= azmul(pz[j], z[j-k] );
		}
		py[0] -= azmul(pz[j], z[j]);
	}
}


// --------------------------- Divvp -----------------------------------------
/*!
Compute forward mode Taylor coefficients for result of op = DivvvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x / y
\endverbatim
In the documentation below,
this operations is for the case where x is a variable and y is a parameter.

\copydetails CppAD::local::forward_binary_op
*/

template <class Base>
inline void forward_divvp_op(
	size_t        p           ,
	size_t        q           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(DivvpOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(DivvpOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( p <= q );

	// Taylor coefficients corresponding to arguments and result
	Base* x = taylor + arg[0] * cap_order;
	Base* z = taylor + i_z    * cap_order;

	// Parameter value
	Base y = parameter[ arg[1] ];

	// Using CondExp and multiple levels of AD, it can make sense
	// to divide by zero so do not make it an error.
	for(size_t d = p; d <= q; d++)
		z[d] = x[d] / y;
}
/*!
Multiple direction forward mode Taylor coefficients for op = DivvvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x / y
\endverbatim
In the documentation below,
this operations is for the case where x is a variable and y is a parameter.

\copydetails CppAD::local::forward_binary_op_dir
*/

template <class Base>
inline void forward_divvp_op_dir(
	size_t        q           ,
	size_t        r           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(DivvpOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(DivvpOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( 0 < q  );

	// Taylor coefficients corresponding to arguments and result
	size_t num_taylor_per_var = (cap_order-1) * r + 1;
	Base* x = taylor + arg[0] * num_taylor_per_var;
	Base* z = taylor +    i_z * num_taylor_per_var;

	// Parameter value
	Base y = parameter[ arg[1] ];

	// Using CondExp and multiple levels of AD, it can make sense
	// to divide by zero so do not make it an error.
	size_t m = (q-1)*r + 1;
	for(size_t ell = 0; ell < r; ell++)
		z[m + ell] = x[m + ell] / y;
}


/*!
Compute zero order forward mode Taylor coefficients for result of op = DivvvOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x / y
\endverbatim
In the documentation below,
this operations is for the case where x is a variable and y is a parameter.

\copydetails CppAD::local::forward_binary_op_0
*/

template <class Base>
inline void forward_divvp_op_0(
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(DivvpOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(DivvpOp) == 1 );

	// Parameter value
	Base y = parameter[ arg[1] ];

	// Taylor coefficients corresponding to arguments and result
	Base* x = taylor + arg[0] * cap_order;
	Base* z = taylor + i_z    * cap_order;

	z[0] = x[0] / y;
}

/*!
Compute reverse mode partial derivative for result of op = DivvpOp.

The C++ source code corresponding to this operation is
\verbatim
	z = x / y
\endverbatim
In the documentation below,
this operations is for the case where x is a variable and y is a parameter.

\copydetails CppAD::local::reverse_binary_op
*/

template <class Base>
inline void reverse_divvp_op(
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
	CPPAD_ASSERT_UNKNOWN( NumArg(DivvpOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(DivvpOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( d < cap_order );
	CPPAD_ASSERT_UNKNOWN( d < nc_partial );

	// Argument values
	Base  y = parameter[ arg[1] ];

	// Partial derivatives corresponding to arguments and result
	Base* px = partial + arg[0] * nc_partial;
	Base* pz = partial + i_z    * nc_partial;

	// Using CondExp, it can make sense to divide by zero
	// so do not make it an error.
	Base inv_y = Base(1.0) / y;

	// number of indices to access
	size_t j = d + 1;
	while(j)
	{	--j;
		px[j] += azmul(pz[j], inv_y);
	}
}

} } // END_CPPAD_LOCAL_NAMESPACE
# endif
