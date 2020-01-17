# ifndef CPPAD_LOCAL_ERF_OP_HPP
# define CPPAD_LOCAL_ERF_OP_HPP
# if CPPAD_USE_CPLUSPLUS_2011

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

# include <cppad/local/mul_op.hpp>
# include <cppad/local/sub_op.hpp>
# include <cppad/local/exp_op.hpp>


namespace CppAD { namespace local { // BEGIN_CPPAD_LOCAL_NAMESPACE
/*!
\file erf_op.hpp
Forward and reverse mode calculations for z = erf(x).
*/

/*!
Forward mode Taylor coefficient for result of op = ErfOp.

The C++ source code corresponding to this operation is
\verbatim
	z = erf(x)
\endverbatim

\tparam Base
base type for the operator; i.e., this operation was recorded
using AD< \a Base > and computations by this routine are done using type
\a Base.

\param p
lowest order of the Taylor coefficients that we are computing.

\param q
highest order of the Taylor coefficients that we are computing.

\param i_z
variable index corresponding to the last (primary) result for this operation;
i.e. the row index in \a taylor corresponding to z.
The auxillary results are called y_j have index \a i_z - j.

\param arg
arg[0]: is the variable index corresponding to x.
\n
arg[1]: is the parameter index corresponding to the value zero.
\n
\arg[2]: is  the parameter index correspodning to the value 2 / sqrt(pi).

\param parameter
parameter[ arg[1] ] is the value zero,
and parameter[ arg[2] ] is the value 2 / sqrt(pi).

\param cap_order
maximum number of orders that will fit in the \c taylor array.

\param taylor
\b Input:
taylor [ arg[0] * cap_order + k ]
for k = 0 , ... , q,
is the k-th order Taylor coefficient corresponding to x.
\n
\b Input:
taylor [ i_z * cap_order + k ]
for k = 0 , ... , p - 1,
is the k-th order Taylor coefficient corresponding to z.
\n
\b Input:
taylor [ ( i_z - j) * cap_order + k ]
for k = 0 , ... , p-1,
and j = 0 , ... , 4,
is the k-th order Taylor coefficient corresponding to the j-th result for z.
\n
\b Output:
taylor [ (i_z-j) * cap_order + k ],
for k = p , ... , q,
and j = 0 , ... , 4,
is the k-th order Taylor coefficient corresponding to the j-th result for z.

\par Checked Assertions
\li NumArg(op) == 3
\li NumRes(op) == 5
\li q < cap_order
\li p <= q
\li std::numeric_limits<addr_t>::max() >= i_z + 2
*/
template <class Base>
inline void forward_erf_op(
	size_t        p           ,
	size_t        q           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(ErfOp) == 3 );
	CPPAD_ASSERT_UNKNOWN( NumRes(ErfOp) == 5 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( p <= q );
	CPPAD_ASSERT_UNKNOWN( std::numeric_limits<addr_t>::max() >= i_z + 2 );

	// array used to pass parameter values for sub-operations
	addr_t addr[2];

	// convert from final result to first result
	i_z -= 4; // 4 = NumRes(ErfOp) - 1;

	// z_0 = x * x
	addr[0] = arg[0]; // x
	addr[1] = arg[0]; // x
	forward_mulvv_op(p, q, i_z+0, addr, parameter, cap_order, taylor);

	// z_1 = - x * x
	addr[0] = arg[1];           // zero
	addr[1] = addr_t( i_z );    // z_0
	forward_subpv_op(p, q, i_z+1, addr, parameter, cap_order, taylor);

	// z_2 = exp( - x * x )
	forward_exp_op(p, q, i_z+2, i_z+1, cap_order, taylor);

	// z_3 = (2 / sqrt(pi)) * exp( - x * x )
	addr[0] = arg[2];            // 2 / sqrt(pi)
	addr[1] = addr_t( i_z + 2 ); // z_2
	forward_mulpv_op(p, q, i_z+3, addr, parameter, cap_order, taylor);

	// pointers to taylor coefficients for x , z_3, and z_4
	Base* x    = taylor + arg[0]  * cap_order;
	Base* z_3  = taylor + (i_z+3) * cap_order;
	Base* z_4  = taylor + (i_z+4) * cap_order;

	// calculte z_4 coefficients
	if( p == 0 )
	{	// z4 (t) = erf[x(t)]
		z_4[0] = erf(x[0]);
		p++;
	}
	for(size_t j = p; j <= q; j++)
	{	// z_4' (t) = erf'[x(t)] * x'(t) = z3(t) * x'(t)
		// z_4[1] + 2 * z_4[2] * t +  ... =
		// (z_3[0] + z_3[1] * t +  ...) * (x[1] + 2 * x[2] * t +  ...)
		Base base_j = static_cast<Base>(double(j));
		z_4[j]      = static_cast<Base>(0);
		for(size_t k = 1; k <= j; k++)
			z_4[j] += (Base(double(k)) / base_j) * x[k] * z_3[j-k];
	}
}

/*!
Zero order Forward mode Taylor coefficient for result of op = ErfOp.

The C++ source code corresponding to this operation is
\verbatim
	z = erf(x)
\endverbatim

\tparam Base
base type for the operator; i.e., this operation was recorded
using AD< \a Base > and computations by this routine are done using type
\a Base.

\param i_z
variable index corresponding to the last (primary) result for this operation;
i.e. the row index in \a taylor corresponding to z.
The auxillary results are called y_j have index \a i_z - j.

\param arg
arg[0]: is the variable index corresponding to x.
\n
arg[1]: is the parameter index corresponding to the value zero.
\n
\arg[2]: is  the parameter index correspodning to the value 2 / sqrt(pi).

\param parameter
parameter[ arg[1] ] is the value zero,
and parameter[ arg[2] ] is the value 2 / sqrt(pi).

\param cap_order
maximum number of orders that will fit in the \c taylor array.

\param taylor
\b Input:
taylor [ arg[0] * cap_order + 0 ]
is the zero order Taylor coefficient corresponding to x.
\n
\b Input:
taylor [ i_z * cap_order + 0 ]
is the zero order Taylor coefficient corresponding to z.
\n
\b Output:
taylor [ (i_z-j) * cap_order + 0 ],
for j = 0 , ... , 4,
is the zero order Taylor coefficient for j-th result corresponding to z.

\par Checked Assertions
\li NumArg(op) == 3
\li NumRes(op) == 5
\li q < cap_order
\li p <= q
\li std::numeric_limits<addr_t>::max() >= i_z + 2
*/
template <class Base>
inline void forward_erf_op_0(
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(ErfOp) == 3 );
	CPPAD_ASSERT_UNKNOWN( NumRes(ErfOp) == 5 );
	CPPAD_ASSERT_UNKNOWN( 0 < cap_order );
	CPPAD_ASSERT_UNKNOWN( std::numeric_limits<addr_t>::max() >= i_z + 2 );

	// array used to pass parameter values for sub-operations
	addr_t addr[2];

	// convert from final result to first result
	i_z -= 4; // 4 = NumRes(ErfOp) - 1;

	// z_0 = x * x
	addr[0] = arg[0]; // x
	addr[1] = arg[0]; // x
	forward_mulvv_op_0(i_z+0, addr, parameter, cap_order, taylor);

	// z_1 = - x * x
	addr[0] = arg[1];       // zero
	addr[1] = addr_t(i_z);  // z_0
	forward_subpv_op_0(i_z+1, addr, parameter, cap_order, taylor);

	// z_2 = exp( - x * x )
	forward_exp_op_0(i_z+2, i_z+1, cap_order, taylor);

	// z_3 = (2 / sqrt(pi)) * exp( - x * x )
	addr[0] = arg[2];          // 2 / sqrt(pi)
	addr[1] = addr_t(i_z + 2); // z_2
	forward_mulpv_op_0(i_z+3, addr, parameter, cap_order, taylor);

	// zero order Taylor coefficient for z_4
	Base* x    = taylor + arg[0]  * cap_order;
	Base* z_4  = taylor + (i_z + 4) * cap_order;
	z_4[0] = erf(x[0]);
}
/*!
Forward mode Taylor coefficient for result of op = ErfOp.

The C++ source code corresponding to this operation is
\verbatim
	z = erf(x)
\endverbatim

\tparam Base
base type for the operator; i.e., this operation was recorded
using AD< \a Base > and computations by this routine are done using type
\a Base.

\param q
order of the Taylor coefficients that we are computing.

\param r
number of directions for the Taylor coefficients that we afre computing.

\param i_z
variable index corresponding to the last (primary) result for this operation;
i.e. the row index in \a taylor corresponding to z.
The auxillary results have index i_z - j for j = 0 , ... , 4
(and include z).

\param arg
arg[0]: is the variable index corresponding to x.
\n
arg[1]: is the parameter index corresponding to the value zero.
\n
\arg[2]: is  the parameter index correspodning to the value 2 / sqrt(pi).

\param parameter
parameter[ arg[1] ] is the value zero,
and parameter[ arg[2] ] is the value 2 / sqrt(pi).

\param cap_order
maximum number of orders that will fit in the \c taylor array.

\par tpv
We use the notation
<code>tpv = (cap_order-1) * r + 1</code>
which is the number of Taylor coefficients per variable

\param taylor
\b Input: If x is a variable,
<code>taylor [ arg[0] * tpv + 0 ]</code>,
is the zero order Taylor coefficient for all directions and
<code>taylor [ arg[0] * tpv + (k-1)*r + ell + 1 ]</code>,
for k = 1 , ... , q,
ell = 0, ..., r-1,
is the k-th order Taylor coefficient
corresponding to x and the ell-th direction.
\n
\b Input:
taylor [ (i_z - j) * tpv + 0 ]
is the zero order Taylor coefficient for all directions and the
j-th result for z.
for k = 1 , ... , q-1,
ell = 0, ... , r-1,
<code>
taylor[ (i_z - j) * tpv + (k-1)*r + ell + 1]
</code>
is the Taylor coefficient for the k-th order, ell-th direction,
and j-th auzillary result.
\n
\b Output:
taylor [ (i_z-j) * tpv + (q-1)*r + ell + 1 ],
for ell = 0 , ... , r-1,
is the Taylor coefficient for the q-th order, ell-th direction,
and j-th auzillary result.

\par Checked Assertions
\li NumArg(op) == 3
\li NumRes(op) == 5
\li 0 < q < cap_order
*/
template <class Base>
inline void forward_erf_op_dir(
	size_t        q           ,
	size_t        r           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(ErfOp) == 3 );
	CPPAD_ASSERT_UNKNOWN( NumRes(ErfOp) == 5 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( 0 < q );
	CPPAD_ASSERT_UNKNOWN( std::numeric_limits<addr_t>::max() >= i_z + 2 );

	// array used to pass parameter values for sub-operations
	addr_t addr[2];

	// convert from final result to first result
	i_z -= 4; // 4 = NumRes(ErfOp) - 1;

	// z_0 = x * x
	addr[0] = arg[0]; // x
	addr[1] = arg[0]; // x
	forward_mulvv_op_dir(q, r, i_z+0, addr, parameter, cap_order, taylor);

	// z_1 = - x * x
	addr[0] = arg[1];         // zero
	addr[1] = addr_t( i_z );  // z_0
	forward_subpv_op_dir(q, r, i_z+1, addr, parameter, cap_order, taylor);

	// z_2 = exp( - x * x )
	forward_exp_op_dir(q, r, i_z+2, i_z+1, cap_order, taylor);

	// z_3 = (2 / sqrt(pi)) * exp( - x * x )
	addr[0] = arg[2];            // 2 / sqrt(pi)
	addr[1] = addr_t( i_z + 2 ); // z_2
	forward_mulpv_op_dir(q, r, i_z+3, addr, parameter, cap_order, taylor);

	// pointers to taylor coefficients for x , z_3, and z_4
	size_t num_taylor_per_var = (cap_order - 1) * r + 1;
	Base* x    = taylor + arg[0]  * num_taylor_per_var;
	Base* z_3  = taylor + (i_z+3) * num_taylor_per_var;
	Base* z_4  = taylor + (i_z+4) * num_taylor_per_var;

	// z_4' (t) = erf'[x(t)] * x'(t) = z3(t) * x'(t)
	// z_4[1] + 2 * z_4[2] * t +  ... =
	// (z_3[0] + z_3[1] * t +  ...) * (x[1] + 2 * x[2] * t +  ...)
	Base base_q = static_cast<Base>(double(q));
	for(size_t ell = 0; ell < r; ell++)
	{	// index in z_4 and x for q-th order term
		size_t m = (q-1)*r + ell + 1;
		// initialize q-th order term summation
		z_4[m] = z_3[0] * x[m];
		for(size_t k = 1; k < q; k++)
		{	size_t x_index  = (k-1)*r + ell + 1;
			size_t z3_index = (q-k-1)*r + ell + 1;
			z_4[m]         += (Base(double(k)) / base_q) * x[x_index] * z_3[z3_index];
		}
	}
}

/*!
Compute reverse mode partial derivatives for result of op = ErfOp.

The C++ source code corresponding to this operation is
\verbatim
	z = erf(x)
\endverbatim

\tparam Base
base type for the operator; i.e., this operation was recorded
using AD< \a Base > and computations by this routine are done using type
\a Base.

\param d
highest order Taylor of the Taylor coefficients that we are computing
the partial derivatives with respect to.

\param i_z
variable index corresponding to the last (primary) result for this operation;
i.e. the row index in \a taylor corresponding to z.
The auxillary results are called y_j have index \a i_z - j.

\param arg
arg[0]: is the variable index corresponding to x.
\n
arg[1]: is the parameter index corresponding to the value zero.
\n
\arg[2]: is  the parameter index correspodning to the value 2 / sqrt(pi).

\param parameter
parameter[ arg[1] ] is the value zero,
and parameter[ arg[2] ] is the value 2 / sqrt(pi).

\param cap_order
maximum number of orders that will fit in the \c taylor array.

\param taylor
\b Input:
taylor [ arg[0] * cap_order + k ]
for k = 0 , ... , d,
is the k-th order Taylor coefficient corresponding to x.
\n
taylor [ (i_z - j) * cap_order + k ]
for k = 0 , ... , d,
and for j = 0 , ... , 4,
is the k-th order Taylor coefficient corresponding to the j-th result
for this operation.

\param nc_partial
number of columns in the matrix containing all the partial derivatives

\param partial
\b Input:
partial [ arg[0] * nc_partial + k ]
for k = 0 , ... , d,
is the partial derivative of G( z , x , w , u , ... ) with respect to
the k-th order Taylor coefficient for x.
\n
\b Input:
partial [ (i_z - j) * nc_partial + k ]
for k = 0 , ... , d,
and for j = 0 , ... , 4,
is the partial derivative of G( z , x , w , u , ... ) with respect to
the k-th order Taylor coefficient for the j-th result of this operation.
\n
\b Output:
partial [ arg[0] * nc_partial + k ]
for k = 0 , ... , d,
is the partial derivative of H( x , w , u , ... ) with respect to
the k-th order Taylor coefficient for x.
\n
\b Output:
partial [ (i_z-j) * nc_partial + k ]
for k = 0 , ... , d,
and for j = 0 , ... , 4,
may be used as work space; i.e., may change in an unspecified manner.

\par Checked Assertions
\li NumArg(op) == 3
\li NumRes(op) == 5
\li q < cap_order
\li p <= q
*/
template <class Base>
inline void reverse_erf_op(
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
	CPPAD_ASSERT_UNKNOWN( NumArg(ErfOp) == 3 );
	CPPAD_ASSERT_UNKNOWN( NumRes(ErfOp) == 5 );
	CPPAD_ASSERT_UNKNOWN( d < cap_order );
	CPPAD_ASSERT_UNKNOWN( std::numeric_limits<addr_t>::max() >= i_z + 2 );

	// array used to pass parameter values for sub-operations
	addr_t addr[2];

	// If pz is zero, make sure this operation has no effect
	// (zero times infinity or nan would be non-zero).
	Base* pz  = partial + i_z * nc_partial;
	bool skip(true);
	for(size_t i_d = 0; i_d <= d; i_d++)
		skip &= IdenticalZero(pz[i_d]);
	if( skip )
		return;

	// convert from final result to first result
	i_z -= 4; // 4 = NumRes(ErfOp) - 1;

	// Taylor coefficients and partials corresponding to x
	const Base* x  = taylor  + arg[0]  * cap_order;
	Base* px       = partial + arg[0] * nc_partial;

	// Taylor coefficients and partials corresponding to z_3
	const Base* z_3  = taylor  + (i_z+3) * cap_order;
	Base* pz_3       = partial + (i_z+3) * nc_partial;

	// Taylor coefficients and partials corresponding to z_4
	Base* pz_4 = partial + (i_z+4) * nc_partial;

	// Reverse z_4
	size_t j = d;
	while(j)
	{	pz_4[j] /= Base(double(j));
		for(size_t k = 1; k <= j; k++)
		{	px[k]     += azmul(pz_4[j], z_3[j-k]) * Base(double(k));
			pz_3[j-k] += azmul(pz_4[j], x[k]) * Base(double(k));
		}
		j--;
	}
	px[0] += azmul(pz_4[0], z_3[0]);

	// z_3 = (2 / sqrt(pi)) * exp( - x * x )
	addr[0] = arg[2];            // 2 / sqrt(pi)
	addr[1] = addr_t( i_z + 2 ); // z_2
	reverse_mulpv_op(
		d, i_z+3, addr, parameter, cap_order, taylor, nc_partial, partial
	);

	// z_2 = exp( - x * x )
	reverse_exp_op(
		d, i_z+2, i_z+1, cap_order, taylor, nc_partial, partial
	);

	// z_1 = - x * x
	addr[0] = arg[1];           // zero
	addr[1] = addr_t( i_z );    // z_0
	reverse_subpv_op(
		d, i_z+1, addr, parameter, cap_order, taylor, nc_partial, partial
	);

	// z_0 = x * x
	addr[0] = arg[0]; // x
	addr[1] = arg[0]; // x
	reverse_mulvv_op(
		d, i_z+0, addr, parameter, cap_order, taylor, nc_partial, partial
	);

}


} } // END_CPPAD_LOCAL_NAMESPACE
# endif // CPPAD_USE_CPLUSPLUS_2011
# endif // CPPAD_ERF_OP_INCLUDED
