// $Id: csum_op.hpp 3845 2016-11-19 01:50:47Z bradbell $
# ifndef CPPAD_LOCAL_CSUM_OP_HPP
# define CPPAD_LOCAL_CSUM_OP_HPP

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
\file csum_op.hpp
Forward, reverse and sparsity calculations for cummulative summation.
*/

/*!
Compute forward mode Taylor coefficients for result of op = CsumOp.

This operation is
\verbatim
	z = s + x(1) + ... + x(m) - y(1) - ... - y(n).
\endverbatim

\tparam Base
base type for the operator; i.e., this operation was recorded
using AD< \a Base > and computations by this routine are done using type
\a Base.

\param p
lowest order of the Taylor coefficient that we are computing.

\param q
highest order of the Taylor coefficient that we are computing.

\param i_z
variable index corresponding to the result for this operation;
i.e. the row index in \a taylor corresponding to z.

\param arg
\a arg[0]
is the number of addition variables in this cummulative summation; i.e.,
<tt>m</tt>.
\n
\a arg[1]
is the number of subtraction variables in this cummulative summation; i.e.,
\c m.
\n
<tt>parameter[ arg[2] ]</tt>
is the parameter value \c s in this cummunative summation.
\n
<tt>arg[2+i]</tt>
for <tt>i = 1 , ... , m</tt> is the variable index of <tt>x(i)</tt>.
\n
<tt>arg[2+arg[0]+i]</tt>
for <tt>i = 1 , ... , n</tt> is the variable index of <tt>y(i)</tt>.

\param num_par
is the number of parameters in \a parameter.

\param parameter
is the parameter vector for this operation sequence.

\param cap_order
number of colums in the matrix containing all the Taylor coefficients.

\param taylor
\b Input: <tt>taylor [ arg[2+i] * cap_order + k ]</tt>
for <tt>i = 1 , ... , m</tt>
and <tt>k = 0 , ... , q</tt>
is the k-th order Taylor coefficient corresponding to <tt>x(i)</tt>
\n
\b Input: <tt>taylor [ arg[2+m+i] * cap_order + k ]</tt>
for <tt>i = 1 , ... , n</tt>
and <tt>k = 0 , ... , q</tt>
is the k-th order Taylor coefficient corresponding to <tt>y(i)</tt>
\n
\b Input: <tt>taylor [ i_z * cap_order + k ]</tt>
for k = 0 , ... , p,
is the k-th order Taylor coefficient corresponding to z.
\n
\b Output: <tt>taylor [ i_z * cap_order + k ]</tt>
for k = p , ... , q,
is the \a k-th order Taylor coefficient corresponding to z.
*/
template <class Base>
inline void forward_csum_op(
	size_t        p           ,
	size_t        q           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	size_t        num_par     ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{	Base zero(0);
	size_t i, j, k;

	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumRes(CSumOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( p <= q );
	CPPAD_ASSERT_UNKNOWN( size_t(arg[2]) < num_par );
	CPPAD_ASSERT_UNKNOWN(
		arg[0] + arg[1] == arg[ arg[0] + arg[1] + 3 ]
	);

	// Taylor coefficients corresponding to result
	Base* z = taylor + i_z    * cap_order;
	for(k = p; k <= q; k++)
		z[k] = zero;
	if( p == 0 )
		z[p] = parameter[ arg[2] ];
	Base* x;
	i = arg[0];
	j = 2;
	while(i--)
	{	CPPAD_ASSERT_UNKNOWN( size_t(arg[j+1]) < i_z );
		x     = taylor + arg[++j] * cap_order;
		for(k = p; k <= q; k++)
			z[k] += x[k];
	}
	i = arg[1];
	while(i--)
	{	CPPAD_ASSERT_UNKNOWN( size_t(arg[j+1]) < i_z );
		x     = taylor + arg[++j] * cap_order;
		for(k = p; k <= q; k++)
			z[k] -= x[k];
	}
}

/*!
Multiple direction forward mode Taylor coefficients for op = CsumOp.

This operation is
\verbatim
	z = s + x(1) + ... + x(m) - y(1) - ... - y(n).
\endverbatim

\tparam Base
base type for the operator; i.e., this operation was recorded
using AD<Base> and computations by this routine are done using type
\a Base.

\param q
order ot the Taylor coefficients that we are computing.

\param r
number of directions for Taylor coefficients that we are computing.

\param i_z
variable index corresponding to the result for this operation;
i.e. the row index in \a taylor corresponding to z.

\param arg
\a arg[0]
is the number of addition variables in this cummulative summation; i.e.,
<tt>m</tt>.
\n
\a arg[1]
is the number of subtraction variables in this cummulative summation; i.e.,
\c m.
\n
<tt>parameter[ arg[2] ]</tt>
is the parameter value \c s in this cummunative summation.
\n
<tt>arg[2+i]</tt>
for <tt>i = 1 , ... , m</tt> is the variable index of <tt>x(i)</tt>.
\n
<tt>arg[2+arg[0]+i]</tt>
for <tt>i = 1 , ... , n</tt> is the variable index of <tt>y(i)</tt>.

\param num_par
is the number of parameters in \a parameter.

\param parameter
is the parameter vector for this operation sequence.

\param cap_order
number of colums in the matrix containing all the Taylor coefficients.

\param taylor
\b Input: <tt>taylor [ arg[2+i]*((cap_order-1)*r + 1) + 0 ]</tt>
for <tt>i = 1 , ... , m</tt>
is the 0-th order Taylor coefficient corresponding to <tt>x(i)</tt> and
<tt>taylor [ arg[2+i]*((cap_order-1)*r + 1) + (q-1)*r + ell + 1 ]</tt>
for <tt>i = 1 , ... , m</tt>,
<tt>ell = 0 , ... , r-1</tt>
is the q-th order Taylor coefficient corresponding to <tt>x(i)</tt>
and direction ell.
\n
\b Input: <tt>taylor [ arg[2+m+i]*((cap_order-1)*r + 1) + 0 ]</tt>
for <tt>i = 1 , ... , n</tt>
is the 0-th order Taylor coefficient corresponding to <tt>y(i)</tt> and
<tt>taylor [ arg[2+m+i]*((cap_order-1)*r + 1) + (q-1)*r + ell + 1 ]</tt>
for <tt>i = 1 , ... , n</tt>,
<tt>ell = 0 , ... , r-1</tt>
is the q-th order Taylor coefficient corresponding to <tt>y(i)</tt>
and direction ell.
\n
\b Output: <tt>taylor [ i_z*((cap_order-1)*r+1) + (q-1)*r + ell + 1 ]</tt>
is the \a q-th order Taylor coefficient corresponding to z
for direction <tt>ell = 0 , ... , r-1</tt>.
*/
template <class Base>
inline void forward_csum_op_dir(
	size_t        q           ,
	size_t        r           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	size_t        num_par     ,
	const Base*   parameter   ,
	size_t        cap_order   ,
	Base*         taylor      )
{	Base zero(0);
	size_t i, j, ell;

	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumRes(CSumOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( 0 < q );
	CPPAD_ASSERT_UNKNOWN( size_t(arg[2]) < num_par );
	CPPAD_ASSERT_UNKNOWN(
		arg[0] + arg[1] == arg[ arg[0] + arg[1] + 3 ]
	);

	// Taylor coefficients corresponding to result
	size_t num_taylor_per_var = (cap_order-1) * r + 1;
	size_t m                  = (q-1)*r + 1;
	Base* z = taylor + i_z * num_taylor_per_var + m;
	for(ell = 0; ell < r; ell++)
		z[ell] = zero;
	Base* x;
	i = arg[0];
	j = 2;
	while(i--)
	{	CPPAD_ASSERT_UNKNOWN( size_t(arg[j+1]) < i_z );
		x = taylor + arg[++j] * num_taylor_per_var + m;
		for(ell = 0; ell < r; ell++)
			z[ell] += x[ell];
	}
	i = arg[1];
	while(i--)
	{	CPPAD_ASSERT_UNKNOWN( size_t(arg[j+1]) < i_z );
		x = taylor + arg[++j] * num_taylor_per_var + m;
		for(ell = 0; ell < r; ell++)
			z[ell] -= x[ell];
	}
}

/*!
Compute reverse mode Taylor coefficients for result of op = CsumOp.

This operation is
\verbatim
	z = q + x(1) + ... + x(m) - y(1) - ... - y(n).
	H(y, x, w, ...) = G[ z(x, y), y, x, w, ... ]
\endverbatim

\tparam Base
base type for the operator; i.e., this operation was recorded
using AD< \a Base > and computations by this routine are done using type
\a Base.

\param d
order the highest order Taylor coefficient that we are computing
the partial derivatives with respect to.

\param i_z
variable index corresponding to the result for this operation;
i.e. the row index in \a taylor corresponding to z.

\param arg
\a arg[0]
is the number of addition variables in this cummulative summation; i.e.,
<tt>m</tt>.
\n
\a arg[1]
is the number of subtraction variables in this cummulative summation; i.e.,
\c m.
\n
<tt>parameter[ arg[2] ]</tt>
is the parameter value \c q in this cummunative summation.
\n
<tt>arg[2+i]</tt>
for <tt>i = 1 , ... , m</tt> is the value <tt>x(i)</tt>.
\n
<tt>arg[2+arg[0]+i]</tt>
for <tt>i = 1 , ... , n</tt> is the value <tt>y(i)</tt>.

\param nc_partial
number of colums in the matrix containing all the partial derivatives.

\param partial
\b Input: <tt>partial [ arg[2+i] * nc_partial + k ]</tt>
for <tt>i = 1 , ... , m</tt>
and <tt>k = 0 , ... , d</tt>
is the partial derivative of G(z, y, x, w, ...) with respect to the
k-th order Taylor coefficient corresponding to <tt>x(i)</tt>
\n
\b Input: <tt>partial [ arg[2+m+i] * nc_partial + k ]</tt>
for <tt>i = 1 , ... , n</tt>
and <tt>k = 0 , ... , d</tt>
is the partial derivative of G(z, y, x, w, ...) with respect to the
k-th order Taylor coefficient corresponding to <tt>y(i)</tt>
\n
\b Input: <tt>partial [ i_z * nc_partial + k ]</tt>
for <tt>i = 1 , ... , n</tt>
and <tt>k = 0 , ... , d</tt>
is the partial derivative of G(z, y, x, w, ...) with respect to the
k-th order Taylor coefficient corresponding to \c z.
\n
\b Output: <tt>partial [ arg[2+i] * nc_partial + k ]</tt>
for <tt>i = 1 , ... , m</tt>
and <tt>k = 0 , ... , d</tt>
is the partial derivative of H(y, x, w, ...) with respect to the
k-th order Taylor coefficient corresponding to <tt>x(i)</tt>
\n
\b Output: <tt>partial [ arg[2+m+i] * nc_partial + k ]</tt>
for <tt>i = 1 , ... , n</tt>
and <tt>k = 0 , ... , d</tt>
is the partial derivative of H(y, x, w, ...) with respect to the
k-th order Taylor coefficient corresponding to <tt>y(i)</tt>
*/

template <class Base>
inline void reverse_csum_op(
	size_t        d           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	size_t        nc_partial  ,
	Base*         partial     )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumRes(CSumOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( d < nc_partial );

	// Taylor coefficients and partial derivative corresponding to result
	Base* pz = partial + i_z * nc_partial;
	Base* px;
	size_t i, j, k;
	size_t d1 = d + 1;
	i = arg[0];
	j = 2;
	while(i--)
	{	CPPAD_ASSERT_UNKNOWN( size_t(arg[j+1]) < i_z );
		px    = partial + arg[++j] * nc_partial;
		k = d1;
		while(k--)
			px[k] += pz[k];
	}
	i = arg[1];
	while(i--)
	{	CPPAD_ASSERT_UNKNOWN( size_t(arg[j+1]) < i_z );
		px    = partial + arg[++j] * nc_partial;
		k = d1;
		while(k--)
			px[k] -= pz[k];
	}
}


/*!
Forward mode Jacobian sparsity pattern for CSumOp operator.

This operation is
\verbatim
	z = q + x(1) + ... + x(m) - y(1) - ... - y(n).
\endverbatim

\tparam Vector_set
is the type used for vectors of sets. It can be either
sparse_pack or sparse_list.

\param i_z
variable index corresponding to the result for this operation;
i.e. the index in \a sparsity corresponding to z.

\param arg
\a arg[0]
is the number of addition variables in this cummulative summation; i.e.,
<tt>m + n</tt>.
\n
\a arg[1]
is the number of subtraction variables in this cummulative summation; i.e.,
\c m.
\n
<tt>parameter[ arg[2] ]</tt>
is the parameter value \c q in this cummunative summation.
\n
<tt>arg[2+i]</tt>
for <tt>i = 1 , ... , m</tt> is the value <tt>x(i)</tt>.
\n
<tt>arg[2+arg[1]+i]</tt>
for <tt>i = 1 , ... , n</tt> is the value <tt>y(i)</tt>.

\param sparsity
\b Input:
For <tt>i = 1 , ... , m</tt>,
the set with index \a arg[2+i] in \a sparsity
is the sparsity bit pattern for <tt>x(i)</tt>.
This identifies which of the independent variables the variable
<tt>x(i)</tt> depends on.
\n
\b Input:
For <tt>i = 1 , ... , n</tt>,
the set with index \a arg[2+arg[0]+i] in \a sparsity
is the sparsity bit pattern for <tt>x(i)</tt>.
This identifies which of the independent variables the variable
<tt>y(i)</tt> depends on.
\n
\b Output:
The set with index \a i_z in \a sparsity
is the sparsity bit pattern for z.
This identifies which of the independent variables the variable z
depends on.
*/

template <class Vector_set>
inline void forward_sparse_jacobian_csum_op(
	size_t           i_z         ,
	const addr_t*    arg         ,
	Vector_set&      sparsity    )
{	sparsity.clear(i_z);

	size_t i, j;
	i = arg[0] + arg[1];
	j = 2;
	while(i--)
	{	CPPAD_ASSERT_UNKNOWN( size_t(arg[j+1]) < i_z );
		sparsity.binary_union(
			i_z        , // index in sparsity for result
			i_z        , // index in sparsity for left operand
			arg[++j]   , // index for right operand
			sparsity     // sparsity vector for right operand
		);
	}
}

/*!
Reverse mode Jacobian sparsity pattern for CSumOp operator.

This operation is
\verbatim
	z = q + x(1) + ... + x(m) - y(1) - ... - y(n).
	H(y, x, w, ...) = G[ z(x, y), y, x, w, ... ]
\endverbatim

\tparam Vector_set
is the type used for vectors of sets. It can be either
sparse_pack or sparse_list.

\param i_z
variable index corresponding to the result for this operation;
i.e. the index in \a sparsity corresponding to z.

\param arg
\a arg[0]
is the number of addition variables in this cummulative summation; i.e.,
<tt>m + n</tt>.
\n
\a arg[1]
is the number of subtraction variables in this cummulative summation; i.e.,
\c m.
\n
<tt>parameter[ arg[2] ]</tt>
is the parameter value \c q in this cummunative summation.
\n
<tt>arg[2+i]</tt>
for <tt>i = 1 , ... , m</tt> is the value <tt>x(i)</tt>.
\n
<tt>arg[2+arg[1]+i]</tt>
for <tt>i = 1 , ... , n</tt> is the value <tt>y(i)</tt>.

\param sparsity
For <tt>i = 1 , ... , m</tt>,
the set with index \a arg[2+i] in \a sparsity
is the sparsity bit pattern for <tt>x(i)</tt>.
This identifies which of the dependent variables depend on <tt>x(i)</tt>.
On input, the sparsity patter corresponds to \c G,
and on ouput it corresponds to \c H.
\n
For <tt>i = 1 , ... , m</tt>,
the set with index \a arg[2+arg[0]+i] in \a sparsity
is the sparsity bit pattern for <tt>y(i)</tt>.
This identifies which of the dependent variables depend on <tt>y(i)</tt>.
On input, the sparsity patter corresponds to \c G,
and on ouput it corresponds to \c H.
\n
\b Input:
The set with index \a i_z in \a sparsity
is the sparsity bit pattern for z.
On input it corresponds to \c G and on output it is undefined.
*/

template <class Vector_set>
inline void reverse_sparse_jacobian_csum_op(
	size_t           i_z         ,
	const addr_t*    arg         ,
	Vector_set&      sparsity    )
{
	size_t i, j;
	i = arg[0] + arg[1];
	j = 2;
	while(i--)
	{	++j;
		CPPAD_ASSERT_UNKNOWN( size_t(arg[j]) < i_z );
		sparsity.binary_union(
			arg[j]     , // index in sparsity for result
			arg[j]     , // index in sparsity for left operand
			i_z        , // index for right operand
			sparsity     // sparsity vector for right operand
		);
	}
}
/*!
Reverse mode Hessian sparsity pattern for CSumOp operator.

This operation is
\verbatim
	z = q + x(1) + ... + x(m) - y(1) - ... - y(n).
	H(y, x, w, ...) = G[ z(x, y), y, x, w, ... ]
\endverbatim

\tparam Vector_set
is the type used for vectors of sets. It can be either
sparse_pack or sparse_list.

\param i_z
variable index corresponding to the result for this operation;
i.e. the index in \a sparsity corresponding to z.

\param arg
\a arg[0]
is the number of addition variables in this cummulative summation; i.e.,
<tt>m + n</tt>.
\n
\a arg[1]
is the number of subtraction variables in this cummulative summation; i.e.,
\c m.
\n
<tt>parameter[ arg[2] ]</tt>
is the parameter value \c q in this cummunative summation.
\n
<tt>arg[2+i]</tt>
for <tt>i = 1 , ... , m</tt> is the value <tt>x(i)</tt>.
\n
<tt>arg[2+arg[0]+i]</tt>
for <tt>i = 1 , ... , n</tt> is the value <tt>y(i)</tt>.

\param rev_jacobian
<tt>rev_jacobian[i_z]</tt>
is all false (true) if the Jabobian of G with respect to z must be zero
(may be non-zero).
\n
\n
For <tt>i = 1 , ... , m</tt>
<tt>rev_jacobian[ arg[2+i] ]</tt>
is all false (true) if the Jacobian with respect to <tt>x(i)</tt>
is zero (may be non-zero).
On input, it corresponds to the function G,
and on output it corresponds to the function H.
\n
\n
For <tt>i = 1 , ... , n</tt>
<tt>rev_jacobian[ arg[2+arg[0]+i] ]</tt>
is all false (true) if the Jacobian with respect to <tt>y(i)</tt>
is zero (may be non-zero).
On input, it corresponds to the function G,
and on output it corresponds to the function H.

\param rev_hes_sparsity
The set with index \a i_z in in \a rev_hes_sparsity
is the Hessian sparsity pattern for the fucntion G
where one of the partials derivative is with respect to z.
\n
\n
For <tt>i = 1 , ... , m</tt>
The set with index <tt>arg[2+i]</tt> in \a rev_hes_sparsity
is the Hessian sparsity pattern
where one of the partials derivative is with respect to <tt>x(i)</tt>.
On input, it corresponds to the function G,
and on output it corresponds to the function H.
\n
\n
For <tt>i = 1 , ... , n</tt>
The set with index <tt>arg[2+arg[0]+i]</tt> in \a rev_hes_sparsity
is the Hessian sparsity pattern
where one of the partials derivative is with respect to <tt>y(i)</tt>.
On input, it corresponds to the function G,
and on output it corresponds to the function H.
*/

template <class Vector_set>
inline void reverse_sparse_hessian_csum_op(
	size_t           i_z                 ,
	const addr_t*    arg                 ,
	bool*            rev_jacobian        ,
	Vector_set&      rev_hes_sparsity    )
{
	size_t i, j;
	i = arg[0] + arg[1];
	j = 2;
	while(i--)
	{	++j;
		CPPAD_ASSERT_UNKNOWN( size_t(arg[j]) < i_z );
		rev_hes_sparsity.binary_union(
		arg[j]             , // index in sparsity for result
		arg[j]             , // index in sparsity for left operand
		i_z                , // index for right operand
		rev_hes_sparsity     // sparsity vector for right operand
		);
		rev_jacobian[arg[j]] |= rev_jacobian[i_z];
	}
}

} } // END_CPPAD_LOCAL_NAMESPACE
# endif
