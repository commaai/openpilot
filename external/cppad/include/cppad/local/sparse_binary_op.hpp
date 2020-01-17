// $Id: sparse_binary_op.hpp 3865 2017-01-19 01:57:55Z bradbell $
# ifndef CPPAD_LOCAL_SPARSE_BINARY_OP_HPP
# define CPPAD_LOCAL_SPARSE_BINARY_OP_HPP
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
\file sparse_binary_op.hpp
Forward and reverse mode sparsity patterns for binary operators.
*/


/*!
Forward mode Jacobian sparsity pattern for all binary operators.

The C++ source code corresponding to a binary operation has the form
\verbatim
	z = fun(x, y)
\endverbatim
where fun is a C++ binary function and both x and y are variables,
or it has the form
\verbatim
	z = x op y
\endverbatim
where op is a C++ binary unary operator and both x and y are variables.

\tparam Vector_set
is the type used for vectors of sets. It can be either
sparse_pack or sparse_list.

\param i_z
variable index corresponding to the result for this operation;
i.e., z.

\param arg
\a arg[0]
variable index corresponding to the left operand for this operator;
i.e., x.
\n
\n arg[1]
variable index corresponding to the right operand for this operator;
i.e., y.

\param sparsity
\b Input:
The set with index \a arg[0] in \a sparsity
is the sparsity bit pattern for x.
This identifies which of the independent variables the variable x
depends on.
\n
\n
\b Input:
The set with index \a arg[1] in \a sparsity
is the sparsity bit pattern for y.
This identifies which of the independent variables the variable y
depends on.
\n
\n
\b Output:
The set with index \a i_z in \a sparsity
is the sparsity bit pattern for z.
This identifies which of the independent variables the variable z
depends on.

\par Checked Assertions:
\li \a arg[0] < \a i_z
\li \a arg[1] < \a i_z
*/

template <class Vector_set>
inline void forward_sparse_jacobian_binary_op(
	size_t            i_z           ,
	const addr_t*     arg           ,
	Vector_set&       sparsity      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < i_z );
	CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < i_z );

	sparsity.binary_union(i_z, arg[0], arg[1], sparsity);

	return;
}

/*!
Reverse mode Jacobian sparsity pattern for all binary operators.

The C++ source code corresponding to a unary operation has the form
\verbatim
	z = fun(x, y)
\endverbatim
where fun is a C++ unary function and x and y are variables,
or it has the form
\verbatim
	z = x op y
\endverbatim
where op is a C++ bianry operator and x and y are variables.

This routine is given the sparsity patterns
for a function G(z, y, x, ... )
and it uses them to compute the sparsity patterns for
\verbatim
	H( y, x, w , u , ... ) = G[ z(x,y) , y , x , w , u , ... ]
\endverbatim

\tparam Vector_set
is the type used for vectors of sets. It can be either
sparse_pack or sparse_list.

\param i_z
variable index corresponding to the result for this operation;
i.e., z.

\param arg
\a arg[0]
variable index corresponding to the left operand for this operator;
i.e., x.

\n
\n arg[1]
variable index corresponding to the right operand for this operator;
i.e., y.

\param sparsity
The set with index \a i_z in \a sparsity
is the sparsity pattern for z corresponding ot the function G.
\n
\n
The set with index \a arg[0] in \a sparsity
is the sparsity pattern for x.
On input, it corresponds to the function G,
and on output it corresponds to H.
\n
\n
The set with index \a arg[1] in \a sparsity
is the sparsity pattern for y.
On input, it corresponds to the function G,
and on output it corresponds to H.
\n
\n

\par Checked Assertions:
\li \a arg[0] < \a i_z
\li \a arg[1] < \a i_z
*/
template <class Vector_set>
inline void reverse_sparse_jacobian_binary_op(
	size_t              i_z           ,
	const addr_t*       arg           ,
	Vector_set&         sparsity      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < i_z );
	CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < i_z );

	sparsity.binary_union(arg[0], arg[0], i_z, sparsity);
	sparsity.binary_union(arg[1], arg[1], i_z, sparsity);

	return;
}
// ---------------------------------------------------------------------------
/*!
Reverse mode Hessian sparsity pattern for add and subtract operators.

The C++ source code corresponding to a unary operation has the form
\verbatim
	z = x op y
\endverbatim
where op is + or - and x, y are variables.

\copydetails CppAD::local::reverse_sparse_hessian_binary_op
*/
template <class Vector_set>
inline void reverse_sparse_hessian_addsub_op(
	size_t               i_z                ,
	const addr_t*        arg                ,
	bool*                jac_reverse        ,
	const Vector_set&    for_jac_sparsity   ,
	Vector_set&          rev_hes_sparsity   )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < i_z );
	CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < i_z );

	rev_hes_sparsity.binary_union(arg[0], arg[0], i_z, rev_hes_sparsity);
	rev_hes_sparsity.binary_union(arg[1], arg[1], i_z, rev_hes_sparsity);

	jac_reverse[arg[0]] |= jac_reverse[i_z];
	jac_reverse[arg[1]] |= jac_reverse[i_z];

	return;
}

/*!
Reverse mode Hessian sparsity pattern for multiplication operator.

The C++ source code corresponding to a unary operation has the form
\verbatim
	z = x * y
\endverbatim
where x and y are variables.

\copydetails CppAD::local::reverse_sparse_hessian_binary_op
*/
template <class Vector_set>
inline void reverse_sparse_hessian_mul_op(
	size_t               i_z                ,
	const addr_t*        arg                ,
	bool*                jac_reverse        ,
	const Vector_set&    for_jac_sparsity   ,
	Vector_set&          rev_hes_sparsity   )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < i_z );
	CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < i_z );

	rev_hes_sparsity.binary_union(arg[0], arg[0], i_z, rev_hes_sparsity);
	rev_hes_sparsity.binary_union(arg[1], arg[1], i_z, rev_hes_sparsity);

	if( jac_reverse[i_z] )
	{	rev_hes_sparsity.binary_union(
			arg[0], arg[0], arg[1], for_jac_sparsity);
		rev_hes_sparsity.binary_union(
			arg[1], arg[1], arg[0], for_jac_sparsity);
	}

	jac_reverse[arg[0]] |= jac_reverse[i_z];
	jac_reverse[arg[1]] |= jac_reverse[i_z];
	return;
}

/*!
Reverse mode Hessian sparsity pattern for division operator.

The C++ source code corresponding to a unary operation has the form
\verbatim
	z = x / y
\endverbatim
where x and y are variables.

\copydetails CppAD::local::reverse_sparse_hessian_binary_op
*/
template <class Vector_set>
inline void reverse_sparse_hessian_div_op(
	size_t               i_z                ,
	const addr_t*        arg                ,
	bool*                jac_reverse        ,
	const Vector_set&    for_jac_sparsity   ,
	Vector_set&          rev_hes_sparsity   )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < i_z );
	CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < i_z );

	rev_hes_sparsity.binary_union(arg[0], arg[0], i_z, rev_hes_sparsity);
	rev_hes_sparsity.binary_union(arg[1], arg[1], i_z, rev_hes_sparsity);

	if( jac_reverse[i_z] )
	{	rev_hes_sparsity.binary_union(
			arg[0], arg[0], arg[1], for_jac_sparsity);
		rev_hes_sparsity.binary_union(
			arg[1], arg[1], arg[0], for_jac_sparsity);
		rev_hes_sparsity.binary_union(
			arg[1], arg[1], arg[1], for_jac_sparsity);
	}

	jac_reverse[arg[0]] |= jac_reverse[i_z];
	jac_reverse[arg[1]] |= jac_reverse[i_z];
	return;
}

/*!
Reverse mode Hessian sparsity pattern for power function.

The C++ source code corresponding to a unary operation has the form
\verbatim
	z = pow(x, y)
\endverbatim
where x and y are variables.

\copydetails CppAD::local::reverse_sparse_hessian_binary_op
*/
template <class Vector_set>
inline void reverse_sparse_hessian_pow_op(
	size_t               i_z                ,
	const addr_t*        arg                ,
	bool*                jac_reverse        ,
	const Vector_set&    for_jac_sparsity   ,
	Vector_set&          rev_hes_sparsity   )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < i_z );
	CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < i_z );

	rev_hes_sparsity.binary_union(arg[0], arg[0], i_z, rev_hes_sparsity);
	rev_hes_sparsity.binary_union(arg[1], arg[1], i_z, rev_hes_sparsity);

	if( jac_reverse[i_z] )
	{
		rev_hes_sparsity.binary_union(
			arg[0], arg[0], arg[0], for_jac_sparsity);
		rev_hes_sparsity.binary_union(
			arg[0], arg[0], arg[1], for_jac_sparsity);

		rev_hes_sparsity.binary_union(
			arg[1], arg[1], arg[0], for_jac_sparsity);
		rev_hes_sparsity.binary_union(
			arg[1], arg[1], arg[1], for_jac_sparsity);
	}

	// I cannot think of a case where this is necessary, but it including
	// it makes it like the other cases.
	jac_reverse[arg[0]] |= jac_reverse[i_z];
	jac_reverse[arg[1]] |= jac_reverse[i_z];
	return;
}
// ---------------------------------------------------------------------------
/*!
Forward mode Hessian sparsity pattern for multiplication operator.

The C++ source code corresponding to this operation is
\verbatim
        w(x) = v0(x) * v1(x)
\endverbatim

\param arg
is the index of the argument vector for the multiplication operation; i.e.,
arg[0], arg[1] are the left and right operands.

\param for_jac_sparsity
for_jac_sparsity(arg[0]) constains the Jacobian sparsity for v0(x),
for_jac_sparsity(arg[1]) constains the Jacobian sparsity for v1(x).

\param for_hes_sparsity
On input, for_hes_sparsity includes the Hessian sparsity for v0(x)
and v1(x); i.e., the sparsity can be a super set.
Upon return it includes the Hessian sparsity for  w(x)
*/
template <class Vector_set>
inline void forward_sparse_hessian_mul_op(
	const addr_t*       arg               ,
	const Vector_set&   for_jac_sparsity  ,
	Vector_set&         for_hes_sparsity  )
{	// --------------------------------------------------
	// set of independent variables that v0 depends on
	typename Vector_set::const_iterator itr_0(for_jac_sparsity, arg[0]);

	// loop over dependent variables with non-zero partial
	size_t i_x = *itr_0;
	while( i_x < for_jac_sparsity.end() )
	{	// N(i_x) = N(i_x) union L(v1)
		for_hes_sparsity.binary_union(i_x, i_x, arg[1], for_jac_sparsity);
		i_x = *(++itr_0);
	}
	// --------------------------------------------------
	// set of independent variables that v1 depends on
	typename Vector_set::const_iterator itr_1(for_jac_sparsity, arg[1]);

	// loop over dependent variables with non-zero partial
	i_x = *itr_1;
	while( i_x < for_jac_sparsity.end() )
	{	// N(i_x) = N(i_x) union L(v0)
		for_hes_sparsity.binary_union(i_x, i_x, arg[0], for_jac_sparsity);
		i_x = *(++itr_1);
	}
	return;
}
/*!
Forward mode Hessian sparsity pattern for division operator.

The C++ source code corresponding to this operation is
\verbatim
        w(x) = v0(x) / v1(x)
\endverbatim

\param arg
is the index of the argument vector for the division operation; i.e.,
arg[0], arg[1] are the left and right operands.

\param for_jac_sparsity
for_jac_sparsity(arg[0]) constains the Jacobian sparsity for v0(x),
for_jac_sparsity(arg[1]) constains the Jacobian sparsity for v1(x).

\param for_hes_sparsity
On input, for_hes_sparsity includes the Hessian sparsity for v0(x)
and v1(x); i.e., the sparsity can be a super set.
Upon return it includes the Hessian sparsity for  w(x)
*/
template <class Vector_set>
inline void forward_sparse_hessian_div_op(
	const addr_t*       arg               ,
	const Vector_set&   for_jac_sparsity  ,
	Vector_set&         for_hes_sparsity  )
{	// --------------------------------------------------
	// set of independent variables that v0 depends on
	typename Vector_set::const_iterator itr_0(for_jac_sparsity, arg[0]);

	// loop over dependent variables with non-zero partial
	size_t i_x = *itr_0;
	while( i_x < for_jac_sparsity.end() )
	{	// N(i_x) = N(i_x) union L(v1)
		for_hes_sparsity.binary_union(i_x, i_x, arg[1], for_jac_sparsity);
		i_x = *(++itr_0);
	}
	// --------------------------------------------------
	// set of independent variables that v1 depends on
	typename Vector_set::const_iterator itr_1(for_jac_sparsity, arg[1]);

	// loop over dependent variables with non-zero partial
	i_x = *itr_1;
	while( i_x < for_jac_sparsity.end() )
	{	// N(i_x) = N(i_x) union L(v0)
		for_hes_sparsity.binary_union(i_x, i_x, arg[0], for_jac_sparsity);
		// N(i_x) = N(i_x) union L(v1)
		for_hes_sparsity.binary_union(i_x, i_x, arg[1], for_jac_sparsity);
		i_x = *(++itr_1);
	}
	return;
}
/*!
Forward mode Hessian sparsity pattern for power operator.

The C++ source code corresponding to this operation is
\verbatim
        w(x) = pow( v0(x) , v1(x) )
\endverbatim

\param arg
is the index of the argument vector for the power operation; i.e.,
arg[0], arg[1] are the left and right operands.

\param for_jac_sparsity
for_jac_sparsity(arg[0]) constains the Jacobian sparsity for v0(x),
for_jac_sparsity(arg[1]) constains the Jacobian sparsity for v1(x).

\param for_hes_sparsity
On input, for_hes_sparsity includes the Hessian sparsity for v0(x)
and v1(x); i.e., the sparsity can be a super set.
Upon return it includes the Hessian sparsity for  w(x)
*/
template <class Vector_set>
inline void forward_sparse_hessian_pow_op(
	const addr_t*       arg               ,
	const Vector_set&   for_jac_sparsity  ,
	Vector_set&         for_hes_sparsity  )
{	// --------------------------------------------------
	// set of independent variables that v0 depends on
	typename Vector_set::const_iterator itr_0(for_jac_sparsity, arg[0]);

	// loop over dependent variables with non-zero partial
	size_t i_x = *itr_0;
	while( i_x < for_jac_sparsity.end() )
	{	// N(i_x) = N(i_x) union L(v0)
		for_hes_sparsity.binary_union(i_x, i_x, arg[0], for_jac_sparsity);
		// N(i_x) = N(i_x) union L(v1)
		for_hes_sparsity.binary_union(i_x, i_x, arg[1], for_jac_sparsity);
		i_x = *(++itr_0);
	}
	// --------------------------------------------------
	// set of independent variables that v1 depends on
	typename Vector_set::const_iterator itr_1(for_jac_sparsity, arg[1]);

	// loop over dependent variables with non-zero partial
	i_x = *itr_1;
	while( i_x < for_jac_sparsity.end() )
	{	// N(i_x) = N(i_x) union L(v0)
		for_hes_sparsity.binary_union(i_x, i_x, arg[0], for_jac_sparsity);
		// N(i_x) = N(i_x) union L(v1)
		for_hes_sparsity.binary_union(i_x, i_x, arg[1], for_jac_sparsity);
		i_x = *(++itr_1);
	}
	return;
}
// ---------------------------------------------------------------------------
} } // END_CPPAD_LOCAL_NAMESPACE
# endif
