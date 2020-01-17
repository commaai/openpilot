// $Id: sparse_unary_op.hpp 3865 2017-01-19 01:57:55Z bradbell $
# ifndef CPPAD_LOCAL_SPARSE_UNARY_OP_HPP
# define CPPAD_LOCAL_SPARSE_UNARY_OP_HPP
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
\file sparse_unary_op.hpp
Forward and reverse mode sparsity patterns for unary operators.
*/


/*!
Forward mode Jacobian sparsity pattern for all unary operators.

The C++ source code corresponding to a unary operation has the form
\verbatim
	z = fun(x)
\endverbatim
where fun is a C++ unary function, or it has the form
\verbatim
	z = x op q
\endverbatim
where op is a C++ binary unary operator and q is a parameter.

\tparam Vector_set
is the type used for vectors of sets. It can be either
sparse_pack or sparse_list.

\param i_z
variable index corresponding to the result for this operation;
i.e., z.

\param i_x
variable index corresponding to the argument for this operator;
i.e., x.


\param sparsity
\b Input: The set with index \a arg[0] in \a sparsity
is the sparsity bit pattern for x.
This identifies which of the independent variables the variable x
depends on.
\n
\n
\b Output: The set with index \a i_z in \a sparsity
is the sparsity bit pattern for z.
This identifies which of the independent variables the variable z
depends on.
\n

\par Checked Assertions:
\li \a i_x < \a i_z
*/

template <class Vector_set>
inline void forward_sparse_jacobian_unary_op(
	size_t            i_z           ,
	size_t            i_x           ,
	Vector_set&       sparsity      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( i_x < i_z );

	sparsity.assignment(i_z, i_x, sparsity);
}
/*!
Reverse mode Jacobian sparsity pattern for all unary operators.

The C++ source code corresponding to a unary operation has the form
\verbatim
	z = fun(x)
\endverbatim
where fun is a C++ unary function, or it has the form
\verbatim
	z = x op q
\endverbatim
where op is a C++ bianry operator and q is a parameter.

This routine is given the sparsity patterns
for a function G(z, y, ... )
and it uses them to compute the sparsity patterns for
\verbatim
	H( x , w , u , ... ) = G[ z(x) , x , w , u , ... ]
\endverbatim

\tparam Vector_set
is the type used for vectors of sets. It can be either
sparse_pack or sparse_list.


\param i_z
variable index corresponding to the result for this operation;
i.e. the row index in sparsity corresponding to z.

\param i_x
variable index corresponding to the argument for this operator;
i.e. the row index in sparsity corresponding to x.

\param sparsity
\b Input:
The set with index \a i_z in \a sparsity
is the sparsity bit pattern for G with respect to the variable z.
\n
\b Input:
The set with index \a i_x in \a sparsity
is the sparsity bit pattern for G with respect to the variable x.
\n
\b Output:
The set with index \a i_x in \a sparsity
is the sparsity bit pattern for H with respect to the variable x.

\par Checked Assertions:
\li \a i_x < \a i_z
*/

template <class Vector_set>
inline void reverse_sparse_jacobian_unary_op(
	size_t     i_z                     ,
	size_t     i_x                     ,
	Vector_set&            sparsity    )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( i_x < i_z );

	sparsity.binary_union(i_x, i_x, i_z, sparsity);

	return;
}
// ---------------------------------------------------------------------------
/*!
Reverse mode Hessian sparsity pattern for linear unary operators.

The C++ source code corresponding to this operation is
\verbatim
        z = fun(x)
\endverbatim
where fun is a linear functions; e.g. abs, or
\verbatim
	z = x op q
\endverbatim
where op is a C++ binary operator and q is a parameter.

\copydetails CppAD::local::reverse_sparse_hessian_unary_op
*/
template <class Vector_set>
inline void reverse_sparse_hessian_linear_unary_op(
	size_t              i_z               ,
	size_t              i_x               ,
	bool*               rev_jacobian      ,
	const Vector_set&   for_jac_sparsity  ,
	Vector_set&         rev_hes_sparsity  )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( i_x < i_z );

	rev_hes_sparsity.binary_union(i_x, i_x, i_z, rev_hes_sparsity);

	rev_jacobian[i_x] |= rev_jacobian[i_z];
	return;
}

/*!
Reverse mode Hessian sparsity pattern for non-linear unary operators.

The C++ source code corresponding to this operation is
\verbatim
        z = fun(x)
\endverbatim
where fun is a non-linear functions; e.g. sin. or
\verbatim
	z = q / x
\endverbatim
where q is a parameter.


\copydetails CppAD::local::reverse_sparse_hessian_unary_op
*/
template <class Vector_set>
inline void reverse_sparse_hessian_nonlinear_unary_op(
	size_t              i_z               ,
	size_t              i_x               ,
	bool*               rev_jacobian      ,
	const Vector_set&   for_jac_sparsity  ,
	Vector_set&         rev_hes_sparsity  )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( i_x < i_z );

	rev_hes_sparsity.binary_union(i_x, i_x, i_z, rev_hes_sparsity);
	if( rev_jacobian[i_z] )
		rev_hes_sparsity.binary_union(i_x, i_x, i_x, for_jac_sparsity);

	rev_jacobian[i_x] |= rev_jacobian[i_z];
	return;
}

// ---------------------------------------------------------------------------
/*!
Forward mode Hessian sparsity pattern for non-linear unary operators.

The C++ source code corresponding to this operation is
\verbatim
        w(x) = fun( v(x) )
\endverbatim
where fun is a non-linear function.

\param i_v
is the index of the argument variable v

\param for_jac_sparsity
for_jac_sparsity(i_v) constains the Jacobian sparsity for v(x).

\param for_hes_sparsity
On input, for_hes_sparsity includes the Hessian sparsity for v(x); i.e.,
the sparsity can be a super set.
Upon return it includes the Hessian sparsity for  w(x)
*/
template <class Vector_set>
inline void forward_sparse_hessian_nonlinear_unary_op(
	size_t              i_v               ,
	const Vector_set&   for_jac_sparsity  ,
	Vector_set&         for_hes_sparsity  )
{
	// set of independent variables that v depends on
	typename Vector_set::const_iterator itr(for_jac_sparsity, i_v);

	// next independent variables that v depends on
	size_t i_x = *itr;

	// loop over dependent variables with non-zero partial
	while( i_x < for_jac_sparsity.end() )
	{	// N(i_x) = N(i_x) union L(i_v)
		for_hes_sparsity.binary_union(i_x, i_x, i_v, for_jac_sparsity);
		i_x = *(++itr);
	}
	return;
}

} } // END_CPPAD_LOCAL_NAMESPACE
# endif
