# ifndef CPPAD_LOCAL_DISCRETE_OP_HPP
# define CPPAD_LOCAL_DISCRETE_OP_HPP

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
\file discrete_op.hpp
Forward mode for z = f(x) where f is piecewise constant.
*/


/*!
forward mode Taylor coefficient for result of op = DisOp.

The C++ source code corresponding to this operation is
\verbatim
	z = f(x)
\endverbatim
where f is a piecewise constant function (and it's derivative is always
calculated as zero).

\tparam Base
base type for the operator; i.e., this operation was recorded
using AD< \a Base > and computations by this routine are done using type
\a Base .

\param p
is the lowest order Taylor coefficient that will be calculated.

\param q
is the highest order Taylor coefficient that will be calculated.

\param r
is the number of directions, for each order,
that will be calculated (except for order zero wich only has one direction).

\param i_z
variable index corresponding to the result for this operation;
i.e. the row index in \a taylor corresponding to z.

\param arg
\a arg[0]
\n
is the index, in the order of the discrete functions defined by the user,
for this discrete function.
\n
\n
\a arg[1]
variable index corresponding to the argument for this operator;
i.e. the row index in \a taylor corresponding to x.

\param cap_order
maximum number of orders that will fit in the taylor array.

\par tpv
We use the notation
<code>tpv = (cap_order-1) * r + 1</code>
which is the number of Taylor coefficients per variable

\param taylor
\b Input: <code>taylor [ arg[1] * tpv + 0 ]</code>
is the zero order Taylor coefficient corresponding to x.
\n
\b Output: if <code>p == 0</code>
<code>taylor [ i_z * tpv + 0 ]</code>
is the zero order Taylor coefficient corresponding to z.
For k = max(p, 1), ... , q,
<code>taylor [ i_z * tpv + (k-1)*r + 1 + ell ]</code>
is the k-th order Taylor coefficient corresponding to z
(which is zero).

\par Checked Assertions where op is the unary operator with one result:
\li NumArg(op) == 2
\li NumRes(op) == 1
\li q < cap_order
\li 0 < r
*/
template <class Base>
inline void forward_dis_op(
	size_t        p           ,
	size_t        q           ,
	size_t        r           ,
	size_t        i_z         ,
	const addr_t* arg         ,
	size_t        cap_order   ,
	Base*         taylor      )
{
	// check assumptions
	CPPAD_ASSERT_UNKNOWN( NumArg(DisOp) == 2 );
	CPPAD_ASSERT_UNKNOWN( NumRes(DisOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );
	CPPAD_ASSERT_UNKNOWN( 0 < r );

	// Taylor coefficients corresponding to argument and result
	size_t num_taylor_per_var = (cap_order-1) * r + 1;
	Base* x = taylor + arg[1] * num_taylor_per_var;
	Base* z = taylor +    i_z * num_taylor_per_var;

	if( p == 0 )
	{	z[0]  = discrete<Base>::eval(arg[0], x[0]);
		p++;
	}
	for(size_t ell = 0; ell < r; ell++)
		for(size_t k = p; k <= q; k++)
			z[ (k-1) * r + 1 + ell ] = Base(0.0);
}


} } // END_CPPAD_LOCAL_NAMESPACE
# endif
