// $Id: cond_op.hpp 3845 2016-11-19 01:50:47Z bradbell $
# ifndef CPPAD_LOCAL_COND_OP_HPP
# define CPPAD_LOCAL_COND_OP_HPP
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
\file cond_op.hpp
Forward, reverse, and sparse operations for conditional expressions.
*/

/*!
Shared documentation for conditional expressions (not called).

<!-- define conditional_exp_op -->
The C++ source code coresponding to this operation is
\verbatim
	z = CondExpRel(y_0, y_1, y_2, y_3)
\endverbatim
where Rel is one of the following: Lt, Le, Eq, Ge, Gt.

\tparam Base
base type for the operator; i.e., this operation was recorded
using AD< \a Base > and computations by this routine are done using type
\a Base.

\param i_z
is the AD variable index corresponding to the variable z.

\param arg
\n
\a arg[0]
is static cast to size_t from the enum type
\verbatim
	enum CompareOp {
		CompareLt,
		CompareLe,
		CompareEq,
		CompareGe,
		CompareGt,
		CompareNe
	}
\endverbatim
for this operation.
Note that arg[0] cannot be equal to CompareNe.
\n
\n
\a arg[1] & 1
\n
If this is zero, y_0 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 2
\n
If this is zero, y_1 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 4
\n
If this is zero, y_2 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 8
\n
If this is zero, y_3 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[2 + j ] for j = 0, 1, 2, 3
\n
is the index corresponding to y_j.

\param num_par
is the total number of values in the vector \a parameter.

\param parameter
For j = 0, 1, 2, 3,
if y_j is a parameter, \a parameter [ arg[2 + j] ] is its value.

\param cap_order
number of columns in the matrix containing the Taylor coefficients.

\par Checked Assertions
\li NumArg(CExpOp) == 6
\li NumRes(CExpOp) == 1
\li arg[0] < static_cast<size_t> ( CompareNe )
\li arg[1] != 0; i.e., not all of y_0, y_1, y_2, y_3 are parameters.
\li For j = 0, 1, 2, 3 if y_j is a parameter, arg[2+j] < num_par.
<!-- end conditional_exp_op -->
*/
template <class Base>
inline void conditional_exp_op(
	size_t         i_z         ,
	const addr_t*  arg         ,
	size_t         num_par     ,
	const Base*    parameter   ,
	size_t         cap_order   )
{	// This routine is only for documentation, it should never be used
	CPPAD_ASSERT_UNKNOWN( false );
}

/*!
Shared documentation for conditional expression sparse operations (not called).

<!-- define sparse_conditional_exp_op -->
The C++ source code coresponding to this operation is
\verbatim
	z = CondExpRel(y_0, y_1, y_2, y_3)
\endverbatim
where Rel is one of the following: Lt, Le, Eq, Ge, Gt.

\tparam Vector_set
is the type used for vectors of sets. It can be either
sparse_pack or sparse_list.

\param i_z
is the AD variable index corresponding to the variable z.

\param arg
\n
\a arg[0]
is static cast to size_t from the enum type
\verbatim
	enum CompareOp {
		CompareLt,
		CompareLe,
		CompareEq,
		CompareGe,
		CompareGt,
		CompareNe
	}
\endverbatim
for this operation.
Note that arg[0] cannot be equal to CompareNe.
\n
\n
\a arg[1] & 1
\n
If this is zero, y_0 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 2
\n
If this is zero, y_1 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 4
\n
If this is zero, y_2 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 8
\n
If this is zero, y_3 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[2 + j ] for j = 0, 1, 2, 3
\n
is the index corresponding to y_j.

\param num_par
is the total number of values in the vector \a parameter.

\par Checked Assertions
\li NumArg(CExpOp) == 6
\li NumRes(CExpOp) == 1
\li arg[0] < static_cast<size_t> ( CompareNe )
\li arg[1] != 0; i.e., not all of y_0, y_1, y_2, y_3 are parameters.
\li For j = 0, 1, 2, 3 if y_j is a parameter, arg[2+j] < num_par.
<!-- end sparse_conditional_exp_op -->
*/
template <class Vector_set>
inline void sparse_conditional_exp_op(
	size_t         i_z           ,
	const addr_t*  arg           ,
	size_t         num_par       )
{	// This routine is only for documentation, it should never be used
	CPPAD_ASSERT_UNKNOWN( false );
}

/*!
Compute forward mode Taylor coefficients for op = CExpOp.

<!-- replace conditional_exp_op -->
The C++ source code coresponding to this operation is
\verbatim
	z = CondExpRel(y_0, y_1, y_2, y_3)
\endverbatim
where Rel is one of the following: Lt, Le, Eq, Ge, Gt.

\tparam Base
base type for the operator; i.e., this operation was recorded
using AD< \a Base > and computations by this routine are done using type
\a Base.

\param i_z
is the AD variable index corresponding to the variable z.

\param arg
\n
\a arg[0]
is static cast to size_t from the enum type
\verbatim
	enum CompareOp {
		CompareLt,
		CompareLe,
		CompareEq,
		CompareGe,
		CompareGt,
		CompareNe
	}
\endverbatim
for this operation.
Note that arg[0] cannot be equal to CompareNe.
\n
\n
\a arg[1] & 1
\n
If this is zero, y_0 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 2
\n
If this is zero, y_1 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 4
\n
If this is zero, y_2 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 8
\n
If this is zero, y_3 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[2 + j ] for j = 0, 1, 2, 3
\n
is the index corresponding to y_j.

\param num_par
is the total number of values in the vector \a parameter.

\param parameter
For j = 0, 1, 2, 3,
if y_j is a parameter, \a parameter [ arg[2 + j] ] is its value.

\param cap_order
number of columns in the matrix containing the Taylor coefficients.

\par Checked Assertions
\li NumArg(CExpOp) == 6
\li NumRes(CExpOp) == 1
\li arg[0] < static_cast<size_t> ( CompareNe )
\li arg[1] != 0; i.e., not all of y_0, y_1, y_2, y_3 are parameters.
\li For j = 0, 1, 2, 3 if y_j is a parameter, arg[2+j] < num_par.
<!-- end conditional_exp_op -->

\param p
is the lowest order of the Taylor coefficient of z that we are computing.

\param q
is the highest order of the Taylor coefficient of z that we are computing.

\param taylor
\b Input:
For j = 0, 1, 2, 3 and k = 0 , ... , q,
if y_j is a variable then
<code>taylor [ arg[2+j] * cap_order + k ]</code>
is the k-th order Taylor coefficient corresponding to y_j.
\n
\b Input: <code>taylor [ i_z * cap_order + k ]</code>
for k = 0 , ... , p-1,
is the k-th order Taylor coefficient corresponding to z.
\n
\b Output: <code>taylor [ i_z * cap_order + k ]</code>
for k = p , ... , q,
is the k-th order Taylor coefficient corresponding to z.

*/
template <class Base>
inline void forward_cond_op(
	size_t         p           ,
	size_t         q           ,
	size_t         i_z         ,
	const addr_t*  arg         ,
	size_t         num_par     ,
	const Base*    parameter   ,
	size_t         cap_order   ,
	Base*          taylor      )
{	Base y_0, y_1, y_2, y_3;
	Base zero(0);
	Base* z = taylor + i_z * cap_order;

	CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < static_cast<size_t> (CompareNe) );
	CPPAD_ASSERT_UNKNOWN( NumArg(CExpOp) == 6 );
	CPPAD_ASSERT_UNKNOWN( NumRes(CExpOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( arg[1] != 0 );

	if( arg[1] & 1 )
	{
		y_0 = taylor[ arg[2] * cap_order + 0 ];
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( size_t(arg[2]) < num_par );
		y_0 = parameter[ arg[2] ];
	}
	if( arg[1] & 2 )
	{
		y_1 = taylor[ arg[3] * cap_order + 0 ];
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( size_t(arg[3]) < num_par );
		y_1 = parameter[ arg[3] ];
	}
	if( p == 0 )
	{	if( arg[1] & 4 )
		{
			y_2 = taylor[ arg[4] * cap_order + 0 ];
		}
		else
		{	CPPAD_ASSERT_UNKNOWN( size_t(arg[4]) < num_par );
			y_2 = parameter[ arg[4] ];
		}
		if( arg[1] & 8 )
		{
			y_3 = taylor[ arg[5] * cap_order + 0 ];
		}
		else
		{	CPPAD_ASSERT_UNKNOWN( size_t(arg[5]) < num_par );
			y_3 = parameter[ arg[5] ];
		}
		z[0] = CondExpOp(
			CompareOp( arg[0] ),
			y_0,
			y_1,
			y_2,
			y_3
		);
		p++;
	}
	for(size_t d = p; d <= q; d++)
	{	if( arg[1] & 4 )
		{
			y_2 = taylor[ arg[4] * cap_order + d];
		}
		else	y_2 = zero;
		if( arg[1] & 8 )
		{
			y_3 = taylor[ arg[5] * cap_order + d];
		}
		else	y_3 = zero;
		z[d] = CondExpOp(
			CompareOp( arg[0] ),
			y_0,
			y_1,
			y_2,
			y_3
		);
	}
	return;
}

/*!
Multiple directions forward mode Taylor coefficients for op = CExpOp.

<!-- replace conditional_exp_op -->
The C++ source code coresponding to this operation is
\verbatim
	z = CondExpRel(y_0, y_1, y_2, y_3)
\endverbatim
where Rel is one of the following: Lt, Le, Eq, Ge, Gt.

\tparam Base
base type for the operator; i.e., this operation was recorded
using AD< \a Base > and computations by this routine are done using type
\a Base.

\param i_z
is the AD variable index corresponding to the variable z.

\param arg
\n
\a arg[0]
is static cast to size_t from the enum type
\verbatim
	enum CompareOp {
		CompareLt,
		CompareLe,
		CompareEq,
		CompareGe,
		CompareGt,
		CompareNe
	}
\endverbatim
for this operation.
Note that arg[0] cannot be equal to CompareNe.
\n
\n
\a arg[1] & 1
\n
If this is zero, y_0 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 2
\n
If this is zero, y_1 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 4
\n
If this is zero, y_2 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 8
\n
If this is zero, y_3 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[2 + j ] for j = 0, 1, 2, 3
\n
is the index corresponding to y_j.

\param num_par
is the total number of values in the vector \a parameter.

\param parameter
For j = 0, 1, 2, 3,
if y_j is a parameter, \a parameter [ arg[2 + j] ] is its value.

\param cap_order
number of columns in the matrix containing the Taylor coefficients.

\par Checked Assertions
\li NumArg(CExpOp) == 6
\li NumRes(CExpOp) == 1
\li arg[0] < static_cast<size_t> ( CompareNe )
\li arg[1] != 0; i.e., not all of y_0, y_1, y_2, y_3 are parameters.
\li For j = 0, 1, 2, 3 if y_j is a parameter, arg[2+j] < num_par.
<!-- end conditional_exp_op -->

\param q
is order of the Taylor coefficient of z that we are computing.

\param r
is the number of Taylor coefficient directions that we are computing.

\par tpv
We use the notation
<code>tpv = (cap_order-1) * r + 1</code>
which is the number of Taylor coefficients per variable

\param taylor
\b Input:
For j = 0, 1, 2, 3, k = 1, ..., q,
if y_j is a variable then
<code>taylor [ arg[2+j] * tpv + 0 ]</code>
is the zero order Taylor coefficient corresponding to y_j and
<code>taylor [ arg[2+j] * tpv + (k-1)*r+1+ell</code> is its
k-th order Taylor coefficient in the ell-th direction.
\n
\b Input:
For j = 0, 1, 2, 3, k = 1, ..., q-1,
<code>taylor [ i_z * tpv + 0 ]</code>
is the zero order Taylor coefficient corresponding to z and
<code>taylor [ i_z * tpv + (k-1)*r+1+ell</code> is its
k-th order Taylor coefficient in the ell-th direction.
\n
\b Output: <code>taylor [ i_z * tpv + (q-1)*r+1+ell ]</code>
is the q-th order Taylor coefficient corresponding to z
in the ell-th direction.
*/
template <class Base>
inline void forward_cond_op_dir(
	size_t         q           ,
	size_t         r           ,
	size_t         i_z         ,
	const addr_t*  arg         ,
	size_t         num_par     ,
	const Base*    parameter   ,
	size_t         cap_order   ,
	Base*          taylor      )
{	Base y_0, y_1, y_2, y_3;
	Base zero(0);
	size_t num_taylor_per_var = (cap_order-1) * r + 1;
	Base* z = taylor + i_z * num_taylor_per_var;

	CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < static_cast<size_t> (CompareNe) );
	CPPAD_ASSERT_UNKNOWN( NumArg(CExpOp) == 6 );
	CPPAD_ASSERT_UNKNOWN( NumRes(CExpOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( arg[1] != 0 );
	CPPAD_ASSERT_UNKNOWN( 0 < q );
	CPPAD_ASSERT_UNKNOWN( q < cap_order );

	if( arg[1] & 1 )
	{
		y_0 = taylor[ arg[2] * num_taylor_per_var + 0 ];
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( size_t(arg[2]) < num_par );
		y_0 = parameter[ arg[2] ];
	}
	if( arg[1] & 2 )
	{
		y_1 = taylor[ arg[3] * num_taylor_per_var + 0 ];
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( size_t(arg[3]) < num_par );
		y_1 = parameter[ arg[3] ];
	}
	size_t m = (q-1) * r + 1;
	for(size_t ell = 0; ell < r; ell++)
	{	if( arg[1] & 4 )
		{
			y_2 = taylor[ arg[4] * num_taylor_per_var + m + ell];
		}
		else	y_2 = zero;
		if( arg[1] & 8 )
		{
			y_3 = taylor[ arg[5] * num_taylor_per_var + m + ell];
		}
		else	y_3 = zero;
		z[m+ell] = CondExpOp(
			CompareOp( arg[0] ),
			y_0,
			y_1,
			y_2,
			y_3
		);
	}
	return;
}

/*!
Compute zero order forward mode Taylor coefficients for op = CExpOp.

<!-- replace conditional_exp_op -->
The C++ source code coresponding to this operation is
\verbatim
	z = CondExpRel(y_0, y_1, y_2, y_3)
\endverbatim
where Rel is one of the following: Lt, Le, Eq, Ge, Gt.

\tparam Base
base type for the operator; i.e., this operation was recorded
using AD< \a Base > and computations by this routine are done using type
\a Base.

\param i_z
is the AD variable index corresponding to the variable z.

\param arg
\n
\a arg[0]
is static cast to size_t from the enum type
\verbatim
	enum CompareOp {
		CompareLt,
		CompareLe,
		CompareEq,
		CompareGe,
		CompareGt,
		CompareNe
	}
\endverbatim
for this operation.
Note that arg[0] cannot be equal to CompareNe.
\n
\n
\a arg[1] & 1
\n
If this is zero, y_0 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 2
\n
If this is zero, y_1 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 4
\n
If this is zero, y_2 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 8
\n
If this is zero, y_3 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[2 + j ] for j = 0, 1, 2, 3
\n
is the index corresponding to y_j.

\param num_par
is the total number of values in the vector \a parameter.

\param parameter
For j = 0, 1, 2, 3,
if y_j is a parameter, \a parameter [ arg[2 + j] ] is its value.

\param cap_order
number of columns in the matrix containing the Taylor coefficients.

\par Checked Assertions
\li NumArg(CExpOp) == 6
\li NumRes(CExpOp) == 1
\li arg[0] < static_cast<size_t> ( CompareNe )
\li arg[1] != 0; i.e., not all of y_0, y_1, y_2, y_3 are parameters.
\li For j = 0, 1, 2, 3 if y_j is a parameter, arg[2+j] < num_par.
<!-- end conditional_exp_op -->

\param taylor
\b Input:
For j = 0, 1, 2, 3,
if y_j is a variable then
\a taylor [ \a arg[2+j] * cap_order + 0 ]
is the zero order Taylor coefficient corresponding to y_j.
\n
\b Output: \a taylor [ \a i_z * \a cap_order + 0 ]
is the zero order Taylor coefficient corresponding to z.
*/
template <class Base>
inline void forward_cond_op_0(
	size_t         i_z         ,
	const addr_t*  arg         ,
	size_t         num_par     ,
	const Base*    parameter   ,
	size_t         cap_order   ,
	Base*          taylor      )
{	Base y_0, y_1, y_2, y_3;
	Base* z;

	CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < static_cast<size_t> (CompareNe) );
	CPPAD_ASSERT_UNKNOWN( NumArg(CExpOp) == 6 );
	CPPAD_ASSERT_UNKNOWN( NumRes(CExpOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( arg[1] != 0 );

	if( arg[1] & 1 )
	{
		y_0 = taylor[ arg[2] * cap_order + 0 ];
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( size_t(arg[2]) < num_par );
		y_0 = parameter[ arg[2] ];
	}
	if( arg[1] & 2 )
	{
		y_1 = taylor[ arg[3] * cap_order + 0 ];
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( size_t(arg[3]) < num_par );
		y_1 = parameter[ arg[3] ];
	}
	if( arg[1] & 4 )
	{
		y_2 = taylor[ arg[4] * cap_order + 0 ];
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( size_t(arg[4]) < num_par );
		y_2 = parameter[ arg[4] ];
	}
	if( arg[1] & 8 )
	{
		y_3 = taylor[ arg[5] * cap_order + 0 ];
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( size_t(arg[5]) < num_par );
		y_3 = parameter[ arg[5] ];
	}
	z = taylor + i_z * cap_order;
	z[0] = CondExpOp(
		CompareOp( arg[0] ),
		y_0,
		y_1,
		y_2,
		y_3
	);
	return;
}

/*!
Compute reverse mode Taylor coefficients for op = CExpOp.

This routine is given the partial derivatives of a function
G( z , y , x , w , ... )
and it uses them to compute the partial derivatives of
\verbatim
	H( y , x , w , u , ... ) = G[ z(y) , y , x , w , u , ... ]
\endverbatim
where y above represents y_0, y_1, y_2, y_3.

<!-- replace conditional_exp_op -->
The C++ source code coresponding to this operation is
\verbatim
	z = CondExpRel(y_0, y_1, y_2, y_3)
\endverbatim
where Rel is one of the following: Lt, Le, Eq, Ge, Gt.

\tparam Base
base type for the operator; i.e., this operation was recorded
using AD< \a Base > and computations by this routine are done using type
\a Base.

\param i_z
is the AD variable index corresponding to the variable z.

\param arg
\n
\a arg[0]
is static cast to size_t from the enum type
\verbatim
	enum CompareOp {
		CompareLt,
		CompareLe,
		CompareEq,
		CompareGe,
		CompareGt,
		CompareNe
	}
\endverbatim
for this operation.
Note that arg[0] cannot be equal to CompareNe.
\n
\n
\a arg[1] & 1
\n
If this is zero, y_0 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 2
\n
If this is zero, y_1 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 4
\n
If this is zero, y_2 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 8
\n
If this is zero, y_3 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[2 + j ] for j = 0, 1, 2, 3
\n
is the index corresponding to y_j.

\param num_par
is the total number of values in the vector \a parameter.

\param parameter
For j = 0, 1, 2, 3,
if y_j is a parameter, \a parameter [ arg[2 + j] ] is its value.

\param cap_order
number of columns in the matrix containing the Taylor coefficients.

\par Checked Assertions
\li NumArg(CExpOp) == 6
\li NumRes(CExpOp) == 1
\li arg[0] < static_cast<size_t> ( CompareNe )
\li arg[1] != 0; i.e., not all of y_0, y_1, y_2, y_3 are parameters.
\li For j = 0, 1, 2, 3 if y_j is a parameter, arg[2+j] < num_par.
<!-- end conditional_exp_op -->

\param d
is the order of the Taylor coefficient of z that we are  computing.

\param taylor
\b Input:
For j = 0, 1, 2, 3 and k = 0 , ... , \a d,
if y_j is a variable then
\a taylor [ \a arg[2+j] * cap_order + k ]
is the k-th order Taylor coefficient corresponding to y_j.
\n
\a taylor [ \a i_z * \a cap_order + k ]
for k = 0 , ... , \a d
is the k-th order Taylor coefficient corresponding to z.

\param nc_partial
number of columns in the matrix containing the Taylor coefficients.

\param partial
\b Input:
For j = 0, 1, 2, 3 and k = 0 , ... , \a d,
if y_j is a variable then
\a partial [ \a arg[2+j] * nc_partial + k ]
is the partial derivative of G( z , y , x , w , u , ... )
with respect to the k-th order Taylor coefficient corresponding to y_j.
\n
\b Input: \a partial [ \a i_z * \a cap_order + k ]
for k = 0 , ... , \a d
is the partial derivative of G( z , y , x , w , u , ... )
with respect to the k-th order Taylor coefficient corresponding to z.
\n
\b Output:
For j = 0, 1, 2, 3 and k = 0 , ... , \a d,
if y_j is a variable then
\a partial [ \a arg[2+j] * nc_partial + k ]
is the partial derivative of H( y , x , w , u , ... )
with respect to the k-th order Taylor coefficient corresponding to y_j.

*/
template <class Base>
inline void reverse_cond_op(
	size_t         d           ,
	size_t         i_z         ,
	const addr_t*  arg         ,
	size_t         num_par     ,
	const Base*    parameter   ,
	size_t         cap_order   ,
	const Base*    taylor      ,
	size_t         nc_partial  ,
	Base*          partial     )
{	Base y_0, y_1;
	Base zero(0);
	Base* pz;
	Base* py_2;
	Base* py_3;

	CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < static_cast<size_t> (CompareNe) );
	CPPAD_ASSERT_UNKNOWN( NumArg(CExpOp) == 6 );
	CPPAD_ASSERT_UNKNOWN( NumRes(CExpOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( arg[1] != 0 );

	pz = partial + i_z * nc_partial + 0;
	if( arg[1] & 1 )
	{
		y_0 = taylor[ arg[2] * cap_order + 0 ];
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( size_t(arg[2]) < num_par );
		y_0 = parameter[ arg[2] ];
	}
	if( arg[1] & 2 )
	{
		y_1 = taylor[ arg[3] * cap_order + 0 ];
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( size_t(arg[3]) < num_par );
		y_1 = parameter[ arg[3] ];
	}
	if( arg[1] & 4 )
	{
		py_2 = partial + arg[4] * nc_partial;
		size_t j = d + 1;
		while(j--)
		{	py_2[j] += CondExpOp(
				CompareOp( arg[0] ),
				y_0,
				y_1,
				pz[j],
				zero
			);
		}
	}
	if( arg[1] & 8 )
	{
		py_3 = partial + arg[5] * nc_partial;
		size_t j = d + 1;
		while(j--)
		{	py_3[j] += CondExpOp(
				CompareOp( arg[0] ),
				y_0,
				y_1,
				zero,
				pz[j]
			);
		}
	}
	return;
}

/*!
Compute forward Jacobian sparsity patterns for op = CExpOp.

<!-- replace sparse_conditional_exp_op -->
The C++ source code coresponding to this operation is
\verbatim
	z = CondExpRel(y_0, y_1, y_2, y_3)
\endverbatim
where Rel is one of the following: Lt, Le, Eq, Ge, Gt.

\tparam Vector_set
is the type used for vectors of sets. It can be either
sparse_pack or sparse_list.

\param i_z
is the AD variable index corresponding to the variable z.

\param arg
\n
\a arg[0]
is static cast to size_t from the enum type
\verbatim
	enum CompareOp {
		CompareLt,
		CompareLe,
		CompareEq,
		CompareGe,
		CompareGt,
		CompareNe
	}
\endverbatim
for this operation.
Note that arg[0] cannot be equal to CompareNe.
\n
\n
\a arg[1] & 1
\n
If this is zero, y_0 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 2
\n
If this is zero, y_1 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 4
\n
If this is zero, y_2 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 8
\n
If this is zero, y_3 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[2 + j ] for j = 0, 1, 2, 3
\n
is the index corresponding to y_j.

\param num_par
is the total number of values in the vector \a parameter.

\par Checked Assertions
\li NumArg(CExpOp) == 6
\li NumRes(CExpOp) == 1
\li arg[0] < static_cast<size_t> ( CompareNe )
\li arg[1] != 0; i.e., not all of y_0, y_1, y_2, y_3 are parameters.
\li For j = 0, 1, 2, 3 if y_j is a parameter, arg[2+j] < num_par.
<!-- end sparse_conditional_exp_op -->

\param dependency
Are the derivatives with respect to left and right of the expression below
considered to be non-zero:
\code
	CondExpRel(left, right, if_true, if_false)
\endcode
This is used by the optimizer to obtain the correct dependency relations.

\param sparsity
\b Input:
if y_2 is a variable, the set with index t is
the sparsity pattern corresponding to y_2.
This identifies which of the independent variables the variable y_2
depends on.
\n
\b Input:
if y_3 is a variable, the set with index t is
the sparsity pattern corresponding to y_3.
This identifies which of the independent variables the variable y_3
depends on.
\n
\b Output:
The set with index T is
the sparsity pattern corresponding to z.
This identifies which of the independent variables the variable z
depends on.
*/
template <class Vector_set>
inline void forward_sparse_jacobian_cond_op(
	bool               dependency    ,
	size_t             i_z           ,
	const addr_t*      arg           ,
	size_t             num_par       ,
	Vector_set&        sparsity      )
{
	CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < static_cast<size_t> (CompareNe) );
	CPPAD_ASSERT_UNKNOWN( NumArg(CExpOp) == 6 );
	CPPAD_ASSERT_UNKNOWN( NumRes(CExpOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( arg[1] != 0 );
# ifndef NDEBUG
	size_t k = 1;
	for( size_t j = 0; j < 4; j++)
	{	if( ! ( arg[1] & k ) )
			CPPAD_ASSERT_UNKNOWN( size_t(arg[2+j]) < num_par );
		k *= 2;
	}
# endif
	sparsity.clear(i_z);
	if( dependency )
	{	if( arg[1] & 1 )
			sparsity.binary_union(i_z, i_z, arg[2], sparsity);
		if( arg[1] & 2 )
			sparsity.binary_union(i_z, i_z, arg[3], sparsity);
	}
	if( arg[1] & 4 )
		sparsity.binary_union(i_z, i_z, arg[4], sparsity);
	if( arg[1] & 8 )
		sparsity.binary_union(i_z, i_z, arg[5], sparsity);
	return;
}

/*!
Compute reverse Jacobian sparsity patterns for op = CExpOp.

This routine is given the sparsity patterns
for a function G(z, y, x, ... )
and it uses them to compute the sparsity patterns for
\verbatim
	H( y, x, w , u , ... ) = G[ z(x,y) , y , x , w , u , ... ]
\endverbatim
where y represents the combination of y_0, y_1, y_2, and y_3.

<!-- replace sparse_conditional_exp_op -->
The C++ source code coresponding to this operation is
\verbatim
	z = CondExpRel(y_0, y_1, y_2, y_3)
\endverbatim
where Rel is one of the following: Lt, Le, Eq, Ge, Gt.

\tparam Vector_set
is the type used for vectors of sets. It can be either
sparse_pack or sparse_list.

\param i_z
is the AD variable index corresponding to the variable z.

\param arg
\n
\a arg[0]
is static cast to size_t from the enum type
\verbatim
	enum CompareOp {
		CompareLt,
		CompareLe,
		CompareEq,
		CompareGe,
		CompareGt,
		CompareNe
	}
\endverbatim
for this operation.
Note that arg[0] cannot be equal to CompareNe.
\n
\n
\a arg[1] & 1
\n
If this is zero, y_0 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 2
\n
If this is zero, y_1 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 4
\n
If this is zero, y_2 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 8
\n
If this is zero, y_3 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[2 + j ] for j = 0, 1, 2, 3
\n
is the index corresponding to y_j.

\param num_par
is the total number of values in the vector \a parameter.

\par Checked Assertions
\li NumArg(CExpOp) == 6
\li NumRes(CExpOp) == 1
\li arg[0] < static_cast<size_t> ( CompareNe )
\li arg[1] != 0; i.e., not all of y_0, y_1, y_2, y_3 are parameters.
\li For j = 0, 1, 2, 3 if y_j is a parameter, arg[2+j] < num_par.
<!-- end sparse_conditional_exp_op -->

\param dependency
Are the derivatives with respect to left and right of the expression below
considered to be non-zero:
\code
	CondExpRel(left, right, if_true, if_false)
\endcode
This is used by the optimizer to obtain the correct dependency relations.


\param sparsity
if y_2 is a variable, the set with index t is
the sparsity pattern corresponding to y_2.
This identifies which of the dependent variables depend on the variable y_2.
On input, this pattern corresponds to the function G.
On ouput, it corresponds to the function H.
\n
\n
if y_3 is a variable, the set with index t is
the sparsity pattern corresponding to y_3.
This identifies which of the dependent variables depeond on the variable y_3.
On input, this pattern corresponds to the function G.
On ouput, it corresponds to the function H.
\n
\b Output:
The set with index T is
the sparsity pattern corresponding to z.
This identifies which of the dependent variables depend on the variable z.
On input and output, this pattern corresponds to the function G.
*/
template <class Vector_set>
inline void reverse_sparse_jacobian_cond_op(
	bool                dependency    ,
	size_t              i_z           ,
	const addr_t*       arg           ,
	size_t              num_par       ,
	Vector_set&         sparsity      )
{
	CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < static_cast<size_t> (CompareNe) );
	CPPAD_ASSERT_UNKNOWN( NumArg(CExpOp) == 6 );
	CPPAD_ASSERT_UNKNOWN( NumRes(CExpOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( arg[1] != 0 );
# ifndef NDEBUG
	size_t k = 1;
	for( size_t j = 0; j < 4; j++)
	{	if( ! ( arg[1] & k ) )
			CPPAD_ASSERT_UNKNOWN( size_t(arg[2+j]) < num_par );
		k *= 2;
	}
# endif
	if( dependency )
	{	if( arg[1] & 1 )
			sparsity.binary_union(arg[2], arg[2], i_z, sparsity);
		if( arg[1] & 2 )
			sparsity.binary_union(arg[3], arg[3], i_z, sparsity);
	}
	// --------------------------------------------------------------------
	if( arg[1] & 4 )
		sparsity.binary_union(arg[4], arg[4], i_z, sparsity);
	if( arg[1] & 8 )
		sparsity.binary_union(arg[5], arg[5], i_z, sparsity);
	return;
}

/*!
Compute reverse Hessian sparsity patterns for op = CExpOp.

This routine is given the sparsity patterns
for a function G(z, y, x, ... )
and it uses them to compute the sparsity patterns for
\verbatim
	H( y, x, w , u , ... ) = G[ z(x,y) , y , x , w , u , ... ]
\endverbatim
where y represents the combination of y_0, y_1, y_2, and y_3.

<!-- replace sparse_conditional_exp_op -->
The C++ source code coresponding to this operation is
\verbatim
	z = CondExpRel(y_0, y_1, y_2, y_3)
\endverbatim
where Rel is one of the following: Lt, Le, Eq, Ge, Gt.

\tparam Vector_set
is the type used for vectors of sets. It can be either
sparse_pack or sparse_list.

\param i_z
is the AD variable index corresponding to the variable z.

\param arg
\n
\a arg[0]
is static cast to size_t from the enum type
\verbatim
	enum CompareOp {
		CompareLt,
		CompareLe,
		CompareEq,
		CompareGe,
		CompareGt,
		CompareNe
	}
\endverbatim
for this operation.
Note that arg[0] cannot be equal to CompareNe.
\n
\n
\a arg[1] & 1
\n
If this is zero, y_0 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 2
\n
If this is zero, y_1 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 4
\n
If this is zero, y_2 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[1] & 8
\n
If this is zero, y_3 is a parameter. Otherwise it is a variable.
\n
\n
\a arg[2 + j ] for j = 0, 1, 2, 3
\n
is the index corresponding to y_j.

\param num_par
is the total number of values in the vector \a parameter.

\par Checked Assertions
\li NumArg(CExpOp) == 6
\li NumRes(CExpOp) == 1
\li arg[0] < static_cast<size_t> ( CompareNe )
\li arg[1] != 0; i.e., not all of y_0, y_1, y_2, y_3 are parameters.
\li For j = 0, 1, 2, 3 if y_j is a parameter, arg[2+j] < num_par.
<!-- end sparse_conditional_exp_op -->


\param jac_reverse
\a jac_reverse[i_z]
is false (true) if the Jacobian of G with respect to z is always zero
(may be non-zero).
\n
\n
\a jac_reverse[ arg[4] ]
If y_2 is a variable,
\a jac_reverse[ arg[4] ]
is false (true) if the Jacobian with respect to y_2 is always zero
(may be non-zero).
On input, it corresponds to the function G,
and on output it corresponds to the function H.
\n
\n
\a jac_reverse[ arg[5] ]
If y_3 is a variable,
\a jac_reverse[ arg[5] ]
is false (true) if the Jacobian with respect to y_3 is always zero
(may be non-zero).
On input, it corresponds to the function G,
and on output it corresponds to the function H.

\param hes_sparsity
The set with index \a i_z in \a hes_sparsity
is the Hessian sparsity pattern for the function G
where one of the partials is with respect to z.
\n
\n
If y_2 is a variable,
the set with index \a arg[4] in \a hes_sparsity
is the Hessian sparsity pattern
where one of the partials is with respect to y_2.
On input, this pattern corresponds to the function G.
On output, this pattern corresponds to the function H.
\n
\n
If y_3 is a variable,
the set with index \a arg[5] in \a hes_sparsity
is the Hessian sparsity pattern
where one of the partials is with respect to y_3.
On input, this pattern corresponds to the function G.
On output, this pattern corresponds to the function H.
*/
template <class Vector_set>
inline void reverse_sparse_hessian_cond_op(
	size_t               i_z           ,
	const addr_t*        arg           ,
	size_t               num_par       ,
	bool*                jac_reverse   ,
	Vector_set&          hes_sparsity  )
{

	CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < static_cast<size_t> (CompareNe) );
	CPPAD_ASSERT_UNKNOWN( NumArg(CExpOp) == 6 );
	CPPAD_ASSERT_UNKNOWN( NumRes(CExpOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( arg[1] != 0 );
# ifndef NDEBUG
	size_t k = 1;
	for( size_t j = 0; j < 4; j++)
	{	if( ! ( arg[1] & k ) )
			CPPAD_ASSERT_UNKNOWN( size_t(arg[2+j]) < num_par );
		k *= 2;
	}
# endif
	if( arg[1] & 4 )
	{
		hes_sparsity.binary_union(arg[4], arg[4], i_z, hes_sparsity);
		jac_reverse[ arg[4] ] |= jac_reverse[i_z];
	}
	if( arg[1] & 8 )
	{
		hes_sparsity.binary_union(arg[5], arg[5], i_z, hes_sparsity);
		jac_reverse[ arg[5] ] |= jac_reverse[i_z];
	}
	return;
}

} } // END_CPPAD_LOCAL_NAMESPACE
# endif
