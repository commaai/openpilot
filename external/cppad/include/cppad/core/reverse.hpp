// $Id$
# ifndef CPPAD_CORE_REVERSE_HPP
# define CPPAD_CORE_REVERSE_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

# include <algorithm>
# include <cppad/local/pod_vector.hpp>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
/*!
\file reverse.hpp
Compute derivatives using reverse mode.
*/


/*!
Use reverse mode to compute derivative of forward mode Taylor coefficients.

The function
\f$ X : {\rm R} \times {\rm R}^{n \times q} \rightarrow {\rm R} \f$
is defined by
\f[
X(t , u) = \sum_{k=0}^{q-1} u^{(k)} t^k
\f]
The function
\f$ Y : {\rm R} \times {\rm R}^{n \times q} \rightarrow {\rm R} \f$
is defined by
\f[
Y(t , u) = F[ X(t, u) ]
\f]
The function
\f$ W : {\rm R}^{n \times q} \rightarrow {\rm R} \f$ is defined by
\f[
W(u) = \sum_{k=0}^{q-1} ( w^{(k)} )^{\rm T}
\frac{1}{k !} \frac{ \partial^k } { t^k } Y(0, u)
\f]

\tparam Base
base type for the operator; i.e., this operation sequence was recorded
using AD< \a Base > and computations by this routine are done using type
\a Base.

\tparam VectorBase
is a Simple Vector class with elements of type \a Base.

\param q
is the number of the number of Taylor coefficients that are being
differentiated (per variable).

\param w
is the weighting for each of the Taylor coefficients corresponding
to dependent variables.
If the argument \a w has size <tt>m * q </tt>,
for \f$ k = 0 , \ldots , q-1 \f$ and \f$ i = 0, \ldots , m-1 \f$,
\f[
	w_i^{(k)} = w [ i * q + k ]
\f]
If the argument \a w has size \c m ,
for \f$ k = 0 , \ldots , q-1 \f$ and \f$ i = 0, \ldots , m-1 \f$,
\f[
w_i^{(k)} = \left\{ \begin{array}{ll}
	w [ i ] & {\rm if} \; k = q-1
	\\
	0       & {\rm otherwise}
\end{array} \right.
\f]

\return
Is a vector \f$ dw \f$ such that
for \f$ j = 0 , \ldots , n-1 \f$ and
\f$ k = 0 , \ldots , q-1 \f$
\f[
	dw[ j * q + k ] = W^{(1)} ( x )_{j,k}
\f]
where the matrix \f$ x \f$ is the value for \f$ u \f$
that corresponding to the forward mode Taylor coefficients
for the independent variables as specified by previous calls to Forward.

*/
template <typename Base>
template <typename VectorBase>
VectorBase ADFun<Base>::Reverse(size_t q, const VectorBase &w)
{	// constants
	const Base zero(0);

	// temporary indices
	size_t i, j, k;

	// number of independent variables
	size_t n = ind_taddr_.size();

	// number of dependent variables
	size_t m = dep_taddr_.size();

	local::pod_vector<Base> Partial;
	Partial.extend(num_var_tape_  * q);

	// update maximum memory requirement
	// memoryMax = std::max( memoryMax,
	//	Memory() + num_var_tape_  * q * sizeof(Base)
	// );

	// check VectorBase is Simple Vector class with Base type elements
	CheckSimpleVector<Base, VectorBase>();

	CPPAD_ASSERT_KNOWN(
		size_t(w.size()) == m || size_t(w.size()) == (m * q),
		"Argument w to Reverse does not have length equal to\n"
		"the dimension of the range for the corresponding ADFun."
	);
	CPPAD_ASSERT_KNOWN(
		q > 0,
		"The first argument to Reverse must be greater than zero."
	);
	CPPAD_ASSERT_KNOWN(
		num_order_taylor_ >= q,
		"Less that q taylor_ coefficients are currently stored"
		" in this ADFun object."
	);
	// special case where multiple forward directions have been computed,
	// but we are only using the one direction zero order results
	if( (q == 1) & (num_direction_taylor_ > 1) )
	{	num_order_taylor_ = 1;        // number of orders to copy
		size_t c = cap_order_taylor_; // keep the same capacity setting
		size_t r = 1;                 // only keep one direction
		capacity_order(c, r);
	}
	CPPAD_ASSERT_KNOWN(
		num_direction_taylor_ == 1,
		"Reverse mode for Forward(q, r, xq) with more than one direction"
		"\n(r > 1) is not yet supported for q > 1."
	);
	// initialize entire Partial matrix to zero
	for(i = 0; i < num_var_tape_; i++)
		for(j = 0; j < q; j++)
			Partial[i * q + j] = zero;

	// set the dependent variable direction
	// (use += because two dependent variables can point to same location)
	for(i = 0; i < m; i++)
	{	CPPAD_ASSERT_UNKNOWN( dep_taddr_[i] < num_var_tape_  );
		if( size_t(w.size()) == m )
			Partial[dep_taddr_[i] * q + q - 1] += w[i];
		else
		{	for(k = 0; k < q; k++)
				// ? should use += here, first make test to demonstrate bug
				Partial[ dep_taddr_[i] * q + k ] = w[i * q + k ];
		}
	}

	// evaluate the derivatives
	CPPAD_ASSERT_UNKNOWN( cskip_op_.size() == play_.num_op_rec() );
	CPPAD_ASSERT_UNKNOWN( load_op_.size()  == play_.num_load_op_rec() );
	local::ReverseSweep(
		q - 1,
		n,
		num_var_tape_,
		&play_,
		cap_order_taylor_,
		taylor_.data(),
		q,
		Partial.data(),
		cskip_op_.data(),
		load_op_
	);

	// return the derivative values
	VectorBase value(n * q);
	for(j = 0; j < n; j++)
	{	CPPAD_ASSERT_UNKNOWN( ind_taddr_[j] < num_var_tape_  );

		// independent variable taddr equals its operator taddr
		CPPAD_ASSERT_UNKNOWN( play_.GetOp( ind_taddr_[j] ) == local::InvOp );

		// by the Reverse Identity Theorem
		// partial of y^{(k)} w.r.t. u^{(0)} is equal to
		// partial of y^{(q-1)} w.r.t. u^{(q - 1 - k)}
		if( size_t(w.size()) == m )
		{	for(k = 0; k < q; k++)
				value[j * q + k ] =
					Partial[ind_taddr_[j] * q + q - 1 - k];
		}
		else
		{	for(k = 0; k < q; k++)
				value[j * q + k ] =
					Partial[ind_taddr_[j] * q + k];
		}
	}
	CPPAD_ASSERT_KNOWN( ! ( hasnan(value) && check_for_nan_ ) ,
		"dw = f.Reverse(q, w): has a nan,\n"
		"but none of its Taylor coefficents are nan."
	);

	return value;
}


} // END_CPPAD_NAMESPACE
# endif
