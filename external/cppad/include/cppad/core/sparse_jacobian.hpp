# ifndef CPPAD_CORE_SPARSE_JACOBIAN_HPP
# define CPPAD_CORE_SPARSE_JACOBIAN_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

// maximum number of sparse directions to compute at the same time

// # define CPPAD_SPARSE_JACOBIAN_MAX_MULTIPLE_DIRECTION 1
# define CPPAD_SPARSE_JACOBIAN_MAX_MULTIPLE_DIRECTION 64

/*
$begin sparse_jacobian$$
$spell
	cppad
	colpack
	cmake
	recomputed
	valarray
	std
	CppAD
	Bool
	jac
	Jacobian
	Jacobians
	const
	Taylor
$$

$section Sparse Jacobian$$

$head Syntax$$
$icode%jac% = %f%.SparseJacobian(%x%)
%jac% = %f%.SparseJacobian(%x%, %p%)
%n_sweep% = %f%.SparseJacobianForward(%x%, %p%, %row%, %col%, %jac%, %work%)
%n_sweep% = %f%.SparseJacobianReverse(%x%, %p%, %row%, %col%, %jac%, %work%)
%$$

$head Purpose$$
We use $latex n$$ for the $cref/domain/seq_property/Domain/$$ size,
and $latex m$$ for the $cref/range/seq_property/Range/$$ size of $icode f$$.
We use $latex F : \B{R}^n \rightarrow \B{R}^m$$ do denote the
$cref/AD function/glossary/AD Function/$$
corresponding to $icode f$$.
The syntax above sets $icode jac$$ to the Jacobian
$latex \[
	jac = F^{(1)} (x)
\] $$
This routine takes advantage of the sparsity of the Jacobian
in order to reduce the amount of computation necessary.
If $icode row$$ and $icode col$$ are present, it also takes
advantage of the reduced set of elements of the Jacobian that
need to be computed.
One can use speed tests (e.g. $cref speed_test$$)
to verify that results are computed faster
than when using the routine $cref Jacobian$$.

$head f$$
The object $icode f$$ has prototype
$codei%
	ADFun<%Base%> %f%
%$$
Note that the $cref ADFun$$ object $icode f$$ is not $code const$$
(see $cref/Uses Forward/sparse_jacobian/Uses Forward/$$ below).

$head x$$
The argument $icode x$$ has prototype
$codei%
	const %VectorBase%& %x%
%$$
(see $cref/VectorBase/sparse_jacobian/VectorBase/$$ below)
and its size
must be equal to $icode n$$, the dimension of the
$cref/domain/seq_property/Domain/$$ space for $icode f$$.
It specifies
that point at which to evaluate the Jacobian.

$head p$$
The argument $icode p$$ is optional and has prototype
$codei%
	const %VectorSet%& %p%
%$$
(see $cref/VectorSet/sparse_jacobian/VectorSet/$$ below).
If it has elements of type $code bool$$,
its size is $latex m * n$$.
If it has elements of type $code std::set<size_t>$$,
its size is $latex m$$ and all its set elements are between
zero and $latex n - 1$$.
It specifies a
$cref/sparsity pattern/glossary/Sparsity Pattern/$$
for the Jacobian $latex F^{(1)} (x)$$.
$pre

$$
If this sparsity pattern does not change between calls to
$codei SparseJacobian$$, it should be faster to calculate $icode p$$ once
(using $cref ForSparseJac$$ or $cref RevSparseJac$$)
and then pass $icode p$$ to $codei SparseJacobian$$.
Furthermore, if you specify $icode work$$ in the calling sequence,
it is not necessary to keep the sparsity pattern; see the heading
$cref/p/sparse_jacobian/work/p/$$ under the $icode work$$ description.
$pre

$$
In addition,
if you specify $icode p$$, CppAD will use the same
type of sparsity representation
(vectors of $code bool$$ or vectors of $code std::set<size_t>$$)
for its internal calculations.
Otherwise, the representation
for the internal calculations is unspecified.

$head row, col$$
The arguments $icode row$$ and $icode col$$ are optional and have prototype
$codei%
	const %VectorSize%& %row%
	const %VectorSize%& %col%
%$$
(see $cref/VectorSize/sparse_jacobian/VectorSize/$$ below).
They specify which rows and columns of $latex F^{(1)} (x)$$ are
computes and in what order.
Not all the non-zero entries in $latex F^{(1)} (x)$$ need be computed,
but all the entries specified by $icode row$$ and $icode col$$
must be possibly non-zero in the sparsity pattern.
We use $latex K$$ to denote the value $icode%jac%.size()%$$
which must also equal the size of $icode row$$ and $icode col$$.
Furthermore,
for $latex k = 0 , \ldots , K-1$$, it must hold that
$latex row[k] < m$$ and $latex col[k] < n$$.

$head jac$$
The result $icode jac$$ has prototype
$codei%
	%VectorBase%& %jac%
%$$
In the case where the arguments $icode row$$ and $icode col$$ are not present,
the size of $icode jac$$ is $latex m * n$$ and
for $latex i = 0 , \ldots , m-1$$,
$latex j = 0 , \ldots , n-1$$,
$latex \[
	jac [ i * n + j ] = \D{ F_i }{ x_j } (x)
\] $$
$pre

$$
In the case where the arguments $icode row$$ and $icode col$$ are present,
we use $latex K$$ to denote the size of $icode jac$$.
The input value of its elements does not matter.
Upon return, for $latex k = 0 , \ldots , K - 1$$,
$latex \[
	jac [ k ] = \D{ F_i }{ x_j } (x)
	\; , \;
	\; {\rm where} \;
	i = row[k]
	\; {\rm and } \;
	j = col[k]
\] $$

$head work$$
If this argument is present, it has prototype
$codei%
	sparse_jacobian_work& %work%
%$$
This object can only be used with the routines
$code SparseJacobianForward$$ and $code SparseJacobianReverse$$.
During its the first use, information is stored in $icode work$$.
This is used to reduce the work done by future calls to the same mode
(forward or reverse),
the same $icode f$$, $icode p$$, $icode row$$, and $icode col$$.
If a future call is for a different mode,
or any of these values have changed,
you must first call $icode%work%.clear()%$$
to inform CppAD that this information needs to be recomputed.

$subhead color_method$$
The coloring algorithm determines which columns (forward mode)
or rows (reverse mode) can be computed during the same sweep.
This field has prototype
$codei%
	std::string %work%.color_method
%$$
and its default value (after a constructor or $code clear()$$)
is $code "cppad"$$.
If $cref colpack_prefix$$ is specified on the
$cref/cmake command/cmake/CMake Command/$$ line,
you can set this method to $code "colpack"$$.
This value only matters on the first call to $code sparse_jacobian$$
that follows the $icode work$$ constructor or a call to
$icode%work%.clear()%$$.

$subhead p$$
If $icode work$$ is present, and it is not the first call after
its construction or a clear,
the sparsity pattern $icode p$$ is not used.
This enables one to free the sparsity pattern
and still compute corresponding sparse Jacobians.

$head n_sweep$$
The return value $icode n_sweep$$ has prototype
$codei%
	size_t %n_sweep%
%$$
If $code SparseJacobianForward$$ ($code SparseJacobianReverse$$) is used,
$icode n_sweep$$ is the number of first order forward (reverse) sweeps
used to compute the requested Jacobian values.
(This is also the number of colors determined by the coloring method
mentioned above).
This is proportional to the total work that $code SparseJacobian$$ does,
not counting the zero order forward sweep,
or the work to combine multiple columns (rows) into a single sweep.

$head VectorBase$$
The type $icode VectorBase$$ must be a $cref SimpleVector$$ class with
$cref/elements of type/SimpleVector/Elements of Specified Type/$$
$icode Base$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$head VectorSet$$
The type $icode VectorSet$$ must be a $cref SimpleVector$$ class with
$cref/elements of type/SimpleVector/Elements of Specified Type/$$
$code bool$$ or $code std::set<size_t>$$;
see $cref/sparsity pattern/glossary/Sparsity Pattern/$$ for a discussion
of the difference.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$subhead Restrictions$$
If $icode VectorSet$$ has elements of $code std::set<size_t>$$,
then $icode%p%[%i%]%$$ must return a reference (not a copy) to the
corresponding set.
According to section 26.3.2.3 of the 1998 C++ standard,
$code std::valarray< std::set<size_t> >$$ does not satisfy
this condition.

$head VectorSize$$
The type $icode VectorSize$$ must be a $cref SimpleVector$$ class with
$cref/elements of type/SimpleVector/Elements of Specified Type/$$
$code size_t$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$head Uses Forward$$
After each call to $cref Forward$$,
the object $icode f$$ contains the corresponding
$cref/Taylor coefficients/glossary/Taylor Coefficient/$$.
After a call to any of the sparse Jacobian routines,
the zero order Taylor coefficients correspond to
$icode%f%.Forward(0, %x%)%$$
and the other coefficients are unspecified.

After $code SparseJacobian$$,
the previous calls to $cref Forward$$ are undefined.

$head Example$$
$children%
	example/sparse/sparse_jacobian.cpp
%$$
The routine
$cref sparse_jacobian.cpp$$
is examples and tests of $code sparse_jacobian$$.
It return $code true$$, if it succeeds and $code false$$ otherwise.

$end
==============================================================================
*/
# include <cppad/local/std_set.hpp>
# include <cppad/local/color_general.hpp>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
/*!
\file sparse_jacobian.hpp
Sparse Jacobian driver routine and helper functions.
*/
// ===========================================================================
/*!
class used by SparseJacobian to hold information so it does not need to be
recomputed.
*/
class sparse_jacobian_work {
	public:
		/// Coloring method: "cppad", or "colpack"
		/// (this field is set by user)
		std::string color_method;
		/// indices that sort the user row and col arrays by color
		CppAD::vector<size_t> order;
		/// results of the coloring algorithm
		CppAD::vector<size_t> color;

		/// constructor
		sparse_jacobian_work(void) : color_method("cppad")
		{ }
		/// reset coloring method to its default and
		/// inform CppAD that color and order need to be recomputed
		void clear(void)
		{	color_method = "cppad";
			order.clear();
			color.clear();
		}
};
// ===========================================================================
/*!
Private helper function forward mode cases

\tparam Base
is the base type for the recording that is stored in this
<code>ADFun<Base></code> object.

\tparam VectorBase
is a simple vector class with elements of type \a Base.

\tparam VectorSet
is either sparse_pack or sparse_list.

\tparam VectorSize
is a simple vector class with elements of type \c size_t.

\param x [in]
is a vector specifing the point at which to compute the Jacobian.

\param p_transpose [in]
If <code>work.color.size() != 0</code>,
then \c p_transpose is not used.
Otherwise, it is a
sparsity pattern for the transpose of the Jacobian of this ADFun<Base> object.
Note that we do not change the values in \c p_transpose,
but is not \c const because we use its iterator facility.

\param row [in]
is the vector of row indices for the returned Jacobian values.

\param col [in]
is the vector of columns indices for the returned Jacobian values.
It must have the same size as \c row.

\param jac [out]
is the vector of Jacobian values. We use \c K to denote the size of \c jac.
The return value <code>jac[k]</code> is the partial of the
<code>row[k]</code> range component of the function with respect
the the <code>col[k]</code> domain component of its argument.

\param work
<code>work.color_method</code> is an input. The rest of
this structure contains information that is computed by \c SparseJacobainFor.
If the sparsity pattern, \c row vector, or \c col vectors
are not the same between calls to \c SparseJacobianFor,
\c work.clear() must be called to reinitialize \c work.

\return
Is the number of first order forward sweeps used to compute the
requested Jacobian values. The total work, not counting the zero order
forward sweep, or the time to combine computations, is proportional to this
return value.
*/
template<class Base>
template <class VectorBase, class VectorSet, class VectorSize>
size_t ADFun<Base>::SparseJacobianFor(
	const VectorBase&            x           ,
	      VectorSet&             p_transpose ,
	const VectorSize&            row         ,
	const VectorSize&            col         ,
	      VectorBase&            jac         ,
	       sparse_jacobian_work& work        )
{
	size_t j, k, ell;

	CppAD::vector<size_t>& order(work.order);
	CppAD::vector<size_t>& color(work.color);

	size_t m = Range();
	size_t n = Domain();

	// some values
	const Base zero(0);
	const Base one(1);

	// check VectorBase is Simple Vector class with Base type elements
	CheckSimpleVector<Base, VectorBase>();

	CPPAD_ASSERT_UNKNOWN( size_t(x.size()) == n );
	CPPAD_ASSERT_UNKNOWN( color.size() == 0 || color.size() == n );

	// number of components of Jacobian that are required
	size_t K = size_t(jac.size());
	CPPAD_ASSERT_UNKNOWN( size_t( row.size() ) == K );
	CPPAD_ASSERT_UNKNOWN( size_t( col.size() ) == K );

	// Point at which we are evaluating the Jacobian
	Forward(0, x);

	// check for case where nothing (except Forward above) to do
	if( K == 0 )
		return 0;

	if( color.size() == 0 )
	{
		CPPAD_ASSERT_UNKNOWN( p_transpose.n_set() ==  n );
		CPPAD_ASSERT_UNKNOWN( p_transpose.end() ==  m );

		// execute coloring algorithm
		color.resize(n);
		if(	work.color_method == "cppad" )
			local::color_general_cppad(p_transpose, col, row, color);
		else if( work.color_method == "colpack" )
		{
# if CPPAD_HAS_COLPACK
			local::color_general_colpack(p_transpose, col, row, color);
# else
			CPPAD_ASSERT_KNOWN(
				false,
				"SparseJacobianForward: work.color_method = colpack "
				"and colpack_prefix missing from cmake command line."
			);
# endif
		}
		else CPPAD_ASSERT_KNOWN(
			false,
			"SparseJacobianForward: work.color_method is not valid."
		);

		// put sorting indices in color order
		VectorSize key(K);
		order.resize(K);
		for(k = 0; k < K; k++)
			key[k] = color[ col[k] ];
		index_sort(key, order);
	}
	size_t n_color = 1;
	for(j = 0; j < n; j++) if( color[j] < n )
		n_color = std::max(n_color, color[j] + 1);

	// initialize the return value
	for(k = 0; k < K; k++)
		jac[k] = zero;

# if CPPAD_SPARSE_JACOBIAN_MAX_MULTIPLE_DIRECTION == 1
	// direction vector and return values for calls to forward
	VectorBase dx(n), dy(m);

	// loop over colors
	k = 0;
	for(ell = 0; ell < n_color; ell++)
	{	CPPAD_ASSERT_UNKNOWN( color[ col[ order[k] ] ] == ell );

		// combine all columns with this color
		for(j = 0; j < n; j++)
		{	dx[j] = zero;
			if( color[j] == ell )
				dx[j] = one;
		}
		// call forward mode for all these columns at once
		dy = Forward(1, dx);

		// set the corresponding components of the result
		while( k < K && color[ col[order[k]] ] == ell )
		{	jac[ order[k] ] = dy[row[order[k]]];
			k++;
		}
	}
# else
	// abbreviation for this value
	size_t max_r = CPPAD_SPARSE_JACOBIAN_MAX_MULTIPLE_DIRECTION;
	CPPAD_ASSERT_UNKNOWN( max_r > 1 );

	// count the number of colors done so far
	size_t count_color = 0;
	// count the sparse matrix entries done so far
	k = 0;
	while( count_color < n_color )
	{	// number of colors we will do this time
		size_t r = std::min(max_r , n_color - count_color);
		VectorBase dx(n * r), dy(m * r);

		// loop over colors we will do this tme
		for(ell = 0; ell < r; ell++)
		{	// combine all columns with this color
			for(j = 0; j < n; j++)
			{	dx[j * r + ell] = zero;
				if( color[j] == ell + count_color )
					dx[j * r + ell] = one;
			}
		}
		size_t q           = 1;
		dy = Forward(q, r, dx);

		// store results
		for(ell = 0; ell < r; ell++)
		{	// set the components of the result for this color
			while( k < K && color[ col[order[k]] ] == ell + count_color )
			{	jac[ order[k] ] = dy[ row[order[k]] * r + ell ];
				k++;
			}
		}
		count_color += r;
	}
# endif
	return n_color;
}
/*!
Private helper function for reverse mode cases.

\tparam Base
is the base type for the recording that is stored in this
<code>ADFun<Base></code> object.

\tparam VectorBase
is a simple vector class with elements of type \a Base.

\tparam VectorSet
is either sparse_pack or sparse_list.

\tparam VectorSize
is a simple vector class with elements of type \c size_t.

\param x [in]
is a vector specifing the point at which to compute the Jacobian.

\param p [in]
If <code>work.color.size() != 0</code>, then \c p is not used.
Otherwise, it is a
sparsity pattern for the Jacobian of this ADFun<Base> object.
Note that we do not change the values in \c p,
but is not \c const because we use its iterator facility.

\param row [in]
is the vector of row indices for the returned Jacobian values.

\param col [in]
is the vector of columns indices for the returned Jacobian values.
It must have the same size as \c row.

\param jac [out]
is the vector of Jacobian values.
It must have the same size as \c row.
The return value <code>jac[k]</code> is the partial of the
<code>row[k]</code> range component of the function with respect
the the <code>col[k]</code> domain component of its argument.

\param work
<code>work.color_method</code> is an input. The rest of
This structure contains information that is computed by \c SparseJacobainRev.
If the sparsity pattern, \c row vector, or \c col vectors
are not the same between calls to \c SparseJacobianRev,
\c work.clear() must be called to reinitialize \c work.

\return
Is the number of first order reverse sweeps used to compute the
reverse Jacobian values. The total work, not counting the zero order
forward sweep, or the time to combine computations, is proportional to this
return value.
*/
template<class Base>
template <class VectorBase, class VectorSet, class VectorSize>
size_t ADFun<Base>::SparseJacobianRev(
	const VectorBase&           x           ,
	      VectorSet&            p           ,
	const VectorSize&           row         ,
	const VectorSize&           col         ,
	      VectorBase&           jac         ,
	      sparse_jacobian_work& work        )
{
	size_t i, k, ell;

	CppAD::vector<size_t>& order(work.order);
	CppAD::vector<size_t>& color(work.color);

	size_t m = Range();
	size_t n = Domain();

	// some values
	const Base zero(0);
	const Base one(1);

	// check VectorBase is Simple Vector class with Base type elements
	CheckSimpleVector<Base, VectorBase>();

	CPPAD_ASSERT_UNKNOWN( size_t(x.size()) == n );
	CPPAD_ASSERT_UNKNOWN (color.size() == m || color.size() == 0 );

	// number of components of Jacobian that are required
	size_t K = size_t(jac.size());
	CPPAD_ASSERT_UNKNOWN( size_t( size_t( row.size() ) ) == K );
	CPPAD_ASSERT_UNKNOWN( size_t( size_t( col.size() ) ) == K );

	// Point at which we are evaluating the Jacobian
	Forward(0, x);

	// check for case where nothing (except Forward above) to do
	if( K == 0 )
		return 0;

	if( color.size() == 0 )
	{
		CPPAD_ASSERT_UNKNOWN( p.n_set() == m );
		CPPAD_ASSERT_UNKNOWN( p.end()   == n );

		// execute the coloring algorithm
		color.resize(m);
		if(	work.color_method == "cppad" )
			local::color_general_cppad(p, row, col, color);
		else if( work.color_method == "colpack" )
		{
# if CPPAD_HAS_COLPACK
			local::color_general_colpack(p, row, col, color);
# else
			CPPAD_ASSERT_KNOWN(
				false,
				"SparseJacobianReverse: work.color_method = colpack "
				"and colpack_prefix missing from cmake command line."
			);
# endif
		}
		else CPPAD_ASSERT_KNOWN(
			false,
			"SparseJacobianReverse: work.color_method is not valid."
		);

		// put sorting indices in color order
		VectorSize key(K);
		order.resize(K);
		for(k = 0; k < K; k++)
			key[k] = color[ row[k] ];
		index_sort(key, order);
	}
	size_t n_color = 1;
	for(i = 0; i < m; i++) if( color[i] < m )
		n_color = std::max(n_color, color[i] + 1);

	// weighting vector for calls to reverse
	VectorBase w(m);

	// location for return values from Reverse
	VectorBase dw(n);

	// initialize the return value
	for(k = 0; k < K; k++)
		jac[k] = zero;

	// loop over colors
	k = 0;
	for(ell = 0; ell < n_color; ell++)
	{	CPPAD_ASSERT_UNKNOWN( color[ row[ order[k] ] ] == ell );

		// combine all the rows with this color
		for(i = 0; i < m; i++)
		{	w[i] = zero;
			if( color[i] == ell )
				w[i] = one;
		}
		// call reverse mode for all these rows at once
		dw = Reverse(1, w);

		// set the corresponding components of the result
		while( k < K && color[ row[order[k]] ]  == ell )
		{	jac[ order[k] ] = dw[col[order[k]]];
			k++;
		}
	}
	return n_color;
}
// ==========================================================================
// Public Member functions
// ==========================================================================
/*!
Compute user specified subset of a sparse Jacobian using forward mode.

The C++ source code corresponding to this operation is
\verbatim
	SparceJacobianForward(x, p, row, col, jac, work)
\endverbatim

\tparam Base
is the base type for the recording that is stored in this
<code>ADFun<Base></code> object.

\tparam VectorBase
is a simple vector class with elements of type \a Base.

\tparam VectorSet
is a simple vector class with elements of type
\c bool or \c std::set<size_t>.

\tparam VectorSize
is a simple vector class with elements of type \c size_t.

\param x [in]
is a vector specifing the point at which to compute the Jacobian.

\param p [in]
is the sparsity pattern for the Jacobian that we are calculating.

\param row [in]
is the vector of row indices for the returned Jacobian values.

\param col [in]
is the vector of columns indices for the returned Jacobian values.
It must have the same size as \c row.

\param jac [out]
is the vector of Jacobian values.
It must have the same size as \c row.
The return value <code>jac[k]</code> is the partial of the
<code>row[k]</code> range component of the function with respect
the the <code>col[k]</code> domain component of its argument.

\param work [in,out]
this structure contains information that depends on the function object,
sparsity pattern, \c row vector, and \c col vector.
If they are not the same between calls to \c SparseJacobianForward,
\c work.clear() must be called to reinitialize them.

\return
Is the number of first order forward sweeps used to compute the
requested Jacobian values. The total work, not counting the zero order
forward sweep, or the time to combine computations, is proportional to this
return value.
*/
template<class Base>
template <class VectorBase, class VectorSet, class VectorSize>
size_t ADFun<Base>::SparseJacobianForward(
	const VectorBase&     x    ,
	const VectorSet&      p    ,
	const VectorSize&     row  ,
	const VectorSize&     col  ,
	VectorBase&           jac  ,
	sparse_jacobian_work& work )
{
	size_t n = Domain();
	size_t m = Range();
	size_t K = jac.size();
# ifndef NDEBUG
	size_t k;
	CPPAD_ASSERT_KNOWN(
		size_t(x.size()) == n ,
		"SparseJacobianForward: size of x not equal domain dimension for f."
	);
	CPPAD_ASSERT_KNOWN(
		size_t(row.size()) == K && size_t(col.size()) == K ,
		"SparseJacobianForward: either r or c does not have "
		"the same size as jac."
	);
	CPPAD_ASSERT_KNOWN(
		work.color.size() == 0 || work.color.size() == n,
		"SparseJacobianForward: invalid value in work."
	);
	for(k = 0; k < K; k++)
	{	CPPAD_ASSERT_KNOWN(
			row[k] < m,
			"SparseJacobianForward: invalid value in r."
		);
		CPPAD_ASSERT_KNOWN(
			col[k] < n,
			"SparseJacobianForward: invalid value in c."
		);
	}
	if( work.color.size() != 0 )
		for(size_t j = 0; j < n; j++) CPPAD_ASSERT_KNOWN(
			work.color[j] <= n,
			"SparseJacobianForward: invalid value in work."
	);
# endif
	// check for case where there is nothing to compute
	size_t n_sweep = 0;
	if( K == 0 )
		return n_sweep;

	typedef typename VectorSet::value_type Set_type;
	typedef typename local::internal_sparsity<Set_type>::pattern_type Pattern_type;
	Pattern_type s_transpose;
	if( work.color.size() == 0 )
	{	bool transpose = true;
		const char* error_msg = "SparseJacobianForward: transposed sparsity"
		" pattern does not have proper row or column dimension";
		sparsity_user2internal(s_transpose, p, n, m, transpose, error_msg);
	}
	n_sweep = SparseJacobianFor(x, s_transpose, row, col, jac, work);
	return n_sweep;
}
/*!
Compute user specified subset of a sparse Jacobian using forward mode.

The C++ source code corresponding to this operation is
\verbatim
	SparceJacobianReverse(x, p, row, col, jac, work)
\endverbatim

\tparam Base
is the base type for the recording that is stored in this
<code>ADFun<Base></code> object.

\tparam VectorBase
is a simple vector class with elements of type \a Base.

\tparam VectorSet
is a simple vector class with elements of type
\c bool or \c std::set<size_t>.

\tparam VectorSize
is a simple vector class with elements of type \c size_t.

\param x [in]
is a vector specifing the point at which to compute the Jacobian.

\param p [in]
is the sparsity pattern for the Jacobian that we are calculating.

\param row [in]
is the vector of row indices for the returned Jacobian values.

\param col [in]
is the vector of columns indices for the returned Jacobian values.
It must have the same size as \c row.

\param jac [out]
is the vector of Jacobian values.
It must have the same size as \c row.
The return value <code>jac[k]</code> is the partial of the
<code>row[k]</code> range component of the function with respect
the the <code>col[k]</code> domain component of its argument.

\param work [in,out]
this structure contains information that depends on the function object,
sparsity pattern, \c row vector, and \c col vector.
If they are not the same between calls to \c SparseJacobianReverse,
\c work.clear() must be called to reinitialize them.

\return
Is the number of first order reverse sweeps used to compute the
reverse Jacobian values. The total work, not counting the zero order
forward sweep, or the time to combine computations, is proportional to this
return value.
*/
template<class Base>
template <class VectorBase, class VectorSet, class VectorSize>
size_t ADFun<Base>::SparseJacobianReverse(
	const VectorBase&     x    ,
	const VectorSet&      p    ,
	const VectorSize&     row  ,
	const VectorSize&     col  ,
	VectorBase&           jac  ,
	sparse_jacobian_work& work )
{
	size_t m = Range();
	size_t n = Domain();
	size_t K = jac.size();
# ifndef NDEBUG
	size_t k;
	CPPAD_ASSERT_KNOWN(
		size_t(x.size()) == n ,
		"SparseJacobianReverse: size of x not equal domain dimension for f."
	);
	CPPAD_ASSERT_KNOWN(
		size_t(row.size()) == K && size_t(col.size()) == K ,
		"SparseJacobianReverse: either r or c does not have "
		"the same size as jac."
	);
	CPPAD_ASSERT_KNOWN(
		work.color.size() == 0 || work.color.size() == m,
		"SparseJacobianReverse: invalid value in work."
	);
	for(k = 0; k < K; k++)
	{	CPPAD_ASSERT_KNOWN(
			row[k] < m,
			"SparseJacobianReverse: invalid value in r."
		);
		CPPAD_ASSERT_KNOWN(
			col[k] < n,
			"SparseJacobianReverse: invalid value in c."
		);
	}
	if( work.color.size() != 0 )
		for(size_t i = 0; i < m; i++) CPPAD_ASSERT_KNOWN(
			work.color[i] <= m,
			"SparseJacobianReverse: invalid value in work."
	);
# endif
	// check for case where there is nothing to compute
	size_t n_sweep = 0;
	if( K == 0 )
		return n_sweep;

	typedef typename VectorSet::value_type Set_type;
	typedef typename local::internal_sparsity<Set_type>::pattern_type Pattern_type;
	Pattern_type s;
	if( work.color.size() == 0 )
	{	bool transpose = false;
		const char* error_msg = "SparseJacobianReverse: sparsity"
		" pattern does not have proper row or column dimension";
		sparsity_user2internal(s, p, m, n, transpose, error_msg);
	}
	n_sweep = SparseJacobianRev(x, s, row, col, jac, work);
	return n_sweep;
}
/*!
Compute a sparse Jacobian.

The C++ source code corresponding to this operation is
\verbatim
	jac = SparseJacobian(x, p)
\endverbatim

\tparam Base
is the base type for the recording that is stored in this
<code>ADFun<Base></code> object.

\tparam VectorBase
is a simple vector class with elements of type \a Base.

\tparam VectorSet
is a simple vector class with elements of type
\c bool or \c std::set<size_t>.

\param x [in]
is a vector specifing the point at which to compute the Jacobian.

\param p [in]
is the sparsity pattern for the Jacobian that we are calculating.

\return
Will be a vector if size \c m * n containing the Jacobian at the
specified point (in row major order).
*/
template <class Base>
template <class VectorBase, class VectorSet>
VectorBase ADFun<Base>::SparseJacobian(
	const VectorBase& x, const VectorSet& p
)
{	size_t i, j, k;

	size_t m = Range();
	size_t n = Domain();
	VectorBase jac(m * n);

	CPPAD_ASSERT_KNOWN(
		size_t(x.size()) == n,
		"SparseJacobian: size of x not equal domain size for f."
	);
	CheckSimpleVector<Base, VectorBase>();

	typedef typename VectorSet::value_type Set_type;
	typedef typename local::internal_sparsity<Set_type>::pattern_type Pattern_type;

	// initialize the return value as zero
	Base zero(0);
	for(i = 0; i < m; i++)
		for(j = 0; j < n; j++)
			jac[i * n + j] = zero;

	sparse_jacobian_work work;
	CppAD::vector<size_t> row;
	CppAD::vector<size_t> col;
	if( n <= m )
	{
		// need an internal copy of sparsity pattern
		Pattern_type s_transpose;
		bool transpose = true;
		const char* error_msg = "SparseJacobian: transposed sparsity"
		" pattern does not have proper row or column dimension";
		sparsity_user2internal(s_transpose, p, n, m, transpose, error_msg);

		k = 0;
		for(j = 0; j < n; j++)
		{	typename Pattern_type::const_iterator itr(s_transpose, j);
			i = *itr;
			while( i != s_transpose.end() )
			{	row.push_back(i);
				col.push_back(j);
				k++;
				i = *(++itr);
			}
		}
		size_t K = k;
		VectorBase J(K);

		// now we have folded this into the following case
		SparseJacobianFor(x, s_transpose, row, col, J, work);

		// now set the non-zero return values
		for(k = 0; k < K; k++)
			jac[ row[k] * n + col[k] ] = J[k];
	}
	else
	{
		// need an internal copy of sparsity pattern
		Pattern_type s;
		bool transpose = false;
		const char* error_msg = "SparseJacobian: sparsity"
		" pattern does not have proper row or column dimension";
		sparsity_user2internal(s, p, m, n, transpose, error_msg);

		k = 0;
		for(i = 0; i < m; i++)
		{	typename Pattern_type::const_iterator itr(s, i);
			j = *itr;
			while( j != s.end() )
			{	row.push_back(i);
				col.push_back(j);
				k++;
				j = *(++itr);
			}
		}
		size_t K = k;
		VectorBase J(K);

		// now we have folded this into the following case
		SparseJacobianRev(x, s, row, col, J, work);

		// now set the non-zero return values
		for(k = 0; k < K; k++)
			jac[ row[k] * n + col[k] ] = J[k];
	}

	return jac;
}

/*!
Compute a sparse Jacobian.

The C++ source code corresponding to this operation is
\verbatim
	jac = SparseJacobian(x)
\endverbatim

\tparam Base
is the base type for the recording that is stored in this
<code>ADFun<Base></code> object.

\tparam VectorBase
is a simple vector class with elements of the \a Base.

\param x [in]
is a vector specifing the point at which to compute the Jacobian.

\return
Will be a vector of size \c m * n containing the Jacobian at the
specified point (in row major order).
*/
template <class Base>
template <class VectorBase>
VectorBase ADFun<Base>::SparseJacobian( const VectorBase& x )
{	typedef CppAD::vectorBool   VectorBool;

	size_t m = Range();
	size_t n = Domain();

	// sparsity pattern for Jacobian
	VectorBool p(m * n);

	if( n <= m )
	{	size_t j, k;

		// use forward mode
		VectorBool r(n * n);
		for(j = 0; j < n; j++)
		{	for(k = 0; k < n; k++)
				r[j * n + k] = false;
			r[j * n + j] = true;
		}
		p = ForSparseJac(n, r);
	}
	else
	{	size_t i, k;

		// use reverse mode
		VectorBool s(m * m);
		for(i = 0; i < m; i++)
		{	for(k = 0; k < m; k++)
				s[i * m + k] = false;
			s[i * m + i] = true;
		}
		p = RevSparseJac(m, s);
	}
	return SparseJacobian(x, p);
}

} // END_CPPAD_NAMESPACE
# undef CPPAD_SPARSE_JACOBIAN_MAX_MULTIPLE_DIRECTION
# endif
