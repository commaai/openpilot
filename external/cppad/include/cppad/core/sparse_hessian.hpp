# ifndef CPPAD_CORE_SPARSE_HESSIAN_HPP
# define CPPAD_CORE_SPARSE_HESSIAN_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin sparse_hessian$$
$spell
	jacobian
	recomputed
	CppAD
	valarray
	std
	Bool
	hes
	const
	Taylor
	cppad
	cmake
	colpack
$$

$section Sparse Hessian$$

$head Syntax$$
$icode%hes% = %f%.SparseHessian(%x%, %w%)
%hes% = %f%.SparseHessian(%x%, %w%, %p%)
%n_sweep% = %f%.SparseHessian(%x%, %w%, %p%, %row%, %col%, %hes%, %work%)
%$$

$head Purpose$$
We use $latex n$$ for the $cref/domain/seq_property/Domain/$$ size,
and $latex m$$ for the $cref/range/seq_property/Range/$$ size of $icode f$$.
We use $latex F : \B{R}^n \rightarrow \B{R}^m$$ do denote the
$cref/AD function/glossary/AD Function/$$
corresponding to $icode f$$.
The syntax above sets $icode hes$$ to the Hessian
$latex \[
	H(x) = \dpow{2}{x} \sum_{i=1}^m w_i F_i (x)
\] $$
This routine takes advantage of the sparsity of the Hessian
in order to reduce the amount of computation necessary.
If $icode row$$ and $icode col$$ are present, it also takes
advantage of the reduced set of elements of the Hessian that
need to be computed.
One can use speed tests (e.g. $cref speed_test$$)
to verify that results are computed faster
than when using the routine $cref Hessian$$.

$head f$$
The object $icode f$$ has prototype
$codei%
	ADFun<%Base%> %f%
%$$
Note that the $cref ADFun$$ object $icode f$$ is not $code const$$
(see $cref/Uses Forward/sparse_hessian/Uses Forward/$$ below).

$head x$$
The argument $icode x$$ has prototype
$codei%
	const %VectorBase%& %x%
%$$
(see $cref/VectorBase/sparse_hessian/VectorBase/$$ below)
and its size
must be equal to $icode n$$, the dimension of the
$cref/domain/seq_property/Domain/$$ space for $icode f$$.
It specifies
that point at which to evaluate the Hessian.

$head w$$
The argument $icode w$$ has prototype
$codei%
	const %VectorBase%& %w%
%$$
and size $latex m$$.
It specifies the value of $latex w_i$$ in the expression
for $icode hes$$.
The more components of $latex w$$ that are identically zero,
the more sparse the resulting Hessian may be (and hence the more efficient
the calculation of $icode hes$$ may be).

$head p$$
The argument $icode p$$ is optional and has prototype
$codei%
	const %VectorSet%& %p%
%$$
(see $cref/VectorSet/sparse_hessian/VectorSet/$$ below)
If it has elements of type $code bool$$,
its size is $latex n * n$$.
If it has elements of type $code std::set<size_t>$$,
its size is $latex n$$ and all its set elements are between
zero and $latex n - 1$$.
It specifies a
$cref/sparsity pattern/glossary/Sparsity Pattern/$$
for the Hessian $latex H(x)$$.

$subhead Purpose$$
If this sparsity pattern does not change between calls to
$codei SparseHessian$$, it should be faster to calculate $icode p$$ once and
pass this argument to $codei SparseHessian$$.
If you specify $icode p$$, CppAD will use the same
type of sparsity representation
(vectors of $code bool$$ or vectors of $code std::set<size_t>$$)
for its internal calculations.
Otherwise, the representation
for the internal calculations is unspecified.

$subhead work$$
If you specify $icode work$$ in the calling sequence,
it is not necessary to keep the sparsity pattern; see the heading
$cref/p/sparse_hessian/work/p/$$ under the $icode work$$ description.

$subhead Column Subset$$
If the arguments $icode row$$ and $icode col$$ are present,
and $cref/color_method/sparse_hessian/work/color_method/$$ is
$code cppad.general$$ or $code cppad.symmetric$$,
it is not necessary to compute the entire sparsity pattern.
Only the following subset of column values will matter:
$codei%
	{ %col%[%k%] : %k% = 0 , %...% , %K%-1 }
%$$.


$head row, col$$
The arguments $icode row$$ and $icode col$$ are optional and have prototype
$codei%
	const %VectorSize%& %row%
	const %VectorSize%& %col%
%$$
(see $cref/VectorSize/sparse_hessian/VectorSize/$$ below).
They specify which rows and columns of $latex H (x)$$ are
returned and in what order.
We use $latex K$$ to denote the value $icode%hes%.size()%$$
which must also equal the size of $icode row$$ and $icode col$$.
Furthermore,
for $latex k = 0 , \ldots , K-1$$, it must hold that
$latex row[k] < n$$ and $latex col[k] < n$$.
In addition,
all of the $latex (row[k], col[k])$$ pairs must correspond to a true value
in the sparsity pattern $icode p$$.

$head hes$$
The result $icode hes$$ has prototype
$codei%
	%VectorBase% %hes%
%$$
In the case where $icode row$$ and $icode col$$ are not present,
the size of $icode hes$$ is $latex n * n$$ and
its size is $latex n * n$$.
In this case, for $latex i = 0 , \ldots , n - 1 $$
and $latex ell = 0 , \ldots , n - 1$$
$latex \[
	hes [ j * n + \ell ] = \DD{ w^{\rm T} F }{ x_j }{ x_\ell } ( x )
\] $$
$pre

$$
In the case where the arguments $icode row$$ and $icode col$$ are present,
we use $latex K$$ to denote the size of $icode hes$$.
The input value of its elements does not matter.
Upon return, for $latex k = 0 , \ldots , K - 1$$,
$latex \[
	hes [ k ] = \DD{ w^{\rm T} F }{ x_j }{ x_\ell } (x)
	\; , \;
	\; {\rm where} \;
	j = row[k]
	\; {\rm and } \;
	\ell = col[k]
\] $$

$head work$$
If this argument is present, it has prototype
$codei%
	sparse_hessian_work& %work%
%$$
This object can only be used with the routines $code SparseHessian$$.
During its the first use, information is stored in $icode work$$.
This is used to reduce the work done by future calls to $code SparseHessian$$
with the same $icode f$$, $icode p$$, $icode row$$, and $icode col$$.
If a future call is made where any of these values have changed,
you must first call $icode%work%.clear()%$$
to inform CppAD that this information needs to be recomputed.

$subhead color_method$$
The coloring algorithm determines which rows and columns
can be computed during the same sweep.
This field has prototype
$codei%
	std::string %work%.color_method
%$$
This value only matters on the first call to $code sparse_hessian$$ that
follows the $icode work$$ constructor or a call to
$icode%work%.clear()%$$.
$codei%

"cppad.symmetric"
%$$
This is the default coloring method (after a constructor or $code clear()$$).
It takes advantage of the fact that the Hessian matrix
is symmetric to find a coloring that requires fewer
$cref/sweeps/sparse_hessian/n_sweep/$$.
$codei%

"cppad.general"
%$$
This is the same as the $code "cppad"$$ method for the
$cref/sparse_jacobian/sparse_jacobian/work/color_method/$$ calculation.
$codei%

"colpack.symmetric"
%$$
This method requires that
$cref colpack_prefix$$ was specified on the
$cref/cmake command/cmake/CMake Command/$$ line.
It also takes advantage of the fact that the Hessian matrix is symmetric.
$codei%

"colpack.general"
%$$
This is the same as the $code "colpack"$$ method for the
$cref/sparse_jacobian/sparse_jacobian/work/color_method/$$ calculation.

$subhead colpack.star Deprecated 2017-06-01$$
The $code colpack.star$$ method is deprecated.
It is the same as the $code colpack.symmetric$$
which should be used instead.

$subhead p$$
If $icode work$$ is present, and it is not the first call after
its construction or a clear,
the sparsity pattern $icode p$$ is not used.
This enables one to free the sparsity pattern
and still compute corresponding sparse Hessians.

$head n_sweep$$
The return value $icode n_sweep$$ has prototype
$codei%
	size_t %n_sweep%
%$$
It is the number of first order forward sweeps
used to compute the requested Hessian values.
Each first forward sweep is followed by a second order reverse sweep
so it is also the number of reverse sweeps.
This is proportional to the total work that $code SparseHessian$$ does,
not counting the zero order forward sweep,
or the work to combine multiple columns into a single
forward-reverse sweep pair.

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
After a call to any of the sparse Hessian routines,
the zero order Taylor coefficients correspond to
$icode%f%.Forward(0, %x%)%$$
and the other coefficients are unspecified.

$children%
	example/sparse/sparse_hessian.cpp%
	example/sparse/sub_sparse_hes.cpp%
	example/sparse/sparse_sub_hes.cpp
%$$

$head Example$$
The routine
$cref sparse_hessian.cpp$$
is examples and tests of $code sparse_hessian$$.
It return $code true$$, if it succeeds and $code false$$ otherwise.

$head Subset Hessian$$
The routine
$cref sub_sparse_hes.cpp$$
is an example and test that compute a sparse Hessian
for a subset of the variables.
It returns $code true$$, for success, and $code false$$ otherwise.

$end
-----------------------------------------------------------------------------
*/
# include <cppad/local/std_set.hpp>
# include <cppad/local/color_general.hpp>
# include <cppad/local/color_symmetric.hpp>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
/*!
\file sparse_hessian.hpp
Sparse Hessian driver routine and helper functions.
*/
// ===========================================================================
/*!
class used by SparseHessian to hold information
so it does not need to be recomputed.
*/
class sparse_hessian_work {
	public:
		/// Coloring method: "cppad", or "colpack"
		/// (this field is set by user)
		std::string color_method;
		/// row and column indicies for return values
		/// (some may be reflected by star coloring algorithm)
		CppAD::vector<size_t> row;
		CppAD::vector<size_t> col;
		/// indices that sort the user row and col arrays by color
		CppAD::vector<size_t> order;
		/// results of the coloring algorithm
		CppAD::vector<size_t> color;

		/// constructor
		sparse_hessian_work(void) : color_method("cppad.symmetric")
		{ }
		/// inform CppAD that this information needs to be recomputed
		void clear(void)
		{	color_method = "cppad.symmetric";
			row.clear();
			col.clear();
			order.clear();
			color.clear();
		}
};
// ===========================================================================
/*!
Private helper function that does computation for all Sparse Hessian cases.

\tparam Base
is the base type for the recording that is stored in this ADFun<Base object.

\tparam VectorBase
is a simple vector class with elements of type \a Base.

\tparam VectorSet
is a simple vector class with elements of type
\c bool or \c std::set<size_t>.

\tparam VectorSize
is sparse_pack or sparse_list.

\param x [in]
is a vector specifing the point at which to compute the Hessian.

\param w [in]
is the weighting vector that defines a scalar valued function by
a weighted sum of the components of the vector valued function
$latex F(x)$$.

\param sparsity [in]
is the sparsity pattern for the Hessian that we are calculating.

\param user_row [in]
is the vector of row indices for the returned Hessian values.

\param user_col [in]
is the vector of columns indices for the returned Hessian values.
It must have the same size as user_row.

\param hes [out]
is the vector of Hessian values.
It must have the same size as user_row.
The return value <code>hes[k]</code> is the second partial of
\f$ w^{\rm T} F(x)\f$ with respect to the
<code>row[k]</code> and <code>col[k]</code> component of \f$ x\f$.

\param work
This structure contains information that is computed by \c SparseHessianCompute.
If the sparsity pattern, \c row vector, or \c col vectors
are not the same between calls to \c SparseHessianCompute,
\c work.clear() must be called to reinitialize \c work.

\return
Is the number of first order forward sweeps used to compute the
requested Hessian values.
(This is also equal to the number of second order reverse sweeps.)
The total work, not counting the zero order
forward sweep, or the time to combine computations, is proportional to this
return value.
*/
template<class Base>
template <class VectorBase, class VectorSet, class VectorSize>
size_t ADFun<Base>::SparseHessianCompute(
	const VectorBase&           x           ,
	const VectorBase&           w           ,
	      VectorSet&            sparsity    ,
	const VectorSize&           user_row    ,
	const VectorSize&           user_col    ,
	      VectorBase&           hes         ,
	      sparse_hessian_work&  work        )
{
	using   CppAD::vectorBool;
	size_t i, k, ell;

	CppAD::vector<size_t>& row(work.row);
	CppAD::vector<size_t>& col(work.col);
	CppAD::vector<size_t>& color(work.color);
	CppAD::vector<size_t>& order(work.order);

	size_t n = Domain();

	// some values
	const Base zero(0);
	const Base one(1);

	// check VectorBase is Simple Vector class with Base type elements
	CheckSimpleVector<Base, VectorBase>();

	// number of components of Hessian that are required
	size_t K = hes.size();
	CPPAD_ASSERT_UNKNOWN( size_t( user_row.size() ) == K );
	CPPAD_ASSERT_UNKNOWN( size_t( user_col.size() ) == K );

	CPPAD_ASSERT_UNKNOWN( size_t(x.size()) == n );
	CPPAD_ASSERT_UNKNOWN( color.size() == 0 || color.size() == n );
	CPPAD_ASSERT_UNKNOWN( row.size() == 0   || row.size() == K );
	CPPAD_ASSERT_UNKNOWN( col.size() == 0   || col.size() == K );


	// Point at which we are evaluating the Hessian
	Forward(0, x);

	// check for case where nothing (except Forward above) to do
	if( K == 0 )
		return 0;

	// Rows of the Hessian (i below) correspond to the forward mode index
	// and columns (j below) correspond to the reverse mode index.
	if( color.size() == 0 )
	{
		CPPAD_ASSERT_UNKNOWN( sparsity.n_set() ==  n );
		CPPAD_ASSERT_UNKNOWN( sparsity.end() ==  n );

		// copy user rwo and col to work space
		row.resize(K);
		col.resize(K);
		for(k = 0; k < K; k++)
		{	row[k] = user_row[k];
			col[k] = user_col[k];
		}

		// execute coloring algorithm
		color.resize(n);
		if( work.color_method == "cppad.general" )
			local::color_general_cppad(sparsity, row, col, color);
		else if( work.color_method == "cppad.symmetric" )
			local::color_symmetric_cppad(sparsity, row, col, color);
		else if( work.color_method == "colpack.general" )
		{
# if CPPAD_HAS_COLPACK
			local::color_general_colpack(sparsity, row, col, color);
# else
			CPPAD_ASSERT_KNOWN(
				false,
				"SparseHessian: work.color_method = colpack.general "
				"and colpack_prefix missing from cmake command line."
			);
# endif
		}
		else if(
			work.color_method == "colpack.symmetric" ||
			work.color_method == "colpack.star"
		)
		{
# if CPPAD_HAS_COLPACK
			local::color_symmetric_colpack(sparsity, row, col, color);
# else
			CPPAD_ASSERT_KNOWN(
				false,
				"SparseHessian: work.color_method is "
				"colpack.symmetric or colpack.star\n"
				"and colpack_prefix missing from cmake command line."
			);
# endif
		}
		else
		{	CPPAD_ASSERT_KNOWN(
				false,
				"SparseHessian: work.color_method is not valid."
			);
		}

		// put sorting indices in color order
		VectorSize key(K);
		order.resize(K);
		for(k = 0; k < K; k++)
			key[k] = color[ row[k] ];
		index_sort(key, order);

	}
	size_t n_color = 1;
	for(ell = 0; ell < n; ell++) if( color[ell] < n )
		n_color = std::max(n_color, color[ell] + 1);

	// direction vector for calls to forward (rows of the Hessian)
	VectorBase u(n);

	// location for return values from reverse (columns of the Hessian)
	VectorBase ddw(2 * n);

	// initialize the return value
	for(k = 0; k < K; k++)
		hes[k] = zero;

	// loop over colors
# ifndef NDEBUG
	const std::string& coloring = work.color_method;
# endif
	k = 0;
	for(ell = 0; ell < n_color; ell++)
	if( k == K )
	{	// kludge because colpack returns colors that are not used
		// (it does not know about the subset corresponding to row, col)
		CPPAD_ASSERT_UNKNOWN(
			coloring == "colpack.general" ||
			coloring == "colpack.symmetic" ||
			coloring == "colpack.star"
		);
	}
	else if( color[ row[ order[k] ] ] != ell )
	{	// kludge because colpack returns colors that are not used
		// (it does not know about the subset corresponding to row, col)
		CPPAD_ASSERT_UNKNOWN(
			coloring == "colpack.general" ||
			coloring == "colpack.symmetic" ||
			coloring == "colpack.star"
		);
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( color[ row[ order[k] ] ] == ell );

		// combine all rows with this color
		for(i = 0; i < n; i++)
		{	u[i] = zero;
			if( color[i] == ell )
				u[i] = one;
		}
		// call forward mode for all these rows at once
		Forward(1, u);

		// evaluate derivative of w^T * F'(x) * u
		ddw = Reverse(2, w);

		// set the corresponding components of the result
		while( k < K && color[ row[ order[k] ] ] == ell )
		{	hes[ order[k] ] = ddw[ col[ order[k] ] * 2 + 1 ];
			k++;
		}
	}
	return n_color;
}
// ===========================================================================
// Public Member Functions
// ===========================================================================
/*!
Compute user specified subset of a sparse Hessian.

The C++ source code corresponding to this operation is
\verbatim
	SparceHessian(x, w, p, row, col, hes, work)
\endverbatim

\tparam Base
is the base type for the recording that is stored in this ADFun<Base object.

\tparam VectorBase
is a simple vector class with elements of type \a Base.

\tparam VectorSet
is a simple vector class with elements of type
\c bool or \c std::set<size_t>.

\tparam VectorSize
is a simple vector class with elements of type \c size_t.

\param x [in]
is a vector specifing the point at which to compute the Hessian.

\param w [in]
is the weighting vector that defines a scalar valued function by
a weighted sum of the components of the vector valued function
$latex F(x)$$.

\param p [in]
is the sparsity pattern for the Hessian that we are calculating.

\param row [in]
is the vector of row indices for the returned Hessian values.

\param col [in]
is the vector of columns indices for the returned Hessian values.
It must have the same size are r.

\param hes [out]
is the vector of Hessian values.
It must have the same size are r.
The return value <code>hes[k]</code> is the second partial of
\f$ w^{\rm T} F(x)\f$ with respect to the
<code>row[k]</code> and <code>col[k]</code> component of \f$ x\f$.

\param work
This structure contains information that is computed by \c SparseHessianCompute.
If the sparsity pattern, \c row vector, or \c col vectors
are not the same between calls to \c SparseHessian,
\c work.clear() must be called to reinitialize \c work.

\return
Is the number of first order forward sweeps used to compute the
requested Hessian values.
(This is also equal to the number of second order reverse sweeps.)
The total work, not counting the zero order
forward sweep, or the time to combine computations, is proportional to this
return value.
*/
template<class Base>
template <class VectorBase, class VectorSet, class VectorSize>
size_t ADFun<Base>::SparseHessian(
	const VectorBase&     x    ,
	const VectorBase&     w    ,
	const VectorSet&      p    ,
	const VectorSize&     row  ,
	const VectorSize&     col  ,
	VectorBase&           hes  ,
	sparse_hessian_work&  work )
{
	size_t n    = Domain();
	size_t K    = hes.size();
# ifndef NDEBUG
	size_t k;
	CPPAD_ASSERT_KNOWN(
		size_t(x.size()) == n ,
		"SparseHessian: size of x not equal domain dimension for f."
	);
	CPPAD_ASSERT_KNOWN(
		size_t(row.size()) == K && size_t(col.size()) == K ,
		"SparseHessian: either r or c does not have the same size as ehs."
	);
	CPPAD_ASSERT_KNOWN(
		work.color.size() == 0 || work.color.size() == n,
		"SparseHessian: invalid value in work."
	);
	for(k = 0; k < K; k++)
	{	CPPAD_ASSERT_KNOWN(
			row[k] < n,
			"SparseHessian: invalid value in r."
		);
		CPPAD_ASSERT_KNOWN(
			col[k] < n,
			"SparseHessian: invalid value in c."
		);
	}
	if( work.color.size() != 0 )
		for(size_t j = 0; j < n; j++) CPPAD_ASSERT_KNOWN(
			work.color[j] <= n,
			"SparseHessian: invalid value in work."
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
		const char* error_msg = "SparseHessian: sparsity pattern"
		" does not have proper row or column dimension";
		sparsity_user2internal(s, p, n, n, transpose, error_msg);
	}
	n_sweep = SparseHessianCompute(x, w, s, row, col, hes, work);
	return n_sweep;
}
/*!
Compute a sparse Hessian.

The C++ source code coresponding to this operation is
\verbatim
	hes = SparseHessian(x, w, p)
\endverbatim


\tparam Base
is the base type for the recording that is stored in this
ADFun<Base object.

\tparam VectorBase
is a simple vector class with elements of the \a Base.

\tparam VectorSet
is a simple vector class with elements of type
\c bool or \c std::set<size_t>.

\param x [in]
is a vector specifing the point at which to compute the Hessian.

\param w [in]
The Hessian is computed for a weighted sum of the components
of the function corresponding to this ADFun<Base> object.
The argument \a w specifies the weights for each component.
It must have size equal to the range dimension for this ADFun<Base> object.

\param p [in]
is a sparsity pattern for the Hessian.

\return
Will be a vector of size \c n * n containing the Hessian of
at the point specified by \a x
(where \c n is the domain dimension for this ADFun<Base> object).
*/
template <class Base>
template <class VectorBase, class VectorSet>
VectorBase ADFun<Base>::SparseHessian(
	const VectorBase& x, const VectorBase& w, const VectorSet& p
)
{	size_t i, j, k;

	size_t n = Domain();
	VectorBase hes(n * n);

	CPPAD_ASSERT_KNOWN(
		size_t(x.size()) == n,
		"SparseHessian: size of x not equal domain size for f."
	);

	typedef typename VectorSet::value_type Set_type;
	typedef typename local::internal_sparsity<Set_type>::pattern_type Pattern_type;

	// initialize the return value as zero
	Base zero(0);
	for(i = 0; i < n; i++)
		for(j = 0; j < n; j++)
			hes[i * n + j] = zero;

	// arguments to SparseHessianCompute
	Pattern_type          s;
	CppAD::vector<size_t> row;
	CppAD::vector<size_t> col;
	sparse_hessian_work   work;
	bool transpose = false;
	const char* error_msg = "SparseHessian: sparsity pattern"
	" does not have proper row or column dimension";
	sparsity_user2internal(s, p, n, n, transpose, error_msg);
	k = 0;
	for(i = 0; i < n; i++)
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
	VectorBase H(K);

	// now we have folded this into the following case
	SparseHessianCompute(x, w, s, row, col, H, work);

	// now set the non-zero return values
	for(k = 0; k < K; k++)
		hes[ row[k] * n + col[k] ] = H[k];

	return hes;
}
/*!
Compute a sparse Hessian

The C++ source code coresponding to this operation is
\verbatim
	hes = SparseHessian(x, w)
\endverbatim


\tparam Base
is the base type for the recording that is stored in this
ADFun<Base object.

\tparam VectorBase
is a simple vector class with elements of the \a Base.

\param x [in]
is a vector specifing the point at which to compute the Hessian.

\param w [in]
The Hessian is computed for a weighted sum of the components
of the function corresponding to this ADFun<Base> object.
The argument \a w specifies the weights for each component.
It must have size equal to the range dimension for this ADFun<Base> object.

\return
Will be a vector of size \c n * n containing the Hessian of
at the point specified by \a x
(where \c n is the domain dimension for this ADFun<Base> object).
*/
template <class Base>
template <class VectorBase>
VectorBase ADFun<Base>::SparseHessian(const VectorBase &x, const VectorBase &w)
{	size_t i, j, k;
	typedef CppAD::vectorBool VectorBool;

	size_t m = Range();
	size_t n = Domain();

	// determine the sparsity pattern p for Hessian of w^T F
	VectorBool r(n * n);
	for(j = 0; j < n; j++)
	{	for(k = 0; k < n; k++)
			r[j * n + k] = false;
		r[j * n + j] = true;
	}
	ForSparseJac(n, r);
	//
	VectorBool s(m);
	for(i = 0; i < m; i++)
		s[i] = w[i] != 0;
	VectorBool p = RevSparseHes(n, s);

	// compute sparse Hessian
	return SparseHessian(x, w, p);
}

} // END_CPPAD_NAMESPACE
# endif
