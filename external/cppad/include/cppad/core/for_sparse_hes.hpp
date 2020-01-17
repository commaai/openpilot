# ifndef CPPAD_CORE_FOR_SPARSE_HES_HPP
# define CPPAD_CORE_FOR_SPARSE_HES_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin ForSparseHes$$
$spell
	Andrea Walther
	std
	VecAD
	Jacobian
	Jac
	Hessian
	Hes
	const
	Bool
	Dep
	proportional
	var
	cpp
$$

$section Hessian Sparsity Pattern: Forward Mode$$

$head Syntax$$
$icode%h% = %f%.ForSparseHes(%r%, %s%)
%$$

$head Purpose$$
We use $latex F : \B{R}^n \rightarrow \B{R}^m$$ to denote the
$cref/AD function/glossary/AD Function/$$ corresponding to $icode f$$.
we define
$latex \[
\begin{array}{rcl}
H(x)
& = & \partial_x \left[ \partial_u S \cdot F[ x + R \cdot u ] \right]_{u=0}
\\
& = & R^\R{T} \cdot (S \cdot F)^{(2)} ( x ) \cdot R
\end{array}
\] $$
Where $latex R \in \B{R}^{n \times n}$$ is a diagonal matrix
and $latex S \in \B{R}^{1 \times m}$$ is a row vector.
Given a
$cref/sparsity pattern/glossary/Sparsity Pattern/$$
for the diagonal of $latex R$$ and the vector $latex S$$,
$code ForSparseHes$$ returns a sparsity pattern for the $latex H(x)$$.

$head f$$
The object $icode f$$ has prototype
$codei%
	const ADFun<%Base%> %f%
%$$

$head x$$
If the operation sequence in $icode f$$ is
$cref/independent/glossary/Operation/Independent/$$ of
the independent variables in $latex x \in B^n$$,
the sparsity pattern is valid for all values of
(even if it has $cref CondExp$$ or $cref VecAD$$ operations).

$head r$$
The argument $icode r$$ has prototype
$codei%
	const %VectorSet%& %r%
%$$
(see $cref/VectorSet/ForSparseHes/VectorSet/$$ below)
If it has elements of type $code bool$$,
its size is $latex n$$.
If it has elements of type $code std::set<size_t>$$,
its size is one and all the elements of $icode%s%[0]%$$
are between zero and $latex n - 1$$.
It specifies a
$cref/sparsity pattern/glossary/Sparsity Pattern/$$
for the diagonal of $latex R$$.
The fewer non-zero elements in this sparsity pattern,
the faster the calculation should be and the more sparse
$latex H(x)$$ should be.

$head s$$
The argument $icode s$$ has prototype
$codei%
	const %VectorSet%& %s%
%$$
(see $cref/VectorSet/ForSparseHes/VectorSet/$$ below)
If it has elements of type $code bool$$,
its size is $latex m$$.
If it has elements of type $code std::set<size_t>$$,
its size is one and all the elements of $icode%s%[0]%$$
are between zero and $latex m - 1$$.
It specifies a
$cref/sparsity pattern/glossary/Sparsity Pattern/$$
for the vector $icode S$$.
The fewer non-zero elements in this sparsity pattern,
the faster the calculation should be and the more sparse
$latex H(x)$$ should be.

$head h$$
The result $icode h$$ has prototype
$codei%
	%VectorSet%& %h%
%$$
(see $cref/VectorSet/ForSparseHes/VectorSet/$$ below).
If $icode h$$ has elements of type $code bool$$,
its size is $latex n * n$$.
If it has elements of type $code std::set<size_t>$$,
its size is $latex n$$ and all the set elements are between
zero and $icode%n%-1%$$ inclusive.
It specifies a
$cref/sparsity pattern/glossary/Sparsity Pattern/$$
for the matrix $latex H(x)$$.

$head VectorSet$$
The type $icode VectorSet$$ must be a $cref SimpleVector$$ class with
$cref/elements of type/SimpleVector/Elements of Specified Type/$$
$code bool$$ or $code std::set<size_t>$$;
see $cref/sparsity pattern/glossary/Sparsity Pattern/$$ for a discussion
of the difference.
The type of the elements of
$cref/VectorSet/ForSparseHes/VectorSet/$$ must be the
same as the type of the elements of $icode r$$.

$head Algorithm$$
See Algorithm II in
$italic Computing sparse Hessians with automatic differentiation$$
by Andrea Walther.
Note that $icode s$$ provides the information so that
'dead ends' are not included in the sparsity pattern.

$head Example$$
$children%
	example/sparse/for_sparse_hes.cpp
%$$
The file
$cref for_sparse_hes.cpp$$
contains an example and test of this operation.
It returns true if it succeeds and false otherwise.

$end
-----------------------------------------------------------------------------
*/
# include <algorithm>
# include <cppad/local/pod_vector.hpp>
# include <cppad/local/std_set.hpp>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
/*!
\file for_sparse_hes.hpp
Forward mode Hessian sparsity patterns.
*/
// ===========================================================================
// ForSparseHesCase
/*!
Private helper function for ForSparseHes(q, s) bool sparsity.

All of the description in the public member function ForSparseHes(q, s)
applies.

\param set_type
is a \c bool value. This argument is used to dispatch to the proper source
code depending on the vlaue of \c VectorSet::value_type.

\param r
See \c ForSparseHes(r, s).

\param s
See \c ForSparseHes(r, s).

\param h
is the return value for the corresponging call to \c ForSparseJac(q, s).
*/
template <class Base>
template <class VectorSet>
void ADFun<Base>::ForSparseHesCase(
	bool              set_type         ,
	const VectorSet&  r                ,
	const VectorSet&  s                ,
	VectorSet&        h                )
{	size_t n = Domain();
	size_t m = Range();
	//
	// check Vector is Simple VectorSet class with bool elements
	CheckSimpleVector<bool, VectorSet>();
	//
	CPPAD_ASSERT_KNOWN(
		size_t(r.size()) == n,
		"ForSparseHes: size of r is not equal to\n"
		"domain dimension for ADFun object."
	);
	CPPAD_ASSERT_KNOWN(
		size_t(s.size()) == m,
		"ForSparseHes: size of s is not equal to\n"
		"range dimension for ADFun object."
	);
	//
	// sparsity pattern corresponding to r
	local::sparse_pack for_jac_pattern;
	for_jac_pattern.resize(num_var_tape_, n + 1);
	for(size_t i = 0; i < n; i++)
	{	CPPAD_ASSERT_UNKNOWN( ind_taddr_[i] < n + 1 );
		// ind_taddr_[i] is operator taddr for i-th independent variable
		CPPAD_ASSERT_UNKNOWN( play_.GetOp( ind_taddr_[i] ) == local::InvOp );
		//
		if( r[i] )
			for_jac_pattern.add_element( ind_taddr_[i], ind_taddr_[i] );
	}
	// compute forward Jacobiain sparsity pattern
	bool dependency = false;
	local::ForJacSweep(
		dependency,
		n,
		num_var_tape_,
		&play_,
		for_jac_pattern
	);
	// sparsity pattern correspnding to s
	local::sparse_pack rev_jac_pattern;
	rev_jac_pattern.resize(num_var_tape_, 1);
	for(size_t i = 0; i < m; i++)
	{	CPPAD_ASSERT_UNKNOWN( dep_taddr_[i] < num_var_tape_ );
		if( s[i] )
			rev_jac_pattern.add_element( dep_taddr_[i], 0);
	}
	// compute reverse sparsity pattern for dependency analysis
	// (note that we are only want non-zero derivatives not true dependency)
	local::RevJacSweep(
		dependency,
		n,
		num_var_tape_,
		&play_,
		rev_jac_pattern
	);
	// vector of sets that will hold the forward Hessain values
	local::sparse_pack for_hes_pattern;
	for_hes_pattern.resize(n+1, n+1);
	//
	// compute the Hessian sparsity patterns
	local::ForHesSweep(
		n,
		num_var_tape_,
		&play_,
		for_jac_pattern,
		rev_jac_pattern,
		for_hes_pattern
	);
	// initialize return values corresponding to independent variables
	h.resize(n * n);
	for(size_t i = 0; i < n; i++)
	{	for(size_t j = 0; j < n; j++)
			h[ i * n + j ] = false;
	}
	// copy to result pattern
	CPPAD_ASSERT_UNKNOWN( for_hes_pattern.end() == n+1 );
	for(size_t i = 0; i < n; i++)
	{	// ind_taddr_[i] is operator taddr for i-th independent variable
		CPPAD_ASSERT_UNKNOWN( ind_taddr_[i] == i + 1 );
		CPPAD_ASSERT_UNKNOWN( play_.GetOp( ind_taddr_[i] ) == local::InvOp );

		// extract the result from for_hes_pattern
		local::sparse_pack::const_iterator itr(for_hes_pattern, ind_taddr_[i] );
		size_t j = *itr;
		while( j < for_hes_pattern.end() )
		{	CPPAD_ASSERT_UNKNOWN( 0 < j )
			h[ i * n + (j-1) ] = true;
			j = *(++itr);
		}
	}
}
/*!
Private helper function for ForSparseHes(q, s) set sparsity.

All of the description in the public member function ForSparseHes(q, s)
applies.

\param set_type
is a \c std::set<size_t> value.
This argument is used to dispatch to the proper source
code depending on the vlaue of \c VectorSet::value_type.

\param r
See \c ForSparseHes(r, s).

\param s
See \c ForSparseHes(q, s).

\param h
is the return value for the corresponging call to \c ForSparseJac(q, s).
*/
template <class Base>
template <class VectorSet>
void ADFun<Base>::ForSparseHesCase(
	const std::set<size_t>&   set_type         ,
	const VectorSet&          r                ,
	const VectorSet&          s                ,
	VectorSet&                h                )
{	size_t n = Domain();
# ifndef NDEBUG
	size_t m = Range();
# endif
	std::set<size_t>::const_iterator itr_1;
	//
	// check VectorSet is Simple Vector class with sets for elements
	CheckSimpleVector<std::set<size_t>, VectorSet>(
		local::one_element_std_set<size_t>(), local::two_element_std_set<size_t>()
	);
	CPPAD_ASSERT_KNOWN(
		r.size() == 1,
		"ForSparseHes: size of s is not equal to one."
	);
	CPPAD_ASSERT_KNOWN(
		s.size() == 1,
		"ForSparseHes: size of s is not equal to one."
	);
	//
	// sparsity pattern corresponding to r
	local::sparse_list for_jac_pattern;
	for_jac_pattern.resize(num_var_tape_, n + 1);
	itr_1 = r[0].begin();
	while( itr_1 != r[0].end() )
	{	size_t i = *itr_1++;
		CPPAD_ASSERT_UNKNOWN( ind_taddr_[i] < n + 1 );
		// ind_taddr_[i] is operator taddr for i-th independent variable
		CPPAD_ASSERT_UNKNOWN( play_.GetOp( ind_taddr_[i] ) == local::InvOp );
		//
		for_jac_pattern.add_element( ind_taddr_[i], ind_taddr_[i] );
	}
	// compute forward Jacobiain sparsity pattern
	bool dependency = false;
	local::ForJacSweep(
		dependency,
		n,
		num_var_tape_,
		&play_,
		for_jac_pattern
	);
	// sparsity pattern correspnding to s
	local::sparse_list rev_jac_pattern;
	rev_jac_pattern.resize(num_var_tape_, 1);
	itr_1 = s[0].begin();
	while( itr_1 != s[0].end() )
	{	size_t i = *itr_1++;
		CPPAD_ASSERT_KNOWN(
			i < m,
			"ForSparseHes: an element of the set s[0] has value "
			"greater than or equal m"
		);
		CPPAD_ASSERT_UNKNOWN( dep_taddr_[i] < num_var_tape_ );
		rev_jac_pattern.add_element( dep_taddr_[i], 0);
	}
	//
	// compute reverse sparsity pattern for dependency analysis
	// (note that we are only want non-zero derivatives not true dependency)
	local::RevJacSweep(
		dependency,
		n,
		num_var_tape_,
		&play_,
		rev_jac_pattern
	);
	//
	// vector of sets that will hold reverse Hessain values
	local::sparse_list for_hes_pattern;
	for_hes_pattern.resize(n+1, n+1);
	//
	// compute the Hessian sparsity patterns
	local::ForHesSweep(
		n,
		num_var_tape_,
		&play_,
		for_jac_pattern,
		rev_jac_pattern,
		for_hes_pattern
	);
	// return values corresponding to independent variables
	// j is index corresponding to reverse mode partial
	h.resize(n);
	CPPAD_ASSERT_UNKNOWN( for_hes_pattern.end() == n+1 );
	for(size_t i = 0; i < n; i++)
	{	CPPAD_ASSERT_UNKNOWN( ind_taddr_[i] == i + 1 );
		CPPAD_ASSERT_UNKNOWN( play_.GetOp( ind_taddr_[i] ) == local::InvOp );

		// extract the result from for_hes_pattern
		local::sparse_list::const_iterator itr_2(for_hes_pattern, ind_taddr_[i] );
		size_t j = *itr_2;
		while( j < for_hes_pattern.end() )
		{	CPPAD_ASSERT_UNKNOWN( 0 < j )
				h[i].insert(j-1);
			j = *(++itr_2);
		}
	}
}

// ===========================================================================
// ForSparseHes

/*!
User API for Hessian sparsity patterns using reverse mode.

The C++ source code corresponding to this operation is
\verbatim
	h = f.ForSparseHes(q, r)
\endverbatim

\tparam Base
is the base type for this recording.

\tparam VectorSet
is a simple vector with elements of type \c bool
or \c std::set<size_t>.

\param r
is a vector with size \c n that specifies the sparsity pattern
for the diagonal of the matrix \f$ R \f$,
where \c n is the number of independent variables
corresponding to the operation sequence stored in \a play.

\param s
is a vector with size \c m that specifies the sparsity pattern
for the vector \f$ S \f$,
where \c m is the number of dependent variables
corresponding to the operation sequence stored in \a play.

\return
The return vector is a sparsity pattern for \f$ H(x) \f$
\f[
	H(x) = R^T ( S * F)^{(2)} (x) R
\f]
where \f$ F \f$ is the function corresponding to the operation sequence
and \a x is any argument value.
*/

template <class Base>
template <class VectorSet>
VectorSet ADFun<Base>::ForSparseHes(
	const VectorSet& r, const VectorSet& s
)
{	VectorSet h;
	typedef typename VectorSet::value_type Set_type;

	// Should check to make sure q is same as in previous call to
	// forward sparse Jacobian.
	ForSparseHesCase(
		Set_type()    ,
		r             ,
		s             ,
		h
	);

	return h;
}
// ===========================================================================
// ForSparseHesCheckpoint
/*!
Hessian sparsity patterns calculation used by checkpoint functions.

\tparam Base
is the base type for this recording.

\param r
is a vector with size n that specifies the sparsity pattern
for the diagonal of \f$ R \f$,
where n is the number of independent variables
corresponding to the operation sequence stored in play_.

\param s
is a vector with size m that specifies the sparsity pattern
for the vector \f$ S \f$,
where m is the number of dependent variables
corresponding to the operation sequence stored in play_.

\param h
The input size and elements of h do not matter.
On output, h is the sparsity pattern for the matrix \f$ H(x) R \f$.

\par Assumptions
The forward jacobian sparsity pattern must be currently stored
in this ADFUN object.
*/

// The checkpoint class is not yet using forward sparse Hessians.
# ifdef CPPAD_NOT_DEFINED
template <class Base>
void ADFun<Base>::ForSparseHesCheckpoint(
	vector<bool>&                 r         ,
	vector<bool>&                 s         ,
	local::sparse_list&                  h         )
{
	size_t n = Domain();
	size_t m = Range();

	// checkpoint functions should get this right
	CPPAD_ASSERT_UNKNOWN( for_jac_sparse_pack_.n_set() == 0 );
	CPPAD_ASSERT_UNKNOWN( for_jac_sparse_set_.n_set() == 0   );
	CPPAD_ASSERT_UNKNOWN( s.size()                    == m );

	// Array that holds the reverse Jacobiain dependcy flags.
	// Initialize as true for dependent variables, flase for others.
	local::pod_vector<bool> RevJac;
	RevJac.extend(num_var_tape_);
	for(size_t i = 0; i < num_var_tape_; i++)
		RevJac[i] = false;
	for(size_t i = 0; i < m; i++)
	{	CPPAD_ASSERT_UNKNOWN( dep_taddr_[i] < num_var_tape_ )
		RevJac[ dep_taddr_[i] ] = s[i];
	}

	// holds forward Hessian sparsity pattern for all variables
	local::sparse_list for_hes_pattern;
	for_hes_pattern.resize(n+1, n+1);

	// compute Hessian sparsity pattern for all variables
	local::ForHesSweep(
		n,
		num_var_tape_,
		&play_,
		for_jac_sparse_set_,
		RevJac.data(),
		for_hes_pattern
	);

	// dimension the return value
	if( transpose )
		h.resize(n, n);
	else
		h.resize(n, n);

	// j is index corresponding to reverse mode partial
	for(size_t j = 0; j < n; j++)
	{	CPPAD_ASSERT_UNKNOWN( ind_taddr_[j] < num_var_tape_ );

		// ind_taddr_[j] is operator taddr for j-th independent variable
		CPPAD_ASSERT_UNKNOWN( ind_taddr_[j] == j + 1 );
		CPPAD_ASSERT_UNKNOWN( play_.GetOp( ind_taddr_[j] ) == local::InvOp );

		// extract the result from for_hes_pattern
		CPPAD_ASSERT_UNKNOWN( for_hes_pattern.end() == q );
		local::sparse_list::const_iterator itr(for_hes_pattern, .j + 1);
		size_t i = *itr;
		while( i < q )
		{	if( transpose )
				h.add_element(j,  i);
			else	h.add_element(i, j);
			i = *(++itr);
		}
	}
}
# endif

} // END_CPPAD_NAMESPACE
# endif
