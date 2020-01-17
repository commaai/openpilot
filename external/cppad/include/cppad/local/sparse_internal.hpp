# ifndef CPPAD_LOCAL_SPARSE_INTERNAL_HPP
# define CPPAD_LOCAL_SPARSE_INTERNAL_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

// necessary definitions
# include <cppad/core/define.hpp>
# include <cppad/local/sparse_pack.hpp>
# include <cppad/local/sparse_list.hpp>

namespace CppAD { namespace local { // BEGIN_CPPAD_LOCAL_NAMESPACE
/*!
\file sparse_internal.hpp
Routines that enable code to be independent of which internal spasity pattern
is used.
*/
// ---------------------------------------------------------------------------
/*!
Template structure used obtain the internal sparsity pattern type
form the corresponding element type.
The general form is not valid, must use a specialization.

\tparam Element_type
type of an element in the sparsity structrue.

\par <code>internal_sparsity<Element_type>::pattern_type</code>
is the type of the corresponding internal sparsity pattern.
*/
template <class Element_type> struct internal_sparsity;
/// Specilization for \c bool elements.
template <>
struct internal_sparsity<bool>
{
	typedef sparse_pack pattern_type;
};
/// Specilization for <code>std::set<size_t></code> elements.
template <>
struct internal_sparsity< std::set<size_t> >
{
	typedef sparse_list pattern_type;
};
// ---------------------------------------------------------------------------
/*!
Update the internal sparsity pattern for a sub-set of rows

\tparam SizeVector
The type used for index sparsity patterns. This is a simple vector
with elements of type size_t.

\tparam InternalSparsitiy
The type used for intenal sparsity patterns. This can be either
sparse_pack or sparse_list.

\param zero_empty
If this is true, the internal sparstity pattern corresponds to row zero
must be empty on input and will be emtpy output; i.e., any corresponding
values in pattern_in will be ignored.

\param input_empty
If this is true, the initial sparsity pattern for row
internal_index[i] is empty for all i.
In this case, one is setting the sparsity patterns; i.e.,
the output pattern in row internal_index[i] is the corresponding
entries in pattern.

\param transpose
If this is true, pattern_in is transposed.

\param internal_index
This specifies the sub-set of rows in internal_sparsity that we are updating.
If traspose is false (true),
this is the mapping from row (column) index in pattern_in to the corresponding
row index in the internal_pattern.

\param internal_pattern
On input, the number of sets internal_pattern.n_set(),
and possible elements internal_pattern.end(), have been set.
If input_empty is true, and all of the sets
in internal_index are empty on input.
On output, the entries in pattern_in are added to internal_pattern.
To be specific, suppose transpose is false, and (i, j) is a possibly
non-zero entry in pattern_in, the entry (internal_index[i], j) is added
to internal_pattern.
On the other hand, if transpose is true,
the entry (internal_index[j], i) is added to internal_pattern.

\param pattern_in
This is the sparsity pattern for variables,
or its transpose, depending on the value of transpose.
*/
template <class SizeVector, class InternalSparsity>
void set_internal_sparsity(
	bool                          zero_empty       ,
	bool                          input_empty      ,
	bool                          transpose        ,
	const vector<size_t>&         internal_index   ,
	InternalSparsity&             internal_pattern ,
	const sparse_rc<SizeVector>&  pattern_in       )
{
# ifndef NDEBUG
	size_t nr = internal_index.size();
	size_t nc = internal_pattern.end();
	if( transpose )
	{	CPPAD_ASSERT_UNKNOWN( pattern_in.nr() == nc );
		CPPAD_ASSERT_UNKNOWN( pattern_in.nc() == nr );
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( pattern_in.nr() == nr );
		CPPAD_ASSERT_UNKNOWN( pattern_in.nc() == nc );
	}
	if( input_empty ) for(size_t i = 0; i < nr; i++)
	{	size_t i_var = internal_index[i];
		CPPAD_ASSERT_UNKNOWN( internal_pattern.number_elements(i_var) == 0 );
	}
# endif
	const SizeVector& row( pattern_in.row() );
	const SizeVector& col( pattern_in.col() );
	size_t nnz = row.size();
	for(size_t k = 0; k < nnz; k++)
	{	size_t r = row[k];
		size_t c = col[k];
		if( transpose )
			std::swap(r, c);
		//
		size_t i_var = internal_index[r];
		CPPAD_ASSERT_UNKNOWN( i_var < internal_pattern.n_set() );
		CPPAD_ASSERT_UNKNOWN( c < nc );
		bool ignore  = zero_empty && i_var == 0;
		if( ! ignore )
			internal_pattern.add_element( internal_index[r], c );
	}
}
template <class InternalSparsity>
void set_internal_sparsity(
	bool                          zero_empty       ,
	bool                          input_empty      ,
	bool                          transpose        ,
	const vector<size_t>&         internal_index   ,
	InternalSparsity&             internal_pattern ,
	const vectorBool&             pattern_in       )
{	size_t nr = internal_index.size();
	size_t nc = internal_pattern.end();
# ifndef NDEBUG
	CPPAD_ASSERT_UNKNOWN( pattern_in.size() == nr * nc );
	if( input_empty ) for(size_t i = 0; i < nr; i++)
	{	size_t i_var = internal_index[i];
		CPPAD_ASSERT_UNKNOWN( internal_pattern.number_elements(i_var) == 0 );
	}
# endif
	for(size_t i = 0; i < nr; i++)
	{	for(size_t j = 0; j < nc; j++)
		{	bool flag = pattern_in[i * nc + j];
			if( transpose )
				flag = pattern_in[j * nr + i];
			if( flag )
			{	size_t i_var = internal_index[i];
				CPPAD_ASSERT_UNKNOWN( i_var < internal_pattern.n_set() );
				CPPAD_ASSERT_UNKNOWN( j < nc );
				bool ignore  = zero_empty && i_var == 0;
				if( ! ignore )
					internal_pattern.add_element( i_var, j);
			}
		}
	}
	return;
}
template <class InternalSparsity>
void set_internal_sparsity(
	bool                          zero_empty       ,
	bool                          input_empty      ,
	bool                          transpose        ,
	const vector<size_t>&         internal_index   ,
	InternalSparsity&             internal_pattern ,
	const vector<bool>&           pattern_in       )
{	size_t nr = internal_index.size();
	size_t nc = internal_pattern.end();
# ifndef NDEBUG
	CPPAD_ASSERT_UNKNOWN( pattern_in.size() == nr * nc );
	if( input_empty ) for(size_t i = 0; i < nr; i++)
	{	size_t i_var = internal_index[i];
		CPPAD_ASSERT_UNKNOWN( internal_pattern.number_elements(i_var) == 0 );
	}
# endif
	for(size_t i = 0; i < nr; i++)
	{	for(size_t j = 0; j < nc; j++)
		{	bool flag = pattern_in[i * nc + j];
			if( transpose )
				flag = pattern_in[j * nr + i];
			if( flag )
			{	size_t i_var = internal_index[i];
				CPPAD_ASSERT_UNKNOWN( i_var < internal_pattern.n_set() );
				CPPAD_ASSERT_UNKNOWN( j < nc );
				bool ignore  = zero_empty && i_var == 0;
				if( ! ignore )
					internal_pattern.add_element( i_var, j);
			}
		}
	}
	return;
}
template <class InternalSparsity>
void set_internal_sparsity(
	bool                               zero_empty       ,
	bool                               input_empty      ,
	bool                               transpose        ,
	const vector<size_t>&              internal_index   ,
	InternalSparsity&                  internal_pattern ,
	const vector< std::set<size_t> >&  pattern_in       )
{	size_t nr = internal_index.size();
	size_t nc = internal_pattern.end();
# ifndef NDEBUG
	if( input_empty ) for(size_t i = 0; i < nr; i++)
	{	size_t i_var = internal_index[i];
		CPPAD_ASSERT_UNKNOWN( internal_pattern.number_elements(i_var) == 0 );
	}
# endif
	if( transpose )
	{	CPPAD_ASSERT_UNKNOWN( pattern_in.size() == nc );
		for(size_t j = 0; j < nc; j++)
		{	std::set<size_t>::const_iterator itr( pattern_in[j].begin() );
			while( itr != pattern_in[j].end() )
			{	size_t i = *itr;
				size_t i_var = internal_index[i];
				CPPAD_ASSERT_UNKNOWN( i_var < internal_pattern.n_set() );
				CPPAD_ASSERT_UNKNOWN( j < nc );
				bool ignore  = zero_empty && i_var == 0;
				if( ! ignore )
					internal_pattern.add_element( i_var, j);
				++itr;
			}
		}
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( pattern_in.size() == nr );
		for(size_t i = 0; i < nr; i++)
		{	std::set<size_t>::const_iterator itr( pattern_in[i].begin() );
			while( itr != pattern_in[i].end() )
			{	size_t j = *itr;
				size_t i_var = internal_index[i];
				CPPAD_ASSERT_UNKNOWN( i_var < internal_pattern.n_set() );
				CPPAD_ASSERT_UNKNOWN( j < nc );
				bool ignore  = zero_empty && i_var == 0;
				if( ! ignore )
					internal_pattern.add_element( i_var, j);
				++itr;
			}
		}
	}
	return;
}
// ---------------------------------------------------------------------------
/*!
Get sparsity pattern for a sub-set of variables

\tparam SizeVector
The type used for index sparsity patterns. This is a simple vector
with elements of type size_t.

\tparam InternalSparsitiy
The type used for intenal sparsity patterns. This can be either
sparse_pack or sparse_list.

\param transpose
If this is true, pattern_out is transposed.

\param internal_index
If transpose is false (true)
this is the mapping from row (column) an index in pattern_out
to the corresponding row index in internal_pattern.

\param internal_pattern
This is the internal sparsity pattern.

\param pattern_out
The input value of pattern_out does not matter.
Upon return it is an index sparsity pattern for each of the variables
in internal_index, or its transpose, depending on the value of transpose.
*/
template <class SizeVector, class InternalSparsity>
void get_internal_sparsity(
	bool                          transpose         ,
	const vector<size_t>&         internal_index    ,
	const InternalSparsity&       internal_pattern  ,
	sparse_rc<SizeVector>&        pattern_out        )
{	typedef typename InternalSparsity::const_iterator iterator;
	// number variables
	size_t nr = internal_index.size();
	// column size of interanl sparstiy pattern
	size_t nc = internal_pattern.end();
	// determine nnz, the number of possibly non-zero index pairs
	size_t nnz = 0;
	for(size_t i = 0; i < nr; i++)
	{	CPPAD_ASSERT_UNKNOWN( internal_index[i] < internal_pattern.n_set() );
		iterator itr(internal_pattern, internal_index[i]);
		size_t j = *itr;
		while( j < nc )
		{	++nnz;
			j = *(++itr);
		}
	}
	// transposed
	if( transpose )
	{	pattern_out.resize(nc, nr, nnz);
		//
		size_t k = 0;
		for(size_t i = 0; i < nr; i++)
		{	iterator itr(internal_pattern, internal_index[i]);
			size_t j = *itr;
			while( j < nc )
			{	pattern_out.set(k++, j, i);
				j = *(++itr);
			}
		}
		return;
	}
	// not transposed
	pattern_out.resize(nr, nc, nnz);
	//
	size_t k = 0;
	for(size_t i = 0; i < nr; i++)
	{	iterator itr(internal_pattern, internal_index[i]);
		size_t j = *itr;
		while( j < nc )
		{	pattern_out.set(k++, i, j);
			j = *(++itr);
		}
	}
	return;
}
template <class InternalSparsity>
void get_internal_sparsity(
	bool                          transpose         ,
	const vector<size_t>&         internal_index    ,
	const InternalSparsity&       internal_pattern  ,
	vectorBool&                   pattern_out       )
{	typedef typename InternalSparsity::const_iterator iterator;
	// number variables
	size_t nr = internal_index.size();
	//
	// column size of interanl sparstiy pattern
	size_t nc = internal_pattern.end();
	//
	pattern_out.resize(nr * nc);
	for(size_t ij = 0; ij < nr * nc; ij++)
		pattern_out[ij] = false;
	//
	for(size_t i = 0; i < nr; i++)
	{	CPPAD_ASSERT_UNKNOWN( internal_index[i] < internal_pattern.n_set() );
		iterator itr(internal_pattern, internal_index[i]);
		size_t j = *itr;
		while( j < nc )
		{	if( transpose )
				pattern_out[j * nr + i] = true;
			else
				pattern_out[i * nc + j] = true;
			j = *(++itr);
		}
	}
	return;
}
template <class InternalSparsity>
void get_internal_sparsity(
	bool                          transpose         ,
	const vector<size_t>&         internal_index    ,
	const InternalSparsity&       internal_pattern  ,
	vector<bool>&                 pattern_out       )
{	typedef typename InternalSparsity::const_iterator iterator;
	// number variables
	size_t nr = internal_index.size();
	//
	// column size of interanl sparstiy pattern
	size_t nc = internal_pattern.end();
	//
	pattern_out.resize(nr * nc);
	for(size_t ij = 0; ij < nr * nc; ij++)
		pattern_out[ij] = false;
	//
	for(size_t i = 0; i < nr; i++)
	{	CPPAD_ASSERT_UNKNOWN( internal_index[i] < internal_pattern.n_set() );
		iterator itr(internal_pattern, internal_index[i]);
		size_t j = *itr;
		while( j < nc )
		{	if( transpose )
				pattern_out[j * nr + i] = true;
			else
				pattern_out[i * nc + j] = true;
			j = *(++itr);
		}
	}
	return;
}
template <class InternalSparsity>
void get_internal_sparsity(
	bool                          transpose         ,
	const vector<size_t>&         internal_index    ,
	const InternalSparsity&       internal_pattern  ,
	vector< std::set<size_t> >&   pattern_out       )
{	typedef typename InternalSparsity::const_iterator iterator;
	// number variables
	size_t nr = internal_index.size();
	//
	// column size of interanl sparstiy pattern
	size_t nc = internal_pattern.end();
	//
	if( transpose )
		pattern_out.resize(nc);
	else
		pattern_out.resize(nr);
	for(size_t k = 0; k < pattern_out.size(); k++)
		pattern_out[k].clear();
	//
	for(size_t i = 0; i < nr; i++)
	{	CPPAD_ASSERT_UNKNOWN( internal_index[i] < internal_pattern.n_set() );
		iterator itr(internal_pattern, internal_index[i]);
		size_t j = *itr;
		while( j < nc )
		{	if( transpose )
				pattern_out[j].insert(i);
			else
				pattern_out[i].insert(j);
			j = *(++itr);
		}
	}
	return;
}



} } // END_CPPAD_LOCAL_NAMESPACE

# endif
