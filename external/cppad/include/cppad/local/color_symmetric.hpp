# ifndef CPPAD_LOCAL_COLOR_SYMMETRIC_HPP
# define CPPAD_LOCAL_COLOR_SYMMETRIC_HPP

# include <cppad/configure.hpp>
# include <cppad/local/cppad_colpack.hpp>

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
\file color_symmetric.hpp
Coloring algorithm for a symmetric sparse matrix.
*/
// --------------------------------------------------------------------------
/*!
CppAD algorithm for determining which rows of a symmetric sparse matrix can be
computed together.

\tparam VectorSize
is a simple vector class with elements of type size_t.

\tparam VectorSet
is an unspecified type with the exception that it must support the
operations under pattern and the following operations where
p is a VectorSet object:
\n
<code>VectorSet p</code>
Constructs a new vector of sets object.
\n
<code>p.resize(ns, ne)</code>
resizes \c p to ns sets with elements between zero and \c ne.
All of the sets are initially empty.
\n
<code>p.add_element(s, e)</code>
add element \c e to set with index \c s.

\param pattern [in]
Is a representation of the sparsity pattern for the matrix.
\n
<code>m = pattern.n_set()</code>
\n
sets m to the number of rows (and columns) in the sparse matrix.
All of the row indices are less than this value.
\n
<code>n = pattern.end()</code>
\n
sets n to the number of columns in the sparse matrix
(which must be equal to the number of rows).
All of the column indices are less than this value.
\n
<code>VectorSet::const_iterator itr(pattern, i)</code>
constructs an iterator that starts iterating over
columns in the i-th row of the sparsity pattern.
\n
<code>j = *itr</code>
Sets j to the next possibly non-zero column.
\n
<code>++itr</code>
Advances to the next possibly non-zero column.
\n

\param row [in/out]
is a vector specifying which row indices to compute.

\param col [in/out]
is a vector, with the same size as row,
that specifies which column indices to compute.
\n
\n
Input:
For each  valid index \c k, the index pair
<code>(row[k], col[k])</code> must be present in the sparsity pattern.
It may be that some entries in the sparsity pattern do not need to be computed;
i.e, do not appear in the set of
<code>(row[k], col[k])</code> entries.
\n
\n
Output:
On output, some of row and column indices may have been swapped
\code
	std::swap( row[k], col[k] )
\endcode
So the the the color for row[k] can be used to compute entry
(row[k], col[k]).

\param color [out]
is a vector with size m.
The input value of its elements does not matter.
Upon return, it is a coloring for the rows of the sparse matrix.
Note that if color[i] == m, then there is no index k for which
row[k] == i (for the return value of row).
\n
\n
Fix any (i, j) in the sparsity pattern.
Suppose that there is a row index i1 with
i1 != i, color[i1] == color[i] and (i1, j) is in the sparsity pattern.
If follows that for all j1 with
j1 != j and color[j1] == color[j],
(j1, i ) is not in the sparsity pattern.
\n
\n
This routine tries to minimize, with respect to the choice of colors,
the maximum, with respect to k, of <code>color[ row[k] ]</code>.
*/
template <class VectorSet>
void color_symmetric_cppad(
	const VectorSet&        pattern   ,
	CppAD::vector<size_t>&  row       ,
	CppAD::vector<size_t>&  col       ,
	CppAD::vector<size_t>&  color     )
{	size_t o1, o2, i1, i2, j1, j2, k1, c1, c2;

	size_t K = row.size();
	size_t m = pattern.n_set();
	CPPAD_ASSERT_UNKNOWN( m == pattern.end() );
	CPPAD_ASSERT_UNKNOWN( color.size() == m );
	CPPAD_ASSERT_UNKNOWN( col.size()   == K );

	// row, column pairs that appear in ( row[k], col[k] )
	CppAD::vector< std::set<size_t> > pair_needed(m);
	std::set<size_t>::iterator itr1, itr2;
	for(k1 = 0;  k1 < K; k1++)
	{	CPPAD_ASSERT_UNKNOWN( pattern.is_element(row[k1], col[k1]) );
		pair_needed[ row[k1] ].insert( col[k1] );
		pair_needed[ col[k1] ].insert( row[k1] );
	}

	// order the rows decending by number of pairs needed
	CppAD::vector<size_t> key(m), order2row(m);
	for(i1 = 0; i1 < m; i1++)
	{	CPPAD_ASSERT_UNKNOWN( pair_needed[i1].size() <= m );
		key[i1] = m - pair_needed[i1].size();
	}
	CppAD::index_sort(key, order2row);

	// mapping from order index to row index
	CppAD::vector<size_t> row2order(m);
	for(o1 = 0; o1 < m; o1++)
		row2order[ order2row[o1] ] = o1;

	// initial coloring
	color.resize(m);
	c1 = 0;
	for(o1 = 0; o1 < m; o1++)
	{	i1 = order2row[o1];
		if( pair_needed[i1].empty() )
			color[i1] = m;
		else
			color[i1] = c1++;
	}

	// which colors are forbidden for this row
	CppAD::vector<bool> forbidden(m);

	// must start with row zero so that we remove results computed for it
	for(o1 = 0; o1 < m; o1++) // for each row that appears (in order)
	if( color[ order2row[o1] ] < m )
	{	i1 = order2row[o1];
		c1 = color[i1];

		// initial all colors as ok for this row
		// (value of forbidden for c > c1 does not matter)
		for(c2 = 0; c2 <= c1; c2++)
			forbidden[c2] = false;

		// -----------------------------------------------------
		// Forbid grouping with rows that would destroy results that are
		// needed for this row.
		itr1 = pair_needed[i1].begin();
		while( itr1 != pair_needed[i1].end() )
		{	// entry (i1, j1) is needed for this row
			j1 = *itr1;

			// Forbid rows i2 != i1 that have non-zero sparsity at (i2, j1).
			// Note that this is the same as non-zero sparsity at (j1, i2)
			typename VectorSet::const_iterator pattern_itr(pattern, j1);
			i2 = *pattern_itr;
			while( i2 != pattern.end() )
			{	c2 = color[i2];
				if( c2 < c1 )
					forbidden[c2] = true;
				i2 = *(++pattern_itr);
			}
			itr1++;
		}
		// -----------------------------------------------------
		// Forbid grouping with rows that this row would destroy results for
		for(o2 = 0; o2 < o1; o2++)
		{	i2 = order2row[o2];
			c2 = color[i2];
			itr2 = pair_needed[i2].begin();
			while( itr2 != pair_needed[i2].end() )
			{	j2 = *itr2;
				// row i2 needs pair (i2, j2).
				// Forbid grouping with i1 if (i1, j2) has non-zero sparsity
				if( pattern.is_element(i1, j2) )
					forbidden[c2] = true;
				itr2++;
			}
		}

		// pick the color with smallest index
		c2 = 0;
		while( forbidden[c2] )
		{	c2++;
			CPPAD_ASSERT_UNKNOWN( c2 <= c1 );
		}
		color[i1] = c2;

		// no longer need results that are computed by this row
		itr1 = pair_needed[i1].begin();
		while( itr1 != pair_needed[i1].end() )
		{	j1 = *itr1;
			if( row2order[j1] > o1 )
			{	itr2 = pair_needed[j1].find(i1);
				if( itr2 != pair_needed[j1].end() )
				{	pair_needed[j1].erase(itr2);
					if( pair_needed[j1].empty() )
						color[j1] = m;
				}
			}
			itr1++;
		}
	}

	// determine which sparsity entries need to be reflected
	for(k1 = 0; k1 < row.size(); k1++)
	{	i1   = row[k1];
		j1   = col[k1];
		itr1 = pair_needed[i1].find(j1);
		if( itr1 == pair_needed[i1].end() )
		{	row[k1] = j1;
			col[k1] = i1;
# ifndef NDEBUG
			itr1 = pair_needed[j1].find(i1);
			CPPAD_ASSERT_UNKNOWN( itr1 != pair_needed[j1].end() );
# endif
		}
	}
	return;
}

// --------------------------------------------------------------------------
/*!
Colpack algorithm for determining which rows of a symmetric sparse matrix
can be computed together.

\copydetails CppAD::local::color_symmetric_cppad
*/
template <class VectorSet>
void color_symmetric_colpack(
	const VectorSet&        pattern   ,
	CppAD::vector<size_t>&  row       ,
	CppAD::vector<size_t>&  col       ,
	CppAD::vector<size_t>&  color     )
{
# if ! CPPAD_HAS_COLPACK
	CPPAD_ASSERT_UNKNOWN(false);
	return;
# else
	size_t i, j, k;
	size_t m = pattern.n_set();
	CPPAD_ASSERT_UNKNOWN( m == pattern.end() );
	CPPAD_ASSERT_UNKNOWN( row.size() == col.size() );

	// Determine number of non-zero entries in each row
	CppAD::vector<size_t> n_nonzero(m);
	size_t n_nonzero_total = 0;
	for(i = 0; i < m; i++)
	{	n_nonzero[i] = 0;
		typename VectorSet::const_iterator pattern_itr(pattern, i);
		j = *pattern_itr;
		while( j != pattern.end() )
		{	n_nonzero[i]++;
			j = *(++pattern_itr);
		}
		n_nonzero_total += n_nonzero[i];
	}

	// Allocate memory and fill in Adolc sparsity pattern
	CppAD::vector<unsigned int*> adolc_pattern(m);
	CppAD::vector<unsigned int>  adolc_memory(m + n_nonzero_total);
	size_t i_memory = 0;
	for(i = 0; i < m; i++)
	{	adolc_pattern[i]    = adolc_memory.data() + i_memory;
		CPPAD_ASSERT_KNOWN(
			std::numeric_limits<unsigned int>::max() >= n_nonzero[i],
			"Matrix is too large for colpack"
		);
		adolc_pattern[i][0] = static_cast<unsigned int>( n_nonzero[i] );
		typename VectorSet::const_iterator pattern_itr(pattern, i);
		j = *pattern_itr;
		k = 1;
		while(j != pattern.end() )
		{
			CPPAD_ASSERT_KNOWN(
				std::numeric_limits<unsigned int>::max() >= j,
				"Matrix is too large for colpack"
			);
			adolc_pattern[i][k++] = static_cast<unsigned int>( j );
			j = *(++pattern_itr);
		}
		CPPAD_ASSERT_UNKNOWN( k == 1 + n_nonzero[i] );
		i_memory += k;
	}
	CPPAD_ASSERT_UNKNOWN( i_memory == m + n_nonzero_total );

	// Must use an external routine for this part of the calculation because
	// ColPack/ColPackHeaders.h has as 'using namespace std' at global level.
	cppad_colpack_symmetric(color, m, adolc_pattern);

	// determine which sparsity entries need to be reflected
	size_t i1, i2, j1, j2, k1, k2;
	for(k1 = 0; k1 < row.size(); k1++)
	{	i1 = row[k1];
		j1 = col[k1];
		bool reflect = false;
		for(i2 = 0; i2 < m; i2++) if( (i1 != i2) & (color[i1]==color[i2]) )
		{	for(k2 = 1; k2 <= adolc_pattern[i2][0]; k2++)
			{	j2 = adolc_pattern[i2][k2];
				reflect |= (j1 == j2);
			}
		}
		if( reflect )
		{	row[k1] = j1;
			col[k1] = i1;
		}
	}
	return;
# endif // CPPAD_HAS_COLPACK
}

} } // END_CPPAD_LOCAL_NAMESPACE

# endif
