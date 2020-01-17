// $Id: cppad_colpack.hpp 3845 2016-11-19 01:50:47Z bradbell $
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
# ifndef CPPAD_LOCAL_CPPAD_COLPACK_HPP
# define CPPAD_LOCAL_CPPAD_COLPACK_HPP
# if CPPAD_HAS_COLPACK

namespace CppAD { namespace local { // BEGIN_CPPAD_LOCAL_NAMESPACE
/*!
\file cppad_colpack.hpp
External interface to Colpack routines used by cppad.
*/
// ---------------------------------------------------------------------------
/*!
Link from CppAD to ColPack used for general sparse matrices.

This CppAD library routine is necessary because
<code>ColPack/ColPackHeaders.h</code> has a
<code>using namespace std</code> at the global level.

\param m [in]
is the number of rows in the sparse matrix

\param n [in]
is the nubmer of columns in the sparse matrix.

\param adolc_pattern [in]
This vector has size \c m,
<code>adolc_pattern[i][0]</code> is the number of non-zeros in row \c i.
For <code>j = 1 , ... , adolc_sparsity[i]<code>,
<code>adolc_pattern[i][j]</code> is the column index (base zero) for the
non-zeros in row \c i.

\param color [out]
is a vector with size \c m.
The input value of its elements does not matter.
Upon return, it is a coloring for the rows of the sparse matrix.
\n
\n
If for some \c i, <code>color[i] == m</code>, then
<code>adolc_pattern[i][0] == 0</code>.
Otherwise, <code>color[i] < m</code>.
\n
\n
Suppose two differen rows, <code>i != r</code> have the same color.
It follows that for all column indices \c j;
it is not the case that both
<code>(i, j)</code> and <code>(r, j)</code> appear in the sparsity pattern.
\n
\n
This routine tries to minimize, with respect to the choice of colors,
the number of colors.
*/
extern void cppad_colpack_general(
	      CppAD::vector<size_t>&         color         ,
	size_t                               m             ,
	size_t                               n             ,
	const CppAD::vector<unsigned int*>&  adolc_pattern
);

/*!
Link from CppAD to ColPack used for symmetric sparse matrices
(not yet used or tested).

This CppAD library routine is necessary because
<code>ColPack/ColPackHeaders.h</code> has a
<code>using namespace std</code> at the global level.

\param n [in]
is the nubmer of rows and columns in the symmetric sparse matrix.

\param adolc_pattern [in]
This vector has size \c n,
<code>adolc_pattern[i][0]</code> is the number of non-zeros in row \c i.
For <code>j = 1 , ... , adolc_sparsity[i]<code>,
<code>adolc_pattern[i][j]</code> is the column index (base zero) for the
non-zeros in row \c i.

\param color [out]
The input value of its elements does not matter.
Upon return, it is a coloring for the rows of the sparse matrix.
The properties of this coloring have not yet been determined; see
Efficient Computation of Sparse Hessians Using Coloring
and Automatic Differentiation (pdf/ad/gebemedhin14.pdf)
*/
extern void cppad_colpack_symmetric(
	      CppAD::vector<size_t>&         color         ,
	size_t                               n             ,
	const CppAD::vector<unsigned int*>&  adolc_pattern
);

} } // END_CPPAD_LOCAL_NAMESPACE

# endif
# endif

