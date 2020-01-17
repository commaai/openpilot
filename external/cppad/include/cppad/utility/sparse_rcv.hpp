# ifndef CPPAD_UTILITY_SPARSE_RCV_HPP
# define CPPAD_UTILITY_SPARSE_RCV_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
------------------------------------------------------------------------------
$begin sparse_rcv$$
$spell
	CppAD
	nr
	nc
	const
	var
	nnz
	cppad
	hpp
	rcv
	rc
$$
$section Sparse Matrix Row, Column, Value Representation$$

$head Syntax$$
$codei%# include <cppad/utility/sparse_rcv.hpp>
%$$
$codei%sparse_rcv<%SizeVector%, %ValueVector%>  %empty%
%$$
$codei%sparse_rcv<%SizeVector%, %ValueVector%>  %matrix%(%pattern%)
%$$
$codei%target% = %matrix%
%$$
$icode%matrix%.set(%k%, %v%)
%$$
$icode%nr% = %matrix%.nr()
%$$
$icode%nc% = %matrix%.nc()
%$$
$icode%nnz% = %matrix%.nnz()
%$$
$codei%const %SizeVector%& %row%( %matrix%.row() )
%$$
$codei%const %SizeVector%& %col%( %matrix%.col() )
%$$
$codei%const %ValueVector%& %val%( %matrix%.val() )
%$$
$icode%row_major% = %matrix%.row_major()
%$$
$icode%col_major% = %matrix%.col_major()
%$$

$head SizeVector$$
We use $cref/SizeVector/sparse_rc/SizeVector/$$ to denote the
$cref SimpleVector$$ class corresponding to $icode pattern$$.

$head ValueVector$$
We use $icode ValueVector$$ to denote the
$cref SimpleVector$$ class corresponding to $icode val$$.

$head empty$$
This is an empty sparse matrix object. To be specific,
the corresponding number of rows $icode nr$$,
number of columns $icode nc$$,
and number of possibly non-zero values $icode nnz$$,
are all zero.


$head pattern$$
This argument has prototype
$codei%
	const sparse_rc<%SizeVector%>& %pattern%
%$$
It specifies the number of rows, number of columns and
the possibly non-zero entries in the $icode matrix$$.

$head matrix$$
This is a sparse matrix object with the sparsity specified by $icode pattern$$.
Only the $icode val$$ vector can be changed. All other values returned by
$icode matrix$$ are fixed during the constructor and constant there after.
The $icode val$$ vector is only changed by the constructor
and the $code set$$ function.
There is one exception to the rule, where $icode matrix$$ corresponds
to $icode target$$ for an assignment statement.

$head target$$
The target of the assignment statement must have prototype
$codei%
	sparse_rcv<%SizeVector%, %ValueVector%>  %target%
%$$
After this assignment statement, $icode target$$ is an independent copy
of $icode matrix$$; i.e. it has all the same values as $icode matrix$$
and changes to $icode target$$ do not affect $icode matrix$$.

$head nr$$
This return value has prototype
$codei%
	size_t %nr%
%$$
and is the number of rows in $icode matrix$$.

$head nc$$
This argument and return value has prototype
$codei%
	size_t %nc%
%$$
and is the number of columns in $icode matrix$$.

$head nnz$$
We use the notation $icode nnz$$ to denote the number of
possibly non-zero entries in $icode matrix$$.

$head set$$
This function sets the value
$codei%
	%val%[%k%] = %v%
%$$

$subhead k$$
This argument has type
$codei%
	size_t %k%
%$$
and must be less than $icode nnz$$.

$subhead v$$
This argument has type
$codei%
	const %ValueVector%::value_type& %v%
%$$
It specifies the value assigned to $icode%val%[%k%]%$$.


$head row$$
This vector has size $icode nnz$$ and
$icode%row%[%k%]%$$
is the row index of the $th k$$ possibly non-zero
element in $icode matrix$$.

$head col$$
This vector has size $icode nnz$$ and
$icode%col[%k%]%$$ is the column index of the $th k$$ possibly non-zero
element in $icode matrix$$

$head val$$
This vector has size $icode nnz$$ and
$icode%val[%k%]%$$ is value of the $th k$$ possibly non-zero entry
in the sparse matrix (the value may be zero).

$head row_major$$
This vector has prototype
$codei%
	%SizeVector% %row_major%
%$$
and its size $icode nnz$$.
It sorts the sparsity pattern in row-major order.
To be specific,
$codei%
	%col%[ %row_major%[%k%] ] <= %col%[ %row_major%[%k%+1] ]
%$$
and if $icode%col%[ %row_major%[%k%] ] == %col%[ %row_major%[%k%+1] ]%$$,
$codei%
	%row%[ %row_major%[%k%] ] < %row%[ %row_major%[%k%+1] ]
%$$
This routine generates an assert if there are two entries with the same
row and column values (if $code NDEBUG$$ is not defined).

$head col_major$$
This vector has prototype
$codei%
	%SizeVector% %col_major%
%$$
and its size $icode nnz$$.
It sorts the sparsity pattern in column-major order.
To be specific,
$codei%
	%row%[ %col_major%[%k%] ] <= %row%[ %col_major%[%k%+1] ]
%$$
and if $icode%row%[ %col_major%[%k%] ] == %row%[ %col_major%[%k%+1] ]%$$,
$codei%
	%col%[ %col_major%[%k%] ] < %col%[ %col_major%[%k%+1] ]
%$$
This routine generates an assert if there are two entries with the same
row and column values (if $code NDEBUG$$ is not defined).

$children%
	example/utility/sparse_rcv.cpp
%$$
$head Example$$
The file $cref sparse_rcv.cpp$$
contains an example and test of this class.
It returns true if it succeeds and false otherwise.

$end
*/
/*!
\file sparse_rcv.hpp
A sparse matrix class.
*/
# include <cppad/utility/sparse_rc.hpp>

namespace CppAD { // BEGIN CPPAD_NAMESPACE

/// Sparse matrices with elements of type Scalar
template <class SizeVector, class ValueVector>
class sparse_rcv {
private:
	/// sparsity pattern
	sparse_rc<SizeVector> pattern_;
	/// value_type
	typedef typename ValueVector::value_type value_type;
	/// val_[k] is the value for the k-th possibly non-zero entry in the matrix
	ValueVector    val_;
public:
	// ------------------------------------------------------------------------
	/// default constructor
	sparse_rcv(void)
	: pattern_(0, 0, 0), val_(0)
	{ }
	/// constructor
	sparse_rcv(const sparse_rc<SizeVector>& pattern )
	:
	pattern_(pattern)    ,
	val_(pattern_.nnz())
	{ }
	/// assignment
	void operator=(const sparse_rcv& matrix)
	{	pattern_ = matrix.pattern_;
		// simple vector assignment requires vectors to have same size
		val_.resize( matrix.nnz() );
		val_ = matrix.val();
	}
	// ------------------------------------------------------------------------
	void set(size_t k, const value_type& v)
	{	CPPAD_ASSERT_KNOWN(
			pattern_.nnz(),
			"The index k is not less than nnz in sparse_rcv::set"
		);
		val_[k] = v;
	}
	/// number of rows in matrix
	size_t nr(void) const
	{	return pattern_.nr(); }
	/// number of columns in matrix
	size_t nc(void) const
	{	return pattern_.nc(); }
	/// number of possibly non-zero elements in matrix
	size_t nnz(void) const
	{	return pattern_.nnz(); }
	/// row indices
	const SizeVector& row(void) const
	{	return pattern_.row(); }
	/// column indices
	const SizeVector& col(void) const
	{	return pattern_.col(); }
	/// value for possibly non-zero elements
	const ValueVector& val(void) const
	{	return val_; }
	/// row-major order
	SizeVector row_major(void) const
	{	return pattern_.row_major(); }
	/// column-major indices
	SizeVector col_major(void) const
	{	return pattern_.col_major(); }
};

} // END_CPPAD_NAMESPACE

# endif
