# ifndef CPPAD_UTILITY_INDEX_SORT_HPP
# define CPPAD_UTILITY_INDEX_SORT_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin index_sort$$
$spell
	cppad.hpp
	ind
	const
$$

$section Returns Indices that Sort a Vector$$
$mindex index_sort$$


$head Syntax$$
$codei%# include <cppad/utility/index_sort.hpp>
%$$
$codei%index_sort(%keys%, %ind%)%$$

$head keys$$
The argument $icode keys$$ has prototype
$codei%
	const %VectorKey%& %keys%
%$$
where $icode VectorKey$$ is
a $cref SimpleVector$$ class with elements that support the $code <$$
operation.

$head ind$$
The argument $icode ind$$ has prototype
$codei%
	%VectorSize%& %ind%
%$$
where $icode VectorSize$$ is
a $cref SimpleVector$$ class with elements of type $code size_t$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$subhead Input$$
The size of $icode ind$$ must be the same as the size of $icode keys$$
and the value of its input elements does not matter.

$subhead Return$$
Upon return, $icode ind$$ is a permutation of the set of indices
that yields increasing order for $icode keys$$.
In other words, for all $icode%i% != %j%$$,
$codei%
	%ind%[%i%] != %ind%[%j%]
%$$
and for $icode%i% = 0 , %...% , %size%-2%$$,
$codei%
	( %keys%[ %ind%[%i%+1] ] < %keys%[ %ind%[%i%] ] ) == false
%$$


$head Example$$
$children%
	example/utility/index_sort.cpp
%$$
The file $cref index_sort.cpp$$ contains an example
and test of this routine.
It return true if it succeeds and false otherwise.

$end
*/
# include <algorithm>
# include <cppad/utility/thread_alloc.hpp>
# include <cppad/utility/check_simple_vector.hpp>
# include <cppad/core/define.hpp>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
/*!
\file index_sort.hpp
File used to implement the CppAD index sort utility
*/

/*!
Helper class used by index_sort
*/
template <class Compare>
class index_sort_element {
private:
	/// key used to determine position of this element
	Compare key_;
	/// index vlaue corresponding to this key
	size_t  index_;
public:
	/// operator requried by std::sort
	bool operator<(const index_sort_element& other) const
	{	return key_ < other.key_; }
	/// set the key for this element
	void set_key(const Compare& value)
	{	key_ = value; }
	/// set the index for this element
	void set_index(const size_t& index)
	{	index_ = index; }
	/// get the key for this element
	Compare get_key(void) const
	{	return key_; }
	/// get the index for this element
	size_t get_index(void) const
	{	return index_; }
};

/*!
Compute the indices that sort a vector of keys

\tparam VectorKey
Simple vector type that deterimene the sorting order by \c < operator
on its elements.

\tparam VectorSize
Simple vector type with elements of \c size_t
that is used to return index values.

\param keys [in]
values that determine the sorting order.

\param ind [out]
must have the same size as \c keys.
The input value of its elements does not matter.
The output value of its elements satisfy
\code
( keys[ ind[i] ] < keys[ ind[i+1] ] ) == false
\endcode
*/
template <class VectorKey, class VectorSize>
void index_sort(const VectorKey& keys, VectorSize& ind)
{	typedef typename VectorKey::value_type Compare;
	CheckSimpleVector<size_t, VectorSize>();

	typedef index_sort_element<Compare> element;

	CPPAD_ASSERT_KNOWN(
		size_t(keys.size()) == size_t(ind.size()),
		"index_sort: vector sizes do not match"
	);

	size_t size_work = size_t(keys.size());
	size_t size_out;
	element* work =
		thread_alloc::create_array<element>(size_work, size_out);

	// copy initial order into work
	size_t i;
	for(i = 0; i < size_work; i++)
	{	work[i].set_key( keys[i] );
		work[i].set_index( i );
	}

	// sort the work array
	std::sort(work, work+size_work);

	// copy the indices to the output vector
	for(i = 0; i < size_work; i++)
		ind[i] = work[i].get_index();

	// we are done with this work array
	thread_alloc::delete_array(work);

	return;
}

} // END_CPPAD_NAMESPACE

# endif
