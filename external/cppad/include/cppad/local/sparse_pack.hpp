# ifndef CPPAD_LOCAL_SPARSE_PACK_HPP
# define CPPAD_LOCAL_SPARSE_PACK_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
# include <cppad/core/cppad_assert.hpp>
# include <cppad/local/pod_vector.hpp>

namespace CppAD { namespace local { // BEGIN_CPPAD_LOCAL_NAMESPACE
/*!
\file sparse_pack.hpp
Vector of sets of positive integers stored as a packed array of bools.
*/

// ==========================================================================
/*!
Vector of sets of postivie integers, each set stored as a packed boolean array.
*/

class sparse_pack_const_iterator;
class sparse_pack {
	friend class sparse_pack_const_iterator;
private:
	/// Type used to pack elements (should be the same as corresponding
	/// typedef in multiple_n_bit() in test_more/sparse_hacobian.cpp)
	typedef size_t Pack;
	/// Number of bits per Pack value
	const size_t n_bit_;
	/// Number of sets that we are representing
	/// (set by constructor and resize).
	size_t n_set_;
	/// Possible elements in each set are 0, 1, ..., end_ - 1
	/// (set by constructor and resize).
	size_t end_;
	/// Number of \c Pack values necessary to represent \c end_ bits.
	/// (set by constructor and resize).
	size_t n_pack_;
	/// Data for all the sets.
	pod_vector<Pack>  data_;
public:
	/// declare a const iterator
	typedef sparse_pack_const_iterator const_iterator;

	// -----------------------------------------------------------------
	/*! Default constructor (no sets)
	*/
	sparse_pack(void) :
	n_bit_( std::numeric_limits<Pack>::digits ),
	n_set_(0)      ,
	end_(0)        ,
	n_pack_(0)
	{ }
	// -----------------------------------------------------------------
	/*! Make use of copy constructor an error

	\param v
	vector that we are attempting to make a copy of.
	*/
	sparse_pack(const sparse_pack& v) :
	n_bit_( std::numeric_limits<Pack>::digits )
	{	// Error:
		// Probably a sparse_pack argument has been passed by value
		CPPAD_ASSERT_UNKNOWN(0);
	}
	// -----------------------------------------------------------------
	/*! Destructor
	*/
	~sparse_pack(void)
	{ }
	// -----------------------------------------------------------------
	/*! Change number of sets, set end, and initialize all sets as empty

	If \c n_set_in is zero, any memory currently allocated for this object
	is freed. Otherwise, new memory may be allocated for the sets (if needed).

	\param n_set_in
	is the number of sets in this vector of sets.

	\param end_in
	is the maximum element plus one. The minimum element is 0 and
	end must be greater than zero (unless n_set is also zero).
	*/
	void resize(size_t n_set_in, size_t end_in)
	{	CPPAD_ASSERT_UNKNOWN( n_set_in == 0 || 0 < end_in );
		n_set_          = n_set_in;
		end_            = end_in;
		if( n_set_ == 0 )
		{	data_.free();
			return;
		}
		// now start a new vector with empty sets
		Pack zero(0);
		data_.erase();

		n_pack_         = ( 1 + (end_ - 1) / n_bit_ );
		size_t i        = n_set_ * n_pack_;

		if( i > 0 )
		{	data_.extend(i);
			while(i--)
				data_[i] = zero;
		}
	}
	// -----------------------------------------------------------------
	/*!
	Count number of elements in a set.

	\param index
	is the index in of the set we are counting the elements of.
	*/
	size_t number_elements(size_t index) const
	{	static Pack one(1);
		CPPAD_ASSERT_UNKNOWN( index < n_set_ );
		size_t count  = 0;
		for(size_t k = 0; k < n_pack_; k++)
		{	Pack   unit = data_[ index * n_pack_ + k ];
			Pack   mask = one;
			size_t n    = std::min(n_bit_, end_ - n_bit_ * k);
			for(size_t bit = 0; bit < n; bit++)
			{	CPPAD_ASSERT_UNKNOWN( mask > one || bit == 0);
				if( mask & unit )
					++count;
				mask = mask << 1;
			}
		}
		return count;
	}
	// -----------------------------------------------------------------
	/*! Add one element to a set.

	\param index
	is the index for this set in the vector of sets.

	\param element
	is the element we are adding to the set.

	\par Checked Assertions
	\li index    < n_set_
	\li element  < end_
	*/
	void add_element(size_t index, size_t element)
	{	static Pack one(1);
		CPPAD_ASSERT_UNKNOWN( index   < n_set_ );
		CPPAD_ASSERT_UNKNOWN( element < end_ );
		size_t j  = element / n_bit_;
		size_t k  = element - j * n_bit_;
		Pack mask = one << k;
		data_[ index * n_pack_ + j] |= mask;
	}
	// -----------------------------------------------------------------
	/*! Is an element of a set.

	\param index
	is the index for this set in the vector of sets.

	\param element
	is the element we are checking to see if it is in the set.

	\par Checked Assertions
	\li index    < n_set_
	\li element  < end_
	*/
	bool is_element(size_t index, size_t element) const
	{	static Pack one(1);
		static Pack zero(0);
		CPPAD_ASSERT_UNKNOWN( index   < n_set_ );
		CPPAD_ASSERT_UNKNOWN( element < end_ );
		size_t j  = element / n_bit_;
		size_t k  = element - j * n_bit_;
		Pack mask = one << k;
		return (data_[ index * n_pack_ + j] & mask) != zero;
	}
	// -----------------------------------------------------------------
	/*! Assign the empty set to one of the sets.

	\param target
	is the index of the set we are setting to the empty set.

	\par Checked Assertions
	\li target < n_set_
	*/
	void clear(size_t target)
	{	// value with all its bits set to false
		static Pack zero(0);
		CPPAD_ASSERT_UNKNOWN( target < n_set_ );
		size_t t = target * n_pack_;

		size_t j = n_pack_;
		while(j--)
			data_[t++] = zero;
	}
	// -----------------------------------------------------------------
	/*! Assign one set equal to another set.

	\param this_target
	is the index (in this \c sparse_pack object) of the set being assinged.

	\param other_value
	is the index (in the other \c sparse_pack object) of the
	that we are using as the value to assign to the target set.

	\param other
	is the other \c sparse_pack object (which may be the same as this
	\c sparse_pack object).

	\par Checked Assertions
	\li this_target  < n_set_
	\li other_value  < other.n_set_
	\li n_pack_     == other.n_pack_
	*/
	void assignment(
		size_t               this_target  ,
		size_t               other_value  ,
		const sparse_pack&   other        )
	{	CPPAD_ASSERT_UNKNOWN( this_target  <   n_set_        );
		CPPAD_ASSERT_UNKNOWN( other_value  <   other.n_set_  );
		CPPAD_ASSERT_UNKNOWN( n_pack_      ==  other.n_pack_ );
		size_t t = this_target * n_pack_;
		size_t v = other_value * n_pack_;

		size_t j = n_pack_;
		while(j--)
			data_[t++] = other.data_[v++];
	}

	// -----------------------------------------------------------------
	/*! Assing a set equal to the union of two other sets.

	\param this_target
	is the index (in this \c sparse_pack object) of the set being assinged.

	\param this_left
	is the index (in this \c sparse_pack object) of the
	left operand for the union operation.
	It is OK for \a this_target and \a this_left to be the same value.

	\param other_right
	is the index (in the other \c sparse_pack object) of the
	right operand for the union operation.
	It is OK for \a this_target and \a other_right to be the same value.

	\param other
	is the other \c sparse_pack object (which may be the same as this
	\c sparse_pack object).

	\par Checked Assertions
	\li this_target <  n_set_
	\li this_left   <  n_set_
	\li other_right <  other.n_set_
	\li n_pack_     == other.n_pack_
	*/
	void binary_union(
		size_t                  this_target  ,
		size_t                  this_left    ,
		size_t                  other_right  ,
		const sparse_pack&      other        )
	{	CPPAD_ASSERT_UNKNOWN( this_target < n_set_         );
		CPPAD_ASSERT_UNKNOWN( this_left   < n_set_         );
		CPPAD_ASSERT_UNKNOWN( other_right < other.n_set_   );
		CPPAD_ASSERT_UNKNOWN( n_pack_    ==  other.n_pack_ );

		size_t t = this_target * n_pack_;
		size_t l  = this_left  * n_pack_;
		size_t r  = other_right * n_pack_;

		size_t j = n_pack_;
		while(j--)
			data_[t++] = ( data_[l++] | other.data_[r++] );
	}
	// -----------------------------------------------------------------
	/*! Assing a set equal to the intersection of two other sets.

	\param this_target
	is the index (in this \c sparse_pack object) of the set being assinged.

	\param this_left
	is the index (in this \c sparse_pack object) of the
	left operand for the intersection operation.
	It is OK for \a this_target and \a this_left to be the same value.

	\param other_right
	is the index (in the other \c sparse_pack object) of the
	right operand for the intersection operation.
	It is OK for \a this_target and \a other_right to be the same value.

	\param other
	is the other \c sparse_pack object (which may be the same as this
	\c sparse_pack object).

	\par Checked Assertions
	\li this_target <  n_set_
	\li this_left   <  n_set_
	\li other_right <  other.n_set_
	\li n_pack_     == other.n_pack_
	*/
	void binary_intersection(
		size_t                  this_target  ,
		size_t                  this_left    ,
		size_t                  other_right  ,
		const sparse_pack&      other        )
	{	CPPAD_ASSERT_UNKNOWN( this_target < n_set_         );
		CPPAD_ASSERT_UNKNOWN( this_left   < n_set_         );
		CPPAD_ASSERT_UNKNOWN( other_right < other.n_set_   );
		CPPAD_ASSERT_UNKNOWN( n_pack_    ==  other.n_pack_ );

		size_t t = this_target * n_pack_;
		size_t l  = this_left  * n_pack_;
		size_t r  = other_right * n_pack_;

		size_t j = n_pack_;
		while(j--)
			data_[t++] = ( data_[l++] & other.data_[r++] );
	}
	// -----------------------------------------------------------------
	/*! Fetch n_set for vector of sets object.

	\return
	Number of from sets for this vector of sets object
	*/
	size_t n_set(void) const
	{	return n_set_; }
	// -----------------------------------------------------------------
	/*! Fetch end for this vector of sets object.

	\return
	is the maximum element value plus one (the minimum element value is 0).
	*/
	size_t end(void) const
	{	return end_; }
	// -----------------------------------------------------------------
	/*! Amount of memory used by this vector of sets

	\return
	The amount of memory in units of type unsigned char memory.
	*/
	size_t memory(void) const
	{	return data_.capacity() * sizeof(Pack);
	}
	/*!
	Print the vector of sets (used for debugging)
	*/
	void print(void) const;
};
// ==========================================================================
/*!
cons_iterator for one set of positive integers in a sparse_pack object.
*/
class sparse_pack_const_iterator {
private:
	/// Type used to pack elements in sparse_pack
	typedef sparse_pack::Pack Pack;

	/// data for the entire vector of sets
	const pod_vector<Pack>&  data_;

	/// Number of bits per Pack value
	const size_t             n_bit_;

	/// Number of Pack values necessary to represent end_ bits.
	const size_t             n_pack_;

	/// Possible elements in each set are 0, 1, ..., end_ - 1;
	const size_t             end_;

	/// index of this set in the vector of sets;
	const size_t             index_;

	/// value of the next element in this set
	/// (use end_ for no such element exists; i.e., past end of the set).
	size_t                   next_element_;
public:
	/// construct a const_iterator for a set in a sparse_pack object
	sparse_pack_const_iterator (const sparse_pack& pack, size_t index)
	:
	data_      ( pack.data_ )         ,
	n_bit_     ( pack.n_bit_ )        ,
	n_pack_    ( pack.n_pack_ )       ,
	end_       ( pack.end_ )          ,
	index_     ( index )
	{	static Pack one(1);
		CPPAD_ASSERT_UNKNOWN( index < pack.n_set_ );
		//
		next_element_ = 0;
		if( next_element_ < end_ )
		{	Pack check = data_[ index_ * n_pack_ + 0 ];
			if( check & one )
				return;
		}
		// element with index zero is not in this set of integers,
		// advance to first element or end
		++(*this);
	}

	/// advance to next element in this set
	sparse_pack_const_iterator& operator++(void)
	{	static Pack one(1);
		CPPAD_ASSERT_UNKNOWN( next_element_ <= end_ );
		if( next_element_ == end_ )
			return *this;
		//
		++next_element_;
		if( next_element_ == end_ )
			return *this;
		//
		// initialize packed data index
		size_t j  = next_element_ / n_bit_;

		// initialize bit index
		size_t k  = next_element_ - j * n_bit_;

		// initialize mask
		size_t mask = one << k;

		// start search at this packed value
		Pack check = data_[ index_ * n_pack_ + j ];
		//
		while( true )
		{	// check if this element is in the set
			if( check & mask )
				return *this;

			// increment next element before checking this one
			next_element_++;
			if( next_element_ == end_ )
				return *this;

			// shift mask to left one bit so corresponds to next_element_
			// (use mask <<= 1. not one << k, so compiler knows value)
			k++;
			mask <<= 1;
			CPPAD_ASSERT_UNKNOWN( k <= n_bit_ );

			// check if we must go to next packed data index
			if( k == n_bit_ )
			{	// get next packed value
				k     = 0;
				mask  = one;
				j++;
				CPPAD_ASSERT_UNKNOWN( j < n_pack_ );
				check = data_[ index_ * n_pack_ + j ];
			}
		}
		// should never get here
		CPPAD_ASSERT_UNKNOWN(false);
		return *this;
	}

	/// obtain value of this element of the set of positive integers
	/// (end_ for no such element)
	size_t operator*(void) const
	{	return next_element_; }
};
// =========================================================================
/*!
Print the vector of sets (used for debugging)
*/
inline void sparse_pack::print(void) const
{	std::cout << "sparse_pack:\n";
	for(size_t i = 0; i < n_set(); i++)
	{	std::cout << "set[" << i << "] = {";
		const_iterator itr(*this, i);
		while( *itr != end() )
		{	std::cout << *itr;
			if( *(++itr) != end() )
				std::cout << ",";
		}
		std::cout << "}\n";
	}
	return;
}

// ==========================================================================

/*!
Copy a user vector of sets sparsity pattern to an internal sparse_pack object.

\tparam VectorSet
is a simple vector with elements of type std::set<size_t>.

\param internal
The input value of sparisty does not matter.
Upon return it contains the same sparsity pattern as \c user
(or the transposed sparsity pattern).

\param user
sparsity pattern that we are placing internal.

\param n_set
number of sets (rows) in the internal sparsity pattern.

\param end
end of set value (number of columns) in the interanl sparsity pattern.

\param transpose
if true, the user sparsity patter is the transposed.

\param error_msg
is the error message to display if some values in the user sparstiy
pattern are not valid.
*/
template<class VectorSet>
void sparsity_user2internal(
	sparse_pack&            internal  ,
	const VectorSet&        user      ,
	size_t                  n_set     ,
	size_t                  end       ,
	bool                    transpose ,
	const char*             error_msg )
{	CPPAD_ASSERT_KNOWN(size_t( user.size() ) == n_set * end, error_msg );

	// size of internal sparsity pattern
	internal.resize(n_set, end);

	if( transpose )
	{	// transposed pattern case
		for(size_t j = 0; j < end; j++)
		{	for(size_t i = 0; i < n_set; i++)
			{	if( user[ j * n_set + i ] )
					internal.add_element(i, j);
			}
		}
		return;
	}
	else
	{	for(size_t i = 0; i < n_set; i++)
		{	for(size_t j = 0; j < end; j++)
			{	if( user[ i * end + j ] )
				internal.add_element(i, j);
			}
		}
	}
	return;
}

} } // END_CPPAD_LOCAL_NAMESPACE
# endif
