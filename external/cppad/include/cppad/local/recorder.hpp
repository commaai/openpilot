# ifndef CPPAD_LOCAL_RECORDER_HPP
# define CPPAD_LOCAL_RECORDER_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
# include <cppad/core/hash_code.hpp>
# include <cppad/local/pod_vector.hpp>

namespace CppAD { namespace local { // BEGIN_CPPAD_LOCAL_NAMESPACE
/*!
\file recorder.hpp
File used to define the recorder class.
*/

/*!
Class used to store an operation sequence while it is being recorded
(the operation sequence is copied to the player class for playback).

\tparam Base
This is an AD< \a Base > operation sequence recording; i.e.,
it records operations of type AD< \a Base >.
*/
template <class Base>
class recorder {
	friend class player<Base>;

private:
	/// operator index at which to abort recording with an error
	/// (do not abort when zero)
	size_t abort_op_index_;

	/// offset for this thread in the static hash table
	const size_t thread_offset_;

	/// Number of variables in the recording.
	size_t    num_var_rec_;

	/// Number vecad load operations (LdpOp or LdvOp) currently in recording.
	size_t	num_load_op_rec_;

	/// The operators in the recording.
	pod_vector<CPPAD_OP_CODE_TYPE> op_rec_;

	/// The VecAD indices in the recording.
	pod_vector<addr_t> vecad_ind_rec_;

	/// The argument indices in the recording
	pod_vector<addr_t> op_arg_rec_;

	/// The parameters in the recording.
	/// Note that Base may not be plain old data, so use false in consructor.
	pod_vector<Base> par_rec_;

	/// Character strings ('\\0' terminated) in the recording.
	pod_vector<char> text_rec_;
// ---------------------- Public Functions -----------------------------------
public:
	/// Default constructor
	recorder(void) :
	thread_offset_( thread_alloc::thread_num() * CPPAD_HASH_TABLE_SIZE ) ,
	num_var_rec_(0)                                      ,
	num_load_op_rec_(0)                                  ,
	op_rec_( std::numeric_limits<addr_t>::max() )        ,
	vecad_ind_rec_( std::numeric_limits<addr_t>::max() ) ,
	op_arg_rec_( std::numeric_limits<addr_t>::max() )    ,
	par_rec_( std::numeric_limits<addr_t>::max() )       ,
	text_rec_( std::numeric_limits<addr_t>::max() )
	{
		abort_op_index_ = 0;
	}

	/// Set the abort index
	void set_abort_op_index(size_t abort_op_index)
	{	abort_op_index_ = abort_op_index; }

	/// Get the abort index
	size_t get_abort_op_index(void)
	{	return abort_op_index_; }

	/// Destructor
	~recorder(void)
	{ }

	/*!
	Frees all information in recording.

	Frees the operation sequence store in this recording
	(the operation sequence is empty after this operation).
	The buffers used to store the current recording are returned
	to the system (so as to conserve on memory).
	*/
	void free(void)
	{	num_var_rec_     = 0;
		num_load_op_rec_ = 0;
		op_rec_.free();
		vecad_ind_rec_.free();
		op_arg_rec_.free();
		par_rec_.free();
		text_rec_.free();
	}
	/// Put next operator in the operation sequence.
	inline addr_t PutOp(OpCode op);
	/// Put a vecad load operator in the operation sequence (special case)
	inline addr_t PutLoadOp(OpCode op);
	/// Add a value to the end of the current vector of VecAD indices.
	inline addr_t PutVecInd(size_t vec_ind);
	/// Find or add a parameter to the current vector of parameters.
	inline addr_t PutPar(const Base &par);
	/// Put one operation argument index in the recording
	inline void PutArg(addr_t arg0);
	/// Put two operation argument index in the recording
	inline void PutArg(addr_t arg0, addr_t arg1);
	/// Put three operation argument index in the recording
	inline void PutArg(addr_t arg0, addr_t arg1, addr_t arg2);
	/// Put four operation argument index in the recording
	inline void PutArg(addr_t arg0, addr_t arg1, addr_t arg2, addr_t arg3);
	/// Put five operation argument index in the recording
	inline void PutArg(addr_t arg0, addr_t arg1, addr_t arg2, addr_t arg3,
		addr_t arg4);
	/// Put six operation argument index in the recording
	inline void PutArg(addr_t arg0, addr_t arg1, addr_t arg2, addr_t arg3,
		addr_t arg4, addr_t arg5);

	// Reserve space for a specified number of arguments
	inline size_t ReserveArg(size_t n_arg);

	// Replace an argument value
	void ReplaceArg(size_t i_arg, size_t value);

	/// Put a character string in the text for this recording.
	inline addr_t PutTxt(const char *text);

	/// Number of variables currently stored in the recording.
	size_t num_var_rec(void) const
	{	return num_var_rec_; }

	/// Number of LdpOp and LdvOp operations currently in the recording.
	size_t num_load_op_rec(void) const
	{	return num_load_op_rec_; }

	/// Number of operators currently stored in the recording.
	size_t num_op_rec(void) const
	{	return  op_rec_.size(); }

	/// Approximate amount of memory used by the recording
	size_t Memory(void) const
	{	return op_rec_.capacity()        * sizeof(CPPAD_OP_CODE_TYPE)
		     + vecad_ind_rec_.capacity() * sizeof(size_t)
		     + op_arg_rec_.capacity()    * sizeof(addr_t)
		     + par_rec_.capacity()       * sizeof(Base)
		     + text_rec_.capacity()      * sizeof(char);
	}
};

/*!
Put next operator in the operation sequence.

This sets the op code for the next operation in this recording.
This call must be followed by putting the corresponding
\verbatim
	NumArg(op)
\endverbatim
argument indices in the recording.

\param op
Is the op code corresponding to the the operation that is being
recorded (which must not be LdpOp or LdvOp).

\return
The return value is the index of the primary (last) variable
corresponding to the result of this operation.
The number of variables corresponding to the operation is given by
\verbatim
	NumRes(op)
\endverbatim
With each call to PutOp or PutLoadOp,
the return index increases by the number of variables corresponding
to the call.
This index starts at zero after the default constructor
and after each call to Erase.
*/
template <class Base>
inline addr_t recorder<Base>::PutOp(OpCode op)
{	size_t i    = op_rec_.extend(1);
	CPPAD_ASSERT_KNOWN(
		(abort_op_index_ == 0) || (abort_op_index_ != i),
		"Operator index equals abort_op_index in Independent"
	);
	op_rec_[i]  = static_cast<CPPAD_OP_CODE_TYPE>(op);
	CPPAD_ASSERT_UNKNOWN( op_rec_.size() == i + 1 );
	CPPAD_ASSERT_UNKNOWN( (op != LdpOp) & (op != LdvOp) );

	// first operator should be a BeginOp and NumRes( BeginOp ) > 0
	num_var_rec_ += NumRes(op);
	CPPAD_ASSERT_UNKNOWN( num_var_rec_ > 0 );

	// index of last variable corresponding to this operation
	// (if NumRes(op) > 0)
	CPPAD_ASSERT_KNOWN(
		(size_t) std::numeric_limits<addr_t>::max() >= num_var_rec_ - 1,
		"cppad_tape_addr_type maximum value has been exceeded"
	)

	return static_cast<addr_t>( num_var_rec_ - 1 );
}

/*!
Put next LdpOp or LdvOp operator in operation sequence (special cases).

This sets the op code for the next operation in this recording.
This call must be followed by putting the corresponding
\verbatim
	NumArg(op)
\endverbatim
argument indices in the recording.

\param op
Is the op code corresponding to the the operation that is being
recorded (which must be LdpOp or LdvOp).

\return
The return value is the index of the primary (last) variable
corresponding to the result of this operation.
The number of variables corresponding to the operation is given by
\verbatim
	NumRes(op)
\endverbatim
which must be one for this operation.
With each call to PutLoadOp or PutOp,
the return index increases by the number of variables corresponding
to this call to the call.
This index starts at zero after the default constructor
and after each call to Erase.

\par num_load_op_rec()
The return value for <code>num_load_op_rec()</code>
increases by one after each call to this function
(and starts at zero after the default constructor or Erase).
*/
template <class Base>
inline addr_t recorder<Base>::PutLoadOp(OpCode op)
{	size_t i    = op_rec_.extend(1);
	CPPAD_ASSERT_KNOWN(
		(abort_op_index_ == 0) || (abort_op_index_ != i),
		"This is the abort operator index specified by "
		"Independent(x, abort_op_index)."
	);
	op_rec_[i]  = static_cast<CPPAD_OP_CODE_TYPE>(op);
	CPPAD_ASSERT_UNKNOWN( op_rec_.size() == i + 1 );
	CPPAD_ASSERT_UNKNOWN( (op == LdpOp) | (op == LdvOp) );

	// first operator should be a BeginOp and NumRes( BeginOp ) > 0
	num_var_rec_ += NumRes(op);
	CPPAD_ASSERT_UNKNOWN( num_var_rec_ > 0 );

	// count this vecad load operation
	num_load_op_rec_++;

	// index of last variable corresponding to this operation
	// (if NumRes(op) > 0)
	CPPAD_ASSERT_KNOWN(
		(size_t) std::numeric_limits<addr_t>::max() >= num_var_rec_ - 1,
		"cppad_tape_addr_type maximum value has been exceeded"
	)
	return static_cast<addr_t>( num_var_rec_ - 1 );
}

/*!
Add a value to the end of the current vector of VecAD indices.

For each VecAD vector, this routine is used to store the length
of the vector followed by the parameter index corresponding to each
value in the vector.
This value for the elements of the VecAD vector corresponds to the
beginning of the operation sequence.

\param vec_ind
is the index to be palced at the end of the vector of VecAD indices.

\return
is the index in the vector of VecAD indices corresponding to this value.
This index starts at zero after the recorder default constructor
and after each call to Erase.
It increments by one for each call to PutVecInd..
*/
template <class Base>
inline addr_t recorder<Base>::PutVecInd(size_t vec_ind)
{	size_t i          = vecad_ind_rec_.extend(1);
	CPPAD_ASSERT_UNKNOWN( std::numeric_limits<addr_t>::max() >= vec_ind );
	vecad_ind_rec_[i] = addr_t( vec_ind );
	CPPAD_ASSERT_UNKNOWN( vecad_ind_rec_.size() == i + 1 );

	CPPAD_ASSERT_KNOWN(
		std::numeric_limits<addr_t>::max() >= i,
		"cppad_tape_addr_type maximum value has been exceeded"
	);
	return static_cast<addr_t>( i );
}

/*!
Find or add a parameter to the current vector of parameters.

\param par
is the parameter to be found or placed in the vector of parameters.

\return
is the index in the parameter vector corresponding to this parameter value.
This value is not necessarily placed at the end of the vector
(because values that are identically equal may be reused).
*/
template <class Base>
addr_t recorder<Base>::PutPar(const Base &par)
{	static size_t   hash_table[CPPAD_HASH_TABLE_SIZE * CPPAD_MAX_NUM_THREADS];
	size_t          i;
	size_t          code;

	CPPAD_ASSERT_UNKNOWN(
		thread_offset_ / CPPAD_HASH_TABLE_SIZE
		==
		thread_alloc::thread_num()
	);

	// get hash code for this value
	code = static_cast<size_t>( hash_code(par) );
	CPPAD_ASSERT_UNKNOWN( code < CPPAD_HASH_TABLE_SIZE );

	// If we have a match, return the parameter index
	i = hash_table[code + thread_offset_];
	if( i < par_rec_.size() && IdenticalEqualPar(par_rec_[i], par) )
	{	CPPAD_ASSERT_KNOWN(
			static_cast<size_t>( std::numeric_limits<addr_t>::max() ) >= i,
			"cppad_tape_addr_type maximum value has been exceeded"
		)
		return static_cast<addr_t>( i );
	}

	// place a new value in the table
	i           = par_rec_.extend(1);
	par_rec_[i] = par;
	CPPAD_ASSERT_UNKNOWN( par_rec_.size() == i + 1 );

	// make the hash code point to this new value
	hash_table[code + thread_offset_] = i;

	// return the parameter index
	CPPAD_ASSERT_KNOWN(
		static_cast<size_t>( std::numeric_limits<addr_t>::max() ) >= i,
		"cppad_tape_addr_type maximum value has been exceeded"
	)
	return static_cast<addr_t>( i );
}
// -------------------------- PutArg --------------------------------------
/*!
Prototype for putting operation argument indices in the recording.

The following syntax
\verbatim
	rec.PutArg(arg0)
	rec.PutArg(arg0, arg1)
	.
	.
	.
	rec.PutArg(arg0, arg1, ..., arg5)
\endverbatim
places the values passed to PutArg at the current end of the
operation argument indices for the recording.
\a arg0 comes before \a arg1, etc.
The proper number of operation argument indices
corresponding to the operation code op is given by
\verbatim
	NumArg(op)
\endverbatim
The number of the operation argument indices starts at zero
after the default constructor and each call to Erase.
It increases by the number of indices placed by each call to PutArg.
*/
inline void prototype_put_arg(void)
{	// This routine should not be called
	CPPAD_ASSERT_UNKNOWN(false);
}
/*!
Put one operation argument index in the recording

\param arg0
The operation argument index

\copydetails prototype_put_arg
*/
template <class Base>
inline void recorder<Base>::PutArg(addr_t arg0)
{
	size_t i       = op_arg_rec_.extend(1);
	op_arg_rec_[i] =  static_cast<addr_t>( arg0 );
	CPPAD_ASSERT_UNKNOWN( op_arg_rec_.size() == i + 1 );
}
/*!
Put two operation argument index in the recording

\param arg0
First operation argument index.

\param arg1
Second operation argument index.

\copydetails prototype_put_arg
*/
template <class Base>
inline void recorder<Base>::PutArg(addr_t arg0, addr_t arg1)
{
	size_t i         = op_arg_rec_.extend(2);
	op_arg_rec_[i++] =  static_cast<addr_t>( arg0 );
	op_arg_rec_[i]   =  static_cast<addr_t>( arg1 );
	CPPAD_ASSERT_UNKNOWN( op_arg_rec_.size() == i + 1 );
}
/*!
Put three operation argument index in the recording

\param arg0
First operation argument index.

\param arg1
Second operation argument index.

\param arg2
Third operation argument index.

\copydetails prototype_put_arg
*/
template <class Base>
inline void recorder<Base>::PutArg(addr_t arg0, addr_t arg1, addr_t arg2)
{
	size_t i         = op_arg_rec_.extend(3);
	op_arg_rec_[i++] =  static_cast<addr_t>( arg0 );
	op_arg_rec_[i++] =  static_cast<addr_t>( arg1 );
	op_arg_rec_[i]   =  static_cast<addr_t>( arg2 );
	CPPAD_ASSERT_UNKNOWN( op_arg_rec_.size() == i + 1 );
}
/*!
Put four operation argument index in the recording

\param arg0
First operation argument index.

\param arg1
Second operation argument index.

\param arg2
Third operation argument index.

\param arg3
Fourth operation argument index.

\copydetails prototype_put_arg
*/
template <class Base>
inline void recorder<Base>::PutArg(addr_t arg0, addr_t arg1, addr_t arg2,
	addr_t arg3)
{
	size_t i         = op_arg_rec_.extend(4);
	op_arg_rec_[i++] =  static_cast<addr_t>( arg0 );
	op_arg_rec_[i++] =  static_cast<addr_t>( arg1 );
	op_arg_rec_[i++] =  static_cast<addr_t>( arg2 );
	op_arg_rec_[i]   =  static_cast<addr_t>( arg3 );
	CPPAD_ASSERT_UNKNOWN( op_arg_rec_.size() == i + 1 );

}
/*!
Put five operation argument index in the recording

\param arg0
First operation argument index.

\param arg1
Second operation argument index.

\param arg2
Third operation argument index.

\param arg3
Fourth operation argument index.

\param arg4
Fifth operation argument index.

\copydetails prototype_put_arg
*/
template <class Base>
inline void recorder<Base>::PutArg(addr_t arg0, addr_t arg1, addr_t arg2,
	addr_t arg3, addr_t arg4)
{
	size_t i         = op_arg_rec_.extend(5);
	op_arg_rec_[i++] =  static_cast<addr_t>( arg0 );
	op_arg_rec_[i++] =  static_cast<addr_t>( arg1 );
	op_arg_rec_[i++] =  static_cast<addr_t>( arg2 );
	op_arg_rec_[i++] =  static_cast<addr_t>( arg3 );
	op_arg_rec_[i]   =  static_cast<addr_t>( arg4 );
	CPPAD_ASSERT_UNKNOWN( op_arg_rec_.size() == i + 1 );

}
/*!
Put six operation argument index in the recording

\param arg0
First operation argument index.

\param arg1
Second operation argument index.

\param arg2
Third operation argument index.

\param arg3
Fourth operation argument index.

\param arg4
Fifth operation argument index.

\param arg5
Sixth operation argument index.

\copydetails prototype_put_arg
*/
template <class Base>
inline void recorder<Base>::PutArg(addr_t arg0, addr_t arg1, addr_t arg2,
	addr_t arg3, addr_t arg4, addr_t arg5)
{
	size_t i         = op_arg_rec_.extend(6);
	op_arg_rec_[i++] =  static_cast<addr_t>( arg0 );
	op_arg_rec_[i++] =  static_cast<addr_t>( arg1 );
	op_arg_rec_[i++] =  static_cast<addr_t>( arg2 );
	op_arg_rec_[i++] =  static_cast<addr_t>( arg3 );
	op_arg_rec_[i++] =  static_cast<addr_t>( arg4 );
	op_arg_rec_[i]   =  static_cast<addr_t>( arg5 );
	CPPAD_ASSERT_UNKNOWN( op_arg_rec_.size() == i + 1 );
}
// --------------------------------------------------------------------------
/*!
Reserve space for arguments, but delay placing values there.

\param n_arg
number of arguements to reserve space for

\return
is the index in the argument vector corresponding to the
first of the arguments being reserved.
*/
template <class Base>
inline size_t recorder<Base>::ReserveArg(size_t n_arg)
{
	size_t i = op_arg_rec_.extend(n_arg);
	CPPAD_ASSERT_UNKNOWN( op_arg_rec_.size() == i + n_arg );
	return i;
}

/*!
\brief
Replace an argument value in the recording
(intended to fill in reserved values).

\param i_arg
is the index, in argument vector, for the value that is replaced.

\param value
is the new value for the argument with the specified index.
*/
template <class Base>
inline void recorder<Base>::ReplaceArg(size_t i_arg, size_t value)
{	op_arg_rec_[i_arg] =  static_cast<addr_t>( value ); }
// --------------------------------------------------------------------------
/*!
Put a character string in the text for this recording.

\param text
is a '\\0' terminated character string that is to be put in the
vector of characters corresponding to this recording.
The terminator '\\0' will be included.

\return
is the offset with in the text vector for this recording at which
the character string starts.
*/
template <class Base>
inline addr_t recorder<Base>::PutTxt(const char *text)
{
	// determine length of the text including terminating '\0'
	size_t n = 0;
	while( text[n] != '\0' )
		n++;
	CPPAD_ASSERT_UNKNOWN( n <= 1000 );
	n++;
	CPPAD_ASSERT_UNKNOWN( text[n-1] == '\0' );

	// copy text including terminating '\0'
	size_t i = text_rec_.extend(n);
	size_t j;
	for(j = 0; j < n; j++)
		text_rec_[i + j] = text[j];
	CPPAD_ASSERT_UNKNOWN( text_rec_.size() == i + n );

	CPPAD_ASSERT_KNOWN(
		std::numeric_limits<addr_t>::max() >= i,
		"cppad_tape_addr_type maximum value has been exceeded"
	);
	//
	return static_cast<addr_t>( i );
}
// -------------------------------------------------------------------------


} } // END_CPPAD_LOCAL_NAMESPACE
# endif
