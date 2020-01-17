// $Id: player.hpp 3941 2017-06-02 05:36:10Z bradbell $
# ifndef CPPAD_LOCAL_PLAYER_HPP
# define CPPAD_LOCAL_PLAYER_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

# include <cppad/local/user_state.hpp>

namespace CppAD { namespace local { // BEGIN_CPPAD_LOCAL_NAMESPACE
/*!
\file player.hpp
File used to define the player class.
*/


/*!
Class used to store and play back an operation sequence recording.

\tparam Base
These were AD< Base > operations when recorded. Operations during playback
are done using the type Base .
*/
template <class Base>
class player {
private:
	// ----------------------------------------------------------------------
	// Variables that define the recording
	// ----------------------------------------------------------------------
	/// Number of variables in the recording.
	size_t num_var_rec_;

	/// number of vecad load opeations in the reconding
	size_t num_load_op_rec_;

	/// Number of VecAD vectors in the recording
	size_t num_vecad_vec_rec_;

	/// The operators in the recording.
	pod_vector<CPPAD_OP_CODE_TYPE> op_rec_;

	/// The VecAD indices in the recording.
	pod_vector<addr_t> vecad_ind_rec_;

	/// The operation argument indices in the recording
	pod_vector<addr_t> op_arg_rec_;

	/// The parameters in the recording.
	/// Note that Base may not be plain old data, so use false in consructor.
	pod_vector<Base> par_rec_;

	/// Character strings ('\\0' terminated) in the recording.
	pod_vector<char> text_rec_;

	// ----------------------------------------------------------------------
	// Variables used for iterating thorough operators in the recording
	// ----------------------------------------------------------------------
	/// Current operator
	OpCode op_;

	/// Index in recording corresponding to current operator
	size_t op_index_;

	/// Current offset of the argument indices in op_arg_rec_
	const addr_t* op_arg_;

	/// Index for primary (last) variable corresponding to current operator
	size_t var_index_;

	/// index for the current user atomic function
	size_t user_index_;

	/// Flag indicating that a special function must be called before next
	/// This flags is not used when NDEBUG is defined, but kept in this case
	/// so that debug and release versions of CppAD can be mixed.
	bool      special_before_next_;

public:
	// =================================================================
	/// constructor
	player(void) :
	num_var_rec_(0)                                      ,
	num_load_op_rec_(0)                                  ,
	op_rec_( std::numeric_limits<addr_t>::max() )        ,
	vecad_ind_rec_( std::numeric_limits<addr_t>::max() ) ,
	op_arg_rec_( std::numeric_limits<addr_t>::max() )    ,
	par_rec_( std::numeric_limits<addr_t>::max() )       ,
	text_rec_( std::numeric_limits<addr_t>::max() )
	{ }

	// =================================================================
	/// destructor
	~player(void)
	{ }

	// ===============================================================
	/*!
	Moving an operation sequence from a recorder to this player

	\param rec
	the object that was used to record the operation sequence. After this
	operation, the state of the recording is no longer defined. For example,
	the pod_vector member variables in this have been swapped with
	 rec .
	*/
	void get(recorder<Base>& rec)
	{	size_t i;

		// just set size_t values
		num_var_rec_        = rec.num_var_rec_;
		num_load_op_rec_    = rec.num_load_op_rec_;

		// op_rec_
		op_rec_.swap(rec.op_rec_);

		// vec_ind_rec_
		vecad_ind_rec_.swap(rec.vecad_ind_rec_);

		// op_arg_rec_
		op_arg_rec_.swap(rec.op_arg_rec_);

		// par_rec_
		par_rec_.swap(rec.par_rec_);

		// text_rec_
		text_rec_.swap(rec.text_rec_);

		// set the number of VecAD vectors
		num_vecad_vec_rec_ = 0;
		for(i = 0; i < vecad_ind_rec_.size(); i += vecad_ind_rec_[i] + 1)
			num_vecad_vec_rec_++;

		// vecad_ind_rec_ contains size of each VecAD followed by
		// the parameter indices used to iniialize it.
		CPPAD_ASSERT_UNKNOWN( i == vecad_ind_rec_.size() );
	}
	// ===============================================================
	/*!
	Copying an operation sequence from another player to this one

	\param play
	the object that contains the operatoion sequence to copy.
	*/
	void operator=(const player& play)
	{
		num_var_rec_        = play.num_var_rec_;
		num_load_op_rec_    = play.num_load_op_rec_;
		op_rec_             = play.op_rec_;
		num_vecad_vec_rec_  = play.num_vecad_vec_rec_;
		vecad_ind_rec_      = play.vecad_ind_rec_;
		op_arg_rec_         = play.op_arg_rec_;
		par_rec_            = play.par_rec_;
		text_rec_           = play.text_rec_;
	}
	// ===============================================================
	/// Erase the recording stored in the player
	void Erase(void)
	{
		num_var_rec_       = 0;
		num_load_op_rec_   = 0;
		num_vecad_vec_rec_ = 0;

		op_rec_.erase();
		vecad_ind_rec_.erase();
		op_arg_rec_.erase();
		par_rec_.erase();
		text_rec_.erase();
	}
	// ================================================================
	// const functions that retrieve infromation from this player
	// ================================================================
	/*!
	\brief
	fetch an operator from the recording.

	\return
	the i-th operator in the recording.

	\param i
	the index of the operator in recording
	*/
	OpCode GetOp (size_t i) const
	{	return OpCode(op_rec_[i]); }

	/*!
	\brief
	Fetch a VecAD index from the recording.

	\return
	the i-th VecAD index in the recording.

	\param i
	the index of the VecAD index in recording
	*/
	size_t GetVecInd (size_t i) const
	{	return vecad_ind_rec_[i]; }

	/*!
	\brief
	Fetch a parameter from the recording.

	\return
	the i-th parameter in the recording.

	\param i
	the index of the parameter in recording
	*/
	Base GetPar(size_t i) const
	{	return par_rec_[i]; }

	/*!
	\brief
	Fetch entire parameter vector from the recording.

	\return
	the entire parameter vector.

	*/
	const Base* GetPar(void) const
	{	return par_rec_.data(); }

	/*!
	\brief
	Fetch a '\\0' terminated string from the recording.

	\return
	the beginning of the string.

	\param i
	the index where the string begins.
	*/
	const char *GetTxt(size_t i) const
	{	CPPAD_ASSERT_UNKNOWN(i < text_rec_.size() );
		return text_rec_.data() + i;
	}

	/// Fetch number of variables in the recording.
	size_t num_var_rec(void) const
	{	return num_var_rec_; }

	/// Fetch number of vecad load operations
	size_t num_load_op_rec(void) const
	{	return num_load_op_rec_; }

	/// Fetch number of operators in the recording.
	size_t num_op_rec(void) const
	{	return op_rec_.size(); }

	/// Fetch number of VecAD indices in the recording.
	size_t num_vec_ind_rec(void) const
	{	return vecad_ind_rec_.size(); }

	/// Fetch number of VecAD vectors in the recording
	size_t num_vecad_vec_rec(void) const
	{	return num_vecad_vec_rec_; }

	/// Fetch number of argument indices in the recording.
	size_t num_op_arg_rec(void) const
	{	return op_arg_rec_.size(); }

	/// Fetch number of parameters in the recording.
	size_t num_par_rec(void) const
	{	return par_rec_.size(); }

	/// Fetch number of characters (representing strings) in the recording.
	size_t num_text_rec(void) const
	{	return text_rec_.size(); }

	/// Fetch a rough measure of amount of memory used to store recording
	/// (just lengths, not capacities).
	size_t Memory(void) const
	{	return op_rec_.size()        * sizeof(OpCode)
		     + op_arg_rec_.size()    * sizeof(addr_t)
		     + par_rec_.size()       * sizeof(Base)
		     + text_rec_.size()      * sizeof(char)
		     + vecad_ind_rec_.size() * sizeof(addr_t)
		;
	}
	// =====================================================================
	// Forward iteration over operations in this player
	// =====================================================================
	/*!
	Start a play back of the recording during a forward sweep.

	Use repeated calls to forward_next to play back one operator at a time.

	\param op [out]
	The input value of op does not matter. Its output value is the
	first operator in the recording; i.e., BeginOp.

	\param op_arg [out]
	The input value of op_arg does not matter. Its output value is the
	beginning of the vector of argument indices for the first operation;
	i.e., 0

	\param op_index [out]
	The input value of op_index does not matter. Its output value
	is the index of the next first operator in the recording; i.e., 0.

	\param var_index [out]
	The input value of var_index does not matter. Its output value is the
	index of the primary (last) result corresponding to the the first
	operator (which must be a BeginOp); i.e., 0.
	*/
	void forward_start(
		OpCode&        op         ,
		const addr_t*& op_arg     ,
		size_t&        op_index   ,
		size_t&        var_index  )
	{
		op        = op_          = OpCode( op_rec_[0] );
		op_arg    = op_arg_      = op_arg_rec_.data();
		op_index  = op_index_    = 0;
		var_index = var_index_   = 0;
# ifndef NDEBUG
		special_before_next_     = false;
		CPPAD_ASSERT_UNKNOWN( op_ == BeginOp );
		CPPAD_ASSERT_NARG_NRES(op_, 1, 1);
# endif
		return;
	}

	/*!
	Fetch the next operator during a forward sweep.

	Use forward_start to initialize forward play back to the first operator;
	i.e., the BeginOp at the beginning of the recording.
	We use the notation forward_routine to denote the set
	forward_start, forward_next, forward_csum, forward_cskip, forward_user.

	\param op [in,out]
	The input value of op must be its output value from the
	previous call to a forward_routine.
	Its output value is the next operator in the recording.
	For speed, forward_next does not check for the special cases
	where op == CSumOp (op == CSkipOp). In this case
	some of the return values from forward_next must be corrected by a call
	to forward_csum (forward_cskip).
	In addition, for speed, extra information that is only used by the
	UserOp, UsrapOp, UsravOp, UsrrpOp, UsrrvOp operations is not returned
	for all operations. If this information is needed, then forward_user
	should be called after each call to forward_next.

	\param op_arg [in,out]
	The input value of op_arg must be its output value form the
	previous call to a forward routine.
	Its output value is the
	beginning of the vector of argument indices for this operation.

	\param op_index [in,out]
	The input value of op_index must be its output value form the
	previous call to a forward routine.
	Its output value is the index of this operator in the recording.
	Thus the ouput value following the previous call to forward_start is one.
	In addition,
	the output value increases by one with each call to forward_next.

	\param var_index [in,out]
	The input value of var_index must be its output value form the
	previous call to a forward routine.
	Its output value is the
	index of the primary (last) result corresponding to the operator op.
	*/
	void forward_next(
		OpCode&        op         ,
		const addr_t*& op_arg     ,
		size_t&        op_index   ,
		size_t&        var_index  )
	{
		CPPAD_ASSERT_UNKNOWN( ! special_before_next_ );
		CPPAD_ASSERT_UNKNOWN( op_       == op );
		CPPAD_ASSERT_UNKNOWN( op_arg    == op_arg_ );
		CPPAD_ASSERT_UNKNOWN( op_index  == op_index_ );
		CPPAD_ASSERT_UNKNOWN( var_index == var_index_ );

		// index for the next operator
		op_index    = ++op_index_;

		// first argument for next operator
		op_arg      = op_arg_    += NumArg(op_);

		// next operator
		op          = op_         = OpCode( op_rec_[ op_index_ ] );

		// index for last result for next operator
		var_index   = var_index_ += NumRes(op);

# ifndef NDEBUG
		special_before_next_ = (op == CSumOp) | (op == CSkipOp);
		//
		CPPAD_ASSERT_UNKNOWN( op_arg_rec_.data() <= op_arg_ );
		CPPAD_ASSERT_UNKNOWN(
			op_arg_ + NumArg(op) <= op_arg_rec_.data() + op_arg_rec_.size()
		);
		CPPAD_ASSERT_UNKNOWN( var_index_ < num_var_rec_ );
# endif
	}
	/*!
	Correct forward_next return values when op == CSumOp.

	\param op [in]
	The input value of op must be the return value from the previous
	call to forward_next and must be CSumOp. It is not modified.

	\param op_arg [in,out]
	The input value of op_arg must be the return value from the
	previous call to forward_next. Its output value is the
	beginning of the vector of argument indices for the next operation.

	\param op_index [in]
	The input value of op_index must be the return value from the
	previous call to forward_next. Its is not modified.

	\param var_index [in]
	The input value of var_index must be the return value from the
	previous call to forward_next. It is not modified.
	*/
	void forward_csum(
		const OpCode&  op         ,
		const addr_t*& op_arg     ,
		const size_t&  op_index   ,
		const size_t&  var_index  )
	{
		CPPAD_ASSERT_UNKNOWN( op_       == op );
		CPPAD_ASSERT_UNKNOWN( op_arg    == op_arg_ );
		CPPAD_ASSERT_UNKNOWN( op_index  == op_index_ );
		CPPAD_ASSERT_UNKNOWN( var_index == var_index_ );

		CPPAD_ASSERT_UNKNOWN( op == CSumOp );
		CPPAD_ASSERT_UNKNOWN( NumArg(CSumOp) == 0 );
		CPPAD_ASSERT_UNKNOWN(
		op_arg[0] + op_arg[1] == op_arg[ 3 + op_arg[0] + op_arg[1] ]
		);
		/*
		The only thing that really needs fixing is op_arg_.
		Actual number of arugments for this operator is
			op_arg[0] + op_arg[1] + 4.
		We must change op_arg_ so that when you add NumArg(CSumOp)
		you get first argument for next operator in sequence.
		*/
		op_arg = op_arg_ += op_arg[0] + op_arg[1] + 4;

# ifndef NDEBUG
		CPPAD_ASSERT_UNKNOWN( special_before_next_ );
		special_before_next_ = false;
		//
		CPPAD_ASSERT_UNKNOWN( op_arg_rec_.data() <= op_arg_ );
		CPPAD_ASSERT_UNKNOWN(
			op_arg_ + NumArg(op) <= op_arg_rec_.data() + op_arg_rec_.size()
		);
		CPPAD_ASSERT_UNKNOWN( var_index_ < num_var_rec_ );
# endif
	}
	/*!
	Correct forward_next return values when op == CSkipOp.

	\param op [in]
	The input value of op must be the return value from the previous
	call to forward_next and must be CSkipOp. It is not modified.

	\param op_arg [in,out]
	The input value of op_arg must be the return value from the
	previous call to forward_next. Its output value is the
	beginning of the vector of argument indices for the next operation.

	\param op_index [in]
	The input value of op_index must be the return value from the
	previous call to forward_next. Its is not modified.

	\param var_index [in]
	The input value of var_index must be the return value from the
	previous call to forward_next. It is not modified.
	*/
	void forward_cskip(
		const OpCode&  op         ,
		const addr_t*& op_arg     ,
		const size_t&  op_index   ,
		const size_t&  var_index  )
	{
		CPPAD_ASSERT_UNKNOWN( op_       == op );
		CPPAD_ASSERT_UNKNOWN( op_arg    == op_arg_ );
		CPPAD_ASSERT_UNKNOWN( op_index  == op_index_ );
		CPPAD_ASSERT_UNKNOWN( var_index == var_index_ );

		CPPAD_ASSERT_UNKNOWN( op == CSkipOp );
		CPPAD_ASSERT_UNKNOWN( NumArg(CSkipOp) == 0 );
		CPPAD_ASSERT_UNKNOWN(
		op_arg[4] + op_arg[5] == op_arg[ 6 + op_arg[4] + op_arg[5] ]
		);
		/*
		The only thing that really needs fixing is op_arg_.
		Actual number of arugments for this operator is
			7 + op_arg[4] + op_arg[5]
		We must change op_arg_ so that when you add NumArg(CSkipOp)
		you get first argument for next operator in sequence.
		*/
		op_arg = op_arg_ += 7 + op_arg[4] + op_arg[5];

# ifndef NDEBUG
		CPPAD_ASSERT_UNKNOWN( special_before_next_ );
		special_before_next_ = false;
		//
		CPPAD_ASSERT_UNKNOWN( op_arg_rec_.data() <= op_arg_ );
		CPPAD_ASSERT_UNKNOWN(
			op_arg_ + NumArg(op) <= op_arg_rec_.data() + op_arg_rec_.size()
		);
		CPPAD_ASSERT_UNKNOWN( var_index_ < num_var_rec_ );
# endif
	}
	/*!
	Extra information when forward_next returns one of the following op values:
	UserOp, UsrapOp, UsravOp, UsrrpOp, UsrrvOp.

	\param op [in]
	The value of op must be the return value from the previous
	call to forward_next and one of those listed above.

	\param user_state [in,out]
	This should be initialized to start_user before each call to
	forward_start and not otherwise changed by the calling program.
	Upon return it is the state of the user atomic call as follows:
	\li start_user next user operator will be UserOp at beginning of a call
	\li arg_user next operator will be UsrapOp or UsravOp.
	\li ret_user next operator will be UsrrpOp or UsrrvOp.
	\li end_user next operator will be UserOp at end of a call

	\param user_old [in,out]
	This should not be changed by the calling program.
	Upon return it is the extra information used by the old_atomic interface.

	\param user_m [in,out]
	This should not be changed by the calling program.
	Upon return it is the number of results for this user atomic function.

	\param user_n [in,out]
	This should not be changed by the calling program.
	Upon return it is the number of arguments to this user atomic function.

	\param user_i [in,out]
	This should not be changed by the calling program.
	Upon return it is the index for the next result for this
	user atomic function; i.e., the next UsrrpOp or UsrrvOp.
	If there are no more results, the return value is user_m.

	\param user_j [in,out]
	This should not be changed by the calling program.
	Upon return it is the index for the next argument for this
	user atomic function; i.e., the next UsrapOp or UsravOp.
	If there are no more arguments, the return value is user_n.

	\return
	the return value is a pointer to the atomic_base<Base> object
	for the correspnding function. If the corresponding user function
	has been deleted, an CPPAD_ASSERT_KNOWN is generated and a null pointer
	is returned.

	\par Initialization
	The initial value of user_old, user_m, user_n, user_i, user_j
	do not matter. They may be initialized to avoid compiler warnings.
	*/
	atomic_base<Base>* forward_user(
		const OpCode&    op         ,
		enum_user_state& user_state ,
		size_t&          user_old   ,
		size_t&          user_m     ,
		size_t&          user_n     ,
		size_t&          user_i     ,
		size_t&          user_j     )
	{	atomic_base<Base>* user_atom;
		switch(op)
		{
			case UserOp:
			CPPAD_ASSERT_NARG_NRES(op, 4, 0);
			if( user_state == start_user )
			{
				// forward_user arguments determined by values in UserOp
				user_index_ = op_arg_[0];
				user_old    = op_arg_[1];
				user_n      = op_arg_[2];
				user_m      = op_arg_[3];
				CPPAD_ASSERT_UNKNOWN( user_n > 0 );

				// other forward_user arguments
				user_j     = 0;
				user_i     = 0;
				user_state = arg_user;

# ifndef NDEBUG
				user_atom = atomic_base<Base>::class_object(user_index_);
				if( user_atom == CPPAD_NULL )
				{	// user_atom is null so cannot use user_atom->afun_name()
					std::string msg =
						atomic_base<Base>::class_name(user_index_)
						+ ": atomic_base function has been deleted";
					CPPAD_ASSERT_KNOWN(false, msg.c_str() );
				}
# endif
			}
			else
			{	// copy of UsrOp at end of this atomic sequence
				CPPAD_ASSERT_UNKNOWN( user_state == end_user );
				CPPAD_ASSERT_UNKNOWN( user_index_ == size_t(op_arg_[0]) );
				CPPAD_ASSERT_UNKNOWN( user_old   == size_t(op_arg_[1]) );
				CPPAD_ASSERT_UNKNOWN( user_n     == size_t(op_arg_[2]) );
				CPPAD_ASSERT_UNKNOWN( user_m     == size_t(op_arg_[3]) );
				CPPAD_ASSERT_UNKNOWN( user_j     == user_n );
				CPPAD_ASSERT_UNKNOWN( user_i     == user_m );
				user_state = start_user;
			}
			break;

			case UsrapOp:
			case UsravOp:
			CPPAD_ASSERT_UNKNOWN( NumArg(op) == 1 );
			CPPAD_ASSERT_UNKNOWN( user_state == arg_user );
			CPPAD_ASSERT_UNKNOWN( user_i == 0 );
			CPPAD_ASSERT_UNKNOWN( user_j < user_n );
			++user_j;
			if( user_j == user_n )
				user_state = ret_user;
			break;

			case UsrrpOp:
			case UsrrvOp:
			CPPAD_ASSERT_UNKNOWN( NumArg(op) == 1 || op == UsrrvOp );
			CPPAD_ASSERT_UNKNOWN( NumArg(op) == 0 || op == UsrrpOp );
			CPPAD_ASSERT_UNKNOWN( user_state == ret_user );
			CPPAD_ASSERT_UNKNOWN( user_i < user_m );
			CPPAD_ASSERT_UNKNOWN( user_j == user_n );
			++user_i;
			if( user_i == user_m )
				user_state = end_user;
			break;

			default:
			CPPAD_ASSERT_UNKNOWN(false);
		}
		// the atomic_base object corresponding to this user function
		user_atom = atomic_base<Base>::class_object(user_index_);
		CPPAD_ASSERT_UNKNOWN( user_atom != CPPAD_NULL );
		return user_atom;
	}
	// =====================================================================
	// Reverse iteration over operations in this player
	// =====================================================================
	/*!
	Start a play back of the recording during a reverse sweep.

	Use repeated calls to reverse_next to play back one operator at a time.

	\param op [out]
	The input value of op does not matter. Its output value is the
	last operator in the recording; i.e., EndOp.

	\param op_arg [out]
	The input value of op_arg does not matter. Its output value is the
	beginning of the vector of argument indices for the last operation;
	(there are no arguments for the last operation so op_arg is invalid).

	\param op_index [out[
	The input value of op_index does not matter. Its output value
	is the index of the last operator in the recording.

	\param var_index [out]
	The input value of var_index does not matter. Its output value is the
	index of the primary (last) result corresponding to the the last
	operator (which must be a EndOp).
	(there are no results for the last operation so var_index is invalid).
	*/

	void reverse_start(
		OpCode&        op         ,
		const addr_t*& op_arg     ,
		size_t&        op_index   ,
		size_t&        var_index  )
	{
		op_arg      = op_arg_     = op_arg_rec_.data() + op_arg_rec_.size();
		op_index    = op_index_   = op_rec_.size() - 1;
		var_index   = var_index_  = num_var_rec_ - 1;
		op          = op_         = OpCode( op_rec_[ op_index_ ] );
# ifndef NDEBUG
		special_before_next_ = false;
		CPPAD_ASSERT_UNKNOWN( op_ == EndOp );
		CPPAD_ASSERT_NARG_NRES(op, 0, 0);
# endif
		return;
	}

	/*!
	Fetch the next operator during a reverse sweep.

	Use reverse_start to initialize reverse play back to the last operator;
	i.e., the EndOp at the end of the recording.
	We use the notation reverse_routine to denote the set
	reverse_start, reverse_next, reverse_csum, reverse_cskip, reverse_user.

	\param op [in,out]
	The input value of op must be its output value from the
	previous call to a reverse_routine.
	Its output value is the next operator in the recording (in reverse order).
	For speed, reverse_next does not check for the special cases
	where op == CSumOp (op == CSkipOp). In this case
	some of the return values from reverse_next must be corrected by a call
	to reverse_csum (reverse_cskip).
	In addition, for speed, extra information that is only used by the
	UserOp, UsrapOp, UsravOp, UsrrpOp, UsrrvOp operations is not returned
	for all operations. If this information is needed, then reverse_user
	should be called after each call to reverse_next.

	\param op_arg [in,out]
	The input value of op_arg must be its output value from the
	previous call to a reverse_routine.
	Its output value is the
	beginning of the vector of argument indices for this operation.

	\param op_index [in,out]
	The input value of op_index must be its output value from the
	previous call to a reverse_routine.
	Its output value
	is the index of this operator in the recording. Thus the output
	value following the previous call to reverse_start is equal to
	the number of operators in the recording minus one.
	In addition, the output value decreases by one with each call to
	reverse_next.
	The last operator, BeginOp, sets op_index equal to 0.

	\param var_index [in,out]
	The input value of var_index must be its output value from the
	previous call to a reverse_routine.
	Its output value is the
	index of the primary (last) result corresponding to the operator op.
	The last operator sets var_index equal to 0 (corresponding to BeginOp
	at beginning of operation sequence).
	*/
	void reverse_next(
		OpCode&        op         ,
		const addr_t*& op_arg     ,
		size_t&        op_index   ,
		size_t&        var_index  )
	{
		CPPAD_ASSERT_UNKNOWN( ! special_before_next_ );
		CPPAD_ASSERT_UNKNOWN( op_       == op );
		CPPAD_ASSERT_UNKNOWN( op_arg    == op_arg_ );
		CPPAD_ASSERT_UNKNOWN( op_index  == op_index_ );
		CPPAD_ASSERT_UNKNOWN( var_index == var_index_ );

		// index of the last result for the next operator
		CPPAD_ASSERT_UNKNOWN( var_index_ >= NumRes(op_) );
		var_index   = var_index_ -= NumRes(op_);

		// next operator
		CPPAD_ASSERT_UNKNOWN( op_index_  > 0 );
		op_index    = --op_index_;                                  // index
		op          = op_         = OpCode( op_rec_[ op_index_ ] ); // value

		// first argument for next operator
		op_arg      = op_arg_    -= NumArg(op);

# ifndef NDEBUG
		special_before_next_ = (op == CSumOp) | (op == CSkipOp);
		//
		CPPAD_ASSERT_UNKNOWN( op_arg_rec_.data() <= op_arg_ );
		CPPAD_ASSERT_UNKNOWN(
			op_arg_ + NumArg(op) <= op_arg_rec_.data() + op_arg_rec_.size()
		);
# endif
	}
	/*!
	Correct reverse_next return values when op == CSumOp.

	\param op [in]
	The input value of op must be the return value from the previous
	call to reverse_next and must be CSumOp. It is not modified.

	\param op_arg [in,out]
	The input value of op_arg must be the return value from the
	previous call to reverse_next. Its output value is the
	beginning of the vector of argument indices for this operation.

	\param op_index [in]
	The input value of op_index must be the return value from the
	previous call to reverse_next. It is not modified.

	\param var_index [in]
	The input value of var_index must be the return value from the
	previous call to reverse_next. It is not modified.
	*/

	void reverse_csum(
		const OpCode&  op         ,
		const addr_t*& op_arg     ,
		const size_t&  op_index   ,
		const size_t&  var_index  )
	{
		CPPAD_ASSERT_UNKNOWN( op_       == op );
		CPPAD_ASSERT_UNKNOWN( op_arg    == op_arg_ );
		CPPAD_ASSERT_UNKNOWN( op_index  == op_index_ );
		CPPAD_ASSERT_UNKNOWN( var_index == var_index_ );

		CPPAD_ASSERT_UNKNOWN( op == CSumOp );
		CPPAD_ASSERT_UNKNOWN( NumArg(CSumOp) == 0 );
		/*
		The variables that need fixing are op_arg_ and op_arg. Currently,
		op_arg points to the last argument for the previous operator.
		*/
		// last argument for this csum operation
		--op_arg;
		// first argument for this csum operation
		op_arg = op_arg_ -= (op_arg[0] + 4);
		// now op_arg points to the first argument for this csum operator

		CPPAD_ASSERT_UNKNOWN(
		op_arg[0] + op_arg[1] == op_arg[ 3 + op_arg[0] + op_arg[1] ]
		);
# ifndef NDEBUG
		CPPAD_ASSERT_UNKNOWN( special_before_next_ );
		special_before_next_ = false;
		//
		CPPAD_ASSERT_UNKNOWN( op_index_ < op_rec_.size() );
		CPPAD_ASSERT_UNKNOWN( op_arg_rec_.data() <= op_arg_ );
		CPPAD_ASSERT_UNKNOWN( var_index_ < num_var_rec_ );
# endif
	}
	/*!
	Correct reverse_next return values when op == CSkipOp.

	\param op [int]
	The input value of op must be the return value from the previous
	call to reverse_next and must be CSkipOp. It is not modified.

	\param op_arg [in,out]
	The input value of op_arg must be the return value from the
	previous call to reverse_next. Its output value is the
	beginning of the vector of argument indices for this operation.

	\param op_index [in]
	The input value of op_index must be the return value from the
	previous call to reverse_next. It is not modified.

	\param var_index [in]
	The input value of var_index must be the return value from the
	previous call to reverse_next. It is not modified.
	*/

	void reverse_cskip(
		const OpCode&  op         ,
		const addr_t*& op_arg     ,
		const size_t&  op_index   ,
		const size_t&  var_index  )
	{
		CPPAD_ASSERT_UNKNOWN( op_       == op );
		CPPAD_ASSERT_UNKNOWN( op_arg    == op_arg_ );
		CPPAD_ASSERT_UNKNOWN( op_index  == op_index_ );
		CPPAD_ASSERT_UNKNOWN( var_index == var_index_ );

		CPPAD_ASSERT_UNKNOWN( op == CSkipOp );
		CPPAD_ASSERT_UNKNOWN( NumArg(CSkipOp) == 0 );
		/*
		The variables that need fixing are op_arg_ and op_arg. Currently,
		op_arg points to the last arugment for the previous operator.
		*/
		// last argument for this cskip operation
		--op_arg;
		// first argument for this cskip operation
		op_arg = op_arg_ -= (op_arg[0] + 7);

		CPPAD_ASSERT_UNKNOWN(
		op_arg[4] + op_arg[5] == op_arg[ 6 + op_arg[4] + op_arg[5] ]
		);
# ifndef NDEBUG
		CPPAD_ASSERT_UNKNOWN( special_before_next_ );
		special_before_next_ = false;
		//
		CPPAD_ASSERT_UNKNOWN( op_index_ < op_rec_.size() );
		CPPAD_ASSERT_UNKNOWN( op_arg_rec_.data() <= op_arg_ );
		CPPAD_ASSERT_UNKNOWN( var_index_ < num_var_rec_ );
# endif
	}
	/*!
	Extra information when reverse_next returns one of the following op values:
	UserOp, UsrapOp, UsravOp, UsrrpOp, UsrrvOp.

	\param op [in]
	The value of op must be the return value from the previous
	call to reverse_next and one of those listed above.

	\param user_state [in,out]
	This should be initialized to end_user before each call to
	reverse_start and not otherwise changed by the calling program.
	Upon return it is the state of the user atomic call as follows:
	\li end_user next user operator will be UserOp at end of a call
	\li ret_user next operator will be UsrrpOp or UsrrvOp.
	\li arg_user next operator will be UsrapOp or UsravOp.
	\li start_user next operator will be UserOp at beginning of a call

	\param user_old [in,out]
	This should not be changed by the calling program.
	Upon return it is the extra information used by the old_atomic interface.

	\param user_m [in,out]
	This should not be changed by the calling program.
	Upon return it is the number of results for this user atomic function.

	\param user_n [in,out]
	This should not be changed by the calling program.
	Upon return it is the number of arguments to this user atomic function.

	\param user_i [in,out]
	This should not be changed by the calling program.
	Upon return it is the index for this result for this
	user atomic function; i.e., this UsrrpOp or UsrrvOp.
	If the input value of user_state is end_user, the return value is user_m.

	\param user_j [in,out]
	This should not be changed by the calling program.
	Upon return it is the index for this argument for this
	user atomic function; i.e., this UsrapOp or UsravOp.
	If the input value of user_state is end_user, the return value is user_n.

	\return
	the return value is a pointer to the atomic_base<Base> object
	for the correspnding function. If the corresponding user function
	has been deleted, an CPPAD_ASSERT_KNOWN is generated and a null pointer
	is returned.

	\par Initialization
	The initial value of user_old, user_m, user_n, user_i, user_j
	do not matter. They may be initialized to avoid compiler warnings.
	*/
	atomic_base<Base>* reverse_user(
		const OpCode&    op         ,
		enum_user_state& user_state ,
		size_t&          user_old   ,
		size_t&          user_m     ,
		size_t&          user_n     ,
		size_t&          user_i     ,
		size_t&          user_j     )
	{	atomic_base<Base>* user_atom;
		switch(op)
		{
			case UserOp:
			CPPAD_ASSERT_NARG_NRES(op, 4, 0);
			if( user_state == end_user )
			{
				// reverse_user arguments determined by values in UserOp
				user_index_ = op_arg_[0];
				user_old    = op_arg_[1];
				user_n      = op_arg_[2];
				user_m      = op_arg_[3];
				CPPAD_ASSERT_UNKNOWN( user_n > 0 );

				// other reverse_user arguments
				user_j     = user_n;
				user_i     = user_m;
				user_state = ret_user;

				// the atomic_base object corresponding to this user function
# ifndef NDEBUG
				user_atom = atomic_base<Base>::class_object(user_index_);
				if( user_atom == CPPAD_NULL )
				{	// user_atom is null so cannot use user_atom->afun_name()
					std::string msg =
						atomic_base<Base>::class_name(user_index_)
						+ ": atomic_base function has been deleted";
					CPPAD_ASSERT_KNOWN(false, msg.c_str() );
				}
# endif
			}
			else
			{	// copy of UsrOp at end of this atomic sequence
				CPPAD_ASSERT_UNKNOWN( user_state == start_user );
				CPPAD_ASSERT_UNKNOWN( user_index_ == size_t(op_arg_[0]) );
				CPPAD_ASSERT_UNKNOWN( user_old   == size_t(op_arg_[1]) );
				CPPAD_ASSERT_UNKNOWN( user_n     == size_t(op_arg_[2]) );
				CPPAD_ASSERT_UNKNOWN( user_m     == size_t(op_arg_[3]) );
				CPPAD_ASSERT_UNKNOWN( user_j     == 0 );
				CPPAD_ASSERT_UNKNOWN( user_i     == 0 );
				user_state = end_user;
			}
			break;

			case UsrapOp:
			case UsravOp:
			CPPAD_ASSERT_UNKNOWN( NumArg(op) == 1 );
			CPPAD_ASSERT_UNKNOWN( user_state == arg_user );
			CPPAD_ASSERT_UNKNOWN( user_i == 0 );
			CPPAD_ASSERT_UNKNOWN( user_j <= user_n );
			CPPAD_ASSERT_UNKNOWN( 0 < user_j );
			--user_j;
			if( user_j == 0 )
				user_state = start_user;
			break;

			case UsrrpOp:
			case UsrrvOp:
			CPPAD_ASSERT_UNKNOWN( NumArg(op) == 1 || op == UsrrvOp );
			CPPAD_ASSERT_UNKNOWN( NumArg(op) == 0 || op == UsrrpOp );
			CPPAD_ASSERT_UNKNOWN( user_state == ret_user );
			CPPAD_ASSERT_UNKNOWN( user_i <= user_m );
			CPPAD_ASSERT_UNKNOWN( user_j == user_n );
			CPPAD_ASSERT_UNKNOWN( 0 < user_i );
			--user_i;
			if( user_i == 0 )
				user_state = arg_user;
			break;

			default:
			CPPAD_ASSERT_UNKNOWN(false);
		}
		// the atomic_base object corresponding to this user function
		user_atom = atomic_base<Base>::class_object(user_index_);
		CPPAD_ASSERT_UNKNOWN( user_atom != CPPAD_NULL );
		return user_atom;
	}

};

} } // END_CPPAD_lOCAL_NAMESPACE
# endif
