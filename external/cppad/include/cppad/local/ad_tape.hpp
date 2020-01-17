# ifndef CPPAD_LOCAL_AD_TAPE_HPP
# define CPPAD_LOCAL_AD_TAPE_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
# include <cppad/core/define.hpp>

namespace CppAD { namespace local { // BEGIN_CPPAD_LOCAL__NAMESPACE

/*!
Class used to hold tape that records AD<Base> operations.

\tparam Base
An <tt>AD<Base></tt> object is used to recording <tt>AD<Base></tt> operations.
*/

template <class Base>
class ADTape {
	// Friends =============================================================

	// classes -------------------------------------------------------------
	friend class AD<Base>;
	friend class ADFun<Base>;
	friend class atomic_base<Base>;
	friend class discrete<Base>;
	friend class VecAD<Base>;
	friend class VecAD_reference<Base>;

	// functions -----------------------------------------------------------
	// PrintFor
	friend void CppAD::PrintFor <Base> (
		const AD<Base>&    flag   ,
		const char*        before ,
		const AD<Base>&    var    ,
		const char*        after
	);
	// CondExpOp
	friend AD<Base> CppAD::CondExpOp <Base> (
		enum CompareOp  cop          ,
		const AD<Base> &left         ,
		const AD<Base> &right        ,
		const AD<Base> &trueCase     ,
		const AD<Base> &falseCase
	);
	// pow
	friend AD<Base> CppAD::pow <Base>
		(const AD<Base> &x, const AD<Base> &y);
	// azmul
	friend AD<Base> CppAD::azmul <Base>
		(const AD<Base> &x, const AD<Base> &y);
	// Parameter
	friend bool CppAD::Parameter     <Base>
		(const AD<Base> &u);
	// Variable
	friend bool CppAD::Variable      <Base>
		(const AD<Base> &u);
	// operators -----------------------------------------------------------
	// arithematic binary operators
	friend AD<Base> CppAD::operator + <Base>
		(const AD<Base> &left, const AD<Base> &right);
	friend AD<Base> CppAD::operator - <Base>
		(const AD<Base> &left, const AD<Base> &right);
	friend AD<Base> CppAD::operator * <Base>
		(const AD<Base> &left, const AD<Base> &right);
	friend AD<Base> CppAD::operator / <Base>
		(const AD<Base> &left, const AD<Base> &right);

	// comparison operators
	friend bool CppAD::operator < <Base>
		(const AD<Base> &left, const AD<Base> &right);
	friend bool CppAD::operator <= <Base>
		(const AD<Base> &left, const AD<Base> &right);
	friend bool CppAD::operator > <Base>
		(const AD<Base> &left, const AD<Base> &right);
	friend bool CppAD::operator >= <Base>
		(const AD<Base> &left, const AD<Base> &right);
	friend bool CppAD::operator == <Base>
		(const AD<Base> &left, const AD<Base> &right);
	friend bool CppAD::operator != <Base>
		(const AD<Base> &left, const AD<Base> &right);
	// ======================================================================

// --------------------------------------------------------------------------
private:
	// ----------------------------------------------------------------------
	// private data
	/*!
	Unique identifier for this tape.  It is always greater than
	CPPAD_MAX_NUM_THREADS, and different for every tape (even ones that have
	been deleted). In addition, id_ % CPPAD_MAX_NUM_THREADS is the thread
	number for this tape. Set by Independent and effectively const
	*/
	tape_id_t                    id_;
	/// Number of independent variables in this tapes reconding.
	/// Set by Independent and effectively const
	size_t         size_independent_;
	/// This is where the information is recorded.
	local::recorder<Base>              Rec_;
	// ----------------------------------------------------------------------
	// private functions
	//
	// add a parameter to the tape
	addr_t RecordParOp(const Base &x);

	// see CondExp.h
	void RecordCondExp(
		enum CompareOp  cop           ,
		AD<Base>       &returnValue   ,
		const AD<Base> &left          ,
		const AD<Base> &right         ,
		const AD<Base> &trueCase      ,
		const AD<Base> &falseCase
	);

	// place a VecAD object in the tape
	size_t AddVec(
		size_t                   length,
		const pod_vector<Base>&  data
	);

public:
	// default constructor and destructor

	// public function only used by CppAD::Independent
	template <typename VectorADBase>
	void Independent(VectorADBase &u);
	template <typename VectorADBase>
	void Independent(VectorADBase &u, size_t abort_op_index);

};
// ---------------------------------------------------------------------------
// Private functions
//

/*!
Place a parameter in the tape.

On rare occations it is necessary to place a parameter in the tape; e.g.,
when it is one of the dependent variabes.

\param z
value of the parameter that we are placing in the tape.

\return
variable index (for this recording) correpsonding to the parameter.

\par 2DO
All these operates are preformed in \c Rec_, so we should
move this routine from <tt>ADTape<Base></tt> to <tt>recorder<Base></tt>.
*/
template <class Base>
addr_t ADTape<Base>::RecordParOp(const Base &z)
{	addr_t z_taddr;
	addr_t ind;
	CPPAD_ASSERT_UNKNOWN( NumRes(ParOp) == 1 );
	CPPAD_ASSERT_UNKNOWN( NumArg(ParOp) == 1 );
	z_taddr = Rec_.PutOp(ParOp);
	ind     = Rec_.PutPar(z);
	Rec_.PutArg(ind);

	return z_taddr;
}

/*!
Put initialization for a VecAD<Base> object in the tape.

This routine should be called once for each VecAD object when just
before it changes from a parameter to a variable.

\param length
size of the <tt>VecAD<Base></tt> object.

\param data
initial values for the <tt>VecAD<Base></tt> object
(values before it becomes a variable).

\return
index of the start of this vector in the list of vector indices.
The value for this vector index is the length of the vector.
There are \c length indices following for this vector.
The values for these vector indices are the corresponding
parameter indices in the tape for the initial value of the corresponding
vec_ad element.

\par 2DO
All these operates are preformed in \c Rec_, so we should
move this routine from <tt>ADTape<Base></tt> to <tt>recorder<Base></tt>.
*/
template <class Base>
size_t ADTape<Base>::AddVec(size_t length, const pod_vector<Base>& data)
{	CPPAD_ASSERT_UNKNOWN( length > 0 );
	size_t i;
	size_t value_index;

	// store the length in VecInd
	size_t start = Rec_.PutVecInd(length);

	// store indices of the values in VecInd
	for(i = 0; i < length; i++)
	{
		value_index = Rec_.PutPar( data[i] );
		Rec_.PutVecInd( value_index );
	}

	// return the taddr of the length (where the vector starts)
	return start;
}

} } // END_CPPAD_LOCAL_NAMESPACE

# endif
