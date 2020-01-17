# ifndef CPPAD_CORE_VEC_AD_HPP
# define CPPAD_CORE_VEC_AD_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin VecAD$$
$spell
	cppad.hpp
	CondExpGt
	grep
	Ld
	vp
	Lu
	wc
	op
	Ldp
	Ldv
	Taylor
	VecAD
	const
	Cpp
$$


$section AD Vectors that Record Index Operations$$
$mindex VecAD tape reference VecAD<Base>$$


$head Syntax$$
$codei%VecAD<%Base%> %v%(%n%)%$$
$pre
$$
$icode%v%.size()%$$
$pre
$$
$icode%b% = %v%[%i%]%$$
$pre
$$
$icode%r% = %v%[%x%]%$$

$head Purpose$$
If either $icode v$$ or $icode x$$ is a
$cref/variable/glossary/Variable/$$,
the indexing operation
$codei%
	%r% = %v%[%x%]
%$$
is recorded in the corresponding
AD of $icode Base$$
$cref/operation sequence/glossary/Operation/Sequence/$$ and
transferred to the corresponding $cref ADFun$$ object $icode f$$.
Such an index can change each time
zero order $cref/f.Forward/Forward/$$ is used; i.e.,
$icode f$$ is evaluated with new value for the
$cref/independent variables/glossary/Tape/Independent Variable/$$.
Note that the value of $icode y$$ depends on the value of $icode x$$
in a discrete fashion and CppAD computes its partial derivative with
respect to $icode x$$ as zero.

$head Alternatives$$
If only the values in the vector,
and not the indices,
depend on the independent variables,
the class $icode%Vector%< AD<%Base%> >%$$ is much more efficient for
storing AD values where $icode Vector$$ is any
$cref SimpleVector$$ template class,
If only the indices,
and not the values in the vector,
depend on the independent variables,
The $cref Discrete$$ functions are a much more efficient
way to represent these vectors.

$head VecAD<Base>::reference$$
The result $icode r$$ has type
$codei%
	VecAD<%Base%>::reference
%$$
which is very much like the $codei%AD<%Base%>%$$ type
with some notable exceptions:

$subhead Exceptions$$
$list number$$
The object $icode r$$ cannot be used with the
$cref Value$$ function to compute the corresponding $icode Base$$ value.
If $icode v$$ and $icode i$$ are not $cref/variables/glossary/Variable/$$
$codei%
	%b% = v[%i%]
%$$
can be used to compute the corresponding $icode Base$$ value.

$lnext
The object $icode r$$ cannot be used
with the $cref/compound assignments operators/Arithmetic/$$
$code +=$$,
$code -=$$,
$code *=$$, or
$code /=$$.
For example, the following syntax is not valid:
$codei%
	%v%[%x%] += %z%;
%$$
no matter what the types of $icode z$$.

$lnext
Assignment to $icode r$$ returns a $code void$$.
For example, the following syntax is not valid:
$codei%
	%z% = %v%[%x%] = %u%;
%$$
no matter what the types of $icode z$$, and $icode u$$.

$lnext
The $cref CondExp$$ functions do not accept
$codei%VecAD<%Base%>::reference%$$ arguments.
For example, the following syntax is not valid:
$codei%
	CondExpGt(%v%[%x%], %z%, %u%, %v%)
%$$
no matter what the types of $icode z$$, $icode u$$, and $icode v$$.

$lnext
The $cref/Parameter and Variable/ParVar/$$ functions cannot be used with
$codei%VecAD<%Base%>::reference%$$ arguments like $icode r$$,
use the entire $codei%VecAD<%Base%>%$$ vector instead; i.e. $icode v$$.

$lnext
The vectors passed to $cref Independent$$ must have elements
of type $codei%AD<%Base%>%$$; i.e., $cref VecAD$$ vectors
cannot be passed to $code Independent$$.

$lnext
If one uses this type in a
AD of $icode Base$$
$cref/operation sequence/glossary/Operation/Sequence/$$,
$cref/sparsity pattern/glossary/Sparsity Pattern/$$ calculations
($cref sparsity_pattern$$)
are less efficient because the dependence of different
elements of the vector cannot be separated.

$lend

$head Constructor$$

$subhead v$$
The syntax
$codei%
	VecAD<%Base%> %v%(%n%)
%$$
creates an $code VecAD$$ object $icode v$$ with
$icode n$$ elements.
The initial value of the elements of $icode v$$ is unspecified.

$head n$$
The argument $icode n$$ has prototype
$codei%
	size_t %n%
%$$

$head size$$
The syntax
$codei%
	%v%.size()
%$$
returns the number of elements in the vector $icode v$$;
i.e., the value of $icode n$$ when it was constructed.

$head size_t Indexing$$
We refer to the syntax
$codei%
	%b% = %v%[%i%]
%$$
as $code size_t$$ indexing of a $code VecAD$$ object.
This indexing is only valid if the vector $icode v$$ is a
$cref/parameter/ParVar/$$; i.e.,
it does not depend on the independent variables.

$subhead i$$
The operand $icode i$$ has prototype
$codei%
	size_t %i%
%$$
It must be greater than or equal zero
and less than $icode n$$; i.e., less than
the number of elements in $icode v$$.

$subhead b$$
The result $icode b$$ has prototype
$codei%
	%Base% %b%
%$$
and is a reference to the $th i$$ element in the vector $icode v$$.
It can be used to change the element value;
for example,
$codei%
	%v%[%i%] = %c%
%$$
is valid where $icode c$$ is a $icode Base$$ object.
The reference $icode b$$ is no longer valid once the
destructor for $icode v$$ is called; for example,
when $icode v$$ falls out of scope.

$head AD Indexing$$
We refer to the syntax
$codei%
	%r% = %v%[%x%]
%$$
as AD indexing of a $code VecAD$$ object.

$subhead x$$
The argument $icode x$$ has prototype
$codei%
	const AD<%Base%> &%x%
%$$
The value of $icode x$$ must be greater than or equal zero
and less than $icode n$$; i.e., less than
the number of elements in $icode v$$.

$subhead r$$
The result $icode r$$ has prototype
$codei%
	VecAD<%Base%>::reference %r%
%$$
The object $icode r$$ has an AD type and its
operations are recorded as part of the same
AD of $icode Base$$
$cref/operation sequence/glossary/Operation/Sequence/$$ as
for $codei%AD<%Base%>%$$ objects.
It acts as a reference to the
element with index $latex {\rm floor} (x)$$ in the vector $icode v$$
($latex {\rm floor} (x)$$ is
the greatest integer less than or equal $icode x$$).
Because it is a reference, it can be used to change the element value;
for example,
$codei%
	%v%[%x%] = %z%
%$$
is valid where $icode z$$ is an
$codei%VecAD<%Base%>::reference%$$ object.
As a reference, $icode r$$ is no longer valid once the
destructor for $icode v$$ is called; for example,
when $icode v$$ falls out of scope.

$head Example$$
$children%
	example/general/vec_ad.cpp
%$$
The file
$cref vec_ad.cpp$$
contains an example and test using $code VecAD$$ vectors.
It returns true if it succeeds and false otherwise.


$head Speed and Memory$$
The $cref VecAD$$ vector type is inefficient because every
time an element of a vector is accessed, a new CppAD
$cref/variable/glossary/Variable/$$ is created on the tape
using either the $code Ldp$$ or $code Ldv$$ operation
(unless all of the elements of the vector are
$cref/parameters/glossary/Parameter/$$).
The effect of this can be seen by executing the following steps:

$list number$$
In the file $code cppad/local/forward1sweep.h$$,
change the definition of $code CPPAD_FORWARD1SWEEP_TRACE$$ to
$codep
	# define CPPAD_FORWARD1SWEEP_TRACE 1
$$
$lnext
In the $code Example$$ directory, execute the command
$codep
	./test_one.sh lu_vec_ad_ok.cpp lu_vec_ad.cpp -DNDEBUG > lu_vec_ad_ok.log
$$
This will write a trace of all the forward tape operations,
for the test case $cref lu_vec_ad_ok.cpp$$,
to the file $code lu_vec_ad_ok.log$$.
$lnext
In the $code Example$$ directory execute the commands
$codep
	grep "op="           lu_vec_ad_ok.log | wc -l
	grep "op=Ld[vp]"     lu_vec_ad_ok.log | wc -l
	grep "op=St[vp][vp]" lu_vec_ad_ok.log | wc -l
$$
The first command counts the number of operators in the tracing,
the second counts the number of VecAD load operations,
and the third counts the number of VecAD store operations.
(For CppAD version 05-11-20 these counts were 956, 348, and 118
respectively.)
$lend

$end
------------------------------------------------------------------------
*/
# include <cppad/local/pod_vector.hpp>

namespace CppAD { //  BEGIN_CPPAD_NAMESPACE
/*!
\file vec_ad.hpp
Defines the VecAD<Base> class.
*/

/*!
\def CPPAD_VEC_AD_COMPUTED_ASSIGNMENT(op, name)
Prints an error message if the correspinding compound assignment is used.

THis macro is used to print an error message if any of the
compound assignments are used with the VecAD_reference class.
The argument \c op is one of the following:
+= , -= , *= , /=.
The argument \c name, is a string literal with the name of the
compound assignment \c op.
*/
# define CPPAD_VEC_AD_COMPUTED_ASSIGNMENT(op, name)                     \
VecAD_reference& operator op (const VecAD_reference<Base> &right)       \
{	CPPAD_ASSERT_KNOWN(                                                \
		false,                                                        \
		"Cannot use a ADVec element on left side of" name             \
	);                                                                 \
	return *this;                                                      \
}                                                                       \
VecAD_reference& operator op (const AD<Base> &right)                    \
{	CPPAD_ASSERT_KNOWN(                                                \
		false,                                                        \
		"Cannot use a ADVec element on left side of" name             \
	);                                                                 \
	return *this;                                                      \
}                                                                       \
VecAD_reference& operator op (const Base &right)                        \
{	CPPAD_ASSERT_KNOWN(                                                \
		false,                                                        \
		"Cannot use a ADVec element on left side of" name             \
	);                                                                 \
	return *this;                                                      \
}

/*!
Class used to hold a reference to an element of a VecAD object.

\tparam Base
Elements of this class act like an AD<Base> (in a restricted sense),
in addition they track (on the tape) the index they correspond to.
*/
template <class Base>
class VecAD_reference {
	friend bool  Parameter<Base> (const VecAD<Base> &vec);
	friend bool  Variable<Base>  (const VecAD<Base> &vec);
	friend class VecAD<Base>;
	friend class local::ADTape<Base>;

private:
	/// pointer to vecad vector that this is a element of
	VecAD<Base> *vec_;
	/// index in vecad vector that this element corresponds to
	AD<Base>     ind_;         // index for this element
public:
	/*!
	consructor

	\param vec
	value of vec_

	\param ind
	value of ind_
	*/
	VecAD_reference(VecAD<Base> *vec, const AD<Base>& ind)
		: vec_( vec ) , ind_(ind)
	{ }

	// assignment operators
	inline void operator = (const VecAD_reference<Base> &right);
	void operator = (const AD<Base> &right);
	void operator = (const Base     &right);
	void operator = (int             right);

	// compound assignments
	CPPAD_VEC_AD_COMPUTED_ASSIGNMENT( += , " += " )
	CPPAD_VEC_AD_COMPUTED_ASSIGNMENT( -= , " -= " )
	CPPAD_VEC_AD_COMPUTED_ASSIGNMENT( *= , " *= " )
	CPPAD_VEC_AD_COMPUTED_ASSIGNMENT( /= , " /= " )


	/// Conversion from VecAD_reference to AD<Base>.
	/// puts the correspond vecad load instruction in the tape.
	AD<Base> ADBase(void) const
	{	AD<Base> result;

		size_t i = static_cast<size_t>( Integer(ind_) );
		CPPAD_ASSERT_UNKNOWN( i < vec_->length_ );

		// AD<Base> value corresponding to this element
		result.value_ = vec_->data_[i];

		// this address will be recorded in tape and must be
		// zero for parameters
		CPPAD_ASSERT_UNKNOWN( Parameter(result) );
		result.taddr_ = 0;

		// index corresponding to this element
		if( Variable(*vec_) )
		{
			local::ADTape<Base>* tape = AD<Base>::tape_ptr(vec_->tape_id_);
			CPPAD_ASSERT_UNKNOWN( tape != CPPAD_NULL );
			CPPAD_ASSERT_UNKNOWN( vec_->offset_ > 0  );

			size_t load_op_index = tape->Rec_.num_load_op_rec();
			if( IdenticalPar(ind_) )
			{	CPPAD_ASSERT_UNKNOWN( local::NumRes(local::LdpOp) == 1 );
				CPPAD_ASSERT_UNKNOWN( local::NumArg(local::LdpOp) == 3 );

				// put operand addresses in tape
				tape->Rec_.PutArg(
					(addr_t) vec_->offset_, (addr_t) i, (addr_t) load_op_index
				);
				// put operator in the tape, ind_ is a parameter
				result.taddr_ = tape->Rec_.PutLoadOp(local::LdpOp);
				// change result to variable for this load
				result.tape_id_ = tape->id_;
			}
			else
			{	CPPAD_ASSERT_UNKNOWN( local::NumRes(local::LdvOp) == 1 );
				CPPAD_ASSERT_UNKNOWN( local::NumArg(local::LdvOp) == 3 );
				addr_t ind_taddr;
				if( Parameter(ind_) )
				{	// kludge that should not be needed
					// if ind_ instead of i is used for index
					// in the tape
					ind_taddr  = tape->RecordParOp(
						ind_.value_
					);
				}
				else	ind_taddr = ind_.taddr_;
				CPPAD_ASSERT_UNKNOWN( ind_taddr > 0 );

				// put operand addresses in tape
				// (value of third arugment does not matter)
				tape->Rec_.PutArg(
					(addr_t) vec_->offset_,
					(addr_t) ind_taddr,
					(addr_t) load_op_index
				);
				// put operator in the tape, ind_ is a variable
				result.taddr_ = tape->Rec_.PutLoadOp(local::LdvOp);
				// change result to variable for this load
				result.tape_id_ = tape->id_;
			}
		}
		return result;
	}
};

/*!
Vector of AD objects that tracks indexing operations on the tape.
*/
template <class Base>
class VecAD {
	friend bool  Parameter<Base> (const VecAD<Base> &vec);
	friend bool  Variable<Base>  (const VecAD<Base> &vec);
	friend class local::ADTape<Base>;
	friend class VecAD_reference<Base>;

	friend std::ostream& operator << <Base>
		(std::ostream &os, const VecAD<Base> &vec_);
private:
	/// size of this VecAD vector
	const  size_t   length_;

	/// elements of this vector
	local::pod_vector<Base> data_;

	/// offset in cummulate vector corresponding to this object
	size_t offset_;

	/// tape id corresponding to the offset
	tape_id_t tape_id_;
public:
	/// declare the user's view of this type here
	typedef VecAD_reference<Base> reference;

	/// default constructor
	/// initialize tape_id_ same as for default constructor; see default.hpp
	VecAD(void)
	: length_(0)
	, offset_(0)
	, tape_id_(0)
	{	CPPAD_ASSERT_UNKNOWN( Parameter(*this) ); }

	/// sizing constructor
	/// initialize tape_id_ same as for parameters; see ad_copy.hpp
	VecAD(size_t n)
	: length_(n)
	, offset_(0)
	, tape_id_(0)
	{	if( length_ > 0 )
		{	size_t i;
			Base zero(0);
			data_.extend(length_);

			// Initialize data to zero so all have same value.
			// This uses less memory and avoids a valgrind error
			// during TapeRec<Base>::PutPar
			for(i = 0; i < length_; i++)
				data_[i] = zero;
		}
		CPPAD_ASSERT_UNKNOWN( Parameter(*this) );
	}

	/// destructor
	~VecAD(void)
	{ }

	/// number of elements in the vector
	size_t size(void)
	{	return length_; }

	/// element access (not taped)
	///
	/// \param i
	/// element index
	Base &operator[](size_t i)
	{
		CPPAD_ASSERT_KNOWN(
			Parameter(*this),
			"VecAD: cannot use size_t indexing because this"
			" VecAD vector is a variable."
		);
		CPPAD_ASSERT_KNOWN(
			i < length_,
			"VecAD: element index is >= vector length"
		);

		return data_[i];
	}

	/*! delayed taped elemement access

	\param x
	element index

	\par
	This operation may convert this vector from a parameter to a variable
	*/
	VecAD_reference<Base> operator[](const AD<Base> &x)
	{
		CPPAD_ASSERT_KNOWN(
			0 <= Integer(x),
			"VecAD: element index is less than zero"
		);
		CPPAD_ASSERT_KNOWN(
			static_cast<size_t>( Integer(x) ) < length_,
			"VecAD: element index is >= vector length"
		);

		// if no need to track indexing operation, return now
		if( Parameter(*this) & Parameter(x) )
			return VecAD_reference<Base>(this, x);

		CPPAD_ASSERT_KNOWN(
			Parameter(*this) | Parameter(x) | (tape_id_ == x.tape_id_),
			"VecAD: vector and index are variables for"
			" different tapes."
		);

		if( Parameter(*this) )
		{	// must place a copy of vector in tape
			offset_ =
			AD<Base>::tape_ptr(x.tape_id_)->AddVec(length_, data_);

			// Advance pointer by one so starts at first component of this
			// vector; i.e., skip lenght at begining (so is always > 0)
			offset_++;

			// tape id corresponding to this offest
			tape_id_ = x.tape_id_;
		}

		return VecAD_reference<Base>(this, x);
	}

};


/*!
Taped setting of element to a value.

\param y
value that element is set to.
*/
template <class Base>
void VecAD_reference<Base>::operator=(const AD<Base> &y)
{
	if( Parameter(y) )
	{	// fold into the Base type assignment
		*this = y.value_;
		return;
	}
	CPPAD_ASSERT_UNKNOWN( y.taddr_ > 0 );

	CPPAD_ASSERT_KNOWN(
		Parameter(*vec_) | (vec_->tape_id_ == y.tape_id_),
		"VecAD assignment: vector and new element value are variables"
		"\nfor different tapes."
	);

	local::ADTape<Base>* tape = AD<Base>::tape_ptr(y.tape_id_);
	CPPAD_ASSERT_UNKNOWN( tape != CPPAD_NULL );
	if( Parameter(*vec_) )
	{	// must place a copy of vector in tape
		vec_->offset_ = tape->AddVec(vec_->length_, vec_->data_);

		// advance offset to be start of vector plus one
		(vec_->offset_)++;

		// tape id corresponding to this offest
		vec_->tape_id_ = y.tape_id_;
	}
	CPPAD_ASSERT_UNKNOWN( Variable(*vec_) );


	// index in vector for this element
	size_t i = static_cast<size_t>( Integer(ind_) );
	CPPAD_ASSERT_UNKNOWN( i < vec_->length_ );

	// assign value for this element (as an AD<Base> object)
	vec_->data_[i] = y.value_;

	// record the setting of this array element
	CPPAD_ASSERT_UNKNOWN( vec_->offset_ > 0 );
	if( Parameter(ind_) )
	{	CPPAD_ASSERT_UNKNOWN( local::NumArg(local::StpvOp) == 3 );
		CPPAD_ASSERT_UNKNOWN( local::NumRes(local::StpvOp) == 0 );

		// put operand addresses in tape
		tape->Rec_.PutArg((addr_t) vec_->offset_, (addr_t) i, y.taddr_);

		// put operator in the tape, ind_ is parameter, y is variable
		tape->Rec_.PutOp(local::StpvOp);
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( local::NumArg(local::StvvOp) == 3 );
		CPPAD_ASSERT_UNKNOWN( local::NumRes(local::StvvOp) == 0 );
		CPPAD_ASSERT_UNKNOWN( ind_.taddr_ > 0 );

		// put operand addresses in tape
		tape->Rec_.PutArg((addr_t) vec_->offset_, ind_.taddr_, y.taddr_);

		// put operator in the tape, ind_ is variable, y is variable
		tape->Rec_.PutOp(local::StvvOp);
	}
}


/*!
Taped setting of element to a value.

\param y
value that element is set to.
*/
template <class Base>
void VecAD_reference<Base>::operator=(const Base &y)
{
	size_t i = static_cast<size_t>( Integer(ind_) );
	CPPAD_ASSERT_UNKNOWN( i < vec_->length_ );

	// assign value for this element
	vec_->data_[i] = y;

	// check if this ADVec object is a parameter
	if( Parameter(*vec_) )
		return;

	local::ADTape<Base>* tape = AD<Base>::tape_ptr(vec_->tape_id_);
	CPPAD_ASSERT_UNKNOWN( tape != CPPAD_NULL );

	// put value of the parameter y in the tape
	addr_t p = tape->Rec_.PutPar(y);

	// record the setting of this array element
	CPPAD_ASSERT_UNKNOWN( vec_->offset_ > 0 );
	if( Parameter(ind_) )
	{	CPPAD_ASSERT_UNKNOWN( local::NumArg(local::StppOp) == 3 );
		CPPAD_ASSERT_UNKNOWN( local::NumRes(local::StppOp) == 0 );

		// put operand addresses in tape
		tape->Rec_.PutArg((addr_t) vec_->offset_, (addr_t) i, p);

		// put operator in the tape, ind_ is parameter, y is parameter
		tape->Rec_.PutOp(local::StppOp);
	}
	else
	{	CPPAD_ASSERT_UNKNOWN( local::NumArg(local::StvpOp) == 3 );
		CPPAD_ASSERT_UNKNOWN( local::NumRes(local::StvpOp) == 0 );
		CPPAD_ASSERT_UNKNOWN( ind_.taddr_ > 0 );

		// put operand addresses in tape
		tape->Rec_.PutArg((addr_t) vec_->offset_, ind_.taddr_, p);

		// put operator in the tape, ind_ is variable, y is parameter
		tape->Rec_.PutOp(local::StvpOp);
	}
}

/*!
Taped setting of element to a value.

\param y
value that element is set to.

\par
this case gets folded into case where value is AD<Base>.
*/
template <class Base>
inline void VecAD_reference<Base>::operator=
(const VecAD_reference<Base> &y)
{	*this = y.ADBase(); }

/*!
Taped setting of element to a value.

\param y
value that element is set to.

\par
this case gets folded into case where value is Base.
*/
template <class Base>
inline void VecAD_reference<Base>::operator=(int y)
{	*this = Base(y); }


} // END_CPPAD_NAMESPACE

// preprocessor symbols that are local to this file
# undef CPPAD_VEC_AD_COMPUTED_ASSIGNMENT

# endif
