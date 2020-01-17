# ifndef CPPAD_CORE_DISCRETE_HPP
# define CPPAD_CORE_DISCRETE_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin Discrete$$
$spell
	retaping
	namespace
	std
	Eq
	Cpp
	const
	inline
	Geq
$$

$section Discrete AD Functions$$
$mindex CPPAD_DISCRETE_FUNCTION$$


$head Syntax$$
$codei%CPPAD_DISCRETE_FUNCTION(%Base%, %name%)
%$$
$icode%y%  = %name%(%x%)
%$$
$icode%ay% = %name%(%ax%)
%$$


$head Purpose$$
Record the evaluation of a discrete function as part
of an $codei%AD<%Base%>%$$
$cref/operation sequence/glossary/Operation/Sequence/$$.
The value of a discrete function can depend on the
$cref/independent variables/glossary/Tape/Independent Variable/$$,
but its derivative is identically zero.
For example, suppose that the integer part of
a $cref/variable/glossary/Variable/$$ $icode x$$ is the
index into an array of values.

$head Base$$
This is the
$cref/base type/base_require/$$
corresponding to the operations sequence;
i.e., use of the $icode name$$ with arguments of type
$codei%AD<%Base%>%$$ can be recorded in an operation sequence.

$head name$$
This is the name of the function (as it is used in the source code).
The user must provide a version of $icode name$$
where the argument has type $icode Base$$.
CppAD uses this to create a version of $icode name$$
where the argument has type $codei%AD<%Base%>%$$.

$head x$$
The argument $icode x$$ has prototype
$codei%
	const %Base%& %x%
%$$
It is the value at which the user provided version of $icode name$$
is to be evaluated.

$head y$$
The result $icode y$$ has prototype
$codei%
	%Base% %y%
%$$
It is the return value for the user provided version of $icode name$$.

$head ax$$
The argument $icode ax$$ has prototype
$codei%
	const AD<%Base%>& %ax%
%$$
It is the value at which the CppAD provided version of $icode name$$
is to be evaluated.

$head ay$$
The result $icode ay$$ has prototype
$codei%
	AD<%Base%> %ay%
%$$
It is the return value for the CppAD provided version of $icode name$$.


$head Create AD Version$$
The preprocessor macro invocation
$codei%
	CPPAD_DISCRETE_FUNCTION(%Base%, %name%)
%$$
defines the $codei%AD<%Base%>%$$ version of $icode name$$.
This can be with in a namespace (not the $code CppAD$$ namespace)
but must be outside of any routine.

$head Operation Sequence$$
This is an AD of $icode Base$$
$cref/atomic operation/glossary/Operation/Atomic/$$
and hence is part of the current
AD of $icode Base$$
$cref/operation sequence/glossary/Operation/Sequence/$$.

$head Derivatives$$
During a zero order $cref Forward$$ operation,
an $cref ADFun$$ object will compute the value of $icode name$$
using the user provided $icode Base$$ version of this routine.
All the derivatives of $icode name$$ will be evaluated as zero.

$head Parallel Mode$$
The first call to
$codei%
	%ay% = %name%(%ax%)
%$$
must not be in $cref/parallel/ta_in_parallel/$$ execution mode.


$head Example$$
$children%
	example/general/tape_index.cpp%
	example/general/interp_onetape.cpp%
	example/general/interp_retape.cpp
%$$
The file
$cref tape_index.cpp$$
contains an example and test that uses a discrete function
to vary an array index during $cref Forward$$ mode calculations.
The file
$cref interp_onetape.cpp$$
contains an example and test that uses discrete
functions to avoid retaping a calculation that requires interpolation.
(The file
$cref interp_retape.cpp$$
shows how interpolation can be done with retaping.)

$head CppADCreateDiscrete Deprecated 2007-07-28$$
The preprocessor symbol $code CppADCreateDiscrete$$
is defined to be the same as $code CPPAD_DISCRETE_FUNCTION$$
but its use is deprecated.

$end
------------------------------------------------------------------------------
*/
# include <vector>
# include <cppad/core/cppad_assert.hpp>

// needed before one can use CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL
# include <cppad/utility/thread_alloc.hpp>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
/*!
\file discrete.hpp
user define discrete functions
*/

/*!
\def CPPAD_DISCRETE_FUNCTION(Base, name)
Defines the function <code>name(ax, ay)</code>
where \c ax and \c ay are vectors with <code>AD<Base></code> elements.

\par Base
is the base type for the discrete function.

\par name
is the name of the user defined function that corresponding to this operation.
*/

# define CPPAD_DISCRETE_FUNCTION(Base, name)            \
inline CppAD::AD<Base> name (const CppAD::AD<Base>& ax) \
{                                                       \
     static CppAD::discrete<Base> fun(#name, name);     \
                                                        \
     return fun.ad(ax);                                 \
}
# define CppADCreateDiscrete CPPAD_DISCRETE_FUNCTION


/*
Class that acutally implemnets the <code>ay = name(ax)</code> call.

A new discrete function is generated for ech time the user invokes
the CPPAD_DISCRETE_FUNCTION macro; see static object in that macro.
*/
template <class Base>
class discrete {
	/// parallel_ad needs to call List to initialize static
	template <class Type>
	friend void parallel_ad(void);

	/// type for the user routine that computes function values
	typedef Base (*F) (const Base& x);
private:
	/// name of this user defined function
	const std::string name_;
	/// user's implementation of the function for Base operations
	const F              f_;
	/// index of this objec in the vector of all objects for this class
	const size_t     index_;

	/*!
	List of all objects in this class.

	If we use CppAD::vector for this vector, it will appear that
	there is a memory leak because this list is not distroyed before
	thread_alloc::free_available(thread) is called by the testing routines.
	*/
	static std::vector<discrete *>& List(void)
	{	CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL;
		static std::vector<discrete *> list;
		return list;
	}
public:
	/*!
	Constructor called for each invocation of CPPAD_DISCRETE_FUNCTION.

	Put this object in the list of all objects for this class and set
	the constant private data name_, f_, and index_.

	\param Name
	is the user's name for this discrete function.

	\param f
	user routine that implements this function for \c Base class.

	\par
	This constructor can ont be used in parallel mode because it changes
	the static object \c List.
	*/
	discrete(const char* Name, F f) :
	name_(Name)
	, f_(f)
	, index_( List().size() )
	{
		CPPAD_ASSERT_KNOWN(
			! thread_alloc::in_parallel() ,
			"discrete: First call the function *Name is in parallel mode."
		);
		List().push_back(this);
	}

	/*!
	Implement the user call to <code>ay = name(ax)</code>.

	\param ax
	is the argument for this call.

	\return
	the return value is called \c ay above.
	*/
	AD<Base> ad(const AD<Base> &ax) const
	{	AD<Base> ay;

		ay.value_ = f_(ax.value_);
		if( Variable(ax) )
		{	local::ADTape<Base> *tape = ax.tape_this();
			CPPAD_ASSERT_UNKNOWN( local::NumRes(local::DisOp) == 1 );
			CPPAD_ASSERT_UNKNOWN( local::NumArg(local::DisOp) == 2 );

			// put operand addresses in the tape
			CPPAD_ASSERT_KNOWN(
				std::numeric_limits<addr_t>::max() >= index_,
				"discrete: cppad_tape_addr_type maximum not large enough"
			);
			tape->Rec_.PutArg(addr_t(index_), ax.taddr_);
			// put operator in the tape
			ay.taddr_ = tape->Rec_.PutOp(local::DisOp);
			// make result a variable
			ay.tape_id_    = tape->id_;

			CPPAD_ASSERT_UNKNOWN( Variable(ay) );
		}
		return ay;
	}

	/// Name corresponding to a discrete object
	static const char* name(size_t index)
	{	return List()[index]->name_.c_str(); }

	/*!
	Link from forward mode sweep to users routine

	\param index
	index for this function in the list of all discrete object

	\param x
	argument value at which to evaluate this function
	*/
	static Base eval(size_t index, const Base& x)
	{
		CPPAD_ASSERT_UNKNOWN(index < List().size() );

		return List()[index]->f_(x);
	}
};

} // END_CPPAD_NAMESPACE
# endif
