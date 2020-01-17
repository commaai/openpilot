# ifndef CPPAD_UTILITY_TRACK_NEW_DEL_HPP
# define CPPAD_UTILITY_TRACK_NEW_DEL_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin TrackNewDel$$
$spell
	cppad.hpp
	Cpp
	newptr
	Vec
	oldptr
	newlen
	ncopy
	const
$$

$section Routines That Track Use of New and Delete$$
$mindex memory NDEBUG CPPAD_TRACK_NEW_VEC CppADTrackNewVec CPPAD_TRACK_DEL_VEC CppADTrackDelVec CPPAD_TRACK_EXTEND CppADTrackExtend CPPAD_TRACK_COUNT thread multi$$

$head Deprecated 2007-07-23$$
All these routines have been deprecated.
You should use the $cref thread_alloc$$ memory allocator instead
(which works better in both a single thread and
properly in multi-threading environment).

$head Syntax$$
$codei%# include <cppad/utility/track_new_del.hpp>
%$$
$icode%newptr% = TrackNewVec(%file%, %line%, %newlen%, %oldptr%)
%$$
$codei%TrackDelVec(%file%, %line%, %oldptr%)
%$$
$icode%newptr% = TrackExtend(%file%, %line%, %newlen%, %ncopy%, %oldptr%)
%$$
$icode%count% = TrackCount(%file%, %line%)%$$


$head Purpose$$
These routines
aid in the use of $code new[]$$ and  $code delete[]$$
during the execution of a C++ program.

$head Include$$
The file $code cppad/track_new_del.hpp$$ is included by
$code cppad/cppad.hpp$$
but it can also be included separately with out the rest of the
CppAD include files.


$head file$$
The argument $icode file$$ has prototype
$codei%
	const char *%file%
%$$
It should be the source code file name
where the call to $code TrackNew$$ is located.
The best way to accomplish this is the use the preprocessor symbol
$code __FILE__$$ for this argument.

$head line$$
The argument $icode line$$ has prototype
$codei%
	int %line%
%$$
It should be the source code file line number
where the call to $code TrackNew$$ is located.
The best way to accomplish this is the use the preprocessor symbol
$code __LINE__$$ for this argument.

$head oldptr$$
The argument $icode oldptr$$ has prototype
$codei%
	%Type% *%oldptr%
%$$
This argument is used to identify the type $icode Type$$.

$head newlen$$
The argument $icode newlen$$ has prototype
$codei%
	size_t %newlen%
%$$

$head head newptr$$
The return value $icode newptr$$ has prototype
$codei%
	%Type% *%newptr%
%$$
It points to the newly allocated vector of objects
that were allocated using
$codei%
	new Type[%newlen%]
%$$

$head ncopy$$
The argument $icode ncopy$$ has prototype
$codei%
        size_t %ncopy%
%$$
This specifies the number of elements that are copied from
the old array to the new array.
The value of $icode ncopy$$
must be less than or equal $icode newlen$$.

$head TrackNewVec$$
If $code NDEBUG$$ is defined, this routine only sets
$codei%
	%newptr% = %Type% new[%newlen%]
%$$
The value of $icode oldptr$$ does not matter
(except that it is used to identify $icode Type$$).
If $code NDEBUG$$ is not defined, $code TrackNewVec$$ also
tracks the this memory allocation.
In this case, if memory cannot be allocated
$cref ErrorHandler$$ is used to generate a message
stating that there was not sufficient memory.

$subhead Macro$$
The preprocessor macro call
$codei%
	CPPAD_TRACK_NEW_VEC(%newlen%, %oldptr%)
%$$
expands to
$codei%
	CppAD::TrackNewVec(__FILE__, __LINE__, %newlen%, %oldptr%)
%$$

$subhead Previously Deprecated$$
The preprocessor macro $code CppADTrackNewVec$$ is the
same as $code CPPAD_TRACK_NEW_VEC$$ and was previously deprecated.

$head TrackDelVec$$
This routine is used to a vector of objects
that have been allocated using $code TrackNew$$ or $code TrackExtend$$.
If $code NDEBUG$$ is defined, this routine only frees memory with
$codei%
	delete [] %oldptr%
%$$
If $code NDEBUG$$ is not defined, $code TrackDelete$$ also checks that
$icode oldptr$$ was allocated by $code TrackNew$$ or $code TrackExtend$$
and has not yet been freed.
If this is not the case,
$cref ErrorHandler$$ is used to generate an error message.

$subhead Macro$$
The preprocessor macro call
$codei%
	CPPAD_TRACK_DEL_VEC(%oldptr%)
%$$
expands to
$codei%
	CppAD::TrackDelVec(__FILE__, __LINE__, %oldptr%)
%$$

$subhead Previously Deprecated$$
The preprocessor macro $code CppADTrackDelVec$$ is the
same as $code CPPAD_TRACK_DEL_VEC$$ was previously deprecated.

$head TrackExtend$$
This routine is used to
allocate a new vector (using $code TrackNewVec$$),
copy $icode ncopy$$ elements from the old vector to the new vector.
If $icode ncopy$$ is greater than zero, $icode oldptr$$
must have been allocated using $code TrackNewVec$$ or $code TrackExtend$$.
In this case, the vector pointed to by $icode oldptr$$
must be have at least $icode ncopy$$ elements
and it will be deleted (using $code TrackDelVec$$).
Note that the dependence of $code TrackExtend$$ on $code NDEBUG$$
is indirectly through the routines $code TrackNewVec$$ and
$code TrackDelVec$$.

$subhead Macro$$
The preprocessor macro call
$codei%
	CPPAD_TRACK_EXTEND(%newlen%, %ncopy%, %oldptr%)
%$$
expands to
$codei%
	CppAD::TrackExtend(__FILE__, __LINE__, %newlen%, %ncopy%, %oldptr%)
%$$

$subhead Previously Deprecated$$
The preprocessor macro $code CppADTrackExtend$$ is the
same as $code CPPAD_TRACK_EXTEND$$ and was previously deprecated.

$head TrackCount$$
The return value $icode count$$ has prototype
$codei%
	size_t %count%
%$$
If $code NDEBUG$$ is defined, $icode count$$ will be zero.
Otherwise, it will be
the number of vectors that
have been allocated
(by $code TrackNewVec$$ or $code TrackExtend$$)
and not yet freed
(by $code TrackDelete$$).

$subhead Macro$$
The preprocessor macro call
$codei%
	CPPAD_TRACK_COUNT()
%$$
expands to
$codei%
	CppAD::TrackCount(__FILE__, __LINE__)
%$$

$subhead Previously Deprecated$$
The preprocessor macro $code CppADTrackCount$$ is the
same as $code CPPAD_TRACK_COUNT$$ and was previously deprecated.

$head Multi-Threading$$
These routines cannot be used $cref/in_parallel/ta_in_parallel/$$
execution mode.
Use the $cref thread_alloc$$ routines instead.

$head Example$$
$children%
	example/deprecated/track_new_del.cpp
%$$
The file $cref TrackNewDel.cpp$$
contains an example and test of these functions.
It returns true, if it succeeds, and false otherwise.

$end
------------------------------------------------------------------------------
*/
# include <cppad/core/define.hpp>
# include <cppad/core/cppad_assert.hpp>
# include <cppad/utility/thread_alloc.hpp>
# include <sstream>
# include <string>

# ifndef CPPAD_TRACK_DEBUG
# define CPPAD_TRACK_DEBUG 0
# endif

// -------------------------------------------------------------------------
# define CPPAD_TRACK_NEW_VEC(newlen, oldptr) \
	CppAD::TrackNewVec(__FILE__, __LINE__, newlen, oldptr)

# define CPPAD_TRACK_DEL_VEC(oldptr) \
	CppAD::TrackDelVec(__FILE__, __LINE__, oldptr)

# define CPPAD_TRACK_EXTEND(newlen, ncopy, oldptr) \
	CppAD::TrackExtend(__FILE__, __LINE__, newlen, ncopy, oldptr)

# define CPPAD_TRACK_COUNT() \
	CppAD::TrackCount(__FILE__, __LINE__)
// -------------------------------------------------------------------------
# define CppADTrackNewVec CPPAD_TRACK_NEW_VEC
# define CppADTrackDelVec CPPAD_TRACK_DEL_VEC
# define CppADTrackExtend CPPAD_TRACK_EXTEND
# define CppADTrackCount  CPPAD_TRACK_COUNT
// -------------------------------------------------------------------------
namespace CppAD { // Begin CppAD namespace

// TrackElement ------------------------------------------------------------
class TrackElement {

public:
	std::string   file;   // corresponding file name
	int           line;   // corresponding line number
	void          *ptr;   // value returned by TrackNew
	TrackElement *next;   // next element in linked list

	// default contructor (used to initialize root)
	TrackElement(void)
	: file(""), line(0), ptr(CPPAD_NULL), next(CPPAD_NULL)
	{ }

	TrackElement(const char *f, int l, void *p)
	: file(f), line(l), ptr(p), next(CPPAD_NULL)
	{	CPPAD_ASSERT_UNKNOWN( p != CPPAD_NULL);
	}

	// There is only one tracking list and it starts it here
	static TrackElement *Root(void)
	{	CPPAD_ASSERT_UNKNOWN( ! thread_alloc::in_parallel() );
		static TrackElement root;
		return &root;
	}

	// Print one tracking element
	static void Print(TrackElement* E)
	{
		CPPAD_ASSERT_UNKNOWN( ! thread_alloc::in_parallel() );
		using std::cout;
		cout << "E = "         << E;
		cout << ", E->next = " << E->next;
		cout << ", E->ptr  = " << E->ptr;
		cout << ", E->line = " << E->line;
		cout << ", E->file = " << E->file;
		cout << std::endl;
	}

	// Print the linked list for a thread
	static void Print(void)
	{
		CPPAD_ASSERT_UNKNOWN( ! thread_alloc::in_parallel() );
		using std::cout;
		using std::endl;
		TrackElement *E = Root();
		// convert int(size_t) to avoid warning on _MSC_VER systems
		cout << "Begin Track List" << endl;
		while( E->next != CPPAD_NULL )
		{	E = E->next;
			Print(E);
		}
		cout << "End Track List:" << endl;
		cout << endl;
	}
};


// TrackError ----------------------------------------------------------------
inline void TrackError(
	const char *routine,
	const char *file,
	int         line,
	const char *msg )
{
	CPPAD_ASSERT_UNKNOWN( ! thread_alloc::in_parallel() );
	std::ostringstream buf;
	buf << routine
	    << ": at line "
	    << line
	    << " in file "
	    << file
	    << std::endl
	    << msg;
	std::string str = buf.str();
	size_t      n   = str.size();
	size_t i;
	char *message = new char[n + 1];
	for(i = 0; i < n; i++)
		message[i] = str[i];
	message[n] = '\0';
	CPPAD_ASSERT_KNOWN( false , message);
}

// TrackNewVec ---------------------------------------------------------------
# ifdef NDEBUG
template <class Type>
inline Type *TrackNewVec(
	const char *file, int line, size_t len, Type * /* oldptr */ )
{
# if CPPAD_TRACK_DEBUG
	static bool first = true;
	if( first )
	{	std::cout << "NDEBUG is defined for TrackNewVec" << std::endl;
		first = false;
	}
# endif
	return (new Type[len]);
}

# else

template <class Type>
Type *TrackNewVec(
	const char *file          ,
	int         line          ,
	size_t      len           ,
	Type       * /* oldptr */ )
{
	CPPAD_ASSERT_KNOWN(
		! thread_alloc::in_parallel() ,
		"attempt to use TrackNewVec in parallel execution mode."
	);
	// try to allocate the new memrory
	Type *newptr = CPPAD_NULL;
	try
	{	newptr = new Type[len];
	}
	catch(...)
	{	TrackError("TrackNewVec", file, line,
			"Cannot allocate sufficient memory"
		);
	}
	// create tracking element
	void *vptr = static_cast<void *>(newptr);
	TrackElement *E = new TrackElement(file, line, vptr);

	// get the root
	TrackElement *root = TrackElement::Root();

	// put this elemenent at the front of linked list
	E->next    = root->next;
	root->next = E;

# if CPPAD_TRACK_DEBUG
	std::cout << "TrackNewVec: ";
	TrackElement::Print(E);
# endif

	return newptr;
}

# endif

// TrackDelVec --------------------------------------------------------------
# ifdef NDEBUG
template <class Type>
inline void TrackDelVec(const char *file, int line, Type *oldptr)
{
# if CPPAD_TRACK_DEBUG
	static bool first = true;
	if( first )
	{	std::cout << "NDEBUG is defined in TrackDelVec" << std::endl;
		first = false;
	}
# endif
	 delete [] oldptr;
}

# else

template <class Type>
void TrackDelVec(
	const char *file    ,
	int         line    ,
	Type       *oldptr  )
{
	CPPAD_ASSERT_KNOWN(
		! thread_alloc::in_parallel() ,
		"attempt to use TrackDelVec in parallel execution mode."
	);
	TrackElement        *P;
	TrackElement        *E;

	// search list for pointer
	P          = TrackElement::Root();
	E          = P->next;
	void *vptr = static_cast<void *>(oldptr);
	while(E != CPPAD_NULL && E->ptr != vptr)
	{	P = E;
		E = E->next;
	}

	// check if pointer was not in list
	if( E == CPPAD_NULL || E->ptr != vptr ) TrackError(
		"TrackDelVec", file, line,
		"Invalid value for the argument oldptr.\n"
		"Possible linking of debug and NDEBUG compilations of CppAD."
	);

# if CPPAD_TRACK_DEBUG
	std::cout << "TrackDelVec: ";
	TrackElement::Print(E);
# endif

	// remove tracking element from list
	P->next = E->next;

	// delete allocated pointer
	delete [] oldptr;

	// delete tracking element
	delete E;

	return;
}

# endif

// TrackExtend --------------------------------------------------------------
template <class Type>
Type *TrackExtend(
	const char *file    ,
	int         line    ,
	size_t      newlen  ,
	size_t      ncopy   ,
	Type       *oldptr  )
{
	CPPAD_ASSERT_KNOWN(
		! thread_alloc::in_parallel() ,
		"attempt to use TrackExtend in parallel execution mode."
	);

# if CPPAD_TRACK_DEBUG
	using std::cout;
	cout << "TrackExtend: file = " << file;
	cout << ", line = " << line;
	cout << ", newlen = " << newlen;
	cout << ", ncopy = " << ncopy;
	cout << ", oldptr = " << oldptr;
	cout << std::endl;
# endif
	CPPAD_ASSERT_KNOWN(
		ncopy <= newlen,
		"TrackExtend: ncopy is greater than newlen."
	);

	// allocate the new memrory
	Type *newptr = TrackNewVec(file, line, newlen, oldptr);

	// copy the data
	size_t i;
	for(i = 0; i < ncopy; i++)
		newptr[i] = oldptr[i];

	// delete the old vector
	if( ncopy > 0 )
		TrackDelVec(file, line, oldptr);

	return newptr;
}

// TrackCount --------------------------------------------------------------
inline size_t TrackCount(const char *file, int line)
{
	CPPAD_ASSERT_KNOWN(
		! thread_alloc::in_parallel() ,
		"attempt to use TrackCount in parallel execution mode."
	);
	size_t count = 0;
	TrackElement *E = TrackElement::Root();
	while( E->next != CPPAD_NULL )
	{	++count;
		E = E->next;
	}
	return count;
}
// ---------------------------------------------------------------------------

} // End CppAD namespace

// preprocessor symbols local to this file
# undef CPPAD_TRACK_DEBUG

# endif
