// $Id: omp_alloc.hpp 3804 2016-03-20 15:08:46Z bradbell $
# ifndef CPPAD_UTILITY_OMP_ALLOC_HPP
# define CPPAD_UTILITY_OMP_ALLOC_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
# include <cppad/utility/thread_alloc.hpp>
# ifdef _OPENMP
# include <omp.h>
# endif

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
class omp_alloc{
// ============================================================================
public:
/*
$begin omp_max_num_threads$$
$spell
	cppad.hpp
	inv
	CppAD
	num
	omp_alloc
$$
$section Set and Get Maximum Number of Threads for omp_alloc Allocator$$

$head Deprecated 2011-08-31$$
Use the functions $cref/thread_alloc::parallel_setup/ta_parallel_setup/$$
and $cref/thread_alloc:num_threads/ta_num_threads/$$ instead.

$head Syntax$$
$codei%# include <cppad/utility/omp_alloc.hpp>
%$$
$codei%omp_alloc::set_max_num_threads(%number%)
%$$
$icode%number% = omp_alloc::get_max_num_threads()
%$$

$head Purpose$$
By default there is only one thread and all execution is in sequential mode
(not $cref/parallel/omp_in_parallel/$$).

$head number$$
The argument and return value $icode number$$ has prototype
$codei%
	size_t %number%
%$$
and must be greater than zero.

$head set_max_num_threads$$
Informs $cref omp_alloc$$ of the maximum number of OpenMP threads.

$head get_max_num_threads$$
Returns the valued used in the previous call to $code set_max_num_threads$$.
If there was no such previous call, the value one is returned
(and only thread number zero can use $cref omp_alloc$$).

$head Restrictions$$
The function $code set_max_num_threads$$ must be called before
the program enters $cref/parallel/omp_in_parallel/$$ execution mode.
In addition, this function cannot be called while in parallel mode.

$end
*/
	/*!
	Inform omp_alloc of the maximum number of OpenMP threads and enable
	parallel execution mode by initializing all statics in this file.

	\param number [in]
	maximum number of OpenMP threads.
	*/
	static void set_max_num_threads(size_t number)
	{	thread_alloc::parallel_setup(
			number, omp_alloc::in_parallel, omp_alloc::get_thread_num
		);
		thread_alloc::hold_memory(number > 1);
	}
	/*!
	Get the current maximum number of OpenMP threads that omp_alloc can use.

	\return
	maximum number of OpenMP threads.
	*/
	static size_t get_max_num_threads(void)
	{	return thread_alloc::num_threads(); }

/* -----------------------------------------------------------------------
$begin omp_in_parallel$$

$section Is The Current Execution in OpenMP Parallel Mode$$
$mindex in_parallel$$
$spell
	cppad.hpp
	omp_alloc
	bool
$$

$head Deprecated 2011-08-31$$
Use the function $cref/thread_alloc::in_parallel/ta_in_parallel/$$ instead.

$head Syntax$$
$codei%# include <cppad/utility/omp_alloc.hpp>
%$$
$icode%flag% = omp_alloc::in_parallel()%$$

$head Purpose$$
Some of the $cref omp_alloc$$ allocation routines have different
specifications for parallel (not sequential) execution mode.
This routine enables you to determine if the current execution mode
is sequential or parallel.

$head flag$$
The return value has prototype
$codei%
	bool %flag%
%$$
It is true if the current execution is in parallel mode
(possibly multi-threaded) and false otherwise (sequential mode).

$head Example$$
$cref omp_alloc.cpp$$

$end
*/
	/// Are we in a parallel execution state; i.e., is it possible that
	/// other threads are currently executing.
	static bool in_parallel(void)
	{
# ifdef _OPENMP
		return omp_in_parallel() != 0;
# else
		return false;
# endif
	}

/* -----------------------------------------------------------------------
$begin omp_get_thread_num$$
$spell
	cppad.hpp
	CppAD
	num
	omp_alloc
	cppad.hpp
$$

$section Get the Current OpenMP Thread Number$$
$mindex get_thread_num$$

$head Deprecated 2011-08-31$$
Use the function $cref/thread_alloc::thread_num/ta_thread_num/$$ instead.

$head Syntax$$
$codei%# include <cppad/utility/omp_alloc.hpp>
%$$
$icode%thread% = omp_alloc::get_thread_num()%$$

$head Purpose$$
Some of the $cref omp_alloc$$ allocation routines have a thread number.
This routine enables you to determine the current thread.

$head thread$$
The return value $icode thread$$ has prototype
$codei%
	size_t %thread%
%$$
and is the currently executing thread number.
If $code _OPENMP$$ is not defined, $icode thread$$ is zero.

$head Example$$
$cref omp_alloc.cpp$$

$end
*/
	/// Get current OpenMP thread number (zero if _OpenMP not defined).
	static size_t get_thread_num(void)
	{
# ifdef _OPENMP
		size_t thread = static_cast<size_t>( omp_get_thread_num() );
		return thread;
# else
		return 0;
# endif
	}
/* -----------------------------------------------------------------------
$begin omp_get_memory$$
$spell
	cppad.hpp
	num
	ptr
	omp_alloc
$$

$section Get At Least A Specified Amount of Memory$$

$head Deprecated 2011-08-31$$
Use the function $cref/thread_alloc::get_memory/ta_get_memory/$$ instead.

$head Syntax$$
$codei%# include <cppad/utility/omp_alloc.hpp>
%$$
$icode%v_ptr% = omp_alloc::get_memory(%min_bytes%, %cap_bytes%)%$$

$head Purpose$$
Use $cref omp_alloc$$ to obtain a minimum number of bytes of memory
(for use by the $cref/current thread/omp_get_thread_num/$$).

$head min_bytes$$
This argument has prototype
$codei%
	size_t %min_bytes%
%$$
It specifies the minimum number of bytes to allocate.

$head cap_bytes$$
This argument has prototype
$codei%
	size_t& %cap_bytes%
%$$
It's input value does not matter.
Upon return, it is the actual number of bytes (capacity)
that have been allocated for use,
$codei%
	%min_bytes% <= %cap_bytes%
%$$

$head v_ptr$$
The return value $icode v_ptr$$ has prototype
$codei%
	void* %v_ptr%
%$$
It is the location where the $icode cap_bytes$$ of memory
that have been allocated for use begins.

$head Allocation Speed$$
This allocation should be faster if the following conditions hold:
$list number$$
The memory allocated by a previous call to $code get_memory$$
is currently available for use.
$lnext
The current $icode min_bytes$$ is between
the previous $icode min_bytes$$ and previous $icode cap_bytes$$.
$lend

$head Example$$
$cref omp_alloc.cpp$$

$end
*/
	/*!
	Use omp_alloc to get a specified amount of memory.

	If the memory allocated by a previous call to \c get_memory is now
	avaialable, and \c min_bytes is between its previous value
	and the previous \c cap_bytes, this memory allocation will have
	optimal speed. Otherwise, the memory allocation is more complicated and
	may have to wait for other threads to complete an allocation.

	\param min_bytes [in]
	The minimum number of bytes of memory to be obtained for use.

	\param cap_bytes [out]
	The actual number of bytes of memory obtained for use.

	\return
	pointer to the beginning of the memory allocted for use.
	*/
	static void* get_memory(size_t min_bytes, size_t& cap_bytes)
	{	return thread_alloc::get_memory(min_bytes, cap_bytes); }

/* -----------------------------------------------------------------------
$begin omp_return_memory$$
$spell
	cppad.hpp
	ptr
	omp_alloc
$$

$section Return Memory to omp_alloc$$
$mindex return_memory$$

$head Deprecated 2011-08-31$$
Use the function $cref/thread_alloc::return_memory/ta_return_memory/$$ instead.

$head Syntax$$
$codei%# include <cppad/utility/omp_alloc.hpp>
%$$
$codei%omp_alloc::return_memory(%v_ptr%)%$$

$head Purpose$$
If $cref omp_max_num_threads$$ is one,
the memory is returned to the system.
Otherwise, the memory is retained by $cref omp_alloc$$ for quick future use
by the thread that allocated to memory.

$head v_ptr$$
This argument has prototype
$codei%
	void* %v_ptr%
%$$.
It must be a pointer to memory that is currently in use; i.e.
obtained by a previous call to $cref omp_get_memory$$ and not yet returned.

$head Thread$$
Either the $cref/current thread/omp_get_thread_num/$$ must be the same as during
the corresponding call to $cref omp_get_memory$$,
or the current execution mode must be sequential
(not $cref/parallel/omp_in_parallel/$$).

$head NDEBUG$$
If $code NDEBUG$$ is defined, $icode v_ptr$$ is not checked (this is faster).
Otherwise, a list of in use pointers is searched to make sure
that $icode v_ptr$$ is in the list.

$head Example$$
$cref omp_alloc.cpp$$

$end
*/
	/*!
	Return memory that was obtained by \c get_memory.
	If  <code>max_num_threads(0) == 1</code>,
	the memory is returned to the system.
	Otherwise, it is retained by \c omp_alloc and available for use by
	\c get_memory for this thread.

	\param v_ptr [in]
	Value of the pointer returned by \c get_memory and still in use.
	After this call, this pointer will available (and not in use).

	\par
	We must either be in sequential (not parallel) execution mode,
	or the current thread must be the same as for the corresponding call
	to \c get_memory.
	*/
	static void return_memory(void* v_ptr)
	{	thread_alloc::return_memory(v_ptr); }
/* -----------------------------------------------------------------------
$begin omp_free_available$$
$spell
	cppad.hpp
	omp_alloc
$$

$section Free Memory Currently Available for Quick Use by a Thread$$
$mindex free_available$$

$head Deprecated 2011-08-31$$
Use the function $cref/thread_alloc::free_available/ta_free_available/$$
instead.

$head Syntax$$
$codei%# include <cppad/utility/omp_alloc.hpp>
%$$
$codei%omp_alloc::free_available(%thread%)%$$

$head Purpose$$
Free memory, currently available for quick use by a specific thread,
for general future use.

$head thread$$
This argument has prototype
$codei%
	size_t %thread%
%$$
Either $cref omp_get_thread_num$$ must be the same as $icode thread$$,
or the current execution mode must be sequential
(not $cref/parallel/omp_in_parallel/$$).

$head Example$$
$cref omp_alloc.cpp$$

$end
*/
	/*!
	Return all the memory being held as available for a thread to the system.

	\param thread [in]
	this thread that will no longer have any available memory after this call.
	This must either be the thread currently executing, or we must be
	in sequential (not parallel) execution mode.
	*/
	static void free_available(size_t thread)
	{	thread_alloc::free_available(thread); }
/* -----------------------------------------------------------------------
$begin omp_inuse$$
$spell
	cppad.hpp
	num
	inuse
	omp_alloc
$$

$section Amount of Memory a Thread is Currently Using$$
$mindex inuse$$

$head Deprecated 2011-08-31$$

$head Syntax$$
$codei%# include <cppad/utility/omp_alloc.hpp>
%$$
$icode%num_bytes% = omp_alloc::inuse(%thread%)%$$
Use the function $cref/thread_alloc::inuse/ta_inuse/$$ instead.

$head Purpose$$
Memory being managed by $cref omp_alloc$$ has two states,
currently in use by the specified thread,
and quickly available for future use by the specified thread.
This function informs the program how much memory is in use.

$head thread$$
This argument has prototype
$codei%
	size_t %thread%
%$$
Either $cref omp_get_thread_num$$ must be the same as $icode thread$$,
or the current execution mode must be sequential
(not $cref/parallel/omp_in_parallel/$$).

$head num_bytes$$
The return value has prototype
$codei%
	size_t %num_bytes%
%$$
It is the number of bytes currently in use by the specified thread.

$head Example$$
$cref omp_alloc.cpp$$

$end
*/
	/*!
	Determine the amount of memory that is currently inuse.

	\param thread [in]
	Thread for which we are determining the amount of memory
	(must be < CPPAD_MAX_NUM_THREADS).
	Durring parallel execution, this must be the thread
	that is currently executing.

	\return
	The amount of memory in bytes.
	*/
	static size_t inuse(size_t thread)
	{	return thread_alloc::inuse(thread); }
/* -----------------------------------------------------------------------
$begin omp_available$$
$spell
	cppad.hpp
	num
	omp_alloc
$$

$section Amount of Memory Available for Quick Use by a Thread$$

$head Deprecated 2011-08-31$$
Use the function $cref/thread_alloc::available/ta_available/$$ instead.

$head Syntax$$
$codei%# include <cppad/utility/omp_alloc.hpp>
%$$
$icode%num_bytes% = omp_alloc::available(%thread%)%$$

$head Purpose$$
Memory being managed by $cref omp_alloc$$ has two states,
currently in use by the specified thread,
and quickly available for future use by the specified thread.
This function informs the program how much memory is available.

$head thread$$
This argument has prototype
$codei%
	size_t %thread%
%$$
Either $cref omp_get_thread_num$$ must be the same as $icode thread$$,
or the current execution mode must be sequential
(not $cref/parallel/omp_in_parallel/$$).

$head num_bytes$$
The return value has prototype
$codei%
	size_t %num_bytes%
%$$
It is the number of bytes currently available for use by the specified thread.

$head Example$$
$cref omp_alloc.cpp$$

$end
*/
	/*!
	Determine the amount of memory that is currently available for use.

	\copydetails inuse
	*/
	static size_t available(size_t thread)
	{	return thread_alloc::available(thread); }
/* -----------------------------------------------------------------------
$begin omp_create_array$$
$spell
	cppad.hpp
	omp_alloc
	sizeof
$$

$section Allocate Memory and Create A Raw Array$$
$mindex create_array$$

$head Deprecated 2011-08-31$$
Use the function $cref/thread_alloc::create_array/ta_create_array/$$ instead.

$head Syntax$$
$codei%# include <cppad/utility/omp_alloc.hpp>
%$$
$icode%array% = omp_alloc::create_array<%Type%>(%size_min%, %size_out%)%$$.

$head Purpose$$
Create a new raw array using $cref omp_alloc$$ a fast memory allocator
that works well in a multi-threading OpenMP environment.

$head Type$$
The type of the elements of the array.

$head size_min$$
This argument has prototype
$codei%
	size_t %size_min%
%$$
This is the minimum number of elements that there can be
in the resulting $icode array$$.

$head size_out$$
This argument has prototype
$codei%
	size_t& %size_out%
%$$
The input value of this argument does not matter.
Upon return, it is the actual number of elements
in $icode array$$
($icode% size_min %<=% size_out%$$).

$head array$$
The return value $icode array$$ has prototype
$codei%
	%Type%* %array%
%$$
It is array with $icode size_out$$ elements.
The default constructor for $icode Type$$ is used to initialize the
elements of $icode array$$.
Note that $cref omp_delete_array$$
should be used to destroy the array when it is no longer needed.

$head Delta$$
The amount of memory $cref omp_inuse$$ by the current thread,
will increase $icode delta$$ where
$codei%
	sizeof(%Type%) * (%size_out% + 1) > %delta% >= sizeof(%Type%) * %size_out%
%$$
The $cref omp_available$$ memory will decrease by $icode delta$$,
(and the allocation will be faster)
if a previous allocation with $icode size_min$$ between its current value
and $icode size_out$$ is available.

$head Example$$
$cref omp_alloc.cpp$$

$end
*/
	/*!
	Use omp_alloc to Create a Raw Array.

	\tparam Type
	The type of the elements of the array.

	\param size_min [in]
	The minimum number of elements in the array.

	\param size_out [out]
	The actual number of elements in the array.

	\return
	pointer to the first element of the array.
	The default constructor is used to initialize
	all the elements of the array.

	\par
	The \c extra_ field, in the \c omp_alloc node before the return value,
	is set to size_out.
	*/
	template <class Type>
	static Type* create_array(size_t size_min, size_t& size_out)
	{	return thread_alloc::create_array<Type>(size_min, size_out); }
/* -----------------------------------------------------------------------
$begin omp_delete_array$$
$spell
	cppad.hpp
	omp_alloc
	sizeof
$$

$section Return A Raw Array to The Available Memory for a Thread$$
$mindex delete_array$$

$head Deprecated 2011-08-31$$
Use the function $cref/thread_alloc::delete_array/ta_delete_array/$$ instead.

$head Syntax$$
$codei%# include <cppad/utility/omp_alloc.hpp>
%$$
$codei%omp_alloc::delete_array(%array%)%$$.

$head Purpose$$
Returns memory corresponding to a raw array
(create by $cref omp_create_array$$) to the
$cref omp_available$$ memory pool for the current thread.

$head Type$$
The type of the elements of the array.

$head array$$
The argument $icode array$$ has prototype
$codei%
	%Type%* %array%
%$$
It is a value returned by $cref omp_create_array$$ and not yet deleted.
The $icode Type$$ destructor is called for each element in the array.

$head Thread$$
The $cref/current thread/omp_get_thread_num/$$ must be the
same as when $cref omp_create_array$$ returned the value $icode array$$.
There is an exception to this rule:
when the current execution mode is sequential
(not $cref/parallel/omp_in_parallel/$$) the current thread number does not matter.

$head Delta$$
The amount of memory $cref omp_inuse$$ will decrease by $icode delta$$,
and the $cref omp_available$$ memory will increase by $icode delta$$,
where $cref/delta/omp_create_array/Delta/$$
is the same as for the corresponding call to $code create_array$$.

$head Example$$
$cref omp_alloc.cpp$$

$end
*/
	/*!
	Return Memory Used for a Raw Array to the Available Pool.

	\tparam Type
	The type of the elements of the array.

	\param array [in]
	A value returned by \c create_array that has not yet been deleted.
	The \c Type destructor is used to destroy each of the elements
	of the array.

	\par
	Durring parallel execution, the current thread must be the same
	as during the corresponding call to \c create_array.
	*/
	template <class Type>
	static void delete_array(Type* array)
	{	thread_alloc::delete_array(array); }
};
/* --------------------------------------------------------------------------
$begin omp_efficient$$
$spell
	cppad.hpp
	omp_alloc
	ptr
	num
	bool
	const
$$

$section Check If A Memory Allocation is Efficient for Another Use$$

$head Removed$$
This function has been removed because speed tests seem to indicate
it is just as fast, or faster, to free and then reallocate the memory.

$head Syntax$$
$codei%# include <cppad/utility/omp_alloc.hpp>
%$$
$icode%flag% = omp_alloc::efficient(%v_ptr%, %num_bytes%)%$$

$head Purpose$$
Check if memory that is currently in use is an efficient
allocation for a specified number of bytes.

$head v_ptr$$
This argument has prototype
$codei%
	const void* %v_ptr%
%$$.
It must be a pointer to memory that is currently in use; i.e.
obtained by a previous call to $cref omp_get_memory$$ and not yet returned.

$head num_bytes$$
This argument has prototype
$codei%
	size_t %num_bytes%
%$$
It specifies the number of bytes of the memory allocated by $icode v_ptr$$
that we want to use.

$head flag$$
The return value has prototype
$codei%
	bool %flag%
%$$
It is true,
a call to $code get_memory$$ with
$cref/min_bytes/omp_get_memory/min_bytes/$$
equal to $icode num_bytes$$ would result in a value for
$cref/cap_bytes/omp_get_memory/cap_bytes/$$ that is the same as when $code v_ptr$$
was returned by $code get_memory$$; i.e.,
$icode v_ptr$$ is an efficient memory block for $icode num_bytes$$
bytes of information.

$head Thread$$
Either the $cref/current thread/omp_get_thread_num/$$ must be the same as during
the corresponding call to $cref omp_get_memory$$,
or the current execution mode must be sequential
(not $cref/parallel/omp_in_parallel/$$).

$head NDEBUG$$
If $code NDEBUG$$ is defined, $icode v_ptr$$ is not checked (this is faster).
Otherwise, a list of in use pointers is searched to make sure
that $icode v_ptr$$ is in the list.

$end
---------------------------------------------------------------------------
$begin old_max_num_threads$$
$spell
	cppad.hpp
	inv
	CppAD
	num
	omp_alloc
$$
$section Set Maximum Number of Threads for omp_alloc Allocator$$
$mindex max_num_threads$$

$head Removed$$
This function has been removed from the CppAD API.
Use the function $cref/thread_alloc::parallel_setup/ta_parallel_setup/$$
in its place.

$head Syntax$$
$codei%# include <cppad/utility/omp_alloc.hpp>
%$$
$codei%omp_alloc::max_num_threads(%number%)%$$

$head Purpose$$
By default there is only one thread and all execution is in sequential mode
(not $cref/parallel/omp_in_parallel/$$).

$head number$$
The argument $icode number$$ has prototype
$codei%
	size_t %number%
%$$
It must be greater than zero and specifies the maximum number of
OpenMP threads that will be active at one time.

$head Restrictions$$
This function must be called before the program enters
$cref/parallel/omp_in_parallel/$$ execution mode.

$end
-------------------------------------------------------------------------------
*/
} // END_CPPAD_NAMESPACE

# endif
