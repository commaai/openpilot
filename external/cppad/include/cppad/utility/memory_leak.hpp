# ifndef CPPAD_UTILITY_MEMORY_LEAK_HPP
# define CPPAD_UTILITY_MEMORY_LEAK_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin memory_leak$$
$spell
	cppad
	num
	alloc
	hpp
	bool
	inuse
$$

$section Memory Leak Detection$$
$mindex memory_leak check static$$

$head Deprecated 2012-04-06$$
This routine has been deprecated.
You should instead use the routine $cref ta_free_all$$.

$head Syntax$$
$codei%# include <cppad/utility/memory_leak.hpp>
%$$
$icode%flag% = %memory_leak()
%$$
$icode%flag% = %memory_leak%(%add_static%)%$$

$head Purpose$$
This routine checks that the are no memory leaks
caused by improper use of $cref thread_alloc$$ memory allocator.
The deprecated memory allocator $cref TrackNewDel$$ is also checked.
Memory errors in the deprecated $cref omp_alloc$$ allocator are
reported as being in $code thread_alloc$$.

$head thread$$
It is assumed that $cref/in_parallel()/ta_in_parallel/$$ is false
and $cref/thread_num/ta_thread_num/$$ is zero when
$code memory_leak$$ is called.

$head add_static$$
This argument has prototype
$codei%
	size_t %add_static%
%$$
and its default value is zero.
Static variables hold onto memory forever.
If the argument $icode add_static$$ is present (and non-zero),
$code memory_leak$$ adds this amount of memory to the
$cref/inuse/ta_inuse/$$ sum that corresponds to
static variables in the program.
A call with $icode add_static$$ should be make after
a routine that has static variables which
use $cref/get_memory/ta_get_memory/$$ to allocate memory.
The value of $icode add_static$$ should be the difference of
$codei%
	thread_alloc::inuse(0)
%$$
before and after the call.
Since multiple statics may be allocated in different places in the program,
it is expected that there will be multiple calls
that use this option.

$head flag$$
The return value $icode flag$$ has prototype
$codei%
	bool %flag%
%$$
If $icode add_static$$ is non-zero,
the return value for $code memory_leak$$ is false.
Otherwise, the return value for $code memory_leak$$ should be false
(indicating that the only allocated memory corresponds to static variables).

$head inuse$$
It is assumed that, when $code memory_leak$$ is called,
there should not be any memory
$cref/inuse/ta_inuse/$$ or $cref omp_inuse$$ for any thread
(except for inuse memory corresponding to static variables).
If there is, a message is printed and $code memory_leak$$ returns false.

$head available$$
It is assumed that, when $code memory_leak$$ is called,
there should not be any memory
$cref/available/ta_available/$$ or $cref omp_available$$ for any thread;
i.e., it all has been returned to the system.
If there is memory still available for any thread,
$code memory_leak$$ returns false.

$head TRACK_COUNT$$
It is assumed that, when $code memory_leak$$ is called,
$cref/TrackCount/TrackNewDel/TrackCount/$$ will return a zero value.
If it returns a non-zero value,
$code memory_leak$$ returns false.

$head Error Message$$
If this is the first call to $code memory_leak$$, no message is printed.
Otherwise, if it returns true, an error message is printed
to standard output describing the memory leak that was detected.

$end
*/
# include <iostream>
# include <cppad/core/define.hpp>
# include <cppad/utility/omp_alloc.hpp>
# include <cppad/utility/thread_alloc.hpp>
# include <cppad/utility/track_new_del.hpp>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
/*!
\file memory_leak.hpp
File that implements a memory check at end of a CppAD program
*/

/*!
Function that checks
allocator \c thread_alloc for misuse that results in memory leaks.
Deprecated routines in track_new_del.hpp and omp_alloc.hpp are also checked.

\param add_static [in]
The amount specified by \c add_static is added to the amount
of memory that is expected to be used by thread zero for static variables.

\return
If \c add_static is non-zero, the return value is \c false.
Otherwise, if one of the following errors is detected,
the return value is \c true:

\li
Thread zero does not have the expected amount of inuse memory
(for static variables).
\li
A thread, other than thread zero, has any inuse memory.
\li
Any thread has available memory.

\par
If an error is detected, diagnostic information is printed to standard
output.
*/
inline bool memory_leak(size_t add_static = 0)
{	// CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL not necessary given asserts below
	static size_t thread_zero_static_inuse     = 0;
	using std::cout;
	using std::endl;
	using CppAD::thread_alloc;
	using CppAD::omp_alloc;
	// --------------------------------------------------------------------
	CPPAD_ASSERT_KNOWN(
		! thread_alloc::in_parallel(),
		"memory_leak: in_parallel() is true."
	);
	CPPAD_ASSERT_KNOWN(
		thread_alloc::thread_num() == 0,
		"memory_leak: thread_num() is not zero."
	);
	if( add_static != 0 )
	{	thread_zero_static_inuse += add_static;
		return false;
	}
	bool leak                 = false;
	size_t thread             = 0;

	// check that memory in use for thread zero corresponds to statics
	size_t num_bytes = thread_alloc::inuse(thread);
	if( num_bytes != thread_zero_static_inuse )
	{	leak = true;
		cout << "thread zero: static inuse = " << thread_zero_static_inuse;
		cout << ", current inuse(0)= " << num_bytes << endl;
	}
	// check that no memory is currently available for this thread
	num_bytes = thread_alloc::available(thread);
	if( num_bytes != 0 )
	{	leak = true;
		cout << "thread zero: available    = ";
		cout << num_bytes << endl;
	}
	for(thread = 1; thread < CPPAD_MAX_NUM_THREADS; thread++)
	{
		// check that no memory is currently in use for this thread
		num_bytes = thread_alloc::inuse(thread);
		if( num_bytes != 0 )
		{	leak = true;
			cout << "thread " << thread << ": inuse(thread) = ";
			cout << num_bytes << endl;
		}
		// check that no memory is currently available for this thread
		num_bytes = thread_alloc::available(thread);
		if( num_bytes != 0 )
		{	leak = true;
			cout << "thread " << thread << ": available(thread) = ";
			cout << num_bytes << endl;
		}
	}
	// ----------------------------------------------------------------------
	// check track_new_del
	if( CPPAD_TRACK_COUNT() != 0 )
	{	leak = true;
		CppAD::TrackElement::Print();
	}
	return leak;
}

} // END_CPPAD_NAMESPACE
# endif
