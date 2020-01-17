// $Id: elapsed_seconds.hpp 3845 2016-11-19 01:50:47Z bradbell $
# ifndef CPPAD_UTILITY_ELAPSED_SECONDS_HPP
# define CPPAD_UTILITY_ELAPSED_SECONDS_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin elapsed_seconds$$
$spell
	cppad.hpp
	Microsoft
	gettimeofday
	std
	chrono
$$

$section Returns Elapsed Number of Seconds$$
$mindex elapsed_seconds time$$


$head Syntax$$
$codei%# include <cppad/utility/elapsed_seconds.hpp>
%$$
$icode%s% = elapsed_seconds()%$$

$head Purpose$$
This routine is accurate to within .02 seconds
(see $cref elapsed_seconds.cpp$$).
It does not necessary work for time intervals that are greater than a day.
$list number$$
If the C++11 $code std::chrono::steady_clock$$ is available,
it will be used for timing.
$lnext
Otherwise, if running under the Microsoft compiler,
$code ::GetSystemTime$$ will be used for timing.
$lnext
Otherwise, if $code gettimeofday$$ is available, it is used for timing.
$lnext
Otherwise, $code std::clock()$$ will be used for timing.
$lend

$head s$$
is a $code double$$ equal to the
number of seconds since the first call to $code elapsed_seconds$$.

$head Microsoft Systems$$
It you are using $code ::GetSystemTime$$,
you will need to link in the external routine
called $cref microsoft_timer$$.

$children%
	speed/example/elapsed_seconds.cpp
%$$
$head Example$$
The routine $cref elapsed_seconds.cpp$$ is
an example and test of this routine.


$end
-----------------------------------------------------------------------
*/

// For some unknown reason under Fedora (which needs to be understood),
// if you move this include for cppad_assert.hpp below include for define.hpp,
//		cd work/speed/example
//		make test.sh
// fails with the error message 'gettimeofday' not defined.
# include <cppad/core/cppad_assert.hpp>

// define CPPAD_NULL
# include <cppad/core/define.hpp>

// needed before one can use CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL
# include <cppad/utility/thread_alloc.hpp>

# if CPPAD_USE_CPLUSPLUS_2011
# include <chrono>
# elif _MSC_VER
extern double microsoft_timer(void);
# elif CPPAD_HAS_GETTIMEOFDAY
# include <sys/time.h>
# else
# include <ctime>
# endif

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
/*!
\file elapsed_seconds.hpp
\brief Function that returns the elapsed seconds from first call.
*/

/*!
Returns the elapsed number since the first call to this function.

This routine tries is accurate to within .02 seconds.
It does not necessary work for time intervals that are less than a day.
\li
If running under the Microsoft system, it uses \c ::%GetSystemTime for timing.
\li
Otherwise, if \c gettimeofday is available, it is used.
\li
Otherwise, \c std::clock() is used.

\return
The number of seconds since the first call to \c elapsed_seconds.
*/
inline double elapsed_seconds(void)
// --------------------------------------------------------------------------
# if CPPAD_USE_CPLUSPLUS_2011
{	CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL;
	static bool first_ = true;
	static std::chrono::time_point<std::chrono::steady_clock> start_;
	if( first_ )
	{	start_ = std::chrono::steady_clock::now();
		first_ = false;
		return 0.0;
	}
	std::chrono::time_point<std::chrono::steady_clock> now;
    now   = std::chrono::steady_clock::now();
    std::chrono::duration<double> difference = now - start_;
	return difference.count();
}
// --------------------------------------------------------------------------
# elif _MSC_VER
{	return microsoft_timer(); }
// --------------------------------------------------------------------------
# elif CPPAD_HAS_GETTIMEOFDAY
{	CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL;
	static bool           first_ = true;
	static struct timeval tv_;
	struct timeval        tv;
	if( first_ )
	{	gettimeofday(&tv_, CPPAD_NULL);
		first_ = false;
		return 0.;
	}
	gettimeofday(&tv, CPPAD_NULL);
	assert( tv.tv_sec >= tv_.tv_sec );

	double sec  = double(tv.tv_sec -  tv_.tv_sec);
	double usec = double(tv.tv_usec) - double(tv_.tv_usec);
	double diff = sec + 1e-6*usec;

	return diff;
}
// --------------------------------------------------------------------------
# else // Not CPPAD_USE_CPLUSPLUS_2011 or CPPAD_HAS_GETTIMEOFDAY
{	CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL;
	static bool    first_ = true;
	static double  tic_;
	double  tic;
	if( first_ )
	{	tic_ = double(std::clock());
		first_ = false;
		return 0.;
	}
	tic = double( std::clock() );

	double diff = (tic - tic_) / double(CLOCKS_PER_SEC);

	return diff;
}
# endif
// --------------------------------------------------------------------------
} // END_CPPAD_NAMESPACE
# endif
