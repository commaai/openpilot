# ifndef CPPAD_UTILITY_SPEED_TEST_HPP
# define CPPAD_UTILITY_SPEED_TEST_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin speed_test$$
$spell
	gettimeofday
	vec
	cppad.hpp
	Microsoft
	namespace
	std
	const
	cout
	ctime
	ifdef
	const
	endif
	cpp
$$


$section Run One Speed Test and Return Results$$
$mindex speed_test$$

$head Syntax$$
$codei%# include <cppad/utility/speed_test.hpp>
%$$
$icode%rate_vec% = speed_test(%test%, %size_vec%, %time_min%)%$$

$head Purpose$$
The $code speed_test$$ function executes a speed test
for various sized problems
and reports the rate of execution.

$head Motivation$$
It is important to separate small calculation units
and test them individually.
This way individual changes can be tested in the context of the
routine that they are in.
On many machines, accurate timing of a very short execution
sequences is not possible.
In addition,
there may be set up and tear down time for a test that
we do not really want included in the timing.
For this reason $code speed_test$$
automatically determines how many times to
repeat the section of the test that we wish to time.


$head Include$$
The file $code cppad/speed_test.hpp$$ defines the
$code speed_test$$ function.
This file is included by $code cppad/cppad.hpp$$
and it can also be included separately with out the rest of
the $code CppAD$$ routines.

$head Vector$$
We use $icode Vector$$ to denote a
$cref/simple vector class/SimpleVector/$$ with elements
of type $code size_t$$.

$head test$$
The $code speed_test$$ argument $icode test$$ is a function with the syntax
$codei%
	%test%(%size%, %repeat%)
%$$
and its return value is $code void$$.

$subhead size$$
The $icode test$$ argument $icode size$$ has prototype
$codei%
	size_t %size%
%$$
It specifies the size for this test.

$subhead repeat$$
The $icode test$$ argument $icode repeat$$ has prototype
$codei%
	size_t %repeat%
%$$
It specifies the number of times to repeat the test.

$head size_vec$$
The $code speed_test$$ argument $icode size_vec$$ has prototype
$codei%
	const %Vector%& %size_vec%
%$$
This vector determines the size for each of the tests problems.

$head time_min$$
The argument $icode time_min$$ has prototype
$codei%
	double %time_min%
%$$
It specifies the minimum amount of time in seconds
that the $icode test$$ routine should take.
The $icode repeat$$ argument to $icode test$$ is increased
until this amount of execution time is reached.

$head rate_vec$$
The return value $icode rate_vec$$ has prototype
$codei%
	%Vector%& %rate_vec%
%$$
We use $latex n$$ to denote its size which is the same as
the vector $icode size_vec$$.
For $latex i = 0 , \ldots , n-1$$,
$codei%
	%rate_vec%[%i%]
%$$
is the ratio of $icode repeat$$ divided by time in seconds
for the problem with size $icode%size_vec%[%i%]%$$.

$head Timing$$
If your system supports the unix $code gettimeofday$$ function,
it will be used to measure time.
Otherwise,
time is measured by the difference in
$codep
	(double) clock() / (double) CLOCKS_PER_SEC
$$
in the context of the standard $code <ctime>$$ definitions.

$children%
	speed/example/speed_test.cpp
%$$
$head Example$$
The routine $cref speed_test.cpp$$ is an example and test
of $code speed_test$$.

$end
-----------------------------------------------------------------------
*/

# include <cstddef>
# include <cmath>

# include <cppad/utility/check_simple_vector.hpp>
# include <cppad/utility/elapsed_seconds.hpp>


namespace CppAD { // BEGIN CppAD namespace

// implemented as an inline so that can include in multiple link modules
// with this same file
template <class Vector>
inline Vector speed_test(
	void test(size_t size, size_t repeat),
	const Vector& size_vec               ,
	double time_min                      )
{
	// check that size_vec is a simple vector with size_t elements
	CheckSimpleVector<size_t, Vector>();

	size_t   n = size_vec.size();
	Vector rate_vec(n);
	size_t i;
	for(i = 0; i < n; i++)
	{	size_t size   = size_vec[i];
		size_t repeat = 1;
		double s0     = elapsed_seconds();
		double s1     = elapsed_seconds();
		while( s1 - s0 < time_min )
		{	repeat = 2 * repeat;
			s0     = elapsed_seconds();
			test(size, repeat);
			s1     = elapsed_seconds();
		}
		double rate = .5 + double(repeat) / (s1 - s0);
		// first convert to float to avoid warning with g++ -Wconversion
		rate_vec[i] = static_cast<size_t>( static_cast<float>(rate) );
	}
	return rate_vec;
}

} // END CppAD namespace

/*
$begin SpeedTest$$
$spell
	cppad.hpp
	Microsoft
	namespace
	std
	const
	cout
	ctime
	ifdef
	const
	endif
	cpp
$$


$section Run One Speed Test and Print Results$$
$mindex SpeedTest$$

$head Syntax$$

$codei%# include <cppad/utility/speed_test.hpp>
%$$
$codei%SpeedTest(%Test%, %first%, %inc%, %last%)%$$


$head Purpose$$
The $code SpeedTest$$ function executes a speed test
for various sized problems
and reports the results on standard output; i.e. $code std::cout$$.
The size of each test problem is included in its report
(unless $icode first$$ is equal to $icode last$$).

$head Motivation$$
It is important to separate small calculation units
and test them individually.
This way individual changes can be tested in the context of the
routine that they are in.
On many machines, accurate timing of a very short execution
sequences is not possible.
In addition,
there may be set up time for a test that
we do not really want included in the timing.
For this reason $code SpeedTest$$
automatically determines how many times to
repeat the section of the test that we wish to time.


$head Include$$
The file $code speed_test.hpp$$ contains the
$code SpeedTest$$ function.
This file is included by $code cppad/cppad.hpp$$
but it can also be included separately with out the rest of
the $code CppAD$$ routines.

$head Test$$
The $code SpeedTest$$ argument $icode Test$$ is a function with the syntax
$codei%
	%name% = %Test%(%size%, %repeat%)
%$$

$subhead size$$
The $icode Test$$ argument $icode size$$ has prototype
$codei%
	size_t %size%
%$$
It specifies the size for this test.

$subhead repeat$$
The $icode Test$$ argument $icode repeat$$ has prototype
$codei%
	size_t %repeat%
%$$
It specifies the number of times to repeat the test.

$subhead name$$
The $icode Test$$ result $icode name$$ has prototype
$codei%
	std::string %name%
%$$
The results for this test are reported on $code std::cout$$
with $icode name$$ as an identifier for the test.
It is assumed that,
for the duration of this call to $code SpeedTest$$,
$icode Test$$ will always return
the same value for $icode name$$.
If $icode name$$ is the empty string,
no test name is reported by $code SpeedTest$$.

$head first$$
The $code SpeedTest$$ argument $icode first$$ has prototype
$codei%
	size_t %first%
%$$
It specifies the size of the first test problem reported by this call to
$code SpeedTest$$.

$head last$$
The $code SpeedTest$$ argument $icode last$$ has prototype
$codei%
	size_t %last%
%$$
It specifies the size of the last test problem reported by this call to
$code SpeedTest$$.

$head inc$$
The $code SpeedTest$$ argument $icode inc$$ has prototype
$codei%
	int %inc%
%$$
It specifies the increment between problem sizes; i.e.,
all values of $icode size$$ in calls to $icode Test$$ are given by
$codei%
	%size% = %first% + %j% * %inc%
%$$
where $icode j$$ is a positive integer.
The increment can be positive or negative but it cannot be zero.
The values $icode first$$, $icode last$$ and $icode inc$$ must
satisfy the relation
$latex \[
	inc * ( last - first ) \geq 0
\] $$

$head rate$$
The value displayed in the $code rate$$ column on $code std::cout$$
is defined as the value of $icode repeat$$ divided by the
corresponding elapsed execution time in seconds.
The elapsed execution time is measured by the difference in
$codep
	(double) clock() / (double) CLOCKS_PER_SEC
$$
in the context of the standard $code <ctime>$$ definitions.


$head Errors$$
If one of the restrictions above is violated,
the CppAD error handler is used to report the error.
You can redefine this action using the instructions in
$cref ErrorHandler$$

$head Example$$
$children%
	speed/example/speed_program.cpp
%$$
The program $cref speed_program.cpp$$ is an example usage
of $code SpeedTest$$.

$end
-----------------------------------------------------------------------
*/
// BEGIN C++


# include <string>
# include <iostream>
# include <iomanip>
# include <cppad/core/cppad_assert.hpp>

namespace CppAD { // BEGIN CppAD namespace

inline void SpeedTestNdigit(size_t value, size_t &ndigit, size_t &pow10)
{	pow10 = 10;
	ndigit       = 1;
	while( pow10 <= value )
	{	pow10  *= 10;
		ndigit += 1;
	}
}

// implemented as an inline so that can include in multiple link modules
// with this same file
inline void SpeedTest(
	std::string Test(size_t size, size_t repeat),
	size_t first,
	int    inc,
	size_t last
)
{

	using std::cout;
	using std::endl;

	size_t    size;
	size_t    repeat;
	size_t    rate;
	size_t    digit;
	size_t    ndigit;
	size_t    pow10;
	size_t    maxSize;
	size_t    maxSizeDigit;

	double    s0;
	double    s1;

	std::string name;

	CPPAD_ASSERT_KNOWN(
		inc != 0 && first != 0 && last != 0,
		"inc, first, or last is zero in call to SpeedTest"
	);
	CPPAD_ASSERT_KNOWN(
		(inc > 0 && first <= last) || (inc < 0 && first >= last),
		"SpeedTest: increment is positive and first > last or "
		"increment is negative and first < last"
	);

	// compute maxSize
	maxSize = size = first;
	while(  (inc > 0 && size <= last) || (inc < 0 && size >= last) )
	{
		if( size > maxSize )
			maxSize = size;

		// next size
		if( ((int) size) + inc > 0 )
			size += inc;
		else	size  = 0;
	}
	SpeedTestNdigit(maxSize, maxSizeDigit, pow10);

	size = first;
	while(  (inc > 0 && size <= last) || (inc < 0 && size >= last) )
	{
		repeat = 1;
		s0     = elapsed_seconds();
		s1     = elapsed_seconds();
		while( s1 - s0 < 1. )
		{	repeat = 2 * repeat;
			s0     = elapsed_seconds();
			name   = Test(size, repeat);
			s1     = elapsed_seconds();
		}
		double r = .5 + double(repeat) / (s1 - s0);
		// first convert to float to avoid warning with g++ -Wconversion
		rate     = static_cast<size_t>( static_cast<float>( r ) );

		if( size == first && name != "" )
			cout << name << endl;

		if( first != last )
		{
			// convert int(size_t) to avoid warning on _MSC_VER sys
			std::cout << "size = "  << int(size);

			SpeedTestNdigit(size, ndigit, pow10);
			while( ndigit < maxSizeDigit )
			{	cout << " ";
				ndigit++;
			}
			cout << " ";
		}

		cout << "rate = ";
		SpeedTestNdigit(rate, ndigit, pow10);
		while( ndigit > 0 )
		{
			pow10 /= 10;
			digit  = rate / pow10;

			// convert int(size_t) to avoid warning on _MSC_VER sys
			std::cout << int(digit);

			rate    = rate % pow10;
			ndigit -= 1;

			if( (ndigit > 0) && (ndigit % 3 == 0) )
				cout << ",";
		}
		cout << endl;

		// next size
		if( ((int) size) + inc > 0 )
			size += inc;
		else	size  = 0;
	}
	return;
}

} // END CppAD namespace

// END C++
# endif
