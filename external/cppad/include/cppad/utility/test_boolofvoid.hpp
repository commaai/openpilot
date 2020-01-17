# ifndef CPPAD_UTILITY_TEST_BOOLOFVOID_HPP
# define CPPAD_UTILITY_TEST_BOOLOFVOID_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin test_boolofvoid$$
$spell
	boolofvoid
	const
	std
	bool
	ipopt
	cpp
$$

$section Object that Runs a Group of Tests$$

$head Syntax$$
$codei%test_boolofvoid %Run%(%group%, %width%)
%$$
$icode%Run%(%test%, %name%)
%$$
$icode%ok% = %Run%.summary(%memory_ok%)%$$

$head Purpose$$
The object $icode Run$$ is used to run a group of tests functions
and report the results on standard output.

$head group$$
The argument has prototype
$codei%
	const std::string& %group%
%$$
It is the name for this group of tests.

$head width$$
The argument has prototype
$codei%
	size_t %width%
%$$
It is the number of columns used to display the name of each test.
It must be greater than the maximum number of characters in a test name.

$head test$$
The argument has prototype
$codei%
	bool %test%(void)
%$$
It is a function that returns true (when the test passes) and false
otherwise.

$head name$$
The argument has prototype
$codei%
	const std::string& %name%
%$$
It is the name for the corresponding $icode test$$.

$head memory_ok$$
The argument has prototype
$codei%
	bool %memory_ok%
%$$
It is false if a memory leak is detected (and true otherwise).

$head ok$$
This is true if all of the tests pass (including the memory leak test),
otherwise it is false.

$head Example$$
See any of the main programs in the example directory; e.g.,
$code example/ipopt_solve.cpp$$.

$end
*/

# include <string>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE

/// One class object is used to run a group of tests
class test_boolofvoid {
private:
	/// name for the group of test this object will run
	const std::string group_;
	/// number of characters used to display the name for each indiviual test
	/// (must be larger than the number of characters in name for each test)
	const size_t      width_;
	/// number of tests that have passed
	size_t            n_ok_;
	/// number of tests that have failed
	size_t            n_error_;

public:
	/// ctor
	test_boolofvoid(const std::string& group, size_t width) :
	group_(group) ,
	width_(width) ,
	n_ok_(0)      ,
	n_error_(0)
	{	std::cout << "Begin test group " << group_ << std::endl; }
	/// destructor
	~test_boolofvoid(void)
	{	std::cout << "End test group " << group_ << std::endl; }
	/// run one test
	bool operator()(bool test(void), const std::string& name)
	{	CPPAD_ASSERT_KNOWN(
			name.size() < width_ ,
			"test_boolofvoid: name does not have less characters than width"
		);
		std::cout.width( width_ );
		std::cout.setf( std::ios_base::left );
		std::cout << name;
		//
		bool ok = test();
		if( ok )
		{	std::cout << "OK" << std::endl;
			n_ok_++;
		}
		else
		{	std::cout << "Error" << std::endl;
			n_error_++;
		}
		return ok;
	}
	/// nuber of tests that passed
	size_t n_ok(void) const
	{	return n_ok_; }
	/// nuber of tests that failed
	size_t n_error(void) const
	{	return n_error_; }
	/// summary
	bool summary(bool memory_ok )
	{
		std::cout.width( width_ );
		std::cout.setf( std::ios_base::left );
		std::cout << "memory_leak";
		//
		if( memory_ok  )
		{	std::cout << "OK" << std::endl;
			n_ok_++;
		}
		else
		{	std::cout << "Error" << std::endl;
			n_error_++;
		}
		if( n_error_ == 0 )
			std::cout << "All " << n_ok_ << " tests passed." << std::endl;
		else
			std::cout << n_error_ << " tests failed." << std::endl;
		//
		return n_error_ == 0;
	}
};

} // END_CPPAD_NAMESPACE

# endif
