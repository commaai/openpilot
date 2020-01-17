# ifndef CPPAD_UTILITY_CHECK_NUMERIC_TYPE_HPP
# define CPPAD_UTILITY_CHECK_NUMERIC_TYPE_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin CheckNumericType$$
$spell
	alloc
	cppad.hpp
	CppAD
$$

$section Check NumericType Class Concept$$
$mindex numeric CheckNumericType$$


$head Syntax$$
$codei%# include <cppad/utility/check_numeric_type.hpp>
%$$
$codei%CheckNumericType<%NumericType%>()%$$


$head Purpose$$
The syntax
$codei%
	CheckNumericType<%NumericType%>()
%$$
preforms compile and run time checks that the type specified
by $icode NumericType$$ satisfies all the requirements for
a $cref NumericType$$ class.
If a requirement is not satisfied,
a an error message makes it clear what condition is not satisfied.

$head Include$$
The file $code cppad/check_numeric_type.hpp$$ is included by $code cppad/cppad.hpp$$
but it can also be included separately with out the rest
if the CppAD include files.

$head Parallel Mode$$
The routine $cref/thread_alloc::parallel_setup/ta_parallel_setup/$$
must be called before it
can be used in $cref/parallel/ta_in_parallel/$$ mode.

$head Example$$
$children%
	example/utility/check_numeric_type.cpp
%$$
The file $cref check_numeric_type.cpp$$
contains an example and test of this function.
It returns true, if it succeeds an false otherwise.
The comments in this example suggest a way to change the example
so an error message occurs.

$end
---------------------------------------------------------------------------
*/

# include <cstddef>
# include <cppad/utility/thread_alloc.hpp>

namespace CppAD {

# ifdef NDEBUG
	template <class NumericType>
	void CheckNumericType(void)
	{ }
# else
	template <class NumericType>
	NumericType CheckNumericType(void)
	{	// Section 3.6.2 of ISO/IEC 14882:1998(E) states: "The storage for
		// objects with static storage duration (3.7.1) shall be zero-
		// initialized (8.5) before any other initialization takes place."
		static size_t count[CPPAD_MAX_NUM_THREADS];
		size_t thread = thread_alloc::thread_num();
		if( count[thread] > 0  )
			return NumericType(0);
		count[thread]++;
		/*
		contructors
		*/
		NumericType check_NumericType_default_constructor;
		NumericType check_NumericType_constructor_from_int(1);

		const NumericType x(1);

		NumericType check_NumericType_copy_constructor(x);

		// assignment
		NumericType check_NumericType_assignment;
		check_NumericType_assignment = x;

		/*
		unary operators
		*/
		const NumericType check_NumericType_unary_plus(1);
		NumericType check_NumericType_unary_plus_result =
			+ check_NumericType_unary_plus;

		const NumericType check_NumericType_unary_minus(1);
		NumericType check_NumericType_unary_minus_result =
			- check_NumericType_unary_minus;

		/*
		binary operators
		*/
		const NumericType check_NumericType_binary_addition(1);
		NumericType check_NumericType_binary_addition_result =
			check_NumericType_binary_addition + x;

		const NumericType check_NumericType_binary_subtraction(1);
		NumericType check_NumericType_binary_subtraction_result =
			check_NumericType_binary_subtraction - x;

		const NumericType check_NumericType_binary_multiplication(1);
		NumericType check_NumericType_binary_multiplication_result =
			check_NumericType_binary_multiplication * x;

		const NumericType check_NumericType_binary_division(1);
		NumericType check_NumericType_binary_division_result =
			check_NumericType_binary_division / x;

		/*
		compound assignment operators
		*/
		NumericType
		check_NumericType_computed_assignment_addition(1);
		check_NumericType_computed_assignment_addition += x;

		NumericType
		check_NumericType_computed_assignment_subtraction(1);
		check_NumericType_computed_assignment_subtraction -= x;

		NumericType
		check_NumericType_computed_assignment_multiplication(1);
		check_NumericType_computed_assignment_multiplication *= x;

		NumericType
		check_NumericType_computed_assignment_division(1);
		check_NumericType_computed_assignment_division /= x;

		/*
		use all values so as to avoid warnings
		*/
		check_NumericType_default_constructor = x;
		return
			+ check_NumericType_default_constructor
			+ check_NumericType_constructor_from_int
			+ check_NumericType_copy_constructor
			+ check_NumericType_assignment
			+ check_NumericType_unary_plus_result
			+ check_NumericType_unary_minus_result
			+ check_NumericType_binary_addition_result
			+ check_NumericType_binary_subtraction_result
			+ check_NumericType_binary_multiplication_result
			+ check_NumericType_binary_division_result
			+ check_NumericType_computed_assignment_addition
			+ check_NumericType_computed_assignment_subtraction
			+ check_NumericType_computed_assignment_multiplication
			+ check_NumericType_computed_assignment_division
		;
	}
# endif

} // end namespace CppAD

# endif
