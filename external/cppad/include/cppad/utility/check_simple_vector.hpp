# ifndef CPPAD_UTILITY_CHECK_SIMPLE_VECTOR_HPP
# define CPPAD_UTILITY_CHECK_SIMPLE_VECTOR_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin CheckSimpleVector$$
$spell
	alloc
	const
	cppad.hpp
	CppAD
$$

$section Check Simple Vector Concept$$
$mindex CheckSimpleVector$$


$head Syntax$$
$codei%# include <cppad/utility/check_simple_vector.hpp>
%$$
$codei%CheckSimpleVector<%Scalar%, %Vector%>()%$$
$pre
$$
$codei%CheckSimpleVector<%Scalar%, %Vector%>(%x%, %y%)%$$


$head Purpose$$
Preforms compile and run time checks that the type specified
by $icode Vector$$ satisfies all the requirements for
a $cref SimpleVector$$ class with
$cref/elements of type/SimpleVector/Elements of Specified Type/$$
$icode Scalar$$.
If a requirement is not satisfied,
a an error message makes it clear what condition is not satisfied.

$head x, y$$
If the arguments $icode x$$ and $icode y$$ are present,
they have prototype
$codei%
	const %Scalar%& %x%
	const %Scalar%& %y%
%$$
In addition, the check
$codei%
	%x% == %x%
%$$
will return the boolean value $code true$$, and
$codei%
	%x% == %y%
%$$
will return $code false$$.

$head Restrictions$$
If the arguments $icode x$$ and $icode y$$ are not present,
the following extra assumption is made by $code CheckSimpleVector$$:
If $icode x$$ is a $icode Scalar$$ object
$codei%
	%x% = 0
	%y% = 1
%$$
assigns values to the objects $icode x$$ and $icode y$$.
In addition,
$icode%x% == %x%$$ would return the boolean value $code true$$ and
$icode%x% == %y%$$ would return $code false$$.

$head Include$$
The file $code cppad/check_simple_vector.hpp$$ is included by $code cppad/cppad.hpp$$
but it can also be included separately with out the rest
if the CppAD include files.

$head Parallel Mode$$
The routine $cref/thread_alloc::parallel_setup/ta_parallel_setup/$$
must be called before it
can be used in $cref/parallel/ta_in_parallel/$$ mode.

$head Example$$
$children%
	example/utility/check_simple_vector.cpp
%$$
The file $cref check_simple_vector.cpp$$
contains an example and test of this function where $icode S$$
is the same as $icode T$$.
It returns true, if it succeeds an false otherwise.
The comments in this example suggest a way to change the example
so $icode S$$ is not the same as $icode T$$.

$end
---------------------------------------------------------------------------
*/

# include <cstddef>
# include <cppad/core/cppad_assert.hpp>
# include <cppad/core/define.hpp>
# include <cppad/utility/thread_alloc.hpp>

namespace CppAD {

# ifdef NDEBUG
	template <class Scalar, class Vector>
	inline void CheckSimpleVector(const Scalar& x, const Scalar& y)
	{ }
	template <class Scalar, class Vector>
	inline void CheckSimpleVector(void)
	{ }
# else
	template <class S, class T>
	struct ok_if_S_same_as_T { };

	template <class T>
	struct ok_if_S_same_as_T<T,T> { T value; };

	template <class Scalar, class Vector>
	void CheckSimpleVector(const Scalar& x, const Scalar& y)
	{	CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL
		static size_t count;
		if( count > 0  )
			return;
		count++;

		// value_type must be type of elements of Vector
		typedef typename Vector::value_type value_type;

		// check that elements of Vector have type Scalar
		struct ok_if_S_same_as_T<Scalar, value_type> x_copy;
		x_copy.value = x;

		// check default constructor
		Vector d;

		// size member function
		CPPAD_ASSERT_KNOWN(
			d.size() == 0,
			"default construtor result does not have size zero"
		);

		// resize to same size as other vectors in test
		d.resize(1);

		// check sizing constructor
		Vector s(1);

		// check element assignment
		s[0] = y;
		CPPAD_ASSERT_KNOWN(
			s[0] == y,
			"element assignment failed"
		);

		// check copy constructor
		s[0] = x_copy.value;
		const Vector c(s);
		s[0] = y;
		CPPAD_ASSERT_KNOWN(
			c[0] == x,
			"copy constructor is shallow"
		);

		// vector assignment operator
		d[0] = x;
		s    = d;
		s[0] = y;
		CPPAD_ASSERT_KNOWN(
			d[0] == x,
			"assignment operator is shallow"
		);

		// element access, right side const
		// element assignment, left side not const
		d[0] = c[0];
		CPPAD_ASSERT_KNOWN(
			d[0] == x,
			"element assignment from const failed"
		);
	}
	template <class Scalar, class Vector>
	void CheckSimpleVector(void)
	{	Scalar x;
		Scalar y;

		// use assignment and not constructor
		x = 0;
		y = 1;

		CheckSimpleVector<Scalar, Vector>(x, y);
	}

# endif

} // end namespace CppAD

# endif
