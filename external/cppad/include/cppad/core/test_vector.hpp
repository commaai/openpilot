# ifndef CPPAD_CORE_TEST_VECTOR_HPP
# define CPPAD_CORE_TEST_VECTOR_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin test_vector$$
$spell
	autotools
	ifdef
	undef
	Microsofts
	CppADvector
	hpp
	std
	endif
	ublas
	Dir
	valarray
	stdvector
$$


$section Choosing The Vector Testing Template Class$$
$mindex CPPAD_TEST_VECTOR test$$

$head Deprecated 2012-07-03$$
The $code CPPAD_TEST_VECTOR$$ macro has been deprecated,
use $cref/CPPAD_TESTVECTOR/testvector/$$ instead.

$head Syntax$$
$codei%CPPAD_TEST_VECTOR<%Scalar%>
%$$

$head Introduction$$
Many of the CppAD $cref/examples/example/$$ and tests use
the $code CPPAD_TEST_VECTOR$$ template class to pass information.
The default definition for this template class is
$cref/CppAD::vector/CppAD_vector/$$.

$head MS Windows$$
The include path for boost is not defined in the Windows project files.
If we are using Microsofts compiler, the following code overrides the setting
of $code CPPAD_BOOSTVECTOR$$:
$srccode%cpp% */
// The next 7 lines are C++ source code.
# ifdef _MSC_VER
# if CPPAD_BOOSTVECTOR
# undef  CPPAD_BOOSTVECTOR
# define CPPAD_BOOSTVECTOR 0
# undef  CPPAD_CPPADVECTOR
# define CPPAD_CPPADVECTOR 1
# endif
# endif
/* %$$

$head CppAD::vector$$
By default $code CPPAD_CPPADVECTOR$$ is true
and $code CPPAD_TEST_VECTOR$$ is defined by the following source code
$srccode%cpp% */
// The next 3 line are C++ source code.
# if CPPAD_CPPADVECTOR
# define CPPAD_TEST_VECTOR CppAD::vector
# endif
/* %$$
If you specify $code --with-eigenvector$$ on the
$cref/configure/autotools/Configure/$$ command line,
$code CPPAD_EIGENVECTOR$$ is true.
This vector type cannot be supported by $code CPPAD_TEST_VECTOR$$
(use $cref/CPPAD_TESTVECTOR/testvector/$$ for this support)
so $code CppAD::vector$$ is used in this case
$srccode%cpp% */
// The next 3 line are C++ source code.
# if CPPAD_EIGENVECTOR
# define CPPAD_TEST_VECTOR CppAD::vector
# endif
/* %$$


$head std::vector$$
If you specify $code --with-stdvector$$ on the
$cref/configure/autotools/Configure/$$
command line during CppAD installation,
$code CPPAD_STDVECTOR$$ is true
and $code CPPAD_TEST_VECTOR$$ is defined by the following source code
$srccode%cpp% */
// The next 4 lines are C++ source code.
# if CPPAD_STDVECTOR
# include <vector>
# define CPPAD_TEST_VECTOR std::vector
# endif
/* %$$
In this case CppAD will use $code std::vector$$ for its examples and tests.
Use of $code CppAD::vector$$, $code std::vector$$,
and $code std::valarray$$ with CppAD is always tested to some degree.
Specifying $code --with-stdvector$$ will increase the amount of
$code std::vector$$ testing.

$head boost::numeric::ublas::vector$$
If you specify a value for $icode boost_dir$$ on the configure
command line during CppAD installation,
$code CPPAD_BOOSTVECTOR$$ is true
and $code CPPAD_TEST_VECTOR$$ is defined by the following source code
$srccode%cpp% */
// The next 4 lines are C++ source code.
# if CPPAD_BOOSTVECTOR
# include <boost/numeric/ublas/vector.hpp>
# define CPPAD_TEST_VECTOR boost::numeric::ublas::vector
# endif
/* %$$
In this case CppAD will use Ublas vectors for its examples and tests.
Use of $code CppAD::vector$$, $code std::vector$$,
and $code std::valarray$$ with CppAD is always tested to some degree.
Specifying $icode boost_dir$$ will increase the amount of
Ublas vector testing.

$head CppADvector Deprecated 2007-07-28$$
The preprocessor symbol $code CppADvector$$ is defined to
have the same value as $code CPPAD_TEST_VECTOR$$ but its use is deprecated:
$srccode%cpp% */
# define CppADvector CPPAD_TEST_VECTOR
/* %$$
$end
------------------------------------------------------------------------
*/

# endif
