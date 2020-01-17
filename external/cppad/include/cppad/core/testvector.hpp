// $Id$
# ifndef CPPAD_CORE_TESTVECTOR_HPP
# define CPPAD_CORE_TESTVECTOR_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin testvector$$
$spell
	CppAD
	cmake
	testvector
	cppad
	Eigen
	ifdef
	hpp
	std
	endif
	ublas
$$


$section Using The CppAD Test Vector Template Class$$
$mindex CPPAD_TESTVECTOR$$

$head Syntax$$
$codei%CPPAD_TESTVECTOR(%Scalar%)
%$$

$head Purpose$$
Many of the CppAD $cref/examples/example/$$ and tests use
the $code CPPAD_TESTVECTOR$$ template class to pass information to CppAD.
This is not a true template class because it's syntax uses
$codei%(%Scalar%)%$$ instead of $codei%<%Scalar%>%$$.
This enables us to use
$codei%
	Eigen::Matrix<%Scalar%, Eigen::Dynamic, 1>
%$$
as one of the possible cases for this 'template class'.

$head Choice$$
The user can choose, during the install procedure,
which template class to use in the examples and tests; see below.
This shows that any
$cref/simple vector/SimpleVector/$$ class can be used in place of
$codei%
	CPPAD_TESTVECTOR(%Type%)
%$$
When writing their own code,
users can choose a specific simple vector they prefer; for example,
$codei%
	CppAD::vector<%Type%>
%$$


$head CppAD::vector$$
If in the $cref/cmake command/cmake/CMake Command/$$
you specify $cref cppad_testvector$$ to be $code cppad$$,
$code CPPAD_CPPADVECTOR$$ will be true.
In this case,
$code CPPAD_TESTVECTOR$$ is defined by the following source code:
$srccode%cpp% */
# if CPPAD_CPPADVECTOR
# define CPPAD_TESTVECTOR(Scalar) CppAD::vector< Scalar >
# endif
/* %$$
In this case CppAD will use its own vector for
many of its examples and tests.

$head std::vector$$
If in the cmake command
you specify $icode cppad_testvector$$ to be $code std$$,
$code CPPAD_STDVECTOR$$ will be true.
In this case,
$code CPPAD_TESTVECTOR$$ is defined by the following source code:
$srccode%cpp% */
# if CPPAD_STDVECTOR
# include <vector>
# define CPPAD_TESTVECTOR(Scalar) std::vector< Scalar >
# endif
/* %$$
In this case CppAD will use standard vector for
many of its examples and tests.

$head boost::numeric::ublas::vector$$
If in the cmake command
you specify $icode cppad_testvector$$ to be $code boost$$,
$code CPPAD_BOOSTVECTOR$$ will be true.
In this case,
$code CPPAD_TESTVECTOR$$ is defined by the following source code:
$srccode%cpp% */
# if CPPAD_BOOSTVECTOR
# include <boost/numeric/ublas/vector.hpp>
# define CPPAD_TESTVECTOR(Scalar) boost::numeric::ublas::vector< Scalar >
# endif
/* %$$
In this case CppAD will use this boost vector for
many of its examples and tests.

$head Eigen Vectors$$
If in the cmake command
you specify $icode cppad_testvector$$ to be $code eigen$$,
$code CPPAD_EIGENVECTOR$$ will be true.
In this case,
$code CPPAD_TESTVECTOR$$ is defined by the following source code:
$srccode%cpp% */
# if CPPAD_EIGENVECTOR
# include <cppad/example/cppad_eigen.hpp>
# define CPPAD_TESTVECTOR(Scalar) Eigen::Matrix< Scalar , Eigen::Dynamic, 1>
# endif
/* %$$
In this case CppAD will use the Eigen vector
for many of its examples and tests.

$end
------------------------------------------------------------------------
*/

# endif
