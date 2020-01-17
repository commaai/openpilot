# ifndef CPPAD_SPEED_DET_BY_LU_HPP
# define CPPAD_SPEED_DET_BY_LU_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin det_by_lu$$
$spell
	CppAD
	cppad
	lu
	hpp
	typedef
	const
	hpp
	Det
	CPPAD_TESTVECTOR
	namespace
$$

$section Determinant Using Expansion by Lu Factorization$$
$mindex det_by_lu factor$$


$head Syntax$$
$codei%# include <cppad/speed/det_by_lu.hpp>
%$$
$codei%det_by_lu<%Scalar%> %det%(%n%)
%$$
$icode%d% = %det%(%a%)
%$$

$head Inclusion$$
The template class $code det_by_lu$$ is defined in the $code CppAD$$
namespace by including
the file $code cppad/speed/det_by_lu.hpp$$
(relative to the CppAD distribution directory).

$head Constructor$$
The syntax
$codei%
	det_by_lu<%Scalar%> %det%(%n%)
%$$
constructs the object $icode det$$ which can be used for
evaluating the determinant of $icode n$$ by $icode n$$ matrices
using LU factorization.

$head Scalar$$
The type $icode Scalar$$ can be any
$cref NumericType$$

$head n$$
The argument $icode n$$ has prototype
$codei%
	size_t %n%
%$$

$head det$$
The syntax
$codei%
	%d% = %det%(%a%)
%$$
returns the determinant of the matrix $latex A$$ using LU factorization.

$subhead a$$
The argument $icode a$$ has prototype
$codei%
	const %Vector% &%a%
%$$
It must be a $icode Vector$$ with length $latex n * n$$ and with
It must be a $icode Vector$$ with length $latex n * n$$ and with
elements of type $icode Scalar$$.
The elements of the $latex n \times n$$ matrix $latex A$$ are defined,
for $latex i = 0 , \ldots , n-1$$ and $latex j = 0 , \ldots , n-1$$, by
$latex \[
	A_{i,j} = a[ i * m + j]
\] $$

$subhead d$$
The return value $icode d$$ has prototype
$codei%
	%Scalar% %d%
%$$

$head Vector$$
If $icode y$$ is a $icode Vector$$ object,
it must support the syntax
$codei%
	%y%[%i%]
%$$
where $icode i$$ has type $code size_t$$ with value less than $latex n * n$$.
This must return a $icode Scalar$$ value corresponding to the $th i$$
element of the vector $icode y$$.
This is the only requirement of the type $icode Vector$$.

$children%
	speed/example/det_by_lu.cpp%
	omh/det_by_lu_hpp.omh
%$$


$head Example$$
The file
$cref det_by_lu.cpp$$
contains an example and test of $code det_by_lu.hpp$$.
It returns true if it succeeds and false otherwise.

$head Source Code$$
The file
$cref det_by_lu.hpp$$
contains the source for this template function.


$end
---------------------------------------------------------------------------
*/
// BEGIN C++
# include <cppad/utility/vector.hpp>
# include <cppad/utility/lu_solve.hpp>

// BEGIN CppAD namespace
namespace CppAD {

template <class Scalar>
class det_by_lu {
private:
	const size_t m_;
	const size_t n_;
	CppAD::vector<Scalar> A_;
	CppAD::vector<Scalar> B_;
	CppAD::vector<Scalar> X_;
public:
	det_by_lu(size_t n) : m_(0), n_(n), A_(n * n)
	{	}

	template <class Vector>
	inline Scalar operator()(const Vector &x)
	{

		Scalar       logdet;
		Scalar       det;
		int          signdet;
		size_t       i;

		// copy matrix so it is not overwritten
		for(i = 0; i < n_ * n_; i++)
			A_[i] = x[i];

		// comput log determinant
		signdet = CppAD::LuSolve(
			n_, m_, A_, B_, X_, logdet);

/*
		// Do not do this for speed test because it makes floating
		// point operation sequence very simple.
		if( signdet == 0 )
			det = 0;
		else	det =  Scalar( signdet ) * exp( logdet );
*/

		// convert to determinant
		det     = Scalar( signdet ) * exp( logdet );

# ifdef FADBAD
		// Fadbad requires tempories to be set to constants
		for(i = 0; i < n_ * n_; i++)
			A_[i] = 0;
# endif

		return det;
	}
};
} // END CppAD namespace
// END C++
# endif
