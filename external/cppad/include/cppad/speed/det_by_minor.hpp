# ifndef CPPAD_SPEED_DET_BY_MINOR_HPP
# define CPPAD_SPEED_DET_BY_MINOR_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin det_by_minor$$
$spell
	CppAD
	cppad
	typedef
	const
	hpp
	Det
	namespace
$$

$section Determinant Using Expansion by Minors$$



$head Syntax$$
$codei%# include <cppad/speed/det_by_minor.hpp>
%$$
$codei%det_by_minor<%Scalar%> %det%(%n%)
%$$
$icode%d% = %det%(%a%)
%$$

$head Inclusion$$
The template class $code det_by_minor$$ is defined in the $code CppAD$$
namespace by including
the file $code cppad/speed/det_by_minor.hpp$$
(relative to the CppAD distribution directory).

$head Constructor$$
The syntax
$codei%
	det_by_minor<%Scalar%> %det%(%n%)
%$$
constructs the object $icode det$$ which can be used for
evaluating the determinant of $icode n$$ by $icode n$$ matrices
using expansion by minors.

$head Scalar$$
The type $icode Scalar$$ must satisfy the same conditions
as in the function $cref/det_of_minor/det_of_minor/Scalar/$$.

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
returns the determinant of the matrix $icode A$$ using expansion by minors.

$subhead a$$
The argument $icode a$$ has prototype
$codei%
	const %Vector% &%a%
%$$
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
It is equal to the determinant of $latex A$$.

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
	speed/example/det_by_minor.cpp%
	omh/det_by_minor_hpp.omh
%$$


$head Example$$
The file
$cref det_by_minor.cpp$$
contains an example and test of $code det_by_minor.hpp$$.
It returns true if it succeeds and false otherwise.

$head Source Code$$
The file
$cref det_by_minor.hpp$$
contains the source for this template function.


$end
---------------------------------------------------------------------------
*/
// BEGIN C++
# include <cppad/speed/det_of_minor.hpp>
# include <vector>

// BEGIN CppAD namespace
namespace CppAD {

template <class Scalar>
class det_by_minor {
private:
	size_t              m_;

	// made mutable because modified and then restored
	mutable std::vector<size_t> r_;
	mutable std::vector<size_t> c_;

	// make mutable because its value does not matter
	mutable std::vector<Scalar> a_;
public:
	det_by_minor(size_t m) : m_(m) , r_(m + 1) , c_(m + 1), a_(m * m)
	{
		size_t i;

		// values for r and c that correspond to entire matrix
		for(i = 0; i < m; i++)
		{	r_[i] = i+1;
			c_[i] = i+1;
		}
		r_[m] = 0;
		c_[m] = 0;
	}

	template <class Vector>
	inline Scalar operator()(const Vector &x) const
	{	size_t i = m_ * m_;
		while(i--)
			a_[i] = x[i];
		return det_of_minor(a_, m_, m_, r_, c_);
	}

};

} // END CppAD namespace
// END C++
# endif
