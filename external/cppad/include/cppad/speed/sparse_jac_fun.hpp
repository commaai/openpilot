# ifndef CPPAD_SPEED_SPARSE_JAC_FUN_HPP
# define CPPAD_SPEED_SPARSE_JAC_FUN_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin sparse_jac_fun$$
$spell
	Jacobian
	jac
	cppad
	hpp
	fp
	CppAD
	namespace
	const
	bool
	exp
	arg
$$

$section Evaluate a Function That Has a Sparse Jacobian$$
$mindex sparse_jac_fun$$


$head Syntax$$
$codei%# include <cppad/speed/sparse_jac_fun.hpp>
%$$
$codei%sparse_jac_fun(%m%, %n%, %x%, %row%, %col%, %p%, %fp%)%$$

$head Purpose$$
This routine evaluates
$latex f(x)$$ and $latex f^{(1)} (x)$$
where the Jacobian $latex f^{(1)} (x)$$ is sparse.
The function $latex f : \B{R}^n \rightarrow \B{R}^m$$ only depends on the
size and contents of the index vectors $icode row$$ and $icode col$$.
The non-zero entries in the Jacobian of this function have
one of the following forms:
$latex \[
	\D{ f[row[k]]}{x[col[k]]}
\] $$
for some $latex k $$ between zero and $latex K-1$$.
All the other terms of the Jacobian are zero.

$head Inclusion$$
The template function $code sparse_jac_fun$$
is defined in the $code CppAD$$ namespace by including
the file $code cppad/speed/sparse_jac_fun.hpp$$
(relative to the CppAD distribution directory).

$head Float$$
The type $icode Float$$ must be a $cref NumericType$$.
In addition, if $icode y$$ and $icode z$$ are $icode Float$$ objects,
$codei%
	%y% = exp(%z%)
%$$
must set the $icode y$$ equal the exponential of $icode z$$, i.e.,
the derivative of $icode y$$ with respect to $icode z$$ is equal to $icode y$$.

$head FloatVector$$
The type $icode FloatVector$$ is any
$cref SimpleVector$$, or it can be a raw pointer,
with elements of type $icode Float$$.

$head n$$
The argument $icode n$$ has prototype
$codei%
	size_t %n%
%$$
It specifies the dimension for the domain space for $latex f(x)$$.

$head m$$
The argument $icode m$$ has prototype
$codei%
	size_t %m%
%$$
It specifies the dimension for the range space for $latex f(x)$$.

$head x$$
The argument $icode x$$ has prototype
$codei%
	const %FloatVector%& %x%
%$$
It contains the argument value for which the function,
or its derivative, is being evaluated.
We use $latex n$$ to denote the size of the vector $icode x$$.

$head row$$
The argument $icode row$$ has prototype
$codei%
	 const CppAD::vector<size_t>& %row%
%$$
It specifies indices in the range of $latex f(x)$$ for non-zero components
of the Jacobian
(see $cref/purpose/sparse_hes_fun/Purpose/$$ above).
The value $latex K$$ is defined by $icode%K% = %row%.size()%$$.
All the elements of $icode row$$ must be between zero and $icode%m%-1%$$.

$head col$$
The argument $icode col$$ has prototype
$codei%
	 const CppAD::vector<size_t>& %col%
%$$
and its size must be $latex K$$; i.e., the same as $icode row$$.
It specifies the component of $latex x$$ for
the non-zero Jacobian terms.
All the elements of $icode col$$ must be between zero and $icode%n%-1%$$.

$head p$$
The argument $icode p$$ has prototype
$codei%
	size_t %p%
%$$
It is either zero or one and
specifies the order of the derivative of $latex f$$
that is being evaluated, i.e., $latex f^{(p)} (x)$$ is evaluated.

$head fp$$
The argument $icode fp$$ has prototype
$codei%
	%FloatVector%& %fp%
%$$
If $icode%p% = 0%$$, it size is $icode m$$
otherwise its size is $icode K$$.
The input value of the elements of $icode fp$$ does not matter.

$subhead Function$$
If $icode p$$ is zero, $icode fp$$ has size $latex m$$ and
$codei%(%fp%[0]%, ... , %fp%[%m%-1])%$$ is the value of $latex f(x)$$.

$subhead Jacobian$$
If $icode p$$ is one, $icode fp$$ has size $icode K$$ and
for $latex k = 0 , \ldots , K-1$$,
$latex \[
	\D{f[ \R{row}[i] ]}{x[ \R{col}[j] ]} = fp [k]
\] $$

$children%
	speed/example/sparse_jac_fun.cpp%
	omh/sparse_jac_fun.omh
%$$

$head Example$$
The file
$cref sparse_jac_fun.cpp$$
contains an example and test  of $code sparse_jac_fun.hpp$$.
It returns true if it succeeds and false otherwise.

$head Source Code$$
The file
$cref sparse_jac_fun.hpp$$
contains the source code for this template function.

$end
------------------------------------------------------------------------------
*/
// BEGIN C++
# include <cppad/core/cppad_assert.hpp>
# include <cppad/utility/check_numeric_type.hpp>
# include <cppad/utility/vector.hpp>

// following needed by gcc under fedora 17 so that exp(double) is defined
# include <cppad/base_require.hpp>

namespace CppAD {
	template <class Float, class FloatVector>
	void sparse_jac_fun(
		size_t                       m    ,
		size_t                       n    ,
		const FloatVector&           x    ,
		const CppAD::vector<size_t>& row  ,
		const CppAD::vector<size_t>& col  ,
		size_t                       p    ,
		FloatVector&                 fp   )
	{
		// check numeric type specifications
		CheckNumericType<Float>();
		// check value of p
		CPPAD_ASSERT_KNOWN(
			p == 0 || p == 1,
			"sparse_jac_fun: p != 0 and p != 1"
		);
		size_t K = row.size();
		CPPAD_ASSERT_KNOWN(
			K >= m,
			"sparse_jac_fun: row.size() < m"
		);
		size_t i, j, k;

		if( p == 0 )
			for(i = 0; i < m; i++)
				fp[i] = Float(0);

		Float t;
		for(k = 0; k < K; k++)
		{	i    = row[k];
			j    = col[k];
			t    = exp( x[j] * x[j] / 2.0 );
			switch(p)
			{
				case 0:
				fp[i] += t;
				break;

				case 1:
				fp[k] = t * x[j];
				break;
			}
		}
	}
}
// END C++
# endif
