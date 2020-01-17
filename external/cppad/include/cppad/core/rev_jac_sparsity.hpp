# ifndef CPPAD_CORE_REV_JAC_SPARSITY_HPP
# define CPPAD_CORE_REV_JAC_SPARSITY_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin rev_jac_sparsity$$
$spell
	Jacobian
	jac
	bool
	const
	rc
	cpp
$$

$section Reverse Mode Jacobian Sparsity Patterns$$

$head Syntax$$
$icode%f%.rev_jac_sparsity(
	%pattern_in%, %transpose%, %dependency%, %internal_bool%, %pattern_out%
)%$$

$head Purpose$$
We use $latex F : \B{R}^n \rightarrow \B{R}^m$$ to denote the
$cref/AD function/glossary/AD Function/$$ corresponding to
the operation sequence stored in $icode f$$.
Fix $latex R \in \B{R}^{\ell \times m}$$ and define the function
$latex \[
	J(x) = R * F^{(1)} ( x )
\] $$
Given the $cref/sparsity pattern/glossary/Sparsity Pattern/$$ for $latex R$$,
$code rev_jac_sparsity$$ computes a sparsity pattern for $latex J(x)$$.

$head x$$
Note that the sparsity pattern $latex J(x)$$ corresponds to the
operation sequence stored in $icode f$$ and does not depend on
the argument $icode x$$.
(The operation sequence may contain
$cref CondExp$$ and  $cref VecAD$$ operations.)

$head SizeVector$$
The type $icode SizeVector$$ is a $cref SimpleVector$$ class with
$cref/elements of type/SimpleVector/Elements of Specified Type/$$
$code size_t$$.

$head f$$
The object $icode f$$ has prototype
$codei%
	ADFun<%Base%> %f%
%$$

$head pattern_in$$
The argument $icode pattern_in$$ has prototype
$codei%
	const sparse_rc<%SizeVector%>& %pattern_in%
%$$
see $cref sparse_rc$$.
If $icode transpose$$ it is false (true),
$icode pattern_in$$ is a sparsity pattern for $latex R$$ ($latex R^\R{T}$$).

$head transpose$$
This argument has prototype
$codei%
	bool %transpose%
%$$
See $cref/pattern_in/rev_jac_sparsity/pattern_in/$$ above and
$cref/pattern_out/rev_jac_sparsity/pattern_out/$$ below.

$head dependency$$
This argument has prototype
$codei%
	bool %dependency%
%$$
see $cref/pattern_out/rev_jac_sparsity/pattern_out/$$ below.

$head internal_bool$$
If this is true, calculations are done with sets represented by a vector
of boolean values. Otherwise, a vector of sets of integers is used.

$head pattern_out$$
This argument has prototype
$codei%
	sparse_rc<%SizeVector%>& %pattern_out%
%$$
This input value of $icode pattern_out$$ does not matter.
If $icode transpose$$ it is false (true),
upon return $icode pattern_out$$ is a sparsity pattern for
$latex J(x)$$ ($latex J(x)^\R{T}$$).
If $icode dependency$$ is true, $icode pattern_out$$ is a
$cref/dependency pattern/dependency.cpp/Dependency Pattern/$$
instead of sparsity pattern.

$head Sparsity for Entire Jacobian$$
Suppose that
$latex R$$ is the $latex m \times m$$ identity matrix.
In this case, $icode pattern_out$$ is a sparsity pattern for
$latex F^{(1)} ( x )$$  ( $latex F^{(1)} (x)^\R{T}$$ )
if $icode transpose$$ is false (true).

$head Example$$
$children%
	example/sparse/rev_jac_sparsity.cpp
%$$
The file
$cref rev_jac_sparsity.cpp$$
contains an example and test of this operation.
It returns true if it succeeds and false otherwise.

$end
-----------------------------------------------------------------------------
*/
# include <cppad/core/ad_fun.hpp>
# include <cppad/local/sparse_internal.hpp>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE

/*!
Reverse Jacobian sparsity patterns.

\tparam Base
is the base type for this recording.

\tparam SizeVector
is the simple vector with elements of type size_t that is used for
row, column index sparsity patterns.

\param pattern_in
is the sparsity pattern for for R or R^T depending on transpose.

\param transpose
Is the input and returned sparsity pattern transposed.

\param dependency
Are the derivatives with respect to left and right of the expression below
considered to be non-zero:
\code
	CondExpRel(left, right, if_true, if_false)
\endcode
This is used by the optimizer to obtain the correct dependency relations.

\param internal_bool
If this is true, calculations are done with sets represented by a vector
of boolean values. Otherwise, a vector of standard sets is used.

\param pattern_out
The value of transpose is false (true),
the return value is a sparsity pattern for J(x) ( J(x)^T ) where
\f[
	J(x) = R * F^{(1)} (x)
\f]
Here F is the function corresponding to the operation sequence
and x is any argument value.
*/
template <class Base>
template <class SizeVector>
void ADFun<Base>::rev_jac_sparsity(
	const sparse_rc<SizeVector>& pattern_in       ,
	bool                         transpose        ,
	bool                         dependency       ,
	bool                         internal_bool    ,
	sparse_rc<SizeVector>&       pattern_out      )
{	// number or rows, columns, and non-zeros in pattern_in
	size_t nr_in  = pattern_in.nr();
	size_t nc_in  = pattern_in.nc();
	//
	size_t ell = nr_in;
	size_t m   = nc_in;
	if( transpose )
		std::swap(ell, m);
	//
	CPPAD_ASSERT_KNOWN(
		m == Range() ,
		"rev_jac_sparsity: number columns in R "
		"is not equal number of dependent variables."
	);
	// number of independent variables
	size_t n = Domain();
	//
	bool zero_empty  = true;
	bool input_empty = true;
	if( internal_bool )
	{	// allocate memory for bool sparsity calculation
		// (sparsity pattern is emtpy after a resize)
		local::sparse_pack internal_jac;
		internal_jac.resize(num_var_tape_, ell);
		//
		// set sparsity patttern for dependent variables
		local::set_internal_sparsity(
			zero_empty            ,
			input_empty           ,
			! transpose           ,
			dep_taddr_            ,
			internal_jac          ,
			pattern_in
		);

		// compute sparsity for other variables
		local::RevJacSweep(
			dependency,
			n,
			num_var_tape_,
			&play_,
			internal_jac
		);
		// get sparstiy pattern for independent variables
		local::get_internal_sparsity(
			! transpose, ind_taddr_, internal_jac, pattern_out
		);
	}
	else
	{	// allocate memory for bool sparsity calculation
		// (sparsity pattern is emtpy after a resize)
		local::sparse_list internal_jac;
		internal_jac.resize(num_var_tape_, ell);
		//
		// set sparsity patttern for dependent variables
		local::set_internal_sparsity(
			zero_empty            ,
			input_empty           ,
			! transpose           ,
			dep_taddr_            ,
			internal_jac          ,
			pattern_in
		);

		// compute sparsity for other variables
		local::RevJacSweep(
			dependency,
			n,
			num_var_tape_,
			&play_,
			internal_jac
		);
		// get sparstiy pattern for independent variables
		local::get_internal_sparsity(
			! transpose, ind_taddr_, internal_jac, pattern_out
		);
	}
	return;
}
} // END_CPPAD_NAMESPACE
# endif
