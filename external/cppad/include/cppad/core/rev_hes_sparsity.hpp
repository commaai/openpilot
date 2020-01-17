# ifndef CPPAD_CORE_REV_HES_SPARSITY_HPP
# define CPPAD_CORE_REV_HES_SPARSITY_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin rev_hes_sparsity$$
$spell
	Jacobian
	Hessian
	jac
	hes
	bool
	const
	rc
	cpp
$$

$section Reverse Mode Hessian Sparsity Patterns$$

$head Syntax$$
$icode%f%.rev_hes_sparsity(
	%select_range%, %transpose%, %internal_bool%, %pattern_out%
)%$$

$head Purpose$$
We use $latex F : \B{R}^n \rightarrow \B{R}^m$$ to denote the
$cref/AD function/glossary/AD Function/$$ corresponding to
the operation sequence stored in $icode f$$.
Fix $latex R \in \B{R}^{n \times \ell}$$, $latex s \in \B{R}^m$$
and define the function
$latex \[
	H(x) = ( s^\R{T} F )^{(2)} ( x ) R
\] $$
Given a $cref/sparsity pattern/glossary/Sparsity Pattern/$$ for $latex R$$
and for the vector $latex s$$,
$code rev_hes_sparsity$$ computes a sparsity pattern for $latex H(x)$$.

$head x$$
Note that the sparsity pattern $latex H(x)$$ corresponds to the
operation sequence stored in $icode f$$ and does not depend on
the argument $icode x$$.

$head BoolVector$$
The type $icode BoolVector$$ is a $cref SimpleVector$$ class with
$cref/elements of type/SimpleVector/Elements of Specified Type/$$
$code bool$$.

$head SizeVector$$
The type $icode SizeVector$$ is a $cref SimpleVector$$ class with
$cref/elements of type/SimpleVector/Elements of Specified Type/$$
$code size_t$$.

$head f$$
The object $icode f$$ has prototype
$codei%
	ADFun<%Base%> %f%
%$$

$head R$$
The sparsity pattern for the matrix $latex R$$ is specified by
$cref/pattern_in/for_jac_sparsity/pattern_in/$$ in the previous call
$codei%
	%f%.for_jac_sparsity(
		%pattern_in%, %transpose%, %dependency%, %internal_bool%, %pattern_out%
)%$$

$head select_range$$
The argument $icode select_range$$ has prototype
$codei%
	const %BoolVector%& %select_range%
%$$
It has size $latex m$$ and specifies which components of the vector
$latex s$$ are non-zero; i.e., $icode%select_range%[%i%]%$$ is true
if and only if $latex s_i$$ is possibly non-zero.

$head transpose$$
This argument has prototype
$codei%
	bool %transpose%
%$$
See $cref/pattern_out/rev_hes_sparsity/pattern_out/$$ below.

$head internal_bool$$
If this is true, calculations are done with sets represented by a vector
of boolean values. Otherwise, a vector of sets of integers is used.
This must be the same as in the previous call to
$icode%f%.for_jac_sparsity%$$.

$head pattern_out$$
This argument has prototype
$codei%
	sparse_rc<%SizeVector%>& %pattern_out%
%$$
This input value of $icode pattern_out$$ does not matter.
If $icode transpose$$ it is false (true),
upon return $icode pattern_out$$ is a sparsity pattern for
$latex H(x)$$ ($latex H(x)^\R{T}$$).

$head Sparsity for Entire Hessian$$
Suppose that $latex R$$ is the $latex n \times n$$ identity matrix.
In this case, $icode pattern_out$$ is a sparsity pattern for
$latex (s^\R{T} F) F^{(2)} ( x )$$.

$head Example$$
$children%
	example/sparse/rev_hes_sparsity.cpp
%$$
The file
$cref rev_hes_sparsity.cpp$$
contains an example and test of this operation.
It returns true if it succeeds and false otherwise.

$end
-----------------------------------------------------------------------------
*/
# include <cppad/core/ad_fun.hpp>
# include <cppad/local/sparse_internal.hpp>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE

/*!
Reverse Hessian sparsity patterns.

\tparam Base
is the base type for this recording.

\tparam BoolVector
is the simple vector with elements of type bool that is used for
sparsity for the vector s.

\tparam SizeVector
is the simple vector with elements of type size_t that is used for
row, column index sparsity patterns.

\param select_range
is a sparsity pattern for for s.

\param transpose
Is the returned sparsity pattern transposed.

\param internal_bool
If this is true, calculations are done with sets represented by a vector
of boolean values. Otherwise, a vector of standard sets is used.

\param pattern_out
The value of transpose is false (true),
the return value is a sparsity pattern for H(x) ( H(x)^T ) where
\f[
	H(x) = R * F^{(1)} (x)
\f]
Here F is the function corresponding to the operation sequence
and x is any argument value.
*/
template <class Base>
template <class BoolVector, class SizeVector>
void ADFun<Base>::rev_hes_sparsity(
	const BoolVector&            select_range     ,
	bool                         transpose        ,
	bool                         internal_bool    ,
	sparse_rc<SizeVector>&       pattern_out      )
{	size_t n  = Domain();
	size_t m  = Range();
	//
	CPPAD_ASSERT_KNOWN(
		size_t( select_range.size() ) == m,
		"rev_hes_sparsity: size of select_range is not equal to "
		"number of dependent variables"
	);
	//
	// vector that holds reverse Jacobian sparsity flag
	local::pod_vector<bool> rev_jac_pattern;
	rev_jac_pattern.extend(num_var_tape_);
	for(size_t i = 0; i < num_var_tape_; i++)
		rev_jac_pattern[i] = false;
	//
	// initialize rev_jac_pattern for dependent variables
	for(size_t i = 0; i < m; i++)
		rev_jac_pattern[ dep_taddr_[i] ] = select_range[i];
	//
	//
	if( internal_bool )
	{	CPPAD_ASSERT_KNOWN(
			for_jac_sparse_pack_.n_set() > 0,
			"rev_hes_sparsity: previous call to for_jac_sparsity did not "
			"use bool for interanl sparsity patterns."
		);
		// column dimension of internal sparstiy pattern
		size_t ell = for_jac_sparse_pack_.end();
		//
		// allocate memory for bool sparsity calculation
		// (sparsity pattern is emtpy after a resize)
		local::sparse_pack internal_hes;
		internal_hes.resize(num_var_tape_, ell);
		//
		// compute the Hessian sparsity pattern
		local::RevHesSweep(
			n,
			num_var_tape_,
			&play_,
			for_jac_sparse_pack_,
			rev_jac_pattern.data(),
			internal_hes

		);
		// get sparstiy pattern for independent variables
		local::get_internal_sparsity(
			transpose, ind_taddr_, internal_hes, pattern_out
		);
	}
	else
	{	CPPAD_ASSERT_KNOWN(
			for_jac_sparse_set_.n_set() > 0,
			"rev_hes_sparsity: previous call to for_jac_sparsity did not "
			"use bool for interanl sparsity patterns."
		);
		// column dimension of internal sparstiy pattern
		size_t ell = for_jac_sparse_set_.end();
		//
		// allocate memory for bool sparsity calculation
		// (sparsity pattern is emtpy after a resize)
		local::sparse_list internal_hes;
		internal_hes.resize(num_var_tape_, ell);
		//
		// compute the Hessian sparsity pattern
		local::RevHesSweep(
			n,
			num_var_tape_,
			&play_,
			for_jac_sparse_set_,
			rev_jac_pattern.data(),
			internal_hes

		);
		// get sparstiy pattern for independent variables
		local::get_internal_sparsity(
			transpose, ind_taddr_, internal_hes, pattern_out
		);
	}
	return;
}
} // END_CPPAD_NAMESPACE
# endif
