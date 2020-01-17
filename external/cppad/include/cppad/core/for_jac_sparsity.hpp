# ifndef CPPAD_CORE_FOR_JAC_SPARSITY_HPP
# define CPPAD_CORE_FOR_JAC_SPARSITY_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin for_jac_sparsity$$
$spell
	Jacobian
	jac
	bool
	const
	rc
	cpp
$$

$section Forward Mode Jacobian Sparsity Patterns$$

$head Syntax$$
$icode%f%.for_jac_sparsity(
	%pattern_in%, %transpose%, %dependency%, %internal_bool%, %pattern_out%
)%$$

$head Purpose$$
We use $latex F : \B{R}^n \rightarrow \B{R}^m$$ to denote the
$cref/AD function/glossary/AD Function/$$ corresponding to
the operation sequence stored in $icode f$$.
Fix $latex R \in \B{R}^{n \times \ell}$$ and define the function
$latex \[
	J(x) = F^{(1)} ( x ) * R
\] $$
Given the $cref/sparsity pattern/glossary/Sparsity Pattern/$$ for $latex R$$,
$code for_jac_sparsity$$ computes a sparsity pattern for $latex J(x)$$.

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
The $cref ADFun$$ object $icode f$$ is not $code const$$.
After a call to $code for_jac_sparsity$$, a sparsity pattern
for each of the variables in the operation sequence
is held in $icode f$$ for possible later use during
reverse Hessian sparsity calculations.

$subhead size_forward_bool$$
After $code for_jac_sparsity$$, if $icode k$$ is a $code size_t$$ object,
$codei%
	%k% = %f%.size_forward_bool()
%$$
sets $icode k$$ to the amount of memory (in unsigned character units)
used to store the
$cref/boolean vector/glossary/Sparsity Pattern/Boolean Vector/$$
sparsity patterns.
If $icode internal_bool$$ if false, $icode k$$ will be zero.
Otherwise it will be non-zero.
If you do not need this information for $cref RevSparseHes$$
calculations, it can be deleted
(and the corresponding memory freed) using
$codei%
	%f%.size_forward_bool(0)
%$$
after which $icode%f%.size_forward_bool()%$$ will return zero.

$subhead size_forward_set$$
After $code for_jac_sparsity$$, if $icode k$$ is a $code size_t$$ object,
$codei%
	%k% = %f%.size_forward_set()
%$$
sets $icode k$$ to the amount of memory (in unsigned character units)
used to store the
$cref/vector of sets/glossary/Sparsity Pattern/Vector of Sets/$$
sparsity patterns.
If $icode internal_bool$$ if true, $icode k$$ will be zero.
Otherwise it will be non-zero.
If you do not need this information for future $cref rev_hes_sparsity$$
calculations, it can be deleted
(and the corresponding memory freed) using
$codei%
	%f%.size_forward_set(0)
%$$
after which $icode%f%.size_forward_set()%$$ will return zero.

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
See $cref/pattern_in/for_jac_sparsity/pattern_in/$$ above and
$cref/pattern_out/for_jac_sparsity/pattern_out/$$ below.

$head dependency$$
This argument has prototype
$codei%
	bool %dependency%
%$$
see $cref/pattern_out/for_jac_sparsity/pattern_out/$$ below.

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
$latex R$$ is the $latex n \times n$$ identity matrix.
In this case, $icode pattern_out$$ is a sparsity pattern for
$latex F^{(1)} ( x )$$  ( $latex F^{(1)} (x)^\R{T}$$ )
if $icode transpose$$ is false (true).

$head Example$$
$children%
	example/sparse/for_jac_sparsity.cpp
%$$
The file
$cref for_jac_sparsity.cpp$$
contains an example and test of this operation.
It returns true if it succeeds and false otherwise.

$end
-----------------------------------------------------------------------------
*/
# include <cppad/core/ad_fun.hpp>
# include <cppad/local/sparse_internal.hpp>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE

/*!
Forward Jacobian sparsity patterns.

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
of boolean values. Othewise, a vector of standard sets is used.

\param pattern_out
The value of transpose is false (true),
the return value is a sparsity pattern for J(x) ( J(x)^T ) where
\f[
	J(x) = F^{(1)} (x) * R
\f]
Here F is the function corresponding to the operation sequence
and x is any argument value.
*/
template <class Base>
template <class SizeVector>
void ADFun<Base>::for_jac_sparsity(
	const sparse_rc<SizeVector>& pattern_in       ,
	bool                         transpose        ,
	bool                         dependency       ,
	bool                         internal_bool    ,
	sparse_rc<SizeVector>&       pattern_out      )
{	// number or rows, columns, and non-zeros in pattern_in
	size_t nr_in  = pattern_in.nr();
	size_t nc_in  = pattern_in.nc();
	//
	size_t n   = nr_in;
	size_t ell = nc_in;
	if( transpose )
		std::swap(n, ell);
	//
	CPPAD_ASSERT_KNOWN(
		n == Domain() ,
		"for_jac_sparsity: number rows in R "
		"is not equal number of independent variables."
	);
	bool zero_empty  = true;
	bool input_empty = true;
	if( internal_bool )
	{	// allocate memory for bool sparsity calculation
		// (sparsity pattern is emtpy after a resize)
		for_jac_sparse_pack_.resize(num_var_tape_, ell);
		for_jac_sparse_set_.resize(0, 0);
		//
		// set sparsity patttern for independent variables
		local::set_internal_sparsity(
			zero_empty            ,
			input_empty           ,
			transpose             ,
			ind_taddr_            ,
			for_jac_sparse_pack_  ,
			pattern_in
		);

		// compute sparsity for other variables
		local::ForJacSweep(
			dependency,
			n,
			num_var_tape_,
			&play_,
			for_jac_sparse_pack_
		);
		// set the output pattern
		local::get_internal_sparsity(
			transpose, dep_taddr_, for_jac_sparse_pack_, pattern_out
		);
	}
	else
	{
		// allocate memory for set sparsity calculation
		// (sparsity pattern is emtpy after a resize)
		for_jac_sparse_set_.resize(num_var_tape_, ell);
		for_jac_sparse_pack_.resize(0, 0);
		//
		// set sparsity patttern for independent variables
		local::set_internal_sparsity(
			zero_empty            ,
			input_empty           ,
			transpose             ,
			ind_taddr_            ,
			for_jac_sparse_set_   ,
			pattern_in
		);

		// compute sparsity for other variables
		local::ForJacSweep(
			dependency,
			n,
			num_var_tape_,
			&play_,
			for_jac_sparse_set_
		);
		// get the ouput pattern
		local::get_internal_sparsity(
			transpose, dep_taddr_, for_jac_sparse_set_, pattern_out
		);
	}
	return;
}


} // END_CPPAD_NAMESPACE
# endif
