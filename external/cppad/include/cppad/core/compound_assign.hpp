# ifndef CPPAD_CORE_COMPOUND_ASSIGN_HPP
# define CPPAD_CORE_COMPOUND_ASSIGN_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
-------------------------------------------------------------------------------
$begin compound_assign$$
$spell
	Op
	VecAD
	const
$$

$section AD Compound Assignment Operators$$
$mindex + add plus - subtract minus * multiply times / divide multiple$$






$head Syntax$$
$icode%x% %Op% %y%$$

$head Purpose$$
Performs compound assignment operations
where either $icode x$$ has type
$codei%AD<%Base%>%$$.

$head Op$$
The operator $icode Op$$ is one of the following
$table
$bold Op$$  $cnext $bold Meaning$$ $rnext
$code +=$$   $cnext $icode x$$ is assigned $icode x$$ plus $icode y$$ $rnext
$code -=$$   $cnext $icode x$$ is assigned $icode x$$ minus $icode y$$ $rnext
$code *=$$   $cnext $icode x$$ is assigned $icode x$$ times $icode y$$ $rnext
$code /=$$   $cnext $icode x$$ is assigned $icode x$$ divided by $icode y$$
$tend

$head Base$$
The type $icode Base$$ is determined by the operand $icode x$$.

$head x$$
The operand $icode x$$ has the following prototype
$codei%
	AD<%Base%> &%x%
%$$

$head y$$
The operand $icode y$$ has the following prototype
$codei%
	const %Type% &%y%
%$$
where $icode Type$$ is
$codei%VecAD<%Base%>::reference%$$,
$codei%AD<%Base%>%$$,
$icode Base$$, or
$code double$$.

$head Result$$
The result of this assignment
can be used as a reference to $icode x$$.
For example, if $icode z$$ has the following type
$codei%
	AD<%Base%> %z%
%$$
then the syntax
$codei%
	%z% = %x% += %y%
%$$
will compute $icode x$$ plus $icode y$$
and then assign this value to both $icode x$$ and $icode z$$.


$head Operation Sequence$$
This is an $cref/atomic/glossary/Operation/Atomic/$$
$cref/AD of Base/glossary/AD of Base/$$ operation
and hence it is part of the current
AD of $icode Base$$
$cref/operation sequence/glossary/Operation/Sequence/$$.

$children%
	example/general/add_eq.cpp%
	example/general/sub_eq.cpp%
	example/general/mul_eq.cpp%
	example/general/div_eq.cpp
%$$

$head Example$$
The following files contain examples and tests of these functions.
Each test returns true if it succeeds and false otherwise.
$table
$rref AddEq.cpp$$
$rref sub_eq.cpp$$
$rref mul_eq.cpp$$
$rref div_eq.cpp$$
$tend

$head Derivative$$
If $latex f$$ and $latex g$$ are
$cref/Base functions/glossary/Base Function/$$

$subhead Addition$$
$latex \[
	\D{[ f(x) + g(x) ]}{x} = \D{f(x)}{x} + \D{g(x)}{x}
\] $$

$subhead Subtraction$$
$latex \[
	\D{[ f(x) - g(x) ]}{x} = \D{f(x)}{x} - \D{g(x)}{x}
\] $$

$subhead Multiplication$$
$latex \[
	\D{[ f(x) * g(x) ]}{x} = g(x) * \D{f(x)}{x} + f(x) * \D{g(x)}{x}
\] $$

$subhead Division$$
$latex \[
	\D{[ f(x) / g(x) ]}{x} =
		[1/g(x)] * \D{f(x)}{x} - [f(x)/g(x)^2] * \D{g(x)}{x}
\] $$

$end
-----------------------------------------------------------------------------
*/
# include <cppad/core/add_eq.hpp>
# include <cppad/core/sub_eq.hpp>
# include <cppad/core/mul_eq.hpp>
# include <cppad/core/div_eq.hpp>

# endif
