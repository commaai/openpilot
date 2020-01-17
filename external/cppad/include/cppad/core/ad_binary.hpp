# ifndef CPPAD_CORE_AD_BINARY_HPP
# define CPPAD_CORE_AD_BINARY_HPP

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
$begin ad_binary$$
$spell
	Op
	VecAD
	const
$$

$section AD Binary Arithmetic Operators$$
$mindex + add plus - subtract minus * multiply times / divide$$







$head Syntax$$
$icode%z% = %x% %Op% %y%$$

$head Purpose$$
Performs arithmetic operations where either $icode x$$ or $icode y$$
has type
$codei%AD<%Base%>%$$ or
$cref%VecAD<Base>::reference%VecAD%VecAD<Base>::reference%$$.

$head Op$$
The operator $icode Op$$ is one of the following
$table
$bold Op$$  $cnext $bold Meaning$$ $rnext
$code +$$   $cnext $icode z$$ is $icode x$$ plus $icode y$$ $rnext
$code -$$   $cnext $icode z$$ is $icode x$$ minus $icode y$$ $rnext
$code *$$   $cnext $icode z$$ is $icode x$$ times $icode y$$ $rnext
$code /$$   $cnext $icode z$$ is $icode x$$ divided by $icode y$$
$tend

$head Base$$
The type $icode Base$$ is determined by the operand that
has type $codei%AD<%Base%>%$$ or $codei%VecAD<%Base%>::reference%$$.

$head x$$
The operand $icode x$$ has the following prototype
$codei%
	const %Type% &%x%
%$$
where $icode Type$$ is
$codei%VecAD<%Base%>::reference%$$,
$codei%AD<%Base%>%$$,
$icode Base$$, or
$code double$$.

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


$head z$$
The result $icode z$$ has the following prototype
$codei%
	%Type% %z%
%$$
where $icode Type$$ is
$codei%AD<%Base%>%$$.

$head Operation Sequence$$
This is an $cref/atomic/glossary/Operation/Atomic/$$
$cref/AD of Base/glossary/AD of Base/$$ operation
and hence it is part of the current
AD of $icode Base$$
$cref/operation sequence/glossary/Operation/Sequence/$$.

$children%
	example/general/add.cpp%
	example/general/sub.cpp%
	example/general/mul.cpp%
	example/general/div.cpp
%$$

$head Example$$
The following files contain examples and tests of these functions.
Each test returns true if it succeeds and false otherwise.
$table
$rref add.cpp$$
$rref sub.cpp$$
$rref mul.cpp$$
$rref div.cpp$$
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
# include <cppad/core/add.hpp>
# include <cppad/core/sub.hpp>
# include <cppad/core/mul.hpp>
# include <cppad/core/div.hpp>

# endif
