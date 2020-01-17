# ifndef CPPAD_CORE_FUN_CHECK_HPP
# define CPPAD_CORE_FUN_CHECK_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin FunCheck$$
$spell
	exp
	bool
	const
	Taylor
$$


$section Check an ADFun Sequence of Operations$$

$head Syntax$$
$icode%ok% = FunCheck(%f%, %g%, %x%, %r%, %a%)%$$
$pre
$$
$bold See Also$$
$cref CompareChange$$


$head Purpose$$
We use $latex F : B^n \rightarrow B^m$$ to denote the
$cref/AD function/glossary/AD Function/$$ corresponding to $icode f$$.
We use $latex G : B^n \rightarrow B^m$$ to denote the
function corresponding to the C++ function object $icode g$$.
This routine check if
$latex \[
	F(x) = G(x)
\]$$
If $latex F(x) \neq G(x)$$, the
$cref/operation sequence/glossary/Operation/Sequence/$$
corresponding to $icode f$$ does not represents the algorithm used
by $icode g$$ to calculate values for $latex G$$
(see $cref/Discussion/FunCheck/Discussion/$$ below).

$head f$$
The $code FunCheck$$ argument $icode f$$ has prototype
$codei%
	ADFun<%Base%> %f%
%$$
Note that the $cref ADFun$$ object $icode f$$ is not $code const$$
(see $cref/Forward/FunCheck/FunCheck Uses Forward/$$ below).

$head g$$
The $code FunCheck$$ argument $icode g$$ has prototype
$codei%
	%Fun% &%g%
%$$
($icode Fun$$ is defined the properties of $icode g$$).
The C++ function object $icode g$$ supports the syntax
$codei%
	%y% = %g%(%x%)
%$$
which computes $latex y = G(x)$$.

$subhead x$$
The $icode g$$ argument $icode x$$ has prototype
$codei%
	const %Vector% &%x%
%$$
(see $cref/Vector/FunCheck/Vector/$$ below)
and its size
must be equal to $icode n$$, the dimension of the
$cref/domain/seq_property/Domain/$$ space for $icode f$$.

$head y$$
The $icode g$$ result $icode y$$ has prototype
$codei%
	%Vector% %y%
%$$
and its value is $latex G(x)$$.
The size of $icode y$$
is equal to $icode m$$, the dimension of the
$cref/range/seq_property/Range/$$ space for $icode f$$.

$head x$$
The $code FunCheck$$ argument $icode x$$ has prototype
$codei%
	const %Vector% &%x%
%$$
and its size
must be equal to $icode n$$, the dimension of the
$cref/domain/seq_property/Domain/$$ space for $icode f$$.
This specifies that point at which to compare the values
calculated by $icode f$$ and $icode G$$.

$head r$$
The $code FunCheck$$ argument $icode r$$ has prototype
$codei%
	const %Base% &%r%
%$$
It specifies the relative error the element by element
comparison of the value of $latex F(x)$$ and $latex G(x)$$.

$head a$$
The $code FunCheck$$ argument $icode a$$ has prototype
$codei%
	const %Base% &%a%
%$$
It specifies the absolute error the element by element
comparison of the value of $latex F(x)$$ and $latex G(x)$$.

$head ok$$
The $code FunCheck$$ result $icode ok$$ has prototype
$codei%
	bool %ok%
%$$
It is true, if for $latex i = 0 , \ldots , m-1$$
either the relative error bound is satisfied
$latex \[
| F_i (x) - G_i (x) |
\leq
r ( | F_i (x) | + | G_i (x) | )
\] $$
or the absolute error bound is satisfied
$latex \[
	| F_i (x) - G_i (x) | \leq a
\] $$
It is false if for some $latex (i, j)$$ neither
of these bounds is satisfied.

$head Vector$$
The type $icode Vector$$ must be a $cref SimpleVector$$ class with
$cref/elements of type/SimpleVector/Elements of Specified Type/$$
$icode Base$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$head FunCheck Uses Forward$$
After each call to $cref Forward$$,
the object $icode f$$ contains the corresponding
$cref/Taylor coefficients/glossary/Taylor Coefficient/$$.
After $code FunCheck$$,
the previous calls to $cref Forward$$ are undefined.

$head Discussion$$
Suppose that the algorithm corresponding to $icode g$$ contains
$codei%
	if( %x% >= 0 )
		%y% = exp(%x%)
	else	%y% = exp(-%x%)
%$$
where $icode x$$ and $icode y$$ are $codei%AD<double>%$$ objects.
It follows that the
AD of $code double$$ $cref/operation sequence/glossary/Operation/Sequence/$$
depends on the value of $icode x$$.
If the sequence of operations stored in $icode f$$ corresponds to
$icode g$$ with $latex x \geq 0$$,
the function values computed using $icode f$$ when $latex x < 0$$
will not agree with the function values computed by $latex g$$.
This is because the operation sequence corresponding to $icode g$$ changed
(and hence the object $icode f$$ does not represent the function
$latex G$$ for this value of $icode x$$).
In this case, you probably want to re-tape the calculations
performed by $icode g$$ with the
$cref/independent variables/glossary/Tape/Independent Variable/$$
equal to the values in $icode x$$
(so AD operation sequence properly represents the algorithm
for this value of independent variables).


$head Example$$
$children%
	example/general/fun_check.cpp
%$$
The file
$cref fun_check.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$end
---------------------------------------------------------------------------
*/

namespace CppAD {
	template <class Base, class Fun, class Vector>
	bool FunCheck(
		ADFun<Base>  &f ,
		Fun          &g ,
		const Vector &x ,
		const Base   &r ,
		const Base   &a )
	{	bool ok = true;

		size_t m   = f.Range();
		Vector yf  = f.Forward(0, x);
		Vector yg  = g(x);

		size_t i;
		for(i = 0; i < m; i++)
			ok  &= NearEqual(yf[i], yg[i], r, a);
		return ok;
	}
}

# endif
