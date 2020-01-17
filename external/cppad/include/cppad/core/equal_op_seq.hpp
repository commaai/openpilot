# ifndef CPPAD_CORE_EQUAL_OP_SEQ_HPP
# define CPPAD_CORE_EQUAL_OP_SEQ_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
------------------------------------------------------------------------------
$begin EqualOpSeq$$
$spell
	Op
	const
	bool
$$


$section Check if Two Value are Identically Equal$$
$mindex EqualOpSeq operation sequence$$

$head Syntax$$
$icode%b% = EqualOpSeq(%x%, %y%)%$$

$head Purpose$$
Determine if two $icode x$$ and $icode y$$ are identically equal; i.e.,
not only is $icode%x% == %y%$$ true, but
if they are $cref/variables/glossary/Variable/$$,
they correspond have the same
$cref/operation sequence/glossary/Operation/Sequence/$$.

$head Motivation$$
Sometimes it is useful to cache information
and only recalculate when a function's arguments change.
In the case of AD variables,
it may be important not only when the argument values are equal,
but when they are related to the
$cref/independent variables/glossary/Tape/Independent Variable/$$
by the same operation sequence.
After the assignment
$codei%
	%y% = %x%
%$$
these two AD objects would not only have equal values,
but would also correspond to the same operation sequence.

$head x$$
The argument $icode x$$ has prototype
$codei%
	const AD<%Base%> &%x%
%$$

$head y$$
The argument $icode y$$ has prototype
$codei%
	const AD<%Base%> &%y%
%$$

$head b$$
The result $icode b$$ has prototype
$codei%
	bool %b%
%$$
The result is true if and only if one of the following cases holds:

$list number$$
Both $icode x$$ and $icode y$$ are variables
and correspond to the same operation sequence.
$lnext
Both $icode x$$ and $icode y$$ are parameters,
$icode Base$$ is an AD type,
and $codei%EqualOpSeq( Value(%x%) , Value(%y%) )%$$ is true.
$lnext
Both $icode x$$ and $icode y$$ are parameters,
$icode Base$$ is not an AD type,
and $icode%x% == %y%%$$ is true.
$lend


$head Example$$
$children%
	example/general/equal_op_seq.cpp
%$$
The file
$cref equal_op_seq.cpp$$
contains an example and test of $code EqualOpSeq$$.
It returns true if it succeeds and false otherwise.


$end
------------------------------------------------------------------------------
*/


namespace CppAD {
	template <class Base>
	CPPAD_INLINE_FRIEND_TEMPLATE_FUNCTION
	bool EqualOpSeq(const AD<Base> &x, const AD<Base> &y)
	{
		if( Parameter(x) )
		{	if( Parameter(y) )
				return EqualOpSeq(x.value_, y.value_);
			else	return false;
		}
		else if( Parameter(y) )
			return false;

		return (x.taddr_ == y.taddr_);
	}

}

# endif
