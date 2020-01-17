// $Id$
# ifndef CPPAD_CORE_BASE_HASH_HPP
# define CPPAD_CORE_BASE_HASH_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin base_hash$$
$spell
	alloc
	Cpp
	adouble
	valgrind
	const
	inline
$$

$section Base Type Requirements for Hash Coding Values$$

$head Syntax$$
$icode%code% = hash_code(%x%)%$$

$head Purpose$$
CppAD uses a table of $icode Base$$ type values when recording
$codei%AD<%Base%>%$$ operations.
A hashing function is used to reduce number of values stored in this table;
for example, it is not necessary to store the value 3.0 every
time it is used as a $cref/parameter/parvar/$$.

$head Default$$
The default hashing function works with the set of bits that correspond
to a $icode Base$$ value.
In most cases this works well, but in some cases
it does not. For example, in the
$cref base_adolc.hpp$$ case, an $code adouble$$ value can have
fields that are not initialized and $code valgrind$$ reported an error
when these are used to form the hash code.

$head x$$
This argument has prototype
$codei%
	const %Base%& %x
%$$
It is the value we are forming a hash code for.

$head code$$
The return value $icode code$$ has prototype
$codei%
	unsigned short %code%
%$$
It is the hash code corresponding to $icode x$$. This intention is the
commonly used values will have different hash codes.
The hash code must satisfy
$codei%
	%code% < CPPAD_HASH_TABLE_SIZE
%$$
so that it is a valid index into the hash code table.

$head inline$$
If you define this function, it should declare it to be $code inline$$,
so that you do not get multiple definitions from different compilation units.

$head Example$$
See the $code base_alloc$$ $cref/hash_code/base_alloc.hpp/hash_code/$$
and the $code adouble$$ $cref/hash_code/base_adolc.hpp/hash_code/$$.

$end
*/

/*!
\def CPPAD_HASH_TABLE_SIZE
the codes retruned by hash_code are between zero and CPPAD_HASH_TABLE_SIZE
minus one.
*/
# define CPPAD_HASH_TABLE_SIZE 10000

# endif
