# ifndef CPPAD_UTILITY_SET_UNION_HPP
# define CPPAD_UTILITY_SET_UNION_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin set_union$$
$spell
	set
	const
	std
$$

$section Union of Standard Sets$$

$head Syntax$$
$icode%result% = set_union(%left%, %right%)%$$

$head Purpose$$
This is a simplified (and restricted) interface to
the $code std::union$$ operation.

$head Element$$
This is the type of the elements of the sets.

$head left$$
This argument has prototype
$codei%
	const std::set<%Element%>& %left%
%$$

$head right$$
This argument has prototype
$codei%
	const std::set<%Element%>& %right%
%$$

$head result$$
The return value has prototype
$codei%
	std::set<%Element%>& %result%
%$$
It contains the union of $icode left$$ and $icode right$$.
Note that C++11 detects that the return value is a temporary
and uses it for the result instead of making a separate copy.

$children%
	example/utility/set_union.cpp
%$$
$head Example$$
The file $cref set_union.cpp$$ contains an example and test of this
operation. It returns true if the test passes and false otherwise.


$end
*/

# include <set>
# include <algorithm>
# include <iterator>

namespace CppAD {
	template <class Element>
	std::set<Element> set_union(
		const std::set<Element>&     left   ,
		const std::set<Element>&     right  )
	{	std::set<Element> result;
		std::set_union(
			left.begin()              ,
			left.end()                ,
			right.begin()             ,
			right.end()               ,
			std::inserter(result, result.begin())
		);
		return result;
	}
}

# endif
