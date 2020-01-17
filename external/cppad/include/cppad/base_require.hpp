// $Id: base_require.hpp 3845 2016-11-19 01:50:47Z bradbell $
# ifndef CPPAD_BASE_REQUIRE_HPP
# define CPPAD_BASE_REQUIRE_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin base_require$$
$spell
	azmul
	ostream
	alloc
	eps
	std
	Lt
	Le
	Eq
	Ge
	Gt
	cppad.hpp
	namespace
	optimizations
	bool
	const
	CppAD
	enum
	Lt
	Le
	Eq
	Ge
	Gt
	inline
	Op
	std
	CondExp
$$

$section AD<Base> Requirements for a CppAD Base Type$$

$head Syntax$$
$code # include <cppad/base_require.hpp>$$

$head Purpose$$
This section lists the requirements for the type
$icode Base$$ so that the type $codei%AD<%Base%>%$$ can be used.

$head API Warning$$
Defining a CppAD $icode Base$$ type is an advanced use of CppAD.
This part of the CppAD API changes with time. The most common change
is adding more requirements.
Search for $code base_require$$ in the
current $cref whats_new$$ section for these changes.

$head Standard Base Types$$
In the case where $icode Base$$ is
$code float$$,
$code double$$,
$code std::complex<float>$$,
$code std::complex<double>$$,
or $codei%AD<%Other%>%$$,
these requirements are provided by including the file
$code cppad/cppad.hpp$$.

$head Include Order$$
If you are linking a non-standard base type to CppAD,
you must first include the file $code cppad/base_require.hpp$$,
then provide the specifications below,
and then include the file $code cppad/cppad.hpp$$.

$head Numeric Type$$
The type $icode Base$$ must support all the operations for a
$cref NumericType$$.

$head Output Operator$$
The type $icode Base$$ must support the syntax
$codei%
	%os% << %x%
%$$
where $icode os$$ is an $code std::ostream&$$
and $icode x$$ is a $code const base_alloc&$$.
For example, see
$cref/base_alloc/base_alloc.hpp/Output Operator/$$.

$head Integer$$
The type $icode Base$$ must support the syntax
$codei%
	%i% = CppAD::Integer(%x%)
%$$
which converts $icode x$$ to an $code int$$.
The argument $icode x$$ has prototype
$codei%
	const %Base%& %x%
%$$
and the return value $icode i$$ has prototype
$codei%
	int %i%
%$$

$subhead Suggestion$$
In many cases, the $icode Base$$ version of the $code Integer$$ function
can be defined by
$codei%
namespace CppAD {
	inline int Integer(const %Base%& x)
	{	return static_cast<int>(x); }
}
%$$
For example, see
$cref/base_float/base_float.hpp/Integer/$$ and
$cref/base_alloc/base_alloc.hpp/Integer/$$.

$head Absolute Zero, azmul$$
The type $icode Base$$ must support the syntax
$codei%
	%z% = azmul(%x%, %y%)
%$$
see; $cref azmul$$.
The following preprocessor macro invocation suffices
(for most $icode Base$$ types):
$codei%
namespace CppAD {
	CPPAD_AZMUL(%Base%)
}
%$$
where the macro is defined by
$srccode%cpp% */
# define CPPAD_AZMUL(Base) \
    inline Base azmul(const Base& x, const Base& y) \
    {   Base zero(0.0);   \
        if( x == zero ) \
            return zero;  \
        return x * y;     \
    }
/* %$$

$childtable%
	omh/base_require/base_member.omh%
	cppad/core/base_cond_exp.hpp%
	omh/base_require/base_identical.omh%
	omh/base_require/base_ordered.omh%
	cppad/core/base_std_math.hpp%
	cppad/core/base_limits.hpp%
	cppad/core/base_to_string.hpp%
	cppad/core/base_hash.hpp%
	omh/base_require/base_example.omh
%$$

$end
*/

// definitions that must come before base implementations
# include <cppad/utility/error_handler.hpp>
# include <cppad/core/define.hpp>
# include <cppad/core/cppad_assert.hpp>
# include <cppad/local/declare_ad.hpp>

// grouping documentation by feature
# include <cppad/core/base_cond_exp.hpp>
# include <cppad/core/base_std_math.hpp>
# include <cppad/core/base_limits.hpp>
# include <cppad/core/base_to_string.hpp>
# include <cppad/core/base_hash.hpp>

// must define template class numeric_limits before the base cases
# include <cppad/core/numeric_limits.hpp>
# include <cppad/core/epsilon.hpp> // deprecated

// base cases that come with CppAD
# include <cppad/core/base_float.hpp>
# include <cppad/core/base_double.hpp>
# include <cppad/core/base_complex.hpp>

// deprecated base type
# include <cppad/core/zdouble.hpp>

# endif
