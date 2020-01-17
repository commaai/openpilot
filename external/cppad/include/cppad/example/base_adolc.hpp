# ifndef CPPAD_EXAMPLE_BASE_ADOLC_HPP
# define CPPAD_EXAMPLE_BASE_ADOLC_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin base_adolc.hpp$$
$spell
	stringstream
	struct
	string
	setprecision
	str
	valgrind
	azmul
	expm1
	atanh
	acosh
	asinh
	erf
	ifndef
	define
	endif
	Rel
	codassign
	eps
	std
	abs_geq
	fabs
	cppad.hpp
	undef
	Lt
	Le
	Eq
	Ge
	Gt
	namespace
	cassert
	condassign
	hpp
	bool
	const
	Adolc
	adouble
	CondExpOp
	inline
	enum
	CppAD
	pow
	acos
	asin
	atan
	cos
	cosh
	exp
	sqrt
$$


$section Enable use of AD<Base> where Base is Adolc's adouble Type$$

$head Syntax$$
$codei%# include <cppad/example/base_adolc.hpp>
%$$
$children%
	example/general/mul_level_adolc.cpp
%$$

$head Example$$
The file $cref mul_level_adolc.cpp$$ contains an example use of
Adolc's $code adouble$$ type for a CppAD $icode Base$$ type.
It returns true if it succeeds and false otherwise.
The file $cref mul_level_adolc_ode.cpp$$ contains a more realistic
(and complex) example.

$head Include Files$$
This file $code base_adolc.hpp$$ requires $code adouble$$ to be defined.
In addition, it is included before $code <cppad/cppad.hpp>$$,
but it needs to include parts of CppAD that are used by this file.
This is done with the following include commands:
$srccode%cpp% */
# include <adolc/adolc.h>
# include <cppad/base_require.hpp>
/* %$$

$head CondExpOp$$
The type $code adouble$$ supports a conditional assignment function
with the syntax
$codei%
	condassign(%a%, %b%, %c%, %d%)
%$$
which evaluates to
$codei%
	%a% = (%b% > 0) ? %c% : %d%;
%$$
This enables one to include conditionals in the recording of
$code adouble$$ operations and later evaluation for different
values of the independent variables
(in the same spirit as the CppAD $cref CondExp$$ function).
$srccode%cpp% */
namespace CppAD {
	inline adouble CondExpOp(
		enum  CppAD::CompareOp     cop ,
		const adouble            &left ,
		const adouble           &right ,
		const adouble        &trueCase ,
		const adouble       &falseCase )
	{	adouble result;
		switch( cop )
		{
			case CompareLt: // left < right
			condassign(result, right - left, trueCase, falseCase);
			break;

			case CompareLe: // left <= right
			condassign(result, left - right, falseCase, trueCase);
			break;

			case CompareEq: // left == right
			condassign(result, left - right, falseCase, trueCase);
			condassign(result, right - left, falseCase, result);
			break;

			case CompareGe: // left >= right
			condassign(result, right - left, falseCase, trueCase);
			break;

			case CompareGt: // left > right
			condassign(result, left - right, trueCase, falseCase);
			break;
			default:
			CppAD::ErrorHandler::Call(
				true     , __LINE__ , __FILE__ ,
				"CppAD::CondExp",
				"Error: for unknown reason."
			);
			result = trueCase;
		}
		return result;
	}
}
/* %$$

$head CondExpRel$$
The $cref/CPPAD_COND_EXP_REL/base_cond_exp/CondExpRel/$$ macro invocation
$srccode%cpp% */
namespace CppAD {
	CPPAD_COND_EXP_REL(adouble)
}
/* %$$

$head EqualOpSeq$$
The Adolc user interface does not specify a way to determine if
two $code adouble$$ variables correspond to the same operations sequence.
Make $code EqualOpSeq$$ an error if it gets used:
$srccode%cpp% */
namespace CppAD {
	inline bool EqualOpSeq(const adouble &x, const adouble &y)
	{	CppAD::ErrorHandler::Call(
			true     , __LINE__ , __FILE__ ,
			"CppAD::EqualOpSeq(x, y)",
			"Error: adouble does not support EqualOpSeq."
		);
		return false;
	}
}
/* %$$

$head Identical$$
The Adolc user interface does not specify a way to determine if an
$code adouble$$ depends on the independent variables.
To be safe (but slow) return $code false$$ in all the cases below.
$srccode%cpp% */
namespace CppAD {
	inline bool IdenticalPar(const adouble &x)
	{	return false; }
	inline bool IdenticalZero(const adouble &x)
	{	return false; }
	inline bool IdenticalOne(const adouble &x)
	{	return false; }
	inline bool IdenticalEqualPar(const adouble &x, const adouble &y)
	{	return false; }
}
/* %$$

$head Integer$$
$srccode%cpp% */
	inline int Integer(const adouble &x)
	{    return static_cast<int>( x.getValue() ); }
/* %$$

$head azmul$$
$srccode%cpp% */
namespace CppAD {
	CPPAD_AZMUL( adouble )
}
/* %$$

$head Ordered$$
$srccode%cpp% */
namespace CppAD {
	inline bool GreaterThanZero(const adouble &x)
	{    return (x > 0); }
	inline bool GreaterThanOrZero(const adouble &x)
	{    return (x >= 0); }
	inline bool LessThanZero(const adouble &x)
	{    return (x < 0); }
	inline bool LessThanOrZero(const adouble &x)
	{    return (x <= 0); }
	inline bool abs_geq(const adouble& x, const adouble& y)
	{	return fabs(x) >= fabs(y); }
}
/* %$$

$head Unary Standard Math$$
The following $cref/required/base_require/$$ functions
are defined by the Adolc package for the $code adouble$$ base case:
$pre
$$
$code acos$$,
$code asin$$,
$code atan$$,
$code cos$$,
$code cosh$$,
$code exp$$,
$code fabs$$,
$code log$$,
$code sin$$,
$code sinh$$,
$code sqrt$$,
$code tan$$.

$head erf, asinh, acosh, atanh, expm1, log1p$$
If the
$cref/erf, asinh, acosh, atanh, expm1, log1p
	/base_std_math
	/erf, asinh, acosh, atanh, expm1, log1p
/$$,
functions are supported by the compiler,
they must also be supported by a $icode Base$$ type;
The adolc package does not support these functions so make
their use an error:
$srccode%cpp% */
namespace CppAD {
# define CPPAD_BASE_ADOLC_NO_SUPPORT(fun)                         \
    inline adouble fun(const adouble& x)                          \
    {   CPPAD_ASSERT_KNOWN(                                       \
            false,                                                \
            #fun ": adolc does not support this function"         \
        );                                                        \
        return 0.0;                                               \
    }
# if CPPAD_USE_CPLUSPLUS_2011
	CPPAD_BASE_ADOLC_NO_SUPPORT(erf)
	CPPAD_BASE_ADOLC_NO_SUPPORT(asinh)
	CPPAD_BASE_ADOLC_NO_SUPPORT(acosh)
	CPPAD_BASE_ADOLC_NO_SUPPORT(atanh)
	CPPAD_BASE_ADOLC_NO_SUPPORT(expm1)
	CPPAD_BASE_ADOLC_NO_SUPPORT(log1p)
# endif
# undef CPPAD_BASE_ADOLC_NO_SUPPORT
}
/* %$$

$head sign$$
This $cref/required/base_require/$$ function is defined using the
$code codassign$$ function so that its $code adouble$$ operation sequence
does not depend on the value of $icode x$$.
$srccode%cpp% */
namespace CppAD {
	inline adouble sign(const adouble& x)
	{	adouble s_plus, s_minus, half(.5);
		// set s_plus to sign(x)/2,  except for case x == 0, s_plus = -.5
		condassign(s_plus,  +x, -half, +half);
		// set s_minus to -sign(x)/2, except for case x == 0, s_minus = -.5
		condassign(s_minus, -x, -half, +half);
		// set s to sign(x)
		return s_plus - s_minus;
	}
}
/* %$$

$head abs$$
This $cref/required/base_require/$$ function uses the adolc $code fabs$$
function:
$srccode%cpp% */
namespace CppAD {
	inline adouble abs(const adouble& x)
	{	return fabs(x); }
}
/* %$$

$head pow$$
This $cref/required/base_require/$$ function
is defined by the Adolc package for the $code adouble$$ base case.

$head numeric_limits$$
The following defines the CppAD $cref numeric_limits$$
for the type $code adouble$$:
$srccode%cpp% */
namespace CppAD {
	CPPAD_NUMERIC_LIMITS(double, adouble)
}
/* %$$

$head to_string$$
The following defines the CppAD $cref to_string$$ function
for the type $code adouble$$:
$srccode%cpp% */
namespace CppAD {
	template <> struct to_string_struct<adouble>
	{	std::string operator()(const adouble& x)
		{	std::stringstream os;
			int n_digits = 1 + std::numeric_limits<double>::digits10;
			os << std::setprecision(n_digits);
			os << x.value();
			return os.str();
		}
	};
}
/* %$$

$head hash_code$$
It appears that an $code adouble$$ object can have fields
that are not initialized.
This results in a $code valgrind$$ error when these fields are used by the
$cref/default/base_hash/Default/$$ hashing function.
For this reason, the $code adouble$$ class overrides the default definition.
$srccode|cpp| */
namespace CppAD {
	inline unsigned short hash_code(const adouble& x)
	{	unsigned short code = 0;
		double value = x.value();
		if( value == 0.0 )
			return code;
		double log_x = std::log( fabs( value ) );
		// assume log( std::numeric_limits<double>::max() ) is near 700
		code = static_cast<unsigned short>(
			(CPPAD_HASH_TABLE_SIZE / 700 + 1) * log_x
		);
		code = code % CPPAD_HASH_TABLE_SIZE;
		return code;
	}
}
/* |$$
Note that after the hash codes match, the
$cref/Identical/base_adolc.hpp/Identical/$$ function will be used
to make sure two values are the same and one can replace the other.
A more sophisticated implementation of the $code Identical$$ function
would detect which $code adouble$$ values depend on the
$code adouble$$ independent variables (and hence can change).


$end
*/
# endif

