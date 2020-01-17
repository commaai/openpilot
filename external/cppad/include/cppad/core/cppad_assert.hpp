// $Id$
# ifndef CPPAD_CORE_CPPAD_ASSERT_HPP
# define CPPAD_CORE_CPPAD_ASSERT_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*!
\file cppad_assert.hpp
Define the CppAD error checking macros (all of which begin with CPPAD_ASSERT_)
*/

/*
-------------------------------------------------------------------------------
$begin cppad_assert$$
$spell
	CppAD
	exp
	const
	bool
$$


$section CppAD Assertions During Execution$$
$mindex assert macro CPPAD_ASSERT_KNOWN CPPAD_ASSERT_UNKNOWN$$

$head Syntax$$
$codei%CPPAD_ASSERT_KNOWN(%exp%, %msg%)
%$$
$codei%CPPAD_ASSERT_UNKNOWN(%exp%)%$$


$head Purpose$$
These CppAD macros are used to detect and report errors.
They are documented here because they correspond to the C++
source code that the error is reported at.

$head NDEBUG$$
If the preprocessor symbol
$cref/NDEBUG/Faq/Speed/NDEBUG/$$ is defined,
these macros do nothing; i.e., they are optimized out.

$head Restriction$$
The CppAD user should not uses these macros.
You can however write your own macros that do not begin with $code CPPAD$$
and that call the $cref/CppAD error handler/ErrorHandler/$$.

$head Known$$
The $code CPPAD_ASSERT_KNOWN$$ macro is used to check for an error
with a known cause.
For example, many CppAD routines uses these macros
to make sure their arguments conform to their specifications.

$head Unknown$$
The $code CPPAD_ASSERT_UNKNOWN$$ macro is used to check that the
CppAD internal data structures conform as expected.
If this is not the case, CppAD does not know why the error has
occurred; for example, the user may have written past the end
of an allocated array.

$head Exp$$
The argument $icode exp$$ is a C++ source code expression
that results in a $code bool$$ value that should be true.
If it is false, an error has occurred.
This expression may be execute any number of times
(including zero times) so it must have not side effects.

$head Msg$$
The argument $icode msg$$ has prototype
$codei%
	const char *%msg%
%$$
and contains a $code '\0'$$ terminated character string.
This string is a description of the error
corresponding to $icode exp$$ being false.

$head Error Handler$$
These macros use the
$cref/CppAD error handler/ErrorHandler/$$ to report errors.
This error handler can be replaced by the user.

$end
------------------------------------------------------------------------------
*/

# include <cassert>
# include <iostream>
# include <cppad/utility/error_handler.hpp>

/*!
\def CPPAD_ASSERT_KNOWN(exp, msg)
Check that \a exp is true, if not print \a msg and terminate execution.

The C++ expression \a exp is expected to be true.
If it is false,
the CppAD use has made an error that is described by \a msg.
If the preprocessor symbol \a NDEBUG is not defined,
and \a exp is false,
this macro will report the source code line number at
which this expected result occurred.
In addition, it will print the specified error message \a msg.
*/
# ifdef NDEBUG
# define CPPAD_ASSERT_KNOWN(exp, msg)  // do nothing
# else
# define CPPAD_ASSERT_KNOWN(exp, msg)           \
{	if( ! ( exp ) )                         \
	CppAD::ErrorHandler::Call(              \
		true       ,                    \
		__LINE__   ,                    \
		__FILE__   ,                    \
		#exp       ,                    \
		msg        );                   \
}
# endif

/*!
\def CPPAD_ASSERT_UNKNOWN(exp)
Check that \a exp is true, if not terminate execution.

The C++ expression \a exp is expected to be true.
If it is false,
CppAD has detected an error but does not know the cause of the error.
If the preprocessor symbol \a NDEBUG is not defined,
and \a exp is false,
this macro will report the source code line number at
which this expected result occurred.
*/
# ifdef NDEBUG
# define CPPAD_ASSERT_UNKNOWN(exp)      // do nothing
# else
# define CPPAD_ASSERT_UNKNOWN(exp)              \
{	if( ! ( exp ) )                         \
	CppAD::ErrorHandler::Call(              \
		false      ,                    \
		__LINE__   ,                    \
		__FILE__   ,                    \
		#exp       ,                    \
		""         );                   \
}
# endif

/*!
\def CPPAD_ASSERT_NARG_NRES(op, n_arg, n_res)
Check that operator \a op has the specified number of of arguments and results.

If \a NDEBUG is not defined and either the number of arguments
or the number of results are not as expected,
execution is terminated and the source code line number is reported.
*/
# define CPPAD_ASSERT_NARG_NRES(op, n_arg, n_res)   \
	CPPAD_ASSERT_UNKNOWN( NumArg(op) == n_arg ) \
	CPPAD_ASSERT_UNKNOWN( NumRes(op) == n_res )

/*!
\def CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL
Check that the first call to a routine is not during parallel execution mode.

If \c NDEBUG is defined, this macro has no effect
(not even the definition of (\c assert_first_call).
Otherwise, the variable
\code
	static bool assert_first_call
\endcode
is defined and if the first call is executed in parallel mode,
execution is terminated and the source code line number is reported.
*/
# ifdef NDEBUG
# define CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL
# else
# define CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL                           \
	static bool assert_first_call = true;                              \
	if( assert_first_call )                                            \
	{	CPPAD_ASSERT_KNOWN(                                           \
		! (CppAD::thread_alloc::in_parallel() ),                      \
		"In parallel mode and parallel_setup has not been called."    \
		);                                                            \
		assert_first_call = false;                                    \
	}
# endif

/*!
\def CPPAD_ASSERT_ARG_BEFORE_RESULT
Check that operator arguments come before result.

If \c NDEBUG is defined, this macro has no effect,
otherwise it calls the function assert_arg_before_result.
*/
# ifdef NDEBUG
# define CPPAD_ASSERT_ARG_BEFORE_RESULT(op, arg, result)
# else
# define CPPAD_ASSERT_ARG_BEFORE_RESULT(op, arg, result) \
	assert_arg_before_result(op, arg, result)

# endif

# endif
