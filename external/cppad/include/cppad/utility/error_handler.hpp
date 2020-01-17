# ifndef CPPAD_UTILITY_ERROR_HANDLER_HPP
# define CPPAD_UTILITY_ERROR_HANDLER_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin ErrorHandler$$
$spell
	cppad.hpp
	CppAD
	exp
	bool
	const
$$

$section Replacing the CppAD Error Handler$$
$mindex replace assert exception ErrorHandler$$

$head Syntax$$
$codei%# include <cppad/utility/error_handler.hpp>
%$$
$codei%ErrorHandler %info%(%handler%)
%$$
$codei%ErrorHandler::Call(%known%, %line%, %file%, %exp%, %msg%)
%$$

$head Constructor$$
When you construct a $code ErrorHandler$$ object,
the current CppAD error handler is replaced by $icode handler$$.
When the object is destructed, the previous CppAD error handler is restored.

$subhead Parallel Mode$$
The $code ErrorHandler$$ constructor and destructor cannot be called in
$cref/parallel/ta_in_parallel/$$ execution mode.
Furthermore, this rule is not abided by, a raw C++ $code assert$$,
instead of one that uses this error handler, will be generated.

$head Call$$
When $code ErrorHandler::Call$$ is called,
the current CppAD error handler is used to report an error.
This starts out as a default error handler and can be replaced
using the $code ErrorHandler$$ constructor.

$head info$$
The object $icode info$$ is used to store information
that is necessary to restore the previous CppAD error handler.
This is done when the destructor for $icode info$$ is called.


$head handler$$
The argument $icode handler$$ has prototype
$codei%
	void (*%handler%)
		(bool, int, const char *, const char *, const char *);
%$$
When an error is detected,
it is called with the syntax
$codei%
	%handler% (%known%, %line%, %file%, %exp%, %msg%)
%$$
This routine should not return; i.e., upon detection of the error,
the routine calling $icode handler$$ does not know how to proceed.

$head known$$
The $icode handler$$ argument $icode known$$ has prototype
$codei%
	bool %known%
%$$
If it is true, the error being reported is from a know problem.

$head line$$
The $icode handler$$ argument $icode line$$ has prototype
$codei%
	int %line%
%$$
It reports the source code line number where the error is detected.

$head file$$
The $icode handler$$ argument $icode file$$ has prototype
$codei%
	const char *%file%
%$$
and is a $code '\0'$$ terminated character vector.
It reports the source code file where the error is detected.

$head exp$$
The $icode handler$$ argument $icode exp$$ has prototype
$codei%
	const char *%exp%
%$$
and is a $code '\0'$$ terminated character vector.
It is a source code boolean expression that should have been true,
but is false,
and thereby causes this call to $icode handler$$.

$head msg$$
The $icode handler$$ argument $icode msg$$ has prototype
$codei%
	const char *%msg%
%$$
and is a $code '\0'$$ terminated character vector.
It reports the meaning of the error from the C++ programmers point of view.

$children%
	example/utility/error_handler.cpp%
	cppad/core/cppad_assert.hpp
%$$
$head Example$$
The file
$cref error_handler.cpp$$
contains an example and test a test of using this routine.
It returns true if it succeeds and false otherwise.

$end
---------------------------------------------------------------------------
*/

# include <iostream>

# include <cppad/configure.hpp>
# include <cppad/local/set_get_in_parallel.hpp>
# include <cassert>
# include <cstdlib>

namespace CppAD { // BEGIN CppAD namespace

class ErrorHandler {
	template <class Base>
	friend void parallel_ad(void);
public:
	typedef void (*Handler)
		(bool, int, const char *, const char *, const char *);

	// construct a new handler
	ErrorHandler(Handler handler) : previous( Current() )
	{	if( local::set_get_in_parallel(0) )
		{	bool known       = true;
			int  line        = __LINE__;
			const char* file = __FILE__;
			const char* exp  = "! local::set_get_in_parallel(0)";
			const char* msg  =
				"Using ErrorHandler constructor in parallel mode.";
			Call(known, line, file, exp, msg);
		}
		Current() = handler;
	}

	// destructor for an error handler
	~ErrorHandler(void)
	{	if( local::set_get_in_parallel(0) )
		{	bool known       = true;
			int  line        = __LINE__;
			const char* file = __FILE__;
			const char* exp  = "! local::set_get_in_parallel(0)";
			const char* msg  =
				"Using ErrorHandler destructor in parallel mode.";
			Call(known, line, file, exp, msg);
		}
		Current() = previous;
	}

	// report an error
	static void Call(
		bool        known,
		int         line ,
		const char *file ,
		const char *exp  ,
		const char *msg  )
	{	Handler handler = Current();
		handler(known, line, file, exp, msg);
	}

private:
	const Handler previous;

	// The default error handler
	static void Default(
		bool        known,
		int         line ,
		const char *file ,
		const char *exp  ,
		const char *msg  )
	{	using std::cerr;
		using std::endl;

		cerr << CPPAD_PACKAGE_STRING;
		if( known )
			cerr << " error from a known source:" << endl;
		else	cerr << " error from unknown source"  << endl;
		if( msg[0] != '\0' )
			cerr << msg << endl;
		cerr << "Error detected by false result for"  << endl;
		cerr << "    "     << exp                     << endl;
		cerr << "at line " << line << " in the file " << endl;
		cerr << "    "     << file                    << endl;

		// terminate program execution
		assert(false);

		// termination when NDEBUG is defined
		std::exit(1);
	}

	// current error handler
	static Handler &Current(void)
	{	static bool first_call = true;
		static Handler current = Default;
		if( first_call )
		{	if( local::set_get_in_parallel(0) )
			{	bool known       = false;
				int  line        = __LINE__;
				const char* file = __FILE__;
				const char* exp  = "";
				const char* msg  = "";
				Call(known, line, file, exp, msg);
			}
			first_call = false;
		}
		return current;
	}
};

} // END CppAD namespace



# endif
