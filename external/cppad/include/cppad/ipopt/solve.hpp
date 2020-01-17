// $Id: solve.hpp 3804 2016-03-20 15:08:46Z bradbell $
# ifndef CPPAD_IPOPT_SOLVE_HPP
# define CPPAD_IPOPT_SOLVE_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin ipopt_solve$$
$spell
	Jacobian
	Jacobians
	retape
	Bvector
	bool
	infeasibility
	const
	cpp
	cppad
	doesn't
	ADvector
	eval
	fg
	gl
	gu
	hpp
	inf
	ipopt
	maxiter
	naninf
	nf
	ng
	nx
	obj
	optimizer
	std
	xi
	xl
	xu
	zl
	zu
$$

$section Use Ipopt to Solve a Nonlinear Programming Problem$$

$head Syntax$$
$codei%# include <cppad/ipopt/solve.hpp>
%$$
$codei%ipopt::solve(
	%options%, %xi%, %xl%, %xu%, %gl%, %gu%, %fg_eval%, %solution%
)%$$

$head Purpose$$
The function $code ipopt::solve$$ solves nonlinear programming
problems of the form
$latex \[
\begin{array}{rll}
{\rm minimize}      & f (x)
\\
{\rm subject \; to} & gl \leq g(x) \leq gu
\\
                    & xl  \leq x   \leq xu
\end{array}
\] $$
This is done using
$href%
	http://www.coin-or.org/projects/Ipopt.xml%
	Ipopt
%$$
optimizer and CppAD for the derivative and sparsity calculations.

$head Include File$$
Currently, this routine
$cref/ipopt::solve/ipopt_solve/$$ is not included by the command
$codei%
	# include <cppad/cppad.hpp>
%$$
(Doing so would require the ipopt library to link
the corresponding program (even if $code ipopt::solve$$) was not used.)
For this reason,
if you are using $code ipopt::solve$$ you should use
$codei%
	# include <cppad/ipopt/solve.hpp>
%$$
which in turn will also include $code <cppad/cppad.hpp>$$.

$head Bvector$$
The type $icode Bvector$$ must be a $cref SimpleVector$$ class with
$cref/elements of type/SimpleVector/Elements of Specified Type/$$
$code bool$$.

$head Dvector$$
The type $icode DVector$$ must be a $cref SimpleVector$$ class with
$cref/elements of type/SimpleVector/Elements of Specified Type/$$
$code double$$.

$head options$$
The argument $icode options$$ has prototype
$codei%
	const std::string %options%
%$$
It contains a list of options.
Each option, including the last option,
is terminated by the $code '\n'$$ character.
Each line consists of two or three tokens separated by one or more spaces.

$subhead Retape$$
You can set the retape flag with the following syntax:
$codei%
	Retape %value%
%$$
If the value is $code true$$, $code ipopt::solve$$ with retape the
$cref/operation sequence/glossary/Operation/Sequence/$$ for each
new value of $icode x$$.
If the value is $code false$$, $code ipopt::solve$$
will tape the operation sequence at the value
of $icode xi$$ and use that sequence for the entire optimization process.
The default value is $code false$$.

$subhead Sparse$$
You can set the sparse Jacobian and Hessian flag with the following syntax:
$codei%
	Sparse %value% %direction%
%$$
If the value is $code true$$, $code ipopt::solve$$ will use a sparse
matrix representation for the computation of Jacobians and Hessians.
Otherwise, it will use a full matrix representation for
these calculations.
The default for $icode value$$ is $code false$$.
If sparse is true, retape must be false.
$pre

$$
It is unclear if $cref sparse_jacobian$$ would be faster user
forward or reverse mode so you are able to choose the direction.
If
$codei%
	%value% == true && %direction% == forward
%$$
the Jacobians will be calculated using $code SparseJacobianForward$$.
If
$codei%
	%value% == true && %direction% == reverse
%$$
the Jacobians will be calculated using $code SparseJacobianReverse$$.

$subhead String$$
You can set any Ipopt string option using a line with the following syntax:
$codei%
	String %name% %value%
%$$
Here $icode name$$ is any valid Ipopt string option
and $icode value$$ is its setting.

$subhead Numeric$$
You can set any Ipopt numeric option using a line with the following syntax:
$codei%
	Numeric %name% %value%
%$$
Here $icode name$$ is any valid Ipopt numeric option
and $icode value$$ is its setting.

$subhead Integer$$
You can set any Ipopt integer option using a line with the following syntax:
$codei%
	Integer %name% %value%
%$$
Here $icode name$$ is any valid Ipopt integer option
and $icode value$$ is its setting.

$head xi$$
The argument $icode xi$$ has prototype
$codei%
	const %Vector%& %xi%
%$$
and its size is equal to $icode nx$$.
It specifies the initial point where Ipopt starts the optimization process.

$head xl$$
The argument $icode xl$$ has prototype
$codei%
	const %Vector%& %xl%
%$$
and its size is equal to $icode nx$$.
It specifies the lower limits for the argument in the optimization problem.

$head xu$$
The argument $icode xu$$ has prototype
$codei%
	const %Vector%& %xu%
%$$
and its size is equal to $icode nx$$.
It specifies the upper limits for the argument in the optimization problem.

$head gl$$
The argument $icode gl$$ has prototype
$codei%
	const %Vector%& %gl%
%$$
and its size is equal to $icode ng$$.
It specifies the lower limits for the constraints in the optimization problem.

$head gu$$
The argument $icode gu$$ has prototype
$codei%
	const %Vector%& %gu%
%$$
and its size is equal to $icode ng$$.
It specifies the upper limits for the constraints in the optimization problem.

$head fg_eval$$
The argument $icode fg_eval$$ has prototype
$codei%
	%FG_eval% %fg_eval%
%$$
where the class $icode FG_eval$$ is unspecified except for the fact that
it supports the syntax
$codei%
	%FG_eval%::ADvector
	%fg_eval%(%fg%, %x%)
%$$
The type $icode ADvector$$
and the arguments to $icode fg$$, $icode x$$ have the following meaning:

$subhead ADvector$$
The type $icode%FG_eval%::ADvector%$$ must be a $cref SimpleVector$$ class with
$cref/elements of type/SimpleVector/Elements of Specified Type/$$
$code AD<double>$$.

$subhead x$$
The $icode fg_eval$$ argument $icode x$$ has prototype
$codei%
	const %ADvector%& %x%
%$$
where $icode%nx% = %x%.size()%$$.

$subhead fg$$
The $icode fg_eval$$ argument $icode fg$$ has prototype
$codei%
	%ADvector%& %fg%
%$$
where $codei%1 + %ng% = %fg%.size()%$$.
The input value of the elements of $icode fg$$ does not matter.
Upon return from $icode fg_eval$$,
$codei%
	%fg%[0] =%$$ $latex f (x)$$ $codei%
%$$
and   for $latex i = 0, \ldots , ng-1$$,
$codei%
	%fg%[1 + %i%] =%$$ $latex g_i (x)$$

$head solution$$
The argument $icode solution$$ has prototype
$codei%
	ipopt::solve_result<%Dvector%>& %solution%
%$$
After the optimization process is completed, $icode solution$$ contains
the following information:

$subhead status$$
The $icode status$$ field of $icode solution$$ has prototype
$codei%
	ipopt::solve_result<%Dvector%>::status_type %solution%.status
%$$
It is the final Ipopt status for the optimizer.
Here is a list of the possible values for the status:

$table
$icode status$$ $cnext Meaning
$rnext
not_defined $cnext
The optimizer did not return a final status for this problem.
$rnext
unknown $cnext
The status returned by the optimizer is not defined in the Ipopt
documentation for $code finalize_solution$$.
$rnext
success $cnext
Algorithm terminated successfully at a point satisfying the convergence
tolerances (see Ipopt options).
$rnext
maxiter_exceeded $cnext
The maximum number of iterations was exceeded (see Ipopt options).
$rnext
stop_at_tiny_step $cnext
Algorithm terminated because progress was very slow.
$rnext
stop_at_acceptable_point $cnext
Algorithm stopped at a point that was converged,
not to the 'desired' tolerances, but to 'acceptable' tolerances
(see Ipopt options).
$rnext
local_infeasibility $cnext
Algorithm converged to a non-feasible point
(problem may have no solution).
$rnext
user_requested_stop $cnext
This return value should not happen.
$rnext
diverging_iterates $cnext
It the iterates are diverging.
$rnext
restoration_failure $cnext
Restoration phase failed, algorithm doesn't know how to proceed.
$rnext
error_in_step_computation $cnext
An unrecoverable error occurred while Ipopt tried to
compute the search direction.
$rnext
invalid_number_detected $cnext
Algorithm received an invalid number (such as $code nan$$ or $code inf$$)
from the users function $icode%fg_info%.eval%$$ or from the CppAD evaluations
of its derivatives
(see the Ipopt option $code check_derivatives_for_naninf$$).
$rnext
internal_error $cnext
An unknown Ipopt internal error occurred.
Contact the Ipopt authors through the mailing list.
$tend

$subhead x$$
The $code x$$ field of $icode solution$$ has prototype
$codei%
	%Vector% %solution%.x
%$$
and its size is equal to $icode nx$$.
It is the final $latex x$$ value for the optimizer.

$subhead zl$$
The $code zl$$ field of $icode solution$$ has prototype
$codei%
	%Vector% %solution%.zl
%$$
and its size is equal to $icode nx$$.
It is the final Lagrange multipliers for the
lower bounds on $latex x$$.

$subhead zu$$
The $code zu$$ field of $icode solution$$ has prototype
$codei%
	%Vector% %solution%.zu
%$$
and its size is equal to $icode nx$$.
It is the final Lagrange multipliers for the
upper bounds on $latex x$$.

$subhead g$$
The $code g$$ field of $icode solution$$ has prototype
$codei%
	%Vector% %solution%.g
%$$
and its size is equal to $icode ng$$.
It is the final value for the constraint function $latex g(x)$$.

$subhead lambda$$
The $code lambda$$ field of $icode solution$$ has prototype
$codei%
	%Vector%> %solution%.lambda
%$$
and its size is equal to $icode ng$$.
It is the final value for the
Lagrange multipliers corresponding to the constraint function.

$subhead obj_value$$
The $code obj_value$$ field of $icode solution$$ has prototype
$codei%
	double %solution%.obj_value
%$$
It is the final value of the objective function $latex f(x)$$.

$children%
	example/ipopt_solve/get_started.cpp%
	example/ipopt_solve/retape.cpp%
	example/ipopt_solve/ode_inverse.cpp
%$$
$head Example$$
All the examples return true if it succeeds and false otherwise.

$subhead get_started$$
The file
$cref%example/ipopt_solve/get_started.cpp%ipopt_solve_get_started.cpp%$$
is an example and test of $code ipopt::solve$$
taken from the Ipopt manual.

$subhead retape$$
The file
$cref%example/ipopt_solve/retape.cpp%ipopt_solve_retape.cpp%$$
demonstrates when it is necessary to specify
$cref/retape/ipopt_solve/options/Retape/$$ as true.

$subhead ode_inverse$$
The file
$cref%example/ipopt_solve/ode_inverse.cpp%ipopt_solve_ode_inverse.cpp%$$
demonstrates using Ipopt to solve for parameters in an ODE model.

$end
-------------------------------------------------------------------------------
*/
# include <cppad/ipopt/solve_callback.hpp>

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
namespace ipopt {
/*!
\file solve.hpp
\brief Implement the ipopt::solve Nonlinear Programming Solver
*/

/*!
Use Ipopt to Solve a Nonlinear Programming Problem

\tparam Bvector
simple vector class with elements of type bool.

\tparam Dvector
simple vector class with elements of type double.

\tparam FG_eval
function object used to evaluate f(x) and g(x); see fg_eval below.
It must also support
\code
	FG_eval::ADvector
\endcode
to dentify the type used for the arguments to fg_eval.

\param options
list of options, one for each line.
Ipopt options (are optional) and have one of the following forms
\code
	String   name  value
	Numeric  name  value
	Integer  name  value
\endcode
The following other possible options are listed below:
\code
	Retape   value
\endcode


\param xi
initial argument value to start optimization procedure at.

\param xl
lower limit for argument during optimization

\param xu
upper limit for argument during optimization

\param gl
lower limit for g(x) during optimization.

\param gu
upper limit for g(x) during optimization.

\param fg_eval
function that evaluates the objective and constraints using the syntax
\code
	fg_eval(fg, x)
\endcode

\param solution
structure that holds the solution of the optimization.
*/
template <class Dvector, class FG_eval>
void solve(
	const std::string&                   options   ,
	const Dvector&                       xi        ,
	const Dvector&                       xl        ,
	const Dvector&                       xu        ,
	const Dvector&                       gl        ,
	const Dvector&                       gu        ,
	FG_eval&                             fg_eval   ,
	ipopt::solve_result<Dvector>&        solution  )
{	bool ok = true;

	typedef typename FG_eval::ADvector ADvector;

	CPPAD_ASSERT_KNOWN(
		xi.size() == xl.size() && xi.size() == xu.size() ,
		"ipopt::solve: size of xi, xl, and xu are not all equal."
	);
	CPPAD_ASSERT_KNOWN(
		gl.size() == gu.size() ,
		"ipopt::solve: size of gl and gu are not equal."
	);
	size_t nx = xi.size();
	size_t ng = gl.size();

	// Create an IpoptApplication
	using Ipopt::IpoptApplication;
	Ipopt::SmartPtr<IpoptApplication> app = new IpoptApplication();

	// process the options argument
	size_t begin_1, end_1, begin_2, end_2, begin_3, end_3;
	begin_1     = 0;
	bool retape          = false;
	bool sparse_forward  = false;
	bool sparse_reverse  = false;
	while( begin_1 < options.size() )
	{	// split this line into tokens
		while( options[begin_1] == ' ')
			begin_1++;
		end_1   = options.find_first_of(" \n", begin_1);
		begin_2 = end_1;
		while( options[begin_2] == ' ')
			begin_2++;
		end_2   = options.find_first_of(" \n", begin_2);
		begin_3 = end_2;
		while( options[begin_3] == ' ')
			begin_3++;
		end_3   = options.find_first_of(" \n", begin_3);

		// check for errors
		CPPAD_ASSERT_KNOWN(
			(end_1 != std::string::npos)  &
			(end_2 != std::string::npos)  &
			(end_3 != std::string::npos)  ,
			"ipopt::solve: missing '\\n' at end of an option line"
		);
		CPPAD_ASSERT_KNOWN(
			(end_1 > begin_1) & (end_2 > begin_2) ,
			"ipopt::solve: an option line does not have two tokens"
		);

		// get first two tokens
		std::string tok_1 = options.substr(begin_1, end_1 - begin_1);
		std::string tok_2 = options.substr(begin_2, end_2 - begin_2);

		// get third token
		std::string tok_3;
		bool three_tok = false;
		three_tok |= tok_1 == "Sparse";
		three_tok |= tok_1 == "String";
		three_tok |= tok_1 == "Numeric";
		three_tok |= tok_1 == "Integer";
		if( three_tok )
		{	CPPAD_ASSERT_KNOWN(
				(end_3 > begin_3) ,
				"ipopt::solve: a Sparse, String, Numeric, or Integer\n"
				"option line does not have three tokens."
			);
			tok_3 = options.substr(begin_3, end_3 - begin_3);
		}

		// switch on option type
		if( tok_1 == "Retape" )
		{	CPPAD_ASSERT_KNOWN(
				(tok_2 == "true") | (tok_2 == "false") ,
				"ipopt::solve: Retape value is not true or false"
			);
			retape = (tok_2 == "true");
		}
		else if( tok_1 == "Sparse" )
		{	CPPAD_ASSERT_KNOWN(
				(tok_2 == "true") | (tok_2 == "false") ,
				"ipopt::solve: Sparse value is not true or false"
			);
			CPPAD_ASSERT_KNOWN(
				(tok_3 == "forward") | (tok_3 == "reverse") ,
				"ipopt::solve: Sparse direction is not forward or reverse"
			);
			if( tok_2 == "false" )
			{	sparse_forward = false;
				sparse_reverse = false;
			}
			else
			{	sparse_forward = tok_3 == "forward";
				sparse_reverse = tok_3 == "reverse";
			}
		}
		else if ( tok_1 == "String" )
			app->Options()->SetStringValue(tok_2.c_str(), tok_3.c_str());
		else if ( tok_1 == "Numeric" )
		{	Ipopt::Number value = std::atof( tok_3.c_str() );
			app->Options()->SetNumericValue(tok_2.c_str(), value);
		}
		else if ( tok_1 == "Integer" )
		{	Ipopt::Index value = std::atoi( tok_3.c_str() );
			app->Options()->SetIntegerValue(tok_2.c_str(), value);
		}
		else	CPPAD_ASSERT_KNOWN(
			false,
			"ipopt::solve: First token is not one of\n"
			"Retape, Sparse, String, Numeric, Integer"
		);

		begin_1 = end_3;
		while( options[begin_1] == ' ')
			begin_1++;
		if( options[begin_1] != '\n' ) CPPAD_ASSERT_KNOWN(
			false,
			"ipopt::solve: either more than three tokens "
			"or no '\\n' at end of a line"
		);
		begin_1++;
	}
	CPPAD_ASSERT_KNOWN(
		! ( retape & (sparse_forward | sparse_reverse) ) ,
		"ipopt::solve: retape and sparse both true is not supported."
	);

	// Initialize the IpoptApplication and process the options
	Ipopt::ApplicationReturnStatus status = app->Initialize();
	ok    &= status == Ipopt::Solve_Succeeded;
	if( ! ok )
	{	solution.status = solve_result<Dvector>::unknown;
		return;
	}

	// Create an interface from Ipopt to this specific problem.
	// Note the assumption here that ADvector is same as cppd_ipopt::ADvector
	size_t nf = 1;
	Ipopt::SmartPtr<Ipopt::TNLP> cppad_nlp =
	new CppAD::ipopt::solve_callback<Dvector, ADvector, FG_eval>(
		nf,
		nx,
		ng,
		xi,
		xl,
		xu,
		gl,
		gu,
		fg_eval,
		retape,
		sparse_forward,
		sparse_reverse,
		solution
	);

	// Run the IpoptApplication
	app->OptimizeTNLP(cppad_nlp);

	return;
}

} // end ipopt namespace
} // END_CPPAD_NAMESPACE
# endif
