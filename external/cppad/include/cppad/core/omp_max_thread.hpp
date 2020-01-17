// $Id$
# ifndef CPPAD_CORE_OMP_MAX_THREAD_HPP
# define CPPAD_CORE_OMP_MAX_THREAD_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin omp_max_thread$$
$spell
	alloc
	num
	omp
	OpenMp
	CppAD
$$

$section OpenMP Parallel Setup$$
$mindex omp_max_thread$$

$head Deprecated 2011-06-23$$
Use $cref/thread_alloc::parallel_setup/ta_parallel_setup/$$
to set the number of threads.

$head Syntax$$
$codei%AD<%Base%>::omp_max_thread(%number%)
%$$

$head Purpose$$
By default, for each $codei%AD<%Base%>%$$ class there is only one
tape that records $cref/AD of Base/glossary/AD of Base/$$ operations.
This tape is a global variable and hence it cannot be used
by multiple OpenMP threads at the same time.
The $code omp_max_thread$$ function is used to set the
maximum number of OpenMP threads that can be active.
In this case, there is a different tape corresponding to each
$codei%AD<%Base%>%$$ class and thread pair.

$head number$$
The argument $icode number$$ has prototype
$codei%
	size_t %number%
%$$
It must be greater than zero and specifies the maximum number of
OpenMp threads that will be active at one time.


$head Independent$$
Each call to $cref/Independent(x)/Independent/$$
creates a new $cref/active/glossary/Tape/Active/$$ tape.
All of the operations with the corresponding variables
must be preformed by the same OpenMP thread.
This includes the corresponding call to
$cref/f.Dependent(x,y)/Dependent/$$ or the
$cref/ADFun f(x, y)/FunConstruct/Sequence Constructor/$$
during which the tape stops recording and the variables
become parameters.

$head Restriction$$
No tapes can be
$cref/active/glossary/Tape/Active/$$ when this function is called.

$end
-----------------------------------------------------------------------------
*/

// BEGIN CppAD namespace
namespace CppAD {

template <class Base>
void AD<Base>::omp_max_thread(size_t number)
{
# ifdef _OPENMP
	thread_alloc::parallel_setup(
		number, omp_alloc::in_parallel, omp_alloc::get_thread_num
	);
# else
	CPPAD_ASSERT_KNOWN(
		number == 1,
		"omp_max_thread: number > 1 and _OPENMP is not defined"
	);
# endif
	parallel_ad<Base>();
}

} // END CppAD namespace

# endif
