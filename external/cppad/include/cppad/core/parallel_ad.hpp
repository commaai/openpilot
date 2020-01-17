// $Id$
# ifndef CPPAD_CORE_PARALLEL_AD_HPP
# define CPPAD_CORE_PARALLEL_AD_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin parallel_ad$$
$spell
	CppAD
	num
	std
$$

$section Enable AD Calculations During Parallel Mode$$

$head Syntax$$
$codei%parallel_ad<%Base%>()%$$

$head Purpose$$
The function
$codei%parallel_ad<%Base%>()%$$
must be called before any $codei%AD<%Base>%$$ objects are used
in $cref/parallel/ta_in_parallel/$$ mode.
In addition, if this routine is called after one is done using
parallel mode, it will free extra memory used to keep track of
the multiple $codei%AD<%Base%>%$$ tapes required for parallel execution.

$head Discussion$$
By default, for each $codei%AD<%Base%>%$$ class there is only one
tape that records $cref/AD of Base/glossary/AD of Base/$$ operations.
This tape is a global variable and hence it cannot be used
by multiple threads at the same time.
The $cref/parallel_setup/ta_parallel_setup/$$ function informs CppAD of the
maximum number of threads that can be active in parallel mode.
This routine does extra setup
(and teardown) for the particular $icode Base$$ type.

$head CheckSimpleVector$$
This routine has the side effect of calling the routines
$codei%
	CheckSimpleVector< %Type%, CppAD::vector<%Type%> >()
%$$
where $icode Type$$ is $icode Base$$ and $codei%AD<%Base%>%$$.

$head Example$$
The files
$cref team_openmp.cpp$$,
$cref team_bthread.cpp$$, and
$cref team_pthread.cpp$$,
contain examples and tests that implement this function.

$head Restriction$$
This routine cannot be called in parallel mode or while
there is a tape recording $codei%AD<%Base%>%$$ operations.

$end
-----------------------------------------------------------------------------
*/

# include <cppad/local/std_set.hpp>

// BEGIN CppAD namespace
namespace CppAD {

/*!
Enable parallel execution mode with <code>AD<Base></code> by initializing
static variables that my be used.
*/

template <class Base>
void parallel_ad(void)
{	CPPAD_ASSERT_KNOWN(
		! thread_alloc::in_parallel() ,
		"parallel_ad must be called before entering parallel execution mode."
	);
	CPPAD_ASSERT_KNOWN(
		AD<Base>::tape_ptr() == CPPAD_NULL ,
		"parallel_ad cannot be called while a tape recording is in progress"
	);

	// ensure statics in following functions are initialized
	elapsed_seconds();
	ErrorHandler::Current();
	local::NumArg(local::BeginOp);
	local::NumRes(local::BeginOp);
	local::one_element_std_set<size_t>();
	local::two_element_std_set<size_t>();

	// the sparse_pack class has member functions with static data
	local::sparse_pack sp;
	sp.resize(1, 1);       // so can call add_element
	sp.add_element(0, 0);  // has static data
	sp.clear(0);           // has static data
	sp.is_element(0, 0);   // has static data
	local::sparse_pack::const_iterator itr(sp, 0); // has static data
	++itr;                                  // has static data

	// statics that depend on the value of Base
	AD<Base>::tape_id_handle(0);
	AD<Base>::tape_handle(0);
	AD<Base>::tape_manage(tape_manage_clear);
	discrete<Base>::List();
	CheckSimpleVector< Base, CppAD::vector<Base> >();
	CheckSimpleVector< AD<Base>, CppAD::vector< AD<Base> > >();

}

} // END CppAD namespace

# endif
