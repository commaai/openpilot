// $Id: independent.hpp 3845 2016-11-19 01:50:47Z bradbell $
# ifndef CPPAD_LOCAL_INDEPENDENT_HPP
# define CPPAD_LOCAL_INDEPENDENT_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
namespace CppAD { namespace local { //  BEGIN_CPPAD_LOCAL_NAMESPACE
/*
\file local/independent.hpp
Implement the declaration of the independent variables
*/

/*!
Implementation of the declaration of independent variables (in local namespace).

\tparam VectorAD
This is simple vector type with elements of type AD<Base>.

\param x
Vector of the independent variablerd.

\param abort_op_index
operator index at which execution will be aborted (during  the recording
of operations). The value zero corresponds to not aborting (will not match).
*/
template <typename Base>
template <typename VectorAD>
void ADTape<Base>::Independent(VectorAD &x, size_t abort_op_index)
{
	// check VectorAD is Simple Vector class with AD<Base> elements
	CheckSimpleVector< AD<Base>, VectorAD>();

	// dimension of the domain space
	size_t n = x.size();
	CPPAD_ASSERT_KNOWN(
		n > 0,
		"Indepdendent: the argument vector x has zero size"
	);
	CPPAD_ASSERT_UNKNOWN( Rec_.num_var_rec() == 0 );

	// set the abort index before doing anything else
	Rec_.set_abort_op_index(abort_op_index);

	// mark the beginning of the tape and skip the first variable index
	// (zero) because parameters use taddr zero
	CPPAD_ASSERT_NARG_NRES(BeginOp, 1, 1);
	Rec_.PutOp(BeginOp);
	Rec_.PutArg(0);

	// place each of the independent variables in the tape
	CPPAD_ASSERT_NARG_NRES(InvOp, 0, 1);
	size_t j;
	for(j = 0; j < n; j++)
	{	// tape address for this independent variable
		x[j].taddr_ = Rec_.PutOp(InvOp);
		x[j].tape_id_    = id_;
		CPPAD_ASSERT_UNKNOWN( size_t(x[j].taddr_) == j+1 );
		CPPAD_ASSERT_UNKNOWN( Variable(x[j] ) );
	}

	// done specifying all of the independent variables
	size_independent_ = n;
}
} } // END_CPPAD_LOCAL_NAMESPACE

# endif
