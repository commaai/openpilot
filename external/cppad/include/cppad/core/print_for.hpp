# ifndef CPPAD_CORE_PRINT_FOR_HPP
# define CPPAD_CORE_PRINT_FOR_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin PrintFor$$
$spell
	pos
	var
	VecAD
	std
	cout
	const
$$


$section Printing AD Values During Forward Mode$$
$mindex print text output debug$$

$head Syntax$$
$icode%f%.Forward(0, %x%)
%$$
$codei%PrintFor(%before%, %var%)
%$$
$codei%PrintFor(%pos%, %before%, %var%, %after%)
%$$

$head Purpose$$
The $cref/zero order forward/forward_zero/$$ mode command
$codei%
	%f%.Forward(0, %x%)
%$$
assigns the
$cref/independent variable/glossary/Tape/Independent Variable/$$ vector
equal to $icode x$$.
It then computes a value for all of the dependent variables in the
$cref/operation sequence/glossary/Operation/Sequence/$$ corresponding
to $icode f$$.
Putting a $code PrintFor$$ in the operation sequence will
cause the value of $icode var$$, corresponding to $icode x$$,
to be printed during zero order forward operations.

$head f.Forward(0, x)$$
The objects $icode f$$, $icode x$$, and the purpose
for this operation, are documented in $cref Forward$$.

$head pos$$
If present, the argument $icode pos$$ has one of the following prototypes
$codei%
	const AD<%Base%>&               %pos%
	const VecAD<%Base%>::reference& %pos%
%$$
In this case
the text and $icode var$$ will be printed if and only if
$icode pos$$ is not greater than zero and a finite number.

$head before$$
The argument $icode before$$ has prototype
$codei%
	const char* %before%
%$$
This text is written to $code std::cout$$ before $icode var$$.

$head var$$
The argument $icode var$$ has one of the following prototypes
$codei%
	const AD<%Base%>&               %var%
	const VecAD<%Base%>::reference& %var%
%$$
The value of $icode var$$, that corresponds to $icode x$$,
is written to $code std::cout$$ during the execution of
$codei%
	%f%.Forward(0, %x%)
%$$
Note that $icode var$$ may be a
$cref/variable/glossary/Variable/$$ or
$cref/parameter/glossary/Parameter/$$.
(A parameters value does not depend on the value of
the independent variable vector $icode x$$.)

$head after$$
The argument $icode after$$ has prototype
$codei%
	const char* %after%
%$$
This text is written to $code std::cout$$ after $icode var$$.

$head Redirecting Output$$
You can redirect this output to any standard output stream; see the
$cref/s/forward_order/s/$$ in the forward mode documentation.

$head Discussion$$
This is helpful for understanding why tape evaluations
have trouble.
For example, if one of the operations in $icode f$$ is
$codei%log(%var%)%$$ and $icode%var% <= 0%$$,
the corresponding result will be $cref nan$$.

$head Alternative$$
The $cref ad_output$$ section describes the normal
printing of values; i.e., printing when the corresponding
code is executed.

$head Example$$
$children%
	example/print_for/print_for.cpp%
	example/general/print_for.cpp
%$$
The program
$cref print_for_cout.cpp$$
is an example and test that prints to standard output.
The output of this program
states the conditions for passing and failing the test.
The function
$cref print_for_string.cpp$$
is an example and test that prints to an standard string stream.
This function automatically check for correct output.

$end
------------------------------------------------------------------------------
*/

# include <cstring>

namespace CppAD {
	template <class Base>
	void PrintFor(const AD<Base>& pos,
		const char *before, const AD<Base>& var, const char* after)
	{	CPPAD_ASSERT_NARG_NRES(local::PriOp, 5, 0);

		// check for case where we are not recording operations
		local::ADTape<Base>* tape = AD<Base>::tape_ptr();
		if( tape == CPPAD_NULL )
			return;

		CPPAD_ASSERT_KNOWN(
			std::strlen(before) <= 1000 ,
			"PrintFor: length of before is greater than 1000 characters"
		);
		CPPAD_ASSERT_KNOWN(
			std::strlen(after) <= 1000 ,
			"PrintFor: length of after is greater than 1000 characters"
		);
		addr_t ind0, ind1, ind2, ind3, ind4;

		// ind[0] = base 2 representation of the value [Var(pos), Var(var)]
		ind0 = 0;

		// ind[1] = address for pos
		if( Parameter(pos) )
			ind1  = tape->Rec_.PutPar(pos.value_);
		else
		{	ind0 += 1;
			ind1  = pos.taddr_;
		}

		// ind[2] = address of before
		ind2 = tape->Rec_.PutTxt(before);

		// ind[3] = address for var
		if( Parameter(var) )
			ind3  = tape->Rec_.PutPar(var.value_);
		else
		{	ind0 += 2;
			ind3  = var.taddr_;
		}

		// ind[4] = address of after
		ind4 = tape->Rec_.PutTxt(after);

		// put the operator in the tape
		tape->Rec_.PutArg(ind0, ind1, ind2, ind3, ind4);
		tape->Rec_.PutOp(local::PriOp);
	}
	// Fold all other cases into the case above
	template <class Base>
	void PrintFor(const char* before, const AD<Base>& var)
	{	PrintFor(AD<Base>(0), before, var, "" ); }
	//
	template <class Base>
	void PrintFor(const char* before, const VecAD_reference<Base>& var)
	{	PrintFor(AD<Base>(0), before, var.ADBase(), "" ); }
	//
	template <class Base>
	void PrintFor(
		const VecAD_reference<Base>& pos    ,
		const char                  *before ,
		const VecAD_reference<Base>& var    ,
		const char                  *after  )
	{	PrintFor(pos.ADBase(), before, var.ADBase(), after); }
	//
	template <class Base>
	void PrintFor(
		const VecAD_reference<Base>& pos    ,
		const char                  *before ,
		const AD<Base>&              var    ,
		const char                  *after  )
	{	PrintFor(pos.ADBase(), before, var, after); }
	//
	template <class Base>
	void PrintFor(
		const AD<Base>&              pos    ,
		const char                  *before ,
		const VecAD_reference<Base>& var    ,
		const char                  *after  )
	{	PrintFor(pos, before, var.ADBase(), after); }
}

# endif
