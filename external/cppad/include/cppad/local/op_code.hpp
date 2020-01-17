# ifndef CPPAD_LOCAL_OP_CODE_HPP
# define CPPAD_LOCAL_OP_CODE_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
# include <string>
# include <sstream>
# include <iomanip>

# include <cppad/core/define.hpp>
# include <cppad/core/cppad_assert.hpp>

// needed before one can use CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL
# include <cppad/utility/thread_alloc.hpp>

namespace CppAD { namespace local { // BEGIN_CPPAD_LOCAL_NAMESPACE
/*!
\file op_code.hpp
Defines the OpCode enum type and functions related to it.

*/


/*!
Type used to distinguish different AD< \a Base > atomic operations.

Each of the operators ends with the characters Op. Ignoring the Op at the end,
the operators appear in alphabetical order. Binary operation where both
operands have type AD< \a Base > use the following convention for thier endings:
\verbatim
    Ending  Left-Operand  Right-Operand
      pvOp     parameter       variable
      vpOp      variable      parameter
      vvOp      variable       variable
\endverbatim
For example, AddpvOp represents the addition operator where the left
operand is a parameter and the right operand is a variable.
*/
// alphabetical order is checked by bin/check_op_code.sh
enum OpCode {
	AbsOp,    // fabs(variable)
	AcosOp,   // acos(variable)
	AcoshOp,  // acosh(variable)
	AddpvOp,  // parameter  + variable
	AddvvOp,  // variable   + variable
	AsinOp,   // asin(variable)
	AsinhOp,  // asinh(variable)
	AtanOp,   // atan(variable)
	AtanhOp,  // atanh(variable)
	BeginOp,  // used to mark the beginning of the tape
	CExpOp,   // CondExpRel(left, right, trueCase, falseCase)
	// arg[0]     = the Rel operator: Lt, Le, Eq, Ge, Gt, or Ne
	// arg[1] & 1 = is left a variable
	// arg[1] & 2 = is right a variable
	// arg[1] & 4 = is trueCase a variable
	// arg[1] & 8 = is falseCase a variable
	// arg[2]     = index correspoding to left
	// arg[3]     = index correspoding to right
	// arg[4]     = index correspoding to trueCase
	// arg[5]     = index correspoding to falseCase
	CosOp,    // cos(variable)
	CoshOp,   // cosh(variable)
	CSkipOp,  // Conditional skip
	// arg[0]     = the Rel operator: Lt, Le, Eq, Ge, Gt, or Ne
	// arg[1] & 1 = is left a variable
	// arg[1] & 2 = is right a variable
	// arg[2]     = index correspoding to left
	// arg[3]     = index correspoding to right
	// arg[4] = number of operations to skip if CExpOp comparision is true
	// arg[5] = number of operations to skip if CExpOp comparision is false
	// arg[6] -> arg[5+arg[4]]               = skip operations if true
	// arg[6+arg[4]] -> arg[5+arg[4]+arg[5]] = skip operations if false
	// arg[6+arg[4]+arg[5]] = arg[4] + arg[5]
	CSumOp,   // Cummulative summation
	// arg[0] = number of addition variables in summation
	// arg[1] = number of subtraction variables in summation
	// arg[2] = index of parameter that initializes summation
	// arg[3] -> arg[2+arg[0]] = index for positive variables
	// arg[3+arg[0]] -> arg[2+arg[0]+arg[1]] = index for minus variables
	// arg[3+arg[0]+arg[1]] = arg[0] + arg[1]
	DisOp,    // discrete::eval(index, variable)
	DivpvOp,  // parameter  / variable
	DivvpOp,  // variable   / parameter
	DivvvOp,  // variable   / variable
	EndOp,    // used to mark the end of the tape
	EqpvOp,   // parameter  == variable
	EqvvOp,   // variable   == variable
	ErfOp,    // erf(variable)
	ExpOp,    // exp(variable)
	Expm1Op,  // expm1(variable)
	InvOp,    // independent variable
	LdpOp,    // z[parameter]
	LdvOp,    // z[variable]
	LepvOp,   // parameter <= variable
	LevpOp,   // variable  <= parameter
	LevvOp,   // variable  <= variable
	LogOp,    // log(variable)
	Log1pOp,  // log1p(variable)
	LtpvOp,   // parameter < variable
	LtvpOp,   // variable  < parameter
	LtvvOp,   // variable  < variable
	MulpvOp,  // parameter  * variable
	MulvvOp,  // variable   * variable
	NepvOp,   // parameter  != variable
	NevvOp,   // variable   != variable
	ParOp,    // parameter
	PowpvOp,  // pow(parameter,   variable)
	PowvpOp,  // pow(variable,    parameter)
	PowvvOp,  // pow(variable,    variable)
	PriOp,    // PrintFor(text, parameter or variable, parameter or variable)
	SignOp,   // sign(variable)
	SinOp,    // sin(variable)
	SinhOp,   // sinh(variable)
	SqrtOp,   // sqrt(variable)
	StppOp,   // z[parameter] = parameter
	StpvOp,   // z[parameter] = variable
	StvpOp,   // z[variable]  = parameter
	StvvOp,   // z[variable]  = variable
	SubpvOp,  // parameter  - variable
	SubvpOp,  // variable   - parameter
	SubvvOp,  // variable   - variable
	TanOp,    // tan(variable)
	TanhOp,   // tan(variable)
	// user atomic operation codes
	UserOp,   // start of a user atomic operaiton
	// arg[0] = index of the operation if atomic_base<Base> class
	// arg[1] = extra information passed trough by deprecated old atomic class
	// arg[2] = number of arguments to this atomic function
	// arg[3] = number of results for this atomic function
	UsrapOp,  // this user atomic argument is a parameter
	UsravOp,  // this user atomic argument is a variable
	UsrrpOp,  // this user atomic result is a parameter
	UsrrvOp,  // this user atomic result is a variable
	ZmulpvOp, // azmul(parameter, variable)
	ZmulvpOp, // azmul(variabe,  parameter)
	ZmulvvOp, // azmul(variable, variable)
	NumberOp  // number of operator codes (not an operator)
};
// Note that bin/check_op_code.sh assumes the pattern '^\tNumberOp$' occurs
// at the end of this list and only at the end of this list.

/*!
Number of arguments for a specified operator.

\return
Number of arguments corresponding to the specified operator.

\param op
Operator for which we are fetching the number of arugments.

\par NumArgTable
this table specifes the number of arguments stored for each
occurance of the operator that is the i-th value in the OpCode enum type.
For example, for the first three OpCode enum values we have
\verbatim
OpCode   j   NumArgTable[j]  Meaning
AbsOp    0                1  index of variable we are taking absolute value of
AcosOp   1                1  index of variable we are taking acos of
AcoshOp  2                1  index of variable we are taking acosh of
\endverbatim
Note that the meaning of the arguments depends on the operator.
*/
inline size_t NumArg( OpCode op)
{	CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL;

	// agreement with OpCode is checked by bin/check_op_code.sh
	static const size_t NumArgTable[] = {
		1, // AbsOp
		1, // AcosOp
		1, // AcoshOp
		2, // AddpvOp
		2, // AddvvOp
		1, // AsinOp
		1, // AsinhOp
		1, // AtanOp
		1, // AtanhOp
		1, // BeginOp  offset first real argument to have index 1
		6, // CExpOp
		1, // CosOp
		1, // CoshOp
		0, // CSkipOp  (actually has a variable number of arguments, not zero)
		0, // CSumOp   (actually has a variable number of arguments, not zero)
		2, // DisOp
		2, // DivpvOp
		2, // DivvpOp
		2, // DivvvOp
		0, // EndOp
		2, // EqpvOp
		2, // EqvvOp
		3, // ErfOp
		1, // ExpOp
		1, // Expm1Op
		0, // InvOp
		3, // LdpOp
		3, // LdvOp
		2, // LepvOp
		2, // LevpOp
		2, // LevvOp
		1, // LogOp
		1, // Log1pOp
		2, // LtpvOp
		2, // LtvpOp
		2, // LtvvOp
		2, // MulpvOp
		2, // MulvvOp
		2, // NepvOp
		2, // NevvOp
		1, // ParOp
		2, // PowpvOp
		2, // PowvpOp
		2, // PowvvOp
		5, // PriOp
		1, // SignOp
		1, // SinOp
		1, // SinhOp
		1, // SqrtOp
		3, // StppOp
		3, // StpvOp
		3, // StvpOp
		3, // StvvOp
		2, // SubpvOp
		2, // SubvpOp
		2, // SubvvOp
		1, // TanOp
		1, // TanhOp
		4, // UserOp
		1, // UsrapOp
		1, // UsravOp
		1, // UsrrpOp
		0, // UsrrvOp
		2, // ZmulpvOp
		2, // ZmulvpOp
		2, // ZmulvvOp
		0  // NumberOp not used
	};
# ifndef NDEBUG
	// only do these checks once to save time
	static bool first = true;
	if( first )
	{	CPPAD_ASSERT_UNKNOWN( size_t(NumberOp) + 1 ==
			sizeof(NumArgTable) / sizeof(NumArgTable[0])
		);
		CPPAD_ASSERT_UNKNOWN( size_t(NumberOp) <=
			std::numeric_limits<CPPAD_OP_CODE_TYPE>::max()
		);
		first = false;
	}
	// do this check every time
	CPPAD_ASSERT_UNKNOWN( size_t(op) < size_t(NumberOp) );
# endif

	return NumArgTable[op];
}

/*!
Number of variables resulting from the specified operation.

\param op
Operator for which we are fecching the number of results.

\par NumResTable
table specifes the number of varibles that result for each
occurance of the operator that is the i-th value in the OpCode enum type.
For example, for the first three OpCode enum values we have
\verbatim
OpCode   j   NumResTable[j]  Meaning
AbsOp    0                1  variable that is the result of the absolute value
AcosOp   1                2  acos(x) and sqrt(1-x*x) are required for this op
AcoshOp  2                2  acosh(x) and sqrt(x*x-1) are required for this op
\endverbatim
*/
inline size_t NumRes(OpCode op)
{	CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL;

	// agreement with OpCode is checked by bin/check_op_code.sh
	static const size_t NumResTable[] = {
		1, // AbsOp
		2, // AcosOp
		2, // AcoshOp
		1, // AddpvOp
		1, // AddvvOp
		2, // AsinOp
		2, // AsinhOp
		2, // AtanOp
		2, // AtanhOp
		1, // BeginOp  offsets first variable to have index one (not zero)
		1, // CExpOp
		2, // CosOp
		2, // CoshOp
		0, // CSkipOp
		1, // CSumOp
		1, // DisOp
		1, // DivpvOp
		1, // DivvpOp
		1, // DivvvOp
		0, // EndOp
		0, // EqpvOp
		0, // EqvvOp
		5, // ErfOp
		1, // ExpOp
		1, // Expm1Op
		1, // InvOp
		1, // LdpOp
		1, // LdvOp
		0, // LepvOp
		0, // LevpOp
		0, // LevvOp
		1, // LogOp
		1, // Log1pOp
		0, // LtpvOp
		0, // LtvpOp
		0, // LtvvOp
		1, // MulpvOp
		1, // MulvvOp
		0, // NepvOp
		0, // NevvOp
		1, // ParOp
		3, // PowpvOp
		3, // PowvpOp
		3, // PowvvOp
		0, // PriOp
		1, // SignOp
		2, // SinOp
		2, // SinhOp
		1, // SqrtOp
		0, // StppOp
		0, // StpvOp
		0, // StvpOp
		0, // StvvOp
		1, // SubpvOp
		1, // SubvpOp
		1, // SubvvOp
		2, // TanOp
		2, // TanhOp
		0, // UserOp
		0, // UsrapOp
		0, // UsravOp
		0, // UsrrpOp
		1, // UsrrvOp
		1, // ZmulpvOp
		1, // ZmulvpOp
		1, // ZmulvvOp
		0  // NumberOp not used and avoids g++ 4.3.2 warn when pycppad builds
	};
	// check ensuring conversion to size_t is as expected
	CPPAD_ASSERT_UNKNOWN( size_t(NumberOp) + 1 ==
		sizeof(NumResTable) / sizeof(NumResTable[0])
	);
	// this test ensures that all indices are within the table
	CPPAD_ASSERT_UNKNOWN( size_t(op) < size_t(NumberOp) );

	return NumResTable[op];
}


/*!
Fetch the name for a specified operation.

\return
name of the specified operation.

\param op
Operator for which we are fetching the name
*/
inline const char* OpName(OpCode op)
{	// agreement with OpCode is checked by bin/check_op_code.sh
	static const char *OpNameTable[] = {
		"Abs"   ,
		"Acos"  ,
		"Acosh" ,
		"Addpv" ,
		"Addvv" ,
		"Asin"  ,
		"Asinh" ,
		"Atan"  ,
		"Atanh" ,
		"Begin" ,
		"CExp"  ,
		"Cos"   ,
		"Cosh"  ,
		"CSkip" ,
		"CSum"  ,
		"Dis"   ,
		"Divpv" ,
		"Divvp" ,
		"Divvv" ,
		"End"   ,
		"Eqpv"  ,
		"Eqvv"  ,
		"Erf"   ,
		"Exp"   ,
		"Expm1" ,
		"Inv"   ,
		"Ldp"   ,
		"Ldv"   ,
		"Lepv"  ,
		"Levp"  ,
		"Levv"  ,
		"Log"   ,
		"Log1p" ,
		"Ltpv"  ,
		"Ltvp"  ,
		"Ltvv"  ,
		"Mulpv" ,
		"Mulvv" ,
		"Nepv"  ,
		"Nevv"  ,
		"Par"   ,
		"Powpv" ,
		"Powvp" ,
		"Powvv" ,
		"Pri"   ,
		"Sign"  ,
		"Sin"   ,
		"Sinh"  ,
		"Sqrt"  ,
		"Stpp"  ,
		"Stpv"  ,
		"Stvp"  ,
		"Stvv"  ,
		"Subpv" ,
		"Subvp" ,
		"Subvv" ,
		"Tan"   ,
		"Tanh"  ,
		"User"  ,
		"Usrap" ,
		"Usrav" ,
		"Usrrp" ,
		"Usrrv" ,
		"Zmulpv",
		"Zmulvp",
		"Zmulvv",
		"Number"  // not used
	};
	// check ensuring conversion to size_t is as expected
	CPPAD_ASSERT_UNKNOWN(
		size_t(NumberOp) + 1 == sizeof(OpNameTable)/sizeof(OpNameTable[0])
	);
	// this test ensures that all indices are within the table
	CPPAD_ASSERT_UNKNOWN( size_t(op) < size_t(NumberOp) );

	return OpNameTable[op];
}

/*!
Prints a single field corresponding to an operator.

A specified leader is printed in front of the value
and then the value is left justified in the following width character.

\tparam Type
is the type of the value we are printing.

\param os
is the stream that we are printing to.

\param leader
are characters printed before the value.

\param value
is the value being printed.

\param width
is the number of character to print the value in.
If the value does not fit in the width, the value is replace
by width '*' characters.
*/
template <class Type>
void printOpField(
	std::ostream      &os ,
	const char *   leader ,
	const Type     &value ,
	size_t          width )
{
	std::ostringstream buffer;
	std::string        str;

	// first print the leader
	os << leader;

	// print the value into an internal buffer
	buffer << std::setw(width) << value;
	str = buffer.str();

	// length of the string
	size_t len = str.size();
	if( len > width )
	{	size_t i;
		for(i = 0; i < width-1; i++)
			os << str[i];
		os << "*";
		return;
	}

	// count number of spaces at begining
	size_t nspace = 0;
	while(str[nspace] == ' ' && nspace < len)
		nspace++;

	// left justify the string
	size_t i = nspace;
	while( i < len )
		os << str[i++];

	i = width - len + nspace;
	while(i--)
		os << " ";
}

/*!
Prints a single operator and its operands

\tparam Base
Is the base type for these AD< \a Base > operations.

\param os
is the output stream that the information is printed on.

\param play
Is the entire recording for the tape that this operator is in.

\param i_op
is the index for the operator corresponding to this operation.

\param i_var
is the index for the variable corresponding to the result of this operation
(if NumRes(op) > 0).

\param op
The operator code (OpCode) for this operation.

\param ind
is the vector of argument indices for this operation
(must have NumArg(op) elements).
*/
template <class Base>
void printOp(
	std::ostream&          os     ,
	const local::player<Base>*    play   ,
	size_t                 i_op   ,
	size_t                 i_var  ,
	OpCode                 op     ,
	const addr_t*          ind    )
{	size_t i;
	CPPAD_ASSERT_KNOWN(
		! thread_alloc::in_parallel() ,
		"cannot print trace of AD operations in parallel mode"
	);
	static const char *CompareOpName[] =
		{ "Lt", "Le", "Eq", "Ge", "Gt", "Ne" };

	// print operator
	printOpField(os,  "o=",      i_op,  5);
	if( NumRes(op) > 0 && op != BeginOp )
		printOpField(os,  "v=",      i_var, 5);
	else	printOpField(os,  "v=",      "",    5);
	if( op == CExpOp || op == CSkipOp )
	{	printOpField(os, "", OpName(op), 5);
		printOpField(os, "", CompareOpName[ ind[0] ], 3);
	}
	else	printOpField(os, "", OpName(op), 8);

	// print other fields
	size_t ncol = 5;
	switch( op )
	{
		case CSkipOp:
		/*
		ind[0]     = the Rel operator: Lt, Le, Eq, Ge, Gt, or Ne
		ind[1] & 1 = is left a variable
		ind[1] & 2 = is right a variable
		ind[2]     = index correspoding to left
		ind[3]     = index correspoding to right
		ind[4] = number of operations to skip if CExpOp comparision is true
		ind[5] = number of operations to skip if CExpOp comparision is false
		ind[6] -> ind[5+ind[4]]               = skip operations if true
		ind[6+ind[4]] -> ind[5+ind[4]+ind[5]] = skip operations if false
		ind[6+ind[4]+ind[5]] = ind[4] + ind[5]
		*/
		CPPAD_ASSERT_UNKNOWN( ind[6+ind[4]+ind[5]] == ind[4]+ind[5] );
		CPPAD_ASSERT_UNKNOWN(ind[1] != 0);
		if( ind[1] & 1 )
			printOpField(os, " vl=", ind[2], ncol);
		else	printOpField(os, " pl=", play->GetPar(ind[2]), ncol);
		if( ind[1] & 2 )
			printOpField(os, " vr=", ind[3], ncol);
		else	printOpField(os, " pr=", play->GetPar(ind[3]), ncol);
		if( size_t(ind[4]) < 3 )
		{	for(i = 0; i < size_t(ind[4]); i++)
				printOpField(os, " ot=", ind[6+i], ncol);
		}
		else
		{	printOpField(os, "\n\tot=", ind[6+0], ncol);
			for(i = 1; i < size_t(ind[4]); i++)
				printOpField(os, " ot=", ind[6+i], ncol);
		}
		if( size_t(ind[5]) < 3 )
		{	for(i = 0; i < size_t(ind[5]); i++)
				printOpField(os, " of=", ind[6+ind[4]+i], ncol);
		}
		else
		{	printOpField(os, "\n\tof=", ind[6+ind[4]+0], ncol);
			{	for(i = 1; i < size_t(ind[5]); i++)
					printOpField(os, " of=", ind[6+ind[4]+i], ncol);
			}
		}
		break;

		case CSumOp:
		/*
		ind[0] = number of addition variables in summation
		ind[1] = number of subtraction variables in summation
		ind[2] = index of parameter that initializes summation
		ind[3], ... , ind[2+ind[0]] = index for positive variables
		ind[3+ind[0]], ..., ind[2+ind[0]+ind[1]] = negative variables
		ind[3+ind[0]+ind[1]] == ind[0] + ind[1]
		*/
		CPPAD_ASSERT_UNKNOWN( ind[3+ind[0]+ind[1]] == ind[0]+ind[1] );
		printOpField(os, " pr=", play->GetPar(ind[2]), ncol);
		for(i = 0; i < size_t(ind[0]); i++)
			 printOpField(os, " +v=", ind[3+i], ncol);
		for(i = 0; i < size_t(ind[1]); i++)
			 printOpField(os, " -v=", ind[3+ind[0]+i], ncol);
		break;

		case LdpOp:
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 3 );
		printOpField(os, "off=", ind[0], ncol);
		printOpField(os, "idx=", ind[1], ncol);
		break;

		case LdvOp:
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 3 );
		printOpField(os, "off=", ind[0], ncol);
		printOpField(os, "  v=", ind[1], ncol);
		break;

		case StppOp:
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 3 );
		printOpField(os, "off=", ind[0], ncol);
		printOpField(os, "idx=", ind[1], ncol);
		printOpField(os, " pr=", play->GetPar(ind[2]), ncol);
		break;

		case StpvOp:
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 3 );
		printOpField(os, "off=", ind[0], ncol);
		printOpField(os, "idx=", ind[1], ncol);
		printOpField(os, " vr=", ind[2], ncol);
		break;

		case StvpOp:
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 3 );
		printOpField(os, "off=", ind[0], ncol);
		printOpField(os, " vl=", ind[1], ncol);
		printOpField(os, " pr=", play->GetPar(ind[2]), ncol);
		break;

		case StvvOp:
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 3 );
		printOpField(os, "off=", ind[0], ncol);
		printOpField(os, " vl=", ind[1], ncol);
		printOpField(os, " vr=", ind[2], ncol);
		break;

		case AddvvOp:
		case DivvvOp:
		case EqvvOp:
		case LevvOp:
		case LtvvOp:
		case NevvOp:
		case MulvvOp:
		case PowvvOp:
		case SubvvOp:
		case ZmulvvOp:
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 2 );
		printOpField(os, " vl=", ind[0], ncol);
		printOpField(os, " vr=", ind[1], ncol);
		break;

		case AddpvOp:
		case EqpvOp:
		case DivpvOp:
		case LepvOp:
		case LtpvOp:
		case NepvOp:
		case SubpvOp:
		case MulpvOp:
		case PowpvOp:
		case ZmulpvOp:
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 2 );
		printOpField(os, " pl=", play->GetPar(ind[0]), ncol);
		printOpField(os, " vr=", ind[1], ncol);
		break;

		case DivvpOp:
		case LevpOp:
		case LtvpOp:
		case PowvpOp:
		case SubvpOp:
		case ZmulvpOp:
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 2 );
		printOpField(os, " vl=", ind[0], ncol);
		printOpField(os, " pr=", play->GetPar(ind[1]), ncol);
		break;

		case AbsOp:
		case AcosOp:
		case AcoshOp:
		case AsinOp:
		case AsinhOp:
		case AtanOp:
		case AtanhOp:
		case CosOp:
		case CoshOp:
		case ExpOp:
		case Expm1Op:
		case LogOp:
		case Log1pOp:
		case SignOp:
		case SinOp:
		case SinhOp:
		case SqrtOp:
		case UsravOp:
		case TanOp:
		case TanhOp:
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 1 );
		printOpField(os, "  v=", ind[0], ncol);
		break;

		case ErfOp:
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 3 );
		// ind[1] points to the parameter 0
		// ind[2] points to the parameter 2 / sqrt(pi)
		printOpField(os, "  v=", ind[0], ncol);
		break;

		case ParOp:
		case UsrapOp:
		case UsrrpOp:
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 1 );
		printOpField(os, "  p=", play->GetPar(ind[0]), ncol);
		break;

		case UserOp:
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 4 );
		{	std::string name =  atomic_base<Base>::class_name(ind[0]);
			printOpField(os, " f=",   name.c_str(), ncol);
			printOpField(os, " i=", ind[1], ncol);
			printOpField(os, " n=", ind[2], ncol);
			printOpField(os, " m=", ind[3], ncol);
		}
		break;

		case PriOp:
		CPPAD_ASSERT_NARG_NRES(op, 5, 0);
		if( ind[0] & 1 )
			printOpField(os, " v=", ind[1], ncol);
		else	printOpField(os, " p=", play->GetPar(ind[1]), ncol);
		os << "before=\"" << play->GetTxt(ind[2]) << "\"";
		if( ind[0] & 2 )
			printOpField(os, " v=", ind[3], ncol);
		else	printOpField(os, " p=", play->GetPar(ind[3]), ncol);
		os << "after=\"" << play->GetTxt(ind[4]) << "\"";
		break;

		case BeginOp:
		// argument not used (created by independent)
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 1 );
		break;

		case EndOp:
		case InvOp:
		case UsrrvOp:
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 0 );
		break;

		case DisOp:
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 2 );
		{	const char* name = discrete<Base>::name(ind[0]);
			printOpField(os, " f=", name, ncol);
			printOpField(os, " x=", ind[1], ncol);
		}
		break;


		case CExpOp:
		CPPAD_ASSERT_UNKNOWN(ind[1] != 0);
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 6 );
		if( ind[1] & 1 )
			printOpField(os, " vl=", ind[2], ncol);
		else	printOpField(os, " pl=", play->GetPar(ind[2]), ncol);
		if( ind[1] & 2 )
			printOpField(os, " vr=", ind[3], ncol);
		else	printOpField(os, " pr=", play->GetPar(ind[3]), ncol);
		if( ind[1] & 4 )
			printOpField(os, " vt=", ind[4], ncol);
		else	printOpField(os, " pt=", play->GetPar(ind[4]), ncol);
		if( ind[1] & 8 )
			printOpField(os, " vf=", ind[5], ncol);
		else	printOpField(os, " pf=", play->GetPar(ind[5]), ncol);
		break;

		default:
		CPPAD_ASSERT_UNKNOWN(0);
	}
}

/*!
Prints the result values correspnding to an operator.

\tparam Base
Is the base type for these AD< \a Base > operations.

\tparam Value
Determines the type of the values that we are printing.

\param os
is the output stream that the information is printed on.

\param nfz
is the number of forward sweep calculated values of type Value
that correspond to this operation
(ignored if NumRes(op) == 0).

\param fz
points to the first forward calculated value
that correspond to this operation
(ignored if NumRes(op) == 0).

\param nrz
is the number of reverse sweep calculated values of type Value
that correspond to this operation
(ignored if NumRes(op) == 0).

\param rz
points to the first reverse calculated value
that correspond to this operation
(ignored if NumRes(op) == 0).
*/
template <class Value>
void printOpResult(
	std::ostream          &os     ,
	size_t                 nfz    ,
	const  Value          *fz     ,
	size_t                 nrz    ,
	const  Value          *rz     )
{
	size_t k;
	for(k = 0; k < nfz; k++)
		os << "| fz[" << k << "]=" << fz[k];
	for(k = 0; k < nrz; k++)
		os << "| rz[" << k << "]=" << rz[k];
}

/*!
If NDEBUG is not defined, assert that arguments come before result.

\param op
Operator for which we are checking order.
All the operators are checked except for those of the form UserOp or Usr..Op.

\param result
is the variable index for the result.

\param arg
is a vector of lenght NumArg(op) pointing to the arguments
for this operation.
*/
inline void assert_arg_before_result(
	OpCode op, const addr_t* arg, size_t result
)
{
	switch( op )
	{

		// These cases are not included below
		case UserOp:
		case UsrapOp:
		case UsravOp:
		case UsrrpOp:
		case UsrrvOp:
		break;
		// ------------------------------------------------------------------

		// 0 arguments
		case CSkipOp:
		case CSumOp:
		case EndOp:
		case InvOp:
		break;
		// ------------------------------------------------------------------

		// 1 argument, but is not used
		case BeginOp:
		break;

		// 1 argument , 1 result
		case AbsOp:
		case ExpOp:
		case Expm1Op:
		case LogOp:
		case Log1pOp:
		case ParOp:
		case SignOp:
		case SqrtOp:
		CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < result );
		break;

		// 1 argument, 2 results
		case AcosOp:
		case AcoshOp:
		case AsinOp:
		case AsinhOp:
		case AtanOp:
		case AtanhOp:
		case CosOp:
		case CoshOp:
		case SinOp:
		case SinhOp:
		case TanOp:
		case TanhOp:
		CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) + 1 < result );
		break;

		// 1 argument, 5 results
		case ErfOp:
		CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) + 4 < result );
		break;
		// ------------------------------------------------------------------
		// 2 arguments, no results
		case LepvOp:
		case LtpvOp:
		case EqpvOp:
		case NepvOp:
		CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) <= result );
		break;
		//
		case LevpOp:
		case LtvpOp:
		CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) <= result );
		break;
		//
		case LevvOp:
		case LtvvOp:
		case EqvvOp:
		case NevvOp:
		CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) <= result );
		CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) <= result );
		break;

		// 2 arguments (both variables), 1 results
		case AddvvOp:
		case DivvvOp:
		case MulvvOp:
		case SubvvOp:
		case ZmulvvOp:
		CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < result );
		CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < result );
		break;

		// 2 arguments (first variables), 1 results
		case DivvpOp:
		case SubvpOp:
		case ZmulvpOp:
		CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < result );
		break;

		// 2 arguments (second variables), 1 results
		case AddpvOp:
		case DisOp:
		case DivpvOp:
		case MulpvOp:
		case SubpvOp:
		case ZmulpvOp:
		CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < result );
		break;

		// 2 arguments (both variables), 3 results
		case PowvvOp:
		CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) + 2 < result );
		CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) + 2 < result );
		break;

		// 2 arguments (first variable), 3 results
		case PowvpOp:
		CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) + 2 < result );
		break;

		// 2 arguments (second variable), 3 results
		case PowpvOp:
		CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) + 2 < result );
		break;
		// ------------------------------------------------------------------

		// 3 arguments, none variables
		case LdpOp:
		case StppOp:
		break;

		// 3 arguments, second variable, 1 result
		case LdvOp:
		CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < result );
		break;

		// 3 arguments, third variable, no result
		case StpvOp:
		CPPAD_ASSERT_UNKNOWN( size_t(arg[2]) <= result );
		break;

		// 3 arguments, second variable, no result
		case StvpOp:
		CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) <= result );
		break;

		// 3 arguments, second and third variable, no result
		case StvvOp:
		CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) <= result );
		CPPAD_ASSERT_UNKNOWN( size_t(arg[2]) <= result );
		break;
		// ------------------------------------------------------------------

		// 5 arguments, no result
		case PriOp:
		if( arg[0] & 1 )
		{	CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) <= result );
		}
		if( arg[0] & 2 )
		{	CPPAD_ASSERT_UNKNOWN( size_t(arg[3]) <= result );
		}
		break;
		// ------------------------------------------------------------------

		// 6 arguments, 1 result
		case CExpOp:
		if( arg[1] & 1 )
		{	CPPAD_ASSERT_UNKNOWN( size_t(arg[2]) < result );
		}
		if( arg[1] & 2 )
		{	CPPAD_ASSERT_UNKNOWN( size_t(arg[3]) < result );
		}
		if( arg[1] & 4 )
		{	CPPAD_ASSERT_UNKNOWN( size_t(arg[4]) < result );
		}
		if( arg[1] & 8 )
		{	CPPAD_ASSERT_UNKNOWN( size_t(arg[5]) < result );
		}
		break;
		// ------------------------------------------------------------------

		default:
		CPPAD_ASSERT_UNKNOWN(false);
		break;

	}
	return;
}

} } // END_CPPAD_LOCAL_NAMESPACE
# endif
