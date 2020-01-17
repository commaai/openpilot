# ifndef CPPAD_CORE_STD_MATH_98_HPP
# define CPPAD_CORE_STD_MATH_98_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
-------------------------------------------------------------------------------
$begin acos$$
$spell
	acos
$$

$section Inverse Sine Function: acos$$

$head Syntax$$
$icode%y% = acos(%x%)%$$

$head x, y$$
See the $cref/possible types/unary_standard_math/Possible Types/$$
for a unary standard math function.

$head Atomic$$
This is an $cref/atomic operation/glossary/Operation/Atomic/$$.

$head Derivative$$
$latex \[
\begin{array}{lcr}
	\R{acos}^{(1)} (x) & = & - (1 - x * x)^{-1/2}
\end{array}
\] $$

$head Example$$
$children%
	example/general/acos.cpp
%$$
The file
$cref acos.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$end
-------------------------------------------------------------------------------
$begin asin$$
$spell
	asin
$$

$section Inverse Sine Function: asin$$

$head Syntax$$
$icode%y% = asin(%x%)%$$

$head x, y$$
See the $cref/possible types/unary_standard_math/Possible Types/$$
for a unary standard math function.

$head Atomic$$
This is an $cref/atomic operation/glossary/Operation/Atomic/$$.

$head Derivative$$
$latex \[
\begin{array}{lcr}
	\R{asin}^{(1)} (x) & = & (1 - x * x)^{-1/2}
\end{array}
\] $$

$head Example$$
$children%
	example/general/asin.cpp
%$$
The file
$cref asin.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$end
-------------------------------------------------------------------------------
$begin atan$$
$spell
	atan
$$

$section Inverse Tangent Function: atan$$

$head Syntax$$
$icode%y% = atan(%x%)%$$

$head x, y$$
See the $cref/possible types/unary_standard_math/Possible Types/$$
for a unary standard math function.

$head Atomic$$
This is an $cref/atomic operation/glossary/Operation/Atomic/$$.

$head Derivative$$
$latex \[
\begin{array}{lcr}
	\R{atan}^{(1)} (x) & = & \frac{1}{1 + x^2}
\end{array}
\] $$

$head Example$$
$children%
	example/general/atan.cpp
%$$
The file
$cref atan.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$end
-------------------------------------------------------------------------------
$begin cos$$
$spell
	cos
$$

$section The Cosine Function: cos$$

$head Syntax$$
$icode%y% = cos(%x%)%$$

$head x, y$$
See the $cref/possible types/unary_standard_math/Possible Types/$$
for a unary standard math function.

$head Atomic$$
This is an $cref/atomic operation/glossary/Operation/Atomic/$$.

$head Derivative$$
$latex \[
\begin{array}{lcr}
	\R{cos}^{(1)} (x) & = & - \sin(x)
\end{array}
\] $$

$head Example$$
$children%
	example/general/cos.cpp
%$$
The file
$cref cos.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$end
-------------------------------------------------------------------------------
$begin cosh$$
$spell
	cosh
$$

$section The Hyperbolic Cosine Function: cosh$$

$head Syntax$$
$icode%y% = cosh(%x%)%$$

$head x, y$$
See the $cref/possible types/unary_standard_math/Possible Types/$$
for a unary standard math function.

$head Atomic$$
This is an $cref/atomic operation/glossary/Operation/Atomic/$$.

$head Derivative$$
$latex \[
\begin{array}{lcr}
	\R{cosh}^{(1)} (x) & = &  \sinh(x)
\end{array}
\] $$

$head Example$$
$children%
	example/general/cosh.cpp
%$$
The file
$cref cosh.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$end
-------------------------------------------------------------------------------
$begin exp$$
$spell
	exp
$$

$section The Exponential Function: exp$$

$head Syntax$$
$icode%y% = exp(%x%)%$$

$head x, y$$
See the $cref/possible types/unary_standard_math/Possible Types/$$
for a unary standard math function.

$head Atomic$$
This is an $cref/atomic operation/glossary/Operation/Atomic/$$.

$head Derivative$$
$latex \[
\begin{array}{lcr}
	\R{exp}^{(1)} (x) & = &  \exp(x)
\end{array}
\] $$

$head Example$$
$children%
	example/general/exp.cpp
%$$
The file
$cref exp.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$end
-------------------------------------------------------------------------------
$begin log$$
$spell
$$

$section The Exponential Function: log$$

$head Syntax$$
$icode%y% = log(%x%)%$$

$head x, y$$
See the $cref/possible types/unary_standard_math/Possible Types/$$
for a unary standard math function.

$head Atomic$$
This is an $cref/atomic operation/glossary/Operation/Atomic/$$.

$head Derivative$$
$latex \[
\begin{array}{lcr}
	\R{log}^{(1)} (x) & = &  \frac{1}{x}
\end{array}
\] $$

$head Example$$
$children%
	example/general/log.cpp
%$$
The file
$cref log.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$end
-------------------------------------------------------------------------------
$begin log10$$
$spell
	CppAD
$$

$section The Base 10 Logarithm Function: log10$$

$head Syntax$$
$icode%y% = log10(%x%)%$$

$head x, y$$
See the $cref/possible types/unary_standard_math/Possible Types/$$
for a unary standard math function.

$head Method$$
CppAD uses the representation
$latex \[
\begin{array}{lcr}
        {\rm log10} (x) & = & \log(x) / \log(10)
\end{array}
\] $$

$head Example$$
$children%
	example/general/log10.cpp
%$$
The file
$cref log10.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$end
-------------------------------------------------------------------------------
$begin sin$$
$spell
	sin
$$

$section The Sine Function: sin$$

$head Syntax$$
$icode%y% = sin(%x%)%$$

$head x, y$$
See the $cref/possible types/unary_standard_math/Possible Types/$$
for a unary standard math function.

$head Atomic$$
This is an $cref/atomic operation/glossary/Operation/Atomic/$$.

$head Derivative$$
$latex \[
\begin{array}{lcr}
	\R{sin}^{(1)} (x) & = &  \cos(x)
\end{array}
\] $$

$head Example$$
$children%
	example/general/sin.cpp
%$$
The file
$cref sin.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$end
-------------------------------------------------------------------------------
$begin sinh$$
$spell
	sinh
$$

$section The Hyperbolic Sine Function: sinh$$

$head Syntax$$
$icode%y% = sinh(%x%)%$$

$head x, y$$
See the $cref/possible types/unary_standard_math/Possible Types/$$
for a unary standard math function.

$head Atomic$$
This is an $cref/atomic operation/glossary/Operation/Atomic/$$.

$head Derivative$$
$latex \[
\begin{array}{lcr}
	\R{sinh}^{(1)} (x) & = &  \cosh(x)
\end{array}
\] $$

$head Example$$
$children%
	example/general/sinh.cpp
%$$
The file
$cref sinh.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$end
-------------------------------------------------------------------------------
$begin sqrt$$
$spell
	sqrt
$$

$section The Square Root Function: sqrt$$

$head Syntax$$
$icode%y% = sqrt(%x%)%$$

$head x, y$$
See the $cref/possible types/unary_standard_math/Possible Types/$$
for a unary standard math function.

$head Atomic$$
This is an $cref/atomic operation/glossary/Operation/Atomic/$$.

$head Derivative$$
$latex \[
\begin{array}{lcr}
	\R{sqrt}^{(1)} (x) & = &  \frac{1}{2 \R{sqrt} (x) }
\end{array}
\] $$

$head Example$$
$children%
	example/general/sqrt.cpp
%$$
The file
$cref sqrt.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$end
-------------------------------------------------------------------------------
$begin tan$$
$spell
	tan
$$

$section The Tangent Function: tan$$

$head Syntax$$
$icode%y% = tan(%x%)%$$

$head x, y$$
See the $cref/possible types/unary_standard_math/Possible Types/$$
for a unary standard math function.

$head Atomic$$
This is an $cref/atomic operation/glossary/Operation/Atomic/$$.

$head Derivative$$
$latex \[
\begin{array}{lcr}
	\R{tan}^{(1)} (x) & = &  1 + \tan (x)^2
\end{array}
\] $$

$head Example$$
$children%
	example/general/tan.cpp
%$$
The file
$cref tan.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$end
-------------------------------------------------------------------------------
$begin tanh$$
$spell
	tanh
$$

$section The Hyperbolic Tangent Function: tanh$$

$head Syntax$$
$icode%y% = tanh(%x%)%$$

$head x, y$$
See the $cref/possible types/unary_standard_math/Possible Types/$$
for a unary standard math function.

$head Atomic$$
This is an $cref/atomic operation/glossary/Operation/Atomic/$$.

$head Derivative$$
$latex \[
\begin{array}{lcr}
	\R{tanh}^{(1)} (x) & = &  1 - \tanh (x)^2
\end{array}
\] $$

$head Example$$
$children%
	example/general/tanh.cpp
%$$
The file
$cref tanh.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$end
-------------------------------------------------------------------------------
*/

/*!
\file std_math_98.hpp
Define AD<Base> standard math functions (using their Base versions)
*/

/*!
\def CPPAD_STANDARD_MATH_UNARY_AD(Name, Op)
Defines function Name with argument type AD<Base> and tape operation Op

The macro defines the function x.Name() where x has type AD<Base>.
It then uses this funciton to define Name(x) where x has type
AD<Base> or VecAD_reference<Base>.

If x is a variable, the tape unary operator Op is used
to record the operation and the result is identified as correspoding
to this operation; i.e., Name(x).taddr_ idendifies the operation and
Name(x).tape_id_ identifies the tape.

This macro is used to define AD<Base> versions of
acos, asin, atan, cos, cosh, exp, fabs, log, sin, sinh, sqrt, tan, tanh.
*/

# define CPPAD_STANDARD_MATH_UNARY_AD(Name, Op)                   \
    template <class Base>                                         \
    inline AD<Base> Name(const AD<Base> &x)                       \
    {   return x.Name##_me(); }                                   \
    template <class Base>                                         \
    inline AD<Base> AD<Base>::Name##_me (void) const              \
    {                                                             \
        AD<Base> result;                                          \
        result.value_ = CppAD::Name(value_);                      \
        CPPAD_ASSERT_UNKNOWN( Parameter(result) );                \
                                                                  \
        if( Variable(*this) )                                     \
        {   CPPAD_ASSERT_UNKNOWN( NumArg(Op) == 1 );              \
            local::ADTape<Base> *tape = tape_this();              \
            tape->Rec_.PutArg(taddr_);                            \
            result.taddr_ = tape->Rec_.PutOp(Op);                 \
            result.tape_id_    = tape->id_;                       \
        }                                                         \
        return result;                                            \
    }                                                             \
    template <class Base>                                         \
    inline AD<Base> Name(const VecAD_reference<Base> &x)          \
    {   return x.ADBase().Name##_me(); }

//  BEGIN CppAD namespace
namespace CppAD {

     CPPAD_STANDARD_MATH_UNARY_AD(acos, local::AcosOp)
     CPPAD_STANDARD_MATH_UNARY_AD(asin, local::AsinOp)
     CPPAD_STANDARD_MATH_UNARY_AD(atan, local::AtanOp)
     CPPAD_STANDARD_MATH_UNARY_AD(cos, local::CosOp)
     CPPAD_STANDARD_MATH_UNARY_AD(cosh, local::CoshOp)
     CPPAD_STANDARD_MATH_UNARY_AD(exp, local::ExpOp)
     CPPAD_STANDARD_MATH_UNARY_AD(fabs, local::AbsOp)
     CPPAD_STANDARD_MATH_UNARY_AD(log, local::LogOp)
     CPPAD_STANDARD_MATH_UNARY_AD(sin, local::SinOp)
     CPPAD_STANDARD_MATH_UNARY_AD(sinh, local::SinhOp)
     CPPAD_STANDARD_MATH_UNARY_AD(sqrt, local::SqrtOp)
     CPPAD_STANDARD_MATH_UNARY_AD(tan, local::TanOp)
     CPPAD_STANDARD_MATH_UNARY_AD(tanh, local::TanhOp)

# if CPPAD_USE_CPLUSPLUS_2011
     CPPAD_STANDARD_MATH_UNARY_AD(asinh, local::AsinhOp)
     CPPAD_STANDARD_MATH_UNARY_AD(acosh, local::AcoshOp)
     CPPAD_STANDARD_MATH_UNARY_AD(atanh, local::AtanhOp)
     CPPAD_STANDARD_MATH_UNARY_AD(expm1, local::Expm1Op)
     CPPAD_STANDARD_MATH_UNARY_AD(log1p, local::Log1pOp)
# endif

# if CPPAD_USE_CPLUSPLUS_2011
	// Error function is a special case
	template <class Base>
	inline AD<Base> erf(const AD<Base> &x)
	{	return x.erf_me(); }
	template <class Base>
	inline AD<Base> AD<Base>::erf_me (void) const
	{
		AD<Base> result;
		result.value_ = CppAD::erf(value_);
		CPPAD_ASSERT_UNKNOWN( Parameter(result) );

		if( Variable(*this) )
		{	CPPAD_ASSERT_UNKNOWN( local::NumArg(local::ErfOp) == 3 );
			local::ADTape<Base> *tape = tape_this();
			// arg[0] = argument to erf function
			tape->Rec_.PutArg(taddr_);
			// arg[1] = zero
			addr_t p  = tape->Rec_.PutPar( Base(0.0) );
			tape->Rec_.PutArg(p);
			// arg[2] = 2 / sqrt(pi)
			p = tape->Rec_.PutPar(Base(
				1.0 / std::sqrt( std::atan(1.0) )
			));
			tape->Rec_.PutArg(p);
			//
			result.taddr_ = tape->Rec_.PutOp(local::ErfOp);
			result.tape_id_    = tape->id_;
		}
		return result;
	}
	template <class Base>
	inline AD<Base> erf(const VecAD_reference<Base> &x)
	{	return x.ADBase().erf_me(); }
# endif

     /*!
	Compute the log of base 10 of x where  has type AD<Base>

	\tparam Base
	is the base type (different from base for log)
	for this AD type, see base_require.

	\param x
	is the argument for the log10 function.

	\result
	if the result is y, then \f$ x = 10^y \f$.
	*/
     template <class Base>
     inline AD<Base> log10(const AD<Base> &x)
     {    return CppAD::log(x) / CppAD::log( Base(10) ); }
     template <class Base>
     inline AD<Base> log10(const VecAD_reference<Base> &x)
     {    return CppAD::log(x.ADBase()) / CppAD::log( Base(10) ); }
}

# undef CPPAD_STANDARD_MATH_UNARY_AD

# endif
