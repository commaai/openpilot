# ifndef CPPAD_CORE_ABS_NORMAL_FUN_HPP
# define CPPAD_CORE_ABS_NORMAL_FUN_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin abs_normal_fun$$
$spell
	const
$$


$section Create An Abs-normal Representation of a Function$$

$head Syntax$$
$icode%f%.abs_normal_fun(%g%, %a%)%$$

$head f$$
The object $icode f$$ has prototype
$codei%
	ADFun<%Base%>& %f%
%$$
It represents a function $latex f : \B{R}^n \rightarrow \B{R}^m$$.
We assume that the only non-smooth terms in the representation are
absolute value functions and use $latex s \in \B{Z}_+$$
to represent the number of these terms.
It is effectively $code const$$, except that some internal state
that is not relevant to the user; see
$cref/const ADFun/wish_list/const ADFun/$$.

$subhead n$$
We use $icode n$$ to denote the dimension of the domain space for $icode f$$.

$subhead m$$
We use $icode m$$ to denote the dimension of the range space for $icode f$$.

$subhead s$$
We use $icode s$$ to denote the number of absolute value terms in $icode f$$.


$head a$$
The object $icode a$$ has prototype
$codei%
	ADFun<%Base%> %a%
%$$
The initial function representation in $icode a$$ is lost.
Upon return it represents the result of the absolute terms
$latex a : \B{R}^n \rightarrow \B{R}^s$$; see $latex a(x)$$ defined below.
Note that $icode a$$ is constructed by copying $icode f$$
and then changing the dependent variables. There may
be many calculations in this representation that are not necessary
and can be removed using
$codei%
	%a%.optimize()
%$$
This optimization is not done automatically by $code abs_normal_fun$$
because it may take a significant amount of time.

$subhead zeta$$
Let $latex \zeta_0 ( x )$$
denote the argument for the first absolute value term in $latex f(x)$$,
$latex \zeta_1 ( x , |\zeta_0 (x)| )$$ for the second term, and so on.

$subhead a(x)$$
For $latex i = 0 , \ldots , {s-1}$$ define
$latex \[
a_i (x)
=
| \zeta_i ( x , a_0 (x) , \ldots , a_{i-1} (x ) ) |
\] $$
This defines $latex a : \B{R}^n \rightarrow \B{R}^s$$.

$head g$$
The object $icode g$$ has prototype
$codei%
	ADFun<%Base%> %g%
%$$
The initial function representation in $icode g$$ is lost.
Upon return it represents the smooth function
$latex g : \B{R}^{n + s} \rightarrow  \B{R}^{m + s}$$ is defined by
$latex \[
g( x , u )
=
\left[ \begin{array}{c} y(x, u) \\ z(x, u) \end{array} \right]
\] $$
were $latex y(x, u)$$ and $latex z(x, u)$$ are defined below.

$subhead z(x, u)$$
Define the smooth function
$latex z : \B{R}^{n + s} \rightarrow  \B{R}^s$$ by
$latex \[
z_i ( x , u ) = \zeta_i ( x , u_0 , \ldots , u_{i-1} )
\] $$
Note that the partial of $latex z_i$$ with respect to $latex u_j$$ is zero
for $latex j \geq i$$.

$subhead y(x, u)$$
There is a smooth function
$latex y : \B{R}^{n + s} \rightarrow  \B{R}^m$$
such that $latex y( x , u ) = f(x)$$ whenever $latex u = a(x)$$.

$head Affine Approximation$$
We define the affine approximations
$latex \[
\begin{array}{rcl}
y[ \hat{x} ]( x , u )
& = &
y ( \hat{x}, a( \hat{x} ) )
	+ \partial_x y ( \hat{x}, a( \hat{x} ) ) ( x - \hat{x} )
	+ \partial_u y ( \hat{x}, a( \hat{x} ) ) ( u - a( \hat{x} ) )
\\
z[ \hat{x} ]( x , u )
& = &
z ( \hat{x}, a( \hat{x} ) )
	+ \partial_x z ( \hat{x}, a( \hat{x} ) ) ( x - \hat{x} )
	+ \partial_u z ( \hat{x}, a( \hat{x} ) ) ( u - a( \hat{x} ) )
\end{array}
\] $$
It follows that
$latex \[
\begin{array}{rcl}
y( x , u )
& = &
y[ \hat{x} ]( x , u ) + o ( x - \hat{x}, u - a( \hat{x} ) )
\\
z( x , u )
& = &
z[ \hat{x} ]( x , u ) + o ( x - \hat{x}, u - a( \hat{x} ) )
\end{array}
\] $$

$head Abs-normal Approximation$$

$subhead Approximating a(x)$$
The function $latex a(x)$$ is not smooth, but it is equal to
$latex | z(x, u) |$$ when $latex u = a(x)$$.
Furthermore
$latex \[
z[ \hat{x} ]( x , u )
=
z ( \hat{x}, a( \hat{x} ) )
	+ \partial_x z ( \hat{x}, a( \hat{x} ) ) ( x - \hat{x} )
	+ \partial_u z ( \hat{x}, a( \hat{x} ) ) ( u - a( \hat{x} ) )
\] $$
The partial of $latex z_i$$ with respect to $latex u_j$$ is zero
for $latex j \geq i$$. It follows that
$latex \[
z_i [ \hat{x} ]( x , u )
=
z_i ( \hat{x}, a( \hat{x} ) )
	+ \partial_x z_i ( \hat{x}, a( \hat{x} ) ) ( x - \hat{x} )
	+ \sum_{j < i} \partial_{u(j)}
		z_i ( \hat{x}, a( \hat{x} ) ) ( u_j - a_j ( \hat{x} ) )
\] $$
Considering the case $latex i = 0$$ we define
$latex \[
a_0 [ \hat{x} ]( x )
=
| z_0 [ \hat{x} ]( x , u ) |
=
\left|
	z_0 ( \hat{x}, a( \hat{x} ) )
	+ \partial_x z_0 ( \hat{x}, a( \hat{x} ) ) ( x - \hat{x} )
\right|
\] $$
It follows that
$latex \[
	a_0 (x) = a_0 [ \hat{x} ]( x ) + o ( x - \hat{x} )
\] $$
In general, we define $latex a_i [ \hat{x} ]$$ using
$latex a_j [ \hat{x} ]$$ for $latex j < i$$ as follows:
$latex \[
a_i [ \hat{x} ]( x )
=
\left |
	z_i ( \hat{x}, a( \hat{x} ) )
	+ \partial_x z_i ( \hat{x}, a( \hat{x} ) ) ( x - \hat{x} )
	+ \sum_{j < i} \partial_{u(j)}
		z_i ( \hat{x}, a( \hat{x} ) )
			( a_j [ \hat{x} ] ( x )  - a_j ( \hat{x} ) )
\right|
\] $$
It follows that
$latex \[
	a (x) = a[ \hat{x} ]( x ) + o ( x - \hat{x} )
\] $$
Note that in the case where $latex z(x, u)$$ and $latex y(x, u)$$ are
affine,
$latex \[
	a[ \hat{x} ]( x ) = a( x )
\] $$


$subhead Approximating f(x)$$
$latex \[
f(x)
=
y ( x , a(x ) )
=
y [ \hat{x} ] ( x , a[ \hat{x} ] ( x ) )
+ o( \Delta x )
\] $$

$head Correspondence to Literature$$
Using the notation
$latex Z = \partial_x z(\hat{x}, \hat{u})$$,
$latex L = \partial_u z(\hat{x}, \hat{u})$$,
$latex J = \partial_x y(\hat{x}, \hat{u})$$,
$latex Y = \partial_u y(\hat{x}, \hat{u})$$,
the approximation for $latex z$$ and $latex y$$ are
$latex \[
\begin{array}{rcl}
z[ \hat{x} ]( x , u )
& = &
z ( \hat{x}, a( \hat{x} ) ) + Z ( x - \hat{x} ) + L ( u - a( \hat{x} ) )
\\
y[ \hat{x} ]( x , u )
& = &
y ( \hat{x}, a( \hat{x} ) ) + J ( x - \hat{x} ) + Y ( u - a( \hat{x} ) )
\end{array}
\] $$
Moving the terms with $latex \hat{x}$$ together, we have
$latex \[
\begin{array}{rcl}
z[ \hat{x} ]( x , u )
& = &
z ( \hat{x}, a( \hat{x} ) ) - Z \hat{x} - L a( \hat{x} )  + Z x + L u
\\
y[ \hat{x} ]( x , u )
& = &
y ( \hat{x}, a( \hat{x} ) ) - J \hat{x} - Y a( \hat{x} )  + J x + Y u
\end{array}
\] $$
Using the notation
$latex c = z ( \hat{x}, \hat{u} ) - Z \hat{x} - L \hat{u}$$,
$latex b = y ( \hat{x}, \hat{u} ) - J \hat{x} - Y \hat{u}$$,
we have
$latex \[
\begin{array}{rcl}
z[ \hat{x} ]( x , u ) & = & c + Z x + L u
\\
y[ \hat{x} ]( x , u ) & = & b + J x + Y u
\end{array}
\] $$
Considering the affine case, where the approximations are exact,
and choosing $latex u = a(x) = |z(x, u)|$$, we obtain
$latex \[
\begin{array}{rcl}
z( x , a(x ) ) & = & c + Z x + L |z( x , a(x ) )|
\\
y( x , a(x ) ) & = & b + J x + Y |z( x , a(x ) )|
\end{array}
\] $$
This is Equation (2) of the
$cref/reference/abs_normal/Reference/$$.

$children%example/abs_normal/get_started.cpp
%$$
$head Example$$
The file $cref abs_get_started.cpp$$ contains
an example and test using this operation.

$end
-------------------------------------------------------------------------------
*/
/*!
file abs_normal_fun.hpp
Create an abs-normal representation of a function
*/

namespace CppAD { // BEGIN_CPPAD_NAMESPACE
/*!
Create an abs-normal representation of an ADFun object.

\tparam Base
base type for this abs-normal form and for the function beging represented;
i.e., f.

\param f
is the function that this object will represent in abs-normal form.
This is effectively const except that the play back state play_
is used.
*/

# define NOT_YET_COMPILING 0

template <class Base>
void ADFun<Base>::abs_normal_fun(ADFun<Base>& g, ADFun<Base>& a)
{	using namespace local;

	// -----------------------------------------------------------------------
	// Forward sweep to determine number of absolute value operations in f
	// -----------------------------------------------------------------------
	// The argument and result index in f for each absolute value operator
	CppAD::vector<addr_t> f_abs_arg;
	CppAD::vector<size_t> f_abs_res;
	//
	OpCode        op;                 // this operator
	const addr_t* arg = CPPAD_NULL;   // arguments for this operator
	size_t        i_op;               // index of this operator
	size_t        i_var;              // variable index for this operator
	play_.forward_start(op, arg, i_op, i_var);
	CPPAD_ASSERT_UNKNOWN( op == BeginOp );
	//
	bool    more_operators = true;
	while( more_operators )
	{
		// next op
		play_.forward_next(op, arg, i_op, i_var);
		switch( op )
		{	// absolute value operator
			case AbsOp:
			CPPAD_ASSERT_NARG_NRES(op, 1, 1);
			f_abs_arg.push_back( arg[0] );
			f_abs_res.push_back( i_var );
			break;

			case CSumOp:
			// CSumOp has a variable number of arguments
			play_.forward_csum(op, arg, i_op, i_var);
			break;

			case CSkipOp:
			// CSkip has a variable number of arguments
			play_.forward_cskip(op, arg, i_op, i_var);
			break;

			case EndOp:
			more_operators = false;
			break;

			default:
			break;
		}
	}
	// ------------------------------------------------------------------------
	// Forward sweep to create new recording
	// ------------------------------------------------------------------------
	// recorder for new operation sequence
	recorder<Base> rec;
	//
	// number of variables in both operation sequences
	// (the AbsOp operators are replace by InvOp operators)
	const size_t num_var = play_.num_var_rec();
	//
	// mapping from old variable index to new variable index
	CPPAD_ASSERT_UNKNOWN(
		std::numeric_limits<addr_t>::max() >= num_var
	);
	CppAD::vector<addr_t> f2g_var(num_var);
	for(i_var = 0; i_var < num_var; i_var++)
		f2g_var[i_var] = addr_t( num_var ); // invalid (should not be used)
	//
	// record the independent variables in f
	play_.forward_start(op, arg, i_op, i_var);
	CPPAD_ASSERT_UNKNOWN( op == BeginOp );
	more_operators   = true;
	while( more_operators )
	{	switch( op )
		{
			// phantom variable
			case BeginOp:
			CPPAD_ASSERT_NARG_NRES(op, 1, 1);
			CPPAD_ASSERT_UNKNOWN( arg[0] == 0 );
			rec.PutArg(0);
			f2g_var[i_var] = rec.PutOp(op);
			break;

			// independent variables
			case InvOp:
			CPPAD_ASSERT_NARG_NRES(op, 0, 1);
			f2g_var[i_var] = rec.PutOp(op);
			break;

			// end of independent variables
			default:
			more_operators = false;
			break;
		}
		if( more_operators )
			play_.forward_next(op, arg, i_op, i_var);
	}
	// add one for the phantom variable
	CPPAD_ASSERT_UNKNOWN( 1 + Domain() == i_var );
	//
	// record the independent variables corresponding AbsOp results
	size_t index_abs;
	for(index_abs = 0; index_abs < f_abs_res.size(); index_abs++)
		f2g_var[ f_abs_res[index_abs] ] = rec.PutOp(InvOp);
	//
	// used to hold new argument vector
	addr_t new_arg[6];
	//
	// Parameters in recording of f
	const Base* f_parameter = play_.GetPar();
	//
	// now loop through the rest of the
	more_operators = true;
	index_abs      = 0;
	while( more_operators )
	{	addr_t mask; // temporary used in some switch cases
		switch( op )
		{
			// check setting of f_abs_arg and f_abs_res;
			case AbsOp:
			CPPAD_ASSERT_NARG_NRES(op, 1, 1);
			CPPAD_ASSERT_UNKNOWN( f_abs_arg[index_abs] ==  arg[0] );
			CPPAD_ASSERT_UNKNOWN( f_abs_res[index_abs] ==  i_var );
			CPPAD_ASSERT_UNKNOWN( f2g_var[i_var] > 0 );
			++index_abs;
			break;

			// These operators come at beginning of take and are handled above
			case InvOp:
			CPPAD_ASSERT_UNKNOWN(false);
			break;

			// ---------------------------------------------------------------
			// Unary operators, argument a parameter, one result
			case ParOp:
			CPPAD_ASSERT_NARG_NRES(op, 1, 1);
			new_arg[0] = rec.PutPar( f_parameter[ arg[0] ] );
			rec.PutArg( new_arg[0] );
			f2g_var[i_var] = rec.PutOp(op);
			break;

			// --------------------------------------------------------------
			// Unary operators, argument a variable, one result
			// (excluding the absolute value operator AbsOp)
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
			case TanOp:
			case TanhOp:
			// some of these operators have an auxillary result; e.g.,
			// sine and cosine are computed togeather.
			CPPAD_ASSERT_UNKNOWN( NumArg(op) ==  1 );
			CPPAD_ASSERT_UNKNOWN( NumRes(op) == 1 || NumRes(op) == 2 );
			CPPAD_ASSERT_UNKNOWN( size_t( f2g_var[ arg[0] ] ) < num_var );
			new_arg[0] = f2g_var[ arg[0] ];
			rec.PutArg( new_arg[0] );
			f2g_var[i_var] = rec.PutOp( op );
			break;

			case ErfOp:
			CPPAD_ASSERT_NARG_NRES(op, 3, 5);
			CPPAD_ASSERT_UNKNOWN( size_t( f2g_var[ arg[0] ] ) < num_var );
			// Error function is a special case
			// second argument is always the parameter 0
			// third argument is always the parameter 2 / sqrt(pi)
			rec.PutArg( rec.PutPar( Base(0.0) ) );
			rec.PutArg( rec.PutPar(
				Base( 1.0 / std::sqrt( std::atan(1.0) ) )
			) );
			f2g_var[i_var] = rec.PutOp(op);
			break;
			// --------------------------------------------------------------
			// Binary operators, left variable, right parameter, one result
			case SubvpOp:
			case DivvpOp:
			case PowvpOp:
			case ZmulvpOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1);
			CPPAD_ASSERT_UNKNOWN( size_t( f2g_var[ arg[0] ] ) < num_var );
			new_arg[0] = f2g_var[ arg[0] ];
			new_arg[1] = rec.PutPar( f_parameter[ arg[1] ] );
			rec.PutArg( new_arg[0], new_arg[1] );
			f2g_var[i_var] = rec.PutOp(op);
			break;
			// ---------------------------------------------------
			// Binary operators, left index, right variable, one result
			case DisOp:
			CPPAD_ASSERT_UNKNOWN( size_t( f2g_var[ arg[1] ] ) < num_var );
			new_arg[0] = arg[0];
			new_arg[1] = f2g_var[ arg[1] ];
			rec.PutArg( new_arg[0], new_arg[1] );
			f2g_var[i_var] = rec.PutOp(op);
			break;

			// --------------------------------------------------------------
			// Binary operators, left parameter, right variable, one result
			case AddpvOp:
			case SubpvOp:
			case MulpvOp:
			case DivpvOp:
			case PowpvOp:
			case ZmulpvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1);
			CPPAD_ASSERT_UNKNOWN( size_t( f2g_var[ arg[1] ] ) < num_var );
			new_arg[0] = rec.PutPar( f_parameter[ arg[0] ] );
			new_arg[1] = f2g_var[ arg[1] ];
			rec.PutArg( new_arg[0], new_arg[1] );
			f2g_var[i_var] = rec.PutOp(op);
			break;
			// --------------------------------------------------------------
			// Binary operators, left and right variables, one result
			case AddvvOp:
			case SubvvOp:
			case MulvvOp:
			case DivvvOp:
			case ZmulvvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 1);
			CPPAD_ASSERT_UNKNOWN( size_t( f2g_var[ arg[0] ] ) < num_var );
			CPPAD_ASSERT_UNKNOWN( size_t( f2g_var[ arg[1] ] ) < num_var );
			new_arg[0] = f2g_var[ arg[0] ];
			new_arg[1] = f2g_var[ arg[1] ];
			rec.PutArg( new_arg[0], new_arg[1] );
			f2g_var[i_var] = rec.PutOp(op);
			break;
			// ---------------------------------------------------
			// Conditional expression operators
			case CExpOp:
			CPPAD_ASSERT_NARG_NRES(op, 6, 1);
			new_arg[0] = arg[0];
			new_arg[1] = arg[1];
			mask = 1;
			for(size_t i = 2; i < 6; i++)
			{	if( arg[1] & mask )
				{	CPPAD_ASSERT_UNKNOWN( size_t(f2g_var[arg[i]]) < num_var );
					new_arg[i] = f2g_var[ arg[i] ];
				}
				else
					new_arg[i] = rec.PutPar( f_parameter[ arg[i] ] );
				mask = mask << 1;
			}
			rec.PutArg(
				new_arg[0] ,
				new_arg[1] ,
				new_arg[2] ,
				new_arg[3] ,
				new_arg[4] ,
				new_arg[5]
			);
			f2g_var[i_var] = rec.PutOp(op);
			break;

			// --------------------------------------------------
			// Operators with no arguments and no results
			case EndOp:
			CPPAD_ASSERT_NARG_NRES(op, 0, 0);
			rec.PutOp(op);
			more_operators = false;
			break;

			// ---------------------------------------------------
			// Operations with two arguments and no results
			case LepvOp:
			case LtpvOp:
			case EqpvOp:
			case NepvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 0);
			new_arg[0] = rec.PutPar( f_parameter[ arg[0] ] );
			new_arg[1] = f2g_var[ arg[1] ];
			rec.PutArg(new_arg[0], new_arg[1]);
			rec.PutOp(op);
			break;
			//
			case LevpOp:
			case LtvpOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 0);
			new_arg[0] = f2g_var[ arg[0] ];
			new_arg[1] = rec.PutPar( f_parameter[ arg[1] ] );
			rec.PutArg(new_arg[0], new_arg[1]);
			rec.PutOp(op);
			break;
			//
			case LevvOp:
			case LtvvOp:
			case EqvvOp:
			case NevvOp:
			CPPAD_ASSERT_NARG_NRES(op, 2, 0);
			new_arg[0] = f2g_var[ arg[0] ];
			new_arg[1] = f2g_var[ arg[1] ];
			rec.PutArg(new_arg[0], new_arg[1]);
			rec.PutOp(op);
			break;

			// ---------------------------------------------------
			// print forward operator
			case PriOp:
			CPPAD_ASSERT_NARG_NRES(op, 5, 0);
			//
			// arg[0]
			new_arg[0] = arg[0];
			//
			// arg[1]
			if( arg[0] & 1 )
			{
				CPPAD_ASSERT_UNKNOWN( size_t( f2g_var[ arg[1] ] )  < num_var );
				new_arg[1] = f2g_var[ arg[1] ];
			}
			else
			{	new_arg[1] = rec.PutPar( f_parameter[ arg[1] ] );
			}
			//
			// arg[3]
			if( arg[0] & 2 )
			{
				CPPAD_ASSERT_UNKNOWN( size_t( f2g_var[ arg[3] ] )  < num_var );
				new_arg[3] = f2g_var[ arg[3] ];
			}
			else
			{	new_arg[3] = rec.PutPar( f_parameter[ arg[3] ] );
			}
			new_arg[2] = rec.PutTxt( play_.GetTxt( arg[2] ) );
			new_arg[4] = rec.PutTxt( play_.GetTxt( arg[4] ) );
			//
			rec.PutArg(
				new_arg[0] ,
				new_arg[1] ,
				new_arg[2] ,
				new_arg[3] ,
				new_arg[4]
			);
			// no result
			rec.PutOp(op);
			break;

			// ---------------------------------------------------
			// VecAD operators

			// Load using a parameter index
			case LdpOp:
			CPPAD_ASSERT_NARG_NRES(op, 3, 1);
			new_arg[0] = arg[0];
			new_arg[1] = arg[1];
			new_arg[2] = arg[2];
			rec.PutArg(
				new_arg[0],
				new_arg[1],
				new_arg[2]
			);
			f2g_var[i_var] = rec.PutLoadOp(op);
			break;

			// Load using a variable index
			case LdvOp:
			CPPAD_ASSERT_NARG_NRES(op, 3, 1);
			CPPAD_ASSERT_UNKNOWN( size_t( f2g_var[ arg[1] ] ) < num_var );
			new_arg[0] = arg[0];
			new_arg[1] = f2g_var[ arg[1] ];
			new_arg[2] = arg[2];
			rec.PutArg(
				new_arg[0],
				new_arg[1],
				new_arg[2]
			);
			f2g_var[i_var] = rec.PutLoadOp(op);
			break;

			// Store a parameter using a parameter index
			case StppOp:
			CPPAD_ASSERT_NARG_NRES(op, 3, 0);
			new_arg[0] = arg[0];
			new_arg[1] = rec.PutPar( f_parameter[ arg[1] ] );
			new_arg[2] = rec.PutPar( f_parameter[ arg[2] ] );
			rec.PutArg(
				new_arg[0],
				new_arg[1],
				new_arg[2]
			);
			rec.PutOp(op);
			break;

			// Store a parameter using a variable index
			case StvpOp:
			CPPAD_ASSERT_NARG_NRES(op, 3, 0);
			CPPAD_ASSERT_UNKNOWN( size_t( f2g_var[ arg[1] ] ) < num_var );
			new_arg[0] = arg[0];
			new_arg[1] = f2g_var[ arg[1] ];
			new_arg[2] = rec.PutPar( f_parameter[ arg[2] ] );
			rec.PutArg(
				new_arg[0],
				new_arg[1],
				new_arg[2]
			);
			rec.PutOp(op);
			break;

			// Store a variable using a parameter index
			case StpvOp:
			CPPAD_ASSERT_NARG_NRES(op, 3, 0);
			CPPAD_ASSERT_UNKNOWN( size_t( f2g_var[ arg[2] ] ) < num_var );
			new_arg[0] = arg[0];
			new_arg[1] = rec.PutPar( f_parameter[ arg[1] ] );
			new_arg[2] = f2g_var[ arg[2] ];
			rec.PutArg(
				new_arg[0],
				new_arg[1],
				new_arg[2]
			);
			rec.PutOp(op);
			break;

			// Store a variable using a variable index
			case StvvOp:
			CPPAD_ASSERT_NARG_NRES(op, 3, 0);
			CPPAD_ASSERT_UNKNOWN( size_t( f2g_var[ arg[1] ] ) < num_var );
			CPPAD_ASSERT_UNKNOWN( size_t( f2g_var[ arg[2] ] ) < num_var );
			new_arg[0] = arg[0];
			new_arg[1] = f2g_var[ arg[1] ];
			new_arg[2] = f2g_var[ arg[2] ];
			rec.PutArg(
				new_arg[0],
				new_arg[1],
				new_arg[2]
			);
			break;

			// -----------------------------------------------------------
			// user atomic function call operators

			case UserOp:
			CPPAD_ASSERT_NARG_NRES(op, 4, 0);
			// atomic_index, user_old, user_n, user_m
			rec.PutArg(arg[0], arg[1], arg[2], arg[3]);
			rec.PutOp(UserOp);
			break;

			case UsrapOp:
			CPPAD_ASSERT_NARG_NRES(op, 1, 0);
			new_arg[0] = rec.PutPar( f_parameter[ arg[0] ] );
			rec.PutArg(new_arg[0]);
			rec.PutOp(UsrapOp);
			break;

			case UsravOp:
			CPPAD_ASSERT_NARG_NRES(op, 1, 0);
			CPPAD_ASSERT_UNKNOWN( size_t( f2g_var[arg[0]] ) < num_var );
			new_arg[0] = f2g_var[ arg[0] ];
			rec.PutArg(new_arg[0]);
			rec.PutOp(UsravOp);
			break;

			case UsrrpOp:
			CPPAD_ASSERT_NARG_NRES(op, 1, 0);
			new_arg[0] = rec.PutPar( f_parameter[ arg[0] ] );
			rec.PutArg(new_arg[0]);
			rec.PutOp(UsrrpOp);
			break;

			case UsrrvOp:
			CPPAD_ASSERT_NARG_NRES(op, 0, 1);
			f2g_var[i_var] = rec.PutOp(UsrrvOp);
			break;
			// ---------------------------------------------------

			// all cases should be handled above
			default:
			CPPAD_ASSERT_UNKNOWN(false);
		}
		if( more_operators )
			play_.forward_next(op, arg, i_op, i_var);
	}
	// Check a few expected results
	CPPAD_ASSERT_UNKNOWN( rec.num_op_rec() == play_.num_op_rec() );
	CPPAD_ASSERT_UNKNOWN( rec.num_var_rec() == play_.num_var_rec() );
	CPPAD_ASSERT_UNKNOWN( rec.num_load_op_rec() == play_.num_load_op_rec() );

	// -----------------------------------------------------------------------
	// Use rec to create the function g
	// -----------------------------------------------------------------------

	// number of variables in the recording
	g.num_var_tape_ = rec.num_var_rec();

	// dimension cskip_op vector to number of operators
	g.cskip_op_.erase();
	g.cskip_op_.extend( rec.num_op_rec() );

	// independent variables in g: (x, u)
	size_t s = f_abs_res.size();
	size_t n = Domain();
	g.ind_taddr_.resize(n + s);
	// (x, u)
	for(size_t j = 0; j < n; j++)
	{	g.ind_taddr_[j] = f2g_var[ ind_taddr_[j] ];
		CPPAD_ASSERT_UNKNOWN( g.ind_taddr_[j] == j + 1 );
	}
	for(size_t j = 0; j < s; j++)
	{	g.ind_taddr_[n + j] = f2g_var[ f_abs_res[j] ];
		CPPAD_ASSERT_UNKNOWN( g.ind_taddr_[n + j] == n + j + 1 );
	}

	// dependent variable in g: (y, z)
	CPPAD_ASSERT_UNKNOWN( s == f_abs_arg.size() );
	size_t m = Range();
	g.dep_taddr_.resize(m + s);
	for(size_t i = 0; i < m; i++)
	{	g.dep_taddr_[i] = f2g_var[ dep_taddr_[i] ];
		CPPAD_ASSERT_UNKNOWN( g.dep_taddr_[i] < num_var );
	}
	for(size_t i = 0; i < s; i++)
	{	g.dep_taddr_[m + i] = f2g_var[ f_abs_arg[i] ];
		CPPAD_ASSERT_UNKNOWN( g.dep_taddr_[m + i] < num_var );
	}

	// which  dependent variables are parameters
	g.dep_parameter_.resize(m + s);
	for(size_t i = 0; i < m; i++)
		g.dep_parameter_[i] = dep_parameter_[i];
	for(size_t i = 0; i < s; i++)
		g.dep_parameter_[m + i] = false;

	// free memory allocated for sparse Jacobian calculation
	// (the resutls are no longer valid)
	g.for_jac_sparse_pack_.resize(0, 0);
	g.for_jac_sparse_set_.resize(0, 0);

	// free taylor coefficient memory
	g.taylor_.free();
	g.num_order_taylor_ = 0;
	g.cap_order_taylor_ = 0;

	// Transferring the recording swaps its vectors so do this last
	// replace the recording in g (this ADFun object)
	g.play_.get(rec);

	// ------------------------------------------------------------------------
	// Create the function a
	// ------------------------------------------------------------------------

	// start with a copy of f
	a = *this;

	// dependent variables in a(x)
	CPPAD_ASSERT_UNKNOWN( s == f_abs_arg.size() );
	a.dep_taddr_.resize(s);
	for(size_t i = 0; i < s; i++)
	{	a.dep_taddr_[i] = f_abs_res[i];
		CPPAD_ASSERT_UNKNOWN( a.dep_taddr_[i] < num_var );
	}

	// free memory allocated for sparse Jacobian calculation
	// (the resutls are no longer valid)
	a.for_jac_sparse_pack_.resize(0, 0);
	a.for_jac_sparse_set_.resize(0, 0);

	// free taylor coefficient memory
	a.taylor_.free();
	a.num_order_taylor_ = 0;
	a.cap_order_taylor_ = 0;
}

} // END_CPPAD_NAMESPACE

# endif
