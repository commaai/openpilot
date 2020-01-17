# ifndef CPPAD_LOCAL_HASH_CODE_HPP
# define CPPAD_LOCAL_HASH_CODE_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
# include <cppad/core/base_hash.hpp>
/*!
\file local/hash_code.hpp
CppAD hashing utility.
*/


namespace CppAD { namespace local { // BEGIN_CPPAD_LOCAL_NAMESPACE
/*!
General purpose hash code for an arbitrary value.

\tparam Value
is the type of the argument being hash coded.
It should be a plain old data class; i.e.,
the values included in the equality operator in the object and
not pointed to by the object.

\param value
the value that we are generating a hash code for.
All of the fields in value should have been set before the hash code
is computed (otherwise undefined values are used).

\return
is a hash code that is between zero and CPPAD_HASH_TABLE_SIZE - 1.

\par Checked Assertions
\li \c std::numeric_limits<unsigned short>::max() >= CPPAD_HASH_TABLE_SIZE
\li \c sizeof(value) is even
\li \c sizeof(unsigned short)  == 2
*/
template <class Value>
unsigned short local_hash_code(const Value& value)
{	CPPAD_ASSERT_UNKNOWN(
		std::numeric_limits<unsigned short>::max()
		>=
		CPPAD_HASH_TABLE_SIZE
	);
	CPPAD_ASSERT_UNKNOWN( sizeof(unsigned short) == 2 );
	CPPAD_ASSERT_UNKNOWN( sizeof(value) % 2  == 0 );
	//
	const unsigned short* v
	         = reinterpret_cast<const unsigned short*>(& value);
	//
	size_t i = sizeof(value) / 2 - 1;
	//
	size_t sum = v[i];
	//
	while(i--)
		sum += v[i];
	//
	unsigned short code = static_cast<unsigned short>(
		sum % CPPAD_HASH_TABLE_SIZE
	);
	return code;
}

/*!
Specialized hash code for a CppAD operator and its arguments.

\param op
is the operator that we are computing a hash code for.
If it is not one of the following operartors, the operator is not
hash coded and zero is returned:

\li unary operators:
AbsOp, AcosOp, AcoshOp, AsinOp, AsinhOp, AtanOp, AtanhOp, CosOp, CoshOp
ExpOp, Expm1Op, LogOp, Log1pOp, SinOp, SinhOp, SqrtOp, TanOp, TanhOp

\li binary operators where first argument is a parameter:
AddpvOp, DivpvOp, MulpvOp, PowpvOp, SubpvOp, ZmulpvOp

\li binary operators where second argument is a parameter:
DivvpOp, PowvpOp, SubvpOp, Zmulvp

\li binary operators where first is an index and second is a variable:
DisOp

\li binary operators where both arguments are variables:
AddvvOp, DivvvOp, MulvvOp, PowvvOp, SubvvOp, ZmulvvOp

\param arg
is a vector of length \c NumArg(op) or 2 (which ever is smaller),
containing the corresponding argument indices for this operator.

\param npar
is the number of parameters corresponding to this operation sequence.

\param par
is a vector of length \a npar containing the parameters
for this operation sequence; i.e.,
given a parameter index of \c i, the corresponding parameter value is
\a par[i].


\return
is a hash code that is between zero and CPPAD_HASH_TABLE_SIZE - 1.

\par Checked Assertions
\c op must be one of the operators specified above. In addition,
\li \c std::numeric_limits<unsigned short>::max() >= CPPAD_HASH_TABLE_SIZE
\li \c sizeof(size_t) is even
\li \c sizeof(Base) is even
\li \c sizeof(unsigned short)  == 2
\li \c size_t(op) < size_t(NumberOp) <= CPPAD_HASH_TABLE_SIZE
\li if the j-th argument for this operation is a parameter, arg[j] < npar.
*/

template <class Base>
unsigned short local_hash_code(
	OpCode        op      ,
	const addr_t* arg     ,
	size_t npar           ,
	const Base* par       )
{	CPPAD_ASSERT_UNKNOWN(
		std::numeric_limits<unsigned short>::max()
		>=
		CPPAD_HASH_TABLE_SIZE
	);
	CPPAD_ASSERT_UNKNOWN( size_t (op) < size_t(NumberOp) );
	CPPAD_ASSERT_UNKNOWN( sizeof(unsigned short) == 2 );
	CPPAD_ASSERT_UNKNOWN( sizeof(addr_t) % 2  == 0 );
	CPPAD_ASSERT_UNKNOWN( sizeof(Base) % 2  == 0 );
	unsigned short op_fac = static_cast<unsigned short> (
		CPPAD_HASH_TABLE_SIZE / static_cast<unsigned short>(NumberOp)
	);
	CPPAD_ASSERT_UNKNOWN( op_fac > 0 );

	// number of shorts per addr_t value
	size_t short_addr_t   = sizeof(addr_t) / 2;

	// number of shorts per Base value
	size_t short_base     = sizeof(Base) /  2;

	// initialize with value that separates operators as much as possible
	unsigned short code = static_cast<unsigned short>(
		static_cast<unsigned short>(op) * op_fac
	);

	// now code in the operands
	size_t i;
	const unsigned short* v;

	// first argument
	switch(op)
	{	// Binary operators where first arugment is a parameter.
		// Code parameters by value instead of
		// by index for two reasons. One, it gives better separation.
		// Two, different indices can be same parameter value.
		case AddpvOp:
		case DivpvOp:
		case MulpvOp:
		case PowpvOp:
		case SubpvOp:
		case ZmulpvOp:
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 2 );
		v = reinterpret_cast<const unsigned short*>(par + arg[0]);
		i = short_base;
		while(i--)
			code += v[i];
		v = reinterpret_cast<const unsigned short*>(arg + 1);
		i = short_addr_t;
		while(i--)
			code += v[i];
		break;

		// Binary operator where first argument is an index and
		// second is a variable (same as both variables).
		case DisOp:

		// Binary operators where both arguments are variables
		case AddvvOp:
		case DivvvOp:
		case MulvvOp:
		case PowvvOp:
		case SubvvOp:
		case ZmulvvOp:
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 2 );
		v = reinterpret_cast<const unsigned short*>(arg + 0);
		i = 2 * short_addr_t;
		while(i--)
			code += v[i];
		break;

		// Binary operators where second arugment is a parameter.
		case DivvpOp:
		case PowvpOp:
		case SubvpOp:
		case ZmulvpOp:
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 2 );
		v = reinterpret_cast<const unsigned short*>(arg + 0);
		i = short_addr_t;
		while(i--)
			code += v[i];
		v = reinterpret_cast<const unsigned short*>(par + arg[1]);
		i = short_base;
		while(i--)
			code += v[i];
		break;

		// Unary operators
		case AbsOp:
		case AcosOp:
		case AcoshOp:
		case AsinOp:
		case AsinhOp:
		case AtanOp:
		case AtanhOp:
		case CosOp:
		case CoshOp:
		case ErfOp:
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
		CPPAD_ASSERT_UNKNOWN( NumArg(op) == 1 || op == ErfOp );
		v = reinterpret_cast<const unsigned short*>(arg + 0);
		i = short_addr_t;
		while(i--)
			code += v[i];
		break;

		// should have been one of he cases above
		default:
		CPPAD_ASSERT_UNKNOWN(false);
	}

	return code % CPPAD_HASH_TABLE_SIZE;
}

} } // END_CPPAD_LOCAL_NAMESPACE

# endif
