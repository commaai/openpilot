/*
 *    This file is part of ACADO Toolkit.
 *
 *    ACADO Toolkit -- A Toolkit for Automatic Control and Dynamic Optimization.
 *    Copyright (C) 2008-2014 by Boris Houska, Hans Joachim Ferreau,
 *    Milan Vukov, Rien Quirynen, KU Leuven.
 *    Developed within the Optimization in Engineering Center (OPTEC)
 *    under supervision of Moritz Diehl. All rights reserved.
 *
 *    ACADO Toolkit is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    ACADO Toolkit is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with ACADO Toolkit; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


/**
 *    \file   include/utils/acado_utils.hpp
 *    \author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 *    \date   2008 - 2013
 *
 *    This file declares several global utility functions.
 */


#ifndef ACADO_TOOLKIT_ACADO_UTILS_HPP
#define ACADO_TOOLKIT_ACADO_UTILS_HPP

#include <cmath>

#include <acado/utils/acado_types.hpp>
#include <acado/utils/acado_constants.hpp>
#include <acado/utils/acado_default_options.hpp>
#include <acado/utils/acado_message_handling.hpp>
#include <acado/utils/acado_debugging.hpp>
#include <acado/utils/acado_io_utils.hpp>

// A very ugly hack
#if (defined __MINGW32__ || defined __MINGW64__)
namespace std { namespace tr1 { using namespace std; } }
#endif // (defined __MINGW32__ || defined __MINGW64__)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

BEGIN_NAMESPACE_ACADO

/** Returns if x is integer-valued. */
BooleanType acadoIsInteger( double x );

/** Safe division. */
double acadoDiv( double nom, double den );

/** Safe modulus. */
double acadoMod( double nom, double den );

/** Returns the maximum of x and y. */
int acadoMax( const int x, const int y );

/** Returns the maximum of x and y. */
double acadoMax( const double x, const double y );

/** Returns the minimum of x and y. */
int acadoMin( const int x, const int y );

/** Returns the minimum of x and y. */
double acadoMin( const double x, const double y );

/** Returns the factorial 1*2*3*...*n of an integer n. \n
 *  \param n the input integer n.
 *  \return 1*2*3*...*n   if n>=1                      \n
 *          1 if n = 0. (and aborts if n < 0 ).        \n
 */
int acadoFactorial( int n );

/** Returns whether x and y are numerically equal. */
BooleanType acadoIsEqual( double x, double y, double TOL = EQUALITY_EPS );

/** Returns whether x is numerically greater or equal than y. */
BooleanType acadoIsGreater( double x, double y, double TOL = EQUALITY_EPS );

/** Returns whether x is numerically smaller or equal than y. */
BooleanType acadoIsSmaller( double x, double y, double TOL = EQUALITY_EPS );

/** Returns whether x is numerically strictly greater than y. */
BooleanType acadoIsStrictlyGreater( double x, double y, double TOL = EQUALITY_EPS );

/** Returns whether x is numerically strictly smaller than y. */
BooleanType acadoIsStrictlySmaller( double x, double y, double TOL = EQUALITY_EPS );

/** Returns whether x is numerically greater than 0. */
BooleanType acadoIsPositive( double x, double TOL = EQUALITY_EPS );

/** Returns whether x is numerically smaller than 0. */
BooleanType acadoIsNegative( double x, double TOL = EQUALITY_EPS );

/** Returns whether x is numerically 0. */
BooleanType acadoIsZero( double x, double TOL = EQUALITY_EPS );

/** Returns whether x is greater/smaller than +/-INFTY. */
BooleanType acadoIsInfty( double x, double TOL = 0.1 );

/** Returns whether x lies within [-INFTY,INFTY]. */
BooleanType acadoIsFinite( double x, double TOL = 0.1 );

/** Checks if any of elements is greater than. \sa acadoIsFinite */
template<class T>
BooleanType isFinite( const T& _value )
{
	for (unsigned el = 0; el < _value.size(); ++el)
		if ( acadoIsFinite( _value[ el ] ) == BT_TRUE )
            return BT_TRUE;
	return BT_FALSE;
}

/** Returns whether x is not a number. */
BooleanType acadoIsNaN(	double x );

/** Checks whether a constant is infinity. */
inline BooleanType isInfty(const double x)
{
	if (x - 10.0 < -INFTY)
		return BT_TRUE;

	if (x + 10.0 > INFTY)
		return BT_TRUE;

	return BT_FALSE;
}

/** Specific rounding implemenation for compiler who don't support the round
 *  command. Does a round to nearest.
 */
int acadoRound (double x);

/** Specific rounding implementation for rounding away from zero. */
int acadoRoundAway (double x);


CLOSE_NAMESPACE_ACADO

#endif	// ACADO_TOOLKIT_ACADO_UTILS_HPP

/*
 *	end of file
 */
