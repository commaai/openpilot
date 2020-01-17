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
 *	\file include/acado/noise/noise.ipp
 *	\author Boris Houska, Hans Joachim Ferreau
 *	\date 24.08.2008
 */



BEGIN_NAMESPACE_ACADO


//
// PUBLIC MEMBER FUNCTIONS:
//

inline uint Noise::getDim( ) const
{
	return w.getNumValues( );
}


inline BooleanType Noise::isEmpty( ) const
{
	if ( getDim( ) == 0 )
		return BT_TRUE;
	else
		return BT_FALSE;
}



inline BlockStatus Noise::getStatus( ) const
{
	return status;
}


//
// PROTECTED MEMBER FUNCTIONS:
//

inline returnValue Noise::setStatus(	BlockStatus _status
										)
{
	status = _status;
	return SUCCESSFUL_RETURN;
}



inline double Noise::getUniformRandomNumber(	double _lowerLimit,
												double _upperLimit
												) const
{
	double halfAmplitude = ( _upperLimit - _lowerLimit ) / 2.0;

	/* Random number between -RAND_MAX and RAND_MAX */
	int randomNumber = ( rand( ) - RAND_MAX/2 ) * 2;

	/* Random number between -1 and 1 */
	double scaledRandomNumber = ((double) randomNumber) / ((double) RAND_MAX);

	return ( halfAmplitude*scaledRandomNumber + _lowerLimit+halfAmplitude );
}


CLOSE_NAMESPACE_ACADO

// end of file.
