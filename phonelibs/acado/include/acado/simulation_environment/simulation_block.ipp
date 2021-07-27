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
*    \file include/acado/simulation_environment/simulation_block.ipp
*    \author Boris Houska, Hans Joachim Ferreau
*/



//
//  PUBLIC MEMBER FUNCTIONS:
//


BEGIN_NAMESPACE_ACADO


inline BooleanType SimulationBlock::isDefined( ) const
{
	if ( status == BS_UNDEFINED )
		return BT_FALSE;
	else
		return BT_TRUE;
}


inline BlockName SimulationBlock::getName( ) const
{
	return name;
}


inline double SimulationBlock::getSamplingTime( ) const
{
	return samplingTime;
}



inline returnValue SimulationBlock::setName(	BlockName _name
												)
{
	name = _name;
	return SUCCESSFUL_RETURN;
}


inline returnValue SimulationBlock::setSamplingTime(	double _samplingTime
														)
{
	if ( acadoIsGreater( _samplingTime,0.0 ) == BT_TRUE )
	{
		samplingTime = _samplingTime;
		return SUCCESSFUL_RETURN;
	}
	else
		return ACADOERROR( RET_INVALID_ARGUMENTS );
}


CLOSE_NAMESPACE_ACADO


/*
 *	end of file
 */
