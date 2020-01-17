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
 *    \file include/acado/process/process.ipp
 *    \author Boris Houska, Hans Joachim Ferreau
 *    \date 24.08.2008
 */



BEGIN_NAMESPACE_ACADO


//
// PUBLIC MEMBER FUNCTIONS:
//

inline returnValue Process::getY(	VariablesGrid& _y
									) const
{
	_y = y;
	return SUCCESSFUL_RETURN;
}



inline uint Process::getNU(	uint stageIdx
							) const
{
	if ( stageIdx >= getNumStages( ) )
		return 0;
	else
		return dynamicSystems[stageIdx]->getNumControls( );
}


inline uint Process::getNP(	uint stageIdx
							) const
{
	if ( stageIdx >= getNumStages( ) )
		return 0;
	else
		return dynamicSystems[stageIdx]->getNumParameters( );
}


inline uint Process::getNW(	uint stageIdx
							) const
{
	if ( stageIdx >= getNumStages( ) )
		return 0;
	else
		return dynamicSystems[stageIdx]->getNumDisturbances( );
}


inline uint Process::getNY(	uint stageIdx
							) const
{
	if ( stageIdx >= getNumStages( ) )
		return 0;
	else
		return dynamicSystems[stageIdx]->getNumOutputs( );
}



inline uint Process::getNumStages( ) const
{
	return nDynSys;
}



inline BooleanType Process::isODE(	uint stageIdx
									) const
{
	if ( stageIdx >= getNumStages( ) )
		return BT_TRUE;
	else
		return dynamicSystems[stageIdx]->isODE( );
}


inline BooleanType Process::isDAE(	uint stageIdx
									) const
{
	if ( stageIdx >= getNumStages( ) )
		return BT_TRUE;
	else
		return dynamicSystems[stageIdx]->isDAE( );
}



inline BooleanType Process::isDiscretized(	uint stageIdx
											) const
{
	if ( stageIdx >= getNumStages( ) )
		return BT_TRUE;
	else
		return dynamicSystems[stageIdx]->isDiscretized( );
}


inline BooleanType Process::isContinuous(	uint stageIdx
											) const
{
	if ( stageIdx >= getNumStages( ) )
		return BT_TRUE;
	else
		return dynamicSystems[stageIdx]->isContinuous( );
}



inline BooleanType Process::hasActuator( ) const
{
	if ( actuator != 0 )
		return BT_TRUE;
	else
		return BT_FALSE;
}


inline BooleanType Process::hasSensor( ) const
{
	if ( sensor != 0 )
		return BT_TRUE;
	else
		return BT_FALSE;
}


inline BooleanType Process::hasProcessDisturbance( ) const
{
	if ( processDisturbance != 0 )
		return BT_TRUE;
	else
		return BT_FALSE;
}



//
// PROTECTED MEMBER FUNCTIONS:
//

inline uint Process::getNX(	uint stageIdx
							) const
{
	if ( stageIdx >= getNumStages( ) )
		return 0;
	else
		return dynamicSystems[stageIdx]->getNumDynamicEquations( );
}


inline uint Process::getNXA(	uint stageIdx
								) const
{
	if ( stageIdx >= getNumStages( ) )
		return 0;
	else
		return dynamicSystems[stageIdx]->getNumAlgebraicEquations( );
}



CLOSE_NAMESPACE_ACADO

// end of file.
