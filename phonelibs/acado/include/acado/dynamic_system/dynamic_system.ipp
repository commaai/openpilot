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
 *    \file include/acado/dynamic_system/dynamic_system.ipp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 13.06.2008
 */


//
// PUBLIC MEMBER FUNCTIONS:
//


BEGIN_NAMESPACE_ACADO



inline returnValue DynamicSystem::getSubsystem(	uint stageIdx,
												DifferentialEquation& _diffEqn,
												OutputFcn& _outputFcn
												) const
{
	if ( ( diffEqn == 0 ) || ( outputFcn == 0 ) )
		return ACADOERROR( RET_MEMBER_NOT_INITIALISED );

	if (stageIdx >= nDiffEqn)
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );

	_diffEqn   = getDifferentialEquation( stageIdx );
	_outputFcn = getOutputFcn( stageIdx );

	return SUCCESSFUL_RETURN;
}


inline const DifferentialEquation& DynamicSystem::getDifferentialEquation(	uint stageIdx
																	) const
{
	ASSERT( diffEqn != 0 );
	ASSERT( stageIdx < nDiffEqn );

	return *(diffEqn[stageIdx]);
}


inline const OutputFcn& DynamicSystem::getOutputFcn(	uint stageIdx
												) const
{
	ASSERT( outputFcn != 0 );
	ASSERT( stageIdx < nDiffEqn );

	return *(outputFcn[stageIdx]);
}


inline returnValue DynamicSystem::getSwitchFunction(	uint idx,
														Function& _switchFcn
														) const
{
	if ( switchFcn == 0 )
		return ACADOERROR( RET_MEMBER_NOT_INITIALISED );

	if (idx > nSwitchFcn)
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );

	_switchFcn = *(switchFcn[idx]);
	return SUCCESSFUL_RETURN;
}


inline returnValue DynamicSystem::getSelectFunction(	Function& _selectFcn
														) const
{
	if ( selectFcn == 0 )
		return ACADOERROR( RET_MEMBER_NOT_INITIALISED );

	_selectFcn = *selectFcn;
	return SUCCESSFUL_RETURN;
}



inline BooleanType DynamicSystem::isODE( ) const
{
	if ( nDiffEqn > 0 )
		return diffEqn[0]->isODE( );
	else
		return BT_TRUE;
}


inline BooleanType DynamicSystem::isDAE( ) const
{
	if ( nDiffEqn > 0 )
		return diffEqn[0]->isDAE( );
	else
		return BT_TRUE;
}



inline BooleanType DynamicSystem::isDiscretized( ) const
{
	if ( nDiffEqn > 0 )
		return diffEqn[0]->isDiscretized( );
	else
		return BT_TRUE;
}


inline BooleanType DynamicSystem::isContinuous( ) const
{
	if ( nDiffEqn > 0 )
	{
		if ( diffEqn[0]->isDiscretized( ) == BT_FALSE )
			return BT_TRUE;
		else
			return BT_FALSE;
	}
	else
		return BT_TRUE;
}



inline double DynamicSystem::getSampleTime( ) const
{
	if ( nDiffEqn > 0 )
		return diffEqn[0]->getStepLength( );
	else
		return -INFTY;
}



inline uint DynamicSystem::getNumDynamicEquations( ) const
{
	if ( nDiffEqn > 0 )
		return diffEqn[0]->getNumDynamicEquations( );
	else
		return 0;
}

inline uint DynamicSystem::getNumAlgebraicEquations( ) const
{
	if ( nDiffEqn > 0 )
		return diffEqn[0]->getNumAlgebraicEquations( );
	else
		return 0;
}


inline uint DynamicSystem::getNumOutputs( ) const
{
	if ( nDiffEqn > 0 )
	{
		if ( outputFcn[0]->isDefined( ) == BT_TRUE )
			return outputFcn[0]->getDim( );
		else
			return diffEqn[0]->getNumDynamicEquations( );
	}
	else
		return 0;
}


inline uint DynamicSystem::getNumControls( ) const
{
	uint n = 0;

	for( uint i=0; i<nDiffEqn; ++i )
		if ( n < (uint) diffEqn[i]->getNU( ) )
			n = (uint) diffEqn[i]->getNU( );

	return n;
}


inline uint DynamicSystem::getNumParameters( ) const
{
	uint n = 0;

	for( uint i=0; i<nDiffEqn; ++i )
		if ( n < (uint) diffEqn[i]->getNP( ) )
			n = (uint) diffEqn[i]->getNP( );

	return n;
}


inline uint DynamicSystem::getNumDisturbances( ) const
{
	uint n = 0;

	for( uint i=0; i<nDiffEqn; ++i )
		if ( n < (uint) diffEqn[i]->getNW( ) )
			n = (uint) diffEqn[i]->getNW( );

	return n;
}



inline uint DynamicSystem::getNumSubsystems( ) const
{
	return nDiffEqn;
}


inline uint DynamicSystem::getNumSwitchFunctions( ) const
{
	return nSwitchFcn;
}


inline BooleanType DynamicSystem::hasImplicitSwitches( ) const
{
	if ( getNumSubsystems( ) > 1 )
		return BT_TRUE;
	else
		return BT_FALSE;
}



CLOSE_NAMESPACE_ACADO

// end of file.
