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
 *    \file include/acado/simulation_environment/simulation_environment.ipp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 24.08.2008
 */



BEGIN_NAMESPACE_ACADO


//
// PUBLIC MEMBER FUNCTIONS:
//

inline uint SimulationEnvironment::getNY( ) const
{
	if ( process != 0 )
		return process->getNY( );
	else
		return 0;
}


inline uint SimulationEnvironment::getNU( ) const
{
	if ( controller != 0 )
		return controller->getNU( );
	else
		return 0;
}


inline uint SimulationEnvironment::getNP( ) const
{
	if ( controller != 0 )
		return controller->getNP( );
	else
		return 0;
}


inline uint SimulationEnvironment::getNumSteps( ) const
{
	return nSteps;
}



inline returnValue SimulationEnvironment::getProcessOutput(	Curve& _processOutput
															) const
{
	_processOutput = processOutput;
	return SUCCESSFUL_RETURN;
}


inline returnValue SimulationEnvironment::getSampledProcessOutput(	VariablesGrid& _sampledProcessOutput
																	)
{
	MatrixVariablesGrid sampledProcessOutput;

	LogRecord tmp;
	tmp << LOG_PROCESS_OUTPUT;
	process->updateLogRecord( tmp );
    tmp.getAll( LOG_PROCESS_OUTPUT,sampledProcessOutput );
	
	DMatrix sampledProcessOutputMatrix( sampledProcessOutput.getMatrix(0) );

	for( uint i=1; i<sampledProcessOutput.getNumPoints()-1; ++i )
		sampledProcessOutputMatrix.appendRows( sampledProcessOutput.getMatrix(i) );

	_sampledProcessOutput = sampledProcessOutputMatrix;
	_sampledProcessOutput.setType( VT_OUTPUT );

	return SUCCESSFUL_RETURN;
}



inline returnValue SimulationEnvironment::getProcessDifferentialStates(	VariablesGrid& _diffStates
																		)
{
	MatrixVariablesGrid tmp;
	if ( process->getAll( LOG_DIFFERENTIAL_STATES,tmp ) != SUCCESSFUL_RETURN )
		return ACADOERROR( RET_MEMBER_NOT_INITIALISED );

	_diffStates.init( );
	
	for( int i=0; i<(int)tmp.getNumPoints()-1; ++i )
		_diffStates.appendTimes( tmp.getMatrix( i ) );

	_diffStates.setType( VT_DIFFERENTIAL_STATE );
	
	return SUCCESSFUL_RETURN;
}


inline returnValue SimulationEnvironment::getProcessAlgebraicStates(	VariablesGrid& _algStates
																		)
{
	MatrixVariablesGrid tmp;
	if( process->getAll( LOG_ALGEBRAIC_STATES,tmp ) != SUCCESSFUL_RETURN )
		return ACADOERROR( RET_MEMBER_NOT_INITIALISED );

	_algStates.init( );
	
	for( int i=0; i<(int)tmp.getNumPoints()-1; ++i )
		_algStates.appendTimes( tmp.getMatrix( i ) );

	_algStates.setType( VT_ALGEBRAIC_STATE );
	
	return SUCCESSFUL_RETURN;
}


inline returnValue SimulationEnvironment::getProcessIntermediateStates(	VariablesGrid& _interStates
																		)
{
	MatrixVariablesGrid tmp;
	if( process->getAll( LOG_INTERMEDIATE_STATES,tmp ) != SUCCESSFUL_RETURN )
		return ACADOERROR( RET_MEMBER_NOT_INITIALISED );

	_interStates.init( );

	for( int i=0; i<(int)tmp.getNumPoints()-1; ++i )
		_interStates.appendTimes( tmp.getMatrix( i ) );

	_interStates.setType( VT_INTERMEDIATE_STATE );
	
	return SUCCESSFUL_RETURN;
}



inline returnValue SimulationEnvironment::getFeedbackControl(	Curve& _feedbackControl
																) const
{
	_feedbackControl = feedbackControl;
	return SUCCESSFUL_RETURN;
}


inline returnValue SimulationEnvironment::getFeedbackControl(	VariablesGrid& _sampledFeedbackControl
																)
{
	if ( feedbackControl.isEmpty( ) == BT_TRUE )
		return ACADOERROR( RET_MEMBER_NOT_INITIALISED );
	
	Grid _samplingGrid( startTime,endTime, 10*(getNumSteps()+1) );

	feedbackControl.discretize( _samplingGrid,_sampledFeedbackControl );
	_sampledFeedbackControl.setType( VT_CONTROL );
	return SUCCESSFUL_RETURN;
}



inline returnValue SimulationEnvironment::getFeedbackParameter(	Curve& _feedbackParameter
																) const
{
	_feedbackParameter = feedbackParameter;
	return SUCCESSFUL_RETURN;
}


inline returnValue SimulationEnvironment::getFeedbackParameter(	VariablesGrid& _sampledFeedbackParameter
																)
{
	Grid _samplingGrid( startTime,endTime-1.0e-6*EPS, 10*(getNumSteps()+1) );

	feedbackParameter.discretize( _samplingGrid,_sampledFeedbackParameter );
	_sampledFeedbackParameter.setType( VT_PARAMETER );
	return SUCCESSFUL_RETURN;
}



CLOSE_NAMESPACE_ACADO

// end of file.
