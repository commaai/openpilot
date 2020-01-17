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
*    \file include/acado/simulation_environment/simulation_environment.hpp
*    \author Hans Joachim Ferreau, Boris Houska
*/


#ifndef ACADO_TOOLKIT_SIMULATION_ENVIRONMENT_HPP
#define ACADO_TOOLKIT_SIMULATION_ENVIRONMENT_HPP


#include <acado/utils/acado_utils.hpp>

#include <acado/simulation_environment/simulation_block.hpp>

#include <acado/clock/clock.hpp>
#include <acado/curve/curve.hpp>

#include <acado/process/process.hpp>
#include <acado/controller/controller.hpp>


BEGIN_NAMESPACE_ACADO



/**
 *	\brief Allows to run closed-loop simulations of dynamic systems.
 *
 *	\ingroup UserInterfaces
 *
 *	The class Simulation Environment is designed to run closed-loop simulations
 *	of dynamic systems.
 *
 *	In a standard setup the Simulation Environment consists of a Process and a 
 *	Controller that are connected by signals. These two main members can be specified
 *	within the constructor or set afterwards by using the methods  "setController"  and  
 *	"setProcess" , respectively.
 *
 *	A simulation has to be initialized by providing the initial value of the differential
 *	states of the dynamic system to be simulated. Afterwards, the simulation can be run
 *	at once or stepped until a given intermediate time.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 *
 */
class SimulationEnvironment : public SimulationBlock
{
	//
	//  PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor. 
		 */
		SimulationEnvironment( );

		/** Constructor which takes the name of the block and the sampling time.
		 *
		 *	@param[in] _startTime		Start time of the simulation.
		 *	@param[in] _endTime			End time of the simulation.
		 *	@param[in] _process			Process used for simulating the dynamic system.
 		 *	@param[in] _controller		Controller used for controlling the dynamic system.
		 *
		 *	\note Only pointers to Process and Controller are stored!
		 */
		SimulationEnvironment(	double _startTime,
								double _endTime,
								Process& _process,
								Controller& _controller
								);
	
		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		SimulationEnvironment(	const SimulationEnvironment &rhs
								);
	
		/** Destructor. 
		 */
		virtual ~SimulationEnvironment();
	
		/** Assignment Operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		SimulationEnvironment& operator=(	const SimulationEnvironment &rhs
											);
	

		/** Assigns new process block to be used for simulation.
		 *
		 *	@param[in]  _process		New process block.
		 *
		 *	\note Only a pointer is stored!
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue setProcess(	Process& _process
								);

		/** Assigns new controller block to be used for simulation.
		 *
		 *	@param[in]  _controller		New controller block.
		 *
		 *	\note Only a pointer is stored!
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue setController(	Controller& _controller
									);


		/** Initializes algebraic states of the process.
		 *
		 *	@param[in]  _xa_init	Initial value for algebraic states.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue initializeAlgebraicStates(	const VariablesGrid& _xa_init
												);

		/** Initializes algebraic states of the process from data file.
		 *
		 *	@param[in]  fileName	Name of file containing initial value for algebraic states.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_FILE_CAN_NOT_BE_OPENED
		 */
		returnValue initializeAlgebraicStates(	const char* fileName
												);


		/** Initializes the simulation with given start values and 
		 *	performs a number of consistency checks.
		 *
		 *	@param[in]  x0_		Initial value for differential states.
		 *	@param[in]  p_		Initial value for parameters.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_ENVIRONMENT_INIT_FAILED, \n
		 *	        RET_NO_CONTROLLER_SPECIFIED, \n
		 *	        RET_NO_PROCESS_SPECIFIED, \n
		 *	        RET_BLOCK_DIMENSION_MISMATCH
		 */
        returnValue init(	const DVector &x0_,
							const DVector &p_ = emptyConstVector
							);


		/** Performs next step of the simulation.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_ENVIRONMENT_STEP_FAILED, \n
		 *	        RET_COMPUTATIONAL_DELAY_TOO_BIG
		 */
		returnValue step( );

		/** Performs next steps of the simulation until given intermediate time.
		 *
		 *	@param[in]  intermediateTime		Intermediate time.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_ENVIRONMENT_STEP_FAILED, \n
		 *	        RET_COMPUTATIONAL_DELAY_TOO_BIG, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue step(	double intermediateTime
							);


		/** Runs the complete simulation.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_ENVIRONMENT_STEP_FAILED, \n
		 *	        RET_COMPUTATIONAL_DELAY_TOO_BIG
		 */
		returnValue run( );


		/** Returns number of process outputs.
		 *
		 *	\return Number of process outputs
		 */
		inline uint getNY( ) const;

		/** Returns number of feedback controls.
		 *
		 *	\return Number of feedback controls
		 */
		inline uint getNU( ) const;

		/** Returns number of feedback parameters.
		 *
		 *	\return Number of feedback parameters
		 */
		inline uint getNP( ) const;


		/** Returns current number of simulation steps.
		 *
		 *	\return Current number of simulation steps
		 */
		inline uint getNumSteps( ) const;


		/** Returns continuous output of the process.
		 *
		 *	@param[out]  _processOutput			Continuous output of the process.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		inline returnValue getProcessOutput(	Curve& _processOutput
												) const;

		/** Returns output of the process at sampling instants.
		 *
		 *	@param[out]  _sampledProcessOutput	Sampled output of the process.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		inline returnValue getSampledProcessOutput(	VariablesGrid& _sampledProcessOutput
													);


		/** Returns differential states of the process over the whole simulation.
		 *
		 *	@param[out]  _diffStates			Differential states of the process.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		inline returnValue getProcessDifferentialStates(	VariablesGrid& _diffStates
															);

		/** Returns algebraic states of the process over the whole simulation.
		 *
		 *	@param[out]  _algStates				Algebraic states of the process.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		inline returnValue getProcessAlgebraicStates(		VariablesGrid& _algStates
															);

		/** Returns intermediate states of the process over the whole simulation.
		 *
		 *	@param[out]  _interStates			Intermediate states of the process.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		inline returnValue getProcessIntermediateStates(	VariablesGrid& _interStates
															);


		/** Returns feedback control signals of the controller over the whole simulation.
		 *
		 *	@param[out]  _feedbackControl			Feedback control signals of the controller.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		inline returnValue getFeedbackControl(	Curve& _feedbackControl
												) const;

		/** Returns feedback control signals of the controller over the whole simulation.
		 *
		 *	@param[out]  _sampledFeedbackControl	Feedback control signals of the controller.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		inline returnValue getFeedbackControl(	VariablesGrid& _sampledFeedbackControl
												);


		/** Returns feedback parameter signals of the controller over the whole simulation.
		 *
		 *	@param[out]  _feedbackParameter			Feedback parameter signals of the controller.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		inline returnValue getFeedbackParameter(	Curve& _feedbackParameter
													) const;

		/** Returns feedback parameter signals of the controller over the whole simulation.
		 *
		 *	@param[out]  _sampledFeedbackParameter	Feedback parameter signals of the controller.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		inline returnValue getFeedbackParameter(	VariablesGrid& _sampledFeedbackParameter
													);



	//
	//  PROTECTED MEMBER FUNCTIONS:
	//
	protected:

		/** Sets-up default options.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue setupOptions( );

		/** Sets-up default logging information.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue setupLogging( );


		/** Returns computational delay used for simulation based on the actual real
		 *	controller runtime and the options set by the user.
		 *
		 *	@param[in]  controllerRuntime	Real controller runtime.
		 *
		 *  \return Computational delay used for simulation
		 */
		double determineComputationalDelay(	double controllerRuntime
											) const;


	//
	//  PROTECTED MEMBERS:
	//
	protected:
		double startTime;							/**< Start time of the simulation. */
		double endTime;								/**< End time of the simulation. */

		Process* process;							/**< Pointer to Process used for simulating the dynamic system. */
		Controller* controller;			 			/**< Pointer to Controller used for controlling the dynamic system. */

		SimulationClock simulationClock;			/**< Clock for managing the simulation time. */

		Curve processOutput;						/**< Curve storing output of the Process during the whole simulation. */
		Curve feedbackControl;						/**< Curve storing control signals from the Controller during the whole simulation. */
		Curve feedbackParameter;					/**< Curve storing parameter signals from the Controller during the whole simulation. */

		uint nSteps;								/**< Number of simulation steps (loops) that have been performed. */
};


CLOSE_NAMESPACE_ACADO



#include <acado/simulation_environment/simulation_environment.ipp>


#endif	// ACADO_TOOLKIT_SIMULATION_ENVIRONMENT_HPP


/*
 *	end of file
 */
