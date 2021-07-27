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
*    \file include/acado/process/process.hpp
*    \author Boris Houska, Hans Joachim Ferreau
*/


#ifndef ACADO_TOOLKIT_PROCESS_HPP
#define ACADO_TOOLKIT_PROCESS_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/simulation_environment/simulation_block.hpp>

#include <acado/noise/noise.hpp>
#include <acado/transfer_device/actuator.hpp>
#include <acado/transfer_device/sensor.hpp>
#include <acado/curve/curve.hpp>
#include <acado/dynamic_system/dynamic_system.hpp>
#include <acado/dynamic_discretization/shooting_method.hpp>


BEGIN_NAMESPACE_ACADO



/** 
 *	\brief  Simulates the process to be controlled based on a dynamic model.
 *
 *	\ingroup UserInterfaces
 *
 *  The class Process is one of the two main building-blocks within the 
 *  SimulationEnvironment and complements the Controller. It simulates the 
 *	process to be controlled based on a dynamic model.
 *
 *	The Process simulates the dynamic model based on the controls, and optionally 
 *	parameters, passed. Before using these inputs, they can be transformed by an
 *	Actuator. After the simulation, the outputs can be transformed by a Sensor.
 *	That way, actuator/sensor delays or noise can be introduced to yield more
 *	realistic simulation results. Moreover, in case the dynamic model depends on 
 *	Disturbances, their values are specified by the user by assigning the 
 *	processDisturbance member.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class Process : public SimulationBlock
{
	//
	//  PUBLIC MEMBER FUNCTIONS:
	//
	public:

        /** Default constructor. 
		 */
        Process( );

		/** Constructor which takes the dynamic system and the type of the 
		 *	integrator used for simulation.
		 *
		 *	@param[in] _dynamicSystem	Dynamic system to be used for simulation.
		 *	@param[in] _integratorType	Type of integrator to be used for simulation.
		 *
		 *	\note This constructor takes the dynamic system of the first model stage,
		 *	      multi-stage models can be simulated by adding further dynamic systems 
		 *	      (however, this feature is not functional yet!).
		 */
		Process(	const DynamicSystem& _dynamicSystem,
					IntegratorType _integratorType = INT_UNKNOWN
					);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
        Process(	const Process& rhs
					);

        /** Destructor. 
		 */
        virtual ~Process( );

        /** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
        Process& operator=(	const Process& rhs
							);


		/** Assigns new dynamic system to be used for simulation. All previously assigned
		 *	dynamic systems will be deleted.
		 *
		 *	@param[in] _dynamicSystem	Dynamic system to be used for simulation.
		 *	@param[in] _integratorType	Type of integrator to be used for simulation.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue setDynamicSystem(	const DynamicSystem& _dynamicSystem,
										IntegratorType _integratorType = INT_UNKNOWN
										);

		/** Assigns new dynamic system stage to be used for simulation.
		 *
		 *	@param[in] _dynamicSystem	Dynamic system to be used for simulation.
		 *	@param[in] _integratorType	Type of integrator to be used for simulation.
		 *
		 *	\note Multi-stage models are not yet supported!
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_NOT_YET_IMPLEMENTED
		 */
		returnValue addDynamicSystemStage(	const DynamicSystem& _dynamicSystem,
											IntegratorType _integratorType = INT_UNKNOWN
											);


		/** Assigns new actuator to be used for simulation.
		 *
		 *	@param[in]  _actuator		New actuator.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue setActuator(	const Actuator& _actuator
									);

		/** Assigns new sensor to be used for simulation.
		 *
		 *	@param[in]  _sensor		New sensor.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue setSensor(	const Sensor& _sensor
								);


		/** Assigns new process disturbance to be used for simulation.
		 *
		 *	@param[in]  _processDisturbance		New sensor.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue setProcessDisturbance(	const Curve& _processDisturbance
											);

		/** Assigns new process disturbance to be used for simulation.
		 *
		 *	@param[in]  _processDisturbance		New sensor.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue setProcessDisturbance(	const VariablesGrid& _processDisturbance
											);

		/** Assigns new process disturbance to be used for simulation.
		 *
		 *	@param[in]  _processDisturbance		New sensor.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_FILE_CAN_NOT_BE_OPENED
		 */
		returnValue setProcessDisturbance(	const char* _processDisturbance
											);


		/** Initializes simulation with given start values.
		 *
		 *	@param[in]  _xStart		Initial value for differential states.
		 *	@param[in]  _xaStart	Initial value for algebraic states.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue initializeStartValues(	const DVector& _xStart,
											const DVector& _xaStart = emptyConstVector
											);

		/** Initializes simulation with given start value for algebraic states.
		 *
		 *	@param[in]  _xaStart	Initial value for algebraic states.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue initializeAlgebraicStates(	const DVector& _xaStart
												);


		/** Initializes the simulation with given start values and 
		 *	performs a number of consistency checks.
		 *
		 *	@param[in]  _startTime	Start time of simulation.
		 *	@param[in]  _xStart		Initial value for differential states.
		 *	@param[in]  _uStart		Initial value for controls.
		 *	@param[in]  _pStart		Initial value for parameters.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_PROCESS_INIT_FAILED, \n
		 *	        RET_NO_DYNAMICSYSTEM_SPECIFIED, \n
		 *	        RET_DIFFERENTIAL_STATE_DIMENSION_MISMATCH, \n
		 *	        RET_ALGEBRAIC_STATE_DIMENSION_MISMATCH, \n
		 *	        RET_CONTROL_DIMENSION_MISMATCH, \n
		 *	        RET_PARAMETER_DIMENSION_MISMATCH, \n
		 *	        RET_DISTURBANCE_DIMENSION_MISMATCH, \n
		 *	        RET_WRONG_DISTURBANCE_HORIZON, \n
		 *	        RET_INCOMPATIBLE_ACTUATOR_SAMPLING_TIME, \n
		 *	        RET_INCOMPATIBLE_SENSOR_SAMPLING_TIME
		 */
        virtual returnValue init(	double _startTime = 0.0,
									const DVector& _xStart = emptyConstVector,
									const DVector& _uStart = emptyConstVector,
									const DVector& _pStart = emptyConstVector
									);


		/** Performs one step of the simulation based on given inputs.
		 *
		 *	@param[in]  _u		Time-varying controls.
		 *	@param[in]  _p		Time-varying parameters.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_PROCESS_STEP_FAILED, \n
		 *	        RET_PROCESS_STEP_FAILED_DISTURBANCE, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
        virtual returnValue step(	const VariablesGrid& _u,
									const VariablesGrid& _p = emptyVariablesGrid
									);

		/** Performs one step of the simulation based on given inputs.
		 *
		 *	@param[in]  _u		Time-varying controls.
		 *	@param[in]  _p		Time-constant parameters.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_PROCESS_STEP_FAILED, \n
		 *	        RET_PROCESS_STEP_FAILED_DISTURBANCE, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
        virtual returnValue step(	const VariablesGrid& _u,
									const DVector& _p
									);

		/** Performs one step of the simulation based on given inputs.
		 *
		 *	@param[in]  startTime	Start time of simulation step.
		 *	@param[in]  endTime		End time of simulation step.
		 *	@param[in]  _u			Time-constant controls.
		 *	@param[in]  _p			Time-constant parameters.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_PROCESS_STEP_FAILED, \n
		 *	        RET_PROCESS_STEP_FAILED_DISTURBANCE, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
        virtual returnValue step(	double startTime,
									double endTime,
									const DVector& _u,
									const DVector& _p = emptyConstVector
									);


		/** Initializes simulation and performs one step based on given inputs.
		 *
		 *	@param[in]  _u		Time-varying controls.
		 *	@param[in]  _p		Time-varying parameters.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_PROCESS_STEP_FAILED, \n
		 *	        RET_PROCESS_STEP_FAILED_DISTURBANCE, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
        virtual returnValue run(	const VariablesGrid& _u,
									const VariablesGrid& _p = emptyVariablesGrid
									);

		/** Initializes simulation and performs one step based on given inputs.
		 *
		 *	@param[in]  _u		Time-varying controls.
		 *	@param[in]  _p		Time-constant parameters.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_PROCESS_STEP_FAILED, \n
		 *	        RET_PROCESS_STEP_FAILED_DISTURBANCE, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
        virtual returnValue run(	const VariablesGrid& _u,
									const DVector& _p
									);

		/** Initializes simulation and performs one step based on given inputs.
		 *
		 *	@param[in]  startTime	Start time of simulation step.
		 *	@param[in]  endTime		End time of simulation step.
		 *	@param[in]  _u			Time-constant controls.
		 *	@param[in]  _p			Time-constant parameters.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_PROCESS_STEP_FAILED, \n
		 *	        RET_PROCESS_STEP_FAILED_DISTURBANCE, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
        virtual returnValue run(	double startTime,
									double endTime,
									const DVector& _u,
									const DVector& _p = emptyConstVector
									);


		/** Returns output of the process.
		 *
		 *	@param[out]  _y			Output of the process.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
        inline returnValue getY(	VariablesGrid& _y
									) const;


		/** Returns number of control signals (at given stage) expected by the process.
		 *
		 *	@param[in]  stageIdx	Index of stage.
		 *
		 *	\return Number of control signals (at given stage)
		 */
		inline uint getNU(	uint stageIdx = 0
							) const;

		/** Returns number of parameter signals (at given stage) expected by the process.
		 *
		 *	@param[in]  stageIdx	Index of stage.
		 *
		 *  \return Number of parameter signals (at given stage)
		 */
		inline uint getNP(	uint stageIdx = 0
							) const;

		/** Returns number of disturbances (at given stage) used within the process.
		 *
		 *	@param[in]  stageIdx	Index of stage.
		 *
		 *  \return Number of disturbances (at given stage)
		 */
		inline uint getNW(	uint stageIdx = 0
							) const;

		/** Returns number of process outputs (at given stage).
		 *
		 *	@param[in]  stageIdx	Index of stage.
		 *
		 *  \return Number of process outputs (at given stage)
		 */
		inline uint getNY(	uint stageIdx = 0
							) const;


		/** Returns number of stages of the dynamic model.
		 *
		 *  \return Number of stages of the dynamic model
		 */
		inline uint getNumStages( ) const;


		/** Returns whether dynamic model (at given stage) is an ODE.
		 *
		 *	@param[in]  stageIdx	Index of stage.
		 *
		 *  \return BT_TRUE  iff dynamic model is an ODE, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType isODE(	uint stageIdx = 0
									) const;

		/** Returns whether dynamic model (at given stage) is a DAE.
		 *
		 *	@param[in]  stageIdx	Index of stage.
		 *
		 *  \return BT_TRUE  iff dynamic model is a DAE, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType isDAE(	uint stageIdx = 0
									) const;


		/** Returns whether dynamic model (at given stage) is discretized in time.
		 *
		 *	@param[in]  stageIdx	Index of stage.
		 *
		 *  \return BT_TRUE  iff dynamic model is discretized in time, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType isDiscretized(	uint stageIdx = 0
											) const;

		/** Returns whether dynamic model (at given stage) is continuous in time.
		 *
		 *	@param[in]  stageIdx	Index of stage.
		 *
		 *  \return BT_TRUE  iff dynamic model is continuous in time, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType isContinuous(	uint stageIdx = 0
											) const;


		/** Returns whether process comprises an actuator.
		 *
		 *  \return BT_TRUE  iff process comprises an actuator, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasActuator( ) const;

		/** Returns whether process comprises a sensor.
		 *
		 *  \return BT_TRUE  iff process comprises a sensor, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasSensor( ) const;

		/** Returns whether process comprises a process disturbance.
		 *
		 *  \return BT_TRUE  iff process comprises a process disturbance, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasProcessDisturbance( ) const;


		/** Customized function for plotting process variables.
		 *
		 *	@param[in] _frequency	Frequency determining at which time instants the window is to be plotted.
         *
         *	\return SUCCESSFUL_RETURN
         */
		virtual returnValue replot(	PlotFrequency _frequency = PLOT_IN_ANY_CASE
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


		/** Returns number of differential states (at given stage) of the dynamic model.
		 *
		 *	@param[in]  stageIdx	Index of stage.
		 *
		 *  \return Number of differential states
		 */
		inline uint getNX(	uint stageIdx = 0
							) const;

		/** Returns number of algebraic states (at given stage) of the dynamic model.
		 *
		 *	@param[in]  stageIdx	Index of stage.
		 *
		 *  \return Number of algebraic states
		 */
		inline uint getNXA(	uint stageIdx = 0
							) const;


		/** Internally adds a new dynamic system stage to be used for simulation.
		 *
		 *	@param[in] _dynamicSystem	Dynamic system to be used for simulation.
		 *	@param[in] stageIntervals	Dummy grid.
		 *	@param[in] _integratorType	Type of integrator to be used for simulation.
		 *
		 *	\note Multi-stage models are not yet supported!
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
        returnValue addStage( 	const DynamicSystem  &dynamicSystem_,
								const Grid           &stageIntervals,
								const IntegratorType &integratorType_ = INT_UNKNOWN
								);


		/** Clears all dynamic systems and all members.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue clear( );


		/** Actually calls the integrator for performing a simulation. All
		 *	simulated results are logged internally.
		 *
		 *	@param[in]  _u		Time-varying controls.
		 *	@param[in]  _p		Time-varying parameters.
		 *	@param[in]  _w		Time-varying disturbances.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue simulate(	const VariablesGrid& _u,
								const VariablesGrid& _p,
								const VariablesGrid& _w
								);


		/** Checks consistency of the given inputs (dimensions, time grids etc.).
		 *
		 *	@param[in]  _u		Time-varying controls.
		 *	@param[in]  _p		Time-varying parameters.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_CONTROL_DIMENSION_MISMATCH, \n
		 *	        RET_PARAMETER_DIMENSION_MISMATCH, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue checkInputConsistency(	const VariablesGrid& _u,
											const VariablesGrid& _p
											) const;


		/** Calculates the process output based on the simulated states by
		 *	evaluating the output function of the dynamic system.
		 *
		 *	@param[in]  _x				Differential states.
		 *	@param[in]  _xComponents	Global components of differential states actually used.
		 *	@param[in]  _xa				Algebraic states.
		 *	@param[in]  _p				Parameters.
		 *	@param[in]  _u				Controls.
		 *	@param[in]  _w				Disturbances.
		 *	@param[in]  _p				Parameters.
		 *	@param[out] _output			Time-varying process output.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue calculateOutput(	OutputFcn& _outputFcn,
										const VariablesGrid* _x,
										const DVector& _xComponents,
										const VariablesGrid* _xa,
										const VariablesGrid* _p,
										const VariablesGrid* _u,
										const VariablesGrid* _w,
										VariablesGrid* _output
										) const;

		/** Projects differential states to global components actually used.
		 *
		 *	@param[in]  _x				Differential states.
		 *	@param[in]  _xComponents	Global components of differential states actually used.
		 *	@param[out] _output			Projected differential states.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue projectToComponents(	const VariablesGrid& _x,
											const DVector& _xComponents,
											VariablesGrid& _output
											) const;


	//
	//  PROTECTED MEMBERS:
	//
	protected:

		DVector x;
		DVector xa;

		uint nDynSys;								/**< Number of dynamic systems. */
		DynamicSystem** dynamicSystems;				/**< Dynamic system to be used for simulation. */

		ShootingMethod* integrationMethod;			/**< Integration method to be used for simulation. */
	
		Actuator* actuator;							/**< Actuator. */
		Sensor* sensor;								/**< Sensor. */
		Curve* processDisturbance;					/**< Process disturbance block. */

		VariablesGrid y;

		double lastTime;


		IntegratorType integratorType;  // sorry -- quick hack.

};


CLOSE_NAMESPACE_ACADO



#include <acado/process/process.ipp>


#endif	// ACADO_TOOLKIT_PROCESS_HPP


/*
 *	end of file
 */
