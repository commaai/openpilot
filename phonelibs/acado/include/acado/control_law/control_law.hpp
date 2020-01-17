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
 *	\file include/acado/control_law/control_law.hpp
 *	\author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_CONTROL_LAW_HPP
#define ACADO_TOOLKIT_CONTROL_LAW_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/simulation_environment/simulation_block.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Base class for interfacing online feedback laws to be used within a Controller.
 *
 *	\ingroup UserInterfaces
 *
 *  The class ControlLaw serves as a base class for interfacing online 
 *	control laws to be used within a Controller. Most prominently, the
 *	control law can be a RealTimeAlgorithm solving dynamic optimization
 *	problems. But also classical feedback laws like LQR or PID controller
 *	or feedforward laws can be interfaced.
 *
 *	After initialization, the ControlLaw is evaluated with a given fixed 
 *	sampling time by calling the step-routines. Additionally, the steps
 *	can be divided into a preparation step and a feedback step that actually
 *	computes the feedback. This feature has mainly been added to deal with 
 *	RealTimeAlgorithm can make use of this division in order to reduce the
 *	feedback delay.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class ControlLaw : public SimulationBlock
{
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor. 
		 */
		ControlLaw( );

		/** Constructor which takes the sampling time.
		 *
		 *	@param[in] _samplingTime	Sampling time.
		 */
		ControlLaw(	double _samplingTime
					);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		ControlLaw(	const ControlLaw& rhs
					);

		/** Destructor.
		 */
		virtual ~ControlLaw( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		ControlLaw& operator=(	const ControlLaw& rhs
								);

		/** Clone constructor (deep copy).
		 *
		 *	\return Pointer to deep copy of base class type
		 */
		virtual ControlLaw* clone( ) const = 0;


		/** Initializes algebraic states of the control law.
		 *
		 *	@param[in]  _xa_init	Initial value for algebraic states.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue initializeAlgebraicStates(	const VariablesGrid& _xa_init
														);

		/** Initializes algebraic states of the control law from data file.
		 *
		 *	@param[in]  fileName	Name of file containing initial value for algebraic states.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_FILE_CAN_NOT_BE_OPENED
		 */
		virtual returnValue initializeAlgebraicStates(	const char* fileName
														);


		/** Initializes controls of the control law.
		 *
		 *	@param[in]  _u_init	Initial value for controls.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue initializeControls(	const VariablesGrid& _u_init
												);

		/** Initializes controls of the control law from data file.
		 *
		 *	@param[in]  fileName	Name of file containing initial value for controls.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_FILE_CAN_NOT_BE_OPENED
		 */
		virtual returnValue initializeControls(	const char* fileName
												);


		/** Initializes the control law with given start values and 
		 *	performs a number of consistency checks.
		 *
		 *	@param[in]  _startTime	Start time.
		 *	@param[in]  _x			Initial value for differential states.
		 *	@param[in]  _p			Initial value for parameters.
		 *	@param[in]  _yRef		Initial value for reference trajectory.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_CONTROLLAW_INIT_FAILED
		 */
		virtual returnValue init(	double startTime = 0.0,
									const DVector& _x = emptyConstVector,
									const DVector& _p = emptyConstVector,
									const VariablesGrid& _yRef = emptyConstVariablesGrid
									) = 0;


		/** Performs next step of the control law based on given inputs.
		 *
		 *	@param[in]  currentTime	Current time.
		 *	@param[in]  _x			Most recent value for differential states.
		 *	@param[in]  _p			Most recent value for parameters.
		 *	@param[in]  _yRef		Current piece of reference trajectory or piece of reference trajectory for next step (required for hotstarting).
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_VECTOR_DIMENSION_MISMATCH, \n
		 *	        RET_CONTROLLAW_STEP_FAILED
		 */
		virtual returnValue step(	double currentTime,
									const DVector& _x,
									const DVector& _p = emptyConstVector,
									const VariablesGrid& _yRef = emptyConstVariablesGrid
									) = 0;
									
		/** Performs next step of the control law based on given inputs.
		 *
		 *	@param[in]  _x			Most recent value for differential states.
		 *	@param[in]  _p			Most recent value for parameters.
		 *	@param[in]  _yRef		Current piece of reference trajectory or piece of reference trajectory for next step (required for hotstarting).
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_VECTOR_DIMENSION_MISMATCH, \n
		 *	        RET_CONTROLLAW_STEP_FAILED
		 */
		virtual returnValue step(	const DVector& _x,
									const DVector& _p = emptyConstVector,
									const VariablesGrid& _yRef = emptyConstVariablesGrid
									);

		/** Performs next feedback step of the control law based on given inputs.
		 *
		 *	@param[in]  currentTime	Current time.
		 *	@param[in]  _x			Most recent value for differential states.
		 *	@param[in]  _p			Most recent value for parameters.
		 *	@param[in]  _yRef		Current piece of reference trajectory (if not specified during previous preparationStep).
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_VECTOR_DIMENSION_MISMATCH, \n
		 *	        RET_CONTROLLAW_STEP_FAILED
		 */
		virtual returnValue feedbackStep(	double currentTime,
											const DVector& _x,
											const DVector& _p = emptyConstVector,
											const VariablesGrid& _yRef = emptyConstVariablesGrid
											);

		/** Performs next preparation step of the control law based on given inputs.
		 *
		 *	@param[in]  nextTime	Time at next step.
		 *	@param[in]  _yRef		Piece of reference trajectory for next step (required for hotstarting).
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue preparationStep(	double nextTime = 0.0,
												const VariablesGrid& _yRef = emptyConstVariablesGrid
												);


		/** Shifts the data for preparating the next real-time step.
		 *
		 *	\return RET_NOT_YET_IMPLEMENTED
		 */
		virtual returnValue shift(	double timeShift = -1.0
									);

									
		/** Returns control signal as determined by the control law.
		 *
		 *	@param[out]  _u		Control signal as determined by the control law.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue getU(	DVector& _u
									) const;

		/** Returns parameter signal as determined by the control law.
		 *
		 *	@param[out]  _p		Parameter signal as determined by the control law.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue getP(	DVector& _p
									) const;


		/** Returns number of (estimated) differential states.
		 *
		 *  \return Number of (estimated) differential states
		 */
		virtual uint getNX( ) const;

		/** Returns number of (estimated) algebraic states.
		 *
		 *  \return Number of (estimated) algebraic states
		 */
		virtual uint getNXA( ) const;

		/** Returns number of controls.
		 *
		 *  \return Number of controls
		 */
		virtual uint getNU( ) const;

		/** Returns number of parameters.
		 *
		 *  \return Number of parameters 
		 */
		virtual uint getNP( ) const;

		/** Returns number of (estimated) disturbances.
		 *
		 *  \return Number of (estimated) disturbances 
		 */
		virtual uint getNW( ) const;

		/** Returns number of process outputs.
		 *
		 *  \return Number of process outputs
		 */
		virtual uint getNY( ) const;


		/** Returns length of the prediction horizon (for the case a predictive control law is used).
		 *
		 *  \return Length of the prediction horizon
		 */
		virtual double getLengthPredictionHorizon( ) const;

		/** Returns length of the control horizon (for the case a predictive control law is used).
		 *
		 *  \return Length of the control horizon
		 */
		virtual double getLengthControlHorizon( ) const;


		/** Returns whether the control law is based on dynamic optimization or 
		 *	a static one.
		 *
		 *  \return BT_TRUE  iff control law is based on dynamic optimization, \n
		 *	        BT_FALSE otherwise
		 */
		virtual BooleanType isDynamic( ) const = 0;

		/** Returns whether the control law is a static one or based on dynamic optimization.
		 *
		 *  \return BT_TRUE  iff control law is a static one, \n
		 *	        BT_FALSE otherwise
		 */
		virtual BooleanType isStatic( ) const = 0;

		/** Returns whether the control law is working in real-time mode.
		 *
		 *  \return BT_TRUE  iff control law is working in real-time mode, \n
		 *	        BT_FALSE otherwise
		 */
		virtual BooleanType isInRealTimeMode( ) const;


	//
	// PROTECTED MEMBER FUNCTIONS:
	//
	protected:



	//
	// DATA MEMBERS:
	//
	protected:

		DVector u;							/**< First piece of time-varying control signals as determined by the control law. */
		DVector p;							/**< Time-constant parameter signals as determined by the control law. */
};


CLOSE_NAMESPACE_ACADO


#include <acado/control_law/control_law.ipp>


#endif  // ACADO_TOOLKIT_CONTROL_LAW_HPP

/*
 *	end of file
 */
