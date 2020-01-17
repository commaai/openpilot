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
 *	\file include/acado/optimization_algorithm/real_time_algorithm.hpp
 *	\author Hans Joachim Ferreau, Boris Houska
 *
 */


#ifndef ACADO_TOOLKIT_REAL_TIME_ALGORITHM_HPP
#define ACADO_TOOLKIT_REAL_TIME_ALGORITHM_HPP


#include <acado/optimization_algorithm/optimization_algorithm.hpp>
#include <acado/control_law/control_law.hpp>


BEGIN_NAMESPACE_ACADO



/** 
 *	\brief User-interface to formulate and solve model predictive control problems.
 *
 *	\ingroup UserInterfaces
 *
 *	The class RealTimeAlgorithm serves as a user-interface to formulate and
 *  solve model predictive control problems.
 *
 *  \author Hans Joachim Ferreau, Boris Houska
 */
class RealTimeAlgorithm : public OptimizationAlgorithmBase, public ControlLaw
{
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor. 
		 */
		RealTimeAlgorithm( );

		/** Constructor which takes the optimal control problem to be solved online
		 *	together with the sampling time.
		 *
		 *	@param[in] ocp_				Optimal control problem to be solved online.
		 *	@param[in] _samplingTime	Sampling time.
		 */
		RealTimeAlgorithm(	const OCP& ocp_,
							double _samplingTime = DEFAULT_SAMPLING_TIME
							);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		RealTimeAlgorithm(	const RealTimeAlgorithm& rhs
							);

		/** Destructor. 
		 */
		virtual ~RealTimeAlgorithm( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		RealTimeAlgorithm& operator=(	const RealTimeAlgorithm& rhs
										);

		/** Clone constructor (deep copy).
		 *
		 *	\return Pointer to deep copy of base class type
		 */
		virtual ControlLaw* clone( ) const;


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


		/** Initializes the (internal) optimization algorithm part of the RealTimeAlgorithm.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_OPTALG_INIT_FAILED
		 */
		virtual returnValue init( );

		/** Initializes the control law with given start values and 
		 *	performs a number of consistency checks.
		 *
		 *	@param[in]  _startTime	Start time.
		 *	@param[in]  _x			Initial value for differential states.
		 *	@param[in]  _p			Initial value for parameters.
		 *	@param[in]  _yRef		Initial value for reference trajectory.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue init(	double startTime,
									const DVector &_x = emptyConstVector,
									const DVector &_p = emptyConstVector,
									const VariablesGrid& _yRef = emptyConstVariablesGrid
									);


		/** Performs next step of the control law based on given inputs.
		 *
		 *	@param[in]  currentTime	Current time.
		 *	@param[in]  _x			Most recent value for differential states.
		 *	@param[in]  _p			Most recent value for parameters.
		 *	@param[in]  _yRef		Current piece of reference trajectory or piece of reference trajectory for next step (required for hotstarting).
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue step(	double currentTime,
									const DVector& _x,
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
		 *	\note If a non-empty reference trajectory is provided, this one is used
		 *	      instead of the possibly set-up build-in one.
		 * 
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue feedbackStep(	double currentTime,
											const DVector &_x,
											const DVector &_p = emptyConstVector,
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


		/** (not yet documented).
		 *
		 *	@param[in]  .		.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue solve(	double startTime,
									const DVector &_x,
									const DVector &_p = emptyConstVector,
									const VariablesGrid& _yRef = emptyConstVariablesGrid
									);


		/** Shifts the data for preparating the next real-time step.
		 *
		 *	\return RET_NOT_YET_IMPLEMENTED
		 */
		virtual returnValue shift(	double timeShift = -1.0
									);


		/** Assigns new reference trajectory for the next real-time step.
		 *
		 *	@param[in]  ref		Current piece of new reference trajectory.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_REFERENCE_SHIFTING_WORKS_FOR_LSQ_TERMS_ONLY, \n
		 *	        RET_MEMBER_NOT_INITIALISED
		 */
		virtual returnValue setReference(	const VariablesGrid &ref
											);

		
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
		virtual BooleanType isDynamic( ) const;

		/** Returns whether the control law is a static one or based on dynamic optimization.
		 *
		 *  \return BT_TRUE  iff control law is a static one, \n
		 *	        BT_FALSE otherwise
		 */
		virtual BooleanType isStatic( ) const;

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

		/** Frees memory of all members of the real time parameters.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue clear( );

		/** (not yet documented).
		 *
		 *	@param[in]  .		.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue allocateNlpSolver(	Objective *F,
												DynamicDiscretization *G,
												Constraint *H
												);

		/** (not yet documented).
		 *
		 *	@param[in]  .		.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue initializeNlpSolver(	const OCPiterate& userInit
													);

		/** (not yet documented).
		 *
		 *	@param[in]  .		.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue initializeObjective(	Objective *F
													);


		/** (not yet documented).
		 *
		 *	@param[in]  .		.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue performFeedbackStep(	double currentTime,
											const DVector &_x,
											const DVector &_p = emptyConstVector
											);

		/** (not yet documented).
		 *
		 *	@param[in]  .		.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue performPreparationStep(	const VariablesGrid& _yRef = emptyConstVariablesGrid,
											BooleanType isLastIteration = BT_TRUE
											);


	//
	// DATA MEMBERS:
	//
	protected:

		DVector* x0;						/**< Deep copy of the most recent initial value of differential states. */
		DVector* p0;						/**< Deep copy of the most recent parameter. */

		VariablesGrid* reference;		/**< Deep copy of the most recent reference. */

};


CLOSE_NAMESPACE_ACADO



#include <acado/optimization_algorithm/real_time_algorithm.ipp>


#endif  // ACADO_TOOLKIT_REAL_TIME_ALGORITHM_HPP

/*
 *   end of file
 */
