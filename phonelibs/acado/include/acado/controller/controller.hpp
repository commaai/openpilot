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
 *    \file include/acado/controller/controller.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_CONTROLLER_HPP
#define ACADO_TOOLKIT_CONTROLLER_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/simulation_environment/simulation_block.hpp>

#include <acado/matrix_vector/matrix_vector.hpp>

#include <acado/control_law/control_law.hpp>
#include <acado/estimator/estimator.hpp>
#include <acado/reference_trajectory/reference_trajectory.hpp>
#include <acado/reference_trajectory/static_reference_trajectory.hpp>




BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Calculates the control inputs of the Process based on the Process outputs.
 *
 *	\ingroup UserInterfaces
 *
 *  The class Controller is one of the two main building-blocks within the 
 *  SimulationEnvironment and complements the Process. It contains an 
 *	online control law (e.g. a DynamicFeedbackLaw comprising a RealTimeAlgorithm) 
 *	for obtaining the control inputs of the Process.
 *
 *	A state/parameter estimator as well as a ReferenceTrajectory can optionally be 
 *	used to provide estimated quantities and a reference values to the control law.
 *	The reference trajectory can either be specified beforehand as member of the 
 *	Controller or, alternatively, provided at each step in order to allow for
 *	reference trajectories that can be adapted online.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class Controller : public SimulationBlock
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:
        /** Default constructor. 
		 */
        Controller( );

		/** Constructor which takes a control law, an estimator and a 
		 *	reference trajectory for computing the control/parameter signals.
		 *
		 *	@param[in] _controlLaw				Control law to be used for computing the control/parameter signals.
		 *	@param[in] _estimator				Estimator for estimating quantities required by the control law based on the process output.
		 *	@param[in] _referenceTrajectory		Reference trajectory to be used by the control law.
		 */
        Controller(	ControlLaw& _controlLaw,
					Estimator& _estimator,
					ReferenceTrajectory& _referenceTrajectory = emptyReferenceTrajectory
					);

		/** Constructor which takes a control law and a reference trajectory 
		 *	for computing the control/parameter signals.
		 *
		 *	@param[in] _controlLaw				Control law to be used for computing the control/parameter signals.
		 *	@param[in] _referenceTrajectory		Reference trajectory to be used by the control law.
		 */
        Controller(	ControlLaw& _controlLaw,
					ReferenceTrajectory& _referenceTrajectory = emptyReferenceTrajectory
					);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
        Controller(	const Controller& rhs
					);

        /** Destructor. 
		 */
        virtual ~Controller( );

        /** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
        Controller& operator=(	const Controller& rhs
								);


		/** Assigns new control law to be used for computing control/parameter signals.
		 *
		 *	@param[in]  _controlLaw		New control law.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue setControlLaw(	ControlLaw& _controlLaw
									);

		/** Assigns new estimator for estimating quantities required by the control law 
		 *	based on the process output.
		 *
		 *	@param[in]  _estimator		New estimator.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue setEstimator(	Estimator& _estimator
									);

		/** Assigns new reference trajectory to be used by the control law.
		 *
		 *	@param[in]  _referenceTrajectory	New reference trajectory.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue setReferenceTrajectory(	ReferenceTrajectory& _referenceTrajectory
											);


		/** Initializes algebraic states of the control law.
		 *
		 *	@param[in]  _xa_init	Initial value for algebraic states.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue initializeAlgebraicStates(	const VariablesGrid& _xa_init
												);

		/** Initializes algebraic states of the control law from data file.
		 *
		 *	@param[in]  fileName	Name of file containing initial value for algebraic states.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_FILE_CAN_NOT_BE_OPENED
		 */
		returnValue initializeAlgebraicStates(	const char* fileName
												);


		/** Initializes the controller with given start values and 
		 *	performs a number of consistency checks.
		 *
		 *	@param[in]  _startTime	Start time.
		 *	@param[in]  _x0			Initial value for differential states.
		 *	@param[in]  _p			Initial value for parameters.
		 *	@param[in]  _yRef		Initial value for reference trajectory.
		 *
		 *	\note If a non-empty reference trajectory is provided, this one is used
		 *	      instead of the possibly set-up build-in one.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_CONTROLLER_INIT_FAILED, \n
		 *	        RET_NO_CONTROLLAW_SPECIFIED, \n
		 *	        RET_BLOCK_DIMENSION_MISMATCH
		 */
        virtual returnValue init(	double startTime = 0.0,
									const DVector& _x0 = emptyConstVector,
									const DVector& _p  = emptyConstVector,
									const VariablesGrid& _yRef = emptyConstVariablesGrid
									);


		/** Performs next step of the contoller based on given inputs.
		 *
		 *	@param[in]  currentTime	Current time.
		 *	@param[in]  _y			Most recent process output.
		 *	@param[in]  _yRef		Current piece of reference trajectory or piece of reference trajectory for next step (required for hotstarting).
		 *
		 *	\note If a non-empty reference trajectory is provided, this one is used
		 *	      instead of the possibly set-up build-in one.
		 * 
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_CONTROLLER_STEP_FAILED, \n
		 *	        RET_NO_CONTROLLAW_SPECIFIED
		 */
        virtual returnValue step(	double currentTime,
									const DVector& _y,
									const VariablesGrid& _yRef = emptyConstVariablesGrid
									);

		/** Performs next step of the contoller based on given inputs.
		 *
		 *	@param[in]  currentTime	Current time.
		 *	@param[in]  dim			Dimension of process output.
		 *	@param[in]  _y			Most recent process output.
		 *	@param[in]  _yRef		Current piece of reference trajectory or piece of reference trajectory for next step (required for hotstarting).
		 *
		 *	\note If a non-empty reference trajectory is provided, this one is used
		 *	      instead of the possibly set-up build-in one.
		 * 
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_CONTROLLER_STEP_FAILED, \n
		 *	        RET_NO_CONTROLLAW_SPECIFIED
		 */
		virtual returnValue step(	double currentTime,
									uint dim,
									const double* const _y,
									const VariablesGrid& _yRef = emptyConstVariablesGrid
									);

		/** Performs next feedback step of the contoller based on given inputs.
		 *
		 *	@param[in]  currentTime	Current time.
		 *	@param[in]  _y			Most recent process output.
		 *	@param[in]  _yRef		Current piece of reference trajectory (if not specified during previous preparationStep).
		 *
		 *	\note If a non-empty reference trajectory is provided, this one is used
		 *	      instead of the possibly set-up build-in one.
		 * 
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_CONTROLLER_STEP_FAILED, \n
		 *	        RET_NO_CONTROLLAW_SPECIFIED
		 */
        virtual returnValue feedbackStep(	double currentTime,
											const DVector& _y,
											const VariablesGrid& _yRef = emptyConstVariablesGrid
											);

		/** Performs next preparation step of the contoller based on given inputs.
		 *
		 *	@param[in]  nextTime	Time at next step.
		 *	@param[in]  _yRef		Piece of reference trajectory for next step (required for hotstarting).
		 *
		 *	\note If a non-empty reference trajectory is provided, this one is used
		 *	      instead of the possibly set-up build-in one.
		 * 
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_CONTROLLER_STEP_FAILED, \n
		 *	        RET_NO_CONTROLLAW_SPECIFIED
		 */
        virtual returnValue preparationStep(	double nextTime = 0.0,
												const VariablesGrid& _yRef = emptyConstVariablesGrid
												);

		virtual returnValue obtainEstimates(	double currentTime,
												const DVector& _y,
												DVector& xEst,
												DVector& pEst
												);


		/** Returns computed control signals.
		 *
		 *	@param[out]  _y			Computed control signals.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
        inline returnValue getU(	DVector& _u
									) const;

		/** Returns computed parameter signals.
		 *
		 *	@param[out]  _y			Computed parameter signals.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
        inline returnValue getP(	DVector& _p
									) const;


		/** Returns number of process outputs expected by the controller.
		 *
		 *	\return Number of process outputs
		 */
		inline uint getNY( ) const;

		/** Returns number of control signals computed by the controller.
		 *
		 *	\return Number of control signals
		 */
		inline uint getNU( ) const;

		/** Returns number of parameter signals computed by the controller.
		 *
		 *	\return Number of parameter signals
		 */
		inline uint getNP( ) const;


		/** Returns whether controller comprises a dynamic control law.
		 *
		 *  \return BT_TRUE  iff controller comprises a dynamic control law, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasDynamicControlLaw( ) const;

		/** Returns whether controller comprises a static control law.
		 *
		 *  \return BT_TRUE  iff controller comprises a static control law, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasStaticControlLaw( ) const;

		/** Returns whether controller comprises an estimator.
		 *
		 *  \return BT_TRUE  iff controller comprises an estimator, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasEstimator( ) const;

		/** Returns whether controller comprises a build-in reference trajectory.
		 *
		 *  \return BT_TRUE  iff controller comprises a build-in reference trajectory, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasReferenceTrajectory( ) const;


		/** Returns sampling time of control law.
		 *
		 *  \return Sampling time of control law.
		 */
		inline double getSamplingTimeControlLaw( );

		/** Returns sampling time of estimator.
		 *
		 *  \return Sampling time of estimator.
		 */
		inline double getSamplingTimeEstimator( );


		/** Determines next sampling instant of controller based on the 
		 *	sampling times of control law and estimator
		 *
		 *	@param[in]  currentTime		Current time.
		 *
		 *  \return Next sampling instant of controller.
		 */
		double getNextSamplingInstant(	double currentTime
										);

		/** Returns previous real runtime of the controller 
		 *	(e.g. for determining computational delay).
		 *
		 *  \return Previous real runtime of the controller.
		 */
		inline double getPreviousRealRuntime( );


		/** Enables the controller.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		inline returnValue enable( );

		/** Disables the controller (i.e. initial values kept and no steps are performed).
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		inline returnValue disable( );



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


		/** Returns current piece of the reference trajectory starting at given time.
		 *
		 *	@param[in]  tStart	Start time of reference piece.
		 *	@param[out] _yRef	Current piece of the reference trajectory.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		virtual returnValue getCurrentReference(	double tStart,
													VariablesGrid& _yRef
													) const;


    //
    // DATA MEMBERS:
    //
    protected:
		
		ControlLaw* controlLaw;						/**< Control law (usually including a dynamic optimizer) to be used for computing the control/parameter signals. */
		Estimator* estimator;						/**< Estimator for estimating quantities required by the control law based on the process output. */
		ReferenceTrajectory* referenceTrajectory;	/**< Reference trajectory to be used by the control law. */
		
		BooleanType isEnabled;						/**< Flag indicating whether controller is enabled or not. */
		
		RealClock controlLawClock;					/**< Clock required to determine runtime of control law. */
};



CLOSE_NAMESPACE_ACADO



#include <acado/controller/controller.ipp>


#endif  // ACADO_TOOLKIT_CONTROLLER_HPP

/*
 *	end of file
 */
