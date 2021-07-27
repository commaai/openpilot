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
 *	\file include/acado/control_law/linear_state_feedback.hpp
 *	\author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_LINEAR_STATE_FEEDBACK_HPP
#define ACADO_TOOLKIT_LINEAR_STATE_FEEDBACK_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/function/function.hpp>

#include <acado/control_law/control_law.hpp>
#include <acado/control_law/clipping_functionality.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Implements a linear state feedback law to be used within a Controller.
 *
 *	\ingroup UserInterfaces
 *
 *  The class LinearStateFeedback implements a linear state feedback law to be used 
 *	within a Controller.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class LinearStateFeedback : public ControlLaw, public ClippingFunctionality
{
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:
		/** Default constructor. 
		 */
		LinearStateFeedback( );

		/** Constructor which takes the gain matrix of the linear feedback law
		 *	together with the sampling time.
		 *
		 *	@param[in] _K				Linear feedback gain matrix.
		 *	@param[in] _samplingTime	Sampling time.
		 */
		LinearStateFeedback(	const DMatrix& _K,
								double _samplingTime = DEFAULT_SAMPLING_TIME
								);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		LinearStateFeedback(	const LinearStateFeedback& rhs
								);

		/** Destructor. 
		 */
		virtual ~LinearStateFeedback( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		LinearStateFeedback& operator=(	const LinearStateFeedback& rhs
										);

		/** Clone constructor (deep copy).
		 *
		 *	\return Pointer to deep copy of base class type
		 */
		virtual ControlLaw* clone( ) const;


		/** Initializes the feedback law with given start values and 
		 *	performs a number of consistency checks.
		 *
		 *	@param[in]  _startTime	Start time.
		 *	@param[in]  _x			Initial value for differential states.
		 *	@param[in]  _p			Initial value for parameters.
		 *	@param[in]  _yRef		Initial value for reference trajectory.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue init(	double startTime = 0.0,
									const DVector &x0_ = emptyConstVector,
									const DVector &p_ = emptyConstVector,
									const VariablesGrid& _yRef = emptyConstVariablesGrid
									);


		/** Performs next step of the feedback law based on given inputs.
		 *
		 *	@param[in]  currentTime	Current time.
		 *	@param[in]  _x			Most recent value for differential states.
		 *	@param[in]  _p			Most recent value for parameters.
		 *	@param[in]  _yRef		Current piece of reference trajectory.
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



	//
	// PROTECTED MEMBER FUNCTIONS:
	//
	protected:



	//
	// DATA MEMBERS:
	//
	protected:

		DMatrix K;						/**< Linear state feedback gain matrix. */

};


CLOSE_NAMESPACE_ACADO



//#include <acado/control_law/linear_state_feedback.ipp>


#endif  // ACADO_TOOLKIT_LINEAR_STATE_FEEDBACK_HPP

/*
 *	end of file
 */
