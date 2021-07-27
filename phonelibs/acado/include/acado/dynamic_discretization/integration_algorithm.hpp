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
 *    \file include/acado/dynamic_discretization/integration_algorithm.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */

#ifndef ACADO_TOOLKIT_INTEGRATION_ALGORITHM_HPP
#define ACADO_TOOLKIT_INTEGRATION_ALGORITHM_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/user_interaction.hpp>
#include <acado/dynamic_discretization/shooting_method.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief User-interface to integrate a DynamicSystem, possibly over multiple stages.
 *
 *	\ingroup UserInterfaces
 *
 *  The class IntegrationAlgorithm serves as a user-interface to integrate 
 *  a DynamicSystem, possibly over multiple stages.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class IntegrationAlgorithm : public UserInteraction
{
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor. */
		IntegrationAlgorithm( );

		/** Copy constructor (deep copy). */
		IntegrationAlgorithm( const IntegrationAlgorithm& rhs );

		/** Destructor. */
		virtual ~IntegrationAlgorithm( );

		/** Assignment operator (deep copy). */
		IntegrationAlgorithm& operator=( const IntegrationAlgorithm& rhs );


        /** Set the Differential Equations stage by stage. */
        virtual returnValue addStage( const DynamicSystem  &dynamicSystem_,
                                      const Grid           &stageIntervals,
                                      const IntegratorType &integratorType_ = INT_UNKNOWN );

		/** Set the Transition stages. */
		virtual returnValue addTransition( const Transition& transition_ );


        /** Deletes all stages and transitions and resets the DynamicDiscretization. */
        virtual returnValue clear();


		/** Evaluates the descretized DifferentialEquation at a specified     \n
		*  VariablesGrid. The results are written into the residuum of the   \n
		*  type VariablesGrid. This routine is for a simple evaluation only. \n
		*  If sensitivities are needed use one of the routines below         \n
		*  instead.                                                          \n
		*                                                                    \n
		*  \return SUCCESSFUL_RETURN                                         \n
		*          RET_INVALID_ARGUMENTS                                     \n
		*          or a specific error message form an underlying            \n
		*          discretization instance.                                  \n
		*/
		returnValue evaluate(	VariablesGrid  *x ,      /**< differential states     */
								VariablesGrid  *xa,      /**< algebraic states        */
								VariablesGrid  *p ,      /**< parameters              */
								VariablesGrid  *u ,      /**< controls                */
								VariablesGrid  *w        /**< disturbances            */
								);

		returnValue evaluate(	OCPiterate& _iter );

		returnValue integrate(	VariablesGrid  *x ,      /**< differential states     */
								VariablesGrid  *xa,      /**< algebraic states        */
								VariablesGrid  *p ,      /**< parameters              */
								VariablesGrid  *u ,      /**< controls                */
								VariablesGrid  *w        /**< disturbances            */
								);


		/** Starts the integration of the right hand side at time t0.    \n
		*  If neither the maximum number of allowed iteration is        \n
		*  exceeded nor any  other error occurs the functions stops the \n
		*  integration at time tend.                                    \n
		*  \return SUCCESFUL_RETURN          or                         \n
		*          a error message that depends on the specific         \n
		*          integration routine. (cf. the corresponding header   \n
		*          file that implements the integration routine)        \n
		*/
		returnValue integrate(	double       t0                  /**< the start time              */,
								double       tend                /**< the end time                */,
								const DVector &x0                 /**< the initial state           */,
								const DVector &xa  = emptyVector  /**< the initial algebraic state */,
								const DVector &p   = emptyVector  /**< the parameters              */,
								const DVector &u   = emptyVector  /**< the controls                */,
								const DVector &w   = emptyVector  /**< the disturbance             */
								);

		/** Starts the integration of the right hand side at time t0.    \n
		*  If neither the maximum number of allowed iteration is        \n
		*  exceeded nor any  other error occurs the functions stops the \n
		*  integration at time tend.                                    \n
		*  In addition, results at intermediate grid points can be      \n
		*  stored. Note that these grid points are for storage only and \n
		*  have nothing to do the integrator steps.                     \n
		*                                                               \n
		*  \return SUCCESFUL_RETURN          or                         \n
		*          a error message that depends on the specific         \n
		*          integration routine. (cf. the corresponding header   \n
		*          file that implements the integration routine)        \n
		*/
		returnValue integrate(	const Grid   &t                  /**< the grid [t0,tend]          */,
								const DVector &x0                 /**< the initial state           */,
								const DVector &xa  = emptyVector  /**< the initial algebraic state */,
								const DVector &p   = emptyVector  /**< the parameters              */,
								const DVector &u   = emptyVector  /**< the controls                */,
								const DVector &w   = emptyVector  /**< the disturbance             */
								);


		/** Define a forward seed in form of a block matrix.   \n
		*  Here, the block matrix should have N block rows,   \n
		*  where N is the number of points of the union grid. \n
		*  The i-th row is associated with the i-th grid      \n
		*  point in the union grid. Note that the direction   \n
		*  can itself be organized in sub-blocks as long as   \n
		*  all dimensions fit together.                       \n
		*                                                     \n
		*  \return SUCCESFUL RETURN                           \n
		*          RET_INPUT_OUT_OF_RANGE                     \n
		*/
		returnValue setForwardSeed( const BlockMatrix &xSeed_,   /**< the seed in x-direction */
									const BlockMatrix &pSeed_ = emptyBlockMatrix,   /**< the seed in p-direction */
									const BlockMatrix &uSeed_ = emptyBlockMatrix,   /**< the seed in u-direction */
									const BlockMatrix &wSeed_ = emptyBlockMatrix    /**< the seed in w-direction */ );

		/** Define a forward seed matrix.    \n
		*  \return SUCCESFUL RETURN         \n
		*          RET_INPUT_OUT_OF_RANGE   \n
		*/
		returnValue setForwardSeed(	const DVector &xSeed                /**< the seed w.r.t states       */,
									const DVector &pSeed = emptyVector  /**< the seed w.r.t parameters   */,
									const DVector &uSeed = emptyVector  /**< the seed w.r.t controls     */,
									const DVector &wSeed = emptyVector  /**< the seed w.r.t disturbances */  );


		/**  Defines the first order forward seed to be         \n
		*   the unit-directions matrix.                        \n
		*                                                      \n
		*   \return SUCCESFUL_RETURN                           \n
		*           RET_INPUT_OUT_OF_RANGE                     \n
		*/
		returnValue setUnitForwardSeed();



		/**  Define a backward seed in form of a block matrix.  \n
		*  Here, the block matrix should have N block columns, \n
		*  where N is the number of points of the union grid.  \n
		*  The i-th column is associated with the i-th grid    \n
		*  point in the union grid. Note that the directions   \n
		*  can itself be organized in sub-blocks as long as    \n
		*  all dimensions fit together.                        \n
		*                                                      \n
		*   \return SUCCESFUL_RETURN                           \n
		*           RET_INPUT_OUT_OF_RANGE                     \n
		*/
		returnValue setBackwardSeed( const BlockMatrix &seed    /**< the seed matrix       */ );

		/**  Define a backward seed           \n
		*   \return SUCCESFUL_RETURN         \n
		*           RET_INPUT_OUT_OF_RANGE   \n
		*/
		returnValue setBackwardSeed(	const DVector &seed  /**< the backward seed      */  );

		/**  Defines the first order backward seed to be        \n
		*   a unit matrix.                                     \n
		*                                                      \n
		*   \return SUCCESFUL_RETURN                           \n
		*           RET_INPUT_OUT_OF_RANGE                     \n
		*/
		returnValue setUnitBackwardSeed();



		/** Deletes all seeds that have been set with the methods above.          \n
		*  This function will also give the corresponding memory free.           \n
		*  \return SUCCESSFUL_RETURN                                             \n
		*          RET_NO_SEED_ALLOCATED                                         \n
		*/
		returnValue deleteAllSeeds();


		/** Evaluates the sensitivities.                                      \n
		*                                                                    \n
		*  \return SUCCESSFUL_RETURN                                         \n
		*          RET_NOT_FROZEN                                            \n
		*/
		returnValue evaluateSensitivities( );

		/** Evaluates the sensitivities.                                      \n
		*                                                                    \n
		*  \return SUCCESSFUL_RETURN                                         \n
		*          RET_NOT_FROZEN                                            \n
		*/
		returnValue integrateSensitivities( );

		/** Evaluates the sensitivities and the hessian.  \n
		*                                                \n
		*  \return SUCCESSFUL_RETURN                     \n
		*          RET_NOT_FROZEN                        \n
		*/
		returnValue evaluateSensitivities(	const BlockMatrix &seed,
											BlockMatrix &hessian
											);

		/** Evaluates the sensitivities and the hessian.  \n
		*                                                \n
		*  \return SUCCESSFUL_RETURN                     \n
		*          RET_NOT_FROZEN                        \n
		*/
		returnValue integrateSensitivities(	const BlockMatrix &seed,
											BlockMatrix &hessian
											);


		returnValue unfreeze( );


		BooleanType isAffine( ) const;


		/** Returns the result for the differential states at 
		*  time tend.                                         
		*                                                     
		*  \param xEnd the result for the states at time tend.
		*                                                     
		*  \return SUCCESSFUL_RETURN
		*/
		returnValue getX(	DVector& xEnd
							) const;

		/** Returns the result for the algebraic states at 
		*  time tend.
		*
		*  \param xaEnd the result for the algebraic states at time tend. 
		*
		*  \return SUCCESSFUL_RETURN
		*/
		returnValue getXA(	DVector& xaEnd
							) const;

		/** Returns the requested output on the specified grid. Note     \n
		*  that the output X will be evaluated based on polynomial      \n
		*  interpolation depending on the order of the integrator.      \n
		*                                                               \n
		*  \param X  the differential states on the grid.               \n
		*                                                               \n
		*  \return SUCCESSFUL_RETURN                                    \n
		*          RET_TRAJECTORY_NOT_FROZEN                            \n
		*/
		returnValue getX(	VariablesGrid& X
							) const;


		/** Returns the requested output on the specified grid. Note     \n
		*  that the output X will be evaluated based on polynomial      \n
		*  interpolation depending on the order of the integrator.      \n
		*                                                               \n
		*  \param XA the algebraic states on the grid.                  \n
		*                                                               \n
		*  \return SUCCESSFUL_RETURN                                    \n
		*          RET_TRAJECTORY_NOT_FROZEN                            \n
		*/
		returnValue getXA(	VariablesGrid& XA
							) const;


		/** Returns the result for the forward sensitivities in BlockMatrix form.       \n
		*                                                                               \n
		*  \return SUCCESSFUL_RETURN                                                    \n
		*          RET_INPUT_OUT_OF_RANGE                                               \n
		*/
		returnValue getForwardSensitivities(	BlockMatrix &D  /**< the result for the
															  *   forward sensitivi-
															  *   ties               */
												) const;

		/** Returns the result for the forward sensitivities at the time tend. \n
		*                                                                     \n
		*  \param Dx    the result for the forward sensitivities.             \n
		*                                                                     \n
		*  \return SUCCESSFUL_RETURN                                          \n
		*          RET_INPUT_OUT_OF_RANGE                                     \n
		*/
		 returnValue getForwardSensitivities(	DVector &Dx
												) const;


		/** Returns the result for the backward sensitivities in BlockMatrix form.       \n
		*                                                                               \n
		*  \return SUCCESSFUL_RETURN                                                    \n
		*          RET_INPUT_OUT_OF_RANGE                                               \n
		*/
		returnValue getBackwardSensitivities(	BlockMatrix &D  /**< the result for the
																*   forward sensitivi-
																*   ties              */
												) const;

		/** Returns the result for the backward sensitivities at the time tend. \n
		*                                                                      \n
		*  \param Dx_x0 backward sensitivities w.r.t. the initial states       \n
		*  \param Dx_p  backward sensitivities w.r.t. the parameters           \n
		*  \param Dx_u  backward sensitivities w.r.t. the controls             \n
		*  \param Dx_w  backward sensitivities w.r.t. the disturbance          \n
		*                                                                      \n
		*  \return SUCCESSFUL_RETURN                                           \n
		*          RET_INPUT_OUT_OF_RANGE                                      \n
		*/
		returnValue getBackwardSensitivities(	DVector &Dx_x0,
												DVector &Dx_p = emptyVector,
												DVector &Dx_u = emptyVector,
												DVector &Dx_w = emptyVector
												) const;



    //
    // PROTECTED MEMBER FUNCTIONS:
    //

    protected:

		virtual returnValue setupOptions( );
		virtual returnValue setupLogging( );


    //
    // PROTECTED MEMBERS:
    //

    protected:

        ShootingMethod* integrationMethod;
		OCPiterate		iter;
};


CLOSE_NAMESPACE_ACADO


//#include <acado/dynamic_discretization/integration_algorithm.ipp>


#endif  // ACADO_TOOLKIT_INTEGRATION_ALGORITHM_HPP

// end of file
