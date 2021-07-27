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
 *    \file include/acado/integrator/integrator.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 */


#ifndef ACADO_TOOLKIT_INTEGRATOR_HPP
#define ACADO_TOOLKIT_INTEGRATOR_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/algorithmic_base.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/variables_grid/variables_grid.hpp>
#include <acado/symbolic_expression/symbolic_expression.hpp>
#include <acado/function/function.hpp>

#include <acado/integrator/integrator_fwd.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Abstract base class for all kinds of algorithms for integrating differential equations (ODEs or DAEs).
 *
 *	\ingroup AlgorithmInterfaces
 *
 *  The class Integrator serves as an abstract base class for all kinds of algorithms
 *	for integrating differential equations (ODEs or DAEs).
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class Integrator : public AlgorithmicBase
{

	friend class SimulationByIntegration;
	friend class ShootingMethod;

	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:


		/** Default constructor. 
		*/
		Integrator( );

		/** Copy constructor. */
		Integrator( const Integrator &arg );

		/** Default destructor. */
		virtual ~Integrator( );

		/** The (virtual) copy constructor */
		virtual Integrator* clone() const = 0;


	// ================================================================================


		/** The initialization routine which takes the right-hand side of \n
		*  the differential equation to be integrated.                   \n
		*                                                                \n
		*  \param rhs  the right-hand side of the ODE/DAE.               \n
		*                                                                \n
		*  \return SUCCESSFUL_RETURN   if all dimension checks succeed.  \n
		*          otherwise: integrator dependent error message.        \n
		*/
		virtual returnValue init( const DifferentialEquation &rhs ) = 0;



		/** The initialization routine which takes the right-hand side of \n
		*  the differential equation to be integrated. In addition a     \n
		*  transition function can be set which is evaluated at the end  \n
		*  of the integration interval.                                  \n
		*                                                                \n
		*  \param rhs  the right-hand side of the ODE/DAE.               \n
		*  \param trs  the transition to be evaluated at the end.        \n
		*                                                                \n
		*  \return SUCCESSFUL_RETURN   if all dimension checks succeed.  \n
		*          otherwise: integrator dependent error message.        \n
		*/
		returnValue init(	const DifferentialEquation &rhs,
							const Transition           &trs
							);


		returnValue setTransition(	const Transition &trs
									);



		// ================================================================================

		/** Freezes the mesh: Storage of the step sizes. If the integrator is     \n
		*  freezed, the mesh will be stored when calling the function integrate  \n
		*  for the first time. If the function integrate is called more than     \n
		*  once the same mesh will be reused (i.e. the step size control will    \n
		*  be turned off). Note that the mesh should be frozen if any kind of    \n
		*  sensitivity generation is used.                                       \n
		*  \return SUCCESSFUL_RETURN                                             \n
		*          RET_ALREADY_FROZEN                                            \n
		*/
		virtual returnValue freezeMesh() = 0;


		/** Freezes the mesh as well as all intermediate values. This function    \n
		*  is necessary for the case that automatic differentiation in backward  \n
		*  mode should is used. (Note: This function might for large right hand  \n
		*  sides lead to memory problems as all intemediate values will be       \n
		*  stored!)                                                              \n
		*  \return SUCCESSFUL_RETURN                                             \n
		*          RET_ALREADY_FROZEN                                            \n
		*/
		virtual returnValue freezeAll() = 0;


		/** Unfreezes the mesh: Gives the memory free that has previously  \n
		*  been allocated by "freeze". If you use the function            \n
		*  integrate after unfreezing the usual step size control will be \n
		*  switched on.                                                   \n
		*  \return SUCCESSFUL_RETURN                                      \n
		*          RET_MESH_ALREADY_UNFREEZED                                \n
		*/
		virtual returnValue unfreeze() = 0;



		// ================================================================================

		/** Starts the integration of the right hand side at time t0.    \n
		*  If neither the maximum number of allowed iteration is        \n
		*  exceeded nor any  other error occurs the functions stops the \n
		*  integration at time tend.                                    \n
		*  \return SUCCESFUL_RETURN          or                         \n
		*          a error message that depends on the specific         \n
		*          integration routine. (cf. the corresponding header   \n
		*          file that implements the integration routine)        \n
		*/
		returnValue integrate(	double t0        /**< the start time              */,
								double tend      /**< the end time                */,
								double *x0       /**< the initial state           */,
								double *xa  = 0  /**< the initial algebraic state */,
								double *p   = 0  /**< the parameters              */,
								double *u   = 0  /**< the controls                */,
								double *w   = 0  /**< the disturbance             */  );



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
		returnValue integrate(	const Grid   &t          /**< the grid [t0,tend]          */,
								double      *x0          /**< the initial state           */,
								double      *xa  = 0     /**< the initial algebraic state */,
								double      *p   = 0     /**< the parameters              */,
								double      *u   = 0     /**< the controls                */,
								double      *w   = 0     /**< the disturbance             */  );



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
								const DVector &w   = emptyVector  /**< the disturbance             */  );



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
								const DVector &w   = emptyVector  /**< the disturbance             */  );



		// ================================================================================


		/** Define a forward seed matrix.    \n
		*  \return SUCCESFUL RETURN         \n
		*          RET_INPUT_OUT_OF_RANGE   \n
		*/
		returnValue setForwardSeed(	const int    &order                /**< the order of the seed.      */,
									const DVector &xSeed                /**< the seed w.r.t states       */,
									const DVector &pSeed = emptyVector  /**< the seed w.r.t parameters   */,
									const DVector &uSeed = emptyVector  /**< the seed w.r.t controls     */,
									const DVector &wSeed = emptyVector  /**< the seed w.r.t disturbances */  );



		// ================================================================================


		/**  Define a backward seed           \n
		*   \return SUCCESFUL_RETURN         \n
		*           RET_INPUT_OUT_OF_RANGE   \n
		*/
		returnValue setBackwardSeed(	const int    &order /**< the order of the seed. */,
										const DVector &seed  /**< the backward seed      */  );



		// ================================================================================

		/** Deletes all seeds that have been set with the methods above.          \n
		*  This function will also give the corresponding memory free.           \n
		*  \return SUCCESSFUL_RETURN                                             \n
		*          RET_NO_SEED_ALLOCATED                                         \n
		*/
		virtual returnValue deleteAllSeeds();


		// ================================================================================


		/**< Integrates forward and/or backward depending on the specified seeds. \n
		*  
		*  \return SUCCESSFUL_RETURN                                            \n
		*          RET_NO_SEED_ALLOCATED                                        \n
		*/
		returnValue integrateSensitivities( );



		/** Sets an initial guess for the differential state derivatives \n
		*  (consistency condition)                                      \n
		*  \return SUCCESSFUL_RETURN                                    \n
		*/
		virtual returnValue setDxInitialization( double *dx0 /**< initial guess
															*   for the differential
															*   state derivatives
															*/  ) = 0;


		// ================================================================================


		/** Returns the result for the differential states at the   \n
		*  time tend.                                              \n
		*                                                          \n
		*  \param xEnd the result for the states at the time tend. \n
		*                                                          \n
		*  \return SUCCESSFUL_RETURN                               \n
		*/
		inline  returnValue getX( DVector &xEnd ) const;


		/** Returns the result for the algebraic states at the                 \n
		*  time tend.                                                         \n
		*                                                                     \n
		*  \param xaEnd the result for the algebraic states at the time tend. \n
		*                                                                     \n
		*  \return SUCCESSFUL_RETURN                                          \n
		*/
		inline  returnValue getXA( DVector &xaEnd ) const;



		/** Returns the requested output on the specified grid. Note     \n
		*  that the output X will be evaluated based on polynomial      \n
		*  interpolation depending on the order of the integrator.      \n
		*                                                               \n
		*  \param X  the differential states on the grid.               \n
		*                                                               \n
		*  \return SUCCESSFUL_RETURN                                    \n
		*          RET_TRAJECTORY_NOT_FROZEN                            \n
		*/
		inline  returnValue getX( VariablesGrid &X ) const;


		/** Returns the requested output on the specified grid. Note     \n
		*  that the output X will be evaluated based on polynomial      \n
		*  interpolation depending on the order of the integrator.      \n
		*                                                               \n
		*  \param XA the algebraic states on the grid.                  \n
		*                                                               \n
		*  \return SUCCESSFUL_RETURN                                    \n
		*          RET_TRAJECTORY_NOT_FROZEN                            \n
		*/
		inline  returnValue getXA( VariablesGrid &XA ) const;


		/** Returns the requested output on the specified grid.  \n
		*  The intermediate states constructed based on linear  \n
		*  interpolation.                                       \n
		*                                                       \n
		*  \param I  the intermediates states on the grid.      \n
		*                                                       \n
		*  \return SUCCESSFUL_RETURN                            \n
		*          RET_TRAJECTORY_NOT_FROZEN                    \n
		*/
		inline  returnValue getI( VariablesGrid &I ) const;



		/** Returns the result for the forward sensitivities at the time tend. \n
		*                                                                     \n
		*  \param Dx    the result for the forward sensitivities.             \n
		*  \param order the order.                                            \n
		*                                                                     \n
		*  \return SUCCESSFUL_RETURN                                          \n
		*          RET_INPUT_OUT_OF_RANGE                                     \n
		*/
		 returnValue getForwardSensitivities(	DVector &Dx,
												int order ) const;


		/** Returns the result for the forward sensitivities on the grid.  \n
		*                                                                 \n
		*  \param Dx    the result for the forward sensitivities.         \n
		*  \param order the order.                                        \n
		*                                                                 \n
		*  \return SUCCESSFUL_RETURN                                      \n
		*          RET_INPUT_OUT_OF_RANGE                                 \n
		*/
		returnValue getForwardSensitivities(	VariablesGrid &Dx,
												int order ) const;


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
												DVector &Dx_p ,
												DVector &Dx_u ,
												DVector &Dx_w ,
												int order      ) const;

		// ================================================================================




		/**  Returns the number of accepted Steps.                                       \n
		*   \return The requested number of accepted steps.                             \n
		*/
		virtual int getNumberOfSteps() const = 0;


		/**  Returns the number of rejected Steps.                                       \n
		*   \return The requested number of rejected steps.                             \n
		*/
		virtual int getNumberOfRejectedSteps() const = 0;



		/**  Returns if integrator is able to handle implicit switches.                  \n
		*   \return BT_TRUE:  if integrator can handle implicit switches.               \n
		*           BT_FALSE: otherwise
		*/
		virtual BooleanType canHandleImplicitSwitches( ) const;


		/**  Returns if the differential equation of the integrator is defined.          \n
		*   \return BT_TRUE:  if differential equation is defined.                      \n
		*           BT_FALSE: otherwise
		*/
		virtual BooleanType isDifferentialEquationDefined( ) const;	


		/**  Returns if the differential equation of the integrator is affine.           \n
		*   \return BT_TRUE:  if differential equation is affine.                       \n
		*           BT_FALSE: otherwise
		*/
		virtual BooleanType isDifferentialEquationAffine( ) const;

		/** Returns the */
		virtual double getDifferentialEquationSampleTime() const;


		/** Returns the current step size */
		virtual double getStepSize() const = 0;


		/** Prints the run-time profile. This routine \n
		*  can be used after an integration run in   \n
		*  order to assess the performance.          \n
		*/
		virtual returnValue printRunTimeProfile() const;


		/**< Integrates forward and/or backward depending on the specified seeds. \n
		*  
		*  \return SUCCESSFUL_RETURN                                            \n
		*          RET_NO_SEED_ALLOCATED                                        \n
		*/
		virtual returnValue evaluateSensitivities() = 0;


	//
	// PROTECTED MEMBER FUNCTIONS:
	//
	protected:

		virtual returnValue setupOptions( );


		// ================================================================================


		/** Starts integration: cf. integrate(...) for  \n
		* more details.                                                \n
		*/
		virtual returnValue evaluate( const DVector &x0    /**< the initial state           */,
									const DVector &xa    /**< the initial algebraic state */,
									const DVector &p     /**< the parameters              */,
									const DVector &u     /**< the controls                */,
									const DVector &w     /**< the disturbance             */,
									const Grid   &t_    /**< the time interval           */  ) = 0;

		/** Evaluates the transtion (protected, only for internal use).
		*/
		virtual returnValue evaluateTransition( const double time   /**< the time            */,
													DVector &xd    /**< the state           */,
												const DVector &xa    /**< the algebraic state */,
												const DVector &p     /**< the parameters      */,
												const DVector &u     /**< the controls        */,
												const DVector &w     /**< the disturbance     */  );


		virtual returnValue diffTransitionForward(       DVector &DX,
												const DVector &DP,
												const DVector &DU,
												const DVector &DW,
												const int    &order );


		virtual returnValue diffTransitionBackward( DVector &DX,
													DVector &DP,
													DVector &DU,
													DVector &DW,
													int    &order );


		// ================================================================================


		/** Define a forward seed.           \n
		*  \return SUCCESFUL RETURN         \n
		*          RET_INPUT_OUT_OF_RANGE   \n
		*/
		virtual returnValue setProtectedForwardSeed( const DVector &xSeed     /**< the seed w.r.t the
																			*  initial states     */,
													const DVector &pSeed     /**< the seed w.r.t the
																			*  parameters         */,
													const DVector &uSeed     /**< the seed w.r.t the
																			*  controls           */,
													const DVector &wSeed     /**< the seed w.r.t the
																			*  disturbances       */,
													const int    &order     /**< the order of the
																			*  seed.              */ ) = 0;

		// ================================================================================


		/**  Define a backward seed           \n
		*   \return SUCCESFUL_RETURN         \n
		*           RET_INPUT_OUT_OF_RANGE   \n
		*/
		virtual returnValue setProtectedBackwardSeed(  const DVector &seed    /**< the seed
																			*   matrix     */,
													const int    &order   /**< the order of the
																			*  seed.              */  ) = 0;


		// ================================================================================



		/** Returns the result for the state at the time tend.                           \n
		*  \return SUCCESSFUL_RETURN                                                    \n
		*/
		virtual returnValue getProtectedX( DVector *xEnd /**< the result for the
														*  states at the time
														*  tend.              */ ) const = 0;


		/** Returns the result for the forward sensitivities at the time tend.           \n
		*  \return SUCCESSFUL_RETURN                                                    \n
		*          RET_INPUT_OUT_OF_RANGE                                               \n
		*/
		virtual returnValue getProtectedForwardSensitivities( DMatrix *Dx  /**< the result for the
																		*   forward sensitivi-
																		*   ties               */,
															int order   /**< the order          */ ) const = 0;


		/** Returns the result for the backward sensitivities at the time tend. \n
		*                                                                      \n
		*  \param Dx_x0 backward sensitivities w.r.t. the initial states       \n
		*  \param Dx_p  backward sensitivities w.r.t. the parameters           \n
		*  \param Dx_u  backward sensitivities w.r.t. the controls             \n
		*  \param Dx_w  backward sensitivities w.r.t. the disturbance          \n
		*  \param order the order of the derivative                            \n
		*                                                                      \n
		*  \return SUCCESSFUL_RETURN                                           \n
		*          RET_INPUT_OUT_OF_RANGE                                      \n
		*/
		virtual returnValue getProtectedBackwardSensitivities( DVector &Dx_x0,
															DVector &Dx_p ,
															DVector &Dx_u ,
															DVector &Dx_w ,
															int order      ) const = 0;


		// ================================================================================



		/** Returns the dimension of the Differential Equation */
		virtual int getDim() const = 0;


		/** Returns the number of Dynamic Equations in the Differential Equation */
		virtual int getDimX() const;



		/** Initializes the options. This routine should be called by the \n
		*  integrators derived from this class in order to initialize    \n
		*  the following members based on the user options:              \n
		*
		*  maxNumberOfSteps
		*  hmin
		*  hmax
		*  tune
		*  TOL
		*  las
		*  (cf. SETTINGS  for more details)
		*/
		void initializeOptions();


		virtual returnValue setupLogging( );


	//
	// DATA MEMBERS:
	//
	protected:


		// DIFFERENTIAL ALGEBRAIC RHS:
		// ---------------------------
		DifferentialEquation *rhs    ;  /**< the right-hand side to be integrated               */
		short int             m      ;  /**< the dimension of the right-hand side               */
		short int             ma     ;  /**< the number of algebraic states                     */
		short int             mdx    ;  /**< the dimension of differential state derivatives    */
		short int             mn     ;  /**< the number of intermediate states in the rhs       */
		short int             mu     ;  /**< the number of controls                             */
		short int             mui    ;  /**< the number of integer controls                     */
		short int             mp     ;  /**< the number of parameters                           */
		short int             mpi    ;  /**< the number of integer parameters                   */
		short int             mw     ;  /**< the number of disturbances                         */
		short int             md     ;  /**< number of differential states                      */


		// TRANSITION:
		// ---------------------------
		Transition        *transition;  /**< the transition to be evaluated at switches         */



		// SETTINGS:
		// ---------
		double  *h                   ;  /**< the initial step size = h[0]                       */
		double   hini                ;  /**< storage of the initial step size                   */
		double   hmin                ;  /**< the minimum step size                              */
		double   hmax                ;  /**< the maximum step size                              */
		double   tune                ;  /**< tuning parameter for the step size control.        */
		double   TOL                 ;  /**< the integration tolerance                          */
		int      las                 ;  /** the type of linear algebra solver to be used        */

		Grid     timeInterval        ;  /**< the time interval                                  */


		// INTERNAL INDEX LISTS:
		// ---------------------
		int    *diff_index         ;  /**< the index list for the differential states            */
		int    *ddiff_index        ;  /**< the index list for the differential state derivatives */
		int    *alg_index          ;  /**< the index list for the algebraic states               */
		int    *control_index      ;  /**< the index list for the controls                       */
		int    *parameter_index    ;  /**< the index list for the parameters                     */
		int    *int_control_index  ;  /**< the index list for the integer controls               */
		int    *int_parameter_index;  /**< the index list for the integer parameters             */
		int    *disturbance_index  ;  /**< the index list for the disturbances                   */
		int     time_index         ;  /**< the time index                                        */


		// OTHERS:
		// -------------------------
		int     maxNumberOfSteps   ;  /**< the maximum number of integrator steps.            */
		int     count              ;  /**< a counter of the (accepted) steps                  */
		int     count2             ;  /**< number of steps after integration                  */
		int     count3             ;  /**< number of steps after integration                  */
		DVector  diff_scale         ;  /**< the scale of the differential states               */


		// PRINT-LEVEL:
		// -------------------------
		int PrintLevel             ;  /**< The PrintLevel (default: LOW)                   */


		// SEED DIMENSIONS:
		// -------------------------
		int        nFDirs          ;  /**< The number of forward directions                   */
		int        nBDirs          ;  /**< The number of backward directions                  */
		int        nFDirs2         ;  /**< The number of second order forward directions      */
		int        nBDirs2         ;  /**< The number of second order backward directions     */


		// THE STATE OF AGGREGATION:
		// -------------------------
		StateOfAggregation soa     ;  /**< the state of aggregation                           */



		// STATISTICS:
		// --------------------------
		RealClock totalTime         ;
		RealClock functionEvaluation;
		int       nFcnEvaluations   ;


		DVector                    xE;

		DVector                    dX;
		DVector                    dP;
		DVector                    dU;
		DVector                    dW;

		DVector                   dXb;
		DVector                   dPb;
		DVector                   dUb;
		DVector                   dWb;

		VariablesGrid         xStore;
		VariablesGrid        dxStore;
		VariablesGrid       ddxStore;
		VariablesGrid         iStore;


};


CLOSE_NAMESPACE_ACADO


#include <acado/integrator/integrator.ipp>


// collect all remaining headers of integrator directory
#include <acado/integrator/integrator_runge_kutta.hpp>
#include <acado/integrator/integrator_runge_kutta12.hpp>
#include <acado/integrator/integrator_runge_kutta23.hpp>
#include <acado/integrator/integrator_runge_kutta45.hpp>
#include <acado/integrator/integrator_runge_kutta78.hpp>
#include <acado/integrator/integrator_discretized_ode.hpp>
#include <acado/integrator/integrator_bdf.hpp>
#include <acado/integrator/integrator_lyapunov.hpp>
#include <acado/integrator/integrator_lyapunov45.hpp>


#endif  // ACADO_TOOLKIT_INTEGRATOR_HPP

// end of file.
