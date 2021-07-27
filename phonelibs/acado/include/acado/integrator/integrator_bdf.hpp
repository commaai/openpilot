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
 *    \file include/acado/integrator/integrator_bdf.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 */


#ifndef ACADO_TOOLKIT_INTEGRATOR_BDF_HPP
#define ACADO_TOOLKIT_INTEGRATOR_BDF_HPP

#include <acado/integrator/integrator_fwd.hpp>

BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Implements the backward-differentiation formula for integrating DAEs.
 *
 *	\ingroup NumericalAlgorithms
 *
 *  The class IntegratorBDF implements the backward-differentiation formula
 *	for integrating differential-algebraic equations (DAEs).
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class IntegratorBDF : public Integrator{


friend class SimulationByIntegration;
friend class ShootingMethod;

//
// PUBLIC MEMBER FUNCTIONS:
//

public:

    /** Default constructor. */
    IntegratorBDF( );

    /** Default constructor. */
    IntegratorBDF( const DifferentialEquation &rhs_ );

    /** Copy constructor (deep copy). */
    IntegratorBDF( const IntegratorBDF& arg );

    /** Destructor. */
    virtual ~IntegratorBDF( );

    /** Assignment operator (deep copy). */
    virtual IntegratorBDF& operator=( const IntegratorBDF& arg );


    /** The (virtual) copy constructor */
    virtual Integrator* clone() const;


   // ================================================================================

    /** The initialization routine which takes the right-hand side of \n
     *  the differential equation to be integrated.                   \n
     *                                                                \n
     *  \param rhs  the right-hand side of the ODE/DAE.               \n
     *                                                                \n
     *  \return SUCCESSFUL_RETURN   if all dimension checks succeed.  \n
     *          otherwise: integrator dependent error message.        \n
     */
    virtual returnValue init( const DifferentialEquation &rhs_ );


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
    inline returnValue init( const DifferentialEquation &rhs_,
                             const Transition           &trs_ );


    // ================================================================================

    /** Freezes the mesh: Storage of the step sizes. If the integrator is     \n
     *  freezed the mesh will be stored when calling the function integrate   \n
     *  for the first time. If the function integrate is called more than     \n
     *  once the same mesh will be reused (i.e. the step size control will    \n
     *  be turned  off). Note that the mesh should be frozen if any kind of   \n
     *  sensitivity generation is used.                                       \n
     *  \return SUCCESSFUL_RETURN                                             \n
     *          RET_ALREADY_FROZEN                                            \n
     */
    virtual returnValue freezeMesh();


    /** Freezes the mesh as well as all intermediate values. This function    \n
     *  is necessary for the case that automatic differentiation in backward  \n
     *  mode should is used. (Note: This function might for large right hand  \n
     *  sides lead to memory problems as all intemediate values will be       \n
     *  stored!)                                                              \n
     *  \return SUCCESSFUL_RETURN                                             \n
     *          RET_ALREADY_FROZEN                                            \n
     */
    virtual returnValue freezeAll();


    /** Unfreezes the mesh: Gives the memory free that has previously  \n
     *  been allocated by "freeze". If you use the function            \n
     *  integrate after unfreezing the usual step size control will be \n
     *  switched on.                                                   \n
     *  \return SUCCESSFUL_RETURN                                      \n
     *          RET_ALREADY_UNFROZED                                   \n
     */
    virtual returnValue unfreeze();


    // ================================================================================


    /** Executes the next single step. This function can be used to    \n
     *  call the integrator step wise. Note that this function is e.g. \n
     *  useful in real-time simulations where after each step a time   \n
     *  out limit has to be checked. This function will usually return \n
     *  \return RET_FINAL_STEP_NOT_PERFORMED_YET                       \n
     *          for the case that the final step has not been          \n
     *          performed yet, i.e. the integration routine is not yet \n
     *          at the time tend.                                      \n
     *          Otherwise it will either return                        \n
     *  \return SUCCESSFUL_RETURN                                      \n
     *          or any other error message that might occur during     \n
     *          an integration step.                                   \n
     *
     *  In most real situations you can define the maximum number of   \n
     *  step sizes to be 1 before calling the function integrate       \n
     *  Then the function integrate should return after one step with  \n
     *  the message                                                    \n
     *  RET_MAXIMUM_NUMBER_OF_STEPS_EXCEEDED. After that you can call  \n
     *  step() until the final time is reached.                        \n
     *  (You can use the PrintLevel NONE to avoid that the message  \n
     *  RET_MAXIMUM_NUMBER_OF_STEPS_EXCEEDED is printed.)              \n
     */
    virtual returnValue step( int number  /**< the step number */ );


    /** Stops the integration even if the final time has not been  \n
     *  reached yet. This function will also give all memory free. \n
     *  In particular, the function unfreeze() will be called.     \n
     *  (This function is designed for the usage in real-time      \n
     *   contexts in order to deal with error messages without     \n
     *   deleting and re-initializing the integrator.)             \n
     *  \return SUCCESSFUL_RETURN                                  \n
     */
    virtual returnValue stop();


    /** Sets an initial guess for the differential state derivatives \n
     *  (consistency condition)                                      \n
     *  \return SUCCESSFUL_RETURN                                    \n
     */
    virtual returnValue setDxInitialization( double *dx0 /**< initial guess
                                                          *   for the differential
                                                          *   state derivatives
                                                          */  );



    // ================================================================================



    /**  Returns the number of accepted Steps.                                       \n
     *   \return The requested number of accepted steps.                             \n
     */
    virtual int getNumberOfSteps() const;


    /**  Returns the number of rejected Steps.                                       \n
     *   \return The requested number of rejected steps.                             \n
     */
    virtual int getNumberOfRejectedSteps() const;



    /** Returns the current step size */
    virtual double getStepSize() const;



//
// PROTECTED MEMBER FUNCTIONS:
//
protected:



    /** Starts integration: cf. integrate(...) for  \n
      * more details.                               \n
      */
    virtual returnValue evaluate( const DVector &x0    /**< the initial state           */,
                                  const DVector &xa    /**< the initial algebraic state */,
                                  const DVector &p     /**< the parameters              */,
                                  const DVector &u     /**< the controls                */,
                                  const DVector &w     /**< the disturbance             */,
                                  const Grid   &t_    /**< the time interval           */  );

    // ================================================================================


    /**< Integrates forward and/or backward depending on the specified seeds. \n
      *  
      *  \return SUCCESSFUL_RETURN                                            \n
      *          RET_NO_SEED_ALLOCATED                                        \n
      */
    virtual returnValue evaluateSensitivities();


    // ================================================================================


    /** Define a forward seed matrix.    \n
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
                                                                          *  seed.              */ );


    // ================================================================================


    /**  Define a backward seed           \n
     *   \return SUCCESFUL_RETURN         \n
     *           RET_INPUT_OUT_OF_RANGE   \n
     */
    virtual returnValue setProtectedBackwardSeed(  const DVector &seed   /**< the seed  matrix  */,
                                                   const int    &order  /**< the order of the
                                                                         *  seed.              */  );


    // ================================================================================



    /** Returns the result for the state at the time tend.                           \n
     *  \return SUCCESSFUL_RETURN                                                    \n
     */
    virtual returnValue getProtectedX(           DVector *xEnd /**< the result for the
                                                               *  states at the time
                                                               *  tend.              */ ) const;


    /** Returns the result for the forward sensitivities at the time tend.           \n
     *  \return SUCCESSFUL_RETURN                                                    \n
     *          RET_INPUT_OUT_OF_RANGE                                               \n
     */
    virtual returnValue getProtectedForwardSensitivities( DMatrix *Dx  /**< the result for the
                                                                       *   forward sensitivi-
                                                                       *   ties               */,
                                                          int order   /**< the order          */ ) const;



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
                                                           int order      ) const;


    // ================================================================================

    /** Returns the dimension of the Differential Equation */
    virtual int getDim() const;


    /** Returns the number of Dynamic Equations in the Differential Equation */
    virtual int getDimX() const;


    /** inializes the Butcher Tableau. (only for internal use) */
    void initializeButcherTableau();

    /** computes start values using a diagonal implicit RK-method.         \n
     *  \return SUCCESSFUL_RETURN
     *
     */
    returnValue rk_start();


    /** starts the newton iteration for the implicit computation of k      \n
     *  \return SUCCESSFUL_RETURN                                          \n
     */
    returnValue rk_start_solve( int stepnumber );


    /** determines the predictor and starts the corrector iterations.      \n
     *  If a corrector is found a local error estimate will be returned.   \n
     *  \return The local error estimate                                   \n
     */
    double determinePredictor( int number, BooleanType ini );


    /** determines the predictor and corrector polynom                     \n
     *  \return SUCCESFUL_RETURN                                           \n
     *          RET_INFO_UNDEFINED                                         \n
     */
    returnValue determineCorrector( int number, BooleanType ini );


    /** prints intermediate results of the BDF steps.                      \n
     */
    void printBDFIntermediateResults();


    /** prints final results of the BDF steps.                             \n
     */
    void printBDFfinalResults();


    /** prints final results of the BDF steps.                             \n
     */
    void printBDFfinalResults2(DMatrix &div);


    /** prints intermediate results of the Runge Kutta starter             \n
     */
    void printRKIntermediateResults();


    /** Decomposes the Jacobian J.               \n
     *  \return SUCCESSFUL_RETURN                \n
     *          RET_THE_DAE_INDEX_IS_TOO_LARGE   \n
     */
    returnValue decomposeJacobian(int index, DMatrix &J );


    /** applies a newton step                                              \n
     *  \return the norm of the increment                                  \n
     */
    double applyNewtonStep( int index, double *etakplus1, const double *etak, const DMatrix &J, const double *FFF );


    /** applies the transpose of M (needed for automatic differentiation   \n
     *  in backward mode)                                                  \n
     *  \return (void)                                                     \n
     */
    void applyMTranspose( int index, double *seed1, DMatrix &J, double *seed2 );


    /** Initializes a second forward seed. (only for internal use)         \n
     */
    returnValue setForwardSeed2( const DVector &xSeed           /**< the seed w.r.t the
                                                                 *  initial states     */,
                                 const DVector &pSeed           /**< the seed w.r.t the
                                                                 *  parameters         */,
                                 const DVector &uSeed           /**< the seed w.r.t the
                                                                 *  controls           */,
                                 const DVector &wSeed           /**< the seed w.r.t the
                                                                 *  disturbances       */   );


    /** Initializes a second backward seed. (only for internal use)        \n
     */
    virtual returnValue setBackwardSeed2( const DVector &seed     /**< the seed
                                                                  *   matrix     */ );


    /** Differentiation of the RK-Starter: \n
     */
    void determineRKEtaGForward();


    /** Differentiation of the RK-Starter: \n
     */
    void determineRKEtaHBackward();


    /** Differentiation of the RK-Starter: \n
     */
    void determineRKEtaGForward2();


    /** Differentiation of the RK-Starter: \n
     */
    void determineRKEtaHBackward2();


    /** Differentiation of the BDF step:   \n
     */
    void determineBDFEtaGForward( int number );

    /** Differentiation of the BDF step:   \n
     */
    void determineBDFEtaHBackward( int number );

    /** Differentiation of the BDF step:   \n
     */
    void determineBDFEtaGForward2( int number );

    /** Differentiation of the BDF step:   \n
     */
    void determineBDFEtaHBackward2( int number );


    /** Delete everything.                 \n
     */
    void deleteAll();


    /** Protected implementation of the copy constructor. \n
     */
    void constructAll( const IntegratorBDF& arg );



    /** This routine is protected and sets up all   \n
     *  variables (i.e. allocates memory etc.).     \n
     *  Note that this routine assumes that the     \n
     *  dimensions are already set correctly and is \n
     *  thus for internal use only.                 \n
     */
    void allocateMemory( );


    /** This routine is protected and is basically used       \n
     *  to set all pointer-valued member to the NULL pointer. \n
     *  In addition some dimensions are initialized with 0 as \n
     *  a default value.
     */
    void initializeVariables();


    /** Relax the algebraic equations */
    void relaxAlgebraic( double *residuum, double timePoint );


    void prepareDividedDifferences( DMatrix &div );

    void interpolate( int number_, DMatrix &div, VariablesGrid &poly );

	void logCurrentIntegratorStep(	const DVector& currentX  = emptyConstVector,
									const DVector& currentXA = emptyConstVector
									);


    void copyBackward( DVector       &Dx_x0,
                       DVector       &Dx_p ,
                       DVector       &Dx_u ,
                       DVector       &Dx_w ,
                       const DMatrix &div    ) const;


// DATA MEMBERS:
//
protected:


    // STORAGE OF THE ALGEBRAIC RESIDUUM :
    // -----------------------------------
    double *initialAlgebraicResiduum;
    double        relaxationConstant;


    // BUTCHER-
    // TABLEAU:
    // ----------
    int      dim               ;  /**< the dimension of the Butcher Tableau.              */
    double **A                 ;  /**< the coefficient A of the Butcher Tableau.          */
    double  *b4                ;  /**< the 4th order coefficients of the Butcher Tableau. */
    double  *b5                ;  /**< the 5th order coefficients of the Butcher Tableau. */
    double  *c                 ;  /**< the time coefficients of the Butcher Tableau.      */


    // RK-STARTER:
    // -----------
    double   *eta4             ;  /**< the result of order 4                              */
    double   *eta5             ;  /**< the result of order 5                              */
    double   *eta4_            ;  /**< the result of order 4                              */
    double   *eta5_            ;  /**< the result of order 5                              */
    double ***k                ;  /**< the intermediate results                           */
    double ***k2               ;  /**< the intermediate results                           */
    double ***l                ;  /**< the intermediate results                           */
    double ***l2               ;  /**< the intermediate results                           */
    double   *iseed            ;  /**< the intermediate seeds                             */


    // BDF-METHOD:
    // -----------
    int      nstep             ; /**< the horizon length of the multistep method          */
    double **psi               ; /**< the time differences (multi-step-sizes).            */
    double  *psi_              ; /**< the time differences (multi-step-sizes). (backup)   */
    double **gamma             ; /**< intermediate coefficients                           */
    DMatrix   nablaY            ; /**< the devided differences                             */
    DMatrix   nablaY_           ; /**< the devided differences (backup)                    */
    DMatrix   phi               ; /**< the modified divided differences                    */
    DMatrix   delta             ; /**< sums over the modified divided differences          */
    DVector   c2                ; /**< the constant in the derivative of the corrector     \n
                                  *   polynom.                                            */

    DMatrix **M                 ; /**< the Jacobians for Newton's method                   */
//    std::vector< Eigen::HouseholderQR< DMatrix::Base > > qr;
    int     *M_index           ; /**< the index of the inverse approximation              */
    int      nOfM              ; /**< number of distinct inverse Jacobian approximations  */
    int      maxNM             ; /**< number of allocated Jacobian storage positions      */

    int     *nOfNewtonSteps    ; /**< the number of newton steps (for each BDF-step)      */
    double **eta               ; /**< the predictor and corrector approximations          */
    double **eta2              ; /**< the predictor and corrector approximations          */
    double   t                 ; /**< the actual time                                     */
    double  *x                 ; /**< the actual state (only internal use)                */

    double  *F                 ;
    double  *F2                ;


    // OTHERS:
    // -------
    double *initial_guess      ;  /**< an initial guess for dot_x and xa                  */
    double  rel_time_scale     ;  /**< relative time scale (intermediate variable)        */


    // SENSITIVITIES:
    // --------------
    int        ndir            ;  /**< number of dependecy directions (only internal use) */

    DVector     fseed           ;  /**< The forward seed (only internal use)               */
    DVector     bseed           ;  /**< The backward seed (only internal use)              */

    DVector     fseed2          ;  /**< The forward seed 2 (only internal use)             */
    DVector     bseed2          ;  /**< The backward seed 2 (only internal use)            */

    double    *G               ;  /**< Sensitivity matrix (only internal use)             */
    double    *etaG            ;  /**< Sensitivity matrix (only internal use)             */
    DMatrix     nablaG          ;  /**< Sensitivity matrix (only internal use)             */
    DMatrix     phiG            ;  /**< Sensitivity matrix (only internal use)             */
    DMatrix     deltaG          ;  /**< Sensitivity matrix (only internal use)             */
    DVector     c2G             ;  /**< Sensitivity matrix (only internal use)             */

    double    *G2              ;  /**< Sensitivity matrix (only internal use)             */
    double    *G3              ;  /**< Sensitivity matrix (only internal use)             */
    double    *etaG2           ;  /**< Sensitivity matrix (only internal use)             */
    double    *etaG3           ;  /**< Sensitivity matrix (only internal use)             */
    DMatrix     nablaG2         ;  /**< Sensitivity matrix (only internal use)             */
    DMatrix     nablaG3         ;  /**< Sensitivity matrix (only internal use)             */
    DMatrix     phiG2           ;  /**< Sensitivity matrix (only internal use)             */
    DMatrix     phiG3           ;  /**< Sensitivity matrix (only internal use)             */
    DMatrix     deltaG2         ;  /**< Sensitivity matrix (only internal use)             */
    DMatrix     deltaG3         ;  /**< Sensitivity matrix (only internal use)             */
    DVector     c2G2            ;  /**< Sensitivity matrix (only internal use)             */
    DVector     c2G3            ;  /**< Sensitivity matrix (only internal use)             */

    double    *H               ;  /**< Sensitivity matrix (only internal use)             */
    double   **etaH            ;  /**< Sensitivity matrix (only internal use)             */
    DMatrix     nablaH          ;  /**< Sensitivity matrix (only internal use)             */
    DMatrix     nablaH_         ;  /**< Sensitivity matrix (only internal use)             */
    DMatrix     deltaH          ;  /**< Sensitivity matrix (only internal use)             */
    DVector     c2H             ;  /**< Sensitivity matrix (only internal use)             */
    double  ***kH              ;  /**< Sensitivity matrix (only internal use)             */
    double   **zH              ;  /**< Sensitivity matrix (only internal use)             */

    double    *H2              ;  /**< Sensitivity matrix (only internal use)             */
    double    *H3              ;  /**< Sensitivity matrix (only internal use)             */
    double   **etaH2           ;  /**< Sensitivity matrix (only internal use)             */
    double   **etaH3           ;  /**< Sensitivity matrix (only internal use)             */
    DMatrix     nablaH2         ;  /**< Sensitivity matrix (only internal use)             */
    DMatrix     nablaH3         ;  /**< Sensitivity matrix (only internal use)             */
    DMatrix     nablaH2_        ;  /**< Sensitivity matrix (only internal use)             */
    DMatrix     nablaH3_        ;  /**< Sensitivity matrix (only internal use)             */
    DMatrix     deltaH2         ;  /**< Sensitivity matrix (only internal use)             */
    DMatrix     deltaH3         ;  /**< Sensitivity matrix (only internal use)             */
    DVector     c2H2            ;  /**< Sensitivity matrix (only internal use)             */
    DVector     c2H3            ;  /**< Sensitivity matrix (only internal use)             */
    double  ***kH2             ;  /**< Sensitivity matrix (only internal use)             */
    double   **zH2             ;  /**< Sensitivity matrix (only internal use)             */
    double  ***kH3             ;  /**< Sensitivity matrix (only internal use)             */
    double   **zH3             ;  /**< Sensitivity matrix (only internal use)             */


    // STORAGE:
    // --------
    int maxAlloc               ;  /**< size of the memory that is allocated to store      \n
                                   *   the trajectory and the mesh.                       */


    // STATISTICS:
    // -----------
    RealClock jacComputation    ;
    RealClock jacDecomposition  ;
    RealClock correctorTime     ;
    int       nJacEvaluations   ;

};


CLOSE_NAMESPACE_ACADO



#include <acado/integrator/integrator_bdf.ipp>


#endif  // ACADO_TOOLKIT_INTEGRATOR_BDF_HPP

// end of file.
