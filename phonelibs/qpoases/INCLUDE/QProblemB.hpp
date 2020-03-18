/*
 *	This file is part of qpOASES.
 *
 *	qpOASES -- An Implementation of the Online Active Set Strategy.
 *	Copyright (C) 2007-2008 by Hans Joachim Ferreau et al. All rights reserved.
 *
 *	qpOASES is free software; you can redistribute it and/or
 *	modify it under the terms of the GNU Lesser General Public
 *	License as published by the Free Software Foundation; either
 *	version 2.1 of the License, or (at your option) any later version.
 *
 *	qpOASES is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *	Lesser General Public License for more details.
 *
 *	You should have received a copy of the GNU Lesser General Public
 *	License along with qpOASES; if not, write to the Free Software
 *	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


/**
 *	\file INCLUDE/QProblemB.hpp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Declaration of the QProblemB class which is able to use the newly
 *	developed online active set strategy for parametric quadratic programming
 *	for problems with (simple) bounds only.
 */



#ifndef QPOASES_QPROBLEMB_HPP
#define QPOASES_QPROBLEMB_HPP


#include <Bounds.hpp>



class SolutionAnalysis;

/** Class for setting up and solving quadratic programs with (simple) bounds only.
 *	The main feature is the possibily to use the newly developed online active set strategy
 *	for parametric quadratic programming.
 *
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 */
class QProblemB
{
	/* allow SolutionAnalysis class to access private members */
	friend class SolutionAnalysis;
	
	/*
	 *	PUBLIC MEMBER FUNCTIONS
	 */
	public:
		/** Default constructor. */
		QProblemB( );

		/** Constructor which takes the QP dimension only. */
		QProblemB(	int _nV						/**< Number of variables. */
					);

		/** Copy constructor (deep copy). */
		QProblemB(	const QProblemB& rhs	/**< Rhs object. */
					);

		/** Destructor. */
		~QProblemB( );

		/** Assignment operator (deep copy). */
		QProblemB& operator=(	const QProblemB& rhs	/**< Rhs object. */
								);


		/** Clears all data structures of QProblemB except for QP data.
		 *	\return SUCCESSFUL_RETURN \n
					RET_RESET_FAILED */
		returnValue reset( );


		/** Initialises a QProblemB with given QP data and solves it
		 *	using an initial homotopy with empty working set (at most nWSR iterations).
		 *	\return SUCCESSFUL_RETURN \n
					RET_INIT_FAILED \n
					RET_INIT_FAILED_CHOLESKY \n
					RET_INIT_FAILED_HOTSTART \n
					RET_INIT_FAILED_INFEASIBILITY \n
					RET_INIT_FAILED_UNBOUNDEDNESS \n
					RET_MAX_NWSR_REACHED \n
					RET_INVALID_ARGUMENTS \n
					RET_INACCURATE_SOLUTION \n
		 			RET_NO_SOLUTION */
		returnValue init(	const real_t* const _H, 		/**< Hessian matrix. */
							const real_t* const _g,			/**< Gradient vector. */
							const real_t* const _lb,		/**< Lower bounds (on variables). \n
																If no lower bounds exist, a NULL pointer can be passed. */
							const real_t* const _ub,		/**< Upper bounds (on variables). \n
																If no upper bounds exist, a NULL pointer can be passed. */
							int& nWSR, 						/**< Input: Maximum number of working set recalculations when using initial homotopy. \n
																Output: Number of performed working set recalculations. */
							const real_t* const yOpt = 0,	/**< Initial guess for dual solution vector. */
				 			real_t* const cputime = 0		/**< Output: CPU time required to initialise QP. */
							);


		/** Initialises a QProblemB with given QP data and solves it
		 *	using an initial homotopy with empty working set (at most nWSR iterations).
		 *	\return SUCCESSFUL_RETURN \n
					RET_INIT_FAILED \n
					RET_INIT_FAILED_CHOLESKY \n
					RET_INIT_FAILED_HOTSTART \n
					RET_INIT_FAILED_INFEASIBILITY \n
					RET_INIT_FAILED_UNBOUNDEDNESS \n
					RET_MAX_NWSR_REACHED \n
					RET_INVALID_ARGUMENTS \n
					RET_INACCURATE_SOLUTION \n
		 			RET_NO_SOLUTION */
		returnValue init(	const real_t* const _H, 		/**< Hessian matrix. */
							const real_t* const _R, 		/**< Cholesky factorization of the Hessian matrix. */
							const real_t* const _g,			/**< Gradient vector. */
							const real_t* const _lb,		/**< Lower bounds (on variables). \n
																If no lower bounds exist, a NULL pointer can be passed. */
							const real_t* const _ub,		/**< Upper bounds (on variables). \n
																If no upper bounds exist, a NULL pointer can be passed. */
							int& nWSR, 						/**< Input: Maximum number of working set recalculations when using initial homotopy. \n
																Output: Number of performed working set recalculations. */
							const real_t* const yOpt = 0,	/**< Initial guess for dual solution vector. */
				 			real_t* const cputime = 0		/**< Output: CPU time required to initialise QP. */
							);


		/** Solves an initialised QProblemB using online active set strategy.
		 *	\return SUCCESSFUL_RETURN \n
					RET_MAX_NWSR_REACHED \n
					RET_HOTSTART_FAILED_AS_QP_NOT_INITIALISED \n
					RET_HOTSTART_FAILED \n
					RET_SHIFT_DETERMINATION_FAILED \n
					RET_STEPDIRECTION_DETERMINATION_FAILED \n
					RET_STEPLENGTH_DETERMINATION_FAILED \n
					RET_HOMOTOPY_STEP_FAILED \n
					RET_HOTSTART_STOPPED_INFEASIBILITY \n
					RET_HOTSTART_STOPPED_UNBOUNDEDNESS \n
					RET_INACCURATE_SOLUTION \n
		 			RET_NO_SOLUTION */
		returnValue hotstart(	const real_t* const g_new,	/**< Gradient of neighbouring QP to be solved. */
								const real_t* const lb_new,	/**< Lower bounds of neighbouring QP to be solved. \n
													 			 If no lower bounds exist, a NULL pointer can be passed. */
								const real_t* const ub_new,	/**< Upper bounds of neighbouring QP to be solved. \n
													 			 If no upper bounds exist, a NULL pointer can be passed. */
								int& nWSR,					/**< Input: Maximum number of working set recalculations; \n
																 Output: Number of performed working set recalculations. */
								real_t* const cputime		/**< Output: CPU time required to solve QP (or to perform nWSR iterations). */
								);


		/** Returns Hessian matrix of the QP (deep copy).
		  *	\return SUCCESSFUL_RETURN */
		inline returnValue getH(	real_t* const _H	/**< Array of appropriate dimension for copying Hessian matrix.*/
									) const;

		/** Returns gradient vector of the QP (deep copy).
		  *	\return SUCCESSFUL_RETURN */
		inline returnValue getG(	real_t* const _g	/**< Array of appropriate dimension for copying gradient vector.*/
									) const;

		/** Returns lower bound vector of the QP (deep copy).
		  *	\return SUCCESSFUL_RETURN */
		inline returnValue getLB(	real_t* const _lb	/**< Array of appropriate dimension for copying lower bound vector.*/
									) const;

		/** Returns single entry of lower bound vector of the QP.
		  *	\return SUCCESSFUL_RETURN \n
					RET_INDEX_OUT_OF_BOUNDS */
		inline returnValue getLB(	int number,		/**< Number of entry to be returned. */
									real_t& value	/**< Output: lb[number].*/
									) const;

		/** Returns upper bound vector of the QP (deep copy).
		  *	\return SUCCESSFUL_RETURN */
		inline returnValue getUB(	real_t* const _ub	/**< Array of appropriate dimension for copying upper bound vector.*/
									) const;

		/** Returns single entry of upper bound vector of the QP.
		  *	\return SUCCESSFUL_RETURN \n
					RET_INDEX_OUT_OF_BOUNDS */
		inline returnValue getUB(	int number,		/**< Number of entry to be returned. */
									real_t& value	/**< Output: ub[number].*/
									) const;


		/** Returns current bounds object of the QP (deep copy).
		  *	\return SUCCESSFUL_RETURN */
		inline returnValue getBounds(	Bounds* const _bounds	/** Output: Bounds object. */
										) const;


		/** Returns the number of variables.
		 *	\return Number of variables. */
		inline int getNV( ) const;

		/** Returns the number of free variables.
		 *	\return Number of free variables. */
		inline int getNFR( );

		/** Returns the number of fixed variables.
		 *	\return Number of fixed variables. */
		inline int getNFX( );

		/** Returns the number of implicitly fixed variables.
		 *	\return Number of implicitly fixed variables. */
		inline int getNFV( ) const;

		/** Returns the dimension of null space.
		 *	\return Dimension of null space. */
		int getNZ( );


		/** Returns the optimal objective function value.
		 *	\return finite value: Optimal objective function value (QP was solved) \n
		 			+infinity:	  QP was not yet solved */
		real_t getObjVal( ) const;

		/** Returns the objective function value at an arbitrary point x.
		 *	\return Objective function value at point x */
		real_t getObjVal(	const real_t* const _x	/**< Point at which the objective function shall be evaluated. */
							) const;

		/** Returns the primal solution vector.
		 *	\return SUCCESSFUL_RETURN \n
					RET_QP_NOT_SOLVED */
		returnValue getPrimalSolution(	real_t* const xOpt			/**< Output: Primal solution vector (if QP has been solved). */
										) const;

		/** Returns the dual solution vector.
		 *	\return SUCCESSFUL_RETURN \n
					RET_QP_NOT_SOLVED */
		returnValue getDualSolution(	real_t* const yOpt	/**< Output: Dual solution vector (if QP has been solved). */
										) const;


		/** Returns status of the solution process.
		 *	\return Status of solution process. */
		inline QProblemStatus getStatus( ) const;


		/** Returns if the QProblem object is initialised.
		 *	\return BT_TRUE:  QProblemB initialised \n
		 			BT_FALSE: QProblemB not initialised */
		inline BooleanType isInitialised( ) const;

		/** Returns if the QP has been solved.
		 *	\return BT_TRUE:  QProblemB solved \n
		 			BT_FALSE: QProblemB not solved */
		inline BooleanType isSolved( ) const;

		/** Returns if the QP is infeasible.
		 *	\return BT_TRUE:  QP infeasible \n
		 			BT_FALSE: QP feasible (or not known to be infeasible!) */
		inline BooleanType isInfeasible( ) const;

		/** Returns if the QP is unbounded.
		 *	\return BT_TRUE:  QP unbounded \n
		 			BT_FALSE: QP unbounded (or not known to be unbounded!) */
		inline BooleanType isUnbounded( ) const;


		/** Returns the print level.
		 *	\return Print level. */
		inline PrintLevel getPrintLevel( ) const;

		/** Changes the print level.
 		 *	\return SUCCESSFUL_RETURN */
		returnValue setPrintLevel(	PrintLevel _printlevel	/**< New print level. */
									);


		/** Returns Hessian type flag (type is not determined due to this call!).
		 *	\return Hessian type. */
		inline HessianType getHessianType( ) const;

		/** Changes the print level.
 		 *	\return SUCCESSFUL_RETURN */
		inline returnValue setHessianType(	HessianType _hessianType /**< New Hessian type. */
											);


	/*
	 *	PROTECTED MEMBER FUNCTIONS
	 */
	protected:
		/** Checks if Hessian happens to be the identity matrix,
		 *  and sets corresponding status flag (otherwise the flag remains unaltered!).
		 *	\return SUCCESSFUL_RETURN */
		returnValue checkForIdentityHessian( );

		/** Determines type of constraints and bounds (i.e. implicitly fixed, unbounded etc.).
		 *	\return SUCCESSFUL_RETURN \n
					RET_SETUPSUBJECTTOTYPE_FAILED */
		returnValue setupSubjectToType( );

		/** Computes the Cholesky decomposition R of the (simply projected) Hessian (i.e. R^T*R = Z^T*H*Z).
		 *  It only works in the case where Z is a simple projection matrix!
		 *	\return SUCCESSFUL_RETURN \n
		 *			RET_INDEXLIST_CORRUPTED */
		returnValue setupCholeskyDecomposition( );


		/** Solves a QProblemB whose QP data is assumed to be stored in the member variables.
		 *  A guess for its primal/dual optimal solution vectors and the corresponding
		 *  optimal working set can be provided.
		 *	\return SUCCESSFUL_RETURN \n
					RET_INIT_FAILED \n
					RET_INIT_FAILED_CHOLESKY \n
					RET_INIT_FAILED_HOTSTART \n
					RET_INIT_FAILED_INFEASIBILITY \n
					RET_INIT_FAILED_UNBOUNDEDNESS \n
					RET_MAX_NWSR_REACHED */
		returnValue solveInitialQP(	const real_t* const xOpt,			/**< Optimal primal solution vector.
																		 *	 A NULL pointer can be passed. */
									const real_t* const yOpt,			/**< Optimal dual solution vector.
																		 *	 A NULL pointer can be passed. */
									const Bounds* const guessedBounds,	/**< Guessed working set for solution (xOpt,yOpt).
																		 *	 A NULL pointer can be passed. */
									int& nWSR, 							/**< Input: Maximum number of working set recalculations; \n
																 		 *	 Output: Number of performed working set recalculations. */
									real_t* const cputime				/**< Output: CPU time required to solve QP (or to perform nWSR iterations). */
									);


		/** Obtains the desired working set for the auxiliary initial QP in
		 *  accordance with the user specifications
		 *	\return SUCCESSFUL_RETURN \n
					RET_OBTAINING_WORKINGSET_FAILED \n
					RET_INVALID_ARGUMENTS */
		returnValue obtainAuxiliaryWorkingSet(	const real_t* const xOpt,			/**< Optimal primal solution vector.
																					 *	 If a NULL pointer is passed, all entries are assumed to be zero. */
												const real_t* const yOpt,			/**< Optimal dual solution vector.
																					 *	 If a NULL pointer is passed, all entries are assumed to be zero. */
												const Bounds* const guessedBounds,	/**< Guessed working set for solution (xOpt,yOpt). */
												Bounds* auxiliaryBounds				/**< Input: Allocated bound object. \n
																					 *	 Ouput: Working set for auxiliary QP. */
												) const;

		/** Setups bound data structure according to auxiliaryBounds.
		 *  (If the working set shall be setup afresh, make sure that
		 *  bounds data structure has been resetted!)
		 *	\return SUCCESSFUL_RETURN \n
					RET_SETUP_WORKINGSET_FAILED \n
					RET_INVALID_ARGUMENTS \n
					RET_UNKNOWN BUG */
		returnValue setupAuxiliaryWorkingSet(	const Bounds* const auxiliaryBounds,	/**< Working set for auxiliary QP. */
												BooleanType setupAfresh					/**< Flag indicating if given working set shall be
																						 *    setup afresh or by updating the current one. */
												);

		/** Setups the optimal primal/dual solution of the auxiliary initial QP.
		 *	\return SUCCESSFUL_RETURN */
		returnValue setupAuxiliaryQPsolution(	const real_t* const xOpt,			/**< Optimal primal solution vector.
																				 	*	 If a NULL pointer is passed, all entries are set to zero. */
												const real_t* const yOpt			/**< Optimal dual solution vector.
																					 *	 If a NULL pointer is passed, all entries are set to zero. */
												);

		/** Setups gradient of the auxiliary initial QP for given
		 *  optimal primal/dual solution and given initial working set
		 *  (assumes that members X, Y and BOUNDS have already been initialised!).
		 *	\return SUCCESSFUL_RETURN */
		returnValue setupAuxiliaryQPgradient( );

		/** Setups bounds of the auxiliary initial QP for given
		 *  optimal primal/dual solution and given initial working set
		 *  (assumes that members X, Y and BOUNDS have already been initialised!).
		 *	\return SUCCESSFUL_RETURN \n
					RET_UNKNOWN BUG */
		returnValue setupAuxiliaryQPbounds( BooleanType useRelaxation	/**< Flag indicating if inactive bounds shall be relaxed. */
											);


		/** Adds a bound to active set (specialised version for the case where no constraints exist).
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_ADDBOUND_FAILED */
		returnValue addBound(	int number,					/**< Number of bound to be added to active set. */
								SubjectToStatus B_status,	/**< Status of new active bound. */
								BooleanType updateCholesky	/**< Flag indicating if Cholesky decomposition shall be updated. */
								);

		/** Removes a bounds from active set (specialised version for the case where no constraints exist).
		 *	\return SUCCESSFUL_RETURN \n
					RET_HESSIAN_NOT_SPD \n
					RET_REMOVEBOUND_FAILED */
		returnValue removeBound(	int number,					/**< Number of bound to be removed from active set. */
									BooleanType updateCholesky	/**< Flag indicating if Cholesky decomposition shall be updated. */
									);


		/** Solves the system Ra = b or R^Ta = b where R is an upper triangular matrix.
		 *	\return SUCCESSFUL_RETURN \n
					RET_DIV_BY_ZERO */
		returnValue backsolveR(	const real_t* const b,	/**< Right hand side vector. */
								BooleanType transposed,	/**< Indicates if the transposed system shall be solved. */
								real_t* const a 		/**< Output: Solution vector */
								);

		/** Solves the system Ra = b or R^Ta = b where R is an upper triangular matrix. \n
		 *  Special variant for the case that this function is called from within "removeBound()".
		 *	\return SUCCESSFUL_RETURN \n
					RET_DIV_BY_ZERO */
		returnValue backsolveR(	const real_t* const b,		/**< Right hand side vector. */
								BooleanType transposed,		/**< Indicates if the transposed system shall be solved. */
								BooleanType removingBound,	/**< Indicates if function is called from "removeBound()". */
								real_t* const a 			/**< Output: Solution vector */
								);


		/** Determines step direction of the shift of the QP data.
		 *	\return SUCCESSFUL_RETURN */
		returnValue hotstart_determineDataShift(const int* const FX_idx, 	/**< Index array of fixed variables. */
												const real_t* const g_new,	/**< New gradient vector. */
												const real_t* const lb_new,	/**< New lower bounds. */
												const real_t* const ub_new,	/**< New upper bounds. */
												real_t* const delta_g,	 	/**< Output: Step direction of gradient vector. */
												real_t* const delta_lb,	 	/**< Output: Step direction of lower bounds. */
												real_t* const delta_ub,	 	/**< Output: Step direction of upper bounds. */
												BooleanType& Delta_bB_isZero/**< Output: Indicates if active bounds are to be shifted. */
												);


		/** Checks if lower/upper bounds remain consistent
		 *  (i.e. if lb <= ub) during the current step.
		 *	\return BT_TRUE iff bounds remain consistent
		 */
		BooleanType areBoundsConsistent(	const real_t* const delta_lb,		/**< Step direction of lower bounds. */
											const real_t* const delta_ub		/**< Step direction of upper bounds. */
											) const;


		/** Setups internal QP data.
		 *	\return SUCCESSFUL_RETURN \n
					RET_INVALID_ARGUMENTS */
		returnValue setupQPdata(	const real_t* const _H, 	/**< Hessian matrix. */
									const real_t* const _R, 	/**< Cholesky factorization of the Hessian matrix. */
									const real_t* const _g,		/**< Gradient vector. */
									const real_t* const _lb,	/**< Lower bounds (on variables). \n
																	 If no lower bounds exist, a NULL pointer can be passed. */
									const real_t* const _ub		/**< Upper bounds (on variables). \n
																	 If no upper bounds exist, a NULL pointer can be passed. */
									);


		/** Sets Hessian matrix of the QP.
		 *	\return SUCCESSFUL_RETURN */
		inline returnValue setH(	const real_t* const H_new	/**< New Hessian matrix (with correct dimension!). */
									);

		/** Changes gradient vector of the QP.
		 *	\return SUCCESSFUL_RETURN */
		inline returnValue setG(	const real_t* const g_new	/**< New gradient vector (with correct dimension!). */
									);

		/** Changes lower bound vector of the QP.
		 *	\return SUCCESSFUL_RETURN */
		inline returnValue setLB(	const real_t* const lb_new	/**< New lower bound vector (with correct dimension!). */
									);

		/** Changes single entry of lower bound vector of the QP.
		 *	\return SUCCESSFUL_RETURN  \n
					RET_INDEX_OUT_OF_BOUNDS */
		inline returnValue setLB(	int number,		/**< Number of entry to be changed. */
									real_t value	/**< New value for entry of lower bound vector. */
									);

		/** Changes upper bound vector of the QP.
		 *	\return SUCCESSFUL_RETURN */
		inline returnValue setUB(	const real_t* const ub_new	/**< New upper bound vector (with correct dimension!). */
									);

		/** Changes single entry of upper bound vector of the QP.
		 *	\return SUCCESSFUL_RETURN  \n
					RET_INDEX_OUT_OF_BOUNDS */
		inline returnValue setUB(	int number,		/**< Number of entry to be changed. */
									real_t value	/**< New value for entry of upper bound vector. */
									);


		/** Computes parameters for the Givens matrix G for which [x,y]*G = [z,0]
		 *	\return SUCCESSFUL_RETURN */
		inline void computeGivens(	real_t xold,	/**< Matrix entry to be normalised. */
									real_t yold,	/**< Matrix entry to be annihilated. */
									real_t& xnew,	/**< Output: Normalised matrix entry. */
									real_t& ynew,	/**< Output: Annihilated matrix entry. */
									real_t& c,		/**< Output: Cosine entry of Givens matrix. */
									real_t& s 		/**< Output: Sine entry of Givens matrix. */
									) const;

		/** Applies Givens matrix determined by c and s (cf. computeGivens).
		 *	\return SUCCESSFUL_RETURN */
		inline void applyGivens(	real_t c,		/**< Cosine entry of Givens matrix. */
									real_t s,		/**< Sine entry of Givens matrix. */
									real_t xold,	/**< Matrix entry to be transformed corresponding to
													 *	 the normalised entry of the original matrix. */
									real_t yold, 	/**< Matrix entry to be transformed corresponding to
													 *	 the annihilated entry of the original matrix. */
									real_t& xnew,	/**< Output: Transformed matrix entry corresponding to
													 *	 the normalised entry of the original matrix. */
									real_t& ynew	/**< Output: Transformed matrix entry corresponding to
													 *	 the annihilated entry of the original matrix. */
									) const;


	/*
	 *	PRIVATE MEMBER FUNCTIONS
	 */
	private:
		/** Determines step direction of the homotopy path.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_STEPDIRECTION_FAILED_CHOLESKY */
		returnValue hotstart_determineStepDirection(const int* const FR_idx, 		/**< Index array of free variables. */
													const int* const FX_idx, 		/**< Index array of fixed variables. */
													const real_t* const delta_g,	/**< Step direction of gradient vector. */
													const real_t* const delta_lb,	/**< Step direction of lower bounds. */
													const real_t* const delta_ub,	/**< Step direction of upper bounds. */
													BooleanType Delta_bB_isZero,	/**< Indicates if active bounds are to be shifted. */
													real_t* const delta_xFX, 		/**< Output: Primal homotopy step direction of fixed variables. */
													real_t* const delta_xFR,	 	/**< Output: Primal homotopy step direction of free variables. */
													real_t* const delta_yFX 		/**< Output: Dual homotopy step direction of fixed variables' multiplier. */
													);

		/** Determines the maximum possible step length along the homotopy path.
		 *	\return SUCCESSFUL_RETURN */
		returnValue hotstart_determineStepLength(	const int* const FR_idx, 		/**< Index array of free variables. */
													const int* const FX_idx, 		/**< Index array of fixed variables. */
													const real_t* const delta_lb,	/**< Step direction of lower bounds. */
													const real_t* const delta_ub,	/**< Step direction of upper bounds. */
													const real_t* const delta_xFR,	/**< Primal homotopy step direction of free variables. */
													const real_t* const delta_yFX,	/**< Dual homotopy step direction of fixed variables' multiplier. */
													int& BC_idx, 					/**< Output: Index of blocking constraint. */
													SubjectToStatus& BC_status		/**< Output: Status of blocking constraint. */
													);

		/** Performs a step along the homotopy path (and updates active set).
		 *	\return  SUCCESSFUL_RETURN \n
		 			 RET_OPTIMAL_SOLUTION_FOUND \n
		 			 RET_REMOVE_FROM_ACTIVESET_FAILED \n
					 RET_ADD_TO_ACTIVESET_FAILED \n
					 RET_QP_INFEASIBLE */
		returnValue hotstart_performStep(	const int* const FR_idx, 			/**< Index array of free variables. */
											const int* const FX_idx, 			/**< Index array of fixed variables. */
											const real_t* const delta_g,	 	/**< Step direction of gradient vector. */
											const real_t* const delta_lb,	 	/**< Step direction of lower bounds. */
											const real_t* const delta_ub,	 	/**< Step direction of upper bounds. */
											const real_t* const delta_xFX, 		/**< Primal homotopy step direction of fixed variables. */
											const real_t* const delta_xFR,	 	/**< Primal homotopy step direction of free variables. */
											const real_t* const delta_yFX, 		/**< Dual homotopy step direction of fixed variables' multiplier. */
											int BC_idx, 						/**< Index of blocking constraint. */
											SubjectToStatus BC_status 			/**< Status of blocking constraint. */
											);


		#ifdef PC_DEBUG  /* Define print functions only for debugging! */

		/** Prints concise information on the current iteration.
		 *	\return  SUCCESSFUL_RETURN \n */
		returnValue printIteration(	int iteration,				/**< Number of current iteration. */
									int BC_idx, 				/**< Index of blocking bound. */
									SubjectToStatus BC_status	/**< Status of blocking bound. */
									);

		#endif  /* PC_DEBUG */


		/** Determines the maximum violation of the KKT optimality conditions
		 *  of the current iterate within the QProblemB object.
		 *	\return SUCCESSFUL_RETURN \n
		 * 			RET_INACCURATE_SOLUTION \n
		 * 			RET_NO_SOLUTION */
		returnValue checkKKTconditions( );


	/*
	 *	PROTECTED MEMBER VARIABLES
	 */
	protected:
		real_t H[NVMAX*NVMAX];		/**< Hessian matrix. */
		BooleanType hasHessian;		/**< Flag indicating whether H contains Hessian or corresponding Cholesky factor R; \sa init. */

		real_t g[NVMAX];			/**< Gradient. */
		real_t lb[NVMAX];			/**< Lower bound vector (on variables). */
		real_t ub[NVMAX];			/**< Upper bound vector (on variables). */

		Bounds bounds;				/**< Data structure for problem's bounds. */

		real_t R[NVMAX*NVMAX];		/**< Cholesky decomposition of H (i.e. H = R^T*R). */
		BooleanType hasCholesky;	/**< Flag indicating whether Cholesky decomposition has already been setup. */

		real_t x[NVMAX];			/**< Primal solution vector. */
		real_t y[NVMAX+NCMAX];		/**< Dual solution vector. */

		real_t tau;					/**< Last homotopy step length. */

		QProblemStatus status;		/**< Current status of the solution process. */

		BooleanType infeasible;		/**< QP infeasible? */
		BooleanType unbounded;		/**< QP unbounded? */

		HessianType hessianType;	/**< Type of Hessian matrix. */

		PrintLevel printlevel;		/**< Print level. */

		int count;					/**< Counts the number of hotstart function calls (internal usage only!). */
};


#include <QProblemB.ipp>

#endif	/* QPOASES_QPROBLEMB_HPP */


/*
 *	end of file
 */
