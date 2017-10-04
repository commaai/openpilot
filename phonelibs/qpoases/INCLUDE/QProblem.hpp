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
 *	\file INCLUDE/QProblem.hpp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Declaration of the QProblem class which is able to use the newly
 *	developed online active set strategy for parametric quadratic programming.
 */



#ifndef QPOASES_QPROBLEM_HPP
#define QPOASES_QPROBLEM_HPP


#include <QProblemB.hpp>
#include <Constraints.hpp>
#include <CyclingManager.hpp>


/** A class for setting up and solving quadratic programs. The main feature is
 *	the possibily to use the newly developed online active set strategy for
 * 	parametric quadratic programming.
 *
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 */
class QProblem : public QProblemB
{
	/* allow SolutionAnalysis class to access private members */
	friend class SolutionAnalysis;
	
	/*
	 *	PUBLIC MEMBER FUNCTIONS
	 */
	public:
		/** Default constructor. */
		QProblem( );

		/** Constructor which takes the QP dimensions only. */
		QProblem(	int _nV,	  				/**< Number of variables. */
					int _nC  					/**< Number of constraints. */
					);

		/** Copy constructor (deep copy). */
		QProblem(	const QProblem& rhs	/**< Rhs object. */
					);

		/** Destructor. */
		~QProblem( );

		/** Assignment operator (deep copy). */
		QProblem& operator=(	const QProblem& rhs		/**< Rhs object. */
								);


		/** Clears all data structures of QProblemB except for QP data.
		 *	\return SUCCESSFUL_RETURN \n
					RET_RESET_FAILED */
		returnValue reset( );


		/** Initialises a QProblem with given QP data and solves it
		 *	using an initial homotopy with empty working set (at most nWSR iterations).
		 *	\return SUCCESSFUL_RETURN \n
					RET_INIT_FAILED \n
					RET_INIT_FAILED_CHOLESKY \n
					RET_INIT_FAILED_TQ \n
					RET_INIT_FAILED_HOTSTART \n
					RET_INIT_FAILED_INFEASIBILITY \n
					RET_INIT_FAILED_UNBOUNDEDNESS \n
					RET_MAX_NWSR_REACHED \n
					RET_INVALID_ARGUMENTS \n
					RET_INACCURATE_SOLUTION \n
		 			RET_NO_SOLUTION */
		returnValue init(	const real_t* const _H, 		/**< Hessian matrix. */
							const real_t* const _g, 		/**< Gradient vector. */
							const real_t* const _A,  		/**< Constraint matrix. */
							const real_t* const _lb,		/**< Lower bound vector (on variables). \n
																If no lower bounds exist, a NULL pointer can be passed. */
							const real_t* const _ub,		/**< Upper bound vector (on variables). \n
																If no upper bounds exist, a NULL pointer can be passed. */
							const real_t* const _lbA,		/**< Lower constraints' bound vector. \n
																If no lower constraints' bounds exist, a NULL pointer can be passed. */
							const real_t* const _ubA,		/**< Upper constraints' bound vector. \n
																If no lower constraints' bounds exist, a NULL pointer can be passed. */
							int& nWSR,						/**< Input: Maximum number of working set recalculations when using initial homotopy.
																Output: Number of performed working set recalculations. */
							const real_t* const yOpt = 0,	/**< Initial guess for dual solution vector. */
							real_t* const cputime = 0		/**< Output: CPU time required to initialise QP. */
							);


		/** Initialises a QProblem with given QP data and solves it
		 *	using an initial homotopy with empty working set (at most nWSR iterations).
		 *	\return SUCCESSFUL_RETURN \n
					RET_INIT_FAILED \n
					RET_INIT_FAILED_CHOLESKY \n
					RET_INIT_FAILED_TQ \n
					RET_INIT_FAILED_HOTSTART \n
					RET_INIT_FAILED_INFEASIBILITY \n
					RET_INIT_FAILED_UNBOUNDEDNESS \n
					RET_MAX_NWSR_REACHED \n
					RET_INVALID_ARGUMENTS \n
					RET_INACCURATE_SOLUTION \n
		 			RET_NO_SOLUTION */
		returnValue init(	const real_t* const _H, 		/**< Hessian matrix. */
							const real_t* const _R, 		/**< Cholesky factorization of the Hessian matrix. */
							const real_t* const _g, 		/**< Gradient vector. */
							const real_t* const _A,  		/**< Constraint matrix. */
							const real_t* const _lb,		/**< Lower bound vector (on variables). \n
																If no lower bounds exist, a NULL pointer can be passed. */
							const real_t* const _ub,		/**< Upper bound vector (on variables). \n
																If no upper bounds exist, a NULL pointer can be passed. */
							const real_t* const _lbA,		/**< Lower constraints' bound vector. \n
																If no lower constraints' bounds exist, a NULL pointer can be passed. */
							const real_t* const _ubA,		/**< Upper constraints' bound vector. \n
																If no lower constraints' bounds exist, a NULL pointer can be passed. */
							int& nWSR,						/**< Input: Maximum number of working set recalculations when using initial homotopy.
																Output: Number of performed working set recalculations. */
							const real_t* const yOpt = 0,	/**< Initial guess for dual solution vector. */
							real_t* const cputime = 0		/**< Output: CPU time required to initialise QP. */
							);


		/** Solves QProblem using online active set strategy.
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
		returnValue hotstart(	const real_t* const g_new,		/**< Gradient of neighbouring QP to be solved. */
								const real_t* const lb_new,		/**< Lower bounds of neighbouring QP to be solved. \n
													 			 	 If no lower bounds exist, a NULL pointer can be passed. */
								const real_t* const ub_new,		/**< Upper bounds of neighbouring QP to be solved. \n
													 			 	 If no upper bounds exist, a NULL pointer can be passed. */
								const real_t* const lbA_new,	/**< Lower constraints' bounds of neighbouring QP to be solved. \n
													 			 	 If no lower constraints' bounds exist, a NULL pointer can be passed. */
								const real_t* const ubA_new,	/**< Upper constraints' bounds of neighbouring QP to be solved. \n
													 			 	 If no upper constraints' bounds exist, a NULL pointer can be passed. */
								int& nWSR,						/**< Input: Maximum number of working set recalculations; \n
															 		 Output: Number of performed working set recalculations. */
								real_t* const cputime			/**< Output: CPU time required to solve QP (or to perform nWSR iterations). */
								);


		/** Returns constraint matrix of the QP (deep copy).
		  *	\return SUCCESSFUL_RETURN */
		inline returnValue getA(	real_t* const _A	/**< Array of appropriate dimension for copying constraint matrix.*/
									) const;

		/** Returns a single row of constraint matrix of the QP (deep copy).
		  *	\return SUCCESSFUL_RETURN \n
					RET_INDEX_OUT_OF_BOUNDS */
		inline returnValue getA(	int number,			/**< Number of entry to be returned. */
									real_t* const row	/**< Array of appropriate dimension for copying (number)th constraint. */
									) const;

		/** Returns lower constraints' bound vector of the QP (deep copy).
		  *	\return SUCCESSFUL_RETURN */
		inline returnValue getLBA(	real_t* const _lbA	/**< Array of appropriate dimension for copying lower constraints' bound vector.*/
									) const;

		/** Returns single entry of lower constraints' bound vector of the QP.
		  *	\return SUCCESSFUL_RETURN \n
					RET_INDEX_OUT_OF_BOUNDS */
		inline returnValue getLBA(	int number,		/**< Number of entry to be returned. */
									real_t& value	/**< Output: lbA[number].*/
									) const;

		/** Returns upper constraints' bound vector of the QP (deep copy).
		  *	\return SUCCESSFUL_RETURN */
		inline returnValue getUBA(	real_t* const _ubA	/**< Array of appropriate dimension for copying upper constraints' bound vector.*/
									) const;

		/** Returns single entry of upper constraints' bound vector of the QP.
		  *	\return SUCCESSFUL_RETURN \n
					RET_INDEX_OUT_OF_BOUNDS */
		inline returnValue getUBA(	int number,		/**< Number of entry to be returned. */
									real_t& value	/**< Output: ubA[number].*/
									) const;


		/** Returns current constraints object of the QP (deep copy).
		  *	\return SUCCESSFUL_RETURN */
		inline returnValue getConstraints(	Constraints* const _constraints	/** Output: Constraints object. */
											) const;


		/** Returns the number of constraints.
		 *	\return Number of constraints. */
		inline int getNC( ) const;

		/** Returns the number of (implicitly defined) equality constraints.
		 *	\return Number of (implicitly defined) equality constraints. */
		inline int getNEC( ) const;

		/** Returns the number of active constraints.
		 *	\return Number of active constraints. */
		inline int getNAC( );

		/** Returns the number of inactive constraints.
		 *	\return Number of inactive constraints. */
		inline int getNIAC( );

		/** Returns the dimension of null space.
		 *	\return Dimension of null space. */
		int getNZ( );


		/** Returns the dual solution vector (deep copy).
		 *	\return SUCCESSFUL_RETURN \n
					RET_QP_NOT_SOLVED */
		returnValue getDualSolution(	real_t* const yOpt	/**< Output: Dual solution vector (if QP has been solved). */
										) const;


	/*
	 *	PROTECTED MEMBER FUNCTIONS
	 */
	protected:
		/** Determines type of constraints and bounds (i.e. implicitly fixed, unbounded etc.).
		 *	\return SUCCESSFUL_RETURN \n
					RET_SETUPSUBJECTTOTYPE_FAILED */
		returnValue setupSubjectToType( );

		/** Computes the Cholesky decomposition R of the projected Hessian (i.e. R^T*R = Z^T*H*Z).
		 *	\return SUCCESSFUL_RETURN \n
		 *			RET_INDEXLIST_CORRUPTED */
		returnValue setupCholeskyDecompositionProjected( );

		/** Initialises TQ factorisation of A (i.e. A*Q = [0 T]) if NO constraint is active.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_INDEXLIST_CORRUPTED */
		returnValue setupTQfactorisation( );


		/** Solves a QProblem whose QP data is assumed to be stored in the member variables.
		 *  A guess for its primal/dual optimal solution vectors and the corresponding
		 *  working sets of bounds and constraints can be provided.
		 *	\return SUCCESSFUL_RETURN \n
					RET_INIT_FAILED \n
					RET_INIT_FAILED_CHOLESKY \n
					RET_INIT_FAILED_TQ \n
					RET_INIT_FAILED_HOTSTART \n
					RET_INIT_FAILED_INFEASIBILITY \n
					RET_INIT_FAILED_UNBOUNDEDNESS \n
					RET_MAX_NWSR_REACHED */
		returnValue solveInitialQP(	const real_t* const xOpt,						/**< Optimal primal solution vector.
																					 *	 A NULL pointer can be passed. */
									const real_t* const yOpt,						/**< Optimal dual solution vector.
																					 *	 A NULL pointer can be passed. */
									const Bounds* const guessedBounds,				/**< Guessed working set of bounds for solution (xOpt,yOpt).
																		 			 *	 A NULL pointer can be passed. */
									const Constraints* const guessedConstraints,	/**< Optimal working set of constraints for solution (xOpt,yOpt).
																		 			 *	 A NULL pointer can be passed. */
									int& nWSR, 										/**< Input: Maximum number of working set recalculations; \n
																 					 *	 Output: Number of performed working set recalculations. */
									real_t* const cputime							/**< Output: CPU time required to solve QP (or to perform nWSR iterations). */
									);

		/** Obtains the desired working set for the auxiliary initial QP in
		 *  accordance with the user specifications
		 *  (assumes that member AX has already been initialised!)
		 *	\return SUCCESSFUL_RETURN \n
					RET_OBTAINING_WORKINGSET_FAILED \n
					RET_INVALID_ARGUMENTS */
		returnValue obtainAuxiliaryWorkingSet(	const real_t* const xOpt,						/**< Optimal primal solution vector.
																								 *	 If a NULL pointer is passed, all entries are assumed to be zero. */
												const real_t* const yOpt,						/**< Optimal dual solution vector.
																								 *	 If a NULL pointer is passed, all entries are assumed to be zero. */
												const Bounds* const guessedBounds,				/**< Guessed working set of bounds for solution (xOpt,yOpt). */
												const Constraints* const guessedConstraints,	/**< Guessed working set for solution (xOpt,yOpt). */
												Bounds* auxiliaryBounds,						/**< Input: Allocated bound object. \n
																								 *	 Ouput: Working set of constraints for auxiliary QP. */
												Constraints* auxiliaryConstraints				/**< Input: Allocated bound object. \n
																								 *	 Ouput: Working set for auxiliary QP. */
												) const;

		/** Setups bound and constraints data structures according to auxiliaryBounds/Constraints.
		 *  (If the working set shall be setup afresh, make sure that
		 *  bounds and constraints data structure have been resetted
		 *  and the TQ factorisation has been initialised!)
		 *	\return SUCCESSFUL_RETURN \n
					RET_SETUP_WORKINGSET_FAILED \n
					RET_INVALID_ARGUMENTS \n
					RET_UNKNOWN BUG */
		returnValue setupAuxiliaryWorkingSet(	const Bounds* const auxiliaryBounds,			/**< Working set of bounds for auxiliary QP. */
												const Constraints* const auxiliaryConstraints,	/**< Working set of constraints for auxiliary QP. */
												BooleanType setupAfresh							/**< Flag indicating if given working set shall be
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
		 *  (assumes that members X, Y and BOUNDS, CONSTRAINTS have already been initialised!).
		 *	\return SUCCESSFUL_RETURN */
		returnValue setupAuxiliaryQPgradient( );

		/** Setups (constraints') bounds of the auxiliary initial QP for given
		 *  optimal primal/dual solution and given initial working set
		 *  (assumes that members X, Y and BOUNDS, CONSTRAINTS have already been initialised!).
		 *	\return SUCCESSFUL_RETURN \n
					RET_UNKNOWN BUG */
		returnValue setupAuxiliaryQPbounds(	const Bounds* const auxiliaryBounds,			/**< Working set of bounds for auxiliary QP. */
											const Constraints* const auxiliaryConstraints,	/**< Working set of constraints for auxiliary QP. */
											BooleanType useRelaxation						/**< Flag indicating if inactive (constraints') bounds shall be relaxed. */
											);


		/** Adds a constraint to active set.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_ADDCONSTRAINT_FAILED \n
					RET_ADDCONSTRAINT_FAILED_INFEASIBILITY \n
					RET_ENSURELI_FAILED */
		returnValue addConstraint(	int number,					/**< Number of constraint to be added to active set. */
									SubjectToStatus C_status,	/**< Status of new active constraint. */
									BooleanType updateCholesky	/**< Flag indicating if Cholesky decomposition shall be updated. */
									);

		/** Checks if new active constraint to be added is linearly dependent from
		 *	from row of the active constraints matrix.
		 *	\return	 RET_LINEARLY_DEPENDENT \n
		 			 RET_LINEARLY_INDEPENDENT \n
					 RET_INDEXLIST_CORRUPTED */
		returnValue addConstraint_checkLI(	int number			/**< Number of constraint to be added to active set. */
											);

		/** Ensures linear independence of constraint matrix when a new constraint is added.
		 * 	To this end a bound or constraint is removed simultaneously if necessary.
		 *	\return	 SUCCESSFUL_RETURN \n
		 			 RET_LI_RESOLVED \n
					 RET_ENSURELI_FAILED \n
					 RET_ENSURELI_FAILED_TQ \n
					 RET_ENSURELI_FAILED_NOINDEX \n
					 RET_REMOVE_FROM_ACTIVESET */
		returnValue addConstraint_ensureLI(	int number,					/**< Number of constraint to be added to active set. */
											SubjectToStatus C_status	/**< Status of new active bound. */
											);

		/** Adds a bound to active set.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_ADDBOUND_FAILED \n
					RET_ADDBOUND_FAILED_INFEASIBILITY \n
					RET_ENSURELI_FAILED */
		returnValue addBound(	int number,					/**< Number of bound to be added to active set. */
								SubjectToStatus B_status,	/**< Status of new active bound. */
								BooleanType updateCholesky	/**< Flag indicating if Cholesky decomposition shall be updated. */
								);

		/** Checks if new active bound to be added is linearly dependent from
		 *	from row of the active constraints matrix.
		 *	\return	 RET_LINEARLY_DEPENDENT \n
		 			 RET_LINEARLY_INDEPENDENT */
		returnValue addBound_checkLI(	int number			/**< Number of bound to be added to active set. */
										);

		/** Ensures linear independence of constraint matrix when a new bound is added.
		 *	To this end a bound or constraint is removed simultaneously if necessary.
		 *	\return	 SUCCESSFUL_RETURN \n
		 			 RET_LI_RESOLVED \n
					 RET_ENSURELI_FAILED \n
					 RET_ENSURELI_FAILED_TQ \n
					 RET_ENSURELI_FAILED_NOINDEX \n
					 RET_REMOVE_FROM_ACTIVESET */
		returnValue addBound_ensureLI(	int number,					/**< Number of bound to be added to active set. */
										SubjectToStatus B_status	/**< Status of new active bound. */
										);

		/** Removes a constraint from active set.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_CONSTRAINT_NOT_ACTIVE \n
					RET_REMOVECONSTRAINT_FAILED \n
					RET_HESSIAN_NOT_SPD */
		returnValue removeConstraint(	int number,					/**< Number of constraint to be removed from active set. */
										BooleanType updateCholesky	/**< Flag indicating if Cholesky decomposition shall be updated. */
										);

		/** Removes a bounds from active set.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_BOUND_NOT_ACTIVE \n
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


		/** Solves the system Ta = b or T^Ta = b where T is a reverse upper triangular matrix.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_DIV_BY_ZERO */
		returnValue backsolveT(	const real_t* const b,	/**< Right hand side vector. */
								BooleanType transposed,	/**< Indicates if the transposed system shall be solved. */
								real_t* const a 		/**< Output: Solution vector */
								);


		/** Determines step direction of the shift of the QP data.
		 *	\return SUCCESSFUL_RETURN */
		returnValue hotstart_determineDataShift(const int* const FX_idx, 	/**< Index array of fixed variables. */
												const int* const AC_idx, 	/**< Index array of active constraints. */
												const real_t* const g_new,	/**< New gradient vector. */
												const real_t* const lbA_new,/**< New lower constraints' bounds. */
												const real_t* const ubA_new,/**< New upper constraints' bounds. */
												const real_t* const lb_new,	/**< New lower bounds. */
												const real_t* const ub_new,	/**< New upper bounds. */
												real_t* const delta_g,	 	/**< Output: Step direction of gradient vector. */
												real_t* const delta_lbA,	/**< Output: Step direction of lower constraints' bounds. */
												real_t* const delta_ubA,	/**< Output: Step direction of upper constraints' bounds. */
												real_t* const delta_lb,	 	/**< Output: Step direction of lower bounds. */
												real_t* const delta_ub,	 	/**< Output: Step direction of upper bounds. */
												BooleanType& Delta_bC_isZero,/**< Output: Indicates if active constraints' bounds are to be shifted. */
												BooleanType& Delta_bB_isZero/**< Output: Indicates if active bounds are to be shifted. */
												);

		/** Determines step direction of the homotopy path.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_STEPDIRECTION_FAILED_TQ \n
					RET_STEPDIRECTION_FAILED_CHOLESKY */
		returnValue hotstart_determineStepDirection(const int* const FR_idx, 		/**< Index array of free variables. */
													const int* const FX_idx, 		/**< Index array of fixed variables. */
													const int* const AC_idx, 		/**< Index array of active constraints. */
													const real_t* const delta_g,	/**< Step direction of gradient vector. */
													const real_t* const delta_lbA,	/**< Step direction of lower constraints' bounds. */
													const real_t* const delta_ubA,	/**< Step direction of upper constraints' bounds. */
													const real_t* const delta_lb,	/**< Step direction of lower bounds. */
													const real_t* const delta_ub,	/**< Step direction of upper bounds. */
													BooleanType Delta_bC_isZero, 	/**< Indicates if active constraints' bounds are to be shifted. */
													BooleanType Delta_bB_isZero,	/**< Indicates if active bounds are to be shifted. */
													real_t* const delta_xFX, 		/**< Output: Primal homotopy step direction of fixed variables. */
													real_t* const delta_xFR,	 	/**< Output: Primal homotopy step direction of free variables. */
													real_t* const delta_yAC, 		/**< Output: Dual homotopy step direction of active constraints' multiplier. */
													real_t* const delta_yFX 		/**< Output: Dual homotopy step direction of fixed variables' multiplier. */
													);

		/** Determines the maximum possible step length along the homotopy path.
		 *	\return SUCCESSFUL_RETURN */
		returnValue hotstart_determineStepLength(	const int* const FR_idx, 			/**< Index array of free variables. */
													const int* const FX_idx, 			/**< Index array of fixed variables. */
													const int* const AC_idx, 			/**< Index array of active constraints. */
													const int* const IAC_idx, 			/**< Index array of inactive constraints. */
													const real_t* const delta_lbA,		/**< Step direction of lower constraints' bounds. */
													const real_t* const delta_ubA,		/**< Step direction of upper constraints' bounds. */
													const real_t* const delta_lb,	 	/**< Step direction of lower bounds. */
													const real_t* const delta_ub,	 	/**< Step direction of upper bounds. */
													const real_t* const delta_xFX, 		/**< Primal homotopy step direction of fixed variables. */
													const real_t* const delta_xFR,		/**< Primal homotopy step direction of free variables. */
													const real_t* const delta_yAC,		/**< Dual homotopy step direction of active constraints' multiplier. */
													const real_t* const delta_yFX,		/**< Dual homotopy step direction of fixed variables' multiplier. */
													real_t* const delta_Ax,				/**< Output: Step in vector Ax. */
													int& BC_idx, 						/**< Output: Index of blocking constraint. */
													SubjectToStatus& BC_status,			/**< Output: Status of blocking constraint. */
													BooleanType& BC_isBound 			/**< Output: Indicates if blocking constraint is a bound. */
													);

		/** Performs a step along the homotopy path (and updates active set).
		 *	\return  SUCCESSFUL_RETURN \n
		 			 RET_OPTIMAL_SOLUTION_FOUND \n
		 			 RET_REMOVE_FROM_ACTIVESET_FAILED \n
					 RET_ADD_TO_ACTIVESET_FAILED \n
					 RET_QP_INFEASIBLE */
		returnValue hotstart_performStep(	const int* const FR_idx, 			/**< Index array of free variables. */
											const int* const FX_idx, 			/**< Index array of fixed variables. */
											const int* const AC_idx, 			/**< Index array of active constraints. */
											const int* const IAC_idx, 			/**< Index array of inactive constraints. */
											const real_t* const delta_g,	 	/**< Step direction of gradient vector. */
											const real_t* const delta_lbA,	 	/**< Step direction of lower constraints' bounds. */
											const real_t* const delta_ubA,	 	/**< Step direction of upper constraints' bounds. */
											const real_t* const delta_lb,	 	/**< Step direction of lower bounds. */
											const real_t* const delta_ub,	 	/**< Step direction of upper bounds. */
											const real_t* const delta_xFX, 		/**< Primal homotopy step direction of fixed variables. */
											const real_t* const delta_xFR,	 	/**< Primal homotopy step direction of free variables. */
											const real_t* const delta_yAC,	 	/**< Dual homotopy step direction of active constraints' multiplier. */
											const real_t* const delta_yFX, 		/**< Dual homotopy step direction of fixed variables' multiplier. */
											const real_t* const delta_Ax, 		/**< Step in vector Ax. */
											int BC_idx, 						/**< Index of blocking constraint. */
											SubjectToStatus BC_status,			/**< Status of blocking constraint. */
											BooleanType BC_isBound 				/**< Indicates if blocking constraint is a bound. */
											);


		/** Checks if lower/upper (constraints') bounds remain consistent
		 *  (i.e. if lb <= ub and lbA <= ubA ) during the current step.
		 *	\return BT_TRUE iff (constraints") bounds remain consistent
		 */
		BooleanType areBoundsConsistent(	const real_t* const delta_lb,	/**< Step direction of lower bounds. */
											const real_t* const delta_ub,	/**< Step direction of upper bounds. */
											const real_t* const delta_lbA,	/**< Step direction of lower constraints' bounds. */
											const real_t* const delta_ubA	/**< Step direction of upper constraints' bounds. */
											) const;


		/** Setups internal QP data.
		 *	\return SUCCESSFUL_RETURN \n
					RET_INVALID_ARGUMENTS */
		returnValue setupQPdata(	const real_t* const _H, 	/**< Hessian matrix. */
									const real_t* const _R, 	/**< Cholesky factorization of the Hessian matrix. */
									const real_t* const _g, 	/**< Gradient vector. */
									const real_t* const _A,  	/**< Constraint matrix. */
									const real_t* const _lb,	/**< Lower bound vector (on variables). \n
																	 If no lower bounds exist, a NULL pointer can be passed. */
									const real_t* const _ub,	/**< Upper bound vector (on variables). \n
																	 If no upper bounds exist, a NULL pointer can be passed. */
									const real_t* const _lbA,	/**< Lower constraints' bound vector. \n
																	 If no lower constraints' bounds exist, a NULL pointer can be passed. */
									const real_t* const _ubA	/**< Upper constraints' bound vector. \n
																	 If no lower constraints' bounds exist, a NULL pointer can be passed. */
									);


		#ifdef PC_DEBUG  /* Define print functions only for debugging! */

		/** Prints concise information on the current iteration.
		 *	\return  SUCCESSFUL_RETURN \n */
		returnValue printIteration(	int iteration,				/**< Number of current iteration. */
									int BC_idx, 				/**< Index of blocking constraint. */
									SubjectToStatus BC_status,	/**< Status of blocking constraint. */
									BooleanType BC_isBound 		/**< Indicates if blocking constraint is a bound. */
									);

		/** Prints concise information on the current iteration.
		 *  NOTE: ONLY DEFINED FOR SUPPRESSING A COMPILER WARNING!!
		 *	\return  SUCCESSFUL_RETURN \n */
		returnValue printIteration(	int iteration,				/**< Number of current iteration. */
									int BC_idx, 				/**< Index of blocking bound. */
									SubjectToStatus BC_status	/**< Status of blocking bound. */
									);

		#endif  /* PC_DEBUG */


		/** Determines the maximum violation of the KKT optimality conditions
		 *  of the current iterate within the QProblem object.
		 *	\return SUCCESSFUL_RETURN \n
		 * 			RET_INACCURATE_SOLUTION \n
		 * 			RET_NO_SOLUTION */
		returnValue checkKKTconditions( );


		/** Sets constraint matrix of the QP. \n
			(Remark: Also internal vector Ax is recomputed!)
		 *	\return SUCCESSFUL_RETURN */
		inline returnValue setA(	const real_t* const A_new	/**< New constraint matrix (with correct dimension!). */
									);

		/** Changes single row of constraint matrix of the QP. \n
			(Remark: Also correponding component of internal vector Ax is recomputed!)
		 *	\return SUCCESSFUL_RETURN  \n
					RET_INDEX_OUT_OF_BOUNDS */
		inline returnValue setA(	int number,					/**< Number of row to be changed. */
									const real_t* const value	/**< New (number)th constraint (with correct dimension!). */
									);


		/** Sets constraints' lower bound vector of the QP.
		 *	\return SUCCESSFUL_RETURN */
		inline returnValue setLBA(	const real_t* const lbA_new	/**< New constraints' lower bound vector (with correct dimension!). */
									);

		/** Changes single entry of lower constraints' bound vector of the QP.
		 *	\return SUCCESSFUL_RETURN  \n
					RET_INDEX_OUT_OF_BOUNDS */
		inline returnValue setLBA(	int number,		/**< Number of entry to be changed. */
									real_t value	/**< New value for entry of lower constraints' bound vector (with correct dimension!). */
									);

		/** Sets constraints' upper bound vector of the QP.
		 *	\return SUCCESSFUL_RETURN */
		inline returnValue setUBA(	const real_t* const ubA_new	/**< New constraints' upper bound vector (with correct dimension!). */
									);

		/** Changes single entry of upper constraints' bound vector of the QP.
		 *	\return SUCCESSFUL_RETURN  \n
					RET_INDEX_OUT_OF_BOUNDS */
		inline returnValue setUBA(	int number,		/**< Number of entry to be changed. */
									real_t value	/**< New value for entry of upper constraints' bound vector (with correct dimension!). */
									);


	/*
	 *	PROTECTED MEMBER VARIABLES
	 */
	protected:
		real_t A[NCMAX_ALLOC*NVMAX];		/**< Constraint matrix. */
		real_t lbA[NCMAX_ALLOC];			/**< Lower constraints' bound vector. */
		real_t ubA[NCMAX_ALLOC];			/**< Upper constraints' bound vector. */

		Constraints constraints;			/**< Data structure for problem's constraints. */

		real_t T[NVMAX*NVMAX];				/**< Reverse triangular matrix, A = [0 T]*Q'. */
		real_t Q[NVMAX*NVMAX];				/**< Orthonormal quadratic matrix, A = [0 T]*Q'. */
		int sizeT;							/**< Matrix T is stored in a (sizeT x sizeT) array. */

		real_t Ax[NCMAX_ALLOC];				/**< Stores the current product A*x (for increased efficiency only). */

		CyclingManager cyclingManager;		/**< Data structure for storing (possible) cycling information (NOT YET IMPLEMENTED!). */
};


#include <QProblem.ipp>

#endif	/* QPOASES_QPROBLEM_HPP */


/*
 *	end of file
 */
