/*
 *	This file is part of qpOASES.
 *
 *	qpOASES -- An Implementation of the Online Active Set Strategy.
 *	Copyright (C) 2007-2015 by Hans Joachim Ferreau, Andreas Potschka,
 *	Christian Kirches et al. All rights reserved.
 *
 *	qpOASES is free software; you can redistribute it and/or
 *	modify it under the terms of the GNU Lesser General Public
 *	License as published by the Free Software Foundation; either
 *	version 2.1 of the License, or (at your option) any later version.
 *
 *	qpOASES is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *	See the GNU Lesser General Public License for more details.
 *
 *	You should have received a copy of the GNU Lesser General Public
 *	License along with qpOASES; if not, write to the Free Software
 *	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


/**
 *	\file include/qpOASES_e/QProblemB.h
 *	\author Hans Joachim Ferreau, Andreas Potschka, Christian Kirches
 *	\version 3.1embedded
 *	\date 2007-2015
 *
 *	Declaration of the QProblemB class which is able to use the newly
 *	developed online active set strategy for parametric quadratic programming
 *	for problems with (simple) bounds only.
 */



#ifndef QPOASES_QPROBLEMB_H
#define QPOASES_QPROBLEMB_H


#include <qpOASES_e/Bounds.h>
#include <qpOASES_e/Options.h>
#include <qpOASES_e/Matrices.h>
#include <qpOASES_e/Flipper.h>


BEGIN_NAMESPACE_QPOASES

typedef struct {
	Bounds *emptyBounds;
	Bounds *auxiliaryBounds;

	real_t *ub_new_far;
	real_t *lb_new_far;

	real_t *g_new;
	real_t *lb_new;
	real_t *ub_new;

	real_t *g_new2;
	real_t *lb_new2;
	real_t *ub_new2;

	real_t *Hx;

	real_t *_H;

	real_t *g_original;
	real_t *lb_original;
	real_t *ub_original;

	real_t *delta_xFR;
	real_t *delta_xFX;
	real_t *delta_yFX;
	real_t *delta_g;
	real_t *delta_lb;
	real_t *delta_ub;

	real_t *gMod;

	real_t *num;
	real_t *den;

	real_t *rhs;
	real_t *r;
} QProblemB_ws;

int QProblemB_ws_calculateMemorySize( unsigned int nV );

char *QProblemB_ws_assignMemory( unsigned int nV, QProblemB_ws **mem, void *raw_memory );

QProblemB_ws *QProblemB_ws_createMemory( unsigned int nV );


/**
 *	\brief Implements the online active set strategy for box-constrained QPs.
 *
 *	Class for setting up and solving quadratic programs with bounds (= box constraints) only.
 *	The main feature is the possibility to use the newly developed online active set strategy
 *	for parametric quadratic programming.
 *
 *	\author Hans Joachim Ferreau, Andreas Potschka, Christian Kirches
 *	\version 3.1embedded
 *	\date 2007-2015
 */
typedef struct
{
	QProblemB_ws *ws;
	Bounds *bounds;					/**< Data structure for problem's bounds. */
	Flipper *flipper;				/**< Struct for making a temporary copy of the matrix factorisations. */

	DenseMatrix* H;					/**< Hessian matrix pointer. */

	Options options;				/**< Struct containing all user-defined options for solving QPs. */
	TabularOutput tabularOutput;	/**< Struct storing information for tabular output (printLevel == PL_TABULAR). */

	real_t *g;						/**< Gradient. */
	real_t *lb;						/**< Lower bound vector (on variables). */
	real_t *ub;						/**< Upper bound vector (on variables). */

	real_t *R;						/**< Cholesky factor of H (i.e. H = R^T*R). */

	real_t *x;						/**< Primal solution vector. */
	real_t *y;						/**< Dual solution vector. */

	real_t *delta_xFR_TMP;			/**< Temporary for determineStepDirection */

	real_t tau;						/**< Last homotopy step length. */
	real_t regVal;					/**< Holds the offset used to regularise Hessian matrix (zero by default). */

	real_t ramp0;					/**< Start value for Ramping Strategy. */
	real_t ramp1;					/**< Final value for Ramping Strategy. */

	QProblemStatus status;			/**< Current status of the solution process. */
	HessianType hessianType;		/**< Type of Hessian matrix. */

	BooleanType haveCholesky;		/**< Flag indicating whether Cholesky decomposition has already been setup. */
	BooleanType infeasible;			/**< QP infeasible? */
	BooleanType unbounded;			/**< QP unbounded? */

	int rampOffset;					/**< Offset index for Ramping. */
	unsigned int count;				/**< Counts the number of hotstart function calls (internal usage only!). */
} QProblemB;

int QProblemB_calculateMemorySize( unsigned int nV );

char *QProblemB_assignMemory( unsigned int nV, QProblemB **mem, void *raw_memory );

QProblemB *QProblemB_createMemory( unsigned int nV );


/** Constructor which takes the QP dimension and Hessian type
 *  information. If the Hessian is the zero (i.e. HST_ZERO) or the
 *  identity matrix (i.e. HST_IDENTITY), respectively, no memory
 *  is allocated for it and a NULL pointer can be passed for it
 *  to the init() functions. */
void QProblemBCON(	QProblemB* _THIS,
					int _nV,						/**< Number of variables. */
					HessianType _hessianType		/**< Type of Hessian matrix. */
					);

void QProblemBCPY(	QProblemB* FROM,
					QProblemB* TO
					);


/** Clears all data structures of QProblemB except for QP data.
 *	\return SUCCESSFUL_RETURN \n
			RET_RESET_FAILED */
returnValue QProblemB_reset( QProblemB* _THIS );


/** Initialises a simply bounded QP problem with given QP data and tries to solve it
 *	using at most nWSR iterations.
 *
 *  Note: This function internally calls solveInitialQP for initialisation!
 *
 *	\return SUCCESSFUL_RETURN \n
			RET_INIT_FAILED \n
			RET_INIT_FAILED_CHOLESKY \n
			RET_INIT_FAILED_HOTSTART \n
			RET_INIT_FAILED_INFEASIBILITY \n
			RET_INIT_FAILED_UNBOUNDEDNESS \n
			RET_MAX_NWSR_REACHED \n
			RET_INVALID_ARGUMENTS */
returnValue QProblemB_initM(	QProblemB* _THIS,
								DenseMatrix *_H,	 		/**< Hessian matrix. */
								const real_t* const _g,		/**< Gradient vector. */
								const real_t* const _lb,	/**< Lower bounds (on variables). \n
																 If no lower bounds exist, a NULL pointer can be passed. */
								const real_t* const _ub,	/**< Upper bounds (on variables). \n
																 If no upper bounds exist, a NULL pointer can be passed. */
								int* nWSR, 					/**< Input: Maximum number of working set recalculations when using initial homotopy. \n
																 Output: Number of performed working set recalculations. */
		 						real_t* const cputime 		/**< Input: Maximum CPU time allowed for QP initialisation. \n
																 Output: CPU time spent for QP initialisation (if pointer passed). */
								);

/** Initialises a simply bounded QP problem with given QP data and tries to solve it
 *	using at most nWSR iterations.
 *
 *  Note: This function internally calls solveInitialQP for initialisation!
 *
 *	\return SUCCESSFUL_RETURN \n
			RET_INIT_FAILED \n
			RET_INIT_FAILED_CHOLESKY \n
			RET_INIT_FAILED_HOTSTART \n
			RET_INIT_FAILED_INFEASIBILITY \n
			RET_INIT_FAILED_UNBOUNDEDNESS \n
			RET_MAX_NWSR_REACHED \n
			RET_INVALID_ARGUMENTS */
returnValue QProblemB_init(	QProblemB* _THIS,
							real_t* const _H, 			/**< Hessian matrix. \n
															 If Hessian matrix is trivial, a NULL pointer can be passed. */
							const real_t* const _g,		/**< Gradient vector. */
							const real_t* const _lb,	/**< Lower bounds (on variables). \n
															 If no lower bounds exist, a NULL pointer can be passed. */
							const real_t* const _ub,	/**< Upper bounds (on variables). \n
															 If no upper bounds exist, a NULL pointer can be passed. */
							int* nWSR, 					/**< Input: Maximum number of working set recalculations when using initial homotopy. \n
															 Output: Number of performed working set recalculations. */
		 					real_t* const cputime 		/**< Input: Maximum CPU time allowed for QP initialisation. \n
															 Output: CPU time spent for QP initialisation (if pointer passed). */
							);

/** Initialises a simply bounded QP problem with given QP data to be read from files and solves it
 *	using at most nWSR iterations.
 *
 *  Note: This function internally calls solveInitialQP for initialisation!
 *
 *	\return SUCCESSFUL_RETURN \n
			RET_INIT_FAILED \n
			RET_INIT_FAILED_CHOLESKY \n
			RET_INIT_FAILED_HOTSTART \n
			RET_INIT_FAILED_INFEASIBILITY \n
			RET_INIT_FAILED_UNBOUNDEDNESS \n
			RET_MAX_NWSR_REACHED \n
			RET_UNABLE_TO_READ_FILE */
returnValue QProblemB_initF(	QProblemB* _THIS,
								const char* const H_file, 	/**< Name of file where Hessian matrix is stored. \n
																 If Hessian matrix is trivial, a NULL pointer can be passed. */
								const char* const g_file,  	/**< Name of file where gradient vector is stored. */
								const char* const lb_file, 	/**< Name of file where lower bound vector. \n
																 If no lower bounds exist, a NULL pointer can be passed. */
								const char* const ub_file, 	/**< Name of file where upper bound vector. \n
																 If no upper bounds exist, a NULL pointer can be passed. */
								int* nWSR, 					/**< Input: Maximum number of working set recalculations when using initial homotopy. \n
																 Output: Number of performed working set recalculations. */
		 						real_t* const cputime 		/**< Input: Maximum CPU time allowed for QP initialisation. \n
																 Output: CPU time spent for QP initialisation (if pointer passed). */
								);

/** Initialises a simply bounded QP problem with given QP data and tries to solve it
 *	using at most nWSR iterations. Depending on the parameter constellation it: \n
 *	1. 0,    0,    0 : starts with xOpt = 0, yOpt = 0 and gB empty (or all implicit equality bounds), \n
 *	2. xOpt, 0,    0 : starts with xOpt, yOpt = 0 and obtain gB by "clipping", \n
 *	3. 0,    yOpt, 0 : starts with xOpt = 0, yOpt and obtain gB from yOpt != 0, \n
 *	4. 0,    0,    gB: starts with xOpt = 0, yOpt = 0 and gB, \n
 *	5. xOpt, yOpt, 0 : starts with xOpt, yOpt and obtain gB from yOpt != 0, \n
 *	6. xOpt, 0,    gB: starts with xOpt, yOpt = 0 and gB, \n
 *	7. xOpt, yOpt, gB: starts with xOpt, yOpt and gB (assume them to be consistent!)
 *
 *  Note: This function internally calls solveInitialQP for initialisation!
 *
 *	\return SUCCESSFUL_RETURN \n
			RET_INIT_FAILED \n
			RET_INIT_FAILED_CHOLESKY \n
			RET_INIT_FAILED_HOTSTART \n
			RET_INIT_FAILED_INFEASIBILITY \n
			RET_INIT_FAILED_UNBOUNDEDNESS \n
			RET_MAX_NWSR_REACHED \n
			RET_INVALID_ARGUMENTS */
returnValue QProblemB_initMW(	QProblemB* _THIS,
								DenseMatrix *_H,				/**< Hessian matrix. */
								const real_t* const _g,			/**< Gradient vector. */
								const real_t* const _lb,		/**< Lower bounds (on variables). \n
																	 If no lower bounds exist, a NULL pointer can be passed. */
								const real_t* const _ub,		/**< Upper bounds (on variables). \n
																	 If no upper bounds exist, a NULL pointer can be passed. */
								int* nWSR, 						/**< Input: Maximum number of working set recalculations when using initial homotopy. \n
																	 Output: Number of performed working set recalculations. */
		 						real_t* const cputime,			/**< Input: Maximum CPU time allowed for QP initialisation. \n
													 				 Output: CPU time spent for QP initialisation. */
								const real_t* const xOpt,		/**< Optimal primal solution vector. A NULL pointer can be passed. \n
																	 (If a null pointer is passed, the old primal solution is kept!) */
								const real_t* const yOpt,		/**< Optimal dual solution vector. A NULL pointer can be passed. \n
																	 (If a null pointer is passed, the old dual solution is kept!) */
								Bounds* const guessedBounds,	/**< Optimal working set of bounds for solution (xOpt,yOpt). \n
																	 (If a null pointer is passed, all bounds are assumed inactive!) */
								const real_t* const _R			/**< Pre-computed (upper triangular) Cholesky factor of Hessian matrix.
																 	 The Cholesky factor must be stored in a real_t array of size nV*nV
																	 in row-major format. Note: Only used if xOpt/yOpt and gB are NULL! \n
																	 (If a null pointer is passed, Cholesky decomposition is computed internally!) */
								);

/** Initialises a simply bounded QP problem with given QP data and tries to solve it
 *	using at most nWSR iterations. Depending on the parameter constellation it: \n
 *	1. 0,    0,    0 : starts with xOpt = 0, yOpt = 0 and gB empty (or all implicit equality bounds), \n
 *	2. xOpt, 0,    0 : starts with xOpt, yOpt = 0 and obtain gB by "clipping", \n
 *	3. 0,    yOpt, 0 : starts with xOpt = 0, yOpt and obtain gB from yOpt != 0, \n
 *	4. 0,    0,    gB: starts with xOpt = 0, yOpt = 0 and gB, \n
 *	5. xOpt, yOpt, 0 : starts with xOpt, yOpt and obtain gB from yOpt != 0, \n
 *	6. xOpt, 0,    gB: starts with xOpt, yOpt = 0 and gB, \n
 *	7. xOpt, yOpt, gB: starts with xOpt, yOpt and gB (assume them to be consistent!)
 *
 *  Note: This function internally calls solveInitialQP for initialisation!
 *
 *	\return SUCCESSFUL_RETURN \n
			RET_INIT_FAILED \n
			RET_INIT_FAILED_CHOLESKY \n
			RET_INIT_FAILED_HOTSTART \n
			RET_INIT_FAILED_INFEASIBILITY \n
			RET_INIT_FAILED_UNBOUNDEDNESS \n
			RET_MAX_NWSR_REACHED \n
			RET_INVALID_ARGUMENTS */
returnValue QProblemB_initW(	QProblemB* _THIS,
								real_t* const _H, 				/**< Hessian matrix. \n
																	 If Hessian matrix is trivial, a NULL pointer can be passed. */
								const real_t* const _g,			/**< Gradient vector. */
								const real_t* const _lb,		/**< Lower bounds (on variables). \n
																	 If no lower bounds exist, a NULL pointer can be passed. */
								const real_t* const _ub,		/**< Upper bounds (on variables). \n
																	 If no upper bounds exist, a NULL pointer can be passed. */
								int* nWSR, 						/**< Input: Maximum number of working set recalculations when using initial homotopy. \n
																	 Output: Number of performed working set recalculations. */
		 						real_t* const cputime,			/**< Input: Maximum CPU time allowed for QP initialisation. \n
													 				 Output: CPU time spent for QP initialisation. */
								const real_t* const xOpt,		/**< Optimal primal solution vector. A NULL pointer can be passed. \n
																	 (If a null pointer is passed, the old primal solution is kept!) */
								const real_t* const yOpt,		/**< Optimal dual solution vector. A NULL pointer can be passed. \n
																	 (If a null pointer is passed, the old dual solution is kept!) */
								Bounds* const guessedBounds,	/**< Optimal working set of bounds for solution (xOpt,yOpt). \n
																	 (If a null pointer is passed, all bounds are assumed inactive!) */
								const real_t* const _R			/**< Pre-computed (upper triangular) Cholesky factor of Hessian matrix.
																 	 The Cholesky factor must be stored in a real_t array of size nV*nV
																	 in row-major format. Note: Only used if xOpt/yOpt and gB are NULL! \n
																	 (If a null pointer is passed, Cholesky decomposition is computed internally!) */
								);

/** Initialises a simply bounded QP problem with given QP data to be read from files and solves it
 *	using at most nWSR iterations. Depending on the parameter constellation it: \n
 *	1. 0,    0,    0 : starts with xOpt = 0, yOpt = 0 and gB empty (or all implicit equality bounds), \n
 *	2. xOpt, 0,    0 : starts with xOpt, yOpt = 0 and obtain gB by "clipping", \n
 *	3. 0,    yOpt, 0 : starts with xOpt = 0, yOpt and obtain gB from yOpt != 0, \n
 *	4. 0,    0,    gB: starts with xOpt = 0, yOpt = 0 and gB, \n
 *	5. xOpt, yOpt, 0 : starts with xOpt, yOpt and obtain gB from yOpt != 0, \n
 *	6. xOpt, 0,    gB: starts with xOpt, yOpt = 0 and gB, \n
 *	7. xOpt, yOpt, gB: starts with xOpt, yOpt and gB (assume them to be consistent!)
 *
 *  Note: This function internally calls solveInitialQP for initialisation!
 *
 *	\return SUCCESSFUL_RETURN \n
			RET_INIT_FAILED \n
			RET_INIT_FAILED_CHOLESKY \n
			RET_INIT_FAILED_HOTSTART \n
			RET_INIT_FAILED_INFEASIBILITY \n
			RET_INIT_FAILED_UNBOUNDEDNESS \n
			RET_MAX_NWSR_REACHED \n
			RET_UNABLE_TO_READ_FILE */
returnValue QProblemB_initFW(	QProblemB* _THIS,
								const char* const H_file, 		/**< Name of file where Hessian matrix is stored. \n
																	 If Hessian matrix is trivial, a NULL pointer can be passed. */
								const char* const g_file,  		/**< Name of file where gradient vector is stored. */
								const char* const lb_file, 		/**< Name of file where lower bound vector. \n
																	 If no lower bounds exist, a NULL pointer can be passed. */
								const char* const ub_file, 		/**< Name of file where upper bound vector. \n
																	 If no upper bounds exist, a NULL pointer can be passed. */
								int* nWSR, 						/**< Input: Maximum number of working set recalculations when using initial homotopy. \n
																	 Output: Number of performed working set recalculations. */
		 						real_t* const cputime,			/**< Input: Maximum CPU time allowed for QP initialisation. \n
													 				 Output: CPU time spent for QP initialisation. */
								const real_t* const xOpt,		/**< Optimal primal solution vector. A NULL pointer can be passed. \n
																	 (If a null pointer is passed, the old primal solution is kept!) */
								const real_t* const yOpt,		/**< Optimal dual solution vector. A NULL pointer can be passed. \n
																	 (If a null pointer is passed, the old dual solution is kept!) */
								Bounds* const guessedBounds,	/**< Optimal working set of bounds for solution (xOpt,yOpt). \n
																	 (If a null pointer is passed, all bounds are assumed inactive!) */
								const char* const R_file		/**< Name of the file where a pre-computed (upper triangular) Cholesky factor
																	 of the Hessian matrix is stored. \n
																	 (If a null pointer is passed, Cholesky decomposition is computed internally!) */
								);


/** Solves an initialised QP sequence using the online active set strategy.
 *	By default, QP solution is started from previous solution.
 *
 *  Note: This function internally calls solveQP/solveRegularisedQP
 *        for solving an initialised QP!
 *
 *	\return SUCCESSFUL_RETURN \n
			RET_MAX_NWSR_REACHED \n
			RET_HOTSTART_FAILED_AS_QP_NOT_INITIALISED \n
			RET_HOTSTART_FAILED \n
			RET_SHIFT_DETERMINATION_FAILED \n
			RET_STEPDIRECTION_DETERMINATION_FAILED \n
			RET_STEPLENGTH_DETERMINATION_FAILED \n
			RET_HOMOTOPY_STEP_FAILED \n
			RET_HOTSTART_STOPPED_INFEASIBILITY \n
			RET_HOTSTART_STOPPED_UNBOUNDEDNESS */
returnValue QProblemB_hotstart(	QProblemB* _THIS,
								const real_t* const g_new,	/**< Gradient of neighbouring QP to be solved. */
								const real_t* const lb_new,	/**< Lower bounds of neighbouring QP to be solved. \n
											 					 If no lower bounds exist, a NULL pointer can be passed. */
								const real_t* const ub_new,	/**< Upper bounds of neighbouring QP to be solved. \n
											 					 If no upper bounds exist, a NULL pointer can be passed. */
								int* nWSR,					/**< Input: Maximum number of working set recalculations; \n
																 Output: Number of performed working set recalculations. */
								real_t* const cputime 		/**< Input: Maximum CPU time allowed for QP solution. \n
																 Output: CPU time spent for QP solution (or to perform nWSR iterations). */
								);

/** Solves an initialised QP sequence using the online active set strategy,
 *  where QP data is read from files. QP solution is started from previous solution.
 *
 *  Note: This function internally calls solveQP/solveRegularisedQP
 *        for solving an initialised QP!
 *
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
			RET_UNABLE_TO_READ_FILE \n
			RET_INVALID_ARGUMENTS */
returnValue QProblemB_hotstartF(	QProblemB* _THIS,
									const char* const g_file, 	/**< Name of file where gradient, of neighbouring QP to be solved, is stored. */
									const char* const lb_file, 	/**< Name of file where lower bounds, of neighbouring QP to be solved, is stored. \n
											 						 If no lower bounds exist, a NULL pointer can be passed. */
									const char* const ub_file, 	/**< Name of file where upper bounds, of neighbouring QP to be solved, is stored. \n
											 						 If no upper bounds exist, a NULL pointer can be passed. */
									int* nWSR, 					/**< Input: Maximum number of working set recalculations; \n
																	 Output: Number of performed working set recalculations. */
									real_t* const cputime 	 	/**< Input: Maximum CPU time allowed for QP solution. \n
																	 Output: CPU time spent for QP solution (or to perform nWSR iterations). */
									);

/** Solves an initialised QP sequence using the online active set strategy.
 *	By default, QP solution is started from previous solution. If a guess
 *	for the working set is provided, an initialised homotopy is performed.
 *
 *  Note: This function internally calls solveQP/solveRegularisedQP
 *        for solving an initialised QP!
 *
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
			RET_SETUP_AUXILIARYQP_FAILED */
returnValue QProblemB_hotstartW(	QProblemB* _THIS,
									const real_t* const g_new,	/**< Gradient of neighbouring QP to be solved. */
									const real_t* const lb_new,	/**< Lower bounds of neighbouring QP to be solved. \n
											 						 If no lower bounds exist, a NULL pointer can be passed. */
									const real_t* const ub_new,	/**< Upper bounds of neighbouring QP to be solved. \n
											 						 If no upper bounds exist, a NULL pointer can be passed. */
									int* nWSR,					/**< Input: Maximum number of working set recalculations; \n
																	 Output: Number of performed working set recalculations. */
									real_t* const cputime,		/**< Input: Maximum CPU time allowed for QP solution. \n
														 			 Output: CPU time spent for QP solution (or to perform nWSR iterations). */
									Bounds* const guessedBounds	/**< Optimal working set of bounds for solution (xOpt,yOpt). \n
																	 (If a null pointer is passed, the previous working set is kept!) */
									);

/** Solves an initialised QP sequence using the online active set strategy,
 *  where QP data is read from files.
 *	By default, QP solution is started from previous solution. If a guess
 *	for the working set is provided, an initialised homotopy is performed.
 *
 *  Note: This function internally calls solveQP/solveRegularisedQP
 *        for solving an initialised QP!
 *
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
			RET_UNABLE_TO_READ_FILE \n
			RET_INVALID_ARGUMENTS \n
			RET_SETUP_AUXILIARYQP_FAILED */
returnValue QProblemB_hotstartFW(	QProblemB* _THIS,
									const char* const g_file,	/**< Name of file where gradient, of neighbouring QP to be solved, is stored. */
									const char* const lb_file,	/**< Name of file where lower bounds, of neighbouring QP to be solved, is stored. \n
											 						 If no lower bounds exist, a NULL pointer can be passed. */
									const char* const ub_file, 	/**< Name of file where upper bounds, of neighbouring QP to be solved, is stored. \n
											 						 If no upper bounds exist, a NULL pointer can be passed. */
									int* nWSR,					/**< Input:  Maximum number of working set recalculations; \n
																	 Output: Number of performed working set recalculations. */
									real_t* const cputime,		/**< Input:  Maximum CPU time allowed for QP solution. \n
														 			 Output: CPU time spent for QP solution (or to perform nWSR iterations). */
									Bounds* const guessedBounds	/**< Optimal working set of bounds for solution (xOpt,yOpt). \n
																	 (If a null pointer is passed, the previous working set is kept!) */
									);


/** Writes a vector with the state of the working set
 *	\return SUCCESSFUL_RETURN \n
 *	        RET_INVALID_ARGUMENTS */
returnValue QProblemB_getWorkingSet(	QProblemB* _THIS,
										real_t* workingSet		/** Output: array containing state of the working set. */
										);

/** Writes a vector with the state of the working set of bounds
 *	\return SUCCESSFUL_RETURN \n
 *	        RET_INVALID_ARGUMENTS */
returnValue QProblemB_getWorkingSetBounds(	QProblemB* _THIS,
											real_t* workingSetB	/** Output: array containing state of the working set of bounds. */
											);

/** Writes a vector with the state of the working set of constraints
 *	\return SUCCESSFUL_RETURN \n
 *	        RET_INVALID_ARGUMENTS */
returnValue QProblemB_getWorkingSetConstraints(	QProblemB* _THIS,
												real_t* workingSetC	/** Output: array containing state of the working set of constraints. */
												);


/** Returns current bounds object of the QP (deep copy).
  *	\return SUCCESSFUL_RETURN \n
  			RET_QPOBJECT_NOT_SETUP */
static inline returnValue QProblemB_getBounds(	QProblemB* _THIS,
												Bounds* _bounds	/** Output: Bounds object. */
												);


/** Returns the number of variables.
 *	\return Number of variables. */
static inline int QProblemB_getNV( QProblemB* _THIS );

/** Returns the number of free variables.
 *	\return Number of free variables. */
static inline int QProblemB_getNFR( QProblemB* _THIS );

/** Returns the number of fixed variables.
 *	\return Number of fixed variables. */
static inline int QProblemB_getNFX( QProblemB* _THIS );

/** Returns the number of implicitly fixed variables.
 *	\return Number of implicitly fixed variables. */
static inline int QProblemB_getNFV( QProblemB* _THIS );

/** Returns the dimension of null space.
 *	\return Dimension of null space. */
int QProblemB_getNZ( QProblemB* _THIS );


/** Returns the optimal objective function value.
 *	\return finite value: Optimal objective function value (QP was solved) \n
 			+infinity:	  QP was not yet solved */
real_t QProblemB_getObjVal( QProblemB* _THIS );

/** Returns the objective function value at an arbitrary point x.
 *	\return Objective function value at point x */
real_t QProblemB_getObjValX(	QProblemB* _THIS,
								const real_t* const _x	/**< Point at which the objective function shall be evaluated. */
								);

/** Returns the primal solution vector.
 *	\return SUCCESSFUL_RETURN \n
			RET_QP_NOT_SOLVED */
returnValue QProblemB_getPrimalSolution(	QProblemB* _THIS,
											real_t* const xOpt			/**< Output: Primal solution vector (if QP has been solved). */
											);

/** Returns the dual solution vector.
 *	\return SUCCESSFUL_RETURN \n
			RET_QP_NOT_SOLVED */
returnValue QProblemB_getDualSolution(	QProblemB* _THIS,
										real_t* const yOpt	/**< Output: Dual solution vector (if QP has been solved). */
										);


/** Returns status of the solution process.
 *	\return Status of solution process. */
static inline QProblemStatus QProblemB_getStatus( QProblemB* _THIS );


/** Returns if the QProblem object is initialised.
 *	\return BT_TRUE:  QProblemB initialised \n
 			BT_FALSE: QProblemB not initialised */
static inline BooleanType QProblemB_isInitialised( QProblemB* _THIS );

/** Returns if the QP has been solved.
 *	\return BT_TRUE:  QProblemB solved \n
 			BT_FALSE: QProblemB not solved */
static inline BooleanType QProblemB_isSolved( QProblemB* _THIS );

/** Returns if the QP is infeasible.
 *	\return BT_TRUE:  QP infeasible \n
 			BT_FALSE: QP feasible (or not known to be infeasible!) */
static inline BooleanType QProblemB_isInfeasible( QProblemB* _THIS );

/** Returns if the QP is unbounded.
 *	\return BT_TRUE:  QP unbounded \n
 			BT_FALSE: QP unbounded (or not known to be unbounded!) */
static inline BooleanType QProblemB_isUnbounded( QProblemB* _THIS );


/** Returns Hessian type flag (type is not determined due to _THIS call!).
 *	\return Hessian type. */
static inline HessianType QProblemB_getHessianType( QProblemB* _THIS );

/** Changes the print level.
 *	\return SUCCESSFUL_RETURN */
static inline returnValue QProblemB_setHessianType(	QProblemB* _THIS,
													HessianType _hessianType /**< New Hessian type. */
													);

/** Returns if the QP has been internally regularised.
 *	\return BT_TRUE:  Hessian is internally regularised for QP solution \n
 			BT_FALSE: No internal Hessian regularisation is used for QP solution */
static inline BooleanType QProblemB_usingRegularisation( QProblemB* _THIS );

/** Returns current options struct.
 *	\return Current options struct. */
static inline Options QProblemB_getOptions( QProblemB* _THIS );

/** Overrides current options with given ones.
 *	\return SUCCESSFUL_RETURN */
static inline returnValue QProblemB_setOptions(	QProblemB* _THIS,
												Options _options	/**< New options. */
												);

/** Returns the print level.
 *	\return Print level. */
static inline PrintLevel QProblemB_getPrintLevel( QProblemB* _THIS );

/** Changes the print level.
 *	\return SUCCESSFUL_RETURN */
returnValue QProblemB_setPrintLevel(	QProblemB* _THIS,
										PrintLevel _printlevel	/**< New print level. */
										);

/** Returns the current number of QP problems solved.
 *	\return Number of QP problems solved. */
static inline unsigned int QProblemB_getCount( QProblemB* _THIS );

/** Resets QP problem counter (to zero).
 *	\return SUCCESSFUL_RETURN. */
static inline returnValue QProblemB_resetCounter( QProblemB* _THIS );


/** Prints concise list of properties of the current QP.
 *	\return  SUCCESSFUL_RETURN \n */
returnValue QProblemB_printProperties( QProblemB* _THIS );

/** Prints a list of all options and their current values.
 *	\return  SUCCESSFUL_RETURN \n */
returnValue QProblemB_printOptions( QProblemB* _THIS );


/** If Hessian type has been set by the user, nothing is done.
 *  Otherwise the Hessian type is set to HST_IDENTITY, HST_ZERO, or
 *  HST_POSDEF (default), respectively.
 *	\return SUCCESSFUL_RETURN \n
			RET_HESSIAN_INDEFINITE */
returnValue QProblemB_determineHessianType( QProblemB* _THIS );

/** Determines type of existing constraints and bounds (i.e. implicitly fixed, unbounded etc.).
 *	\return SUCCESSFUL_RETURN \n
			RET_SETUPSUBJECTTOTYPE_FAILED */
returnValue QProblemB_setupSubjectToType( QProblemB* _THIS );

/** Determines type of new constraints and bounds (i.e. implicitly fixed, unbounded etc.).
 *	\return SUCCESSFUL_RETURN \n
			RET_SETUPSUBJECTTOTYPE_FAILED */
returnValue QProblemB_setupSubjectToTypeNew(	QProblemB* _THIS,
												const real_t* const lb_new,	/**< New lower bounds. */
												const real_t* const ub_new	/**< New upper bounds. */
												);

/** Computes the Cholesky decomposition of the (simply projected) Hessian
 *  (i.e. R^T*R = Z^T*H*Z). It only works in the case where Z is a simple
 *  projection matrix!
 *  Note: If Hessian turns out not to be positive definite, the Hessian type
 *		  is set to HST_SEMIDEF accordingly.
 *	\return SUCCESSFUL_RETURN \n
 *			RET_HESSIAN_NOT_SPD \n
 *			RET_INDEXLIST_CORRUPTED */
returnValue QProblemB_computeCholesky( QProblemB* _THIS );

/** Computes initial Cholesky decomposition of the projected Hessian making
 *  use of the function setupCholeskyDecomposition() or setupCholeskyDecompositionProjected().
 *	\return SUCCESSFUL_RETURN \n
 *			RET_HESSIAN_NOT_SPD \n
 *			RET_INDEXLIST_CORRUPTED */
returnValue QProblemB_setupInitialCholesky( QProblemB* _THIS );


/** Obtains the desired working set for the auxiliary initial QP in
 *  accordance with the user specifications
 *	\return SUCCESSFUL_RETURN \n
			RET_OBTAINING_WORKINGSET_FAILED \n
			RET_INVALID_ARGUMENTS */
returnValue QProblemB_obtainAuxiliaryWorkingSet(	QProblemB* _THIS,
													const real_t* const xOpt,		/**< Optimal primal solution vector.
																					 *	 If a NULL pointer is passed, all entries are assumed to be zero. */
													const real_t* const yOpt,		/**< Optimal dual solution vector.
																					 *	 If a NULL pointer is passed, all entries are assumed to be zero. */
													Bounds* const guessedBounds,	/**< Guessed working set for solution (xOpt,yOpt). */
													Bounds* auxiliaryBounds			/**< Input: Allocated bound object. \n
																					 *	 Ouput: Working set for auxiliary QP. */
													);

/** Decides if lower bounds are smaller than upper bounds
 *
 * \return SUCCESSFUL_RETURN \n
 * 		   RET_QP_INFEASIBLE */
returnValue QProblemB_areBoundsConsistent(	QProblemB* _THIS,
											const real_t* const lb, /**< Vector of lower bounds*/
											const real_t* const ub  /**< Vector of upper bounds*/
											);

/** Solves the system Ra = b or R^Ta = b where R is an upper triangular matrix.
 *	\return SUCCESSFUL_RETURN \n
			RET_DIV_BY_ZERO */
returnValue QProblemB_backsolveR(	QProblemB* _THIS,
									const real_t* const b,	/**< Right hand side vector. */
									BooleanType transposed,	/**< Indicates if the transposed system shall be solved. */
									real_t* const a 		/**< Output: Solution vector */
									);

/** Solves the system Ra = b or R^Ta = b where R is an upper triangular matrix. \n
 *  Special variant for the case that _THIS function is called from within "removeBound()".
 *	\return SUCCESSFUL_RETURN \n
			RET_DIV_BY_ZERO */
returnValue QProblemB_backsolveRrem(	QProblemB* _THIS,
										const real_t* const b,		/**< Right hand side vector. */
										BooleanType transposed,		/**< Indicates if the transposed system shall be solved. */
										BooleanType removingBound,	/**< Indicates if function is called from "removeBound()". */
										real_t* const a 			/**< Output: Solution vector */
										);


/** Determines step direction of the shift of the QP data.
 *	\return SUCCESSFUL_RETURN */
returnValue QProblemB_determineDataShift(	QProblemB* _THIS,
											const real_t* const g_new,	/**< New gradient vector. */
											const real_t* const lb_new,	/**< New lower bounds. */
											const real_t* const ub_new,	/**< New upper bounds. */
											real_t* const delta_g,	 	/**< Output: Step direction of gradient vector. */
											real_t* const delta_lb,	 	/**< Output: Step direction of lower bounds. */
											real_t* const delta_ub,	 	/**< Output: Step direction of upper bounds. */
											BooleanType* Delta_bB_isZero/**< Output: Indicates if active bounds are to be shifted. */
											);


/** Sets up internal QP data.
 *	\return SUCCESSFUL_RETURN \n
			RET_INVALID_ARGUMENTS */
returnValue QProblemB_setupQPdataM(	QProblemB* _THIS,
									DenseMatrix *_H,	 		/**< Hessian matrix.*/
									const real_t* const _g,		/**< Gradient vector. */
									const real_t* const _lb,	/**< Lower bounds (on variables). \n
																	 If no lower bounds exist, a NULL pointer can be passed. */
									const real_t* const _ub		/**< Upper bounds (on variables). \n
																	 If no upper bounds exist, a NULL pointer can be passed. */
									);

/** Sets up internal QP data. If the current Hessian is trivial
 *  (i.e. HST_ZERO or HST_IDENTITY) but a non-trivial one is given,
 *  memory for Hessian is allocated and it is set to the given one.
 *	\return SUCCESSFUL_RETURN \n
			RET_INVALID_ARGUMENTS \n
			RET_NO_HESSIAN_SPECIFIED */
returnValue QProblemB_setupQPdata(	QProblemB* _THIS,
									real_t* const _H, 			/**< Hessian matrix. \n
																	 If Hessian matrix is trivial,a NULL pointer can be passed. */
									const real_t* const _g,		/**< Gradient vector. */
									const real_t* const _lb,	/**< Lower bounds (on variables). \n
																	 If no lower bounds exist, a NULL pointer can be passed. */
									const real_t* const _ub		/**< Upper bounds (on variables). \n
																	 If no upper bounds exist, a NULL pointer can be passed. */
									);

/** Sets up internal QP data by loading it from files. If the current Hessian
 *  is trivial (i.e. HST_ZERO or HST_IDENTITY) but a non-trivial one is given,
 *  memory for Hessian is allocated and it is set to the given one.
 *	\return SUCCESSFUL_RETURN \n
			RET_UNABLE_TO_OPEN_FILE \n
			RET_UNABLE_TO_READ_FILE \n
			RET_INVALID_ARGUMENTS \n
			RET_NO_HESSIAN_SPECIFIED */
returnValue QProblemB_setupQPdataFromFile(	QProblemB* _THIS,
											const char* const H_file, 	/**< Name of file where Hessian matrix, of neighbouring QP to be solved, is stored. \n
														     				 If Hessian matrix is trivial,a NULL pointer can be passed. */
											const char* const g_file, 	/**< Name of file where gradient, of neighbouring QP to be solved, is stored. */
											const char* const lb_file, 	/**< Name of file where lower bounds, of neighbouring QP to be solved, is stored. \n
										 			 						 If no lower bounds exist, a NULL pointer can be passed. */
											const char* const ub_file 	/**< Name of file where upper bounds, of neighbouring QP to be solved, is stored. \n
										 			 						 If no upper bounds exist, a NULL pointer can be passed. */
											);

/** Loads new QP vectors from files (internal members are not affected!).
 *	\return SUCCESSFUL_RETURN \n
			RET_UNABLE_TO_OPEN_FILE \n
			RET_UNABLE_TO_READ_FILE \n
			RET_INVALID_ARGUMENTS */
returnValue QProblemB_loadQPvectorsFromFile(	QProblemB* _THIS,
												const char* const g_file, 	/**< Name of file where gradient, of neighbouring QP to be solved, is stored. */
												const char* const lb_file, 	/**< Name of file where lower bounds, of neighbouring QP to be solved, is stored. \n
										 			 							 If no lower bounds exist, a NULL pointer can be passed. */
												const char* const ub_file, 	/**< Name of file where upper bounds, of neighbouring QP to be solved, is stored. \n
										 			 							 If no upper bounds exist, a NULL pointer can be passed. */
												real_t* const g_new,		/**< Output: Gradient of neighbouring QP to be solved. */
												real_t* const lb_new,		/**< Output: Lower bounds of neighbouring QP to be solved */
												real_t* const ub_new		/**< Output: Upper bounds of neighbouring QP to be solved */
												);


/** Sets internal infeasibility flag and throws given error in case the far bound
 *	strategy is not enabled (as QP might actually not be infeasible in _THIS case).
 *	\return RET_HOTSTART_STOPPED_INFEASIBILITY \n
			RET_ENSURELI_FAILED_CYCLING \n
			RET_ENSURELI_FAILED_NOINDEX */
returnValue QProblemB_setInfeasibilityFlag(	QProblemB* _THIS,
											returnValue returnvalue,	/**< Returnvalue to be tunneled. */
											BooleanType doThrowError	/**< Flag forcing to throw an error. */
											);


/** Determines if next QP iteration can be performed within given CPU time limit.
 *	\return BT_TRUE: CPU time limit is exceeded, stop QP solution. \n
			BT_FALSE: Sufficient CPU time for next QP iteration. */
BooleanType QProblemB_isCPUtimeLimitExceeded(	QProblemB* _THIS,
												const real_t* const cputime,	/**< Maximum CPU time allowed for QP solution. */
												real_t starttime,				/**< Start time of current QP solution. */
												int nWSR						/**< Number of working set recalculations performed so far. */
												);


/** Regularise Hessian matrix by adding a scaled identity matrix to it.
 *	\return SUCCESSFUL_RETURN \n
			RET_HESSIAN_ALREADY_REGULARISED */
returnValue QProblemB_regulariseHessian( QProblemB* _THIS );


/** Sets Hessian matrix of the QP.
 *	\return SUCCESSFUL_RETURN */
static inline returnValue QProblemB_setHM(	QProblemB* _THIS,
											DenseMatrix* H_new	/**< New Hessian matrix. */
											);

/** Sets dense Hessian matrix of the QP.
 *  If a null pointer is passed and
 *  a) hessianType is HST_IDENTITY, nothing is done,
 *  b) hessianType is not HST_IDENTITY, Hessian matrix is set to zero.
 *	\return SUCCESSFUL_RETURN */
static inline returnValue QProblemB_setH(	QProblemB* _THIS,
											real_t* const H_new	/**< New dense Hessian matrix (with correct dimension!). */
											);

/** Changes gradient vector of the QP.
 *	\return SUCCESSFUL_RETURN \n
 *			RET_INVALID_ARGUMENTS */
static inline returnValue QProblemB_setG(	QProblemB* _THIS,
											const real_t* const g_new	/**< New gradient vector (with correct dimension!). */
											);

/** Changes lower bound vector of the QP.
 *	\return SUCCESSFUL_RETURN \n
 *			RET_QPOBJECT_NOT_SETUP */
static inline returnValue QProblemB_setLB(	QProblemB* _THIS,
											const real_t* const lb_new	/**< New lower bound vector (with correct dimension!). */
											);

/** Changes single entry of lower bound vector of the QP.
 *	\return SUCCESSFUL_RETURN  \n
 *			RET_QPOBJECT_NOT_SETUP \n
 *			RET_INDEX_OUT_OF_BOUNDS */
static inline returnValue QProblemB_setLBn(	QProblemB* _THIS,
											int number,		/**< Number of entry to be changed. */
											real_t value	/**< New value for entry of lower bound vector. */
											);

/** Changes upper bound vector of the QP.
 *	\return SUCCESSFUL_RETURN \n
 *			RET_QPOBJECT_NOT_SETUP */
static inline returnValue QProblemB_setUB(	QProblemB* _THIS,
											const real_t* const ub_new	/**< New upper bound vector (with correct dimension!). */
											);

/** Changes single entry of upper bound vector of the QP.
 *	\return SUCCESSFUL_RETURN  \n
 *			RET_QPOBJECT_NOT_SETUP \n
 *			RET_INDEX_OUT_OF_BOUNDS */
static inline returnValue QProblemB_setUBn(	QProblemB* _THIS,
											int number,		/**< Number of entry to be changed. */
											real_t value	/**< New value for entry of upper bound vector. */
											);


/** Computes parameters for the Givens matrix G for which [x,y]*G = [z,0]
 *	\return SUCCESSFUL_RETURN */
static inline void QProblemB_computeGivens(	real_t xold,	/**< Matrix entry to be normalised. */
											real_t yold,	/**< Matrix entry to be annihilated. */
											real_t* xnew,	/**< Output: Normalised matrix entry. */
											real_t* ynew,	/**< Output: Annihilated matrix entry. */
											real_t* c,		/**< Output: Cosine entry of Givens matrix. */
											real_t* s 		/**< Output: Sine entry of Givens matrix. */
											);

/** Applies Givens matrix determined by c and s (cf. computeGivens).
 *	\return SUCCESSFUL_RETURN */
static inline void QProblemB_applyGivens(	real_t c,		/**< Cosine entry of Givens matrix. */
											real_t s,		/**< Sine entry of Givens matrix. */
											real_t nu, 		/**< Further factor: s/(1+c). */
											real_t xold,	/**< Matrix entry to be transformed corresponding to
															 *	 the normalised entry of the original matrix. */
											real_t yold, 	/**< Matrix entry to be transformed corresponding to
															 *	 the annihilated entry of the original matrix. */
											real_t* xnew,	/**< Output: Transformed matrix entry corresponding to
															 *	 the normalised entry of the original matrix. */
											real_t* ynew	/**< Output: Transformed matrix entry corresponding to
															 *	 the annihilated entry of the original matrix. */
											);



/** Compute relative length of homotopy in data space for termination
 *  criterion.
 *  \return Relative length in data space. */
real_t QProblemB_getRelativeHomotopyLength(	QProblemB* _THIS,
											const real_t* const g_new,	/**< Final gradient. */
											const real_t* const lb_new,	/**< Final lower variable bounds. */
											const real_t* const ub_new	/**< Final upper variable bounds. */
											);

/** Ramping Strategy to avoid ties. Modifies homotopy start without
 *  changing current active set.
 *  \return SUCCESSFUL_RETURN */
returnValue QProblemB_performRamping( QProblemB* _THIS );


/** ... */
returnValue QProblemB_updateFarBounds(	QProblemB* _THIS,
										real_t curFarBound,				/**< ... */
										int nRamp,						/**< ... */
										const real_t* const lb_new,		/**< ... */
										real_t* const lb_new_far,		/**< ... */
										const real_t* const ub_new,		/**< ... */
										real_t* const ub_new_far		/**< ... */
										);



/** Performs robustified ratio test yield the maximum possible step length
 *  along the homotopy path.
 *	\return  SUCCESSFUL_RETURN */
returnValue QProblemB_performRatioTestB(	QProblemB* _THIS,
											int nIdx, 					/**< Number of ratios to be checked. */
											const int* const idxList, 	/**< Array containing the indices of all ratios to be checked. */
											Bounds* const subjectTo,	/**< Bound object corresponding to ratios to be checked. */
											const real_t* const num,	/**< Array containing all numerators for performing the ratio test. */
											const real_t* const den,	/**< Array containing all denominators for performing the ratio test. */
											real_t epsNum,				/**< Numerator tolerance. */
											real_t epsDen,				/**< Denominator tolerance. */
											real_t* t,					/**< Output: Maximum possible step length along the homotopy path. */
											int* BC_idx 				/**< Output: Index of blocking constraint. */
											);

/** Checks whether given ratio is blocking, i.e. limits the maximum step length
 *  along the homotopy path to a value lower than given one.
 *	\return  SUCCESSFUL_RETURN */
static inline BooleanType QProblemB_isBlocking(	QProblemB* _THIS,
												real_t num,		/**< Numerator for performing the ratio test. */
												real_t den,		/**< Denominator for performing the ratio test. */
												real_t epsNum,	/**< Numerator tolerance. */
												real_t epsDen,	/**< Denominator tolerance. */
												real_t* t		/**< Input:  Current maximum step length along the homotopy path,
																 *   Output: Updated maximum possible step length along the homotopy path. */
												);


/** Solves a QProblemB whose QP data is assumed to be stored in the member variables.
 *  A guess for its primal/dual optimal solution vectors and the corresponding
 *  optimal working set can be provided.
 *  Note: This function is internally called by all init functions!
 *	\return SUCCESSFUL_RETURN \n
			RET_INIT_FAILED \n
			RET_INIT_FAILED_CHOLESKY \n
			RET_INIT_FAILED_HOTSTART \n
			RET_INIT_FAILED_INFEASIBILITY \n
			RET_INIT_FAILED_UNBOUNDEDNESS \n
			RET_MAX_NWSR_REACHED */
returnValue QProblemB_solveInitialQP(	QProblemB* _THIS,
										const real_t* const xOpt,		/**< Optimal primal solution vector.*/
										const real_t* const yOpt,		/**< Optimal dual solution vector. */
										Bounds* const guessedBounds,	/**< Optimal working set of bounds for solution (xOpt,yOpt). */
										const real_t* const _R,			/**< Pre-computed (upper triangular) Cholesky factor of Hessian matrix. */
										int* nWSR, 						/**< Input:  Maximum number of working set recalculations; \n
														 				 *	 Output: Number of performed working set recalculations. */
										real_t* const cputime			/**< Input:  Maximum CPU time allowed for QP solution. \n
														 				 *	 Output: CPU time spent for QP solution (or to perform nWSR iterations). */
										);

/** Solves an initialised QProblemB using online active set strategy.
 *  Note: This function is internally called by all hotstart functions!
 *	\return SUCCESSFUL_RETURN \n
			RET_MAX_NWSR_REACHED \n
			RET_HOTSTART_FAILED_AS_QP_NOT_INITIALISED \n
			RET_HOTSTART_FAILED \n
			RET_SHIFT_DETERMINATION_FAILED \n
			RET_STEPDIRECTION_DETERMINATION_FAILED \n
			RET_STEPLENGTH_DETERMINATION_FAILED \n
			RET_HOMOTOPY_STEP_FAILED \n
			RET_HOTSTART_STOPPED_INFEASIBILITY \n
			RET_HOTSTART_STOPPED_UNBOUNDEDNESS */
returnValue QProblemB_solveQP(	QProblemB* _THIS,
								const real_t* const g_new,	/**< Gradient of neighbouring QP to be solved. */
								const real_t* const lb_new,	/**< Lower bounds of neighbouring QP to be solved. \n
											 					 If no lower bounds exist, a NULL pointer can be passed. */
								const real_t* const ub_new,	/**< Upper bounds of neighbouring QP to be solved. \n
											 					 If no upper bounds exist, a NULL pointer can be passed. */
								int* nWSR,					/**< Input: Maximum number of working set recalculations; \n
																 Output: Number of performed working set recalculations. */
								real_t* const cputime,		/**< Input: Maximum CPU time allowed for QP solution. \n
																 Output: CPU time spent for QP solution (or to perform nWSR iterations). */
								int  nWSRperformed,			/**< Number of working set recalculations already performed to solve
																 this QP within previous solveQP() calls. This number is
																 always zero, except for successive calls from solveRegularisedQP()
																 or when using the far bound strategy. */
								BooleanType isFirstCall		/**< Indicating whether this is the first call for current QP. */
								);


/** Solves an initialised QProblemB using online active set strategy.
 *  Note: This function is internally called by all hotstart functions!
 *	\return SUCCESSFUL_RETURN \n
			RET_MAX_NWSR_REACHED \n
			RET_HOTSTART_FAILED_AS_QP_NOT_INITIALISED \n
			RET_HOTSTART_FAILED \n
			RET_SHIFT_DETERMINATION_FAILED \n
			RET_STEPDIRECTION_DETERMINATION_FAILED \n
			RET_STEPLENGTH_DETERMINATION_FAILED \n
			RET_HOMOTOPY_STEP_FAILED \n
			RET_HOTSTART_STOPPED_INFEASIBILITY \n
			RET_HOTSTART_STOPPED_UNBOUNDEDNESS */
returnValue QProblemB_solveRegularisedQP(	QProblemB* _THIS,
											const real_t* const g_new,	/**< Gradient of neighbouring QP to be solved. */
											const real_t* const lb_new,	/**< Lower bounds of neighbouring QP to be solved. \n
													 						 If no lower bounds exist, a NULL pointer can be passed. */
											const real_t* const ub_new,	/**< Upper bounds of neighbouring QP to be solved. \n
													 						 If no upper bounds exist, a NULL pointer can be passed. */
											int* nWSR,					/**< Input: Maximum number of working set recalculations; \n
																			 Output: Number of performed working set recalculations. */
											real_t* const cputime,		/**< Input: Maximum CPU time allowed for QP solution. \n
																			 Output: CPU time spent for QP solution (or to perform nWSR iterations). */
											int  nWSRperformed,			/**< Number of working set recalculations already performed to solve
																			 this QP within previous solveRegularisedQP() calls. This number is
																			 always zero, except for successive calls when using the far bound strategy. */
											BooleanType isFirstCall		/**< Indicating whether this is the first call for current QP. */
											);


/** Sets up bound data structure according to auxiliaryBounds.
 *  (If the working set shall be setup afresh, make sure that
 *  bounds data structure has been resetted!)
 *	\return SUCCESSFUL_RETURN \n
			RET_SETUP_WORKINGSET_FAILED \n
			RET_INVALID_ARGUMENTS \n
			RET_UNKNOWN_BUG */
returnValue QProblemB_setupAuxiliaryWorkingSet(	QProblemB* _THIS,
												Bounds* const auxiliaryBounds,	/**< Working set for auxiliary QP. */
												BooleanType setupAfresh			/**< Flag indicating if given working set shall be
																				 *    setup afresh or by updating the current one. */
												);

/** Sets up the optimal primal/dual solution of the auxiliary initial QP.
 *	\return SUCCESSFUL_RETURN */
returnValue QProblemB_setupAuxiliaryQPsolution(	QProblemB* _THIS,
												const real_t* const xOpt,	/**< Optimal primal solution vector.
																		 	 *	 If a NULL pointer is passed, all entries are set to zero. */
												const real_t* const yOpt	/**< Optimal dual solution vector.
																			 *	 If a NULL pointer is passed, all entries are set to zero. */
												);

/** Sets up gradient of the auxiliary initial QP for given
 *  optimal primal/dual solution and given initial working set
 *  (assumes that members X, Y and BOUNDS have already been (ialised!).
 *	\return SUCCESSFUL_RETURN */
returnValue QProblemB_setupAuxiliaryQPgradient( QProblemB* _THIS );

/** Sets up bounds of the auxiliary initial QP for given
 *  optimal primal/dual solution and given initial working set
 *  (assumes that members X, Y and BOUNDS have already been initialised!).
 *	\return SUCCESSFUL_RETURN \n
			RET_UNKNOWN_BUG */
returnValue QProblemB_setupAuxiliaryQPbounds(	QProblemB* _THIS,
												BooleanType useRelaxation	/**< Flag indicating if inactive bounds shall be relaxed. */
												);


/** Updates QP vectors, working sets and internal data structures in order to
	start from an optimal solution corresponding to initial guesses of the working
	set for bounds
 *	\return SUCCESSFUL_RETURN \n
 *			RET_SETUP_AUXILIARYQP_FAILED */
returnValue QProblemB_setupAuxiliaryQP(	QProblemB* _THIS,
										Bounds* const guessedBounds	/**< Initial guess for working set of bounds. */
										);

/** Determines step direction of the homotopy path.
 *	\return SUCCESSFUL_RETURN \n
 			RET_STEPDIRECTION_FAILED_CHOLESKY */
returnValue QProblemB_determineStepDirection(	QProblemB* _THIS,
												const real_t* const delta_g,	/**< Step direction of gradient vector. */
												const real_t* const delta_lb,	/**< Step direction of lower bounds. */
												const real_t* const delta_ub,	/**< Step direction of upper bounds. */
												BooleanType Delta_bB_isZero,	/**< Indicates if active bounds are to be shifted. */
												real_t* const delta_xFX, 		/**< Output: Primal homotopy step direction of fixed variables. */
												real_t* const delta_xFR,	 	/**< Output: Primal homotopy step direction of free variables. */
												real_t* const delta_yFX 		/**< Output: Dual homotopy step direction of fixed variables' multiplier. */
												);

/** Determines the maximum possible step length along the homotopy path
 *  and performs _THIS step (without changing working set).
 *	\return SUCCESSFUL_RETURN \n
 *			RET_QP_INFEASIBLE \n
 */
returnValue QProblemB_performStep(	QProblemB* _THIS,
									const real_t* const delta_g,	/**< Step direction of gradient. */
									const real_t* const delta_lb,	/**< Step direction of lower bounds. */
									const real_t* const delta_ub,	/**< Step direction of upper bounds. */
									const real_t* const delta_xFX, 	/**< Primal homotopy step direction of fixed variables. */
									const real_t* const delta_xFR,	/**< Primal homotopy step direction of free variables. */
									const real_t* const delta_yFX,	/**< Dual homotopy step direction of fixed variables' multiplier. */
									int* BC_idx, 					/**< Output: Index of blocking constraint. */
									SubjectToStatus* BC_status		/**< Output: Status of blocking constraint. */
									);

/** Updates active set.
 *	\return  SUCCESSFUL_RETURN \n
 			 RET_REMOVE_FROM_ACTIVESET_FAILED \n
			 RET_ADD_TO_ACTIVESET_FAILED */
returnValue QProblemB_changeActiveSet(	QProblemB* _THIS,
										int BC_idx, 				/**< Index of blocking constraint. */
										SubjectToStatus BC_status 	/**< Status of blocking constraint. */
										);

/** Drift correction at end of each active set iteration
 *  \return SUCCESSFUL_RETURN */
returnValue QProblemB_performDriftCorrection( QProblemB* _THIS );

/** Determines if it is more efficient to refactorise the matrices when
 *  hotstarting or not (i.e. better to update the existing factorisations).
 *	\return BT_TRUE iff matrices shall be refactorised afresh
 */
BooleanType QProblemB_shallRefactorise(	QProblemB* _THIS,
										Bounds* const guessedBounds	/**< Guessed new working set. */
										);


/** Adds a bound to active set (specialised version for the case where no constraints exist).
 *	\return SUCCESSFUL_RETURN \n
 			RET_ADDBOUND_FAILED */
returnValue QProblemB_addBound(	QProblemB* _THIS,
								int number,					/**< Number of bound to be added to active set. */
								SubjectToStatus B_status,	/**< Status of new active bound. */
								BooleanType updateCholesky	/**< Flag indicating if Cholesky decomposition shall be updated. */
								);

/** Removes a bounds from active set (specialised version for the case where no constraints exist).
 *	\return SUCCESSFUL_RETURN \n
			RET_HESSIAN_NOT_SPD \n
			RET_REMOVEBOUND_FAILED */
returnValue QProblemB_removeBound(	QProblemB* _THIS,
									int number,					/**< Number of bound to be removed from active set. */
									BooleanType updateCholesky	/**< Flag indicating if Cholesky decomposition shall be updated. */
									);


/** Prints concise information on the current iteration.
 *	\return  SUCCESSFUL_RETURN \n */
returnValue QProblemB_printIteration(	QProblemB* _THIS,
										int iter,					/**< Number of current iteration. */
										int BC_idx, 				/**< Index of blocking bound. */
										SubjectToStatus BC_status,	/**< Status of blocking bound. */
										real_t homotopyLength,		/**< Current homotopy distance. */
										BooleanType isFirstCall		/**< Indicating whether this is the first call for current QP. */
										);



/*
 *	g e t B o u n d s
 */
static inline returnValue QProblemB_getBounds( QProblemB* _THIS, Bounds* _bounds )
{
	int nV = QProblemB_getNV( _THIS );

	if ( nV == 0 )
		return THROWERROR( RET_QPOBJECT_NOT_SETUP );

	_bounds = _THIS->bounds;

	return SUCCESSFUL_RETURN;
}


/*
 *	g e t N V
 */
static inline int QProblemB_getNV( QProblemB* _THIS )
{
	return Bounds_getNV( _THIS->bounds );
}


/*
 *	g e t N F R
 */
static inline int QProblemB_getNFR( QProblemB* _THIS )
{
	return Bounds_getNFR( _THIS->bounds );
}


/*
 *	g e t N F X
 */
static inline int QProblemB_getNFX( QProblemB* _THIS )
{
	return Bounds_getNFX( _THIS->bounds );
}


/*
 *	g e t N F V
 */
static inline int QProblemB_getNFV( QProblemB* _THIS )
{
	return Bounds_getNFV( _THIS->bounds );
}


/*
 *	g e t S t a t u s
 */
static inline QProblemStatus QProblemB_getStatus( QProblemB* _THIS )
{
	return _THIS->status;
}


/*
 *	i s I n i t i a l i s e d
 */
static inline BooleanType QProblemB_isInitialised( QProblemB* _THIS )
{
	if ( _THIS->status == QPS_NOTINITIALISED )
		return BT_FALSE;
	else
		return BT_TRUE;
}


/*
 *	i s S o l v e d
 */
static inline BooleanType QProblemB_isSolved( QProblemB* _THIS )
{
	if ( _THIS->status == QPS_SOLVED )
		return BT_TRUE;
	else
		return BT_FALSE;
}


/*
 *	i s I n f e a s i b l e
 */
static inline BooleanType QProblemB_isInfeasible( QProblemB* _THIS )
{
	return _THIS->infeasible;
}


/*
 *	i s U n b o u n d e d
 */
static inline BooleanType QProblemB_isUnbounded( QProblemB* _THIS )
{
	return _THIS->unbounded;
}


/*
 *	g e t H e s s i a n T y p e
 */
static inline HessianType QProblemB_getHessianType( QProblemB* _THIS )
{
	return _THIS->hessianType;
}


/*
 *	s e t H e s s i a n T y p e
 */
static inline returnValue QProblemB_setHessianType( QProblemB* _THIS, HessianType _hessianType )
{
	_THIS->hessianType = _hessianType;
	return SUCCESSFUL_RETURN;
}


/*
 *	u s i n g R e g u l a r i s a t i o n
 */
static inline BooleanType QProblemB_usingRegularisation( QProblemB* _THIS )
{
	if ( _THIS->regVal > QPOASES_ZERO )
		return BT_TRUE;
	else
		return BT_FALSE;
}


/*
 *	g e t O p t i o n s
 */
static inline Options QProblemB_getOptions( QProblemB* _THIS )
{
	return _THIS->options;
}


/*
 *	s e t O p t i o n s
 */
static inline returnValue QProblemB_setOptions(	QProblemB* _THIS,
												Options _options
												)
{
	OptionsCPY( &_options,&(_THIS->options) );
	Options_ensureConsistency( &(_THIS->options) );

	QProblemB_setPrintLevel( _THIS,_THIS->options.printLevel );

	return SUCCESSFUL_RETURN;
}


/*
 *	g e t P r i n t L e v e l
 */
static inline PrintLevel QProblemB_getPrintLevel( QProblemB* _THIS )
{
	return _THIS->options.printLevel;
}



/*
 *	g e t C o u n t
 */
static inline unsigned int QProblemB_getCount( QProblemB* _THIS )
{
	return _THIS->count;
}


/*
 *	r e s e t C o u n t e r
 */
static inline returnValue QProblemB_resetCounter( QProblemB* _THIS )
{
	_THIS->count = 0;
	return SUCCESSFUL_RETURN;
}



/*****************************************************************************
 *  P R O T E C T E D                                                        *
 *****************************************************************************/


/*
 *	s e t H
 */
static inline returnValue QProblemB_setHM( QProblemB* _THIS, DenseMatrix* H_new )
{
	if ( H_new == 0 )
		return QProblemB_setH( _THIS,(real_t*)0 );
	else
		return QProblemB_setH( _THIS,DenseMatrix_getVal(H_new) );
}


/*
 *	s e t H
 */
static inline returnValue QProblemB_setH( QProblemB* _THIS, real_t* const H_new )
{
	/* if null pointer is passed, Hessian is set to zero matrix
	 *                            (or stays identity matrix) */
	if ( H_new == 0 )
	{
		if ( _THIS->hessianType == HST_IDENTITY )
			return SUCCESSFUL_RETURN;

		_THIS->hessianType = HST_ZERO;

		_THIS->H = 0;
	}
	else
	{
		DenseMatrixCON( _THIS->H,QProblemB_getNV( _THIS ),QProblemB_getNV( _THIS ),QProblemB_getNV( _THIS ),H_new );
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t G
 */
static inline returnValue QProblemB_setG( QProblemB* _THIS, const real_t* const g_new )
{
	unsigned int nV = (unsigned int)QProblemB_getNV( _THIS );

	if ( nV == 0 )
		return THROWERROR( RET_QPOBJECT_NOT_SETUP );

	if ( g_new == 0 )
		return THROWERROR( RET_INVALID_ARGUMENTS );

	memcpy( _THIS->g,g_new,nV*sizeof(real_t) );

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t L B
 */
static inline returnValue QProblemB_setLB( QProblemB* _THIS, const real_t* const lb_new )
{
	unsigned int i;
	unsigned int nV = (unsigned int)QProblemB_getNV( _THIS );

	if ( nV == 0 )
		return THROWERROR( RET_QPOBJECT_NOT_SETUP );

	if ( lb_new != 0 )
	{
		memcpy( _THIS->lb,lb_new,nV*sizeof(real_t) );
	}
	else
	{
		/* if no lower bounds are specified, set them to -infinity */
		for( i=0; i<nV; ++i )
			_THIS->lb[i] = -QPOASES_INFTY;
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t L B
 */
static inline returnValue QProblemB_setLBn( QProblemB* _THIS, int number, real_t value )
{
	int nV = QProblemB_getNV( _THIS );

	if ( nV == 0 )
		return THROWERROR( RET_QPOBJECT_NOT_SETUP );

	if ( ( number >= 0 ) && ( number < nV ) )
	{
		_THIS->lb[number] = value;
		return SUCCESSFUL_RETURN;
	}
	else
	{
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );
	}
}


/*
 *	s e t U B
 */
static inline returnValue QProblemB_setUB( QProblemB* _THIS, const real_t* const ub_new )
{
	unsigned int i;
	unsigned int nV = (unsigned int)QProblemB_getNV( _THIS );

	if ( nV == 0 )
		return THROWERROR( RET_QPOBJECT_NOT_SETUP );

	if ( ub_new != 0 )
	{
		memcpy( _THIS->ub,ub_new,nV*sizeof(real_t) );
	}
	else
	{
		/* if no upper bounds are specified, set them to infinity */
		for( i=0; i<nV; ++i )
			_THIS->ub[i] = QPOASES_INFTY;
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t U B
 */
static inline returnValue QProblemB_setUBn( QProblemB* _THIS, int number, real_t value )
{
	int nV = QProblemB_getNV( _THIS );

	if ( nV == 0 )
		return THROWERROR( RET_QPOBJECT_NOT_SETUP );

	if ( ( number >= 0 ) && ( number < nV ) )
	{
		_THIS->ub[number] = value;

		return SUCCESSFUL_RETURN;
	}
	else
	{
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );
	}
}


/*
 *	c o m p u t e G i v e n s
 */
static inline void QProblemB_computeGivens(	real_t xold, real_t yold,
											real_t* xnew, real_t* ynew, real_t* c, real_t* s
											)
{
	real_t t, mu;

	if ( fabs( yold ) <= QPOASES_ZERO )
	{
		*c = 1.0;
		*s = 0.0;

		*xnew = xold;
		*ynew = yold;
	}
	else
	{
		mu = fabs( xold );
		if ( fabs( yold ) > mu )
			mu = fabs( yold );

		t = mu * sqrt( (xold/mu)*(xold/mu) + (yold/mu)*(yold/mu) );

		if ( xold < 0.0 )
		t = -t;

		*c = xold/t;
		*s = yold/t;
		*xnew = t;
		*ynew = 0.0;
	}

	return;
}


/*
 *	a p p l y G i v e n s
 */
static inline void QProblemB_applyGivens(	real_t c, real_t s, real_t nu, real_t xold, real_t yold,
											real_t* xnew, real_t* ynew
											)
{
	#ifdef __USE_THREE_MULTS_GIVENS__

	/* Givens plane rotation requiring only three multiplications,
	 * cf. Hammarling, S.: A note on modifications to the givens plane rotation.
	 * J. Inst. Maths Applics, 13:215-218, 1974. */
	*xnew = xold*c + yold*s;
	*ynew = (*xnew+xold)*nu - yold;

	#else

	/* Usual Givens plane rotation requiring four multiplications. */
	*xnew =  c*xold + s*yold;
	*ynew = -s*xold + c*yold;

	#endif

	return;
}


/*
 * i s B l o c k i n g
 */
static inline BooleanType QProblemB_isBlocking(	QProblemB* _THIS,
												real_t num,
												real_t den,
												real_t epsNum,
												real_t epsDen,
												real_t* t
												)
{
	if ( ( den >= epsDen ) && ( num >= epsNum ) )
	{
		if ( num < (*t)*den )
			return BT_TRUE;
	}

	return BT_FALSE;
}


END_NAMESPACE_QPOASES


#endif	/* QPOASES_QPROBLEMB_H */


/*
 *	end of file
 */
