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
 *	\file include/qpOASES_e/QProblem.h
 *	\author Hans Joachim Ferreau, Andreas Potschka, Christian Kirches
 *	\version 3.1embedded
 *	\date 2007-2015
 *
 *	Declaration of the QProblem class which is able to use the newly
 *	developed online active set strategy for parametric quadratic programming.
 */



#ifndef QPOASES_QPROBLEM_H
#define QPOASES_QPROBLEM_H


#include <qpOASES_e/Bounds.h>
#include <qpOASES_e/Options.h>
#include <qpOASES_e/Matrices.h>
#include <qpOASES_e/Constraints.h>
#include <qpOASES_e/Flipper.h>
#include <qpOASES_e/ConstraintProduct.h>


BEGIN_NAMESPACE_QPOASES

typedef struct {
	Bounds *auxiliaryBounds;
	Constraints *auxiliaryConstraints;

	real_t *ub_new_far;
	real_t *lb_new_far;
	real_t *ubA_new_far;
	real_t *lbA_new_far;

	real_t *g_new;
	real_t *lb_new;
	real_t *ub_new;
	real_t *lbA_new;
	real_t *ubA_new;

	real_t *g_new2;
	real_t *lb_new2;
	real_t *ub_new2;
	real_t *lbA_new2;
	real_t *ubA_new2;

	real_t *delta_xFX5;
	real_t *delta_xFR5;
	real_t *delta_yAC5;
	real_t *delta_yFX5;

	real_t *Hx;

	real_t *_H;

	real_t *g_original;
	real_t *lb_original;
	real_t *ub_original;
	real_t *lbA_original;
	real_t *ubA_original;

	real_t *delta_xFR;
	real_t *delta_xFX;
	real_t *delta_yAC;
	real_t *delta_yFX;
	real_t *delta_g;
	real_t *delta_lb;
	real_t *delta_ub;
	real_t *delta_lbA;
	real_t *delta_ubA;

	real_t *gMod;

	real_t *aFR;
	real_t *wZ;

	real_t *delta_g2;
	real_t *delta_xFX2;
	real_t *delta_xFR2;
	real_t *delta_yAC2;
	real_t *delta_yFX2;
	real_t *nul;
	real_t *Arow;

	real_t *xiC;
	real_t *xiC_TMP;
	real_t *xiB;
	real_t *Arow2;
	real_t *num;

	real_t *w;
	real_t *tmp;

	real_t *delta_g3;
	real_t *delta_xFX3;
	real_t *delta_xFR3;
	real_t *delta_yAC3;
	real_t *delta_yFX3;
	real_t *nul2;

	real_t *xiC2;
	real_t *xiC_TMP2;
	real_t *xiB2;
	real_t *num2;

	real_t *Hz;
	real_t *z;
	real_t *ZHz;
	real_t *r;

	real_t *tmp2;
	real_t *Hz2;
	real_t *z2;
	real_t *r2;
	real_t *rhs;

	real_t *delta_xFX4;
	real_t *delta_xFR4;
	real_t *delta_yAC4;
	real_t *delta_yFX4;
	real_t *nul3;
	real_t *ek;
	real_t *x_W;
	real_t *As;
	real_t *Ax_W;

	real_t *num3;
	real_t *den;
	real_t *delta_Ax_l;
	real_t *delta_Ax_u;
	real_t *delta_Ax;
	real_t *delta_x;

	real_t *_A;

	real_t *grad;
	real_t *AX;
} QProblem_ws;

int QProblem_ws_calculateMemorySize( unsigned int nV, unsigned int nC );

char *QProblem_ws_assignMemory( unsigned int nV, unsigned int nC, QProblem_ws **mem, void *raw_memory );

QProblem_ws *QProblem_ws_createMemory( unsigned int nV, unsigned int nC );

/**
 *	\brief Implements the online active set strategy for QPs with general constraints.
 *
 *	A class for setting up and solving quadratic programs. The main feature is
 *	the possibily to use the newly developed online active set strategy for
 * 	parametric quadratic programming.
 *
 *	\author Hans Joachim Ferreau, Andreas Potschka, Christian Kirches
 *	\version 3.1embedded
 *	\date 2007-2015
 */
typedef struct
{
	QProblem_ws *ws;				/**< Workspace */
	Bounds *bounds;					/**< Data structure for problem's bounds. */
	Constraints *constraints;		/**< Data structure for problem's constraints. */
	Flipper *flipper;				/**< Struct for making a temporary copy of the matrix factorisations. */

	DenseMatrix* H;					/**< Hessian matrix pointer. */
	DenseMatrix* A;					/**< Constraint matrix pointer. */

	Options options;				/**< Struct containing all user-defined options for solving QPs. */
	TabularOutput tabularOutput;	/**< Struct storing information for tabular output (printLevel == PL_TABULAR). */

	real_t *g;						/**< Gradient. */

	real_t *lb;						/**< Lower bound vector (on variables). */
	real_t *ub;						/**< Upper bound vector (on variables). */
	real_t *lbA;					/**< Lower constraints' bound vector. */
	real_t *ubA;					/**< Upper constraints' bound vector. */

	real_t *R;						/**< Cholesky factor of H (i.e. H = R^T*R). */

	real_t *T;						/**< Reverse triangular matrix, A = [0 T]*Q'. */
	real_t *Q;						/**< Orthonormal quadratic matrix, A = [0 T]*Q'. */

	real_t *Ax;						/**< Stores the current A*x \n
									 *	 (for increased efficiency only). */
	real_t *Ax_l;					/**< Stores the current distance to lower constraints' bounds A*x-lbA \n
									 *	 (for increased efficiency only). */
	real_t *Ax_u;					/**< Stores the current distance to lower constraints' bounds ubA-A*x \n
									 *	 (for increased efficiency only). */

	real_t *x;						/**< Primal solution vector. */
	real_t *y;						/**< Dual solution vector. */

	real_t *delta_xFR_TMP;			/**< Temporary for determineStepDirection */
	real_t *tempA;					/**< Temporary for determineStepDirection. */
	real_t *tempB;					/**< Temporary for determineStepDirection. */
	real_t *ZFR_delta_xFRz;			/**< Temporary for determineStepDirection. */
	real_t *delta_xFRy;				/**< Temporary for determineStepDirection. */
	real_t *delta_xFRz;				/**< Temporary for determineStepDirection. */
	real_t *delta_yAC_TMP;			/**< Temporary for determineStepDirection. */

	ConstraintProduct constraintProduct;	/**< Pointer to user-defined constraint product function. */

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

	int sizeT;						/**< Matrix T is stored in a (sizeT x sizeT) array. */
} QProblem;

int QProblem_calculateMemorySize( unsigned int nV, unsigned int nC );

char *QProblem_assignMemory( unsigned int nV, unsigned int nC, QProblem **mem, void *raw_memory );

QProblem *QProblem_createMemory( unsigned int nV, unsigned int nC );


/** Constructor which takes the QP dimension and Hessian type
 *  information. If the Hessian is the zero (i.e. HST_ZERO) or the
 *  identity matrix (i.e. HST_IDENTITY), respectively, no memory
 *  is allocated for it and a NULL pointer can be passed for it
 *  to the init() functions. */
void QProblemCON(	QProblem* _THIS,
					int _nV,	  					/**< Number of variables. */
					int _nC,		  				/**< Number of constraints. */
					HessianType _hessianType 		/**< Type of Hessian matrix. */
					);

/** Copies all members from given rhs object.
 *  \return SUCCESSFUL_RETURN */
void QProblemCPY(	QProblem* FROM,
					QProblem* TO
					);


/** Clears all data structures of QProblem except for QP data.
 *	\return SUCCESSFUL_RETURN \n
			RET_RESET_FAILED */
returnValue QProblem_reset( QProblem* _THIS );


/** Initialises a QP problem with given QP data and tries to solve it
 *	using at most nWSR iterations.
 *
 *  Note: This function internally calls solveInitialQP for initialisation!
 *
 *	\return SUCCESSFUL_RETURN \n
			RET_INIT_FAILED \n
			RET_INIT_FAILED_CHOLESKY \n
			RET_INIT_FAILED_TQ \n
			RET_INIT_FAILED_HOTSTART \n
			RET_INIT_FAILED_INFEASIBILITY \n
			RET_INIT_FAILED_UNBOUNDEDNESS \n
			RET_MAX_NWSR_REACHED \n
			RET_INVALID_ARGUMENTS */
returnValue QProblem_initM(	QProblem* _THIS,
							DenseMatrix *_H,			/**< Hessian matrix. */
							const real_t* const _g, 	/**< Gradient vector. */
							DenseMatrix *_A,  			/**< Constraint matrix. */
							const real_t* const _lb,	/**< Lower bound vector (on variables). \n
															 If no lower bounds exist, a NULL pointer can be passed. */
							const real_t* const _ub,	/**< Upper bound vector (on variables). \n
															 If no upper bounds exist, a NULL pointer can be passed. */
							const real_t* const _lbA,	/**< Lower constraints' bound vector. \n
															 If no lower constraints' bounds exist, a NULL pointer can be passed. */
							const real_t* const _ubA,	/**< Upper constraints' bound vector. \n
															 If no lower constraints' bounds exist, a NULL pointer can be passed. */
							int* nWSR,					/**< Input: Maximum number of working set recalculations when using initial homotopy.
															 Output: Number of performed working set recalculations. */
							real_t* const cputime 		/**< Input: Maximum CPU time allowed for QP initialisation. \n
															 Output: CPU time spent for QP initialisation (if pointer passed). */
							);


/** Initialises a QP problem with given QP data and tries to solve it
 *	using at most nWSR iterations.
 *
 *  Note: This function internally calls solveInitialQP for initialisation!
 *
 *	\return SUCCESSFUL_RETURN \n
			RET_INIT_FAILED \n
			RET_INIT_FAILED_CHOLESKY \n
			RET_INIT_FAILED_TQ \n
			RET_INIT_FAILED_HOTSTART \n
			RET_INIT_FAILED_INFEASIBILITY \n
			RET_INIT_FAILED_UNBOUNDEDNESS \n
			RET_MAX_NWSR_REACHED \n
			RET_INVALID_ARGUMENTS */
returnValue QProblem_init(	QProblem* _THIS,
							real_t* const _H, 			/**< Hessian matrix. \n
															 If Hessian matrix is trivial, a NULL pointer can be passed. */
							const real_t* const _g, 	/**< Gradient vector. */
							real_t* const _A,  			/**< Constraint matrix. */
							const real_t* const _lb,	/**< Lower bound vector (on variables). \n
															 If no lower bounds exist, a NULL pointer can be passed. */
							const real_t* const _ub,	/**< Upper bound vector (on variables). \n
															 If no upper bounds exist, a NULL pointer can be passed. */
							const real_t* const _lbA,	/**< Lower constraints' bound vector. \n
															 If no lower constraints' bounds exist, a NULL pointer can be passed. */
							const real_t* const _ubA,	/**< Upper constraints' bound vector. \n
															 If no lower constraints' bounds exist, a NULL pointer can be passed. */
							int* nWSR,					/**< Input: Maximum number of working set recalculations when using initial homotopy.
															 Output: Number of performed working set recalculations. */
							real_t* const cputime 		/**< Input: Maximum CPU time allowed for QP initialisation. \n
															 Output: CPU time spent for QP initialisation (if pointer passed). */
							);

/** Initialises a QP problem with given QP data to be read from files and tries to solve it
 *	using at most nWSR iterations.
 *
 *  Note: This function internally calls solveInitialQP for initialisation!
 *
 *	\return SUCCESSFUL_RETURN \n
			RET_INIT_FAILED \n
			RET_INIT_FAILED_CHOLESKY \n
			RET_INIT_FAILED_TQ \n
			RET_INIT_FAILED_HOTSTART \n
			RET_INIT_FAILED_INFEASIBILITY \n
			RET_INIT_FAILED_UNBOUNDEDNESS \n
			RET_MAX_NWSR_REACHED \n
			RET_UNABLE_TO_READ_FILE */
returnValue QProblem_initF(	QProblem* _THIS,
							const char* const H_file,	/**< Name of file where Hessian matrix is stored. \n
															 If Hessian matrix is trivial, a NULL pointer can be passed. */
							const char* const g_file,	/**< Name of file where gradient vector is stored. */
							const char* const A_file,	/**< Name of file where constraint matrix is stored. */
							const char* const lb_file,	/**< Name of file where lower bound vector. \n
															If no lower bounds exist, a NULL pointer can be passed. */
							const char* const ub_file,	/**< Name of file where upper bound vector. \n
															If no upper bounds exist, a NULL pointer can be passed. */
							const char* const lbA_file,	/**< Name of file where lower constraints' bound vector. \n
															If no lower constraints' bounds exist, a NULL pointer can be passed. */
							const char* const ubA_file,	/**< Name of file where upper constraints' bound vector. \n
															 If no upper constraints' bounds exist, a NULL pointer can be passed. */
							int* nWSR,					/**< Input: Maximum number of working set recalculations when using initial homotopy.
															 Output: Number of performed working set recalculations. */
							real_t* const cputime	 	/**< Input: Maximum CPU time allowed for QP initialisation. \n
															 Output: CPU time spent for QP initialisation (if pointer passed). */
							);

/** Initialises a QP problem with given QP data and tries to solve it
 *	using at most nWSR iterations. Depending on the parameter constellation it: \n
 *	1. 0,    0,    0    : starts with xOpt = 0, yOpt = 0 and gB/gC empty (or all implicit equality bounds), \n
 *	2. xOpt, 0,    0    : starts with xOpt, yOpt = 0 and obtain gB/gC by "clipping", \n
 *	3. 0,    yOpt, 0    : starts with xOpt = 0, yOpt and obtain gB/gC from yOpt != 0, \n
 *	4. 0,    0,    gB/gC: starts with xOpt = 0, yOpt = 0 and gB/gC, \n
 *	5. xOpt, yOpt, 0    : starts with xOpt, yOpt and obtain gB/gC from yOpt != 0, \n
 *	6. xOpt, 0,    gB/gC: starts with xOpt, yOpt = 0 and gB/gC, \n
 *	7. xOpt, yOpt, gB/gC: starts with xOpt, yOpt and gB/gC (assume them to be consistent!)
 *
 *  Note: This function internally calls solveInitialQP for initialisation!
 *
 *	\return SUCCESSFUL_RETURN \n
			RET_INIT_FAILED \n
			RET_INIT_FAILED_CHOLESKY \n
			RET_INIT_FAILED_TQ \n
			RET_INIT_FAILED_HOTSTART \n
			RET_INIT_FAILED_INFEASIBILITY \n
			RET_INIT_FAILED_UNBOUNDEDNESS \n
			RET_MAX_NWSR_REACHED \n
			RET_INVALID_ARGUMENTS */
returnValue QProblem_initMW(	QProblem* _THIS,
								DenseMatrix *_H, 						/**< Hessian matrix. \n
														    				 If Hessian matrix is trivial, a NULL pointer can be passed. */
								const real_t* const _g, 				/**< Gradient vector. */
								DenseMatrix *_A, 			 			/**< Constraint matrix. */
								const real_t* const _lb,				/**< Lower bound vector (on variables). \n
																			 If no lower bounds exist, a NULL pointer can be passed. */
								const real_t* const _ub,				/**< Upper bound vector (on variables). \n
																			 If no upper bounds exist, a NULL pointer can be passed. */
								const real_t* const _lbA,				/**< Lower constraints' bound vector. \n
																			 If no lower constraints' bounds exist, a NULL pointer can be passed. */
								const real_t* const _ubA,				/**< Upper constraints' bound vector. \n
																			 If no lower constraints' bounds exist, a NULL pointer can be passed. */
								int* nWSR,								/**< Input: Maximum number of working set recalculations when using initial homotopy.
																		 *	 Output: Number of performed working set recalculations. */
								real_t* const cputime,					/**< Input: Maximum CPU time allowed for QP initialisation. \n
																			 Output: CPU time spent for QP initialisation. */
								const real_t* const xOpt,				/**< Optimal primal solution vector. \n
																			 (If a null pointer is passed, the old primal solution is kept!) */
								const real_t* const yOpt,				/**< Optimal dual solution vector. \n
																			 (If a null pointer is passed, the old dual solution is kept!) */
								Bounds* const guessedBounds,			/**< Optimal working set of bounds for solution (xOpt,yOpt). */
								Constraints* const guessedConstraints,	/**< Optimal working set of constraints for solution (xOpt,yOpt). */
								const real_t* const _R					/**< Pre-computed (upper triangular) Cholesky factor of Hessian matrix.
																	 		 The Cholesky factor must be stored in a real_t array of size nV*nV
																			 in row-major format. Note: Only used if xOpt/yOpt and gB are NULL! \n
																			 (If a null pointer is passed, Cholesky decomposition is computed internally!) */
								);

/** Initialises a QP problem with given QP data and tries to solve it
 *	using at most nWSR iterations. Depending on the parameter constellation it: \n
 *	1. 0,    0,    0    : starts with xOpt = 0, yOpt = 0 and gB/gC empty (or all implicit equality bounds), \n
 *	2. xOpt, 0,    0    : starts with xOpt, yOpt = 0 and obtain gB/gC by "clipping", \n
 *	3. 0,    yOpt, 0    : starts with xOpt = 0, yOpt and obtain gB/gC from yOpt != 0, \n
 *	4. 0,    0,    gB/gC: starts with xOpt = 0, yOpt = 0 and gB/gC, \n
 *	5. xOpt, yOpt, 0    : starts with xOpt, yOpt and obtain gB/gC from yOpt != 0, \n
 *	6. xOpt, 0,    gB/gC: starts with xOpt, yOpt = 0 and gB/gC, \n
 *	7. xOpt, yOpt, gB/gC: starts with xOpt, yOpt and gB/gC (assume them to be consistent!)
 *
 *  Note: This function internally calls solveInitialQP for initialisation!
 *
 *	\return SUCCESSFUL_RETURN \n
			RET_INIT_FAILED \n
			RET_INIT_FAILED_CHOLESKY \n
			RET_INIT_FAILED_TQ \n
			RET_INIT_FAILED_HOTSTART \n
			RET_INIT_FAILED_INFEASIBILITY \n
			RET_INIT_FAILED_UNBOUNDEDNESS \n
			RET_MAX_NWSR_REACHED \n
			RET_INVALID_ARGUMENTS */
returnValue QProblem_initW(	QProblem* _THIS,
							real_t* const _H, 						/**< Hessian matrix. \n
														    			 If Hessian matrix is trivial, a NULL pointer can be passed. */
							const real_t* const _g, 				/**< Gradient vector. */
							real_t* const _A,  						/**< Constraint matrix. */
							const real_t* const _lb,				/**< Lower bound vector (on variables). \n
																		 If no lower bounds exist, a NULL pointer can be passed. */
							const real_t* const _ub,				/**< Upper bound vector (on variables). \n
																		 If no upper bounds exist, a NULL pointer can be passed. */
							const real_t* const _lbA,				/**< Lower constraints' bound vector. \n
																		 If no lower constraints' bounds exist, a NULL pointer can be passed. */
							const real_t* const _ubA,				/**< Upper constraints' bound vector. \n
																	 If no lower constraints' bounds exist, a NULL pointer can be passed. */
							int* nWSR,								/**< Input:  Maximum number of working set recalculations when using initial homotopy.
																	 *	 Output: Number of performed working set recalculations. */
							real_t* const cputime,					/**< Input:  Maximum CPU time allowed for QP initialisation. \n
																		 Output: CPU time spent for QP initialisation. */
							const real_t* const xOpt,				/**< Optimal primal solution vector. \n
																		 (If a null pointer is passed, the old primal solution is kept!) */
							const real_t* const yOpt,				/**< Optimal dual solution vector. \n
																		 (If a null pointer is passed, the old dual solution is kept!) */
							Bounds* const guessedBounds,			/**< Optimal working set of bounds for solution (xOpt,yOpt). */
							Constraints* const guessedConstraints,	/**< Optimal working set of constraints for solution (xOpt,yOpt). */
							const real_t* const _R					/**< Pre-computed (upper triangular) Cholesky factor of Hessian matrix.
																	 	 The Cholesky factor must be stored in a real_t array of size nV*nV
																		 in row-major format. Note: Only used if xOpt/yOpt and gB are NULL! \n
																		 (If a null pointer is passed, Cholesky decomposition is computed internally!) */
							);

/** Initialises a QP problem with given QP data to be ream from files and tries to solve it
 *	using at most nWSR iterations. Depending on the parameter constellation it: \n
 *	1. 0,    0,    0    : starts with xOpt = 0, yOpt = 0 and gB/gC empty (or all implicit equality bounds), \n
 *	2. xOpt, 0,    0    : starts with xOpt, yOpt = 0 and obtain gB/gC by "clipping", \n
 *	3. 0,    yOpt, 0    : starts with xOpt = 0, yOpt and obtain gB/gC from yOpt != 0, \n
 *	4. 0,    0,    gB/gC: starts with xOpt = 0, yOpt = 0 and gB/gC, \n
 *	5. xOpt, yOpt, 0    : starts with xOpt, yOpt and obtain gB/gC from yOpt != 0, \n
 *	6. xOpt, 0,    gB/gC: starts with xOpt, yOpt = 0 and gB/gC, \n
 *	7. xOpt, yOpt, gB/gC: starts with xOpt, yOpt and gB/gC (assume them to be consistent!)
 *
 *  Note: This function internally calls solveInitialQP for initialisation!
 *
 *	\return SUCCESSFUL_RETURN \n
			RET_INIT_FAILED \n
			RET_INIT_FAILED_CHOLESKY \n
			RET_INIT_FAILED_TQ \n
			RET_INIT_FAILED_HOTSTART \n
			RET_INIT_FAILED_INFEASIBILITY \n
			RET_INIT_FAILED_UNBOUNDEDNESS \n
			RET_MAX_NWSR_REACHED \n
			RET_UNABLE_TO_READ_FILE */
returnValue QProblem_initFW(	QProblem* _THIS,
								const char* const H_file,				/**< Name of file where Hessian matrix is stored. \n
														     				 If Hessian matrix is trivial, a NULL pointer can be passed. */
								const char* const g_file,				/**< Name of file where gradient vector is stored. */
								const char* const A_file,				/**< Name of file where constraint matrix is stored. */
								const char* const lb_file,				/**< Name of file where lower bound vector. \n
																			 If no lower bounds exist, a NULL pointer can be passed. */
								const char* const ub_file,				/**< Name of file where upper bound vector. \n
																			 If no upper bounds exist, a NULL pointer can be passed. */
								const char* const lbA_file,				/**< Name of file where lower constraints' bound vector. \n
																			 If no lower constraints' bounds exist, a NULL pointer can be passed. */
								const char* const ubA_file,				/**< Name of file where upper constraints' bound vector. \n
																			 If no upper constraints' bounds exist, a NULL pointer can be passed. */
								int* nWSR,								/**< Input:  Maximum number of working set recalculations when using initial homotopy.
																			 Output: Number of performed working set recalculations. */
								real_t* const cputime,					/**< Input:  Maximum CPU time allowed for QP initialisation. \n
																			 Output: CPU time spent for QP initialisation. */
								const real_t* const xOpt,				/**< Optimal primal solution vector. \n
																			 (If a null pointer is passed, the old primal solution is kept!) */
								const real_t* const yOpt,				/**< Optimal dual solution vector. \n
																			 (If a null pointer is passed, the old dual solution is kept!) */
								Bounds* const guessedBounds,			/**< Optimal working set of bounds for solution (xOpt,yOpt). */
								Constraints* const guessedConstraints,	/**< Optimal working set of constraints for solution (xOpt,yOpt). */
								const char* const R_file				/**< Pre-computed (upper triangular) Cholesky factor of Hessian matrix.
																		 	 The Cholesky factor must be stored in a real_t array of size nV*nV
																			 in row-major format. Note: Only used if xOpt/yOpt and gB are NULL! \n
																			 (If a null pointer is passed, Cholesky decomposition is computed internally!) */
								);

/** Solves an initialised QP sequence using the online active set strategy.
 *	QP solution is started from previous solution.
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
returnValue QProblem_hotstart(	QProblem* _THIS,
								const real_t* const g_new,		/**< Gradient of neighbouring QP to be solved. */
								const real_t* const lb_new,		/**< Lower bounds of neighbouring QP to be solved. \n
											 			 			 If no lower bounds exist, a NULL pointer can be passed. */
								const real_t* const ub_new,		/**< Upper bounds of neighbouring QP to be solved. \n
											 			 			 If no upper bounds exist, a NULL pointer can be passed. */
								const real_t* const lbA_new,	/**< Lower constraints' bounds of neighbouring QP to be solved. \n
											 			 			 If no lower constraints' bounds exist, a NULL pointer can be passed. */
								const real_t* const ubA_new,	/**< Upper constraints' bounds of neighbouring QP to be solved. \n
											 			 			 If no upper constraints' bounds exist, a NULL pointer can be passed. */
								int* nWSR,						/**< Input:  Maximum number of working set recalculations; \n
													 				 Output: Number of performed working set recalculations. */
								real_t* const cputime 			/**< Input:  Maximum CPU time allowed for QP solution. \n
														 			 Output: CPU time spent for QP solution (or to perform nWSR iterations). */
								);

/** Solves an initialised QP sequence using the online active set strategy,
 *	where QP data is read from files. QP solution is started from previous solution.
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
returnValue QProblem_hotstartF(	QProblem* _THIS,
								const char* const g_file, 	/**< Name of file where gradient, of neighbouring QP to be solved, is stored. */
								const char* const lb_file, 	/**< Name of file where lower bounds, of neighbouring QP to be solved, is stored. \n
											 					 If no lower bounds exist, a NULL pointer can be passed. */
								const char* const ub_file, 	/**< Name of file where upper bounds, of neighbouring QP to be solved, is stored. \n
											 					 If no upper bounds exist, a NULL pointer can be passed. */
								const char* const lbA_file, /**< Name of file where lower constraints' bounds, of neighbouring QP to be solved, is stored. \n
											 					 If no lower constraints' bounds exist, a NULL pointer can be passed. */
								const char* const ubA_file, /**< Name of file where upper constraints' bounds, of neighbouring QP to be solved, is stored. \n
											 					 If no upper constraints' bounds exist, a NULL pointer can be passed. */
								int* nWSR, 					/**< Input:  Maximum number of working set recalculations; \n
																 Output: Number of performed working set recalculations. */
								real_t* const cputime 		/**< Input:  Maximum CPU time allowed for QP solution. \n
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
returnValue QProblem_hotstartW(	QProblem* _THIS,
								const real_t* const g_new,				/**< Gradient of neighbouring QP to be solved. */
								const real_t* const lb_new,				/**< Lower bounds of neighbouring QP to be solved. \n
											 						 		 If no lower bounds exist, a NULL pointer can be passed. */
								const real_t* const ub_new,				/**< Upper bounds of neighbouring QP to be solved. \n
											 						 		 If no upper bounds exist, a NULL pointer can be passed. */
								const real_t* const lbA_new,			/**< Lower constraints' bounds of neighbouring QP to be solved. \n
											 						 		 If no lower constraints' bounds exist, a NULL pointer can be passed. */
								const real_t* const ubA_new,			/**< Upper constraints' bounds of neighbouring QP to be solved. \n
											 								 If no upper constraints' bounds exist, a NULL pointer can be passed. */
								int* nWSR,								/**< Input: Maximum number of working set recalculations; \n
																			 Output: Number of performed working set recalculations. */
								real_t* const cputime,					/**< Input: Maximum CPU time allowed for QP solution. \n
																	 		 Output: CPU time spent for QP solution (or to perform nWSR iterations). */
								Bounds* const guessedBounds,			/**< Optimal working set of bounds for solution (xOpt,yOpt). \n
																			 (If a null pointer is passed, the previous working set of bounds is kept!) */
								Constraints* const guessedConstraints	/**< Optimal working set of constraints for solution (xOpt,yOpt). \n
																			 (If a null pointer is passed, the previous working set of constraints is kept!) */
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
			RET_SETUP_AUXILIARYQP_FAILED \n
			RET_UNABLE_TO_READ_FILE \n
			RET_INVALID_ARGUMENTS */
returnValue QProblem_hotstartFW(	QProblem* _THIS,
									const char* const g_file, 				/**< Name of file where gradient, of neighbouring QP to be solved, is stored. */
									const char* const lb_file, 				/**< Name of file where lower bounds, of neighbouring QP to be solved, is stored. \n
											 									 If no lower bounds exist, a NULL pointer can be passed. */
									const char* const ub_file, 				/**< Name of file where upper bounds, of neighbouring QP to be solved, is stored. \n
											 									 If no upper bounds exist, a NULL pointer can be passed. */
									const char* const lbA_file, 			/**< Name of file where lower constraints' bounds, of neighbouring QP to be solved, is stored. \n
											 									 If no lower constraints' bounds exist, a NULL pointer can be passed. */
									const char* const ubA_file, 			/**< Name of file where upper constraints' bounds, of neighbouring QP to be solved, is stored. \n
											 									 If no upper constraints' bounds exist, a NULL pointer can be passed. */
									int* nWSR,								/**< Input: Maximum number of working set recalculations; \n
																 				 Output: Number of performed working set recalculations. */
									real_t* const cputime,					/**< Input: Maximum CPU time allowed for QP solution. \n
																	 			 Output: CPU time spent for QP solution (or to perform nWSR iterations). */
									Bounds* const guessedBounds,			/**< Optimal working set of bounds for solution (xOpt,yOpt). \n
																				 (If a null pointer is passed, the previous working set of bounds is kept!) */
									Constraints* const guessedConstraints	/**< Optimal working set of constraints for solution (xOpt,yOpt). \n
																				 (If a null pointer is passed, the previous working set of constraints is kept!) */
									);


/** Solves using the current working set
 *	\return SUCCESSFUL_RETURN \n
 *			RET_STEPDIRECTION_FAILED_TQ \n
 *			RET_STEPDIRECTION_FAILED_CHOLESKY \n
 *			RET_INVALID_ARGUMENTS */
returnValue QProblem_solveCurrentEQP (	QProblem* _THIS,
										const int n_rhs,		/**< Number of consecutive right hand sides */
										const real_t* g_in,		/**< Gradient of neighbouring QP to be solved. */
										const real_t* lb_in,	/**< Lower bounds of neighbouring QP to be solved. \n
																	 If no lower bounds exist, a NULL pointer can be passed. */
										const real_t* ub_in,	/**< Upper bounds of neighbouring QP to be solved. \n
																	 If no upper bounds exist, a NULL pointer can be passed. */
										const real_t* lbA_in,	/**< Lower constraints' bounds of neighbouring QP to be solved. \n
																	 If no lower constraints' bounds exist, a NULL pointer can be passed. */
										const real_t* ubA_in,	/**< Upper constraints' bounds of neighbouring QP to be solved. \n */
										real_t* x_out,			/**< Output: Primal solution */
										real_t* y_out			/**< Output: Dual solution */
										);



/** Returns current constraints object of the QP (deep copy).
  *	\return SUCCESSFUL_RETURN \n
  			RET_QPOBJECT_NOT_SETUP */
static inline returnValue QProblem_getConstraints(	QProblem* _THIS,
													Constraints* _constraints	/** Output: Constraints object. */
													);


/** Returns the number of constraints.
 *	\return Number of constraints. */
static inline int QProblem_getNC( QProblem* _THIS );

/** Returns the number of (implicitly defined) equality constraints.
 *	\return Number of (implicitly defined) equality constraints. */
static inline int QProblem_getNEC( QProblem* _THIS );

/** Returns the number of active constraints.
 *	\return Number of active constraints. */
static inline int QProblem_getNAC( QProblem* _THIS );

/** Returns the number of inactive constraints.
 *	\return Number of inactive constraints. */
static inline int QProblem_getNIAC( QProblem* _THIS );

/** Returns the dimension of null space.
 *	\return Dimension of null space. */
int QProblem_getNZ( QProblem* _THIS );


/** Returns the dual solution vector (deep copy).
 *	\return SUCCESSFUL_RETURN \n
			RET_QP_NOT_SOLVED */
returnValue QProblem_getDualSolution(	QProblem* _THIS,
										real_t* const yOpt	/**< Output: Dual solution vector (if QP has been solved). */
										);


/** Defines user-defined routine for calculating the constraint product A*x
 *	\return  SUCCESSFUL_RETURN \n */
returnValue QProblem_setConstraintProduct(	QProblem* _THIS,
											ConstraintProduct _constraintProduct
											);


/** Prints concise list of properties of the current QP.
 *	\return  SUCCESSFUL_RETURN \n */
returnValue QProblem_printProperties( QProblem* _THIS );



/** Writes a vector with the state of the working set
*	\return SUCCESSFUL_RETURN */
returnValue QProblem_getWorkingSet(	QProblem* _THIS,
									real_t* workingSet	/** Output: array containing state of the working set. */
									);

/** Writes a vector with the state of the working set of bounds
 *	\return SUCCESSFUL_RETURN \n
 *	        RET_INVALID_ARGUMENTS */
returnValue QProblem_getWorkingSetBounds(	QProblem* _THIS,
											real_t* workingSetB	/** Output: array containing state of the working set of bounds. */
											);

/** Writes a vector with the state of the working set of constraints
 *	\return SUCCESSFUL_RETURN \n
 *	        RET_INVALID_ARGUMENTS */
returnValue QProblem_getWorkingSetConstraints(	QProblem* _THIS,
												real_t* workingSetC	/** Output: array containing state of the working set of constraints. */
												);


/** Returns current bounds object of the QP (deep copy).
  *	\return SUCCESSFUL_RETURN \n
  			RET_QPOBJECT_NOT_SETUP */
static inline returnValue QProblem_getBounds(	QProblem* _THIS,
												Bounds* _bounds		/** Output: Bounds object. */
												);


/** Returns the number of variables.
 *	\return Number of variables. */
static inline int QProblem_getNV( QProblem* _THIS );

/** Returns the number of free variables.
 *	\return Number of free variables. */
static inline int QProblem_getNFR( QProblem* _THIS );

/** Returns the number of fixed variables.
 *	\return Number of fixed variables. */
static inline int QProblem_getNFX( QProblem* _THIS );

/** Returns the number of implicitly fixed variables.
 *	\return Number of implicitly fixed variables. */
static inline int QProblem_getNFV( QProblem* _THIS );


/** Returns the optimal objective function value.
 *	\return finite value: Optimal objective function value (QP was solved) \n
 			+infinity:	  QP was not yet solved */
real_t QProblem_getObjVal( QProblem* _THIS );

/** Returns the objective function value at an arbitrary point x.
 *	\return Objective function value at point x */
real_t QProblem_getObjValX(	QProblem* _THIS,
							const real_t* const _x	/**< Point at which the objective function shall be evaluated. */
							);

/** Returns the primal solution vector.
 *	\return SUCCESSFUL_RETURN \n
			RET_QP_NOT_SOLVED */
returnValue QProblem_getPrimalSolution(	QProblem* _THIS,
										real_t* const xOpt	/**< Output: Primal solution vector (if QP has been solved). */
										);


/** Returns status of the solution process.
 *	\return Status of solution process. */
static inline QProblemStatus QProblem_getStatus( QProblem* _THIS );


/** Returns if the QProblem object is initialised.
 *	\return BT_TRUE:  QProblem initialised \n
 			BT_FALSE: QProblem not initialised */
static inline BooleanType QProblem_isInitialised( QProblem* _THIS );

/** Returns if the QP has been solved.
 *	\return BT_TRUE:  QProblem solved \n
 			BT_FALSE: QProblem not solved */
static inline BooleanType QProblem_isSolved( QProblem* _THIS );

/** Returns if the QP is infeasible.
 *	\return BT_TRUE:  QP infeasible \n
 			BT_FALSE: QP feasible (or not known to be infeasible!) */
static inline BooleanType QProblem_isInfeasible( QProblem* _THIS );

/** Returns if the QP is unbounded.
 *	\return BT_TRUE:  QP unbounded \n
 			BT_FALSE: QP unbounded (or not known to be unbounded!) */
static inline BooleanType QProblem_isUnbounded( QProblem* _THIS );


/** Returns Hessian type flag (type is not determined due to _THIS call!).
 *	\return Hessian type. */
static inline HessianType QProblem_getHessianType( QProblem* _THIS );

/** Changes the print level.
 *	\return SUCCESSFUL_RETURN */
static inline returnValue QProblem_setHessianType(	QProblem* _THIS,
													HessianType _hessianType /**< New Hessian type. */
													);

/** Returns if the QP has been internally regularised.
 *	\return BT_TRUE:  Hessian is internally regularised for QP solution \n
 			BT_FALSE: No internal Hessian regularisation is used for QP solution */
static inline BooleanType QProblem_usingRegularisation( QProblem* _THIS );

/** Returns current options struct.
 *	\return Current options struct. */
static inline Options QProblem_getOptions( QProblem* _THIS );

/** Overrides current options with given ones.
 *	\return SUCCESSFUL_RETURN */
static inline returnValue QProblem_setOptions(	QProblem* _THIS,
												Options _options	/**< New options. */
												);

/** Returns the print level.
 *	\return Print level. */
static inline PrintLevel QProblem_getPrintLevel( QProblem* _THIS );

/** Changes the print level.
 *	\return SUCCESSFUL_RETURN */
returnValue QProblem_setPrintLevel(	QProblem* _THIS,
									PrintLevel _printlevel	/**< New print level. */
									);


/** Returns the current number of QP problems solved.
 *	\return Number of QP problems solved. */
static inline unsigned int QProblem_getCount( QProblem* _THIS );

/** Resets QP problem counter (to zero).
 *	\return SUCCESSFUL_RETURN. */
static inline returnValue QProblem_resetCounter( QProblem* _THIS );


/** Prints a list of all options and their current values.
 *	\return  SUCCESSFUL_RETURN \n */
returnValue QProblem_printOptions( QProblem* _THIS );


/** Solves a QProblem whose QP data is assumed to be stored in the member variables.
 *  A guess for its primal/dual optimal solution vectors and the corresponding
 *  working sets of bounds and constraints can be provided.
 *  Note: This function is internally called by all init functions!
 *	\return SUCCESSFUL_RETURN \n
			RET_INIT_FAILED \n
			RET_INIT_FAILED_CHOLESKY \n
			RET_INIT_FAILED_TQ \n
			RET_INIT_FAILED_HOTSTART \n
			RET_INIT_FAILED_INFEASIBILITY \n
			RET_INIT_FAILED_UNBOUNDEDNESS \n
			RET_MAX_NWSR_REACHED */
returnValue QProblem_solveInitialQP(	QProblem* _THIS,
										const real_t* const xOpt,				/**< Optimal primal solution vector.*/
										const real_t* const yOpt,				/**< Optimal dual solution vector. */
										Bounds* const guessedBounds,			/**< Optimal working set of bounds for solution (xOpt,yOpt). */
										Constraints* const guessedConstraints,	/**< Optimal working set of constraints for solution (xOpt,yOpt). */
										const real_t* const _R,					/**< Pre-computed (upper triangular) Cholesky factor of Hessian matrix. */
										int* nWSR, 								/**< Input:  Maximum number of working set recalculations; \n
														 						 *	 Output: Number of performed working set recalculations. */
										real_t* const cputime					/**< Input:  Maximum CPU time allowed for QP solution. \n
																	 			 *	 Output: CPU time spent for QP solution (or to perform nWSR iterations). */
										);

/** Solves QProblem using online active set strategy.
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
returnValue QProblem_solveQP(	QProblem* _THIS,
								const real_t* const g_new,		/**< Gradient of neighbouring QP to be solved. */
								const real_t* const lb_new,		/**< Lower bounds of neighbouring QP to be solved. \n
											 			 			 If no lower bounds exist, a NULL pointer can be passed. */
								const real_t* const ub_new,		/**< Upper bounds of neighbouring QP to be solved. \n
											 			 			 If no upper bounds exist, a NULL pointer can be passed. */
								const real_t* const lbA_new,	/**< Lower constraints' bounds of neighbouring QP to be solved. \n
											 			 			 If no lower constraints' bounds exist, a NULL pointer can be passed. */
								const real_t* const ubA_new,	/**< Upper constraints' bounds of neighbouring QP to be solved. \n
											 			 			 If no upper constraints' bounds exist, a NULL pointer can be passed. */
								int* nWSR,						/**< Input: Maximum number of working set recalculations; \n
													 				 Output: Number of performed working set recalculations. */
								real_t* const cputime,			/**< Input: Maximum CPU time allowed for QP solution. \n
														 			 Output: CPU time spent for QP solution (or to perform nWSR iterations). */
								int  nWSRperformed,				/**< Number of working set recalculations already performed to solve
																	 this QP within previous solveQP() calls. This number is
																	 always zero, except for successive calls from solveRegularisedQP()
																	 or when using the far bound strategy. */
								BooleanType isFirstCall			/**< Indicating whether this is the first call for current QP. */
								);


/** Solves QProblem using online active set strategy.
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
returnValue QProblem_solveRegularisedQP(	QProblem* _THIS,
											const real_t* const g_new,		/**< Gradient of neighbouring QP to be solved. */
											const real_t* const lb_new,		/**< Lower bounds of neighbouring QP to be solved. \n
													 			 				 If no lower bounds exist, a NULL pointer can be passed. */
											const real_t* const ub_new,		/**< Upper bounds of neighbouring QP to be solved. \n
													 			 				 If no upper bounds exist, a NULL pointer can be passed. */
											const real_t* const lbA_new,	/**< Lower constraints' bounds of neighbouring QP to be solved. \n
													 			 				 If no lower constraints' bounds exist, a NULL pointer can be passed. */
											const real_t* const ubA_new,	/**< Upper constraints' bounds of neighbouring QP to be solved. \n
													 			 				 If no upper constraints' bounds exist, a NULL pointer can be passed. */
											int* nWSR,						/**< Input: Maximum number of working set recalculations; \n
															 					 Output: Number of performed working set recalculations. */
											real_t* const cputime,			/**< Input: Maximum CPU time allowed for QP solution. \n
																 				 Output: CPU time spent for QP solution (or to perform nWSR iterations). */
											int  nWSRperformed,				/**< Number of working set recalculations already performed to solve
																				 this QP within previous solveRegularisedQP() calls. This number is
																				 always zero, except for successive calls when using the far bound strategy. */
											BooleanType isFirstCall			/**< Indicating whether this is the first call for current QP. */
											);


/** Determines type of existing constraints and bounds (i.e. implicitly fixed, unbounded etc.).
 *	\return SUCCESSFUL_RETURN \n
			RET_SETUPSUBJECTTOTYPE_FAILED */
returnValue QProblem_setupSubjectToType( QProblem* _THIS );

/** Determines type of new constraints and bounds (i.e. implicitly fixed, unbounded etc.).
 *	\return SUCCESSFUL_RETURN \n
			RET_SETUPSUBJECTTOTYPE_FAILED */
returnValue QProblem_setupSubjectToTypeNew(	QProblem* _THIS,
											const real_t* const lb_new,		/**< New lower bounds. */
											const real_t* const ub_new,		/**< New upper bounds. */
											const real_t* const lbA_new,	/**< New lower constraints' bounds. */
											const real_t* const ubA_new		/**< New upper constraints' bounds. */
											);

/** Computes the Cholesky decomposition of the projected Hessian (i.e. R^T*R = Z^T*H*Z).
 *  Note: If Hessian turns out not to be positive definite, the Hessian type
 *		  is set to HST_SEMIDEF accordingly.
 *	\return SUCCESSFUL_RETURN \n
 *			RET_HESSIAN_NOT_SPD \n
 *			RET_INDEXLIST_CORRUPTED */
returnValue QProblem_computeProjectedCholesky( QProblem* _THIS );

/** Computes initial Cholesky decomposition of the projected Hessian making
 *  use of the function setupCholeskyDecomposition() or setupCholeskyDecompositionProjected().
 *	\return SUCCESSFUL_RETURN \n
 *			RET_HESSIAN_NOT_SPD \n
 *			RET_INDEXLIST_CORRUPTED */
returnValue QProblem_setupInitialCholesky( QProblem* _THIS );

/** Initialises TQ factorisation of A (i.e. A*Q = [0 T]) if NO constraint is active.
 *	\return SUCCESSFUL_RETURN \n
 			RET_INDEXLIST_CORRUPTED */
returnValue QProblem_setupTQfactorisation( QProblem* _THIS );


/** Obtains the desired working set for the auxiliary initial QP in
 *  accordance with the user specifications
 *  (assumes that member AX has already been initialised!)
 *	\return SUCCESSFUL_RETURN \n
			RET_OBTAINING_WORKINGSET_FAILED \n
			RET_INVALID_ARGUMENTS */
returnValue QProblem_obtainAuxiliaryWorkingSet(	QProblem* _THIS,
												const real_t* const xOpt,					/**< Optimal primal solution vector.
																							 *	 If a NULL pointer is passed, all entries are assumed to be zero. */
												const real_t* const yOpt,					/**< Optimal dual solution vector.
																							 *	 If a NULL pointer is passed, all entries are assumed to be zero. */
												Bounds* const guessedBounds,				/**< Guessed working set of bounds for solution (xOpt,yOpt). */
												Constraints* const guessedConstraints,		/**< Guessed working set for solution (xOpt,yOpt). */
												Bounds* auxiliaryBounds,					/**< Input: Allocated bound object. \n
																							 *	 Ouput: Working set of constraints for auxiliary QP. */
												Constraints* auxiliaryConstraints			/**< Input: Allocated bound object. \n
																							 *	 Ouput: Working set for auxiliary QP. */
												);

/** Sets up bound and constraints data structures according to auxiliaryBounds/Constraints.
 *  (If the working set shall be setup afresh, make sure that
 *  bounds and constraints data structure have been resetted
 *  and the TQ factorisation has been initialised!)
 *	\return SUCCESSFUL_RETURN \n
			RET_SETUP_WORKINGSET_FAILED \n
			RET_INVALID_ARGUMENTS \n
			RET_UNKNOWN_BUG */
returnValue QProblem_setupAuxiliaryWorkingSet(	QProblem* _THIS,
												Bounds* const auxiliaryBounds,				/**< Working set of bounds for auxiliary QP. */
												Constraints* const auxiliaryConstraints,	/**< Working set of constraints for auxiliary QP. */
												BooleanType setupAfresh						/**< Flag indicating if given working set shall be
																							 *    setup afresh or by updating the current one. */
												);

/** Sets up the optimal primal/dual solution of the auxiliary initial QP.
 *	\return SUCCESSFUL_RETURN */
returnValue QProblem_setupAuxiliaryQPsolution(	QProblem* _THIS,
												const real_t* const xOpt,	/**< Optimal primal solution vector.
																		 	 *	 If a NULL pointer is passed, all entries are set to zero. */
												const real_t* const yOpt	/**< Optimal dual solution vector.
																			 *	 If a NULL pointer is passed, all entries are set to zero. */
												);

/** Sets up gradient of the auxiliary initial QP for given
 *  optimal primal/dual solution and given initial working set
 *  (assumes that members X, Y and BOUNDS, CONSTRAINTS have already been initialised!).
 *	\return SUCCESSFUL_RETURN */
returnValue QProblem_setupAuxiliaryQPgradient( QProblem* _THIS );

/** Sets up (constraints') bounds of the auxiliary initial QP for given
 *  optimal primal/dual solution and given initial working set
 *  (assumes that members X, Y and BOUNDS, CONSTRAINTS have already been initialised!).
 *	\return SUCCESSFUL_RETURN \n
			RET_UNKNOWN_BUG */
returnValue QProblem_setupAuxiliaryQPbounds(	QProblem* _THIS,
												Bounds* const auxiliaryBounds,				/**< Working set of bounds for auxiliary QP. */
												Constraints* const auxiliaryConstraints,	/**< Working set of constraints for auxiliary QP. */
												BooleanType useRelaxation					/**< Flag indicating if inactive (constraints') bounds shall be relaxed. */
												);


/** Adds a constraint to active set.
 *	\return SUCCESSFUL_RETURN \n
 			RET_ADDCONSTRAINT_FAILED \n
			RET_ADDCONSTRAINT_FAILED_INFEASIBILITY \n
			RET_ENSURELI_FAILED */
returnValue QProblem_addConstraint(	QProblem* _THIS,
									int number,					/**< Number of constraint to be added to active set. */
									SubjectToStatus C_status,	/**< Status of new active constraint. */
									BooleanType updateCholesky,	/**< Flag indicating if Cholesky decomposition shall be updated. */
									BooleanType ensureLI		/**< Ensure linear independence by exchange rules by default. */
									);

/** Checks if new active constraint to be added is linearly dependent from
 *	from row of the active constraints matrix.
 *	\return	 RET_LINEARLY_DEPENDENT \n
 			 RET_LINEARLY_INDEPENDENT \n
			 RET_INDEXLIST_CORRUPTED */
returnValue QProblem_addConstraint_checkLI(	QProblem* _THIS,
											int number			/**< Number of constraint to be added to active set. */
											);

/** Ensures linear independence of constraint matrix when a new constraint is added.
 * 	To _THIS end a bound or constraint is removed simultaneously if necessary.
 *	\return	 SUCCESSFUL_RETURN \n
 			 RET_LI_RESOLVED \n
			 RET_ENSURELI_FAILED \n
			 RET_ENSURELI_FAILED_TQ \n
			 RET_ENSURELI_FAILED_NOINDEX \n
			 RET_REMOVE_FROM_ACTIVESET */
returnValue QProblem_addConstraint_ensureLI(	QProblem* _THIS,
												int number,					/**< Number of constraint to be added to active set. */
												SubjectToStatus C_status	/**< Status of new active bound. */
												);

/** Adds a bound to active set.
 *	\return SUCCESSFUL_RETURN \n
 			RET_ADDBOUND_FAILED \n
			RET_ADDBOUND_FAILED_INFEASIBILITY \n
			RET_ENSURELI_FAILED */
returnValue QProblem_addBound(	QProblem* _THIS,
								int number,					/**< Number of bound to be added to active set. */
								SubjectToStatus B_status,	/**< Status of new active bound. */
								BooleanType updateCholesky,	/**< Flag indicating if Cholesky decomposition shall be updated. */
								BooleanType ensureLI 		/**< Ensure linear independence by exchange rules by default. */
								);

/** Checks if new active bound to be added is linearly dependent from
 *	from row of the active constraints matrix.
 *	\return	 RET_LINEARLY_DEPENDENT \n
 			 RET_LINEARLY_INDEPENDENT */
returnValue QProblem_addBound_checkLI(	QProblem* _THIS,
										int number			/**< Number of bound to be added to active set. */
										);

/** Ensures linear independence of constraint matrix when a new bound is added.
 *	To _THIS end a bound or constraint is removed simultaneously if necessary.
 *	\return	 SUCCESSFUL_RETURN \n
 			 RET_LI_RESOLVED \n
			 RET_ENSURELI_FAILED \n
			 RET_ENSURELI_FAILED_TQ \n
			 RET_ENSURELI_FAILED_NOINDEX \n
			 RET_REMOVE_FROM_ACTIVESET */
returnValue QProblem_addBound_ensureLI(	QProblem* _THIS,
										int number,					/**< Number of bound to be added to active set. */
										SubjectToStatus B_status	/**< Status of new active bound. */
										);

/** Removes a constraint from active set.
 *	\return SUCCESSFUL_RETURN \n
 			RET_CONSTRAINT_NOT_ACTIVE \n
			RET_REMOVECONSTRAINT_FAILED \n
			RET_HESSIAN_NOT_SPD */
returnValue QProblem_removeConstraint(	QProblem* _THIS,
										int number,						/**< Number of constraint to be removed from active set. */
										BooleanType updateCholesky,		/**< Flag indicating if Cholesky decomposition shall be updated. */
										BooleanType allowFlipping,		/**< Flag indicating if flipping bounds are allowed. */
										BooleanType ensureNZC			/**< Flag indicating if non-zero curvature is ensured by exchange rules. */
										);

/** Removes a bounds from active set.
 *	\return SUCCESSFUL_RETURN \n
 			RET_BOUND_NOT_ACTIVE \n
			RET_HESSIAN_NOT_SPD \n
			RET_REMOVEBOUND_FAILED */
returnValue QProblem_removeBound(	QProblem* _THIS,
									int number,						/**< Number of bound to be removed from active set. */
									BooleanType updateCholesky,		/**< Flag indicating if Cholesky decomposition shall be updated. */
									BooleanType allowFlipping,		/**< Flag indicating if flipping bounds are allowed. */
									BooleanType ensureNZC			/**< Flag indicating if non-zero curvature is ensured by exchange rules. */
									);


/** Performs robustified ratio test yield the maximum possible step length
 *  along the homotopy path.
 *	\return  SUCCESSFUL_RETURN */
returnValue QProblem_performPlainRatioTest(	QProblem* _THIS,
											int nIdx, 					/**< Number of ratios to be checked. */
											const int* const idxList, 	/**< Array containing the indices of all ratios to be checked. */
											const real_t* const num,	/**< Array containing all numerators for performing the ratio test. */
											const real_t* const den,	/**< Array containing all denominators for performing the ratio test. */
											real_t epsNum,				/**< Numerator tolerance. */
											real_t epsDen,				/**< Denominator tolerance. */
											real_t* t,					/**< Output: Maximum possible step length along the homotopy path. */
											int* BC_idx 				/**< Output: Index of blocking constraint. */
											);


/** Ensure non-zero curvature by primal jump.
 *  \return SUCCESSFUL_RETURN \n
 *          RET_HOTSTART_STOPPED_UNBOUNDEDNESS */
returnValue QProblem_ensureNonzeroCurvature(	QProblem* _THIS,
												BooleanType removeBoundNotConstraint,	/**< SubjectTo to be removed is a bound. */
												int remIdx,								/**< Index of bound/constraint to be removed. */
												BooleanType* exchangeHappened,			/**< Output: Exchange was necessary to ensure. */
												BooleanType* addBoundNotConstraint,		/**< SubjectTo to be added is a bound. */
												int* addIdx,							/**< Index of bound/constraint to be added. */
												SubjectToStatus* addStatus				/**< Status of bound/constraint to be added. */
												);


/** Solves the system Ta = b or T^Ta = b where T is a reverse upper triangular matrix.
 *	\return SUCCESSFUL_RETURN \n
 			RET_DIV_BY_ZERO */
returnValue QProblem_backsolveT(	QProblem* _THIS,
									const real_t* const b,	/**< Right hand side vector. */
									BooleanType transposed,	/**< Indicates if the transposed system shall be solved. */
									real_t* const a 		/**< Output: Solution vector */
									);


/** Determines step direction of the shift of the QP data.
 *	\return SUCCESSFUL_RETURN */
returnValue QProblem_determineDataShift(	QProblem* _THIS,
											const real_t* const g_new,		/**< New gradient vector. */
											const real_t* const lbA_new,	/**< New lower constraints' bounds. */
											const real_t* const ubA_new,	/**< New upper constraints' bounds. */
											const real_t* const lb_new,		/**< New lower bounds. */
											const real_t* const ub_new,		/**< New upper bounds. */
											real_t* const delta_g,	 		/**< Output: Step direction of gradient vector. */
											real_t* const delta_lbA,		/**< Output: Step direction of lower constraints' bounds. */
											real_t* const delta_ubA,		/**< Output: Step direction of upper constraints' bounds. */
											real_t* const delta_lb,	 		/**< Output: Step direction of lower bounds. */
											real_t* const delta_ub,	 		/**< Output: Step direction of upper bounds. */
											BooleanType* Delta_bC_isZero,	/**< Output: Indicates if active constraints' bounds are to be shifted. */
											BooleanType* Delta_bB_isZero	/**< Output: Indicates if active bounds are to be shifted. */
											);

/** Determines step direction of the homotopy path.
 *	\return SUCCESSFUL_RETURN \n
 			RET_STEPDIRECTION_FAILED_TQ \n
			RET_STEPDIRECTION_FAILED_CHOLESKY */
returnValue QProblem_determineStepDirection(	QProblem* _THIS,
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

/** Determines the maximum possible step length along the homotopy path
 *  and performs _THIS step (without changing working set).
 *	\return SUCCESSFUL_RETURN \n
 * 			RET_ERROR_IN_CONSTRAINTPRODUCT \n
 * 			RET_QP_INFEASIBLE */
returnValue QProblem_performStep(	QProblem* _THIS,
									const real_t* const delta_g,		/**< Step direction of gradient. */
									const real_t* const delta_lbA,		/**< Step direction of lower constraints' bounds. */
									const real_t* const delta_ubA,		/**< Step direction of upper constraints' bounds. */
									const real_t* const delta_lb,	 	/**< Step direction of lower bounds. */
									const real_t* const delta_ub,	 	/**< Step direction of upper bounds. */
									const real_t* const delta_xFX, 		/**< Primal homotopy step direction of fixed variables. */
									const real_t* const delta_xFR,		/**< Primal homotopy step direction of free variables. */
									const real_t* const delta_yAC,		/**< Dual homotopy step direction of active constraints' multiplier. */
									const real_t* const delta_yFX,		/**< Dual homotopy step direction of fixed variables' multiplier. */
									int* BC_idx, 						/**< Output: Index of blocking constraint. */
									SubjectToStatus* BC_status,			/**< Output: Status of blocking constraint. */
									BooleanType* BC_isBound 			/**< Output: Indicates if blocking constraint is a bound. */
									);

/** Updates the active set.
 *	\return  SUCCESSFUL_RETURN \n
 			 RET_REMOVE_FROM_ACTIVESET_FAILED \n
			 RET_ADD_TO_ACTIVESET_FAILED */
returnValue QProblem_changeActiveSet(	QProblem* _THIS,
										int BC_idx, 				/**< Index of blocking constraint. */
										SubjectToStatus BC_status,	/**< Status of blocking constraint. */
										BooleanType BC_isBound 		/**< Indicates if blocking constraint is a bound. */
										);


/** Compute relative length of homotopy in data space for termination
 *  criterion.
 *  \return Relative length in data space. */
real_t QProblem_getRelativeHomotopyLength(	QProblem* _THIS,
											const real_t* const g_new,		/**< Final gradient. */
											const real_t* const lb_new,		/**< Final lower variable bounds. */
											const real_t* const ub_new,		/**< Final upper variable bounds. */
											const real_t* const lbA_new,	/**< Final lower constraint bounds. */
											const real_t* const ubA_new		/**< Final upper constraint bounds. */
											);


/** Ramping Strategy to avoid ties. Modifies homotopy start without
 *  changing current active set.
 *  \return SUCCESSFUL_RETURN */
returnValue QProblem_performRamping( QProblem* _THIS );


/** ... */
returnValue QProblem_updateFarBounds(	QProblem* _THIS,
										real_t curFarBound,				/**< ... */
										int nRamp,						/**< ... */
										const real_t* const lb_new,		/**< ... */
										real_t* const lb_new_far,		/**< ... */
										const real_t* const ub_new,		/**< ... */
										real_t* const ub_new_far,		/**< ... */
										const real_t* const lbA_new,	/**< ... */
										real_t* const lbA_new_far,		/**< ... */
										const real_t* const ubA_new,	/**< ... */
										real_t* const ubA_new_far		/**< ... */
										);

/** ... */
returnValue QProblemBCPY_updateFarBounds(	QProblem* _THIS,
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
returnValue QProblem_performRatioTestC(	QProblem* _THIS,
										int nIdx, 						/**< Number of ratios to be checked. */
										const int* const idxList, 		/**< Array containing the indices of all ratios to be checked. */
										Constraints* const subjectTo,	/**< Constraint object corresponding to ratios to be checked. */
										const real_t* const num,	 	/**< Array containing all numerators for performing the ratio test. */
										const real_t* const den,		/**< Array containing all denominators for performing the ratio test. */
										real_t epsNum,					/**< Numerator tolerance. */
										real_t epsDen,					/**< Denominator tolerance. */
										real_t* t,						/**< Output: Maximum possible step length along the homotopy path. */
										int* BC_idx 					/**< Output: Index of blocking constraint. */
										);


/** Drift correction at end of each active set iteration
 *  \return SUCCESSFUL_RETURN */
returnValue QProblem_performDriftCorrection( QProblem* _THIS );


/** Updates QP vectors, working sets and internal data structures in order to
	start from an optimal solution corresponding to initial guesses of the working
	set for bounds and constraints.
 *	\return SUCCESSFUL_RETURN \n
 *			RET_SETUP_AUXILIARYQP_FAILED \n
			RET_INVALID_ARGUMENTS */
returnValue QProblem_setupAuxiliaryQP(	QProblem* _THIS,
										Bounds* const guessedBounds,			/**< Initial guess for working set of bounds. */
										Constraints* const guessedConstraints	/**< Initial guess for working set of constraints. */
										);

/** Determines if it is more efficient to refactorise the matrices when
 *  hotstarting or not (i.e. better to update the existing factorisations).
 *	\return BT_TRUE iff matrices shall be refactorised afresh
 */
BooleanType QProblem_shallRefactorise(	QProblem* _THIS,
										Bounds* const guessedBounds,			/**< Guessed new working set of bounds. */
										Constraints* const guessedConstraints	/**< Guessed new working set of constraints. */
										);

/** Setups internal QP data.
 *	\return SUCCESSFUL_RETURN \n
			RET_INVALID_ARGUMENTS \n
			RET_UNKNONW_BUG */
returnValue QProblem_setupQPdataM(	QProblem* _THIS,
									DenseMatrix *_H, 			/**< Hessian matrix. \n
																	 If Hessian matrix is trivial,a NULL pointer can be passed. */
									const real_t* const _g, 	/**< Gradient vector. */
									DenseMatrix *_A, 			 /**< Constraint matrix. */
									const real_t* const _lb,	/**< Lower bound vector (on variables). \n
																	 If no lower bounds exist, a NULL pointer can be passed. */
									const real_t* const _ub,	/**< Upper bound vector (on variables). \n
																	 If no upper bounds exist, a NULL pointer can be passed. */
									const real_t* const _lbA,	/**< Lower constraints' bound vector. \n
																	 If no lower constraints' bounds exist, a NULL pointer can be passed. */
									const real_t* const _ubA	/**< Upper constraints' bound vector. \n
																	 If no lower constraints' bounds exist, a NULL pointer can be passed. */
									);


/** Sets up dense internal QP data. If the current Hessian is trivial
 *  (i.e. HST_ZERO or HST_IDENTITY) but a non-trivial one is given,
 *  memory for Hessian is allocated and it is set to the given one.
 *	\return SUCCESSFUL_RETURN \n
			RET_INVALID_ARGUMENTS \n
			RET_UNKNONW_BUG */
returnValue QProblem_setupQPdata(	QProblem* _THIS,
									real_t* const _H, 			/**< Hessian matrix. \n
																	 If Hessian matrix is trivial,a NULL pointer can be passed. */
									const real_t* const _g, 	/**< Gradient vector. */
									real_t* const _A,  			/**< Constraint matrix. */
									const real_t* const _lb,	/**< Lower bound vector (on variables). \n
																	 If no lower bounds exist, a NULL pointer can be passed. */
									const real_t* const _ub,	/**< Upper bound vector (on variables). \n
																	 If no upper bounds exist, a NULL pointer can be passed. */
									const real_t* const _lbA,	/**< Lower constraints' bound vector. \n
																	 If no lower constraints' bounds exist, a NULL pointer can be passed. */
									const real_t* const _ubA	/**< Upper constraints' bound vector. \n
																	 If no lower constraints' bounds exist, a NULL pointer can be passed. */
									);

/** Sets up internal QP data by loading it from files. If the current Hessian
 *  is trivial (i.e. HST_ZERO or HST_IDENTITY) but a non-trivial one is given,
 *  memory for Hessian is allocated and it is set to the given one.
 *	\return SUCCESSFUL_RETURN \n
			RET_UNABLE_TO_OPEN_FILE \n
			RET_UNABLE_TO_READ_FILE \n
			RET_INVALID_ARGUMENTS \n
			RET_UNKNONW_BUG */
returnValue QProblem_setupQPdataFromFile(	QProblem* _THIS,
											const char* const H_file, 	/**< Name of file where Hessian matrix, of neighbouring QP to be solved, is stored. \n
														     				 If Hessian matrix is trivial,a NULL pointer can be passed. */
											const char* const g_file, 	/**< Name of file where gradient, of neighbouring QP to be solved, is stored. */
											const char* const A_file,	/**< Name of file where constraint matrix, of neighbouring QP to be solved, is stored. */
											const char* const lb_file, 	/**< Name of file where lower bounds, of neighbouring QP to be solved, is stored. \n
										 			 						 If no lower bounds exist, a NULL pointer can be passed. */
											const char* const ub_file, 	/**< Name of file where upper bounds, of neighbouring QP to be solved, is stored. \n
										 			 						 If no upper bounds exist, a NULL pointer can be passed. */
											const char* const lbA_file, /**< Name of file where lower constraints' bounds, of neighbouring QP to be solved, is stored. \n
										 			 						 If no lower constraints' bounds exist, a NULL pointer can be passed. */
											const char* const ubA_file	/**< Name of file where upper constraints' bounds, of neighbouring QP to be solved, is stored. \n
										 			 						 If no upper constraints' bounds exist, a NULL pointer can be passed. */
											);

/** Loads new QP vectors from files (internal members are not affected!).
 *	\return SUCCESSFUL_RETURN \n
			RET_UNABLE_TO_OPEN_FILE \n
			RET_UNABLE_TO_READ_FILE \n
			RET_INVALID_ARGUMENTS */
returnValue QProblem_loadQPvectorsFromFile(	QProblem* _THIS,
											const char* const g_file, 	/**< Name of file where gradient, of neighbouring QP to be solved, is stored. */
											const char* const lb_file, 	/**< Name of file where lower bounds, of neighbouring QP to be solved, is stored. \n
										 			 						 If no lower bounds exist, a NULL pointer can be passed. */
											const char* const ub_file, 	/**< Name of file where upper bounds, of neighbouring QP to be solved, is stored. \n
										 			 						 If no upper bounds exist, a NULL pointer can be passed. */
											const char* const lbA_file, /**< Name of file where lower constraints' bounds, of neighbouring QP to be solved, is stored. \n
										 			 						 If no lower constraints' bounds exist, a NULL pointer can be passed. */
											const char* const ubA_file, /**< Name of file where upper constraints' bounds, of neighbouring QP to be solved, is stored. \n
										 			 						 If no upper constraints' bounds exist, a NULL pointer can be passed. */
											real_t* const g_new,		/**< Output: Gradient of neighbouring QP to be solved. */
											real_t* const lb_new,		/**< Output: Lower bounds of neighbouring QP to be solved */
											real_t* const ub_new,		/**< Output: Upper bounds of neighbouring QP to be solved */
											real_t* const lbA_new,		/**< Output: Lower constraints' bounds of neighbouring QP to be solved */
											real_t* const ubA_new		/**< Output: Upper constraints' bounds of neighbouring QP to be solved */
											);


/** Prints concise information on the current iteration.
 *	\return  SUCCESSFUL_RETURN \n */
returnValue QProblem_printIteration(	QProblem* _THIS,
										int iter,					/**< Number of current iteration. */
										int BC_idx, 				/**< Index of blocking constraint. */
										SubjectToStatus BC_status,	/**< Status of blocking constraint. */
										BooleanType BC_isBound,		/**< Indicates if blocking constraint is a bound. */
										real_t homotopyLength,		/**< Current homotopy distance. */
										BooleanType isFirstCall		/**< Indicating whether this is the first call for current QP. */
 										);


/** Sets constraint matrix of the QP. \n
	Note: Also internal vector Ax is recomputed!
 *	\return SUCCESSFUL_RETURN \n
 *			RET_INVALID_ARGUMENTS */
static inline returnValue QProblem_setAM(	QProblem* _THIS,
											DenseMatrix *A_new	/**< New constraint matrix. */
											);

/** Sets dense constraint matrix of the QP. \n
	Note: Also internal vector Ax is recomputed!
 *	\return SUCCESSFUL_RETURN \n
 *			RET_INVALID_ARGUMENTS */
static inline returnValue QProblem_setA(	QProblem* _THIS,
											real_t* const A_new	/**< New dense constraint matrix (with correct dimension!). */
											);


/** Sets constraints' lower bound vector of the QP.
 *	\return SUCCESSFUL_RETURN \n
 *			RET_QPOBJECT_NOT_SETUP */
static inline returnValue QProblem_setLBA(	QProblem* _THIS,
											const real_t* const lbA_new	/**< New constraints' lower bound vector (with correct dimension!). */
											);

/** Changes single entry of lower constraints' bound vector of the QP.
 *	\return SUCCESSFUL_RETURN  \n
 *			RET_QPOBJECT_NOT_SETUP \n
 *			RET_INDEX_OUT_OF_BOUNDS */
static inline returnValue QProblem_setLBAn(	QProblem* _THIS,
											int number,		/**< Number of entry to be changed. */
											real_t value	/**< New value for entry of lower constraints' bound vector (with correct dimension!). */
											);

/** Sets constraints' upper bound vector of the QP.
 *	\return SUCCESSFUL_RETURN \n
 *			RET_QPOBJECT_NOT_SETUP */
static inline returnValue QProblem_setUBA(	QProblem* _THIS,
											const real_t* const ubA_new	/**< New constraints' upper bound vector (with correct dimension!). */
											);

/** Changes single entry of upper constraints' bound vector of the QP.
 *	\return SUCCESSFUL_RETURN  \n
 *			RET_QPOBJECT_NOT_SETUP \n
 *			RET_INDEX_OUT_OF_BOUNDS */
static inline returnValue QProblem_setUBAn(	QProblem* _THIS,
											int number,		/**< Number of entry to be changed. */
											real_t value	/**< New value for entry of upper constraints' bound vector (with correct dimension!). */
											);


/** Decides if lower bounds are smaller than upper bounds
 *
 * \return SUCCESSFUL_RETURN \n
 * 		   RET_QP_INFEASIBLE */
returnValue QProblem_areBoundsConsistent(	QProblem* _THIS,
											const real_t* const lb,		/**< Vector of lower bounds*/
											const real_t* const ub,		/**< Vector of upper bounds*/
											const real_t* const lbA,	/**< Vector of lower constraints*/
											const real_t* const ubA		/**< Vector of upper constraints*/
											);


/** Drops the blocking bound/constraint that led to infeasibility, or finds another
 *  bound/constraint to drop according to drop priorities.
 *  \return SUCCESSFUL_RETURN \n
 */
returnValue QProblem_dropInfeasibles ( 	QProblem* _THIS,
										int BC_number,				/**< Number of the bound or constraint to be added */
										SubjectToStatus BC_status, 	/**< New status of the bound or constraint to be added */
										BooleanType BC_isBound,		/**< Whether a bound or a constraint is to be added */
										real_t *xiB,
										real_t *xiC
										);


/** If Hessian type has been set by the user, nothing is done.
 *  Otherwise the Hessian type is set to HST_IDENTITY, HST_ZERO, or
 *  HST_POSDEF (default), respectively.
 *	\return SUCCESSFUL_RETURN \n
			RET_HESSIAN_INDEFINITE */
returnValue QProblem_determineHessianType( QProblem* _THIS );

/** Computes the Cholesky decomposition of the (simply projected) Hessian
 *  (i.e. R^T*R = Z^T*H*Z). It only works in the case where Z is a simple
 *  projection matrix!
 *  Note: If Hessian turns out not to be positive definite, the Hessian type
 *		  is set to HST_SEMIDEF accordingly.
 *	\return SUCCESSFUL_RETURN \n
 *			RET_HESSIAN_NOT_SPD \n
 *			RET_INDEXLIST_CORRUPTED */
returnValue QProblemBCPY_computeCholesky( QProblem* _THIS );

/** Obtains the desired working set for the auxiliary initial QP in
 *  accordance with the user specifications
 *	\return SUCCESSFUL_RETURN \n
			RET_OBTAINING_WORKINGSET_FAILED \n
			RET_INVALID_ARGUMENTS */
returnValue QProblemBCPY_obtainAuxiliaryWorkingSet(	QProblem* _THIS,
													const real_t* const xOpt,		/**< Optimal primal solution vector.
																					 *	 If a NULL pointer is passed, all entries are assumed to be zero. */
													const real_t* const yOpt,		/**< Optimal dual solution vector.
																					 *	 If a NULL pointer is passed, all entries are assumed to be zero. */
													Bounds* const guessedBounds,	/**< Guessed working set for solution (xOpt,yOpt). */
													Bounds* auxiliaryBounds			/**< Input:  Allocated bound object. \n
																					 *	 Output: Working set for auxiliary QP. */
													);


/** Solves the system Ra = b or R^Ta = b where R is an upper triangular matrix.
 *	\return SUCCESSFUL_RETURN \n
			RET_DIV_BY_ZERO */
returnValue QProblem_backsolveR(	QProblem* _THIS,
									const real_t* const b,	/**< Right hand side vector. */
									BooleanType transposed,	/**< Indicates if the transposed system shall be solved. */
									real_t* const a 		/**< Output: Solution vector */
									);

/** Solves the system Ra = b or R^Ta = b where R is an upper triangular matrix. \n
 *  Special variant for the case that _THIS function is called from within "removeBound()".
 *	\return SUCCESSFUL_RETURN \n
			RET_DIV_BY_ZERO */
returnValue QProblem_backsolveRrem(	QProblem* _THIS,
									const real_t* const b,		/**< Right hand side vector. */
									BooleanType transposed,		/**< Indicates if the transposed system shall be solved. */
									BooleanType removingBound,	/**< Indicates if function is called from "removeBound()". */
									real_t* const a 			/**< Output: Solution vector */
									);


/** Determines step direction of the shift of the QP data.
 *	\return SUCCESSFUL_RETURN */
returnValue QProblemBCPY_determineDataShift(	QProblem* _THIS,
												const real_t* const g_new,		/**< New gradient vector. */
												const real_t* const lb_new,		/**< New lower bounds. */
												const real_t* const ub_new,		/**< New upper bounds. */
												real_t* const delta_g,	 		/**< Output: Step direction of gradient vector. */
												real_t* const delta_lb,	 		/**< Output: Step direction of lower bounds. */
												real_t* const delta_ub,	 		/**< Output: Step direction of upper bounds. */
												BooleanType* Delta_bB_isZero	/**< Output: Indicates if active bounds are to be shifted. */
												);


/** Sets up internal QP data.
 *	\return SUCCESSFUL_RETURN \n
			RET_INVALID_ARGUMENTS */
returnValue QProblemBCPY_setupQPdataM(	QProblem* _THIS,
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
returnValue QProblemBCPY_setupQPdata(	QProblem* _THIS,
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
returnValue QProblemBCPY_setupQPdataFromFile(	QProblem* _THIS,
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
returnValue QProblemBCPY_loadQPvectorsFromFile(	QProblem* _THIS,
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
returnValue QProblem_setInfeasibilityFlag(	QProblem* _THIS,
											returnValue returnvalue,	/**< Returnvalue to be tunneled. */
											BooleanType doThrowError	/**< Flag forcing to throw an error. */
											);


/** Determines if next QP iteration can be performed within given CPU time limit.
 *	\return BT_TRUE: CPU time limit is exceeded, stop QP solution. \n
			BT_FALSE: Sufficient CPU time for next QP iteration. */
BooleanType QProblem_isCPUtimeLimitExceeded(	QProblem* _THIS,
												const real_t* const cputime,	/**< Maximum CPU time allowed for QP solution. */
												real_t starttime,				/**< Start time of current QP solution. */
												int nWSR						/**< Number of working set recalculations performed so far. */
												);


/** Regularise Hessian matrix by adding a scaled identity matrix to it.
 *	\return SUCCESSFUL_RETURN \n
			RET_HESSIAN_ALREADY_REGULARISED */
returnValue QProblem_regulariseHessian( QProblem* _THIS );


/** Sets Hessian matrix of the QP.
 *	\return SUCCESSFUL_RETURN */
static inline returnValue QProblem_setHM(	QProblem* _THIS,
											DenseMatrix* H_new	/**< New Hessian matrix. */
											);

/** Sets dense Hessian matrix of the QP.
 *  If a null pointer is passed and
 *  a) hessianType is HST_IDENTITY, nothing is done,
 *  b) hessianType is not HST_IDENTITY, Hessian matrix is set to zero.
 *	\return SUCCESSFUL_RETURN */
static inline returnValue QProblem_setH(	QProblem* _THIS,
											real_t* const H_new	/**< New dense Hessian matrix (with correct dimension!). */
											);

/** Changes gradient vector of the QP.
 *	\return SUCCESSFUL_RETURN \n
 *			RET_INVALID_ARGUMENTS */
static inline returnValue QProblem_setG(	QProblem* _THIS,
											const real_t* const g_new	/**< New gradient vector (with correct dimension!). */
											);

/** Changes lower bound vector of the QP.
 *	\return SUCCESSFUL_RETURN \n
 *			RET_INVALID_ARGUMENTS */
static inline returnValue QProblem_setLB(	QProblem* _THIS,
											const real_t* const lb_new	/**< New lower bound vector (with correct dimension!). */
											);

/** Changes single entry of lower bound vector of the QP.
 *	\return SUCCESSFUL_RETURN  \n
			RET_INDEX_OUT_OF_BOUNDS */
static inline returnValue QProblem_setLBn(	QProblem* _THIS,
											int number,		/**< Number of entry to be changed. */
											real_t value	/**< New value for entry of lower bound vector. */
											);

/** Changes upper bound vector of the QP.
 *	\return SUCCESSFUL_RETURN \n
 *			RET_INVALID_ARGUMENTS */
static inline returnValue QProblem_setUB(	QProblem* _THIS,
											const real_t* const ub_new	/**< New upper bound vector (with correct dimension!). */
											);

/** Changes single entry of upper bound vector of the QP.
 *	\return SUCCESSFUL_RETURN  \n
			RET_INDEX_OUT_OF_BOUNDS */
static inline returnValue QProblem_setUBn(	QProblem* _THIS,
											int number,		/**< Number of entry to be changed. */
											real_t value	/**< New value for entry of upper bound vector. */
											);



/** Compute relative length of homotopy in data space for termination
 *  criterion.
 *  \return Relative length in data space. */
real_t QProblemBCPY_getRelativeHomotopyLength(	QProblem* _THIS,
												const real_t* const g_new,	/**< Final gradient. */
												const real_t* const lb_new,	/**< Final lower variable bounds. */
												const real_t* const ub_new	/**< Final upper variable bounds. */
												);



/** Performs robustified ratio test yield the maximum possible step length
 *  along the homotopy path.
 *	\return  SUCCESSFUL_RETURN */
returnValue QProblem_performRatioTestB(	QProblem* _THIS,
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
static inline BooleanType QProblem_isBlocking(	QProblem* _THIS,
												real_t num,			/**< Numerator for performing the ratio test. */
												real_t den,			/**< Denominator for performing the ratio test. */
												real_t epsNum,		/**< Numerator tolerance. */
												real_t epsDen,		/**< Denominator tolerance. */
												real_t* t			/**< Input:  Current maximum step length along the homotopy path,
																	 *   Output: Updated maximum possible step length along the homotopy path. */
												);


/** ...
 *	\return SUCCESSFUL_RETURN  \n
			RET_UNABLE_TO_OPEN_FILE */
returnValue QProblem_writeQpDataIntoMatFile(	QProblem* _THIS,
												const char* const filename	/**< Mat file name. */
												);

/** ...
*	\return SUCCESSFUL_RETURN  \n
			RET_UNABLE_TO_OPEN_FILE */
returnValue QProblem_writeQpWorkspaceIntoMatFile(	QProblem* _THIS,
													const char* const filename	/**< Mat file name. */
													);


/*
 *	g e t B o u n d s
 */
static inline returnValue QProblem_getBounds( QProblem* _THIS, Bounds* _bounds )
{
	int nV = QProblem_getNV( _THIS );

	if ( nV == 0 )
		return THROWERROR( RET_QPOBJECT_NOT_SETUP );

	_bounds = _THIS->bounds;

	return SUCCESSFUL_RETURN;
}


/*
 *	g e t N V
 */
static inline int QProblem_getNV( QProblem* _THIS )
{
	return Bounds_getNV( _THIS->bounds );
}


/*
 *	g e t N F R
 */
static inline int QProblem_getNFR( QProblem* _THIS )
{
	return Bounds_getNFR( _THIS->bounds );
}


/*
 *	g e t N F X
 */
static inline int QProblem_getNFX( QProblem* _THIS )
{
	return Bounds_getNFX( _THIS->bounds );
}


/*
 *	g e t N F V
 */
static inline int QProblem_getNFV( QProblem* _THIS )
{
	return Bounds_getNFV( _THIS->bounds );
}


/*
 *	g e t S t a t u s
 */
static inline QProblemStatus QProblem_getStatus( QProblem* _THIS )
{
	return _THIS->status;
}


/*
 *	i s I n i t i a l i s e d
 */
static inline BooleanType QProblem_isInitialised( QProblem* _THIS )
{
	if ( _THIS->status == QPS_NOTINITIALISED )
		return BT_FALSE;
	else
		return BT_TRUE;
}


/*
 *	i s S o l v e d
 */
static inline BooleanType QProblem_isSolved( QProblem* _THIS )
{
	if ( _THIS->status == QPS_SOLVED )
		return BT_TRUE;
	else
		return BT_FALSE;
}


/*
 *	i s I n f e a s i b l e
 */
static inline BooleanType QProblem_isInfeasible( QProblem* _THIS )
{
	return _THIS->infeasible;
}


/*
 *	i s U n b o u n d e d
 */
static inline BooleanType QProblem_isUnbounded( QProblem* _THIS )
{
	return _THIS->unbounded;
}


/*
 *	g e t H e s s i a n T y p e
 */
static inline HessianType QProblem_getHessianType( QProblem* _THIS )
{
	return _THIS->hessianType;
}


/*
 *	s e t H e s s i a n T y p e
 */
static inline returnValue QProblem_setHessianType( QProblem* _THIS, HessianType _hessianType )
{
	_THIS->hessianType = _hessianType;
	return SUCCESSFUL_RETURN;
}


/*
 *	u s i n g R e g u l a r i s a t i o n
 */
static inline BooleanType QProblem_usingRegularisation( QProblem* _THIS )
{
	if ( _THIS->regVal > QPOASES_ZERO )
		return BT_TRUE;
	else
		return BT_FALSE;
}


/*
 *	g e t O p t i o n s
 */
static inline Options QProblem_getOptions( QProblem* _THIS )
{
	return _THIS->options;
}


/*
 *	s e t O p t i o n s
 */
static inline returnValue QProblem_setOptions(	QProblem* _THIS,
												Options _options
												)
{
	OptionsCPY( &_options,&(_THIS->options) );
	Options_ensureConsistency( &(_THIS->options) );

	QProblem_setPrintLevel( _THIS,_THIS->options.printLevel );

	return SUCCESSFUL_RETURN;
}


/*
 *	g e t P r i n t L e v e l
 */
static inline PrintLevel QProblem_getPrintLevel( QProblem* _THIS )
{
	return _THIS->options.printLevel;
}


/*
 *	g e t C o u n t
 */
static inline unsigned int QProblem_getCount( QProblem* _THIS )
{
	return _THIS->count;
}


/*
 *	r e s e t C o u n t e r
 */
static inline returnValue QProblem_resetCounter( QProblem* _THIS )
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
static inline returnValue QProblem_setHM( QProblem* _THIS, DenseMatrix* H_new )
{
	if ( H_new == 0 )
		return QProblem_setH( _THIS,(real_t*)0 );
	else
		return QProblem_setH( _THIS,DenseMatrix_getVal(H_new) );
}


/*
 *	s e t H
 */
static inline returnValue QProblem_setH( QProblem* _THIS, real_t* const H_new )
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
		DenseMatrixCON( _THIS->H,QProblem_getNV( _THIS ),QProblem_getNV( _THIS ),QProblem_getNV( _THIS ),H_new );
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t G
 */
static inline returnValue QProblem_setG( QProblem* _THIS, const real_t* const g_new )
{
	unsigned int nV = (unsigned int)QProblem_getNV( _THIS );

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
static inline returnValue QProblem_setLB( QProblem* _THIS, const real_t* const lb_new )
{
	unsigned int i;
	unsigned int nV = (unsigned int)QProblem_getNV( _THIS );

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
static inline returnValue QProblem_setLBn( QProblem* _THIS, int number, real_t value )
{
	int nV = QProblem_getNV( _THIS );

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
static inline returnValue QProblem_setUB( QProblem* _THIS, const real_t* const ub_new )
{
	unsigned int i;
	unsigned int nV = (unsigned int)QProblem_getNV( _THIS );

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
static inline returnValue QProblem_setUBn( QProblem* _THIS, int number, real_t value )
{
	int nV = QProblem_getNV( _THIS );

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
 * i s B l o c k i n g
 */
static inline BooleanType QProblem_isBlocking(	QProblem* _THIS,
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



/*
 *	g e t C o n s t r a i n t s
 */
static inline returnValue QProblem_getConstraints( QProblem* _THIS, Constraints* _constraints )
{
	int nV = QProblem_getNV( _THIS );

	if ( nV == 0 )
		return THROWERROR( RET_QPOBJECT_NOT_SETUP );

	ConstraintsCPY( _THIS->constraints,_constraints );

	return SUCCESSFUL_RETURN;
}



/*
 *	g e t N C
 */
static inline int QProblem_getNC( QProblem* _THIS )
{
	return Constraints_getNC( _THIS->constraints );
}


/*
 *	g e t N E C
 */
static inline int QProblem_getNEC( QProblem* _THIS )
{
	return Constraints_getNEC( _THIS->constraints );
}


/*
 *	g e t N A C
 */
static inline int QProblem_getNAC( QProblem* _THIS )
{
	return Constraints_getNAC( _THIS->constraints );
}


/*
 *	g e t N I A C
 */
static inline int QProblem_getNIAC( QProblem* _THIS )
{
	return Constraints_getNIAC( _THIS->constraints );
}



/*****************************************************************************
 *  P R O T E C T E D                                                        *
 *****************************************************************************/


/*
 *	s e t A
 */
static inline returnValue QProblem_setAM( QProblem* _THIS, DenseMatrix *A_new )
{
	if ( A_new == 0 )
		return QProblem_setA( _THIS,(real_t*)0 );
	else
		return QProblem_setA( _THIS,DenseMatrix_getVal(A_new) );
}


/*
 *	s e t A
 */
static inline returnValue QProblem_setA( QProblem* _THIS, real_t* const A_new )
{
	int j;
	int nV = QProblem_getNV( _THIS );
	int nC = QProblem_getNC( _THIS );

	if ( nV == 0 )
		return THROWERROR( RET_QPOBJECT_NOT_SETUP );

	if ( A_new == 0 )
		return THROWERROR( RET_INVALID_ARGUMENTS );

	DenseMatrixCON( _THIS->A,QProblem_getNC( _THIS ),QProblem_getNV( _THIS ),QProblem_getNV( _THIS ),A_new );

	DenseMatrix_times( _THIS->A,1, 1.0, _THIS->x, nV, 0.0, _THIS->Ax, nC);

	for( j=0; j<nC; ++j )
	{
		_THIS->Ax_u[j] = _THIS->ubA[j] - _THIS->Ax[j];
		_THIS->Ax_l[j] = _THIS->Ax[j] - _THIS->lbA[j];

		/* (ckirches) disable constraints with empty rows */
		if ( qpOASES_isZero( DenseMatrix_getRowNorm( _THIS->A,j,2 ),QPOASES_ZERO ) == BT_TRUE )
			Constraints_setType( _THIS->constraints,j,ST_DISABLED );
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t L B A
 */
static inline returnValue QProblem_setLBA( QProblem* _THIS, const real_t* const lbA_new )
{
	unsigned int i;
	unsigned int nV = (unsigned int)QProblem_getNV( _THIS );
	unsigned int nC = (unsigned int)QProblem_getNC( _THIS );

	if ( nV == 0 )
		return THROWERROR( RET_QPOBJECT_NOT_SETUP );

	if ( lbA_new != 0 )
	{
		memcpy( _THIS->lbA,lbA_new,nC*sizeof(real_t) );
	}
	else
	{
		/* if no lower constraints' bounds are specified, set them to -infinity */
		for( i=0; i<nC; ++i )
			_THIS->lbA[i] = -QPOASES_INFTY;
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t L B A
 */
static inline returnValue QProblem_setLBAn( QProblem* _THIS, int number, real_t value )
{
	int nV = QProblem_getNV( _THIS );
	int nC = QProblem_getNC( _THIS );

	if ( nV == 0 )
		return THROWERROR( RET_QPOBJECT_NOT_SETUP );

	if ( ( number >= 0 ) && ( number < nC ) )
	{
		_THIS->lbA[number] = value;
		return SUCCESSFUL_RETURN;
	}
	else
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );
}


/*
 *	s e t U B A
 */
static inline returnValue QProblem_setUBA( QProblem* _THIS, const real_t* const ubA_new )
{
	unsigned int i;
	unsigned int nV = (unsigned int)QProblem_getNV( _THIS );
	unsigned int nC = (unsigned int)QProblem_getNC( _THIS );

	if ( nV == 0 )
		return THROWERROR( RET_QPOBJECT_NOT_SETUP );

	if ( ubA_new != 0 )
	{
		memcpy( _THIS->ubA,ubA_new,nC*sizeof(real_t) );
	}
	else
	{
		/* if no upper constraints' bounds are specified, set them to infinity */
		for( i=0; i<nC; ++i )
			_THIS->ubA[i] = QPOASES_INFTY;
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t U B A
 */
static inline returnValue QProblem_setUBAn( QProblem* _THIS, int number, real_t value )
{
	int nV = QProblem_getNV( _THIS );
	int nC = QProblem_getNC( _THIS );

	if ( nV == 0 )
		return THROWERROR( RET_QPOBJECT_NOT_SETUP );

	if ( ( number >= 0 ) && ( number < nC ) )
	{
		_THIS->ubA[number] = value;
		return SUCCESSFUL_RETURN;
	}
	else
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );
}


END_NAMESPACE_QPOASES


#endif	/* QPOASES_QPROBLEM_H */


/*
 *	end of file
 */
