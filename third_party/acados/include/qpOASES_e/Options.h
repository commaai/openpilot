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
 *	\file include/qpOASES_e/Options.h
 *	\author Hans Joachim Ferreau, Andreas Potschka, Christian Kirches
 *	\version 3.1embedded
 *	\date 2007-2015
 *
 *	Declaration of the Options class designed to manage user-specified
 *	options for solving a QProblem.
 */


#ifndef QPOASES_OPTIONS_H
#define QPOASES_OPTIONS_H


#include <qpOASES_e/Utils.h>


BEGIN_NAMESPACE_QPOASES


/**
 *	\brief Manages all user-specified options for solving QPs.
 *
 *	This class manages all user-specified options used for solving
 *	quadratic programs.
 *
 *	\author Hans Joachim Ferreau, Andreas Potschka, Christian Kirches
 *	\version 3.1embedded
 *	\date 2007-2015
 */
typedef struct
{
	PrintLevel printLevel;					/**< Print level. */

	BooleanType enableRamping;				/**< Specifies whether ramping shall be enabled or not. */
	BooleanType enableFarBounds;			/**< Specifies whether far bounds shall be used or not. */
	BooleanType enableFlippingBounds;		/**< Specifies whether flipping bounds shall be used or not. */
	BooleanType enableRegularisation;		/**< Specifies whether Hessian matrix shall be regularised in case semi-definiteness is detected. */
	BooleanType enableFullLITests;			/**< Specifies whether condition-hardened LI test shall be used or not. */
	BooleanType enableNZCTests;				/**< Specifies whether nonzero curvature tests shall be used. */
	int enableDriftCorrection;				/**< Specifies the frequency of drift corrections (0 = off). */
	int enableCholeskyRefactorisation;		/**< Specifies the frequency of full refactorisation of proj. Hessian (otherwise updates). */
	BooleanType enableEqualities;			/**< Specifies whether equalities shall be always treated as active constraints. */

	real_t terminationTolerance;			/**< Termination tolerance. */
	real_t boundTolerance;					/**< Lower/upper (constraints') bound tolerance (an inequality constraint whose lower and upper bounds differ by less is regarded to be an equality constraint). */
	real_t boundRelaxation;					/**< Offset for relaxing (constraints') bounds at beginning of an initial homotopy. It is also as initial value for far bounds. */
	real_t epsNum;							/**< Numerator tolerance for ratio tests. */
	real_t epsDen;							/**< Denominator tolerance for ratio tests. */
	real_t maxPrimalJump;					/**< Maximum allowed jump in primal variables in nonzero curvature tests. */
	real_t maxDualJump;						/**< Maximum allowed jump in dual variables in linear independence tests. */

	real_t initialRamping;					/**< Start value for Ramping Strategy. */
	real_t finalRamping;					/**< Final value for Ramping Strategy. */
	real_t initialFarBounds;				/**< Initial size of Far Bounds. */
	real_t growFarBounds;					/**< Factor to grow Far Bounds. */
	SubjectToStatus initialStatusBounds;	/**< Initial status of bounds at first iteration. */
	real_t epsFlipping;						/**< Tolerance of squared Cholesky diagonal factor which triggers flipping bound. */
	int numRegularisationSteps;				/**< Maximum number of successive regularisation steps. */
	real_t epsRegularisation;				/**< Scaling factor of identity matrix used for Hessian regularisation. */
	int numRefinementSteps;					/**< Maximum number of iterative refinement steps. */
	real_t epsIterRef;						/**< Early termination tolerance for iterative refinement. */
	real_t epsLITests;						/**< Tolerance for linear independence tests. */
	real_t epsNZCTests;						/**< Tolerance for nonzero curvature tests. */

	BooleanType enableDropInfeasibles;		/**< ... */
	int    dropBoundPriority;				/**< ... */
	int    dropEqConPriority;				/**< ... */
	int    dropIneqConPriority;				/**< ... */
} Options;


void OptionsCON(	Options* _THIS
					);

/** Copies all members from given rhs object.
 *  \return SUCCESSFUL_RETURN */
void OptionsCPY(	Options* FROM,
					Options* TO
					);


/** Sets all options to default values.
 *	\return SUCCESSFUL_RETURN */
returnValue Options_setToDefault(	Options* _THIS
									);

/** Sets all options to values resulting in maximum reliabilty.
 *	\return SUCCESSFUL_RETURN */
returnValue Options_setToReliable(	Options* _THIS
									);

/** Sets all options to values resulting in minimum solution time.
 *	\return SUCCESSFUL_RETURN */
returnValue Options_setToMPC(		Options* _THIS
									);

/** Same as setToMPC( ), for ensuring backwards compatibility.
 *	\return SUCCESSFUL_RETURN */
returnValue Options_setToFast(		Options* _THIS
									);


/** Ensures that all options have consistent values by automatically
 *	adjusting inconsistent ones.
 *	Note: This routine cannot (and does not try to) ensure that values
 *	      are set to reasonable values that make the QP solution work!
 *	\return SUCCESSFUL_RETURN \n 
 *		    RET_OPTIONS_ADJUSTED */
returnValue Options_ensureConsistency(	Options* _THIS
										);


/** Prints values of all options.
 *	\return SUCCESSFUL_RETURN */
returnValue Options_print(	Options* _THIS
							);


END_NAMESPACE_QPOASES


#endif	/* QPOASES_OPTIONS_H */


/*
 *	end of file
 */
