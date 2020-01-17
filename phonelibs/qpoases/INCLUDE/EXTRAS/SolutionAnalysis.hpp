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
 *	\file INCLUDE/EXTRAS/SolutionAnalysis.hpp
 *	\author Milan Vukov, Boris Houska, Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2012
 *
 *	Solution analysis class, based on a class in the standard version of the qpOASES
 */


//

#ifndef QPOASES_SOLUTIONANALYSIS_HPP
#define QPOASES_SOLUTIONANALYSIS_HPP

#include <QProblem.hpp>

/** Enables the computation of variance as is in the standard version of qpOASES */
#define QPOASES_USE_OLD_VERSION 0

#if QPOASES_USE_OLD_VERSION
#define KKT_DIM (2 * NVMAX + NCMAX)
#endif

class SolutionAnalysis
{
public:
	
	/** Default constructor. */
	SolutionAnalysis( );
	
	/** Copy constructor (deep copy). */
	SolutionAnalysis( 	const SolutionAnalysis& rhs	/**< Rhs object. */
						);
	
	/** Destructor. */
	~SolutionAnalysis( );
	
	/** Copy asingment operator (deep copy). */
	SolutionAnalysis& operator=(	const SolutionAnalysis& rhs	/**< Rhs object. */
									);
	
	/** A routine for computation of inverse of the Hessian matrix. */
	returnValue getHessianInverse(
									QProblem* qp,			/** QP */
									real_t* hessianInverse	/** Inverse of the Hessian matrix*/
									);
	
	/** A routine for computation of inverse of the Hessian matrix. */
	returnValue getHessianInverse(	QProblemB* qp,			/** QP */
									real_t* hessianInverse	/** Inverse of the Hessian matrix*/
									);

#if QPOASES_USE_OLD_VERSION
	returnValue getVarianceCovariance(
										QProblem* qp,
										real_t* g_b_bA_VAR,
										real_t* Primal_Dual_VAR
										);
#endif
	
private:
	
	real_t delta_g_cov[ NVMAX ];		/** A covariance-vector of g */
	real_t delta_lb_cov[ NVMAX ];		/** A covariance-vector of lb */
	real_t delta_ub_cov[ NVMAX ];		/** A covariance-vector of ub */
	real_t delta_lbA_cov[ NCMAX_ALLOC ];		/** A covariance-vector of lbA */
	real_t delta_ubA_cov[ NCMAX_ALLOC ];		/** A covariance-vector of ubA */
	
#if QPOASES_USE_OLD_VERSION
	real_t K[KKT_DIM * KKT_DIM];		/** A matrix to store an intermediate result */
#endif
	
	int FR_idx[ NVMAX ];				/** Index array for free variables */
	int FX_idx[ NVMAX ];				/** Index array for fixed variables */
	int AC_idx[ NCMAX_ALLOC ];				/** Index array for active constraints */
	
	real_t delta_xFR[ NVMAX ];			/** QP reaction, primal, w.r.t. free */
	real_t delta_xFX[ NVMAX ];			/** QP reaction, primal, w.r.t. fixed */
	real_t delta_yAC[ NVMAX ];			/** QP reaction, dual, w.r.t. active */
	real_t delta_yFX[ NVMAX ];			/** QP reaction, dual, w.r.t. fixed*/
};

#endif // QPOASES_SOLUTIONANALYSIS_HPP
