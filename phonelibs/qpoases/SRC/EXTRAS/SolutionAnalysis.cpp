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
 *	\file SRC/EXTRAS/SolutionAnalysis.cpp
 *	\author Milan Vukov, Boris Houska, Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2012
 *
 *	Solution analysis class, based on a class in the standard version of the qpOASES
 */

#include <EXTRAS/SolutionAnalysis.hpp>

/*
 *	S o l u t i o n A n a l y s i s
 */
SolutionAnalysis::SolutionAnalysis( )
{
	
}

/*
 *	S o l u t i o n A n a l y s i s
 */
SolutionAnalysis::SolutionAnalysis( const SolutionAnalysis& rhs )
{
	
}

/*
 *	~ S o l u t i o n A n a l y s i s
 */
SolutionAnalysis::~SolutionAnalysis( )
{
	
}

/*
 *	o p e r a t o r =
 */
SolutionAnalysis& SolutionAnalysis::operator=( const SolutionAnalysis& rhs )
{
	if ( this != &rhs )
	{
		
	}
	
	return *this;
}

/*
 * g e t H e s s i a n I n v e r s e
 */
returnValue SolutionAnalysis::getHessianInverse( QProblem* qp, real_t* hessianInverse )
{
	returnValue returnvalue; /* the return value */
	BooleanType Delta_bC_isZero = BT_FALSE; /* (just use FALSE here) */
	BooleanType Delta_bB_isZero = BT_FALSE; /* (just use FALSE here) */
	
	register int run1, run2, run3;
	
	register int nFR, nFX;
	
	/* Ask for the number of free and fixed variables, assumes that active set
	 * is constant for the covariance evaluation */
	nFR = qp->getNFR( );
	nFX = qp->getNFX( );
	
	/* Ask for the corresponding index arrays: */
	if ( qp->bounds.getFree( )->getNumberArray( FR_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_HOTSTART_FAILED );
	
	if ( qp->bounds.getFixed( )->getNumberArray( FX_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_HOTSTART_FAILED );
	
	if ( qp->constraints.getActive( )->getNumberArray( AC_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_HOTSTART_FAILED );
	
	/* Initialization: */
	for( run1 = 0; run1 < NVMAX; run1++ )
		delta_g_cov[ run1 ] = 0.0;
	
	for( run1 = 0; run1 < NVMAX; run1++ )
		delta_lb_cov[ run1 ] = 0.0;
	
	for( run1 = 0; run1 < NVMAX; run1++ )
		delta_ub_cov[ run1 ] = 0.0;
	
	for( run1 = 0; run1 < NCMAX; run1++ )
		delta_lbA_cov[ run1 ] = 0.0;
	
	for( run1 = 0; run1 < NCMAX; run1++ )
		delta_ubA_cov[ run1 ] = 0.0;
	
	/* The following loop solves the following:
	 *
	 * KKT * x =
	 *   [delta_g_cov', delta_lbA_cov', delta_ubA_cov', delta_lb_cov', delta_ub_cov]'
	 *
	 * for the first NVMAX (negative) elementary vectors in order to get
	 * transposed inverse of the Hessian. Assuming that the Hessian is
	 * symmetric, the function will return transposed inverse, instead of the
	 * true inverse.
	 *
	 * Note, that we use negative elementary vectors due because internal
	 * implementation of the function hotstart_determineStepDirection requires
	 * so.
	 *
	 * */
	
	for( run3 = 0; run3 < NVMAX; run3++ )
	{
		/* Line wise loading of the corresponding (negative) elementary vector: */
		delta_g_cov[ run3 ] = -1.0;
		
		/* Evaluation of the step: */
		returnvalue = qp->hotstart_determineStepDirection(
			FR_idx, FX_idx, AC_idx,
			delta_g_cov, delta_lbA_cov, delta_ubA_cov, delta_lb_cov, delta_ub_cov,
			Delta_bC_isZero, Delta_bB_isZero,
			delta_xFX, delta_xFR, delta_yAC, delta_yFX
			);
		if ( returnvalue != SUCCESSFUL_RETURN )
		{
			return returnvalue;
		}
		
		/* Line wise storage of the QP reaction: */
		for( run1 = 0; run1 < nFR; run1++ )
		{
			run2 = FR_idx[ run1 ];
			
			hessianInverse[run3 * NVMAX + run2] = delta_xFR[ run1 ];
		} 
		
		for( run1 = 0; run1 < nFX; run1++ )
		{ 
			run2 = FX_idx[ run1 ];
			
			hessianInverse[run3 * NVMAX + run2] = delta_xFX[ run1 ];
		}
		
		/* Prepare for the next iteration */
		delta_g_cov[ run3 ] = 0.0;
	}
	
	// TODO: Perform the transpose of the inverse of the Hessian matrix
	
	return SUCCESSFUL_RETURN; 
}

/*
 * g e t H e s s i a n I n v e r s e
 */
returnValue SolutionAnalysis::getHessianInverse( QProblemB* qp, real_t* hessianInverse )
{
	returnValue returnvalue; /* the return value */
	BooleanType Delta_bB_isZero = BT_FALSE; /* (just use FALSE here) */
	
	register int run1, run2, run3;
	
	register int nFR, nFX;
	
	/* Ask for the number of free and fixed variables, assumes that active set
	 * is constant for the covariance evaluation */
	nFR = qp->getNFR( );
	nFX = qp->getNFX( );
	
	/* Ask for the corresponding index arrays: */
	if ( qp->bounds.getFree( )->getNumberArray( FR_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_HOTSTART_FAILED );
	
	if ( qp->bounds.getFixed( )->getNumberArray( FX_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_HOTSTART_FAILED );
	
	/* Initialization: */
	for( run1 = 0; run1 < NVMAX; run1++ )
		delta_g_cov[ run1 ] = 0.0;
	
	for( run1 = 0; run1 < NVMAX; run1++ )
		delta_lb_cov[ run1 ] = 0.0;
	
	for( run1 = 0; run1 < NVMAX; run1++ )
		delta_ub_cov[ run1 ] = 0.0;
	
	/* The following loop solves the following:
	 *
	 * KKT * x =
	 *   [delta_g_cov', delta_lb_cov', delta_ub_cov']'
	 *
	 * for the first NVMAX (negative) elementary vectors in order to get
	 * transposed inverse of the Hessian. Assuming that the Hessian is
	 * symmetric, the function will return transposed inverse, instead of the
	 * true inverse.
	 *
	 * Note, that we use negative elementary vectors due because internal
	 * implementation of the function hotstart_determineStepDirection requires
	 * so.
	 *
	 * */
	
	for( run3 = 0; run3 < NVMAX; run3++ )
	{
		/* Line wise loading of the corresponding (negative) elementary vector: */
		delta_g_cov[ run3 ] = -1.0;
		
		/* Evaluation of the step: */
		returnvalue = qp->hotstart_determineStepDirection(
			FR_idx, FX_idx,
			delta_g_cov, delta_lb_cov, delta_ub_cov,
			Delta_bB_isZero,
			delta_xFX, delta_xFR, delta_yFX
			);
		if ( returnvalue != SUCCESSFUL_RETURN )
		{
			return returnvalue;
		}
				
		/* Line wise storage of the QP reaction: */
		for( run1 = 0; run1 < nFR; run1++ )
		{
			run2 = FR_idx[ run1 ];
			
			hessianInverse[run3 * NVMAX + run2] = delta_xFR[ run1 ];
		} 
		
		for( run1 = 0; run1 < nFX; run1++ )
		{ 
			run2 = FX_idx[ run1 ];
			
			hessianInverse[run3 * NVMAX + run2] = delta_xFX[ run1 ];
		}
		
		/* Prepare for the next iteration */
		delta_g_cov[ run3 ] = 0.0;
	}
	
	// TODO: Perform the transpose of the inverse of the Hessian matrix
	
	return SUCCESSFUL_RETURN; 
}

/*
 * g e t V a r i a n c e C o v a r i a n c e
 */

#if QPOASES_USE_OLD_VERSION

returnValue SolutionAnalysis::getVarianceCovariance( QProblem* qp, real_t* g_b_bA_VAR, real_t* Primal_Dual_VAR )
{
	int run1, run2, run3; /* simple run variables (for loops). */
	
	returnValue returnvalue; /* the return value */
	BooleanType Delta_bC_isZero = BT_FALSE; /* (just use FALSE here) */
	BooleanType Delta_bB_isZero = BT_FALSE; /* (just use FALSE here) */
	
	/* ASK FOR THE NUMBER OF FREE AND FIXED VARIABLES:
	 * (ASSUMES THAT ACTIVE SET IS CONSTANT FOR THE
	 *  VARIANCE-COVARIANCE EVALUATION)
	 * ----------------------------------------------- */
	int nFR, nFX, nAC;
	
	nFR = qp->getNFR( );
	nFX = qp->getNFX( );
	nAC = qp->getNAC( );
	
	if ( qp->bounds.getFree( )->getNumberArray( FR_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_HOTSTART_FAILED );
	
	if ( qp->bounds.getFixed( )->getNumberArray( FX_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_HOTSTART_FAILED );
	
	if ( qp->constraints.getActive( )->getNumberArray( AC_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_HOTSTART_FAILED );
	
	/* SOME INITIALIZATIONS:
	 * --------------------- */
	for( run1 = 0; run1 < KKT_DIM * KKT_DIM; run1++ )
	{
		K [run1] = 0.0;
		Primal_Dual_VAR[run1] = 0.0;
	}
	
	/* ================================================================= */
	
	/* FIRST MATRIX MULTIPLICATION (OBTAINS THE INTERMEDIATE RESULT
	 *  K := [ ("ACTIVE" KKT-MATRIX OF THE QP)^(-1) * g_b_bA_VAR ]^T )
	 * THE EVALUATION OF THE INVERSE OF THE KKT-MATRIX OF THE QP
	 * WITH RESPECT TO THE CURRENT ACTIVE SET
	 * USES THE EXISTING CHOLESKY AND TQ-DECOMPOSITIONS. FOR DETAILS
	 * cf. THE (protected) FUNCTION determineStepDirection. */
	
	for( run3 = 0; run3 < KKT_DIM; run3++ )
	{
		
		for( run1 = 0; run1 < NVMAX; run1++ )
		{
			delta_g_cov [run1] = g_b_bA_VAR[run3*KKT_DIM+run1];
			delta_lb_cov [run1] = g_b_bA_VAR[run3*KKT_DIM+NVMAX+run1]; /*  LINE-WISE LOADING OF THE INPUT */
			delta_ub_cov [run1] = g_b_bA_VAR[run3*KKT_DIM+NVMAX+run1]; /*  VARIANCE-COVARIANCE            */
		}
		for( run1 = 0; run1 < NCMAX; run1++ )
		{
			delta_lbA_cov [run1] = g_b_bA_VAR[run3*KKT_DIM+2*NVMAX+run1];
			delta_ubA_cov [run1] = g_b_bA_VAR[run3*KKT_DIM+2*NVMAX+run1];
		}
		
		/* EVALUATION OF THE STEP:
		 * ------------------------------------------------------------------------------ */
		
		returnvalue = qp->hotstart_determineStepDirection(
			FR_idx, FX_idx, AC_idx,
			delta_g_cov, delta_lbA_cov, delta_ubA_cov, delta_lb_cov, delta_ub_cov,
			Delta_bC_isZero, Delta_bB_isZero, delta_xFX,delta_xFR,
			delta_yAC,delta_yFX );
		
		/* ------------------------------------------------------------------------------ */
		
		/* STOP THE ALGORITHM IN THE CASE OF NO SUCCESFUL RETURN:
		 * ------------------------------------------------------ */
		if ( returnvalue != SUCCESSFUL_RETURN )
		{
			return returnvalue;
		}
		
		/*  LINE WISE                  */
		/*  STORAGE OF THE QP-REACTION */
		/*  (uses the index list)      */
		
		for( run1=0; run1<nFR; run1++ )
		{
			run2 = FR_idx[run1];
			K[run3*KKT_DIM+run2] = delta_xFR[run1];
		} 
		for( run1=0; run1<nFX; run1++ )
		{ 
			run2 = FX_idx[run1]; 
			K[run3*KKT_DIM+run2] = delta_xFX[run1];
			K[run3*KKT_DIM+NVMAX+run2] = delta_yFX[run1];
		}
		for( run1=0; run1<nAC; run1++ )
		{
			run2 = AC_idx[run1];
			K[run3*KKT_DIM+2*NVMAX+run2] = delta_yAC[run1];
		}
	}
	
	/* ================================================================= */
	
	/* SECOND MATRIX MULTIPLICATION (OBTAINS THE FINAL RESULT
	 * Primal_Dual_VAR := ("ACTIVE" KKT-MATRIX OF THE QP)^(-1) * K )
	 * THE APPLICATION OF THE KKT-INVERSE IS AGAIN REALIZED
	 * BY USING THE PROTECTED FUNCTION
	 * determineStepDirection */
	
	for( run3 = 0; run3 < KKT_DIM; run3++ )
	{
		
		for( run1 = 0; run1 < NVMAX; run1++ )
		{
			delta_g_cov [run1] = K[run3+ run1*KKT_DIM];
			delta_lb_cov [run1] = K[run3+(NVMAX+run1)*KKT_DIM]; /*  ROW WISE LOADING OF THE */
			delta_ub_cov [run1] = K[run3+(NVMAX+run1)*KKT_DIM]; /*  INTERMEDIATE RESULT K   */
		}
		for( run1 = 0; run1 < NCMAX; run1++ )
		{
			delta_lbA_cov [run1] = K[run3+(2*NVMAX+run1)*KKT_DIM];
			delta_ubA_cov [run1] = K[run3+(2*NVMAX+run1)*KKT_DIM];
		}
		
		/* EVALUATION OF THE STEP:
		 * ------------------------------------------------------------------------------ */
		
		returnvalue = qp->hotstart_determineStepDirection(
			FR_idx, FX_idx, AC_idx,
			delta_g_cov, delta_lbA_cov, delta_ubA_cov, delta_lb_cov, delta_ub_cov,
			Delta_bC_isZero, Delta_bB_isZero, delta_xFX,delta_xFR,
			delta_yAC,delta_yFX );
		
		/* ------------------------------------------------------------------------------ */
		
		/* STOP THE ALGORITHM IN THE CASE OF NO SUCCESFUL RETURN:
		 * ------------------------------------------------------ */
		if ( returnvalue != SUCCESSFUL_RETURN )
		{
			return returnvalue;
		}
		
		/*  ROW-WISE STORAGE */
		/*  OF THE RESULT.   */
		
		for( run1=0; run1<nFR; run1++ )
		{
			run2 = FR_idx[run1];
			Primal_Dual_VAR[run3+run2*KKT_DIM] = delta_xFR[run1];
		}
		for( run1=0; run1<nFX; run1++ )
		{ 
			run2 = FX_idx[run1]; 
			Primal_Dual_VAR[run3+run2*KKT_DIM ] = delta_xFX[run1];
			Primal_Dual_VAR[run3+(NVMAX+run2)*KKT_DIM] = delta_yFX[run1];
		}
		for( run1=0; run1<nAC; run1++ )
		{
			run2 = AC_idx[run1];
			Primal_Dual_VAR[run3+(2*NVMAX+run2)*KKT_DIM] = delta_yAC[run1];
		}
	}
	
	return SUCCESSFUL_RETURN;
}

#endif
