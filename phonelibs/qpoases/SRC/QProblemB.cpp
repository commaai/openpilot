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
 *	\file SRC/QProblemB.cpp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Implementation of the QProblemB class which is able to use the newly
 *	developed online active set strategy for parametric quadratic programming.
 */


#include <QProblemB.hpp>

#include <stdio.h>

void printmatrix(char *name, double *A, int m, int n) {
  int i, j;

  printf("%s = [...\n", name);
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++)
        printf("  % 9.4f", A[i*n+j]);
    printf(",\n");
  }
  printf("];\n");
}



/*****************************************************************************
 *  P U B L I C                                                              *
 *****************************************************************************/


/*
 *	Q P r o b l e m B
 */
QProblemB::QProblemB( )
{
	/* reset global message handler */
	getGlobalMessageHandler( )->reset( );

	hasHessian = BT_FALSE;

	bounds.init( 0 );

	hasCholesky = BT_FALSE;

	tau = 0.0;

	hessianType = HST_POSDEF_NULLSPACE; /* Hessian is assumed to be positive definite by default */
	infeasible = BT_FALSE;
	unbounded = BT_FALSE;

	status = QPS_NOTINITIALISED;

	#ifdef PC_DEBUG
	printlevel = PL_MEDIUM;
	setPrintLevel( PL_MEDIUM );
	#else
	printlevel = QPOASES_PRINTLEVEL;
	#endif

	count = 0;
}


/*
 *	Q P r o b l e m B
 */
QProblemB::QProblemB( int _nV )
{
	/* consistency check */
	if ( _nV <= 0 )
	{
		_nV = 1;
		THROWERROR( RET_INVALID_ARGUMENTS );
	}

	hasHessian = BT_FALSE;

	/* reset global message handler */
	getGlobalMessageHandler( )->reset( );

	bounds.init( _nV );

	hasCholesky = BT_FALSE;

	tau = 0.0;

	hessianType = HST_POSDEF_NULLSPACE; /* Hessian is assumed to be positive definite by default */
	infeasible = BT_FALSE;
	unbounded = BT_FALSE;

	status = QPS_NOTINITIALISED;

	#ifdef PC_DEBUG
	printlevel = PL_MEDIUM;
	setPrintLevel( PL_MEDIUM );
	#else
	printlevel = QPOASES_PRINTLEVEL;
	#endif

	count = 0;
}


/*
 *	Q P r o b l e m B
 */
QProblemB::QProblemB( const QProblemB& rhs )
{
	int i, j;

	int _nV = rhs.bounds.getNV( );

	for( i=0; i<_nV; ++i )
		for( j=0; j<_nV; ++j )
			H[i*NVMAX + j] = rhs.H[i*NVMAX + j];

	hasHessian = rhs.hasHessian;

	for( i=0; i<_nV; ++i )
		g[i] = rhs.g[i];

	for( i=0; i<_nV; ++i )
		lb[i] = rhs.lb[i];

	for( i=0; i<_nV; ++i )
		ub[i] = rhs.ub[i];


	bounds = rhs.bounds;

	for( i=0; i<_nV; ++i )
		for( j=0; j<_nV; ++j )
			R[i*NVMAX + j] = rhs.R[i*NVMAX + j];
	hasCholesky = rhs.hasCholesky;

	for( i=0; i<_nV; ++i )
		x[i] = rhs.x[i];

	for( i=0; i<_nV; ++i )
		y[i] = rhs.y[i];

	tau = rhs.tau;

	hessianType = rhs.hessianType;
	infeasible = rhs.infeasible;
	unbounded = rhs.unbounded;

	status = rhs.status;

	printlevel = rhs.printlevel;

	count = rhs.count;
}


/*
 *	~ Q P r o b l e m B
 */
QProblemB::~QProblemB( )
{
}


/*
 *	o p e r a t o r =
 */
QProblemB& QProblemB::operator=( const QProblemB& rhs )
{
	int i, j;

	if ( this != &rhs )
	{
		int _nV = rhs.bounds.getNV( );

		for( i=0; i<_nV; ++i )
			for( j=0; j<_nV; ++j )
				H[i*NVMAX + j] = rhs.H[i*NVMAX + j];

		hasHessian = rhs.hasHessian;

		for( i=0; i<_nV; ++i )
			g[i] = rhs.g[i];

		for( i=0; i<_nV; ++i )
			lb[i] = rhs.lb[i];

		for( i=0; i<_nV; ++i )
			ub[i] = rhs.ub[i];

		bounds = rhs.bounds;

		for( i=0; i<_nV; ++i )
			for( j=0; j<_nV; ++j )
				R[i*NVMAX + j] = rhs.R[i*NVMAX + j];
		hasCholesky = rhs.hasCholesky;


		for( i=0; i<_nV; ++i )
			x[i] = rhs.x[i];

		for( i=0; i<_nV; ++i )
			y[i] = rhs.y[i];

		tau = rhs.tau;

		hessianType = rhs.hessianType;
		infeasible = rhs.infeasible;
		unbounded = rhs.unbounded;

		status = rhs.status;

		printlevel = rhs.printlevel;
		setPrintLevel( rhs.printlevel );

		count = rhs.count;
	}

	return *this;
}


/*
 *	r e s e t
 */
returnValue QProblemB::reset( )
{
	int i, j;
	int nV = getNV( );

	/** 0) Reset has Hessian flag. */
	hasHessian = BT_FALSE;

	/* 1) Reset bounds. */
	bounds.init( nV );

	/* 2) Reset Cholesky decomposition. */
	for( i=0; i<nV; ++i )
		for( j=0; j<nV; ++j )
			R[i*NVMAX + j] = 0.0;
	hasCholesky = BT_FALSE;

	/* 3) Reset steplength and status flags. */
	tau = 0.0;

	hessianType = HST_POSDEF_NULLSPACE; /* Hessian is assumed to be positive definite by default */
	infeasible = BT_FALSE;
	unbounded = BT_FALSE;

	status = QPS_NOTINITIALISED;

	return SUCCESSFUL_RETURN;
}


/*
 *	i n i t
 */
returnValue QProblemB::init(	const real_t* const _H, const real_t* const _g,
								const real_t* const _lb, const real_t* const _ub,
								int& nWSR, const real_t* const yOpt, real_t* const cputime
								)
{
	/* 1) Setup QP data. */
	if (setupQPdata(_H, 0, _g, _lb, _ub) != SUCCESSFUL_RETURN)
		return THROWERROR( RET_INVALID_ARGUMENTS );

	/* 2) Call to main initialisation routine (without any additional information). */
	return solveInitialQP(0, yOpt, 0, nWSR, cputime);
}

returnValue QProblemB::init(	const real_t* const _H, const real_t* const _R, const real_t* const _g,
								const real_t* const _lb, const real_t* const _ub,
								int& nWSR, const real_t* const yOpt, real_t* const cputime
								)
{
	/* 1) Setup QP data. */
	if (setupQPdata(_H, _R, _g, _lb, _ub) != SUCCESSFUL_RETURN)
		return THROWERROR( RET_INVALID_ARGUMENTS );

	/* 2) Call to main initialisation routine (without any additional information). */
	return solveInitialQP(0, yOpt, 0, nWSR, cputime);
}


/*
 *	h o t s t a r t
 */
returnValue QProblemB::hotstart(	const real_t* const g_new, const real_t* const lb_new, const real_t* const ub_new,
									int& nWSR, real_t* const cputime
									)
{
	int l;

	/* consistency check */
	if ( ( getStatus( ) == QPS_NOTINITIALISED )       ||
		 ( getStatus( ) == QPS_PREPARINGAUXILIARYQP ) ||
		 ( getStatus( ) == QPS_PERFORMINGHOMOTOPY )   )
	{
		return THROWERROR( RET_HOTSTART_FAILED_AS_QP_NOT_INITIALISED );
	}

	/* start runtime measurement */
	real_t starttime = 0.0;
	if ( cputime != 0 )
		starttime = getCPUtime( );


	/* I) PREPARATIONS */
	/* 1) Reset status flags and increase QP counter. */
	infeasible = BT_FALSE;
	unbounded = BT_FALSE;

	++count;

	/* 2) Allocate delta vectors of gradient and bounds. */
	returnValue returnvalue;
	BooleanType Delta_bB_isZero;

	int FR_idx[NVMAX];
	int FX_idx[NVMAX];

	real_t delta_g[NVMAX];
	real_t delta_lb[NVMAX];
	real_t delta_ub[NVMAX];

	real_t delta_xFR[NVMAX];
	real_t delta_xFX[NVMAX];
	real_t delta_yFX[NVMAX];

	int BC_idx;
	SubjectToStatus BC_status;

	#ifdef PC_DEBUG
	char messageString[80];
	#endif

	/* II) MAIN HOMOTOPY LOOP */
	for( l=0; l<nWSR; ++l )
	{
		status = QPS_PERFORMINGHOMOTOPY;

		if ( printlevel == PL_HIGH )
		{
			#ifdef PC_DEBUG
			sprintf( messageString,"%d ...",l );
			getGlobalMessageHandler( )->throwInfo( RET_ITERATION_STARTED,messageString,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE );
			#endif
		}

		/* 1) Setup index arrays. */
		if ( bounds.getFree( )->getNumberArray( FR_idx ) != SUCCESSFUL_RETURN )
			return THROWERROR( RET_HOTSTART_FAILED );

		if ( bounds.getFixed( )->getNumberArray( FX_idx ) != SUCCESSFUL_RETURN )
			return THROWERROR( RET_HOTSTART_FAILED );

		/* 2) Initialize shift direction of the gradient and the bounds. */
		returnvalue = hotstart_determineDataShift(  FX_idx,
													g_new,lb_new,ub_new,
													delta_g,delta_lb,delta_ub,
													Delta_bB_isZero
													);
		if ( returnvalue != SUCCESSFUL_RETURN )
		{
			nWSR = l;
			THROWERROR( RET_SHIFT_DETERMINATION_FAILED );
			return returnvalue;
		}

		/* 3) Determination of step direction of X and Y. */
		returnvalue = hotstart_determineStepDirection(	FR_idx,FX_idx,
														delta_g,delta_lb,delta_ub,
														Delta_bB_isZero,
														delta_xFX,delta_xFR,delta_yFX
														);
		if ( returnvalue != SUCCESSFUL_RETURN )
		{
			nWSR = l;
			THROWERROR( RET_STEPDIRECTION_DETERMINATION_FAILED );
			return returnvalue;
		}


		/* 4) Determination of step length TAU. */
		returnvalue = hotstart_determineStepLength(	FR_idx,FX_idx,
													delta_lb,delta_ub,
													delta_xFR,delta_yFX,
													BC_idx,BC_status );
		if ( returnvalue != SUCCESSFUL_RETURN )
		{
			nWSR = l;
			THROWERROR( RET_STEPLENGTH_DETERMINATION_FAILED );
			return returnvalue;
		}

		/* 5) Realization of the homotopy step. */
		returnvalue = hotstart_performStep(	FR_idx,FX_idx,
											delta_g,delta_lb,delta_ub,
											delta_xFX,delta_xFR,delta_yFX,
											BC_idx,BC_status
											);


		if ( returnvalue != SUCCESSFUL_RETURN )
		{
			nWSR = l;

			/* stop runtime measurement */
			if ( cputime != 0 )
				*cputime = getCPUtime( ) - starttime;

			/* optimal solution found? */
			if ( returnvalue == RET_OPTIMAL_SOLUTION_FOUND )
			{
				status = QPS_SOLVED;

				if ( printlevel == PL_HIGH )
					THROWINFO( RET_OPTIMAL_SOLUTION_FOUND );

				#ifdef PC_DEBUG
	 			if ( printIteration( l,BC_idx,BC_status ) != SUCCESSFUL_RETURN )
					THROWERROR( RET_PRINT_ITERATION_FAILED ); /* do not pass this as return value! */
				#endif

				/* check KKT optimality conditions */
				return checkKKTconditions( );
			}
			else
			{
				/* checks for infeasibility... */
				if ( infeasible == BT_TRUE )
				{
					status = QPS_HOMOTOPYQPSOLVED;
					return THROWERROR( RET_HOTSTART_STOPPED_INFEASIBILITY );
				}

				/* ...unboundedness... */
				if ( unbounded == BT_TRUE ) /* not necessary since objective function convex! */
					return THROWERROR( RET_HOTSTART_STOPPED_UNBOUNDEDNESS );

				/* ... and throw unspecific error otherwise */
				THROWERROR( RET_HOMOTOPY_STEP_FAILED );
				return returnvalue;
			}
		}

		/* 6) Output information of successful QP iteration. */
		status = QPS_HOMOTOPYQPSOLVED;

		#ifdef PC_DEBUG
		if ( printIteration( l,BC_idx,BC_status ) != SUCCESSFUL_RETURN )
			THROWERROR( RET_PRINT_ITERATION_FAILED ); /* do not pass this as return value! */
		#endif
	}


	/* stop runtime measurement */
	if ( cputime != 0 )
		*cputime = getCPUtime( ) - starttime;


	/* if programm gets to here, output information that QP could not be solved
	 * within the given maximum numbers of working set changes */
	if ( printlevel == PL_HIGH )
	{
		#ifdef PC_DEBUG
		sprintf( messageString,"(nWSR = %d)",nWSR );
		return getGlobalMessageHandler( )->throwWarning( RET_MAX_NWSR_REACHED,messageString,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE );
		#endif
	}

	/* Finally check KKT optimality conditions. */
	returnValue returnvalueKKTcheck = checkKKTconditions( );

	if ( returnvalueKKTcheck != SUCCESSFUL_RETURN )
		return returnvalueKKTcheck;
	else
		return RET_MAX_NWSR_REACHED;
}


/*
 *	g e t N Z
 */
int QProblemB::getNZ( )
{
	/* if no constraints are present: nZ=nFR */
	return bounds.getFree( )->getLength( );
}


/*
 *	g e t O b j V a l
 */
real_t QProblemB::getObjVal( ) const
{
	real_t objVal;

	/* calculated optimal objective function value
	 * only if current QP has been solved */
	if ( ( getStatus( ) == QPS_AUXILIARYQPSOLVED ) ||
		 ( getStatus( ) == QPS_HOMOTOPYQPSOLVED )  ||
		 ( getStatus( ) == QPS_SOLVED ) )
	{
		objVal = getObjVal( x );
	}
	else
	{
		objVal = INFTY;
	}

	return objVal;
}


/*
 *	g e t O b j V a l
 */
real_t QProblemB::getObjVal( const real_t* const _x ) const
{
	int i, j;
	int nV = getNV( );

	real_t obj_tmp = 0.0;

	for( i=0; i<nV; ++i )
	{
		obj_tmp += _x[i]*g[i];

		for( j=0; j<nV; ++j )
			obj_tmp += 0.5*_x[i]*H[i*NVMAX + j]*_x[j];
	}

	return obj_tmp;
}


/*
 *	g e t P r i m a l S o l u t i o n
 */
returnValue QProblemB::getPrimalSolution( real_t* const xOpt ) const
{
	int i;

	/* return optimal primal solution vector
	 * only if current QP has been solved */
	if ( ( getStatus( ) == QPS_AUXILIARYQPSOLVED ) ||
		 ( getStatus( ) == QPS_HOMOTOPYQPSOLVED )  ||
		 ( getStatus( ) == QPS_SOLVED ) )
	{
		for( i=0; i<getNV( ); ++i )
			xOpt[i] = x[i];

		return SUCCESSFUL_RETURN;
	}
	else
	{
		return RET_QP_NOT_SOLVED;
	}
}


/*
 *	g e t D u a l S o l u t i o n
 */
returnValue QProblemB::getDualSolution( real_t* const yOpt ) const
{
	int i;

	/* return optimal dual solution vector
	 * only if current QP has been solved */
	if ( ( getStatus( ) == QPS_AUXILIARYQPSOLVED ) ||
		 ( getStatus( ) == QPS_HOMOTOPYQPSOLVED )  ||
		 ( getStatus( ) == QPS_SOLVED ) )
	{
		for( i=0; i<getNV( ); ++i )
			yOpt[i] = y[i];

		return SUCCESSFUL_RETURN;
	}
	else
	{
		return RET_QP_NOT_SOLVED;
	}
}


/*
 *	s e t P r i n t L e v e l
 */
returnValue QProblemB::setPrintLevel( PrintLevel _printlevel )
{
	#ifndef __MATLAB__
	if ( ( printlevel >= PL_MEDIUM ) && ( printlevel != _printlevel ) )
		THROWINFO( RET_PRINTLEVEL_CHANGED );
	#endif

	printlevel = _printlevel;

	/* update message handler preferences */
 	switch ( printlevel )
 	{
 		case PL_NONE:
 			getGlobalMessageHandler( )->setErrorVisibilityStatus( VS_HIDDEN );
			getGlobalMessageHandler( )->setWarningVisibilityStatus( VS_HIDDEN );
			getGlobalMessageHandler( )->setInfoVisibilityStatus( VS_HIDDEN );
			break;

		case PL_LOW:
 			getGlobalMessageHandler( )->setErrorVisibilityStatus( VS_VISIBLE );
			getGlobalMessageHandler( )->setWarningVisibilityStatus( VS_HIDDEN );
			getGlobalMessageHandler( )->setInfoVisibilityStatus( VS_HIDDEN );
			break;

 		default: /* PL_MEDIUM, PL_HIGH */
 			getGlobalMessageHandler( )->setErrorVisibilityStatus( VS_VISIBLE );
			getGlobalMessageHandler( )->setWarningVisibilityStatus( VS_VISIBLE );
			getGlobalMessageHandler( )->setInfoVisibilityStatus( VS_VISIBLE );
			break;
 	}

	return SUCCESSFUL_RETURN;
}



/*****************************************************************************
 *  P R O T E C T E D                                                        *
 *****************************************************************************/

/*
 *	c h e c k F o r I d e n t i t y H e s s i a n
 */
returnValue QProblemB::checkForIdentityHessian( )
{
	int i, j;
	int nV = getNV( );

	/* nothing to do as status flag remains unaltered
	 * if Hessian differs from identity matrix */
	if ( hessianType == HST_IDENTITY )
		return SUCCESSFUL_RETURN;

	/* 1) If Hessian differs from identity matrix,
	 *    return without changing the internal HessianType. */
	for ( i=0; i<nV; ++i )
		if ( getAbs( H[i*NVMAX + i] - 1.0 ) > EPS )
			return SUCCESSFUL_RETURN;

	for ( i=0; i<nV; ++i )
	{
		for ( j=0; j<i; ++j )
			if ( ( getAbs( H[i*NVMAX + j] ) > EPS ) || ( getAbs( H[j*NVMAX + i] ) > EPS ) )
				return SUCCESSFUL_RETURN;
	}

	/* 2) If this point is reached, Hessian equals the idetity matrix. */
	hessianType = HST_IDENTITY;

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t u p S u b j e c t T o T y p e
 */
returnValue QProblemB::setupSubjectToType( )
{
	int i;
	int nV = getNV( );


	/* 1) Check if lower bounds are present. */
	bounds.setNoLower( BT_TRUE );
	for( i=0; i<nV; ++i )
		if ( lb[i] > -INFTY )
		{
			bounds.setNoLower( BT_FALSE );
			break;
		}

	/* 2) Check if upper bounds are present. */
	bounds.setNoUpper( BT_TRUE );
	for( i=0; i<nV; ++i )
		if ( ub[i] < INFTY )
		{
			bounds.setNoUpper( BT_FALSE );
			break;
		}

	/* 3) Determine implicitly fixed and unbounded variables. */
	int nFV = 0;
	int nUV = 0;

	for( i=0; i<nV; ++i )
		if ( ( lb[i] < -INFTY + BOUNDTOL ) && ( ub[i] > INFTY - BOUNDTOL ) )
		{
			bounds.setType( i,ST_UNBOUNDED );
			++nUV;
		}
		else
		{
			if ( lb[i] > ub[i] - BOUNDTOL )
			{
				bounds.setType( i,ST_EQUALITY );
				++nFV;
			}
			else
			{
				bounds.setType( i,ST_BOUNDED );
			}
		}

	/* 4) Set dimensions of bounds structure. */
	bounds.setNFV( nFV );
	bounds.setNUV( nUV );
	bounds.setNBV( nV - nFV - nUV );

	return SUCCESSFUL_RETURN;
}


/*
 *	c h o l e s k y D e c o m p o s i t i o n
 */
returnValue QProblemB::setupCholeskyDecomposition( )
{
	int i, j, k, ii, jj;
	int nV  = getNV( );
	int nFR = getNFR( );

	/* If Hessian flag is false, it means that H & R already contain Cholesky
	 * factorization -- provided from outside. */
	if (hasHessian == BT_FALSE)
		return SUCCESSFUL_RETURN;

	/* 1) Initialises R with all zeros. */
	for( i=0; i<nV; ++i )
		for( j=0; j<nV; ++j )
			R[i*NVMAX + j] = 0.0;

	/* 2) Calculate Cholesky decomposition of H (projected to free variables). */
	if ( hessianType == HST_IDENTITY )
	{
		/* if Hessian is identity, so is its Cholesky factor. */
		for( i=0; i<nFR; ++i )
			R[i*NVMAX + i] = 1.0;
	}
	else
	{
		if ( nFR > 0 )
		{
			int FR_idx[NVMAX];
			if ( bounds.getFree( )->getNumberArray( FR_idx ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_INDEXLIST_CORRUPTED );

			/* R'*R = H */
			real_t sum;
			real_t inv;

			for( i=0; i<nFR; ++i )
			{
				/* j == i */
				ii = FR_idx[i];
				sum = H[ii*NVMAX + ii];

				for( k=(i-1); k>=0; --k )
					sum -= R[k*NVMAX + i] * R[k*NVMAX + i];

				if ( sum > 0.0 )
				{
					R[i*NVMAX + i] = sqrt( sum );
					inv = 1.0 / R[i*NVMAX + i];
				}
				else
				{
					hessianType = HST_SEMIDEF;
					return THROWERROR( RET_HESSIAN_NOT_SPD );
				}

				/* j > i */
				for( j=(i+1); j<nFR; ++j )
				{
					jj = FR_idx[j];
					sum = H[jj*NVMAX + ii];

					for( k=(i-1); k>=0; --k )
						sum -= R[k*NVMAX + i] * R[k*NVMAX + j];

					R[i*NVMAX + j] = sum * inv;
				}
			}
		}
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s o l v e I n i t i a l Q P
 */
returnValue QProblemB::solveInitialQP(	const real_t* const xOpt, const real_t* const yOpt,
										const Bounds* const guessedBounds,
										int& nWSR, real_t* const cputime
										)
{
	int i, nFR;
	int nV = getNV( );


	/* start runtime measurement */
	real_t starttime = 0.0;
	if ( cputime != 0 )
		starttime = getCPUtime( );


	status = QPS_NOTINITIALISED;

	/* I) ANALYSE QP DATA: */
	/* 1) Check if Hessian happens to be the identity matrix. */
	if ( checkForIdentityHessian( ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INIT_FAILED );

	/* 2) Setup type of bounds (i.e. unbounded, implicitly fixed etc.). */
	if ( setupSubjectToType( ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INIT_FAILED );

	status = QPS_PREPARINGAUXILIARYQP;


	/* II) SETUP AUXILIARY QP WITH GIVEN OPTIMAL SOLUTION: */
	/* 1) Setup bounds data structure. */
	if ( bounds.setupAllFree( ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INIT_FAILED );

	/* 2) Setup optimal primal/dual solution for auxiliary QP. */
	if ( setupAuxiliaryQPsolution( xOpt,yOpt ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INIT_FAILED );

	/* 3) Obtain linear independent working set for auxiliary QP. */

	static Bounds auxiliaryBounds;

	auxiliaryBounds.init( nV );

	if ( obtainAuxiliaryWorkingSet( xOpt,yOpt,guessedBounds, &auxiliaryBounds ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INIT_FAILED );

	/* 4) Setup working set of auxiliary QP and setup cholesky decomposition. */
	if ( setupAuxiliaryWorkingSet( &auxiliaryBounds,BT_TRUE ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INIT_FAILED );

	nFR = getNFR();
	/* At the moment we can only provide a Cholesky of the Hessian if
	 * the solver is cold-started. */
	if (hasCholesky == BT_FALSE || nFR != nV)
		if (setupCholeskyDecomposition() != SUCCESSFUL_RETURN)
			return THROWERROR( RET_INIT_FAILED_CHOLESKY );

	/* 5) Store original QP formulation... */
	real_t g_original[NVMAX];
	real_t lb_original[NVMAX];
	real_t ub_original[NVMAX];

	for( i=0; i<nV; ++i )
		g_original[i] = g[i];
	for( i=0; i<nV; ++i )
		lb_original[i] = lb[i];
	for( i=0; i<nV; ++i )
		ub_original[i] = ub[i];

	/* ... and setup QP data of an auxiliary QP having an optimal solution
	 * as specified by the user (or xOpt = yOpt = 0, by default). */
	if ( setupAuxiliaryQPgradient( ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INIT_FAILED );

	if ( setupAuxiliaryQPbounds( BT_TRUE ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INIT_FAILED );

	status = QPS_AUXILIARYQPSOLVED;


	/* III) SOLVE ACTUAL INITIAL QP: */
	/* Use hotstart method to find the solution of the original initial QP,... */
	returnValue returnvalue = hotstart( g_original,lb_original,ub_original, nWSR,0 );


	/* ... check for infeasibility and unboundedness... */
	if ( isInfeasible( ) == BT_TRUE )
		return THROWERROR( RET_INIT_FAILED_INFEASIBILITY );

	if ( isUnbounded( ) == BT_TRUE )
		return THROWERROR( RET_INIT_FAILED_UNBOUNDEDNESS );

	/* ... and internal errors. */
	if ( ( returnvalue != SUCCESSFUL_RETURN ) && ( returnvalue != RET_MAX_NWSR_REACHED )  &&
	     ( returnvalue != RET_INACCURATE_SOLUTION ) && ( returnvalue != RET_NO_SOLUTION ) )
		return THROWERROR( RET_INIT_FAILED_HOTSTART );


	/* stop runtime measurement */
	if ( cputime != 0 )
		*cputime = getCPUtime( ) - starttime;

	if ( printlevel == PL_HIGH )
		THROWINFO( RET_INIT_SUCCESSFUL );

	return returnvalue;
}


/*
 *	o b t a i n A u x i l i a r y W o r k i n g S e t
 */
returnValue QProblemB::obtainAuxiliaryWorkingSet(	const real_t* const xOpt, const real_t* const yOpt,
													const Bounds* const guessedBounds, Bounds* auxiliaryBounds
													) const
{
	int i = 0;
	int nV = getNV( );


	/* 1) Ensure that desiredBounds is allocated (and different from guessedBounds). */
	if ( ( auxiliaryBounds == 0 ) || ( auxiliaryBounds == guessedBounds ) )
		return THROWERROR( RET_INVALID_ARGUMENTS );


	/* 2) Setup working set for auxiliary initial QP. */
	if ( guessedBounds != 0 )
	{
		/* If an initial working set is specific, use it!
		 * Moreover, add all implictly fixed variables if specified. */
		for( i=0; i<nV; ++i )
		{
			if ( bounds.getType( i ) == ST_EQUALITY )
			{
				if ( auxiliaryBounds->setupBound( i,ST_LOWER ) != SUCCESSFUL_RETURN )
					return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
			}
			else
			{
				if ( auxiliaryBounds->setupBound( i,guessedBounds->getStatus( i ) ) != SUCCESSFUL_RETURN )
					return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
			}
		}
	}
	else	/* No initial working set specified. */
	{
		if ( ( xOpt != 0 ) && ( yOpt == 0 ) )
		{
			/* Obtain initial working set by "clipping". */
			for( i=0; i<nV; ++i )
			{
				if ( xOpt[i] <= lb[i] + BOUNDTOL )
				{
					if ( auxiliaryBounds->setupBound( i,ST_LOWER ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
					continue;
				}

				if ( xOpt[i] >= ub[i] - BOUNDTOL )
				{
					if ( auxiliaryBounds->setupBound( i,ST_UPPER ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
					continue;
				}

				/* Moreover, add all implictly fixed variables if specified. */
				if ( bounds.getType( i ) == ST_EQUALITY )
				{
					if ( auxiliaryBounds->setupBound( i,ST_LOWER ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
				}
				else
				{
					if ( auxiliaryBounds->setupBound( i,ST_INACTIVE ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
				}
			}
		}

		if ( ( xOpt == 0 ) && ( yOpt != 0 ) )
		{
			/* Obtain initial working set in accordance to sign of dual solution vector. */
			for( i=0; i<nV; ++i )
			{
				if ( yOpt[i] > ZERO )
				{
					if ( auxiliaryBounds->setupBound( i,ST_LOWER ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
					continue;
				}

				if ( yOpt[i] < -ZERO )
				{
					if ( auxiliaryBounds->setupBound( i,ST_UPPER ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
					continue;
				}

				/* Moreover, add all implictly fixed variables if specified. */
				if ( bounds.getType( i ) == ST_EQUALITY )
				{
					if ( auxiliaryBounds->setupBound( i,ST_LOWER ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
				}
				else
				{
					if ( auxiliaryBounds->setupBound( i,ST_INACTIVE ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
				}
			}
		}

		/* If xOpt and yOpt are null pointer and no initial working is specified,
		 * start with empty working set (or implicitly fixed bounds only)
		 * for auxiliary QP. */
		if ( ( xOpt == 0 ) && ( yOpt == 0 ) )
		{
			for( i=0; i<nV; ++i )
			{
				/* Only add all implictly fixed variables if specified. */
				if ( bounds.getType( i ) == ST_EQUALITY )
				{
					if ( auxiliaryBounds->setupBound( i,ST_LOWER ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
				}
				else
				{
					if ( auxiliaryBounds->setupBound( i,ST_INACTIVE ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
				}
			}
		}
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t u p A u x i l i a r y W o r k i n g S e t
 */
returnValue QProblemB::setupAuxiliaryWorkingSet( 	const Bounds* const auxiliaryBounds,
													BooleanType setupAfresh
													)
{
	int i;
	int nV = getNV( );

	/* consistency checks */
	if ( auxiliaryBounds != 0 )
	{
		for( i=0; i<nV; ++i )
			if ( ( bounds.getStatus( i ) == ST_UNDEFINED ) || ( auxiliaryBounds->getStatus( i ) == ST_UNDEFINED ) )
				return THROWERROR( RET_UNKNOWN_BUG );
	}
	else
	{
		return THROWERROR( RET_INVALID_ARGUMENTS );
	}


	/* I) SETUP CHOLESKY FLAG:
	 *    Cholesky decomposition shall only be updated if working set
	 *    shall be updated (i.e. NOT setup afresh!) */
	BooleanType updateCholesky;
	if ( setupAfresh == BT_TRUE )
		updateCholesky = BT_FALSE;
	else
		updateCholesky = BT_TRUE;


	/* II) REMOVE FORMERLY ACTIVE BOUNDS (IF NECESSARY): */
	if ( setupAfresh == BT_FALSE )
	{
		/* Remove all active bounds that shall be inactive AND
		*  all active bounds that are active at the wrong bound. */
		for( i=0; i<nV; ++i )
		{
			if ( ( bounds.getStatus( i ) == ST_LOWER ) && ( auxiliaryBounds->getStatus( i ) != ST_LOWER ) )
				if ( removeBound( i,updateCholesky ) != SUCCESSFUL_RETURN )
					return THROWERROR( RET_SETUP_WORKINGSET_FAILED );

			if ( ( bounds.getStatus( i ) == ST_UPPER ) && ( auxiliaryBounds->getStatus( i ) != ST_UPPER ) )
				if ( removeBound( i,updateCholesky ) != SUCCESSFUL_RETURN )
					return THROWERROR( RET_SETUP_WORKINGSET_FAILED );
		}
	}


	/* III) ADD NEWLY ACTIVE BOUNDS: */
	/*      Add all inactive bounds that shall be active AND
	 *      all formerly active bounds that have been active at the wrong bound. */
	for( i=0; i<nV; ++i )
	{
		if ( ( bounds.getStatus( i ) == ST_INACTIVE ) && ( auxiliaryBounds->getStatus( i ) != ST_INACTIVE ) )
		{
			if ( addBound( i,auxiliaryBounds->getStatus( i ),updateCholesky ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_SETUP_WORKINGSET_FAILED );
		}
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t u p A u x i l i a r y Q P s o l u t i o n
 */
returnValue QProblemB::setupAuxiliaryQPsolution(	const real_t* const xOpt, const real_t* const yOpt
													)
{
	int i;
	int nV = getNV( );


	/* Setup primal/dual solution vectors for auxiliary initial QP:
	 * if a null pointer is passed, a zero vector is assigned;
	 * old solution vector is kept if pointer to internal solution vector is passed. */
	if ( xOpt != 0 )
	{
		if ( xOpt != x )
			for( i=0; i<nV; ++i )
				x[i] = xOpt[i];
	}
	else
	{
		for( i=0; i<nV; ++i )
			x[i] = 0.0;
	}

	if ( yOpt != 0 )
	{
		if ( yOpt != y )
			for( i=0; i<nV; ++i )
				y[i] = yOpt[i];
	}
	else
	{
		for( i=0; i<nV; ++i )
			y[i] = 0.0;
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t u p A u x i l i a r y Q P g r a d i e n t
 */
returnValue QProblemB::setupAuxiliaryQPgradient( )
{
	int i, j;
	int nV = getNV( );


	/* Setup gradient vector: g = -H*x + y'*Id. */
	for ( i=0; i<nV; ++i )
	{
		/* y'*Id */
		g[i] = y[i];

		/* -H*x */
		for ( j=0; j<nV; ++j )
			g[i] -= H[i*NVMAX + j] * x[j];
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t u p A u x i l i a r y Q P b o u n d s
 */
returnValue QProblemB::setupAuxiliaryQPbounds( BooleanType useRelaxation )
{
	int i;
	int nV = getNV( );


	/* Setup bound vectors. */
	for ( i=0; i<nV; ++i )
	{
		switch ( bounds.getStatus( i ) )
		{
			case ST_INACTIVE:
				if ( useRelaxation == BT_TRUE )
				{
					if ( bounds.getType( i ) == ST_EQUALITY )
					{
						lb[i] = x[i];
						ub[i] = x[i];
					}
					else
					{
						lb[i] = x[i] - BOUNDRELAXATION;
						ub[i] = x[i] + BOUNDRELAXATION;
					}
				}
				break;

			case ST_LOWER:
				lb[i] = x[i];
				if ( bounds.getType( i ) == ST_EQUALITY )
				{
					ub[i] = x[i];
				}
				else
				{
					if ( useRelaxation == BT_TRUE )
						ub[i] = x[i] + BOUNDRELAXATION;
				}
				break;

			case ST_UPPER:
				ub[i] = x[i];
				if ( bounds.getType( i ) == ST_EQUALITY )
				{
					lb[i] = x[i];
				}
				else
				{
					if ( useRelaxation == BT_TRUE )
						lb[i] = x[i] - BOUNDRELAXATION;
				}
				break;

			default:
				return THROWERROR( RET_UNKNOWN_BUG );
		}
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	a d d B o u n d
 */
returnValue QProblemB::addBound(	int number, SubjectToStatus B_status,
									BooleanType updateCholesky
									)
{
	int i, j;
	int nFR = getNFR( );


	/* consistency check */
	if ( ( getStatus( ) == QPS_NOTINITIALISED )    ||
		 ( getStatus( ) == QPS_AUXILIARYQPSOLVED ) ||
		 ( getStatus( ) == QPS_HOMOTOPYQPSOLVED )  ||
		 ( getStatus( ) == QPS_SOLVED )            )
	{
		return THROWERROR( RET_UNKNOWN_BUG );
	}

	/* Perform cholesky updates only if QProblemB has been initialised! */
	if ( ( getStatus( ) == QPS_PREPARINGAUXILIARYQP ) || ( updateCholesky == BT_FALSE ) )
	{
		/* UPDATE INDICES */
		if ( bounds.moveFreeToFixed( number,B_status ) != SUCCESSFUL_RETURN )
			return THROWERROR( RET_ADDBOUND_FAILED );

		return SUCCESSFUL_RETURN;
	}


	/* I) PERFORM CHOLESKY UPDATE: */
	/* 1) Index of variable to be added within the list of free variables. */
	int number_idx = bounds.getFree( )->getIndex( number );

	real_t c, s;

	/* 2) Use row-wise Givens rotations to restore upper triangular form of R. */
	for( i=number_idx+1; i<nFR; ++i )
	{
		computeGivens( R[(i-1)*NVMAX + i],R[i*NVMAX + i], R[(i-1)*NVMAX + i],R[i*NVMAX + i],c,s );

		for( j=(1+i); j<nFR; ++j ) /* last column of R is thrown away */
			applyGivens( c,s,R[(i-1)*NVMAX + j],R[i*NVMAX + j], R[(i-1)*NVMAX + j],R[i*NVMAX + j] );
	}

	/* 3) Delete <number_idx>th column and ... */
	for( i=0; i<nFR-1; ++i )
		for( j=number_idx+1; j<nFR; ++j )
			R[i*NVMAX + j-1] = R[i*NVMAX + j];
	/* ... last column of R. */
	for( i=0; i<nFR; ++i )
		R[i*NVMAX + nFR-1] = 0.0;


	/* II) UPDATE INDICES */
	if ( bounds.moveFreeToFixed( number,B_status ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_ADDBOUND_FAILED );


	return SUCCESSFUL_RETURN;
}


returnValue QProblemB::removeBound(	int number,
									BooleanType updateCholesky
									)
{
	int i, ii;
	int nFR = getNFR( );


	/* consistency check */
	if ( ( getStatus( ) == QPS_NOTINITIALISED )    ||
		 ( getStatus( ) == QPS_AUXILIARYQPSOLVED ) ||
		 ( getStatus( ) == QPS_HOMOTOPYQPSOLVED )  ||
		 ( getStatus( ) == QPS_SOLVED )            )
	{
		return THROWERROR( RET_UNKNOWN_BUG );
	}


	/* I) UPDATE INDICES */
	if ( bounds.moveFixedToFree( number ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_REMOVEBOUND_FAILED );

	/* Perform cholesky updates only if QProblemB has been initialised! */
	if ( ( getStatus( ) == QPS_PREPARINGAUXILIARYQP ) || ( updateCholesky == BT_FALSE ) )
		return SUCCESSFUL_RETURN;


	/* II) PERFORM CHOLESKY UPDATE */
	int FR_idx[NVMAX];
	if ( bounds.getFree( )->getNumberArray( FR_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_REMOVEBOUND_FAILED );

	/* 1) Calculate new column of cholesky decomposition. */
	real_t rhs[NVMAX];
	real_t r[NVMAX];
	real_t r0 = H[number*NVMAX + number];

	for( i=0; i<nFR; ++i )
	{
		ii = FR_idx[i];
		rhs[i] = H[number*NVMAX + ii];
	}

	if ( backsolveR( rhs,BT_TRUE,BT_TRUE,r ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_REMOVEBOUND_FAILED );

	for( i=0; i<nFR; ++i )
		r0 -= r[i]*r[i];

	/* 2) Store new column into R. */
	for( i=0; i<nFR; ++i )
		R[i*NVMAX + nFR] = r[i];

	if ( r0 > 0.0 )
		R[nFR*NVMAX + nFR] = sqrt( r0 );
	else
	{
		hessianType = HST_SEMIDEF;
		return THROWERROR( RET_HESSIAN_NOT_SPD );
	}


	return SUCCESSFUL_RETURN;
}


/*
 *	b a c k s o l v e R  (CODE DUPLICATED IN QProblem CLASS!!!)
 */
returnValue QProblemB::backsolveR(	const real_t* const b, BooleanType transposed,
									real_t* const a
									)
{
	/* Call standard backsolve procedure (i.e. removingBound == BT_FALSE). */
	return backsolveR( b,transposed,BT_FALSE,a );
}


/*
 *	b a c k s o l v e R  (CODE DUPLICATED IN QProblem CLASS!!!)
 */
returnValue QProblemB::backsolveR(	const real_t* const b, BooleanType transposed,
									BooleanType removingBound,
									real_t* const a
									)
{
	int i, j;
	int nR = getNZ( );

	real_t sum;

	/* if backsolve is called while removing a bound, reduce nZ by one. */
	if ( removingBound == BT_TRUE )
		--nR;

	/* nothing to do */
	if ( nR <= 0 )
		return SUCCESSFUL_RETURN;


	/* Solve Ra = b, where R might be transposed. */
	if ( transposed == BT_FALSE )
	{
		/* solve Ra = b */
		for( i=(nR-1); i>=0; --i )
		{
			sum = b[i];
			for( j=(i+1); j<nR; ++j )
				sum -= R[i*NVMAX + j] * a[j];

			if ( getAbs( R[i*NVMAX + i] ) > ZERO )
				a[i] = sum / R[i*NVMAX + i];
			else
				return THROWERROR( RET_DIV_BY_ZERO );
		}
	}
	else
	{
		/* solve R^T*a = b */
		for( i=0; i<nR; ++i )
		{
			sum = b[i];

			for( j=0; j<i; ++j )
				sum -= R[j*NVMAX + i] * a[j];

			if ( getAbs( R[i*NVMAX + i] ) > ZERO )
				a[i] = sum / R[i*NVMAX + i];
			else
				return THROWERROR( RET_DIV_BY_ZERO );
		}
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	h o t s t a r t _ d e t e r m i n e D a t a S h i f t
 */
returnValue QProblemB::hotstart_determineDataShift(	const int* const FX_idx,
													const real_t* const g_new, const real_t* const lb_new, const real_t* const ub_new,
													real_t* const delta_g, real_t* const delta_lb, real_t* const delta_ub,
													BooleanType& Delta_bB_isZero
													)
{
	int i, ii;
	int nV  = getNV( );
	int nFX = getNFX( );


	/* 1) Calculate shift directions. */
	for( i=0; i<nV; ++i )
		delta_g[i]  = g_new[i]  - g[i];

	if ( lb_new != 0 )
	{
		for( i=0; i<nV; ++i )
			delta_lb[i] = lb_new[i] - lb[i];
	}
	else
	{
		/* if no lower bounds exist, assume the new lower bounds to be -infinity */
		for( i=0; i<nV; ++i )
			delta_lb[i] = -INFTY - lb[i];
	}

	if ( ub_new != 0 )
	{
		for( i=0; i<nV; ++i )
			delta_ub[i] = ub_new[i] - ub[i];
	}
	else
	{
		/* if no upper bounds exist, assume the new upper bounds to be infinity */
		for( i=0; i<nV; ++i )
			delta_ub[i] = INFTY - ub[i];
	}

	/* 2) Determine if active bounds are to be shifted. */
	Delta_bB_isZero = BT_TRUE;

	for ( i=0; i<nFX; ++i )
	{
		ii = FX_idx[i];

		if ( ( getAbs( delta_lb[ii] ) > EPS ) || ( getAbs( delta_ub[ii] ) > EPS ) )
		{
			Delta_bB_isZero = BT_FALSE;
			break;
		}
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	a r e B o u n d s C o n s i s t e n t
 */
BooleanType QProblemB::areBoundsConsistent(	const real_t* const delta_lb, const real_t* const delta_ub
											) const
{
	int i;

	/* Check if delta_lb[i] is greater than delta_ub[i]
	 * for a component i whose bounds are already (numerically) equal. */
	for( i=0; i<getNV( ); ++i )
		if ( ( lb[i] > ub[i] - BOUNDTOL ) && ( delta_lb[i] > delta_ub[i] + EPS ) )
			return BT_FALSE;

	return BT_TRUE;
}


/*
 *	s e t u p Q P d a t a
 */
returnValue QProblemB::setupQPdata(	const real_t* const _H, const real_t* const _R, const real_t* const _g,
									const real_t* const _lb, const real_t* const _ub
									)
{
	int i, j;
	int nV = getNV( );

	/* 1) Setup Hessian matrix and it's Cholesky factorization. */
	if (_H != 0)
	{
		for( i=0; i<nV; ++i )
			for( j=0; j<nV; ++j )
				H[i*NVMAX + j] = _H[i*nV + j];
		hasHessian = BT_TRUE;
	}
	else
		hasHessian = BT_FALSE;

	if (_R != 0)
	{
		for( i=0; i<nV; ++i )
			for( j=0; j<nV; ++j )
				R[i*NVMAX + j] = _R[i*nV + j];
		hasCholesky = BT_TRUE;

		/* If Hessian is not provided, store it's factorization in H, and that guy
		 * is going to be used for H * x products (R^T * R * x in this case). */
		if (hasHessian == BT_FALSE)
			for( i=0; i<nV; ++i )
				for( j=0; j<nV; ++j )
					H[i*NVMAX + j] = _R[i*nV + j];
	}
	else
		hasCholesky = BT_FALSE;

	if (hasHessian == BT_FALSE && hasCholesky == BT_FALSE)
		return THROWERROR( RET_INVALID_ARGUMENTS );

	/* 2) Setup gradient vector. */
	if ( _g == 0 )
		return THROWERROR( RET_INVALID_ARGUMENTS );

	for( i=0; i<nV; ++i )
		g[i] = _g[i];

	/* 3) Setup lower bounds vector. */
	if ( _lb != 0 )
	{
		for( i=0; i<nV; ++i )
			lb[i] = _lb[i];
	}
	else
	{
		/* if no lower bounds are specified, set them to -infinity */
		for( i=0; i<nV; ++i )
			lb[i] = -INFTY;
	}

	/* 4) Setup upper bounds vector. */
	if ( _ub != 0 )
	{
		for( i=0; i<nV; ++i )
			ub[i] = _ub[i];
	}
	else
	{
		/* if no upper bounds are specified, set them to infinity */
		for( i=0; i<nV; ++i )
			ub[i] = INFTY;
	}

	//printmatrix( "H",H,nV,nV );
	//printmatrix( "R",R,nV,nV );
	//printmatrix( "g",g,1,nV );
	//printmatrix( "lb",lb,1,nV );
	//printmatrix( "ub",ub,1,nV );

	return SUCCESSFUL_RETURN;
}



/*****************************************************************************
 *  P R I V A T E                                                            *
 *****************************************************************************/

/*
 *	h o t s t a r t _ d e t e r m i n e S t e p D i r e c t i o n
 */
returnValue QProblemB::hotstart_determineStepDirection(	const int* const FR_idx, const int* const FX_idx,
														const real_t* const delta_g, const real_t* const delta_lb, const real_t* const delta_ub,
														BooleanType Delta_bB_isZero,
														real_t* const delta_xFX, real_t* const delta_xFR,
														real_t* const delta_yFX
														)
{
	int i, j, ii, jj;
	int nFR = getNFR( );
	int nFX = getNFX( );


	/* initialise auxiliary vectors */
	real_t HMX_delta_xFX[NVMAX];
	for( i=0; i<nFR; ++i )
		HMX_delta_xFX[i] = 0.0;


	/* I) DETERMINE delta_xFX */
	if ( nFX > 0 )
	{
		for( i=0; i<nFX; ++i )
		{
			ii = FX_idx[i];

			if ( bounds.getStatus( ii ) == ST_LOWER )
				delta_xFX[i] = delta_lb[ii];
			else
				delta_xFX[i] = delta_ub[ii];
		}
	}


	/* II) DETERMINE delta_xFR */
	if ( nFR > 0 )
	{
		/* auxiliary variables */
		real_t delta_xFRz_TMP[NVMAX];
		real_t delta_xFRz_RHS[NVMAX];

		/* Determine delta_xFRz. */
		if ( Delta_bB_isZero == BT_FALSE )
		{
			for( i=0; i<nFR; ++i )
			{
				ii = FR_idx[i];
				for( j=0; j<nFX; ++j )
				{
					jj = FX_idx[j];
					HMX_delta_xFX[i] += H[ii*NVMAX + jj] * delta_xFX[j];
				}
			}
		}

		if ( Delta_bB_isZero == BT_TRUE )
		{
			for( j=0; j<nFR; ++j )
			{
				jj = FR_idx[j];
				delta_xFRz_RHS[j] = delta_g[jj];
			}
		}
		else
		{
			for( j=0; j<nFR; ++j )
			{
				jj = FR_idx[j];
				delta_xFRz_RHS[j] = delta_g[jj] + HMX_delta_xFX[j]; /* *ZFR */
			}
		}

		for( i=0; i<nFR; ++i )
			delta_xFR[i] = -delta_xFRz_RHS[i];

		if ( backsolveR( delta_xFR,BT_TRUE,delta_xFRz_TMP ) != SUCCESSFUL_RETURN )
			return THROWERROR( RET_STEPDIRECTION_FAILED_CHOLESKY );

		if ( backsolveR( delta_xFRz_TMP,BT_FALSE,delta_xFR ) != SUCCESSFUL_RETURN )
			return THROWERROR( RET_STEPDIRECTION_FAILED_CHOLESKY );
	}


	/* III) DETERMINE delta_yFX */
	if ( nFX > 0 )
	{
		for( i=0; i<nFX; ++i )
		{
			ii = FX_idx[i];

			delta_yFX[i] = 0.0;
			for( j=0; j<nFR; ++j )
			{
				jj = FR_idx[j];
				delta_yFX[i] += H[ii*NVMAX + jj] * delta_xFR[j];
			}

			if ( Delta_bB_isZero == BT_FALSE )
			{
				for( j=0; j<nFX; ++j )
				{
					jj = FX_idx[j];
					delta_yFX[i] += H[ii*NVMAX + jj] * delta_xFX[j];
				}
			}

			delta_yFX[i] += delta_g[ii];
		}
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	h o t s t a r t _ d e t e r m i n e S t e p L e n g t h
 */
returnValue QProblemB::hotstart_determineStepLength(	const int* const FR_idx, const int* const FX_idx,
														const real_t* const delta_lb, const real_t* const delta_ub,
														const real_t* const delta_xFR,
														const real_t* const delta_yFX,
														int& BC_idx, SubjectToStatus& BC_status
														)
{
	int i, ii;
	int nFR = getNFR( );
	int nFX = getNFX( );

	real_t tau_tmp;
	real_t tau_new = 1.0;

	BC_idx = 0;
	BC_status = ST_UNDEFINED;


	/* I) DETERMINE MAXIMUM DUAL STEPLENGTH, i.e. ensure that
	 *    active dual bounds remain valid (ignoring implicitly fixed variables): */
	for( i=0; i<nFX; ++i )
	{
		ii = FX_idx[i];

		if ( bounds.getType( ii ) != ST_EQUALITY )
		{
			if ( bounds.getStatus( ii ) == ST_LOWER )
			{
				/* 1) Active lower bounds. */
				if ( ( delta_yFX[i] < -ZERO ) && ( y[ii] >= 0.0 ) )
				{
					tau_tmp = y[ii] / ( -delta_yFX[i] );
					if ( tau_tmp < tau_new )
					{
						if ( tau_tmp >= 0.0 )
						{
							tau_new = tau_tmp;
							BC_idx = ii;
							BC_status = ST_INACTIVE;
						}
					}
				}
			}
			else
			{
				/* 2) Active upper bounds. */
				if ( ( delta_yFX[i] > ZERO ) && ( y[ii] <= 0.0 ) )
				{
					tau_tmp = y[ii] / ( -delta_yFX[i] );
					if ( tau_tmp < tau_new )
					{
						if ( tau_tmp >= 0.0 )
						{
							tau_new = tau_tmp;
							BC_idx = ii;
							BC_status = ST_INACTIVE;
						}
					}
				}
			}
		}
	}


	/* II) DETERMINE MAXIMUM PRIMAL STEPLENGTH, i.e. ensure that
	 *     inactive bounds remain valid (ignoring unbounded variables). */
	/* 1) Inactive lower bounds. */
	if ( bounds.isNoLower( ) == BT_FALSE )
	{
		for( i=0; i<nFR; ++i )
		{
			ii = FR_idx[i];

			if ( bounds.getType( ii ) != ST_UNBOUNDED )
			{
				if ( delta_lb[ii] > delta_xFR[i] )
				{
					if ( x[ii] > lb[ii] )
						tau_tmp = ( x[ii] - lb[ii] ) / ( delta_lb[ii] - delta_xFR[i] );
					else
						tau_tmp = 0.0;

					if ( tau_tmp < tau_new )
					{
						if ( tau_tmp >= 0.0 )
						{
							tau_new = tau_tmp;
							BC_idx = ii;
							BC_status = ST_LOWER;
						}
					}
				}
			}
		}
	}

	/* 2) Inactive upper bounds. */
	if ( bounds.isNoUpper( ) == BT_FALSE )
	{
		for( i=0; i<nFR; ++i )
		{
			ii = FR_idx[i];

			if ( bounds.getType( ii ) != ST_UNBOUNDED )
			{
				if ( delta_ub[ii] < delta_xFR[i] )
				{
					if ( x[ii] < ub[ii] )
						tau_tmp = ( x[ii] - ub[ii] ) / ( delta_ub[ii] - delta_xFR[i] );
					else
						tau_tmp = 0.0;

					if ( tau_tmp < tau_new )
					{
						if ( tau_tmp >= 0.0 )
						{
							tau_new = tau_tmp;
							BC_idx = ii;
							BC_status = ST_UPPER;
						}
					}
				}
			}
		}
	}


	/* III) SET MAXIMUM HOMOTOPY STEPLENGTH */
	tau = tau_new;

	if ( printlevel ==  PL_HIGH )
	{
		#ifdef PC_DEBUG
		char messageString[80];

		if ( BC_status == ST_UNDEFINED )
			sprintf( messageString,"Stepsize is %.6e!",tau );
		else
			sprintf( messageString,"Stepsize is %.6e! (BC_idx = %d, BC_status = %d)",tau,BC_idx,BC_status );

		getGlobalMessageHandler( )->throwInfo( RET_STEPSIZE_NONPOSITIVE,messageString,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE );
		#endif
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	h o t s t a r t _ p e r f o r m S t e p
 */
returnValue QProblemB::hotstart_performStep(	const int* const FR_idx, const int* const FX_idx,
												const real_t* const delta_g, const real_t* const  delta_lb, const real_t* const delta_ub,
												const real_t* const delta_xFX, const real_t* const delta_xFR,
												const real_t* const delta_yFX,
												int BC_idx, SubjectToStatus BC_status
												)
{
	int i, ii;
	int nV  = getNV( );
	int nFR = getNFR( );
	int nFX = getNFX( );


	/* I) CHECK BOUNDS' CONSISTENCY */
	if ( areBoundsConsistent( delta_lb,delta_ub ) == BT_FALSE )
	{
		infeasible = BT_TRUE;
		tau = 0.0;

		return THROWERROR( RET_QP_INFEASIBLE );
	}


	/* II) GO TO ACTIVE SET CHANGE */
	if ( tau > ZERO )
	{
		/* 1) Perform step in primal und dual space. */
		for( i=0; i<nFR; ++i )
		{
			ii = FR_idx[i];
			x[ii] += tau*delta_xFR[i];
		}

		for( i=0; i<nFX; ++i )
		{
			ii = FX_idx[i];
			x[ii] += tau*delta_xFX[i];
			y[ii] += tau*delta_yFX[i];
		}

		/* 2) Shift QP data. */
		for( i=0; i<nV; ++i )
		{
			g[i]  += tau*delta_g[i];
			lb[i] += tau*delta_lb[i];
			ub[i] += tau*delta_ub[i];
		}
	}
	else
	{
		/* print a stepsize warning if stepsize is zero */
		#ifdef PC_DEBUG
		char messageString[80];
		sprintf( messageString,"Stepsize is %.6e",tau );
		getGlobalMessageHandler( )->throwWarning( RET_STEPSIZE,messageString,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE );
		#endif
	}


	/* setup output preferences */
	#ifdef PC_DEBUG
	char messageString[80];
  	VisibilityStatus visibilityStatus;

  	if ( printlevel == PL_HIGH )
		visibilityStatus = VS_VISIBLE;
	else
		visibilityStatus = VS_HIDDEN;
	#endif


	/* III) UPDATE ACTIVE SET */
	switch ( BC_status )
	{
		/* Optimal solution found as no working set change detected. */
		case ST_UNDEFINED:
			return RET_OPTIMAL_SOLUTION_FOUND;


		/* Remove one variable from active set. */
		case ST_INACTIVE:
			#ifdef PC_DEBUG
			sprintf( messageString,"bound no. %d.", BC_idx );
			getGlobalMessageHandler( )->throwInfo( RET_REMOVE_FROM_ACTIVESET,messageString,__FUNCTION__,__FILE__,__LINE__,visibilityStatus );
			#endif

			if ( removeBound( BC_idx,BT_TRUE ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_REMOVE_FROM_ACTIVESET_FAILED );

			y[BC_idx] = 0.0;
			break;


		/* Add one variable to active set. */
		default:
			#ifdef PC_DEBUG
			if ( BC_status == ST_LOWER )
				sprintf( messageString,"lower bound no. %d.", BC_idx );
			else
				sprintf( messageString,"upper bound no. %d.", BC_idx );
				getGlobalMessageHandler( )->throwInfo( RET_ADD_TO_ACTIVESET,messageString,__FUNCTION__,__FILE__,__LINE__,visibilityStatus );
			#endif

			if ( addBound( BC_idx,BC_status,BT_TRUE ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_ADD_TO_ACTIVESET_FAILED );
			break;
	}

	return SUCCESSFUL_RETURN;
}


#ifdef PC_DEBUG  /* Define print functions only for debugging! */

/*
 *	p r i n t I t e r a t i o n
 */
returnValue QProblemB::printIteration( 	int iteration,
										int BC_idx,	SubjectToStatus BC_status
		  								)
{
	char myPrintfString[160];

	/* consistency check */
	if ( iteration < 0 )
		return THROWERROR( RET_INVALID_ARGUMENTS );

	/* nothing to do */
	if ( printlevel != PL_MEDIUM )
		return SUCCESSFUL_RETURN;


	/* 1) Print header at first iteration. */
 	if ( iteration == 0 )
	{
		sprintf( myPrintfString,"\n##############  qpOASES  --  QP NO.%4.1d  ###############\n", count );
		myPrintf( myPrintfString );

		sprintf( myPrintfString,"   Iter   |   StepLength    |       Info      |   nFX   \n" );
		myPrintf( myPrintfString );
	}

	/* 2) Print iteration line. */
	if ( BC_status == ST_UNDEFINED )
	{
		sprintf( myPrintfString,"   %4.1d   |   %1.5e   |    QP SOLVED    |  %4.1d   \n", iteration,tau,getNFX( ) );
		myPrintf( myPrintfString );
	}
	else
	{
		char info[8];

		if ( BC_status == ST_INACTIVE )
			sprintf( info,"REM BND" );
		else
			sprintf( info,"ADD BND" );

		sprintf( myPrintfString,"   %4.1d   |   %1.5e   |   %s%4.1d   |  %4.1d   \n", iteration,tau,info,BC_idx,getNFX( ) );
		myPrintf( myPrintfString );
	}

	return SUCCESSFUL_RETURN;
}

#endif  /* PC_DEBUG */



/*
 *	c h e c k K K T c o n d i t i o n s
 */
returnValue QProblemB::checkKKTconditions( )
{
	#ifdef __PERFORM_KKT_TEST__

	int i, j;
	int nV = getNV( );

	real_t tmp;
	real_t maxKKTviolation = 0.0;


	/* 1) Check for Hx + g - y*A' = 0  (here: A = Id). */
	for( i=0; i<nV; ++i )
	{
		tmp = g[i];

		for( j=0; j<nV; ++j )
			tmp += H[i*nV + j] * x[j];

		tmp -= y[i];

		if ( getAbs( tmp ) > maxKKTviolation )
			maxKKTviolation = getAbs( tmp );
	}

	/* 2) Check for lb <= x <= ub. */
	for( i=0; i<nV; ++i )
	{
		if ( lb[i] - x[i] > maxKKTviolation )
			maxKKTviolation = lb[i] - x[i];

		if ( x[i] - ub[i] > maxKKTviolation )
			maxKKTviolation = x[i] - ub[i];
	}

	/* 3) Check for correct sign of y and for complementary slackness. */
	for( i=0; i<nV; ++i )
	{
		switch ( bounds.getStatus( i ) )
		{
			case ST_LOWER:
				if ( -y[i] > maxKKTviolation )
					maxKKTviolation = -y[i];
				if ( getAbs( ( x[i] - lb[i] ) * y[i] ) > maxKKTviolation )
					maxKKTviolation = getAbs( ( x[i] - lb[i] ) * y[i] );
				break;

			case ST_UPPER:
				if ( y[i] > maxKKTviolation )
					maxKKTviolation = y[i];
				if ( getAbs( ( ub[i] - x[i] ) * y[i] ) > maxKKTviolation )
					maxKKTviolation = getAbs( ( ub[i] - x[i] ) * y[i] );
				break;

			default: /* inactive */
			if ( getAbs( y[i] ) > maxKKTviolation )
					maxKKTviolation = getAbs( y[i] );
				break;
		}
	}

	if ( maxKKTviolation > CRITICALACCURACY )
		return RET_NO_SOLUTION;

	if ( maxKKTviolation > DESIREDACCURACY )
		return RET_INACCURATE_SOLUTION;

	#endif /* __PERFORM_KKT_TEST__ */

	return SUCCESSFUL_RETURN;
}



/*
 *	end of file
 */
