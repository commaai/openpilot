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
 *	\file SRC/QProblem.cpp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Implementation of the QProblem class which is able to use the newly
 *	developed online active set strategy for parametric quadratic programming.
 */


#include <QProblem.hpp>

#include <stdio.h>

void printmatrix2(char *name, double *A, int m, int n) {
  int i, j;

  printf("%s = [...\n", name);
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++)
        printf("  % 9.4f", A[i*n+j]);
    printf(",\n");
  }
  printf("];\n");
}

//#define __PERFORM_KKT_TEST__


/*****************************************************************************
 *  P U B L I C                                                              *
 *****************************************************************************/


/*
 *	Q P r o b l e m
 */
QProblem::QProblem( ) : QProblemB( )
{
	constraints.init( 0 );

	sizeT = 0;

	cyclingManager.init( 0,0 );
}


/*
 *	Q P r o b l e m
 */
QProblem::QProblem( int _nV, int _nC ) : QProblemB( _nV )
{
	/* consistency checks */
	if ( _nV <= 0 )
		_nV = 1;

	if ( _nC < 0 )
	{
		_nC = 0;
		THROWERROR( RET_INVALID_ARGUMENTS );
	}

	constraints.init( _nC );


	sizeT = _nC;
	if ( _nC > _nV )
		sizeT = _nV;

	cyclingManager.init( _nV,_nC );
}


/*
 *	Q P r o b l e m
 */
QProblem::QProblem( const QProblem& rhs ) :	QProblemB( rhs )
{
	int i, j;

	int _nV = rhs.bounds.getNV( );
	int _nC = rhs.constraints.getNC( );

	for( i=0; i<_nC; ++i )
		for( j=0; j<_nV; ++j )
			A[i*NVMAX + j] = rhs.A[i*NVMAX + j];

	for( i=0; i<_nC; ++i )
		lbA[i] = rhs.lbA[i];

	for( i=0; i<_nC; ++i )
			ubA[i] = rhs.ubA[i];

	constraints = rhs.constraints;

	for( i=0; i<(_nV+_nC); ++i )
		y[i] = rhs.y[i];


	sizeT = rhs.sizeT;

	for( i=0; i<sizeT; ++i )
		for( j=0; j<sizeT; ++j )
			T[i*NVMAX + j] = rhs.T[i*NVMAX + j];

	for( i=0; i<_nV; ++i )
		for( j=0; j<_nV; ++j )
			Q[i*NVMAX + j] = rhs.Q[i*NVMAX + j];

	for( i=0; i<_nC; ++i )
		Ax[i] = rhs.Ax[i];

	cyclingManager = rhs.cyclingManager;
}


/*
 *	~ Q P r o b l e m
 */
QProblem::~QProblem( )
{
}


/*
 *	o p e r a t o r =
 */
QProblem& QProblem::operator=( const QProblem& rhs )
{
	int i, j;

	if ( this != &rhs )
	{
		QProblemB::operator=( rhs );


		int _nV = rhs.bounds.getNV( );
		int _nC = rhs.constraints.getNC( );

		for( i=0; i<_nC; ++i )
			for( j=0; j<_nV; ++j )
				A[i*NVMAX + j] = rhs.A[i*NVMAX + j];

		for( i=0; i<_nC; ++i )
			lbA[i] = rhs.lbA[i];

		for( i=0; i<_nC; ++i )
			ubA[i] = rhs.ubA[i];

		constraints = rhs.constraints;

		for( i=0; i<(_nV+_nC); ++i )
			y[i] = rhs.y[i];


		sizeT = rhs.sizeT;

		for( i=0; i<sizeT; ++i )
			for( j=0; j<sizeT; ++j )
				T[i*NVMAX + j] = rhs.T[i*NVMAX + j];

		for( i=0; i<_nV; ++i )
			for( j=0; j<_nV; ++j )
				Q[i*NVMAX + j] = rhs.Q[i*NVMAX + j];

		for( i=0; i<_nC; ++i )
			Ax[i] = rhs.Ax[i];

		cyclingManager = rhs.cyclingManager;
	}

	return *this;
}


/*
 *	r e s e t
 */
returnValue QProblem::reset( )
{
	int i, j;
	int nV = getNV( );
	int nC = getNC( );


	/* 1) Reset bounds, Cholesky decomposition and status flags. */
	if ( QProblemB::reset( ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_RESET_FAILED );

	/* 2) Reset constraints. */
	constraints.init( nC );

	/* 3) Reset TQ factorisation. */
	for( i=0; i<sizeT; ++i )
		for( j=0; j<sizeT; ++j )
			T[i*NVMAX + j] = 0.0;

	for( i=0; i<nV; ++i )
		for( j=0; j<nV; ++j )
			Q[i*NVMAX + j] = 0.0;

	/* 4) Reset cycling manager. */
	if ( cyclingManager.clearCyclingData( ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_RESET_FAILED );

	return SUCCESSFUL_RETURN;
}


/*
 *	i n i t
 */
returnValue QProblem::init(	const real_t* const _H, const real_t* const _g, const real_t* const _A,
							const real_t* const _lb, const real_t* const _ub,
							const real_t* const _lbA, const real_t* const _ubA,
							int& nWSR, const real_t* const yOpt, real_t* const cputime
							)
{
	/* 1) Setup QP data. */
	if (setupQPdata(_H, 0, _g, _A, _lb, _ub, _lbA, _ubA) != SUCCESSFUL_RETURN)
		return THROWERROR( RET_INVALID_ARGUMENTS );

	/* 2) Call to main initialisation routine (without any additional information). */
	return solveInitialQP( 0,yOpt,0,0, nWSR,cputime );
}

returnValue QProblem::init(	const real_t* const _H, const real_t* const _R, const real_t* const _g, const real_t* const _A,
							const real_t* const _lb, const real_t* const _ub,
							const real_t* const _lbA, const real_t* const _ubA,
							int& nWSR, const real_t* const yOpt, real_t* const cputime
							)
{
	/* 1) Setup QP data. */
	if (setupQPdata(_H, _R, _g, _A, _lb, _ub, _lbA, _ubA) != SUCCESSFUL_RETURN)
		return THROWERROR( RET_INVALID_ARGUMENTS );

	/* 2) Call to main initialisation routine (without any additional information). */
	return solveInitialQP( 0,yOpt,0,0, nWSR,cputime );
}


/*
 *	h o t s t a r t
 */
returnValue QProblem::hotstart(	const real_t* const g_new, const real_t* const lb_new, const real_t* const ub_new,
								const real_t* const lbA_new, const real_t* const ubA_new,
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
	/* 1) Reset cycling and status flags and increase QP counter. */
	cyclingManager.clearCyclingData( );

	infeasible = BT_FALSE;
	unbounded = BT_FALSE;

	++count;

	/* 2) Allocate delta vectors of gradient and (constraints') bounds. */
	returnValue returnvalue;
	BooleanType Delta_bC_isZero, Delta_bB_isZero;

	int FR_idx[NVMAX];
	int FX_idx[NVMAX];
	int AC_idx[NCMAX_ALLOC];
	int IAC_idx[NCMAX_ALLOC];

	real_t delta_g[NVMAX];
	real_t delta_lb[NVMAX];
	real_t delta_ub[NVMAX];
	real_t delta_lbA[NCMAX_ALLOC];
	real_t delta_ubA[NCMAX_ALLOC];

	real_t delta_xFR[NVMAX];
	real_t delta_xFX[NVMAX];
	real_t delta_yAC[NCMAX_ALLOC];
	real_t delta_yFX[NVMAX];
	real_t delta_Ax[NCMAX_ALLOC];

	int BC_idx;
	SubjectToStatus BC_status;
	BooleanType BC_isBound;

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

		if ( constraints.getActive( )->getNumberArray( AC_idx ) != SUCCESSFUL_RETURN )
			return THROWERROR( RET_HOTSTART_FAILED );

		if ( constraints.getInactive( )->getNumberArray( IAC_idx ) != SUCCESSFUL_RETURN )
			return THROWERROR( RET_HOTSTART_FAILED );

		/* 2) Detemination of shift direction of the gradient and the (constraints') bounds. */
		returnvalue = hotstart_determineDataShift(  FX_idx, AC_idx,
													g_new,lbA_new,ubA_new,lb_new,ub_new,
													delta_g,delta_lbA,delta_ubA,delta_lb,delta_ub,
													Delta_bC_isZero, Delta_bB_isZero );
		if ( returnvalue != SUCCESSFUL_RETURN )
		{
			nWSR = l;
			THROWERROR( RET_SHIFT_DETERMINATION_FAILED );
			return returnvalue;
		}

		/* 3) Determination of step direction of X and Y. */
		returnvalue = hotstart_determineStepDirection(	FR_idx,FX_idx,AC_idx,
														delta_g,delta_lbA,delta_ubA,delta_lb,delta_ub,
														Delta_bC_isZero, Delta_bB_isZero,
														delta_xFX,delta_xFR,delta_yAC,delta_yFX
														);
		if ( returnvalue != SUCCESSFUL_RETURN )
		{
			nWSR = l;
			THROWERROR( RET_STEPDIRECTION_DETERMINATION_FAILED );
			return returnvalue;
		}

		/* 4) Determination of step length TAU. */
		returnvalue = hotstart_determineStepLength(	FR_idx,FX_idx,AC_idx,IAC_idx,
													delta_lbA,delta_ubA,delta_lb,delta_ub,
													delta_xFX,delta_xFR,delta_yAC,delta_yFX,delta_Ax,
													BC_idx,BC_status,BC_isBound
													);
		if ( returnvalue != SUCCESSFUL_RETURN )
		{
			nWSR = l;
			THROWERROR( RET_STEPLENGTH_DETERMINATION_FAILED );
			return returnvalue;
		}

		/* 5) Realisation of the homotopy step. */
		returnvalue = hotstart_performStep(	FR_idx,FX_idx,AC_idx,IAC_idx,
											delta_g,delta_lbA,delta_ubA,delta_lb,delta_ub,
											delta_xFX,delta_xFR,delta_yAC,delta_yFX,delta_Ax,
											BC_idx,BC_status,BC_isBound
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
	 			if ( printIteration( l,BC_idx,BC_status,BC_isBound ) != SUCCESSFUL_RETURN )
					THROWERROR( RET_PRINT_ITERATION_FAILED ); /* do not pass this as return value! */
				#endif

				/* check KKT optimality conditions */
				return checkKKTconditions( );
			}
			else
			{
				/* checks for infeasibility... */
				if ( isInfeasible( ) == BT_TRUE )
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
		if ( printIteration( l,BC_idx,BC_status,BC_isBound ) != SUCCESSFUL_RETURN )
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
int QProblem::getNZ( )
{
	/* nZ = nFR - nAC */
	return bounds.getFree( )->getLength( ) - constraints.getActive( )->getLength( );
}


/*
 *	g e t D u a l S o l u t i o n
 */
returnValue QProblem::getDualSolution( real_t* const yOpt ) const
{
	int i;

	/* return optimal dual solution vector
	 * only if current QP has been solved */
	if ( ( getStatus( ) == QPS_AUXILIARYQPSOLVED ) ||
		 ( getStatus( ) == QPS_HOMOTOPYQPSOLVED )  ||
		 ( getStatus( ) == QPS_SOLVED ) )
	{
		for( i=0; i<getNV( )+getNC( ); ++i )
			yOpt[i] = y[i];

		return SUCCESSFUL_RETURN;
	}
	else
	{
		return RET_QP_NOT_SOLVED;
	}
}



/*****************************************************************************
 *  P R O T E C T E D                                                        *
 *****************************************************************************/

/*
 *	s e t u p S u b j e c t T o T y p e
 */
returnValue QProblem::setupSubjectToType( )
{
	int i;
	int nC = getNC( );


	/* I) SETUP SUBJECTTOTYPE FOR BOUNDS */
	if ( QProblemB::setupSubjectToType( ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_SETUPSUBJECTTOTYPE_FAILED );


	/* II) SETUP SUBJECTTOTYPE FOR CONSTRAINTS */
	/* 1) Check if lower constraints' bounds are present. */
	constraints.setNoLower( BT_TRUE );
	for( i=0; i<nC; ++i )
	{
		if ( lbA[i] > -INFTY )
		{
			constraints.setNoLower( BT_FALSE );
			break;
		}
	}

	/* 2) Check if upper constraints' bounds are present. */
	constraints.setNoUpper( BT_TRUE );
	for( i=0; i<nC; ++i )
	{
		if ( ubA[i] < INFTY )
		{
			constraints.setNoUpper( BT_FALSE );
			break;
		}
	}

	/* 3) Determine implicit equality constraints and unbounded constraints. */
	int nEC = 0;
	int nUC = 0;

	for( i=0; i<nC; ++i )
	{
		if ( ( lbA[i] < -INFTY + BOUNDTOL ) && ( ubA[i] > INFTY - BOUNDTOL ) )
		{
			constraints.setType( i,ST_UNBOUNDED );
			++nUC;
		}
		else
		{
			if ( lbA[i] > ubA[i] - BOUNDTOL )
			{
				constraints.setType( i,ST_EQUALITY );
				++nEC;
			}
			else
			{
				constraints.setType( i,ST_BOUNDED );
			}
		}
	}

	/* 4) Set dimensions of constraints structure. */
	constraints.setNEC( nEC );
	constraints.setNUC( nUC );
	constraints.setNIC( nC - nEC - nUC );

	return SUCCESSFUL_RETURN;
}


/*
 *	c h o l e s k y D e c o m p o s i t i o n P r o j e c t e d
 */
returnValue QProblem::setupCholeskyDecompositionProjected( )
{
	int i, j, k, ii, kk;
	int nV  = getNV( );
	int nFR = getNFR( );
	int nZ  = getNZ( );

	/* 1) Initialises R with all zeros. */
	for( i=0; i<nV; ++i )
		for( j=0; j<nV; ++j )
			R[i*NVMAX + j] = 0.0;

	/* 2) Calculate Cholesky decomposition of projected Hessian Z'*H*Z. */
	if ( hessianType == HST_IDENTITY )
	{
		/* if Hessian is identity, so is its Cholesky factor. */
		for( i=0; i<nV; ++i )
			R[i*NVMAX + i] = 1.0;
	}
	else
	{
		if ( nZ > 0 )
		{
			int FR_idx[NVMAX];
			if ( bounds.getFree( )->getNumberArray( FR_idx ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_INDEXLIST_CORRUPTED );

#if 0
			real_t HZ[NVMAX*NVMAX];
			real_t ZHZ[NVMAX*NVMAX];

			/* calculate H*Z */
			for ( i=0; i<nFR; ++i )
			{
				ii = FR_idx[i];

				for ( j=0; j<nZ; ++j )
				{
					real_t sum = 0.0;
					for ( k=0; k<nFR; ++k )
					{
						kk = FR_idx[k];
						sum += H[ii*NVMAX + kk] * Q[kk*NVMAX + j];
					}
					HZ[i * NVMAX + j] = sum;
				}
			}

			/* calculate Z'*H*Z */
			for ( i=0; i<nZ; ++i )
				for ( j=0; j<nZ; ++j )
				{
					real_t sum = 0.0;
					for ( k=0; k<nFR; ++k )
					{
						kk = FR_idx[k];
						sum += Q[kk*NVMAX + i] * HZ[k*NVMAX + j];
					}
					ZHZ[i * NVMAX + j] = sum;
				}

			/* R'*R = Z'*H*Z */
			real_t sum, inv;

			for( i=0; i<nZ; ++i )
			{
				/* j == i */
				sum = ZHZ[i*NVMAX + i];

				for( k=(i-1); k>=0; --k )
					sum -= R[k*NVMAX + i] * R[k*NVMAX + i];

				if ( sum > 0.0 )
				{
					R[i*NVMAX + i] = sqrt( sum );
					inv = 1.0 / R[i * NVMAX + i];
				}
				else
				{
					hessianType = HST_SEMIDEF;
					return THROWERROR( RET_HESSIAN_NOT_SPD );
				}

				for( j=(i+1); j<nZ; ++j )
				{
					sum = ZHZ[j*NVMAX + i];

					for( k=(i-1); k>=0; --k )
						sum -= R[k*NVMAX + i] * R[k*NVMAX + j];

					R[i*NVMAX + j] = sum * inv;
				}
			}
#else
			real_t HZ[NVMAX];
			real_t ZHZ[NVMAX];

			real_t sum, inv;
			for (j = 0; j < nZ; ++j)
			{
				/* Cache one column of Z. */
				for (i = 0; i < NVMAX; ++i)
					ZHZ[i] = Q[i * NVMAX + j];

				/* Create one column of the product H * Z. */
				for (i = 0; i < nFR; ++i)
				{
					ii = FR_idx[i];

					sum = 0.0;
					for (k = 0; k < nFR; ++k)
					{
						kk = FR_idx[k];
						sum += H[ii * NVMAX + kk] * ZHZ[kk];
					}
					HZ[ii] = sum;
				}

				/* Create one column of the product Z^T * H * Z. */
				for (i = j; i < nZ; ++i)
					ZHZ[ i ] = 0.0;

				for (k = 0; k < nFR; ++k)
				{
					kk = FR_idx[k];
					real_t q = HZ[kk];
					for (i = j; i < nZ; ++i)
					{
						ZHZ[i] += Q[kk * NVMAX + i] * q;
					}
				}

				/* Use the computed column to update the factorization. */
				/* j == i */
				sum = ZHZ[j];

				for (k = (j - 1); k >= 0; --k)
					sum -= R[k * NVMAX + j] * R[k * NVMAX + j];

				if (sum > 0.0)
				{
					R[j * NVMAX + j] = sqrt(sum);
					inv = 1.0 / R[j * NVMAX + j];
				}
				else
				{
					hessianType = HST_SEMIDEF;
					return THROWERROR( RET_HESSIAN_NOT_SPD );
				}

				for (i = (j + 1); i < nZ; ++i)
				{
					sum = ZHZ[i];

					for (k = (j - 1); k >= 0; --k)
						sum -= R[k * NVMAX + j] * R[k * NVMAX + i];

					R[j * NVMAX + i] = sum * inv;
				}
			}
#endif
		}
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t u p T Q f a c t o r i s a t i o n
 */
returnValue QProblem::setupTQfactorisation( )
{
	int i, j, ii;
	int nV  = getNV( );
	int nFR = getNFR( );

	int FR_idx[NVMAX];
	if ( bounds.getFree( )->getNumberArray( FR_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INDEXLIST_CORRUPTED );

	/* 1) Set Q to unity matrix. */
	for( i=0; i<nV; ++i )
		for( j=0; j<nV; ++j )
			Q[i*NVMAX + j] = 0.0;

	for( i=0; i<nFR; ++i )
	{
		ii = FR_idx[i];
		Q[ii*NVMAX + i] = 1.0;
	}

 	/* 2) Set T to zero matrix. */
	for( i=0; i<sizeT; ++i )
		for( j=0; j<sizeT; ++j )
			T[i*NVMAX + j] = 0.0;

	return SUCCESSFUL_RETURN;
}


/*
 *	s o l v e I n i t i a l Q P
 */
returnValue QProblem::solveInitialQP(	const real_t* const xOpt, const real_t* const yOpt,
										const Bounds* const guessedBounds, const Constraints* const guessedConstraints,
										int& nWSR, real_t* const cputime
										)
{
	int i;

	/* some definitions */
	int nV = getNV( );
	int nC = getNC( );


	/* start runtime measurement */
	real_t starttime = 0.0;
	if ( cputime != 0 )
		starttime = getCPUtime( );


	status = QPS_NOTINITIALISED;

	/* I) ANALYSE QP DATA: */
	/* 1) Check if Hessian happens to be the identity matrix. */
	if ( checkForIdentityHessian( ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INIT_FAILED );

	/* 2) Setup type of bounds and constraints (i.e. unbounded, implicitly fixed etc.). */
	if ( setupSubjectToType( ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INIT_FAILED );

	/* 3) Initialise cycling manager. */
	cyclingManager.clearCyclingData( );

	status = QPS_PREPARINGAUXILIARYQP;


	/* II) SETUP AUXILIARY QP WITH GIVEN OPTIMAL SOLUTION: */
	/* 1) Setup bounds and constraints data structure. */
	if ( bounds.setupAllFree( ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INIT_FAILED );

	if ( constraints.setupAllInactive( ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INIT_FAILED );

	/* 2) Setup optimal primal/dual solution for auxiliary QP. */
	if ( setupAuxiliaryQPsolution( xOpt,yOpt ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INIT_FAILED );

	/* 3) Obtain linear independent working set for auxiliary QP. */

	static Bounds auxiliaryBounds;

	auxiliaryBounds.init( nV );

	static Constraints auxiliaryConstraints;

	auxiliaryConstraints.init( nC );

	if ( obtainAuxiliaryWorkingSet(	xOpt,yOpt,guessedBounds,guessedConstraints,
									&auxiliaryBounds,&auxiliaryConstraints ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INIT_FAILED );

	/* 4) Setup working set of auxiliary QP and setup matrix factorisations. */
	if ( setupTQfactorisation( ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INIT_FAILED_TQ );

	if ( setupAuxiliaryWorkingSet( &auxiliaryBounds,&auxiliaryConstraints,BT_TRUE ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INIT_FAILED );

	if ( ( getNAC( ) + getNFX( ) ) == 0 )
	{
		/* Factorise full Hessian if no bounds/constraints are active. */
		if (hasCholesky == BT_FALSE)
			if ( setupCholeskyDecomposition( ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_INIT_FAILED_CHOLESKY );
		/* ... else we use user provided Cholesky factorization. At the moment
		 * we can do that only for cold-started solver. */
	}
	else
	{
		/* Factorise projected Hessian if there active bounds/constraints. */
		if ( setupCholeskyDecompositionProjected( ) != SUCCESSFUL_RETURN )
			return THROWERROR( RET_INIT_FAILED_CHOLESKY );
		/* TODO: use user-supplied Hessian decomposition. R_Z = R * Z. */
	}

	/* 5) Store original QP formulation... */
	real_t g_original[NVMAX];
	real_t lb_original[NVMAX];
	real_t ub_original[NVMAX];
	real_t lbA_original[NCMAX_ALLOC];
	real_t ubA_original[NCMAX_ALLOC];

	for( i=0; i<nV; ++i )
	{
		g_original[i] = g[i];
		lb_original[i] = lb[i];
		ub_original[i] = ub[i];
	}

	for( i=0; i<nC; ++i )
	{
		lbA_original[i] = lbA[i];
		ubA_original[i] = ubA[i];
	}

	/* ... and setup QP data of an auxiliary QP having an optimal solution
	 * as specified by the user (or xOpt = yOpt = 0, by default). */
	if ( setupAuxiliaryQPgradient( ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INIT_FAILED );

	if ( setupAuxiliaryQPbounds( &auxiliaryBounds,&auxiliaryConstraints,BT_TRUE ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INIT_FAILED );

	status = QPS_AUXILIARYQPSOLVED;


	/* III) SOLVE ACTUAL INITIAL QP: */
	/* Use hotstart method to find the solution of the original initial QP,... */
	returnValue returnvalue = hotstart( g_original,lb_original,ub_original,lbA_original,ubA_original, nWSR,0 );


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
returnValue QProblem::obtainAuxiliaryWorkingSet(	const real_t* const xOpt, const real_t* const yOpt,
													const Bounds* const guessedBounds, const Constraints* const guessedConstraints,
													Bounds* auxiliaryBounds, Constraints* auxiliaryConstraints
													) const
{
	int i = 0;
	int nV = getNV( );
	int nC = getNC( );


	/* 1) Ensure that desiredBounds is allocated (and different from guessedBounds). */
	if ( ( auxiliaryBounds == 0 ) || ( auxiliaryBounds == guessedBounds ) )
		return THROWERROR( RET_INVALID_ARGUMENTS );

	if ( ( auxiliaryConstraints == 0 ) || ( auxiliaryConstraints == guessedConstraints ) )
		return THROWERROR( RET_INVALID_ARGUMENTS );


	SubjectToStatus guessedStatus;

	/* 2) Setup working set of bounds for auxiliary initial QP. */
	if ( QProblemB::obtainAuxiliaryWorkingSet( xOpt,yOpt,guessedBounds, auxiliaryBounds ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );

	/* 3) Setup working set of constraints for auxiliary initial QP. */
	if ( guessedConstraints != 0 )
	{
		/* If an initial working set is specific, use it!
		 * Moreover, add all equality constraints if specified. */
		for( i=0; i<nC; ++i )
		{
			guessedStatus = guessedConstraints->getStatus( i );

			if ( constraints.getType( i ) == ST_EQUALITY )
			{
				if ( auxiliaryConstraints->setupConstraint( i,ST_LOWER ) != SUCCESSFUL_RETURN )
					return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
			}
			else
			{
				if ( auxiliaryConstraints->setupConstraint( i,guessedStatus ) != SUCCESSFUL_RETURN )
					return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
			}
		}
	}
	else	/* No initial working set specified. */
	{
		/* Obtain initial working set by "clipping". */
		if ( ( xOpt != 0 ) && ( yOpt == 0 ) )
		{
			for( i=0; i<nC; ++i )
			{
				if ( Ax[i] <= lbA[i] + BOUNDTOL )
				{
					if ( auxiliaryConstraints->setupConstraint( i,ST_LOWER ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
					continue;
				}

				if ( Ax[i] >= ubA[i] - BOUNDTOL )
				{
					if ( auxiliaryConstraints->setupConstraint( i,ST_UPPER ) != SUCCESSFUL_RETURN )
							return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
					continue;
				}

				/* Moreover, add all equality constraints if specified. */
				if ( constraints.getType( i ) == ST_EQUALITY )
				{
					if ( auxiliaryConstraints->setupConstraint( i,ST_LOWER ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
				}
				else
				{
					if ( auxiliaryConstraints->setupConstraint( i,ST_INACTIVE ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
				}
			}
		}

		/* Obtain initial working set in accordance to sign of dual solution vector. */
		if ( ( xOpt == 0 ) && ( yOpt != 0 ) )
		{
			for( i=0; i<nC; ++i )
			{
				if ( yOpt[nV+i] > ZERO )
				{
					if ( auxiliaryConstraints->setupConstraint( i,ST_LOWER ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
					continue;
				}

				if ( yOpt[nV+i] < -ZERO )
				{
					if ( auxiliaryConstraints->setupConstraint( i,ST_UPPER ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
					continue;
				}

				/* Moreover, add all equality constraints if specified. */
				if ( constraints.getType( i ) == ST_EQUALITY )
				{
					if ( auxiliaryConstraints->setupConstraint( i,ST_LOWER ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
				}
				else
				{
					if ( auxiliaryConstraints->setupConstraint( i,ST_INACTIVE ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
				}
			}
		}

		/* If xOpt and yOpt are null pointer and no initial working is specified,
		 * start with empty working set (or implicitly fixed bounds and equality constraints only)
		 * for auxiliary QP. */
		if ( ( xOpt == 0 ) && ( yOpt == 0 ) )
		{
			for( i=0; i<nC; ++i )
			{
				/* Only add all equality constraints if specified. */
				if ( constraints.getType( i ) == ST_EQUALITY )
				{
					if ( auxiliaryConstraints->setupConstraint( i,ST_LOWER ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_OBTAINING_WORKINGSET_FAILED );
				}
				else
				{
					if ( auxiliaryConstraints->setupConstraint( i,ST_INACTIVE ) != SUCCESSFUL_RETURN )
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
returnValue QProblem::setupAuxiliaryWorkingSet(	const Bounds* const auxiliaryBounds,
												const Constraints* const auxiliaryConstraints,
												BooleanType setupAfresh
												)
{
	int i;
	int nV = getNV( );
	int nC = getNC( );

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

	if ( auxiliaryConstraints != 0 )
	{
		for( i=0; i<nC; ++i )
			if ( ( constraints.getStatus( i ) == ST_UNDEFINED ) || ( auxiliaryConstraints->getStatus( i ) == ST_UNDEFINED ) )
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


	/* II) REMOVE FORMERLY ACTIVE (CONSTRAINTS') BOUNDS (IF NECESSARY): */
	if ( setupAfresh == BT_FALSE )
	{
		/* 1) Remove all active constraints that shall be inactive AND
		*    all active constraints that are active at the wrong bound. */
		for( i=0; i<nC; ++i )
		{
			if ( ( constraints.getStatus( i ) == ST_LOWER ) && ( auxiliaryConstraints->getStatus( i ) != ST_LOWER ) )
				if ( removeConstraint( i,updateCholesky ) != SUCCESSFUL_RETURN )
					return THROWERROR( RET_SETUP_WORKINGSET_FAILED );

			if ( ( constraints.getStatus( i ) == ST_UPPER ) && ( auxiliaryConstraints->getStatus( i ) != ST_UPPER ) )
				if ( removeConstraint( i,updateCholesky ) != SUCCESSFUL_RETURN )
					return THROWERROR( RET_SETUP_WORKINGSET_FAILED );
		}

		/* 2) Remove all active bounds that shall be inactive AND
		*    all active bounds that are active at the wrong bound. */
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


	/* III) ADD NEWLY ACTIVE (CONSTRAINTS') BOUNDS: */
	/* 1) Add all inactive bounds that shall be active AND
	 *    all formerly active bounds that have been active at the wrong bound. */
	for( i=0; i<nV; ++i )
	{
		if ( ( bounds.getStatus( i ) == ST_INACTIVE ) && ( auxiliaryBounds->getStatus( i ) != ST_INACTIVE ) )
		{
			/* Add bound only if it is linearly independent from the current working set. */
			if ( addBound_checkLI( i ) == RET_LINEARLY_INDEPENDENT )
			{
				if ( addBound( i,auxiliaryBounds->getStatus( i ),updateCholesky ) != SUCCESSFUL_RETURN )
					return THROWERROR( RET_SETUP_WORKINGSET_FAILED );
			}
		}
	}

	/* 2) Add all inactive constraints that shall be active AND
	 *    all formerly active constraints that have been active at the wrong bound. */
	for( i=0; i<nC; ++i )
	{
		if ( ( auxiliaryConstraints->getStatus( i ) == ST_LOWER ) || ( auxiliaryConstraints->getStatus( i ) == ST_UPPER ) )
		{
			/* formerly inactive */
			if ( constraints.getStatus( i ) == ST_INACTIVE )
			{
				/* Add constraint only if it is linearly independent from the current working set. */
				if ( addConstraint_checkLI( i ) == RET_LINEARLY_INDEPENDENT )
				{
					if ( addConstraint( i,auxiliaryConstraints->getStatus( i ),updateCholesky ) != SUCCESSFUL_RETURN )
						return THROWERROR( RET_SETUP_WORKINGSET_FAILED );
				}
			}
		}
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t u p A u x i l i a r y Q P s o l u t i o n
 */
returnValue QProblem::setupAuxiliaryQPsolution(	const real_t* const xOpt, const real_t* const yOpt
												)
{
	int i, j;
	int nV = getNV( );
	int nC = getNC( );


	/* Setup primal/dual solution vector for auxiliary initial QP:
	 * if a null pointer is passed, a zero vector is assigned;
	 *  old solution vector is kept if pointer to internal solution vevtor is passed. */
	if ( xOpt != 0 )
	{
		if ( xOpt != x )
			for( i=0; i<nV; ++i )
				x[i] = xOpt[i];

		for ( j=0; j<nC; ++j )
		{
			Ax[j] = 0.0;

			for( i=0; i<nV; ++i )
				Ax[j] += A[j*NVMAX + i] * x[i];
		}
	}
	else
	{
		for( i=0; i<nV; ++i )
			x[i] = 0.0;

		for ( j=0; j<nC; ++j )
			Ax[j] = 0.0;
	}

	if ( yOpt != 0 )
	{
		if ( yOpt != y )
			for( i=0; i<nV+nC; ++i )
				y[i] = yOpt[i];
	}
	else
	{
		for( i=0; i<nV+nC; ++i )
			y[i] = 0.0;
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t u p A u x i l i a r y Q P g r a d i e n t
 */
returnValue QProblem::setupAuxiliaryQPgradient( )
{
	int i, j;
	int nV = getNV( );
	int nC = getNC( );


	/* Setup gradient vector: g = -H*x + [Id A]'*[yB yC]. */
	for ( i=0; i<nV; ++i )
	{
		/* Id'*yB */
		g[i] = y[i];

		/* A'*yC */
		for ( j=0; j<nC; ++j )
			g[i] += A[j*NVMAX + i] * y[nV+j];

		/* -H*x */
		for ( j=0; j<nV; ++j )
			g[i] -= H[i*NVMAX + j] * x[j];
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t u p A u x i l i a r y Q P b o u n d s
 */
returnValue QProblem::setupAuxiliaryQPbounds(	const Bounds* const auxiliaryBounds,
												const Constraints* const auxiliaryConstraints,
												BooleanType useRelaxation
												)
{
	int i;
	int nV = getNV( );
	int nC = getNC( );


	/* 1) Setup bound vectors. */
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
						/* If a bound is inactive although it was supposed to be
						* active by the auxiliaryBounds, it could not be added
						* due to linear dependence. Thus set it "strongly inactive". */
						if ( auxiliaryBounds->getStatus( i ) == ST_LOWER )
							lb[i] = x[i];
						else
							lb[i] = x[i] - BOUNDRELAXATION;

						if ( auxiliaryBounds->getStatus( i ) == ST_UPPER )
							ub[i] = x[i];
						else
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

	/* 2) Setup constraints vectors. */
	for ( i=0; i<nC; ++i )
	{
		switch ( constraints.getStatus( i ) )
		{
			case ST_INACTIVE:
				if ( useRelaxation == BT_TRUE )
				{
					if ( constraints.getType( i ) == ST_EQUALITY )
					{
						lbA[i] = Ax[i];
						ubA[i] = Ax[i];
					}
					else
					{
						/* If a constraint is inactive although it was supposed to be
						* active by the auxiliaryConstraints, it could not be added
						* due to linear dependence. Thus set it "strongly inactive". */
						if ( auxiliaryConstraints->getStatus( i ) == ST_LOWER )
							lbA[i] = Ax[i];
						else
							lbA[i] = Ax[i] - BOUNDRELAXATION;

						if ( auxiliaryConstraints->getStatus( i ) == ST_UPPER )
							ubA[i] = Ax[i];
						else
							ubA[i] = Ax[i] + BOUNDRELAXATION;
					}
				}
				break;

			case ST_LOWER:
				lbA[i] = Ax[i];
				if ( constraints.getType( i ) == ST_EQUALITY )
				{
					ubA[i] = Ax[i];
				}
				else
				{
					if ( useRelaxation == BT_TRUE )
						ubA[i] = Ax[i] + BOUNDRELAXATION;
				}
				break;

			case ST_UPPER:
				ubA[i] = Ax[i];
				if ( constraints.getType( i ) == ST_EQUALITY )
				{
					lbA[i] = Ax[i];
				}
				else
				{
					if ( useRelaxation == BT_TRUE )
						lbA[i] = Ax[i] - BOUNDRELAXATION;
				}
				break;

			default:
				return THROWERROR( RET_UNKNOWN_BUG );
		}
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	a d d C o n s t r a i n t
 */
returnValue QProblem::addConstraint(	int number, SubjectToStatus C_status,
										BooleanType updateCholesky
										)
{
	int i, j, ii;

	/* consistency checks */
	if ( constraints.getStatus( number ) != ST_INACTIVE )
		return THROWERROR( RET_CONSTRAINT_ALREADY_ACTIVE );

	if ( ( constraints.getNC( ) - getNAC( ) ) == constraints.getNUC( ) )
		return THROWERROR( RET_ALL_CONSTRAINTS_ACTIVE );

	if ( ( getStatus( ) == QPS_NOTINITIALISED )    ||
		 ( getStatus( ) == QPS_AUXILIARYQPSOLVED ) ||
		 ( getStatus( ) == QPS_HOMOTOPYQPSOLVED )  ||
		 ( getStatus( ) == QPS_SOLVED )            )
	{
		return THROWERROR( RET_UNKNOWN_BUG );
	}


	/* I) ENSURE LINEAR INDEPENDENCE OF THE WORKING SET,
	 *    i.e. remove a constraint or bound if linear dependence occurs. */
	/* check for LI only if Cholesky decomposition shall be updated! */
	if ( updateCholesky == BT_TRUE )
	{
		returnValue ensureLIreturnvalue = addConstraint_ensureLI( number,C_status );

		switch ( ensureLIreturnvalue )
		{
			case SUCCESSFUL_RETURN:
				break;

			case RET_LI_RESOLVED:
				break;

			case RET_ENSURELI_FAILED_NOINDEX:
				return THROWERROR( RET_ADDCONSTRAINT_FAILED_INFEASIBILITY );

			case RET_ENSURELI_FAILED_CYCLING:
				return THROWERROR( RET_ADDCONSTRAINT_FAILED_INFEASIBILITY );

			default:
				return THROWERROR( RET_ENSURELI_FAILED );
		}
	}

	/* some definitions */
	int nFR = getNFR( );
	int nAC = getNAC( );
	int nZ  = getNZ( );

	int tcol = sizeT - nAC;


	int FR_idx[NVMAX];
	if ( bounds.getFree( )->getNumberArray( FR_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_ADDCONSTRAINT_FAILED );

	real_t aFR[NVMAX];
	real_t wZ[NVMAX];
	for( i=0; i<nZ; ++i )
		wZ[i] = 0.0;


	/* II) ADD NEW ACTIVE CONSTRAINT TO MATRIX T: */
	/* 1) Add row [wZ wY] = aFR'*[Z Y] to the end of T: assign aFR. */
	for( i=0; i<nFR; ++i )
	{
		ii = FR_idx[i];
		aFR[i] = A[number*NVMAX + ii];
	}

	/* calculate wZ */
	for( i=0; i<nFR; ++i )
	{
		ii = FR_idx[i];
		for( j=0; j<nZ; ++j )
			wZ[j] += aFR[i] * Q[ii*NVMAX + j];
	}

	/* 2) Calculate wY and store it directly into T. */
	if ( nAC > 0 )
	{
		for( j=0; j<nAC; ++j )
			T[nAC*NVMAX + tcol+j] = 0.0;
		for( i=0; i<nFR; ++i )
		{
			ii = FR_idx[i];
			for( j=0; j<nAC; ++j )
				T[nAC*NVMAX + tcol+j] += aFR[i] * Q[ii*NVMAX + nZ+j];
		}
	}


	real_t c, s;

	if ( nZ > 0 )
	{
		/* II) RESTORE TRIANGULAR FORM OF T: */
		/*     Use column-wise Givens rotations to restore reverse triangular form
		*      of T, simultanenous change of Q (i.e. Z) and R. */
		for( j=0; j<nZ-1; ++j )
		{
			computeGivens( wZ[j+1],wZ[j], wZ[j+1],wZ[j],c,s );

			for( i=0; i<nFR; ++i )
			{
				ii = FR_idx[i];
				applyGivens( c,s,Q[ii*NVMAX + 1+j],Q[ii*NVMAX + j], Q[ii*NVMAX + 1+j],Q[ii*NVMAX + j] );
			}

			if ( ( updateCholesky == BT_TRUE ) && ( hessianType != HST_IDENTITY ) )
			{
				for( i=0; i<=j+1; ++i )
					applyGivens( c,s,R[i*NVMAX + 1+j],R[i*NVMAX + j], R[i*NVMAX + 1+j],R[i*NVMAX + j] );
			}
		}

		T[nAC*NVMAX + tcol-1] = wZ[nZ-1];


		if ( ( updateCholesky == BT_TRUE ) && ( hessianType != HST_IDENTITY ) )
		{
			/* III) RESTORE TRIANGULAR FORM OF R:
			 *      Use row-wise Givens rotations to restore upper triangular form of R. */
			for( i=0; i<nZ-1; ++i )
			{
				computeGivens( R[i*NVMAX + i],R[(1+i)*NVMAX + i], R[i*NVMAX + i],R[(1+i)*NVMAX + i],c,s );

				for( j=(1+i); j<(nZ-1); ++j ) /* last column of R is thrown away */
					applyGivens( c,s,R[i*NVMAX + j],R[(1+i)*NVMAX + j], R[i*NVMAX + j],R[(1+i)*NVMAX + j] );
			}
			/* last column of R is thrown away */
			for( i=0; i<nZ; ++i )
				R[i*NVMAX + nZ-1] = 0.0;
		}
	}


	/* IV) UPDATE INDICES */
	if ( constraints.moveInactiveToActive( number,C_status ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_ADDCONSTRAINT_FAILED );


	return SUCCESSFUL_RETURN;
}



/*
 *	a d d C o n s t r a i n t _ c h e c k L I
 */
returnValue QProblem::addConstraint_checkLI( int number )
{
	int i, j, jj;
	int nFR = getNFR( );
	int nZ  = getNZ( );

	int FR_idx[NVMAX];
	if ( bounds.getFree( )->getNumberArray( FR_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_INDEXLIST_CORRUPTED );

	/* Check if constraint <number> is linearly independent from the
	   the active ones (<=> is element of null space of Afr). */
	real_t sum;

	for( i=0; i<nZ; ++i )
	{
		sum = 0.0;
		for( j=0; j<nFR; ++j )
		{
			jj = FR_idx[j];
			sum += Q[jj*NVMAX + i] * A[number*NVMAX + jj];
		}

		if ( getAbs( sum ) > 10.0*EPS )
			return RET_LINEARLY_INDEPENDENT;
	}

	return RET_LINEARLY_DEPENDENT;
}


/*
 *	a d d C o n s t r a i n t _ e n s u r e L I
 */
returnValue QProblem::addConstraint_ensureLI( int number, SubjectToStatus C_status )
{
	int i, j, ii, jj;
	int nV  = getNV( );
	int nFR = getNFR( );
	int nFX = getNFX( );
	int nAC = getNAC( );
	int nZ  = getNZ( );


	/* I) Check if new constraint is linearly independent from the active ones. */
	returnValue returnvalueCheckLI = addConstraint_checkLI( number );

	if ( returnvalueCheckLI == RET_INDEXLIST_CORRUPTED )
		return THROWERROR( RET_ENSURELI_FAILED );

	if ( returnvalueCheckLI == RET_LINEARLY_INDEPENDENT )
		return SUCCESSFUL_RETURN;


 	/* II) NEW CONSTRAINT IS LINEARLY DEPENDENT: */
	/* 1) Determine coefficients of linear combination,
	 *    cf. M.J. Best. Applied Mathematics and Parallel Computing, chapter:
	 *    An Algorithm for the Solution of the Parametric Quadratic Programming
	 *    Problem, pages 57-76. Physica-Verlag, Heidelberg, 1996. */
	int FR_idx[NVMAX];
	if ( bounds.getFree( )->getNumberArray( FR_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_ENSURELI_FAILED );

	int FX_idx[NVMAX];
	if ( bounds.getFixed( )->getNumberArray( FX_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_ENSURELI_FAILED );

	real_t xiC[NCMAX_ALLOC];
	real_t xiC_TMP[NCMAX_ALLOC];
	real_t xiB[NVMAX];

	/* 2) Calculate xiC */
	if ( nAC > 0 )
	{
		if ( C_status == ST_LOWER )
		{
			for( i=0; i<nAC; ++i )
			{
				xiC_TMP[i] = 0.0;
				for( j=0; j<nFR; ++j )
				{
					jj = FR_idx[j];
					xiC_TMP[i] += Q[jj*NVMAX + nZ+i] * A[number*NVMAX + jj];
				}
			}
		}
		else
		{
			for( i=0; i<nAC; ++i )
			{
				xiC_TMP[i] = 0.0;
				for( j=0; j<nFR; ++j )
				{
					jj = FR_idx[j];
					xiC_TMP[i] -= Q[jj*NVMAX + nZ+i] * A[number*NVMAX + jj];
				}
			}
		}

		if ( backsolveT( xiC_TMP, BT_TRUE, xiC ) != SUCCESSFUL_RETURN )
			return THROWERROR( RET_ENSURELI_FAILED_TQ );
	}

	/* 3) Calculate xiB. */
	int AC_idx[NCMAX_ALLOC];
	if ( constraints.getActive( )->getNumberArray( AC_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_ENSURELI_FAILED );

	if ( C_status == ST_LOWER )
	{
		for( i=0; i<nFX; ++i )
		{
			ii = FX_idx[i];
			xiB[i] = A[number*NVMAX + ii];

			for( j=0; j<nAC; ++j )
			{
				jj = AC_idx[j];
				xiB[i] -= A[jj*NVMAX + ii] * xiC[j];
			}
		}
	}
	else
	{
		for( i=0; i<nFX; ++i )
		{
			ii = FX_idx[i];
			xiB[i] = -A[number*NVMAX + ii];

			for( j=0; j<nAC; ++j )
			{
				jj = AC_idx[j];
				xiB[i] -= A[jj*NVMAX + ii] * xiC[j];
			}
		}
	}


	/* III) DETERMINE CONSTRAINT/BOUND TO BE REMOVED. */
	real_t y_min = INFTY * INFTY;
	int y_min_number = -1;
	BooleanType y_min_isBound = BT_FALSE;

	/* 1) Constraints. */
	for( i=0; i<nAC; ++i )
	{
		ii = AC_idx[i];

		if ( constraints.getStatus( ii ) == ST_LOWER )
		{
			if ( ( xiC[i] > ZERO ) && ( y[nV+ii] >= 0.0 ) )
			{
				if ( y[nV+ii]/xiC[i] < y_min )
				{
					y_min = y[nV+ii]/xiC[i];
					y_min_number = ii;
				}
			}
		}
		else
		{
			if ( ( xiC[i] < -ZERO ) && ( y[nV+ii] <= 0.0 ) )
			{
				if ( y[nV+ii]/xiC[i] < y_min )
				{
					y_min = y[nV+ii]/xiC[i];
					y_min_number = ii;
				}
			}
		}
	}

	/* 2) Bounds. */
	for( i=0; i<nFX; ++i )
	{
		ii = FX_idx[i];

		if ( bounds.getStatus( ii ) == ST_LOWER )
		{
			if ( ( xiB[i] > ZERO ) && ( y[ii] >= 0.0 ) )
			{
				if ( y[ii]/xiB[i] < y_min )
				{
					y_min = y[ii]/xiB[i];
					y_min_number = ii;
					y_min_isBound = BT_TRUE;
				}
			}
		}
		else
		{
			if ( ( xiB[i] < -ZERO ) && ( y[ii] <= 0.0 ) )
			{
				if ( y[ii]/xiB[i] < y_min )
				{
					y_min = y[ii]/xiB[i];
					y_min_number = ii;
					y_min_isBound = BT_TRUE;
				}
			}
		}
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


	/* IV) REMOVE CONSTRAINT/BOUND FOR RESOLVING LINEAR DEPENDENCE: */
	if ( y_min_number >= 0 )
	{
		/* 1) Check for cycling due to infeasibility. */
		if ( ( cyclingManager.getCyclingStatus( number,BT_FALSE ) == CYC_PREV_REMOVED ) &&
			 ( cyclingManager.getCyclingStatus( y_min_number,y_min_isBound ) == CYC_PREV_ADDED ) )
		{
			infeasible = BT_TRUE;

			return THROWERROR( RET_ENSURELI_FAILED_CYCLING );
		}
		else
		{
			/* set cycling data */
			cyclingManager.clearCyclingData( );
			cyclingManager.setCyclingStatus( number,BT_FALSE, CYC_PREV_ADDED );
			cyclingManager.setCyclingStatus( y_min_number,y_min_isBound, CYC_PREV_REMOVED );
		}

		/* 2) Update Lagrange multiplier... */
		for( i=0; i<nAC; ++i )
		{
			ii = AC_idx[i];
			y[nV+ii] -= y_min * xiC[i];
		}
		for( i=0; i<nFX; ++i )
		{
			ii = FX_idx[i];
			y[ii] -= y_min * xiB[i];
		}

		/* ... also for newly active constraint... */
		if ( C_status == ST_LOWER )
			y[nV+number] = y_min;
		else
			y[nV+number] = -y_min;

		/* ... and for constraint to be removed. */
		if ( y_min_isBound == BT_TRUE )
		{
			#ifdef PC_DEBUG
			sprintf( messageString,"bound no. %d.",y_min_number );
			getGlobalMessageHandler( )->throwInfo( RET_REMOVE_FROM_ACTIVESET,messageString,__FUNCTION__,__FILE__,__LINE__,visibilityStatus );
			#endif

			if ( removeBound( y_min_number,BT_TRUE ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_REMOVE_FROM_ACTIVESET_FAILED );

			y[y_min_number] = 0.0;
		}
		else
		{
			#ifdef PC_DEBUG
			sprintf( messageString,"constraint no. %d.",y_min_number );
			getGlobalMessageHandler( )->throwInfo( RET_REMOVE_FROM_ACTIVESET,messageString,__FUNCTION__,__FILE__,__LINE__,visibilityStatus );
			#endif

			if ( removeConstraint( y_min_number,BT_TRUE ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_REMOVE_FROM_ACTIVESET_FAILED );

			y[nV+y_min_number] = 0.0;
		}
	}
	else
	{
		/* no constraint/bound can be removed => QP is infeasible! */
		infeasible = BT_TRUE;

		return THROWERROR( RET_ENSURELI_FAILED_NOINDEX );
	}

	return getGlobalMessageHandler( )->throwInfo( RET_LI_RESOLVED,0,__FUNCTION__,__FILE__,__LINE__,VS_HIDDEN );
}



/*
 *	a d d B o u n d
 */
returnValue QProblem::addBound(	int number, SubjectToStatus B_status,
								BooleanType updateCholesky
								)
{
	int i, j, ii;

	/* consistency checks */
	if ( bounds.getStatus( number ) != ST_INACTIVE )
		return THROWERROR( RET_BOUND_ALREADY_ACTIVE );

	if ( getNFR( ) == bounds.getNUV( ) )
		return THROWERROR( RET_ALL_BOUNDS_ACTIVE );

	if ( ( getStatus( ) == QPS_NOTINITIALISED )    ||
		 ( getStatus( ) == QPS_AUXILIARYQPSOLVED ) ||
		 ( getStatus( ) == QPS_HOMOTOPYQPSOLVED )  ||
 		 ( getStatus( ) == QPS_SOLVED )            )
	{
		return THROWERROR( RET_UNKNOWN_BUG );
	}


	/* I) ENSURE LINEAR INDEPENDENCE OF THE WORKING SET,
	 *    i.e. remove a constraint or bound if linear dependence occurs. */
	/* check for LI only if Cholesky decomposition shall be updated! */
	if ( updateCholesky == BT_TRUE )
	{
		returnValue ensureLIreturnvalue = addBound_ensureLI( number,B_status );

		switch ( ensureLIreturnvalue )
		{
			case SUCCESSFUL_RETURN:
				break;

			case RET_LI_RESOLVED:
				break;

			case RET_ENSURELI_FAILED_NOINDEX:
				return THROWERROR( RET_ADDBOUND_FAILED_INFEASIBILITY );

			case RET_ENSURELI_FAILED_CYCLING:
				return THROWERROR( RET_ADDBOUND_FAILED_INFEASIBILITY );

			default:
				return THROWERROR( RET_ENSURELI_FAILED );
		}
	}


	/* some definitions */
	int nFR = getNFR( );
	int nAC = getNAC( );
	int nZ  = getNZ( );

	int tcol = sizeT - nAC;


	/* I) SWAP INDEXLIST OF FREE VARIABLES:
	 *    move the variable to be fixed to the end of the list of free variables. */
	int lastfreenumber = bounds.getFree( )->getLastNumber( );
	if ( lastfreenumber != number )
		if ( bounds.swapFree( number,lastfreenumber ) != SUCCESSFUL_RETURN )
			THROWERROR( RET_ADDBOUND_FAILED );


	int FR_idx[NVMAX];
	if ( bounds.getFree( )->getNumberArray( FR_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_ADDBOUND_FAILED );

	real_t w[NVMAX];


	/* II) ADD NEW ACTIVE BOUND TO TOP OF MATRIX T: */
	/* 1) add row [wZ wY] = [Z Y](number) at the top of T: assign w */
	for( i=0; i<nFR; ++i )
		w[i] = Q[FR_idx[nFR-1]*NVMAX + i];


	/* 2) Use column-wise Givens rotations to restore reverse triangular form
	 *    of the first row of T, simultanenous change of Q (i.e. Z) and R. */
	real_t c, s;

	for( j=0; j<nZ-1; ++j )
	{
		computeGivens( w[j+1],w[j], w[j+1],w[j],c,s );

		for( i=0; i<nFR; ++i )
		{
			ii = FR_idx[i];
			applyGivens( c,s,Q[ii*NVMAX + 1+j],Q[ii*NVMAX + j], Q[ii*NVMAX + 1+j],Q[ii*NVMAX + j] );
		}

		if ( ( updateCholesky == BT_TRUE ) && ( hessianType != HST_IDENTITY ) )
		{
			for( i=0; i<=j+1; ++i )
				applyGivens( c,s,R[i*NVMAX + 1+j],R[i*NVMAX + j], R[i*NVMAX + 1+j],R[i*NVMAX + j] );
		}
	}


	if ( nAC > 0 )	  /* ( nAC == 0 ) <=> ( nZ == nFR ) <=> Y and T are empty => nothing to do */
	{
		/* store new column a in a temporary vector instead of shifting T one column to the left */
		real_t tmp[NCMAX_ALLOC];
		for( i=0; i<nAC; ++i )
			tmp[i] = 0.0;

		{
			j = nZ-1;

			computeGivens( w[j+1],w[j], w[j+1],w[j],c,s );

			for( i=0; i<nFR; ++i )
			{
				ii = FR_idx[i];
				applyGivens( c,s,Q[ii*NVMAX + 1+j],Q[ii*NVMAX + j], Q[ii*NVMAX + 1+j],Q[ii*NVMAX + j] );
			}

			applyGivens( c,s,T[(nAC-1)*NVMAX + tcol],tmp[nAC-1], tmp[nAC-1],T[(nAC-1)*NVMAX + tcol] );
		}

		for( j=nZ; j<nFR-1; ++j )
		{
			computeGivens( w[j+1],w[j], w[j+1],w[j],c,s );

			for( i=0; i<nFR; ++i )
			{
				ii = FR_idx[i];
				applyGivens( c,s,Q[ii*NVMAX + 1+j],Q[ii*NVMAX + j], Q[ii*NVMAX + 1+j],Q[ii*NVMAX + j] );
			}

			for( i=(nFR-2-j); i<nAC; ++i )
				applyGivens( c,s,T[i*NVMAX + 1+tcol-nZ+j],tmp[i], tmp[i],T[i*NVMAX + 1+tcol-nZ+j] );
		}

	}


	if ( ( updateCholesky == BT_TRUE ) && ( hessianType != HST_IDENTITY ) )
	{
		/* III) RESTORE TRIANGULAR FORM OF R:
		 *      use row-wise Givens rotations to restore upper triangular form of R */
		for( i=0; i<nZ-1; ++i )
		{
			computeGivens( R[i*NVMAX + i],R[(1+i)*NVMAX + i], R[i*NVMAX + i],R[(1+i)*NVMAX + i],c,s );

			for( j=(1+i); j<nZ-1; ++j ) /* last column of R is thrown away */
				applyGivens( c,s,R[i*NVMAX + j],R[(1+i)*NVMAX + j], R[i*NVMAX + j],R[(1+i)*NVMAX + j] );
		}
		/* last column of R is thrown away */
		for( i=0; i<nZ; ++i )
			R[i*NVMAX + nZ-1] = 0.0;
	}


	/* IV) UPDATE INDICES */
	if ( bounds.moveFreeToFixed( number,B_status ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_ADDBOUND_FAILED );

	return SUCCESSFUL_RETURN;
}


/*
 *	a d d B o u n d _ c h e c k L I
 */
returnValue QProblem::addBound_checkLI( int number )
{
	int i;

	/* some definitions */
	int nZ  = getNZ( );

	/* Check if constraint <number> is linearly independent from the
	   the active ones (<=> is element of null space of Afr). */
	for( i=0; i<nZ; ++i )
	{
		if ( getAbs( Q[number*NVMAX + i] ) > EPS )
			return RET_LINEARLY_INDEPENDENT;
	}

	return RET_LINEARLY_DEPENDENT;
}


/*
 *	a d d B o u n d _ e n s u r e L I
 */
returnValue QProblem::addBound_ensureLI( int number, SubjectToStatus B_status )
{
	int i, j, ii, jj;
	int nV  = getNV( );
	int nFX = getNFX( );
	int nAC = getNAC( );
	int nZ  = getNZ( );


	/* I) Check if new constraint is linearly independent from the active ones. */
	returnValue returnvalueCheckLI = addBound_checkLI( number );

	if ( returnvalueCheckLI == RET_LINEARLY_INDEPENDENT )
		return SUCCESSFUL_RETURN;


 	/* II) NEW BOUND IS LINEARLY DEPENDENT: */
	/* 1) Determine coefficients of linear combination,
	 *    cf. M.J. Best. Applied Mathematics and Parallel Computing, chapter:
	 *    An Algorithm for the Solution of the Parametric Quadratic Programming
	 *    Problem, pages 57-76. Physica-Verlag, Heidelberg, 1996. */
	int FR_idx[NVMAX];
	if ( bounds.getFree( )->getNumberArray( FR_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_ENSURELI_FAILED );

	int FX_idx[NVMAX];
	if ( bounds.getFixed( )->getNumberArray( FX_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_ENSURELI_FAILED );

	int AC_idx[NCMAX_ALLOC];
	if ( constraints.getActive( )->getNumberArray( AC_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_ENSURELI_FAILED );

	real_t xiC[NCMAX_ALLOC];
	real_t xiC_TMP[NCMAX_ALLOC];
	real_t xiB[NVMAX];

	/* 2) Calculate xiC. */
	if ( nAC > 0 )
	{
		if ( B_status == ST_LOWER )
		{
			for( i=0; i<nAC; ++i )
				xiC_TMP[i] = Q[number*NVMAX + nZ+i];
		}
		else
		{
			for( i=0; i<nAC; ++i )
				xiC_TMP[i] = -Q[number*NVMAX + nZ+i];
		}

		if ( backsolveT( xiC_TMP, BT_TRUE, xiC ) != SUCCESSFUL_RETURN )
			return THROWERROR( RET_ENSURELI_FAILED_TQ );
	}

	/* 3) Calculate xiB. */
	for( i=0; i<nFX; ++i )
	{
		ii = FX_idx[i];

		xiB[i] = 0.0;
		for( j=0; j<nAC; ++j )
		{
			jj = AC_idx[j];
			xiB[i] -= A[jj*NVMAX + ii] * xiC[j];
		}
	}


	/* III) DETERMINE CONSTRAINT/BOUND TO BE REMOVED. */
	real_t y_min = INFTY * INFTY;
	int y_min_number = -1;
	BooleanType y_min_isBound = BT_FALSE;

	/* 1) Constraints. */
	for( i=0; i<nAC; ++i )
	{
		ii = AC_idx[i];

		if ( constraints.getStatus( ii ) == ST_LOWER )
		{
			if ( ( xiC[i] > ZERO ) && ( y[nV+ii] >= 0.0 ) )
			{
				if ( y[nV+ii]/xiC[i] < y_min )
				{
					y_min = y[nV+ii]/xiC[i];
					y_min_number = ii;
				}
			}
		}
		else
		{
			if ( ( xiC[i] < -ZERO ) && ( y[nV+ii] <= 0.0 ) )
			{
				if ( y[nV+ii]/xiC[i] < y_min )
				{
					y_min = y[nV+ii]/xiC[i];
					y_min_number = ii;
				}
			}
		}
	}

	/* 2) Bounds. */
	for( i=0; i<nFX; ++i )
	{
		ii = FX_idx[i];

		if ( bounds.getStatus( ii ) == ST_LOWER )
		{
			if ( ( xiB[i] > ZERO ) && ( y[ii] >= 0.0 ) )
			{
				if ( y[ii]/xiB[i] < y_min )
				{
					y_min = y[ii]/xiB[i];
					y_min_number = ii;
					y_min_isBound = BT_TRUE;
				}
			}
		}
		else
		{
			if ( ( xiB[i] < -ZERO ) && ( y[ii] <= 0.0 ) )
			{
				if ( y[ii]/xiB[i] < y_min )
				{
					y_min = y[ii]/xiB[i];
					y_min_number = ii;
					y_min_isBound = BT_TRUE;
				}
			}
		}
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


	/* IV) REMOVE CONSTRAINT/BOUND FOR RESOLVING LINEAR DEPENDENCE: */
	if ( y_min_number >= 0 )
	{
		/* 1) Check for cycling due to infeasibility. */
		if ( ( cyclingManager.getCyclingStatus( number,BT_TRUE ) == CYC_PREV_REMOVED ) &&
			 ( cyclingManager.getCyclingStatus( y_min_number,y_min_isBound ) == CYC_PREV_ADDED ) )
		{
			infeasible = BT_TRUE;

			return THROWERROR( RET_ENSURELI_FAILED_CYCLING );
		}
		else
		{
			/* set cycling data */
			cyclingManager.clearCyclingData( );
			cyclingManager.setCyclingStatus( number,BT_TRUE, CYC_PREV_ADDED );
			cyclingManager.setCyclingStatus( y_min_number,y_min_isBound, CYC_PREV_REMOVED );
		}


		/* 2) Update Lagrange multiplier... */
		for( i=0; i<nAC; ++i )
		{
			ii = AC_idx[i];
			y[nV+ii] -= y_min * xiC[i];
		}
		for( i=0; i<nFX; ++i )
		{
			ii = FX_idx[i];
			y[ii] -= y_min * xiB[i];
		}

		/* ... also for newly active bound ... */
		if ( B_status == ST_LOWER )
			y[number] = y_min;
		else
			y[number] = -y_min;

		/* ... and for bound to be removed. */
		if ( y_min_isBound == BT_TRUE )
		{
			#ifdef PC_DEBUG
			sprintf( messageString,"bound no. %d.",y_min_number );
			getGlobalMessageHandler( )->throwInfo( RET_REMOVE_FROM_ACTIVESET,messageString,__FUNCTION__,__FILE__,__LINE__,visibilityStatus );
			#endif

			if ( removeBound( y_min_number,BT_TRUE ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_REMOVE_FROM_ACTIVESET_FAILED );

			y[y_min_number] = 0.0;
		}
		else
		{
			#ifdef PC_DEBUG
			sprintf( messageString,"constraint no. %d.",y_min_number );
			getGlobalMessageHandler( )->throwInfo( RET_REMOVE_FROM_ACTIVESET,messageString,__FUNCTION__,__FILE__,__LINE__,visibilityStatus );
			#endif

			if ( removeConstraint( y_min_number,BT_TRUE ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_REMOVE_FROM_ACTIVESET_FAILED );

			y[nV+y_min_number] = 0.0;
		}
	}
	else
	{
		/* no constraint/bound can be removed => QP is infeasible! */
		infeasible = BT_TRUE;

		return THROWERROR( RET_ENSURELI_FAILED_NOINDEX );
	}

	return getGlobalMessageHandler( )->throwInfo( RET_LI_RESOLVED,0,__FUNCTION__,__FILE__,__LINE__,VS_HIDDEN );
}



/*
 *	r e m o v e C o n s t r a i n t
 */
returnValue QProblem::removeConstraint(	int number,
										BooleanType updateCholesky
										)
{
	int i, j, ii, jj;

	/* consistency check */
	if ( ( getStatus( ) == QPS_NOTINITIALISED )    ||
		 ( getStatus( ) == QPS_AUXILIARYQPSOLVED ) ||
		 ( getStatus( ) == QPS_HOMOTOPYQPSOLVED )  ||
 		 ( getStatus( ) == QPS_SOLVED )            )
	{
		return THROWERROR( RET_UNKNOWN_BUG );
	}

	/* some definitions */
	int nFR = getNFR( );
	int nAC = getNAC( );
	int nZ  = getNZ( );

	int tcol = sizeT - nAC;
	int number_idx = constraints.getActive( )->getIndex( number );


	/* consistency checks */
	if ( constraints.getStatus( number ) == ST_INACTIVE )
		return THROWERROR( RET_CONSTRAINT_NOT_ACTIVE );

	if ( ( number_idx < 0 ) || ( number_idx >= nAC ) )
		return THROWERROR( RET_CONSTRAINT_NOT_ACTIVE );


	int FR_idx[NVMAX];
	if ( bounds.getFree( )->getNumberArray( FR_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_REMOVECONSTRAINT_FAILED );


	/* I) REMOVE <number>th ROW FROM T,
	 *    i.e. shift rows number+1 through nAC  upwards (instead of the actual
	 *    constraint number its corresponding index within matrix A is used). */
	if ( number_idx < nAC-1 )
	{
		for( i=(number_idx+1); i<nAC; ++i )
			for( j=(nAC-i-1); j<nAC; ++j )
				T[(i-1)*NVMAX + tcol+j] = T[i*NVMAX + tcol+j];
		/* gimmick: write zeros into the last row of T */
		for( j=0; j<nAC; ++j )
			T[(nAC-1)*NVMAX + tcol+j] = 0.0;


		/* II) RESTORE TRIANGULAR FORM OF T,
		 *     use column-wise Givens rotations to restore reverse triangular form
		 *     of T simultanenous change of Q (i.e. Y). */
		real_t c, s;

		for( j=(nAC-2-number_idx); j>=0; --j )
		{
			computeGivens( T[(nAC-2-j)*NVMAX + tcol+1+j],T[(nAC-2-j)*NVMAX + tcol+j], T[(nAC-2-j)*NVMAX + tcol+1+j],T[(nAC-2-j)*NVMAX + tcol+j],c,s );

			for( i=(nAC-j-1); i<(nAC-1); ++i )
				applyGivens( c,s,T[i*NVMAX + tcol+1+j],T[i*NVMAX + tcol+j], T[i*NVMAX + tcol+1+j],T[i*NVMAX + tcol+j] );

			for( i=0; i<nFR; ++i )
			{
				ii = FR_idx[i];
				applyGivens( c,s,Q[ii*NVMAX + nZ+1+j],Q[ii*NVMAX + nZ+j], Q[ii*NVMAX + nZ+1+j],Q[ii*NVMAX + nZ+j] );
			}
		}
	}
	else
	{
		/* gimmick: write zeros into the last row of T */
		for( j=0; j<nAC; ++j )
			T[(nAC-1)*NVMAX + tcol+j] = 0.0;
	}


	if ( ( updateCholesky == BT_TRUE ) && ( hessianType != HST_IDENTITY ) )
	{
		/* III) UPDATE CHOLESKY DECOMPOSITION,
		 *      calculate new additional column (i.e. [r sqrt(rho2)]')
		 *      of the Cholesky factor R. */
		real_t Hz[NVMAX];
		for ( i=0; i<nFR; ++i )
			Hz[i] = 0.0;
		real_t rho2 = 0.0;

		/* 1) Calculate Hz = H*z, where z is the new rightmost column of Z
		 *    (i.e. the old leftmost column of Y).  */
		for( j=0; j<nFR; ++j )
		{
			jj = FR_idx[j];
			for( i=0; i<nFR; ++i )
				Hz[i] += H[jj*NVMAX + FR_idx[i]] * Q[jj*NVMAX + nZ];
		}

		if ( nZ > 0 )
		{
			real_t ZHz[NVMAX];
			for ( i=0; i<nZ; ++i )
				ZHz[i] = 0.0;
			real_t r[NVMAX];

			/* 2) Calculate ZHz = Z'*Hz (old Z). */
			for( j=0; j<nFR; ++j )
			{
				jj = FR_idx[j];

				for( i=0; i<nZ; ++i )
					ZHz[i] += Q[jj*NVMAX + i] * Hz[j];
			}

			/* 3) Calculate r = R^-T * ZHz. */
			if ( backsolveR( ZHz,BT_TRUE,r ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_REMOVECONSTRAINT_FAILED );

			/* 4) Calculate rho2 = rho^2 = z'*Hz - r'*r
			 *    and store r into R. */
			for( i=0; i<nZ; ++i )
			{
				rho2 -= r[i]*r[i];
				R[i*NVMAX + nZ] = r[i];
			}
		}

		for( j=0; j<nFR; ++j )
			rho2 += Q[FR_idx[j]*NVMAX + nZ] * Hz[j];

		/* 5) Store rho into R. */
		if ( rho2 > 0.0 )
			R[nZ*NVMAX + nZ] = sqrt( rho2 );
		else
		{
			hessianType = HST_SEMIDEF;
			return THROWERROR( RET_HESSIAN_NOT_SPD );
		}
	}

	/* IV) UPDATE INDICES */
	if ( constraints.moveActiveToInactive( number ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_REMOVECONSTRAINT_FAILED );


	return SUCCESSFUL_RETURN;
}


/*
 *	r e m o v e B o u n d
 */
returnValue QProblem::removeBound(	int number,
									BooleanType updateCholesky
									)
{
	int i, j, ii, jj;

	/* consistency checks */
	if ( bounds.getStatus( number ) == ST_INACTIVE )
		return THROWERROR( RET_BOUND_NOT_ACTIVE );

	if ( ( getStatus( ) == QPS_NOTINITIALISED )    ||
		 ( getStatus( ) == QPS_AUXILIARYQPSOLVED ) ||
		 ( getStatus( ) == QPS_HOMOTOPYQPSOLVED )  ||
 		 ( getStatus( ) == QPS_SOLVED )            )
	{
		return THROWERROR( RET_UNKNOWN_BUG );
	}

	/* some definitions */
	int nFR = getNFR( );
	int nAC = getNAC( );
	int nZ  = getNZ( );

	int tcol = sizeT - nAC;


	/* I) UPDATE INDICES */
	if ( bounds.moveFixedToFree( number ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_REMOVEBOUND_FAILED );


	int FR_idx[NVMAX];
	if ( bounds.getFree( )->getNumberArray( FR_idx ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_REMOVEBOUND_FAILED );

	/* I) APPEND <nFR+1>th UNITY VECOTR TO Q. */
	int nnFRp1 = FR_idx[nFR];
	for( i=0; i<nFR; ++i )
	{
		ii = FR_idx[i];
		Q[ii*NVMAX + nFR] = 0.0;
		Q[nnFRp1*NVMAX + i] = 0.0;
	}
	Q[nnFRp1*NVMAX + nFR] = 1.0;

	if ( nAC > 0 )
	{
		/* store new column a in a temporary vector instead of shifting T one column to the left and appending a */
		int AC_idx[NCMAX_ALLOC];
		if ( constraints.getActive( )->getNumberArray( AC_idx ) != SUCCESSFUL_RETURN )
			return THROWERROR( RET_REMOVEBOUND_FAILED );

		real_t tmp[NCMAX_ALLOC];
		for( i=0; i<nAC; ++i )
		{
			ii = AC_idx[i];
			tmp[i] =  A[ii*NVMAX + number];
		}


		/* II) RESTORE TRIANGULAR FORM OF T,
		 *     use column-wise Givens rotations to restore reverse triangular form
		 *     of T = [T A(:,number)], simultanenous change of Q (i.e. Y and Z). */
		real_t c, s;

		for( j=(nAC-1); j>=0; --j )
		{
			computeGivens( tmp[nAC-1-j],T[(nAC-1-j)*NVMAX + tcol+j],T[(nAC-1-j)*NVMAX + tcol+j],tmp[nAC-1-j],c,s );

			for( i=(nAC-j); i<nAC; ++i )
				applyGivens( c,s,tmp[i],T[i*NVMAX + tcol+j],T[i*NVMAX + tcol+j],tmp[i] );

			for( i=0; i<=nFR; ++i )
			{
				ii = FR_idx[i];
				/* nZ+1+nAC = nFR+1  /  nZ+(1) = nZ+1 */
				applyGivens( c,s,Q[ii*NVMAX + nZ+1+j],Q[ii*NVMAX + nZ+j],Q[ii*NVMAX + nZ+1+j],Q[ii*NVMAX + nZ+j] );
			}
		}
	}


	if ( ( updateCholesky == BT_TRUE ) && ( hessianType != HST_IDENTITY ) )
	{
		/* III) UPDATE CHOLESKY DECOMPOSITION,
		 *      calculate new additional column (i.e. [r sqrt(rho2)]')
		 *      of the Cholesky factor R: */
		real_t z2 = Q[nnFRp1*NVMAX + nZ];
		real_t rho2 = H[nnFRp1*NVMAX + nnFRp1]*z2*z2; /* rho2 = h2*z2*z2 */

		if ( nFR > 0 )
		{
			real_t Hz[NVMAX];
			for( i=0; i<nFR; ++i )
				Hz[i] = 0.0;
			/* 1) Calculate R'*r = Zfr'*Hfr*z1 + z2*Zfr'*h1 =: Zfr'*Hz + z2*Zfr'*h1 =: rhs and
			 *    rho2 = z1'*Hfr*z1 + 2*z2*h1'*z1 + h2*z2^2 - r'*r =: z1'*Hz + 2*z2*h1'*z1 + h2*z2^2 - r'r */
			for( j=0; j<nFR; ++j )
			{
				jj = FR_idx[j];
				for( i=0; i<nFR; ++i )
				{
					ii = FR_idx[i];
					/*			   H * z1 */
					Hz[i] += H[jj*NVMAX + ii] * Q[jj*NVMAX + nZ];
				}
			}

			if ( nZ > 0 )
			{
				real_t r[NVMAX];
				real_t rhs[NVMAX];
				for( i=0; i<nZ; ++i )
					rhs[i] = 0.0;

				/* 2) Calculate rhs. */
				for( j=0; j<nFR; ++j )
				{
					jj = FR_idx[j];
					for( i=0; i<nZ; ++i )
										/* Zfr' * ( Hz + z2*h1 ) */
						rhs[i] += Q[jj*NVMAX + i] * ( Hz[j] + z2 * H[nnFRp1*NVMAX + jj] );
				}

				/* 3) Calculate r = R^-T * rhs. */
				if ( backsolveR( rhs,BT_TRUE,BT_TRUE,r ) != SUCCESSFUL_RETURN )
					return THROWERROR( RET_REMOVEBOUND_FAILED );

				/* 4) Calculate rho2 = rho^2 = z'*Hz - r'*r
				 *    and store r into R. */
				for( i=0; i<nZ; ++i )
				{
					rho2 -= r[i]*r[i];
					R[i*NVMAX + nZ] = r[i];
				}
			}

			for( j=0; j<nFR; ++j )
			{
				jj = FR_idx[j];
							/* z1' * ( Hz + 2*z2*h1 ) */
				rho2 += Q[jj*NVMAX + nZ] * ( Hz[j] + 2.0*z2*H[nnFRp1*NVMAX + jj] );
			}
		}


		/* 5) Store rho into R. */
		if ( rho2 > 0.0 )
			R[nZ*NVMAX + nZ] = sqrt( rho2 );
		else
		{
			hessianType = HST_SEMIDEF;
			return THROWERROR( RET_HESSIAN_NOT_SPD );
		}
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	b a c k s o l v e R  (CODE DUPLICATE OF QProblemB CLASS!!!)
 */
returnValue QProblem::backsolveR(	const real_t* const b, BooleanType transposed,
									real_t* const a
									)
{
	/* Call standard backsolve procedure (i.e. removingBound == BT_FALSE). */
	return backsolveR( b,transposed,BT_FALSE,a );
}


/*
 *	b a c k s o l v e R  (CODE DUPLICATE OF QProblemB CLASS!!!)
 */
returnValue QProblem::backsolveR(	const real_t* const b, BooleanType transposed,
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
 *	b a c k s o l v e T
 */
returnValue QProblem::backsolveT( const real_t* const b, BooleanType transposed, real_t* const a )
{
	int i, j;
	int nT = getNAC( );
	int tcol = sizeT - nT;

	real_t sum;

	/* nothing to do */
	if ( nT <= 0 )
		return SUCCESSFUL_RETURN;


	/* Solve Ta = b, where T might be transposed. */
	if ( transposed == BT_FALSE )
	{
		/* solve Ta = b */
		for( i=0; i<nT; ++i )
		{
			sum = b[i];
			for( j=0; j<i; ++j )
				sum -= T[i*NVMAX + sizeT-1-j] * a[nT-1-j];

			if ( getAbs( T[i*NVMAX + sizeT-1-i] ) > ZERO )
				a[nT-1-i] = sum / T[i*NVMAX + sizeT-1-i];
			else
				return THROWERROR( RET_DIV_BY_ZERO );
		}
	}
	else
	{
		/* solve T^T*a = b */
		for( i=0; i<nT; ++i )
		{
			sum = b[i];
			for( j=0; j<i; ++j )
				sum -= T[(nT-1-j)*NVMAX + tcol+i] * a[nT-1-j];

			if ( getAbs( T[(nT-1-i)*NVMAX + tcol+i] ) > ZERO )
				a[nT-1-i] = sum / T[(nT-1-i)*NVMAX + tcol+i];
			else
				return THROWERROR( RET_DIV_BY_ZERO );
		}
	}


	return SUCCESSFUL_RETURN;
}


/*
 *	h o t s t a r t _ d e t e r m i n e D a t a S h i f t
 */
returnValue QProblem::hotstart_determineDataShift(  const int* const FX_idx, const int* const AC_idx,
													const real_t* const g_new, const real_t* const lbA_new, const real_t* const ubA_new,
													const real_t* const lb_new, const real_t* const ub_new,
													real_t* const delta_g, real_t* const delta_lbA, real_t* const delta_ubA,
													real_t* const delta_lb, real_t* const delta_ub,
													BooleanType& Delta_bC_isZero, BooleanType& Delta_bB_isZero
													)
{
	int i, ii;
	int nC  = getNC( );
	int nAC = getNAC( );


	/* I) DETERMINE DATA SHIFT FOR BOUNDS */
	QProblemB::hotstart_determineDataShift( FX_idx,g_new,lb_new,ub_new, delta_g,delta_lb,delta_ub, Delta_bB_isZero );


	/* II) DETERMINE DATA SHIFT FOR CONSTRAINTS */
	/* 1) Calculate shift directions. */
	for( i=0; i<nC; ++i )
	{
		/* if lower constraints' bounds do not exist, shift them to -infinity */
		if ( lbA_new != 0 )
			delta_lbA[i] = lbA_new[i] - lbA[i];
		else
			delta_lbA[i] = -INFTY - lbA[i];
	}

	for( i=0; i<nC; ++i )
	{
		/* if upper constraints' bounds do not exist, shift them to infinity */
		if ( ubA_new != 0 )
			delta_ubA[i] = ubA_new[i] - ubA[i];
		else
			delta_ubA[i] = INFTY - ubA[i];
	}

	/* 2) Determine if active constraints' bounds are to be shifted. */
	Delta_bC_isZero = BT_TRUE;

	for ( i=0; i<nAC; ++i )
	{
		ii = AC_idx[i];

		if ( ( getAbs( delta_lbA[ii] ) > EPS ) || ( getAbs( delta_ubA[ii] ) > EPS ) )
		{
			Delta_bC_isZero = BT_FALSE;
			break;
		}
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	h o t s t a r t _ d e t e r m i n e S t e p D i r e c t i o n
 */
returnValue QProblem::hotstart_determineStepDirection(	const int* const FR_idx, const int* const FX_idx, const int* const AC_idx,
														const real_t* const delta_g, const real_t* const delta_lbA, const real_t* const delta_ubA,
														const real_t* const delta_lb, const real_t* const delta_ub,
														BooleanType Delta_bC_isZero, BooleanType Delta_bB_isZero,
														real_t* const delta_xFX, real_t* const delta_xFR,
														real_t* const delta_yAC, real_t* const delta_yFX
														)
{
	int i, j, ii, jj;
	int nFR = getNFR( );
	int nFX = getNFX( );
	int nAC = getNAC( );
	int nZ  = getNZ( );

	/* initialise auxiliary vectors */
	real_t HMX_delta_xFX[NVMAX];
	real_t YFR_delta_xFRy[NVMAX];
	real_t ZFR_delta_xFRz[NVMAX];
	real_t HFR_YFR_delta_xFRy[NVMAX];
	for( i=0; i<nFR; ++i )
	{
		delta_xFR[i] = 0.0;
		HMX_delta_xFX[i] = 0.0;
		YFR_delta_xFRy[i] = 0.0;
		ZFR_delta_xFRz[i] = 0.0;
		HFR_YFR_delta_xFRy[i] = 0.0;
	}

	real_t delta_xFRy[NCMAX_ALLOC];
	real_t delta_xFRz[NVMAX];
	for( i=0; i<nZ; ++i )
		delta_xFRz[i] = 0.0;


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
		/* 1) Determine delta_xFRy. */
		if ( nAC > 0 )
		{
			if ( ( Delta_bC_isZero == BT_TRUE ) && ( Delta_bB_isZero == BT_TRUE ) )
			{
				for( i=0; i<nAC; ++i )
					delta_xFRy[i] = 0.0;

				for( i=0; i<nFR; ++i )
					delta_xFR[i] = 0.0;
			}
			else
			{
				/* auxillary variable */
				real_t delta_xFRy_TMP[NCMAX_ALLOC];

				for( i=0; i<nAC; ++i )
				{
					ii = AC_idx[i];

					if ( constraints.getStatus( ii ) == ST_LOWER )
						delta_xFRy_TMP[i] = delta_lbA[ii];
					else
						delta_xFRy_TMP[i] = delta_ubA[ii];

					if ( Delta_bB_isZero == BT_FALSE )
					{
						for( j=0; j<nFX; ++j )
						{
							jj = FX_idx[j];
							delta_xFRy_TMP[i] -= A[ii*NVMAX + jj] * delta_xFX[j];
						}
					}
				}

				if ( backsolveT( delta_xFRy_TMP, BT_FALSE, delta_xFRy ) != SUCCESSFUL_RETURN )
					return THROWERROR( RET_STEPDIRECTION_FAILED_TQ );

				for( i=0; i<nFR; ++i )
				{
					ii = FR_idx[i];
					for( j=0; j<nAC; ++j )
						YFR_delta_xFRy[i] += Q[ii*NVMAX + nZ+j] * delta_xFRy[j];

					/* delta_xFR = YFR*delta_xFRy (+ ZFR*delta_xFRz) */
					delta_xFR[i] = YFR_delta_xFRy[i];
				}
			}
		}

		/* 2) Determine delta_xFRz. */
		if ( hessianType == HST_IDENTITY )
		{
			for( j=0; j<nFR; ++j )
			{
				jj = FR_idx[j];
				for( i=0; i<nZ; ++i )
					delta_xFRz[i] -= Q[jj*NVMAX + i] * delta_g[jj];
			}

			if ( nZ > 0 )
			{
				for( i=0; i<nFR; ++i )
				{
					ii = FR_idx[i];
					for( j=0; j<nZ; ++j )
						ZFR_delta_xFRz[i] += Q[ii*NVMAX + j] * delta_xFRz[j];

					delta_xFR[i] += ZFR_delta_xFRz[i];
				}
			}
		}
		else
		{
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

			if ( nAC > 0 )
			{
				if ( ( Delta_bC_isZero == BT_FALSE ) || ( Delta_bB_isZero == BT_FALSE ) )
				{
					for( i=0; i<nFR; ++i )
					{
						ii = FR_idx[i];
						for( j=0; j<nFR; ++j )
						{
							jj = FR_idx[j];
							HFR_YFR_delta_xFRy[i] += H[ii*NVMAX + jj] * YFR_delta_xFRy[j];
						}
					}
				}
			}


			if ( nZ > 0 )
			{
				/* auxiliary variable */
				real_t delta_xFRz_TMP[NVMAX];
				real_t delta_xFRz_RHS[NVMAX];


				if ( ( nAC > 0 ) && ( nFX > 0 ) && ( Delta_bB_isZero == BT_FALSE ) )
				{
					for( j=0; j<nFR; ++j )
					{
						jj = FR_idx[j];
						delta_xFRz_RHS[j] = delta_g[jj] + HFR_YFR_delta_xFRy[j] + HMX_delta_xFX[j];
					}
				}
				else
				{
					if ( ( nAC == 0 ) && ( Delta_bB_isZero == BT_TRUE ) )
					{
						for( j=0; j<nFR; ++j )
						{
							jj = FR_idx[j];
							delta_xFRz_RHS[j] = delta_g[jj];
						}
					}
					else
					{
						if ( nAC > 0 ) /* => Delta_bB_isZero == BT_TRUE, as BT_FALSE would imply nFX>0 */
						{
							for( j=0; j<nFR; ++j )
							{
								jj = FR_idx[j];
								delta_xFRz_RHS[j] = delta_g[jj] + HFR_YFR_delta_xFRy[j];
							}
						}
						else /* Delta_bB_isZero == BT_FALSE, as nAC==0 */
						{
							for( j=0; j<nFR; ++j )
							{
								jj = FR_idx[j];
								delta_xFRz_RHS[j] = delta_g[jj] + HMX_delta_xFX[j];
							}
						}
					}
				}

				for( j=0; j<nFR; ++j )
				{
					jj = FR_idx[j];
					for( i=0; i<nZ; ++i )
						delta_xFRz[i] -= Q[jj*NVMAX + i] * delta_xFRz_RHS[j];
				}


				if ( backsolveR( delta_xFRz,BT_TRUE,delta_xFRz_TMP ) != SUCCESSFUL_RETURN )
					return THROWERROR( RET_STEPDIRECTION_FAILED_CHOLESKY );

				if ( backsolveR( delta_xFRz_TMP,BT_FALSE,delta_xFRz ) != SUCCESSFUL_RETURN )
					return THROWERROR( RET_STEPDIRECTION_FAILED_CHOLESKY );


				for( i=0; i<nFR; ++i )
				{
					ii = FR_idx[i];
					for( j=0; j<nZ; ++j )
						ZFR_delta_xFRz[i] += Q[ii*NVMAX + j] * delta_xFRz[j];

					delta_xFR[i] += ZFR_delta_xFRz[i];
				}
			}
		}
	}

	/* III) DETERMINE delta_yAC */
	if ( nAC > 0 ) /* => ( nFR = nZ + nAC > 0 ) */
	{
		/* auxiliary variables */
		real_t delta_yAC_TMP[NCMAX_ALLOC];
		for( i=0; i<nAC; ++i )
			delta_yAC_TMP[i] = 0.0;
		real_t delta_yAC_RHS[NVMAX];
		for( i=0; i<nFR; ++i )
			delta_yAC_RHS[i] = 0.0;

		if ( hessianType == HST_IDENTITY )
		{
			/* delta_yAC = (T')^-1 * ( Yfr*delta_gFR + delta_xFRy ) */
			for( j=0; j<nAC; ++j )
			{
				for( i=0; i<nFR; ++i )
				{
					ii = FR_idx[i];
					delta_yAC_TMP[j] += Q[ii*NVMAX + nZ+j] * delta_g[ii];
				}

				delta_yAC_TMP[j] += delta_xFRy[j];
			}
		}
		else
		{
			if ( ( Delta_bC_isZero == BT_TRUE ) && ( Delta_bB_isZero == BT_TRUE ) )
			{
				for( i=0; i<nFR; ++i )
				{
					ii = FR_idx[i];
					delta_yAC_RHS[i] = delta_g[ii];
				}
			}
			else
			{
				for( i=0; i<nFR; ++i )
				{
					ii = FR_idx[i];
					delta_yAC_RHS[i] = HFR_YFR_delta_xFRy[i] + delta_g[ii];
				}
			}

			if ( nZ > 0 )
			{
				for( i=0; i<nFR; ++i )
				{
					ii = FR_idx[i];
					for( j=0; j<nFR; ++j )
					{
						jj = FR_idx[j];
						delta_yAC_RHS[i] += H[ii*NVMAX + jj] * ZFR_delta_xFRz[j];
					}
				}
			}

			if ( nFX > 0 )
			{
				if ( Delta_bB_isZero == BT_FALSE )
				{
					for( i=0; i<nFR; ++i )
						delta_yAC_RHS[i] += HMX_delta_xFX[i];
				}
			}

			for( i=0; i<nAC; ++i)
			{
				for( j=0; j<nFR; ++j )
				{
					jj = FR_idx[j];
					delta_yAC_TMP[i] += Q[jj*NVMAX + nZ+i] * delta_yAC_RHS[j];
				}
			}
		}

		if ( backsolveT( delta_yAC_TMP,BT_TRUE,delta_yAC ) != SUCCESSFUL_RETURN )
			return THROWERROR( RET_STEPDIRECTION_FAILED_TQ );
	}


	/* IV) DETERMINE delta_yFX */
	if ( nFX > 0 )
	{
		for( i=0; i<nFX; ++i )
		{
			ii = FX_idx[i];

			delta_yFX[i] = delta_g[ii];

			for( j=0; j<nAC; ++j )
			{
				jj = AC_idx[j];
				delta_yFX[i] -= A[jj*NVMAX + ii] * delta_yAC[j];
			}

			if ( hessianType == HST_IDENTITY )
			{
				delta_yFX[i] += delta_xFX[i];
			}
			else
			{
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
			}
		}
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	h o t s t a r t _ d e t e r m i n e S t e p L e n g t h
 */
returnValue QProblem::hotstart_determineStepLength(	const int* const FR_idx, const int* const FX_idx, const int* const AC_idx, const int* const IAC_idx,
													const real_t* const delta_lbA, const real_t* const delta_ubA,
													const real_t* const delta_lb, const real_t* const delta_ub,
													const real_t* const delta_xFX, const real_t* const delta_xFR,
													const real_t* const delta_yAC, const real_t* const delta_yFX,
													real_t* const delta_Ax, int& BC_idx, SubjectToStatus& BC_status, BooleanType& BC_isBound
													)
{
	int i, j, ii, jj;
	int nV  = getNV( );
	int nC  = getNC( );
	int nFR = getNFR( );
	int nFX = getNFX( );
	int nAC = getNAC( );
	int nIAC = getNIAC( );

	/* initialise maximum steplength array */
	real_t maxStepLength[2*(NVMAX+NCMAX_ALLOC)];
	for ( i=0; i<2*(nV+nC); ++i )
		maxStepLength[i] = 1.0;


	/* I) DETERMINE MAXIMUM DUAL STEPLENGTH: */
	/* 1) Ensure that active dual constraints' bounds remain valid
	 *    (ignoring inequality constraints).  */
	for( i=0; i<nAC; ++i )
	{
		ii = AC_idx[i];

		if ( constraints.getType( ii ) != ST_EQUALITY )
		{
			if ( constraints.getStatus( ii ) == ST_LOWER )
			{
				/* active lower constraints' bounds */
				if ( delta_yAC[i] < -ZERO )
				{
					if ( y[nV+ii] > 0.0 )
						maxStepLength[nV+ii] = y[nV+ii] / ( -delta_yAC[i] );
					else
						maxStepLength[nV+ii] = 0.0;
				}
			}
			else
			{
				/* active upper constraints' bounds */
				if ( delta_yAC[i] > ZERO )
				{
					if ( y[nV+ii] < 0.0 )
						maxStepLength[nV+ii] = y[nV+ii] / ( -delta_yAC[i] );
					else
						maxStepLength[nV+ii] = 0.0;
				}
			}
		}
	}

	/* 2) Ensure that active dual bounds remain valid
	 *    (ignoring implicitly fixed variables). */
	for( i=0; i<nFX; ++i )
	{
		ii = FX_idx[i];

		if ( bounds.getType( ii ) != ST_EQUALITY )
		{
			if ( bounds.getStatus( ii ) == ST_LOWER )
			{
				/* active lower bounds */
				if ( delta_yFX[i] < -ZERO )
				{
					if ( y[ii] > 0.0 )
						maxStepLength[ii] = y[ii] / ( -delta_yFX[i] );
					else
						maxStepLength[ii] = 0.0;
				}
			}
			else
			{
				/* active upper bounds */
				if ( delta_yFX[i] > ZERO )
				{
					if ( y[ii] < 0.0 )
						maxStepLength[ii] = y[ii] / ( -delta_yFX[i] );
					else
						maxStepLength[ii] = 0.0;
				}
			}
		}
	}


	/* II) DETERMINE MAXIMUM PRIMAL STEPLENGTH */
	/* 1) Ensure that inactive constraints' bounds remain valid
	 *    (ignoring unbounded constraints). */
	real_t delta_x[NVMAX];
	for( j=0; j<nFR; ++j )
	{
		jj = FR_idx[j];
		delta_x[jj] = delta_xFR[j];
	}
	for( j=0; j<nFX; ++j )
	{
		jj = FX_idx[j];
		delta_x[jj] = delta_xFX[j];
	}

	for( i=0; i<nIAC; ++i )
	{
		ii = IAC_idx[i];

		if ( constraints.getType( ii ) != ST_UNBOUNDED )
		{
			delta_Ax[ii] = 0.0;
			for( j=0; j<nV; ++j )
				delta_Ax[ii] += A[ii*NVMAX + j] * delta_x[j]; // POSSIBLE SPEEDUP!

			/* inactive lower constraints' bounds */
			if ( constraints.isNoLower( ) == BT_FALSE )
			{
				if ( delta_lbA[ii] > delta_Ax[ii] )
				{
					if ( Ax[ii] > lbA[ii] )
						maxStepLength[nV+ii] = ( Ax[ii] - lbA[ii] ) / ( delta_lbA[ii] - delta_Ax[ii] );
					else
						maxStepLength[nV+ii] = 0.0;
				}
			}

			/* inactive upper constraints' bounds */
			if ( constraints.isNoUpper( ) == BT_FALSE )
			{
				if ( delta_ubA[ii] < delta_Ax[ii] )
				{
					if ( Ax[ii] < ubA[ii] )
						maxStepLength[nV+nC+nV+ii] = ( Ax[ii] - ubA[ii] ) / ( delta_ubA[ii] - delta_Ax[ii] );
					else
						maxStepLength[nV+nC+nV+ii] = 0.0;
				}
			}
		}
	}


	/* 2) Ensure that inactive bounds remain valid
	 *    (ignoring unbounded variables). */
	/* inactive lower bounds */
	if ( bounds.isNoLower( ) == BT_FALSE )
	{
		for( i=0; i<nFR; ++i )
		{
			ii = FR_idx[i];
			if ( bounds.getType( ii ) != ST_UNBOUNDED )
				if ( delta_lb[ii] > delta_xFR[i] )
				{
					if ( x[ii] > lb[ii] )
						maxStepLength[ii] = ( x[ii] - lb[ii] ) / ( delta_lb[ii] - delta_xFR[i] );
					else
						maxStepLength[ii] = 0.0;
				}
		}
	}

	/* inactive upper bounds */
	if ( bounds.isNoUpper( ) == BT_FALSE )
	{
		for( i=0; i<nFR; ++i )
		{
			ii = FR_idx[i];
			if ( bounds.getType( ii ) != ST_UNBOUNDED )
				if ( delta_ub[ii] < delta_xFR[i] )
				{
					if ( x[ii] < ub[ii] )
						maxStepLength[nV+nC+ii] = ( x[ii] - ub[ii] ) / ( delta_ub[ii] - delta_xFR[i] );
					else
						maxStepLength[nV+nC+ii] = 0.0;
				}
		}
	}


	/* III) DETERMINE MAXIMUM HOMOTOPY STEPLENGTH */
	real_t tau_new = 1.0;

	BC_idx = 0;
	BC_status = ST_UNDEFINED;
	BC_isBound = BT_FALSE;

	for ( i=0; i<nV; ++i )
	{
		/* 1) Consider lower/dual blocking bounds. */
		if ( maxStepLength[i] < tau_new )
		{
			tau_new = maxStepLength[i];
			BC_idx = i;
			BC_isBound = BT_TRUE;
			if ( bounds.getStatus( i ) == ST_INACTIVE ) /* inactive? */
				BC_status = ST_LOWER;
			else
				BC_status = ST_INACTIVE;
		}

		/* 2) Consider upper blocking bounds. */
		if ( maxStepLength[nV+nC+i] < tau_new )
		{
			tau_new = maxStepLength[nV+nC+i];
			BC_idx = i;
			BC_isBound = BT_TRUE;
			BC_status = ST_UPPER;
		}
	}

	for ( i=nV; i<nV+nC; ++i )
	{
		/* 3) Consider lower/dual blocking constraints. */
		if ( maxStepLength[i] < tau_new )
		{
			tau_new = maxStepLength[i];
			BC_idx = i-nV;
			BC_isBound = BT_FALSE;
			if ( constraints.getStatus( i-nV ) == ST_INACTIVE ) /* inactive? */
				BC_status = ST_LOWER;
			else
				BC_status = ST_INACTIVE;
		}

		/* 4) Consider upper blocking constraints. */
		if ( maxStepLength[nV+nC+i] < tau_new )
		{
			tau_new = maxStepLength[nV+nC+i];
			BC_idx = i-nV;
			BC_isBound = BT_FALSE;
			BC_status = ST_UPPER;
		}
	}


	/* IV) CLEAR CYCLING DATA
	 *     if a positive step can be taken */
	if ( tau_new > EPS )
		cyclingManager.clearCyclingData( );


	/* V) SET MAXIMUM HOMOTOPY STEPLENGTH */
	tau = tau_new;

	#ifdef PC_DEBUG
	if ( printlevel == PL_HIGH )
	{

	 	char messageString[80];

		if ( BC_status == ST_UNDEFINED )
			sprintf( messageString,"Stepsize is %.6e!",tau );
		else
			sprintf( messageString,"Stepsize is %.6e! (BC_idx = %d, BC_isBound = %d, BC_status = %d)",tau,BC_idx,BC_isBound,BC_status );

		getGlobalMessageHandler( )->throwInfo( RET_STEPSIZE_NONPOSITIVE,messageString,__FUNCTION__,__FILE__,__LINE__,VS_VISIBLE );
	}
	#endif

	return SUCCESSFUL_RETURN;
}


/*
 *	h o t s t a r t _ p e r f o r m S t e p
 */
returnValue QProblem::hotstart_performStep(	const int* const FR_idx, const int* const FX_idx, const int* const AC_idx, const int* const IAC_idx,
											const real_t* const delta_g, const real_t* const delta_lbA, const real_t* const delta_ubA,
											const real_t* const delta_lb, const real_t* const delta_ub,
											const real_t* const delta_xFX, const real_t* const delta_xFR,
											const real_t* const delta_yAC, const real_t* const delta_yFX,
											const real_t* const delta_Ax, int BC_idx, SubjectToStatus BC_status, BooleanType BC_isBound
											)
{
	int i, j, ii;
	int nV  = getNV( );
	int nC  = getNC( );
	int nFR = getNFR( );
	int nFX = getNFX( );
	int nAC = getNAC( );
	int nIAC = getNIAC( );


	/* I) CHECK (CONSTRAINTS') BOUNDS' CONSISTENCY */
	if ( areBoundsConsistent( delta_lb,delta_ub,delta_lbA,delta_ubA ) == BT_FALSE )
	{
		infeasible = BT_TRUE;
		tau = 0.0;

		return THROWERROR( RET_QP_INFEASIBLE );
	}


	/* II) GO TO ACTIVE SET CHANGE */
	if ( tau > ZERO )
	{
		/* 1) Perform step in primal und dual space... */
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

		for( i=0; i<nAC; ++i )
		{
			ii = AC_idx[i];
			y[nV+ii] += tau*delta_yAC[i];
		}

		/* ... also for Ax. */
		for( i=0; i<nIAC; ++i )
		{
			ii = IAC_idx[i];
			if ( constraints.getType( ii ) != ST_UNBOUNDED )
				Ax[ii] += tau*delta_Ax[ii];
		}
		for( i=0; i<nAC; ++i )
		{
			ii = AC_idx[i];

			Ax[ii] = 0.0;
			for( j=0; j<nV; ++j )
				Ax[ii] += A[ii*NVMAX + j] * x[j];
		}

		/* 2) Shift QP data. */
		for( i=0; i<nV; ++i )
		{
			g[i]  += tau*delta_g[i];
			lb[i] += tau*delta_lb[i];
			ub[i] += tau*delta_ub[i];
		}

		for( i=0; i<nC; ++i )
		{
			lbA[i] += tau*delta_lbA[i];
			ubA[i] += tau*delta_ubA[i];
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
			if ( BC_isBound == BT_TRUE )
			{
				#ifdef PC_DEBUG
				sprintf( messageString,"bound no. %d.", BC_idx );
				getGlobalMessageHandler( )->throwInfo( RET_REMOVE_FROM_ACTIVESET,messageString,__FUNCTION__,__FILE__,__LINE__,visibilityStatus );
				#endif

				if ( removeBound( BC_idx,BT_TRUE ) != SUCCESSFUL_RETURN )
					return THROWERROR( RET_REMOVE_FROM_ACTIVESET_FAILED );

				y[BC_idx] = 0.0;
			}
			else
			{
				#ifdef PC_DEBUG
				sprintf( messageString,"constraint no. %d.", BC_idx );
				getGlobalMessageHandler( )->throwInfo( RET_REMOVE_FROM_ACTIVESET,messageString,__FUNCTION__,__FILE__,__LINE__,visibilityStatus );
				#endif

				if ( removeConstraint( BC_idx,BT_TRUE ) != SUCCESSFUL_RETURN )
					return THROWERROR( RET_REMOVE_FROM_ACTIVESET_FAILED );

				y[nV+BC_idx] = 0.0;
			}
			break;


		/* Add one variable to active set. */
		default:
			if ( BC_isBound == BT_TRUE )
			{
				#ifdef PC_DEBUG
				if ( BC_status == ST_LOWER )
					sprintf( messageString,"lower bound no. %d.", BC_idx );
				else
					sprintf( messageString,"upper bound no. %d.", BC_idx );
				getGlobalMessageHandler( )->throwInfo( RET_ADD_TO_ACTIVESET,messageString,__FUNCTION__,__FILE__,__LINE__,visibilityStatus );
				#endif

				if ( addBound( BC_idx,BC_status,BT_TRUE ) != SUCCESSFUL_RETURN )
					return THROWERROR( RET_ADD_TO_ACTIVESET_FAILED );
			}
			else
			{
				#ifdef PC_DEBUG
				if ( BC_status == ST_LOWER )
					sprintf( messageString,"lower constraint's bound no. %d.", BC_idx );
				else
					sprintf( messageString,"upper constraint's bound no. %d.", BC_idx );
				getGlobalMessageHandler( )->throwInfo( RET_ADD_TO_ACTIVESET,messageString,__FUNCTION__,__FILE__,__LINE__,visibilityStatus );
				#endif

				if ( addConstraint( BC_idx,BC_status,BT_TRUE ) != SUCCESSFUL_RETURN )
					return THROWERROR( RET_ADD_TO_ACTIVESET_FAILED );
			}
			break;
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	a r e B o u n d s C o n s i s t e n t
 */
BooleanType QProblem::areBoundsConsistent(	const real_t* const delta_lb, const real_t* const delta_ub,
											const real_t* const delta_lbA, const real_t* const delta_ubA
											) const
{
	int i;

	/* 1) Check bounds' consistency. */
	if ( QProblemB::areBoundsConsistent( delta_lb,delta_ub ) == BT_FALSE )
		return BT_FALSE;

	/* 2) Check constraints' consistency, i.e.
	 *    check if delta_lb[i] is greater than delta_ub[i]
	 *    for a component i whose bounds are already (numerically) equal. */
	for( i=0; i<getNC( ); ++i )
		if ( ( lbA[i] > ubA[i] - BOUNDTOL ) && ( delta_lbA[i] > delta_ubA[i] + EPS ) )
			return BT_FALSE;

	return BT_TRUE;
}


/*
 *	s e t u p Q P d a t a
 */
returnValue QProblem::setupQPdata(	const real_t* const _H, const real_t* const _R, const real_t* const _g, const real_t* const _A,
									const real_t* const _lb, const real_t* const _ub,
									const real_t* const _lbA, const real_t* const _ubA
									)
{
	int i, j;
	int nV = getNV( );
	int nC = getNC( );


	/* 1) Load Hessian matrix as well as lower and upper bounds vectors. */
	if (QProblemB::setupQPdata(_H, _R, _g, _lb, _ub) != SUCCESSFUL_RETURN)
		return THROWERROR( RET_INVALID_ARGUMENTS );

	/* 2) Load constraint matrix. */
	if ( ( nC > 0 ) && ( _A == 0 ) )
		return THROWERROR( RET_INVALID_ARGUMENTS );

	if ( nC > 0 )
	{
		for( i=0; i<nC; ++i )
			for( j=0; j<nV; ++j )
				A[i*NVMAX + j] = _A[i*nV + j];

		/* 3) Setup lower constraints' bounds vector. */
		if ( _lbA != 0 )
		{
			for( i=0; i<nC; ++i )
				lbA[i] = _lbA[i];
		}
		else
		{
			/* if no lower constraints' bounds are specified, set them to -infinity */
			for( i=0; i<nC; ++i )
				lbA[i] = -INFTY;
		}

		/* 4) Setup upper constraints' bounds vector. */
		if ( _ubA != 0 )
		{
			for( i=0; i<nC; ++i )
				ubA[i] = _ubA[i];
		}
		else
		{
			/* if no upper constraints' bounds are specified, set them to infinity */
			for( i=0; i<nC; ++i )
				ubA[i] = INFTY;
		}
	}

// 	printmatrix2( "A",A,10,20 );
	
// 	printmatrix2( "lbA",lbA,1,nC );
// 	printmatrix2( "ubA",ubA,1,nC );

	return SUCCESSFUL_RETURN;
}



#ifdef PC_DEBUG  /* Define print functions only for debugging! */

/*
 *	p r i n t I t e r a t i o n
 */
returnValue QProblem::printIteration( 	int iteration,
										int BC_idx,	SubjectToStatus BC_status, BooleanType BC_isBound
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

		sprintf( myPrintfString,"  Iter  |  StepLength   |     Info      |  nFX  |  nAC  \n" );
		myPrintf( myPrintfString );
	}

	/* 2) Print iteration line. */
	if ( BC_status == ST_UNDEFINED )
	{
		sprintf( myPrintfString,"  %4.1d  |  %1.5e  |   QP SOLVED   | %4.1d  | %4.1d  \n", iteration,tau,getNFX( ),getNAC( ) );
		myPrintf( myPrintfString );
	}
	else
	{
		char info[8];

		if ( BC_status == ST_INACTIVE )
			sprintf( info,"REM " );
		else
			sprintf( info,"ADD " );

		if ( BC_isBound == BT_TRUE )
			sprintf( &(info[4]),"BND" );
		else
			sprintf( &(info[4]),"CON" );

		sprintf( myPrintfString,"  %4.1d  |  %1.5e  |  %s%4.1d  | %4.1d  | %4.1d  \n", iteration,tau,info,BC_idx,getNFX( ),getNAC( ) );
		myPrintf( myPrintfString );
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	p r i n t I t e r a t i o n
 */
returnValue QProblem::printIteration( 	int iteration,
										int BC_idx,	SubjectToStatus BC_status
		  								)
{
	return printIteration( iteration,BC_idx,BC_status,BT_TRUE );
}

#endif  /* PC_DEBUG */



/*
 *	c h e c k K K T c o n d i t i o n s
 */
returnValue QProblem::checkKKTconditions( )
{
	#ifdef __PERFORM_KKT_TEST__

	int i, j, jj;
	int nV  = getNV( );
	int nC  = getNC( );
	int nAC = getNAC( );

	real_t tmp;
	real_t maxKKTviolation = 0.0;

	int AC_idx[NCMAX_ALLOC];
	constraints.getActive( )->getNumberArray( AC_idx );

	/* 1) check for Hx + g - [yFX yAC]*[Id A]' = 0. */
	for( i=0; i<nV; ++i )
	{
		tmp = g[i];

		for( j=0; j<nV; ++j )
			tmp += H[i*NVMAX + j] * x[j];

		tmp -= y[i];

		/* Only sum over active constraints as y is zero for all inactive ones. */
		for( j=0; j<nAC; ++j )
		{
			jj = AC_idx[j];
			tmp -= A[jj*NVMAX + i] * y[nV+jj];
		}

		if ( getAbs( tmp ) > maxKKTviolation )
			maxKKTviolation = getAbs( tmp );
	}

	/* 2) Check for [lb lbA] <= [Id A]*x <= [ub ubA]. */
	/* lbA <= Ax <= ubA */
	for( i=0; i<nC; ++i )
	{
		if ( lbA[i] - Ax[i] > maxKKTviolation )
			maxKKTviolation = lbA[i] - Ax[i];

		if ( Ax[i] - ubA[i] > maxKKTviolation )
			maxKKTviolation = Ax[i] - ubA[i];
	}

	/* lb <= x <= ub */
	for( i=0; i<nV; ++i )
	{
		if ( lb[i] - x[i] > maxKKTviolation )
			maxKKTviolation = lb[i] - x[i];

		if ( x[i] - ub[i] > maxKKTviolation )
			maxKKTviolation = x[i] - ub[i];
	}

	/* 3) Check for correct sign of y and for complementary slackness. */
	/* bounds */
	for( i=0; i<nV; ++i )
	{
		switch ( bounds.getStatus( i ) )
		{
			case ST_LOWER:
				if ( -y[i] > maxKKTviolation )
					maxKKTviolation = -y[i];
				if ( getAbs( x[i] - lb[i] ) > maxKKTviolation )
					maxKKTviolation = getAbs( x[i] - lb[i] );
				break;

			case ST_UPPER:
				if ( y[i] > maxKKTviolation )
					maxKKTviolation = y[i];
				if ( getAbs( ub[i] - x[i] ) > maxKKTviolation )
					maxKKTviolation = getAbs( ub[i] - x[i] );
				break;

			default: /* inactive */
			if ( getAbs( y[i] ) > maxKKTviolation )
					maxKKTviolation = getAbs( y[i] );
				break;
		}
	}

	/* constraints */
	for( i=0; i<nC; ++i )
	{
		switch ( constraints.getStatus( i ) )
		{
			case ST_LOWER:
				if ( -y[nV+i] > maxKKTviolation )
					maxKKTviolation = -y[nV+i];
				if ( getAbs( Ax[i] - lbA[i] ) > maxKKTviolation )
					maxKKTviolation = getAbs( Ax[i] - lbA[i] );
				break;

			case ST_UPPER:
				if ( y[nV+i] > maxKKTviolation )
					maxKKTviolation = y[nV+i];
				if ( getAbs( ubA[i] - Ax[i] ) > maxKKTviolation )
					maxKKTviolation = getAbs( ubA[i] - Ax[i] );
				break;

			default: /* inactive */
			if ( getAbs( y[nV+i] ) > maxKKTviolation )
					maxKKTviolation = getAbs( y[nV+i] );
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
