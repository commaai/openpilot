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
 *	\file SRC/QProblemB.ipp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Implementation of inlined member functions of the QProblemB class which 
 *	is able to use the newly developed online active set strategy for 
 *	parametric quadratic programming.
 */



#include <math.h>



/*****************************************************************************
 *  P U B L I C                                                              *
 *****************************************************************************/

/*
 *	g e t H
 */
inline returnValue QProblemB::getH( real_t* const _H ) const
{
	int i;

	for ( i=0; i<getNV( )*getNV( ); ++i )
		_H[i] = H[i];

	return SUCCESSFUL_RETURN;
}


/*
 *	g e t G
 */
inline returnValue QProblemB::getG( real_t* const _g ) const
{
	int i;

	for ( i=0; i<getNV( ); ++i )
		_g[i] = g[i];

	return SUCCESSFUL_RETURN;
}


/*
 *	g e t L B
 */
inline returnValue QProblemB::getLB( real_t* const _lb ) const
{
	int i;

	for ( i=0; i<getNV( ); ++i )
		_lb[i] = lb[i];

	return SUCCESSFUL_RETURN;
}


/*
 *	g e t L B
 */
inline returnValue QProblemB::getLB( int number, real_t& value ) const
{
	if ( ( number >= 0 ) && ( number < getNV( ) ) )
	{
		value = lb[number];
		return SUCCESSFUL_RETURN;
	}
	else
	{
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );
	}
}


/*
 *	g e t U B
 */
inline returnValue QProblemB::getUB( real_t* const _ub ) const
{
	int i;

	for ( i=0; i<getNV( ); ++i )
		_ub[i] = ub[i];

	return SUCCESSFUL_RETURN;
}


/*
 *	g e t U B
 */
inline returnValue QProblemB::getUB( int number, real_t& value ) const
{
	if ( ( number >= 0 ) && ( number < getNV( ) ) )
	{
		value = ub[number];
		return SUCCESSFUL_RETURN;
	}
	else
	{
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );
	}
}


/*
 *	g e t B o u n d s
 */
inline returnValue QProblemB::getBounds( Bounds* const _bounds ) const
{
	*_bounds = bounds;
	
	return SUCCESSFUL_RETURN;
}


/*
 *	g e t N V
 */
inline int QProblemB::getNV( ) const
{
	return bounds.getNV( );
}


/*
 *	g e t N F R
 */
inline int QProblemB::getNFR( )
{
	return bounds.getNFR( );
}


/*
 *	g e t N F X
 */
inline int QProblemB::getNFX( )
{
	return bounds.getNFX( );
}


/*
 *	g e t N F V
 */
inline int QProblemB::getNFV( ) const
{
	return bounds.getNFV( );
}


/*
 *	g e t S t a t u s
 */
inline QProblemStatus QProblemB::getStatus( ) const
{
	return status;
}


/*
 *	i s I n i t i a l i s e d
 */
inline BooleanType QProblemB::isInitialised( ) const
{
	if ( status == QPS_NOTINITIALISED )
		return BT_FALSE;
	else
		return BT_TRUE;
}


/*
 *	i s S o l v e d
 */
inline BooleanType QProblemB::isSolved( ) const
{
	if ( status == QPS_SOLVED )
		return BT_TRUE;
	else
		return BT_FALSE;
}


/*
 *	i s I n f e a s i b l e
 */
inline BooleanType QProblemB::isInfeasible( ) const
{
	return infeasible;
}


/*
 *	i s U n b o u n d e d
 */
inline BooleanType QProblemB::isUnbounded( ) const
{
	return unbounded;
}


/*
 *	g e t P r i n t L e v e l
 */
inline PrintLevel QProblemB::getPrintLevel( ) const
{
	return printlevel;
}


/*
 *	g e t H e s s i a n T y p e
 */
inline HessianType QProblemB::getHessianType( ) const
{
	return hessianType;
}


/*
 *	s e t H e s s i a n T y p e
 */
inline returnValue QProblemB::setHessianType( HessianType _hessianType )
{
	hessianType = _hessianType;
	return SUCCESSFUL_RETURN;
}



/*****************************************************************************
 *  P R O T E C T E D                                                        *
 *****************************************************************************/

/*
 *	s e t H
 */
inline returnValue QProblemB::setH( const real_t* const H_new )
{
	int i, j;

	int nV = getNV();

	for( i=0; i<nV; ++i )
		for( j=0; j<nV; ++j )
			H[i*NVMAX + j] = H_new[i*nV + j];

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t G
 */
inline returnValue QProblemB::setG( const real_t* const g_new )
{
	int i;

	int nV = getNV();

	for( i=0; i<nV; ++i )
		g[i] = g_new[i];

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t L B
 */
inline returnValue QProblemB::setLB( const real_t* const lb_new )
{
	int i;

	int nV = getNV();

	for( i=0; i<nV; ++i )
		lb[i] = lb_new[i];

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t L B
 */
inline returnValue QProblemB::setLB( int number, real_t value )
{
	if ( ( number >= 0 ) && ( number < getNV( ) ) )
	{
		lb[number] = value;
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
inline returnValue QProblemB::setUB( const real_t* const ub_new )
{
	int i;

	int nV = getNV();

	for( i=0; i<nV; ++i )
		ub[i] = ub_new[i];

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t U B
 */
inline returnValue QProblemB::setUB( int number, real_t value )
{
	if ( ( number >= 0 ) && ( number < getNV( ) ) )
	{
		ub[number] = value;

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
inline void QProblemB::computeGivens(	real_t xold, real_t yold, real_t& xnew, real_t& ynew,
										real_t& c, real_t& s 
										) const
{
    if ( getAbs( yold ) <= ZERO )
	{
        c = 1.0;
        s = 0.0;
		
		xnew = xold;
		ynew = yold;
	}
    else
	{
		real_t t, mu;

        mu = getAbs( xold );
		if ( getAbs( yold ) > mu )
			mu = getAbs( yold );
		
        t = mu * sqrt( (xold/mu)*(xold/mu) + (yold/mu)*(yold/mu) );
		
		if ( xold < 0.0 )
            t = -t;
		
        c = xold/t;
        s = yold/t;
        xnew = t;
        ynew = 0.0;
	}
	
	return;
}

		
/*
 *	a p p l y G i v e n s
 */
inline void QProblemB::applyGivens(	real_t c, real_t s, real_t xold, real_t yold,
									real_t& xnew, real_t& ynew 
									) const
{
	/* Usual Givens plane rotation requiring four multiplications. */
	xnew =  c*xold + s*yold;
	ynew = -s*xold + c*yold;
// 	double nu = s/(1.0+c);
// 
// 	xnew = xold*c + yold*s;
// 	ynew = (xnew+xold)*nu - yold;
	
	return;
}


/*
 *	end of file
 */
