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
 *	\file SRC/QProblem.ipp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Implementation of inlined member functions of the QProblem class which 
 *	is able to use the newly developed online active set strategy for 
 *	parametric quadratic programming.
 */



/*****************************************************************************
 *  P U B L I C                                                              *
 *****************************************************************************/

/*
 *	g e t A
 */
inline returnValue QProblem::getA( real_t* const _A ) const
{
	int i;

	for ( i=0; i<getNV( )*getNC( ); ++i )
		_A[i] = A[i];

	return SUCCESSFUL_RETURN;
}


/*
 *	g e t A
 */
inline returnValue QProblem::getA( int number, real_t* const row ) const
{
	int nV = getNV( );
		
	if ( ( number >= 0 ) && ( number < getNC( ) ) )
	{
		for ( int i=0; i<nV; ++i )
			row[i] = A[number*NVMAX + i];

		return SUCCESSFUL_RETURN;
	}
	else
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );
}


/*
 *	g e t L B A
 */
inline returnValue QProblem::getLBA( real_t* const _lbA ) const
{
	int i;

	for ( i=0; i<getNC( ); ++i )
		_lbA[i] = lbA[i];

	return SUCCESSFUL_RETURN;
}


/*
 *	g e t L B A
 */
inline returnValue QProblem::getLBA( int number, real_t& value ) const
{
	if ( ( number >= 0 ) && ( number < getNC( ) ) )
	{
		value = lbA[number];
		return SUCCESSFUL_RETURN;
	}
	else
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );
}


/*
 *	g e t U B A
 */
inline returnValue QProblem::getUBA( real_t* const _ubA ) const
{
	int i;

	for ( i=0; i<getNC( ); ++i )
		_ubA[i] = ubA[i];

	return SUCCESSFUL_RETURN;
}


/*
 *	g e t U B A
 */
inline returnValue QProblem::getUBA( int number, real_t& value ) const
{
	if ( ( number >= 0 ) && ( number < getNC( ) ) )
	{
		value = ubA[number];
		return SUCCESSFUL_RETURN;
	}
	else
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );
}


/*
 *	g e t C o n s t r a i n t s
 */
inline returnValue QProblem::getConstraints( Constraints* const _constraints ) const
{
	*_constraints = constraints;
	
	return SUCCESSFUL_RETURN;
}



/*
 *	g e t N C
 */
inline int QProblem::getNC( ) const
{
	return constraints.getNC( );
}


/*
 *	g e t N E C
 */
inline int QProblem::getNEC( ) const
{
	return constraints.getNEC( );
}


/*
 *	g e t N A C
 */
inline int QProblem::getNAC( )
{
	return constraints.getNAC( );
}


/*
 *	g e t N I A C
 */
inline int QProblem::getNIAC( )
{
	return constraints.getNIAC( );
}



/*****************************************************************************
 *  P R O T E C T E D                                                        *
 *****************************************************************************/
 

/*
 *	s e t A
 */
inline returnValue QProblem::setA( const real_t* const A_new )
{
	int i, j;
	int nV = getNV( );
	int nC = getNC( );

	/* Set constraint matrix AND update member AX. */
	for( j=0; j<nC; ++j )
	{
		Ax[j] = 0.0;

		for( i=0; i<nV; ++i )
		{	
			A[j*NVMAX + i] = A_new[j*nV + i];
			Ax[j] += A[j*NVMAX + i] * x[i];
		}
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t A
 */
inline returnValue QProblem::setA( int number, const real_t* const row )
{
	int i;
	int nV = getNV( );

	/* Set constraint matrix AND update member AX. */
	if ( ( number >= 0 ) && ( number < getNC( ) ) )
	{
		Ax[number] = 0.0;

		for( i=0; i<nV; ++i )
		{
			A[number*NVMAX + i] = row[i];
			Ax[number] += A[number*NVMAX + i] * x[i];
		}

		return SUCCESSFUL_RETURN;
	}
	else
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );
}


/*
 *	s e t L B A
 */
inline returnValue QProblem::setLBA( const real_t* const lbA_new )
{
	int i;
	int nC = getNC();

	for( i=0; i<nC; ++i )
		lbA[i] = lbA_new[i];

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t L B A
 */
inline returnValue QProblem::setLBA( int number, real_t value )
{
	if ( ( number >= 0 ) && ( number < getNC( ) ) )
	{
		lbA[number] = value;
		return SUCCESSFUL_RETURN;
	}
	else
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );
}


/*
 *	s e t U B A
 */
inline returnValue QProblem::setUBA( const real_t* const ubA_new )
{
	int i;
	int nC = getNC();

	for( i=0; i<nC; ++i )
		ubA[i] = ubA_new[i];

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t U B A
 */
inline returnValue QProblem::setUBA( int number, real_t value )
{
	if ( ( number >= 0 ) && ( number < getNC( ) ) )
	{
		ubA[number] = value;
		return SUCCESSFUL_RETURN;
	}
	else
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );
}


/*
 *	end of file
 */
