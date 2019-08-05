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
 *	\file SRC/Bounds.cpp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Implementation of the Bounds class designed to manage working sets of
 *	bounds within a QProblem.
 */


#include <Bounds.hpp>



/*****************************************************************************
 *  P U B L I C                                                              *
 *****************************************************************************/


/*
 *	B o u n d s
 */
Bounds::Bounds( ) :	SubjectTo( ),
					nV( 0 ),
					nFV( 0 ),
					nBV( 0 ),
					nUV( 0 )
{
}


/*
 *	B o u n d s
 */
Bounds::Bounds( const Bounds& rhs ) :	SubjectTo( rhs ),
										nV( rhs.nV ),
										nFV( rhs.nFV ),
										nBV( rhs.nBV ),
										nUV( rhs.nUV )
{
	free  = rhs.free;
	fixed = rhs.fixed;
}


/*
 *	~ B o u n d s
 */
Bounds::~Bounds( )
{
}


/*
 *	o p e r a t o r =
 */
Bounds& Bounds::operator=( const Bounds& rhs )
{
	if ( this != &rhs )
	{
		SubjectTo::operator=( rhs );

		nV  = rhs.nV;
		nFV = rhs.nFV;
		nBV = rhs.nBV;
		nUV = rhs.nUV;

		free  = rhs.free;
		fixed = rhs.fixed;
	}

	return *this;
}


/*
 *	i n i t
 */
returnValue Bounds::init( int n )
{
	nV = n;
	nFV = 0;
	nBV = 0;
	nUV = 0;

	free.init( );
	fixed.init( );

	return SubjectTo::init( n );
}


/*
 *	s e t u p B o u n d
 */
returnValue Bounds::setupBound(	int _number, SubjectToStatus _status
								)
{
	/* consistency check */
	if ( ( _number < 0 ) || ( _number >= getNV( ) ) )
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );
	
	/* Add bound index to respective index list. */
	switch ( _status )
	{
		case ST_INACTIVE:
			if ( this->addIndex( this->getFree( ),_number,_status ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_SETUP_BOUND_FAILED );
			break;

		case ST_LOWER:
			if ( this->addIndex( this->getFixed( ),_number,_status ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_SETUP_BOUND_FAILED );
			break;

		case ST_UPPER:
			if ( this->addIndex( this->getFixed( ),_number,_status ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_SETUP_BOUND_FAILED );
			break;

		default:
			return THROWERROR( RET_INVALID_ARGUMENTS );
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t u p A l l F r e e
 */
returnValue Bounds::setupAllFree( )
{
	int i;

	/* 1) Place unbounded variables at the beginning of the index list of free variables. */
	for( i=0; i<nV; ++i )
	{
		if ( getType( i ) == ST_UNBOUNDED )
		{
			if ( setupBound( i,ST_INACTIVE ) != SUCCESSFUL_RETURN )
					return THROWERROR( RET_SETUP_BOUND_FAILED );
		}
	}

	/* 2) Add remaining (i.e. bounded but possibly free) variables to the index list of free variables. */
	for( i=0; i<nV; ++i )
	{
		if ( getType( i ) == ST_BOUNDED )
		{
			if ( setupBound( i,ST_INACTIVE ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_SETUP_BOUND_FAILED );
		}
	}

	/* 3) Place implicitly fixed variables at the end of the index list of free variables. */
	for( i=0; i<nV; ++i )
	{
		if ( getType( i ) == ST_EQUALITY )
		{
			if ( setupBound( i,ST_INACTIVE ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_SETUP_BOUND_FAILED );
		}
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	m o v e F i x e d T o F r e e
 */
returnValue Bounds::moveFixedToFree( int _number )
{
	/* consistency check */
	if ( ( _number < 0 ) || ( _number >= getNV( ) ) )
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );

	/* Move index from indexlist of fixed variables to that of free ones. */
	if ( this->removeIndex( this->getFixed( ),_number ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_MOVING_BOUND_FAILED );

	if ( this->addIndex( this->getFree( ),_number,ST_INACTIVE ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_MOVING_BOUND_FAILED );

	return SUCCESSFUL_RETURN;
}


/*
 *	m o v e F r e e T o F i x e d
 */
returnValue Bounds::moveFreeToFixed(	int _number, SubjectToStatus _status
										)
{
	/* consistency check */
	if ( ( _number < 0 ) || ( _number >= getNV( ) ) )
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );

	/* Move index from indexlist of free variables to that of fixed ones. */
	if ( this->removeIndex( this->getFree( ),_number ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_MOVING_BOUND_FAILED );

	if ( this->addIndex( this->getFixed( ),_number,_status ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_MOVING_BOUND_FAILED );

	return SUCCESSFUL_RETURN;
}


/*
 *	s w a p F r e e
 */
returnValue Bounds::swapFree(	int number1, int number2
								)
{
	/* consistency check */
	if ( ( number1 < 0 ) || ( number1 >= getNV( ) ) || ( number2 < 0 ) || ( number2 >= getNV( ) ) )
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );

	/* Swap index within indexlist of free variables. */
	return this->swapIndex( this->getFree( ),number1,number2 );
}


/*
 *	end of file
 */
