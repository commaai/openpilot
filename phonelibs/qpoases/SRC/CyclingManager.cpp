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
 *	\file SRC/CyclingManager.cpp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Implementation of the CyclingManager class designed to detect
 *	and handle possible cycling during QP iterations.
 *
 */


#include <CyclingManager.hpp>


/*****************************************************************************
 *  P U B L I C                                                              *
 *****************************************************************************/


/*
 *	C y c l i n g M a n a g e r
 */
CyclingManager::CyclingManager( ) :	nV( 0 ),
									nC( 0 )
{
	cyclingDetected = BT_FALSE;
}


/*
 *	C y c l i n g M a n a g e r
 */
CyclingManager::CyclingManager( const CyclingManager& rhs ) :	nV( rhs.nV ),
																nC( rhs.nC ),
																cyclingDetected( rhs.cyclingDetected )
{
	int i;

	for( i=0; i<nV+nC; ++i )
		status[i] = rhs.status[i];
}


/*
 *	~ C y c l i n g M a n a g e r
 */
CyclingManager::~CyclingManager( )
{
}


/*
 *	o p e r a t o r =
 */
CyclingManager& CyclingManager::operator=( const CyclingManager& rhs )
{
	int i;

	if ( this != &rhs )
	{
		nV = rhs.nV;
		nC = rhs.nC;

		for( i=0; i<nV+nC; ++i )
			status[i] = rhs.status[i];

		cyclingDetected = rhs.cyclingDetected;
	}

	return *this;
}



/*
 *	i n i t
 */
returnValue CyclingManager::init( int _nV, int _nC )
{
	nV = _nV;
	nC = _nC;

	cyclingDetected = BT_FALSE;

	return SUCCESSFUL_RETURN;
}



/*
 *	s e t C y c l i n g S t a t u s
 */
returnValue CyclingManager::setCyclingStatus(	int number,
												BooleanType isBound, CyclingStatus _status
												)
{
	if ( isBound == BT_TRUE )
	{
		/* Set cycling status of a bound. */
		if ( ( number >= 0 ) && ( number < nV ) )
		{
			status[number] = _status;
			return SUCCESSFUL_RETURN;
		}
		else
			return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );
	}
	else
	{
		/* Set cycling status of a constraint. */
		if ( ( number >= 0 ) && ( number < nC ) )
		{
			status[nV+number] = _status;
			return SUCCESSFUL_RETURN;
		}
		else
			return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );
	}
}


/*
 *	g e t C y c l i n g S t a t u s
 */
CyclingStatus CyclingManager::getCyclingStatus( int number, BooleanType isBound ) const
{
	if ( isBound == BT_TRUE )
	{
		/* Return cycling status of a bound. */
		if ( ( number >= 0 ) && ( number < nV ) )
			return status[number];
	}
	else
	{
		/* Return cycling status of a constraint. */
		if ( ( number >= 0 ) && ( number < nC ) )
			return status[nV+number];
	}

	return CYC_NOT_INVOLVED;
}


/*
 *	c l e a r C y c l i n g D a t a
 */
returnValue CyclingManager::clearCyclingData( )
{
	int i;

	/* Reset all status values ... */
	for( i=0; i<nV+nC; ++i )
		status[i] = CYC_NOT_INVOLVED;

	/* ... and the main cycling flag. */
	cyclingDetected = BT_FALSE;

	return SUCCESSFUL_RETURN;
}


/*
 *	end of file
 */
