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
 *	\file SRC/Constraints.cpp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Implementation of the Constraints class designed to manage working sets of
 *	constraints within a QProblem.
 */


#include <Constraints.hpp>


/*****************************************************************************
 *  P U B L I C                                                              *
 *****************************************************************************/


/*
 *	C o n s t r a i n t s
 */
Constraints::Constraints( ) :	SubjectTo( ),
								nC( 0 ),
								nEC( 0 ),
								nIC( 0 ),
								nUC( 0 )
{
}


/*
 *	C o n s t r a i n t s
 */
Constraints::Constraints( const Constraints& rhs ) :	SubjectTo( rhs ),
														nC( rhs.nC ),
														nEC( rhs.nEC ),
														nIC( rhs.nIC ),
														nUC( rhs.nUC )
{
	active =   rhs.active;
	inactive = rhs.inactive;
}


/*
 *	~ C o n s t r a i n t s
 */
Constraints::~Constraints( )
{
}


/*
 *	o p e r a t o r =
 */
Constraints& Constraints::operator=( const Constraints& rhs )
{
	if ( this != &rhs )
	{
		SubjectTo::operator=( rhs );

		nC  = rhs.nC;
		nEC = rhs.nEC;
		nIC = rhs.nIC;
		nUC = rhs.nUC;

		active =   rhs.active;
		inactive = rhs.inactive;
	}

	return *this;
}


/*
 *	i n i t
 */
returnValue Constraints::init( int n )
{
	nC = n;
	nEC = 0;
	nIC = 0;
	nUC = 0;

	active.init( );
	inactive.init( );

	return SubjectTo::init( n );
}


/*
 *	s e t u p C o n s t r a i n t
 */
returnValue Constraints::setupConstraint(	int _number, SubjectToStatus _status
											)
{
	/* consistency check */
	if ( ( _number < 0 ) || ( _number >= getNC( ) ) )
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );
	
	/* Add constraint index to respective index list. */
	switch ( _status )
	{
		case ST_INACTIVE:
			if ( this->addIndex( this->getInactive( ),_number,_status ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_SETUP_CONSTRAINT_FAILED );
			break;

		case ST_LOWER:
			if ( this->addIndex( this->getActive( ),_number,_status ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_SETUP_CONSTRAINT_FAILED );
			break;

		case ST_UPPER:
			if ( this->addIndex( this->getActive( ),_number,_status ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_SETUP_CONSTRAINT_FAILED );
			break;

		default:
			return THROWERROR( RET_INVALID_ARGUMENTS );
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	s e t u p A l l I n a c t i v e
 */
returnValue Constraints::setupAllInactive( )
{
	int i;


	/* 1) Place unbounded constraints at the beginning of the index list of inactive constraints. */
	for( i=0; i<nC; ++i )
	{
		if ( getType( i ) == ST_UNBOUNDED )
		{
			if ( setupConstraint( i,ST_INACTIVE ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_SETUP_CONSTRAINT_FAILED );
		}
	}

	/* 2) Add remaining (i.e. "real" inequality) constraints to the index list of inactive constraints. */
	for( i=0; i<nC; ++i )
	{
		if ( getType( i ) == ST_BOUNDED )
		{
			if ( setupConstraint( i,ST_INACTIVE ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_SETUP_CONSTRAINT_FAILED );
		}
	}

	/* 3) Place implicit equality constraints at the end of the index list of inactive constraints. */
	for( i=0; i<nC; ++i )
	{
		if ( getType( i ) == ST_EQUALITY )
		{
			if ( setupConstraint( i,ST_INACTIVE ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_SETUP_CONSTRAINT_FAILED );
		}
	}

	/* 4) Moreover, add all constraints of unknown type. */
	for( i=0; i<nC; ++i )
	{
		if ( getType( i ) == ST_UNKNOWN )
		{
			if ( setupConstraint( i,ST_INACTIVE ) != SUCCESSFUL_RETURN )
				return THROWERROR( RET_SETUP_CONSTRAINT_FAILED );
		}
	}

	return SUCCESSFUL_RETURN;
}


/*
 *	m o v e A c t i v e T o I n a c t i v e
 */
returnValue Constraints::moveActiveToInactive( int _number )
{
	/* consistency check */
	if ( ( _number < 0 ) || ( _number >= getNC( ) ) )
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );

	/* Move index from indexlist of active constraints to that of inactive ones. */
	if ( this->removeIndex( this->getActive( ),_number ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_MOVING_BOUND_FAILED );

	if ( this->addIndex( this->getInactive( ),_number,ST_INACTIVE ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_MOVING_BOUND_FAILED );

	return SUCCESSFUL_RETURN;
}


/*
 *	m o v e I n a c t i v e T o A c t i v e
 */
returnValue Constraints::moveInactiveToActive(	int _number, SubjectToStatus _status
												)
{
	/* consistency check */
	if ( ( _number < 0 ) || ( _number >= getNC( ) ) )
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );

	/* Move index from indexlist of inactive constraints to that of active ones. */
	if ( this->removeIndex( this->getInactive( ),_number ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_MOVING_BOUND_FAILED );

	if ( this->addIndex( this->getActive( ),_number,_status ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_MOVING_BOUND_FAILED );

	return SUCCESSFUL_RETURN;
}



/*
 *	end of file
 */
