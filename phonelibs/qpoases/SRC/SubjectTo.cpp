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
 *	\file SRC/SubjectTo.cpp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Implementation of the SubjectTo class designed to manage working sets of
 *	constraints and bounds within a QProblem.
 */


#include <SubjectTo.hpp>


/*****************************************************************************
 *  P U B L I C                                                              *
 *****************************************************************************/


/*
 *	S u b j e c t T o
 */
SubjectTo::SubjectTo( ) :	noLower( BT_TRUE ),
							noUpper( BT_TRUE ),
							size( 0 )
{
	int i;

	for( i=0; i<size; ++i )
	{
		type[i] = ST_UNKNOWN;
		status[i] = ST_UNDEFINED;
	}
}


/*
 *	S u b j e c t T o
 */
SubjectTo::SubjectTo( const SubjectTo& rhs ) :	noLower( rhs.noLower ),
												noUpper( rhs.noUpper ),
												size( rhs.size )
{
	int i;

	for( i=0; i<size; ++i )
	{
		type[i] = rhs.type[i];
		status[i] = rhs.status[i];
	}
}


/*
 *	~ S u b j e c t T o
 */
SubjectTo::~SubjectTo( )
{
}


/*
 *	o p e r a t o r =
 */
SubjectTo& SubjectTo::operator=( const SubjectTo& rhs )
{
	int i;

	if ( this != &rhs )
	{
		size = rhs.size;

		for( i=0; i<size; ++i )
		{
			type[i] = rhs.type[i];
			status[i] = rhs.status[i];
		}

		noLower = rhs.noLower;
		noUpper = rhs.noUpper;
	}

	return *this;
}



/*
 *	i n i t
 */
returnValue SubjectTo::init( int n )
{
	int i;

	size = n;

	noLower = BT_TRUE;
	noUpper = BT_TRUE;

	for( i=0; i<size; ++i )
	{
		type[i] = ST_UNKNOWN;
		status[i] = ST_UNDEFINED;
	}

	return SUCCESSFUL_RETURN;
}



/*****************************************************************************
 *  P R O T E C T E D                                                        *
 *****************************************************************************/

/*
 *	a d d I n d e x
 */
returnValue SubjectTo::addIndex(	Indexlist* const indexlist,
									int newnumber, SubjectToStatus newstatus
									)
{
	/* consistency check */
	if ( status[newnumber] == newstatus )
		return THROWERROR( RET_INDEX_ALREADY_OF_DESIRED_STATUS );

	status[newnumber] = newstatus;

	if ( indexlist->addNumber( newnumber ) == RET_INDEXLIST_EXCEEDS_MAX_LENGTH )
		return THROWERROR( RET_ADDINDEX_FAILED );

	return SUCCESSFUL_RETURN;
}


/*
 *	r e m o v e I n d e x
 */
returnValue SubjectTo::removeIndex(	Indexlist* const indexlist, 
									int removenumber
									)
{
	status[removenumber] = ST_UNDEFINED;

	if ( indexlist->removeNumber( removenumber ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_UNKNOWN_BUG );

	return SUCCESSFUL_RETURN;
}


/*
 *	s w a p I n d e x
 */
returnValue SubjectTo::swapIndex(	Indexlist* const indexlist,
									int number1, int number2
									)
{
	/* consistency checks */
	if ( status[number1] != status[number2] )
		return THROWERROR( RET_SWAPINDEX_FAILED );

	if ( number1 == number2 )
	{
		THROWWARNING( RET_NOTHING_TO_DO );
		return SUCCESSFUL_RETURN;
	}

	if ( indexlist->swapNumbers( number1,number2 ) != SUCCESSFUL_RETURN )
		return THROWERROR( RET_SWAPINDEX_FAILED );

	return SUCCESSFUL_RETURN;
}


/*
 *	end of file
 */
