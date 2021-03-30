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
 *	\file SRC/Indexlist.ipp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Implementation of inlined member functions of the Indexlist class designed 
 *	to manage index lists of constraints and bounds within a QProblem_SubjectTo.
 */



/*****************************************************************************
 *  P U B L I C                                                              *
 *****************************************************************************/

/*
 *	g e t N u m b e r
 */
inline int Indexlist::getNumber( int physicalindex ) const
{
	/* consistency check */
	if ( ( physicalindex < 0 ) || ( physicalindex > length ) )
		return -RET_INDEXLIST_OUTOFBOUNDS;

	return number[physicalindex];
}


/*
 *	g e t L e n g t h
 */
inline int Indexlist::getLength( )
{
	return length;
}


/*
 *	g e t L a s t N u m b e r
 */
inline int Indexlist::getLastNumber( ) const
{
	return number[last];
}


/*
 *	g e t L a s t N u m b e r
 */
inline BooleanType Indexlist::isMember( int _number ) const
{
	if ( getIndex( _number ) >= 0 )
		return BT_TRUE;
	else
		return BT_FALSE;
}


/*
 *	end of file
 */
