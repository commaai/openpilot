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
 *	\file SRC/Constraints.ipp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Declaration of inlined member functions of the Constraints class designed 
 *	to manage working sets of constraints within a QProblem.
 */



/*****************************************************************************
 *  P U B L I C                                                              *
 *****************************************************************************/

/*
 *	g e t N C
 */
inline int Constraints::getNC( ) const
{
 	return nC;
}
 

/*
 *	g e t N E C
 */
inline int Constraints::getNEC( ) const
{
 	return nEC;
}
 

/*
 *	g e t N I C
 */
inline int Constraints::getNIC( ) const
{
 	return nIC;
}
 

/*
 *	g e t N U C
 */
inline int Constraints::getNUC( ) const
{
 	return nUC;
}


/*
 *	s e t N E C
 */
inline returnValue Constraints::setNEC( int n )
{
 	nEC = n;
	return SUCCESSFUL_RETURN;
}
 

/*
 *	s e t N I C
 */
inline returnValue Constraints::setNIC( int n )
{
 	nIC = n;
	return SUCCESSFUL_RETURN;
}


/*
 *	s e t N U C
 */
inline returnValue Constraints::setNUC( int n )
{
 	nUC = n;
	return SUCCESSFUL_RETURN;
}


/*
 *	g e t N A C
 */
inline int Constraints::getNAC( )
{
 	return active.getLength( );
}


/*
 *	g e t N I A C
 */
inline int Constraints::getNIAC( )
{
 	return inactive.getLength( );
}


/*
 *	g e t A c t i v e
 */
inline Indexlist* Constraints::getActive( )
{
	return &active;
}


/*
 *	g e t I n a c t i v e
 */
inline Indexlist* Constraints::getInactive( )
{
	return &inactive;
}


/*
 *	end of file
 */
