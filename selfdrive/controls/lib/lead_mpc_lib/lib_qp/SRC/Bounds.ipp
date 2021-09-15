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
 *	\file SRC/Bounds.ipp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Implementation of inlined member functions of the Bounds class designed 
 *	to manage working sets of bounds within a QProblem.
 */


/*****************************************************************************
 *  P U B L I C                                                              *
 *****************************************************************************/

/*
 *	g e t N V
 */
inline int Bounds::getNV( ) const
{
 	return nV;
}


/*
 *	g e t N F X
 */
inline int Bounds::getNFV( ) const
{
 	return nFV;
}


/*
 *	g e t N B V
 */
inline int Bounds::getNBV( ) const
{
 	return nBV;
}
 

/*
 *	g e t N U V
 */
inline int Bounds::getNUV( ) const
{
	return nUV;
}



/*
 *	s e t N F X
 */
inline returnValue Bounds::setNFV( int n )
{
 	nFV = n;
	return SUCCESSFUL_RETURN;	
}
 
 
/*
 *	s e t N B V
 */
inline returnValue Bounds::setNBV( int n )
{
 	nBV = n;
	return SUCCESSFUL_RETURN;
}
 

/*
 *	s e t N U V
 */
inline returnValue Bounds::setNUV( int n )
{
	nUV = n;
	return SUCCESSFUL_RETURN;
}


/*
 *	g e t N F R
 */
inline int Bounds::getNFR( )
{
 	return free.getLength( );
}


/*
 *	g e t N F X
 */
inline int Bounds::getNFX( )
{
 	return fixed.getLength( );
}


/*
 *	g e t F r e e
 */
inline Indexlist* Bounds::getFree( )
{
	return &free;
}


/*
 *	g e t F i x e d
 */
inline Indexlist* Bounds::getFixed( )
{
	return &fixed;
}


/*
 *	end of file
 */
