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
 *	\file SRC/Utils.ipp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Implementation of some inlined utilities for working with the different QProblem 
 *  classes.
 */



/*
 *	g e t A b s
 */
inline real_t getAbs( real_t x )
{
	if ( x < 0.0 )
		return -x;
	else
		return x;
}


/*
 *	end of file
 */
