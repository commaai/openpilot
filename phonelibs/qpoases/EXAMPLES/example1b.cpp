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
 *	\file EXAMPLES/example1b.cpp
 *	\author Hans Joachim Ferreau
 *	\version 1.3
 *	\date 2007-2008
 *
 *	Very simple example for testing qpOASES using the QProblemB class.
 */


#include <QProblemB.hpp>


/** Example for qpOASES main function using the QProblemB class. */
int main( )
{
	/* Setup data of first QP. */
	real_t H[2*2] = { 1.0, 0.0, 0.0, 0.5 };
	real_t g[2] = { 1.5, 1.0 };
	real_t lb[2] = { 0.5, -2.0 };
	real_t ub[2] = { 5.0, 2.0 };

	/* Setup data of second QP. */
	real_t g_new[2] = { 1.0, 1.5 };
	real_t lb_new[2] = { 0.0, -1.0 };
	real_t ub_new[2] = { 5.0, -0.5 };


	/* Setting up QProblemB object. */
	QProblemB example( 2 );

	/* Solve first QP. */
	int nWSR = 10;
	example.init( H,g,lb,ub, nWSR,0 );

	/* Solve second QP. */
	nWSR = 10;
	example.hotstart( g_new,lb_new,ub_new, nWSR,0 );

	return 0;
}


/*
 *	end of file
 */
