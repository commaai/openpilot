/*
 *	This file is part of qpOASES.
 *
 *	qpOASES -- An Implementation of the Online Active Set Strategy.
 *	Copyright (C) 2007-2015 by Hans Joachim Ferreau, Andreas Potschka,
 *	Christian Kirches et al. All rights reserved.
 *
 *	qpOASES is free software; you can redistribute it and/or
 *	modify it under the terms of the GNU Lesser General Public
 *	License as published by the Free Software Foundation; either
 *	version 2.1 of the License, or (at your option) any later version.
 *
 *	qpOASES is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *	See the GNU Lesser General Public License for more details.
 *
 *	You should have received a copy of the GNU Lesser General Public
 *	License along with qpOASES; if not, write to the Free Software
 *	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


/**
 *	\file include/qpOASES_e/UnitTesting.h
 *	\author Hans Joachim Ferreau
 *	\version 3.1embedded
 *	\date 2014-2015
 *
 *	Definition of auxiliary functions/macros for unit testing.
 */


#ifndef QPOASES_UNIT_TESTING_H
#define QPOASES_UNIT_TESTING_H


#ifndef TEST_TOL_FACTOR
#define TEST_TOL_FACTOR 1
#endif


/** Return value for tests that passed. */
#define TEST_PASSED 0

/** Return value for tests that failed. */
#define TEST_FAILED 1

/** Return value for tests that could not run due to missing external data. */
#define TEST_DATA_NOT_FOUND 99


/** Macro verifying that two numerical values are equal in order to pass unit test. */
#define QPOASES_TEST_FOR_EQUAL( x,y ) if ( REFER_NAMESPACE_QPOASES isEqual( (x),(y) ) == BT_FALSE ) { return TEST_FAILED; }

/** Macro verifying that two numerical values are close to each other in order to pass unit test. */
#define QPOASES_TEST_FOR_NEAR( x,y )  if ( REFER_NAMESPACE_QPOASES getAbs((x)-(y)) / REFER_NAMESPACE_QPOASES getMax( 1.0,REFER_NAMESPACE_QPOASES getAbs(x) ) >= 1e-10 ) { return TEST_FAILED; }

/** Macro verifying that first quantity is lower or equal than second one in order to pass unit test. */
#define QPOASES_TEST_FOR_TOL( x,tol )  if ( (x) > (tol)*(TEST_TOL_FACTOR) ) { return TEST_FAILED; }

/** Macro verifying that a logical expression holds in order to pass unit test. */
#define QPOASES_TEST_FOR_TRUE( x )  if ( (x) == 0 ) { return TEST_FAILED; }



BEGIN_NAMESPACE_QPOASES


END_NAMESPACE_QPOASES


#endif	/* QPOASES_UNIT_TESTING_H */


/*
 *	end of file
 */
