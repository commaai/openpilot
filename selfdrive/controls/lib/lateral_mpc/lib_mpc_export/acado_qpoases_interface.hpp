/*
 *    This file was auto-generated using the ACADO Toolkit.
 *    
 *    While ACADO Toolkit is free software released under the terms of
 *    the GNU Lesser General Public License (LGPL), the generated code
 *    as such remains the property of the user who used ACADO Toolkit
 *    to generate this code. In particular, user dependent data of the code
 *    do not inherit the GNU LGPL license. On the other hand, parts of the
 *    generated code that are a direct copy of source code from the
 *    ACADO Toolkit or the software tools it is based on, remain, as derived
 *    work, automatically covered by the LGPL license.
 *    
 *    ACADO Toolkit is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *    
 */


#ifndef QPOASES_HEADER
#define QPOASES_HEADER

#ifdef PC_DEBUG
#include <stdio.h>
#endif /* PC_DEBUG */

#include <math.h>

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

/*
 * A set of options for qpOASES
 */

/** Maximum number of optimization variables. */
#define QPOASES_NVMAX      24
/** Maximum number of constraints. */
#define QPOASES_NCMAX      40
/** Maximum number of working set recalculations. */
#define QPOASES_NWSRMAX    500
/** Print level for qpOASES. */
#define QPOASES_PRINTLEVEL PL_NONE
/** The value of EPS */
#define QPOASES_EPS        2.221e-16
/** Internally used floating point type */
typedef double real_t;

/*
 * Forward function declarations
 */

/** A function that calls the QP solver */
EXTERNC int acado_solve( void );

/** Get the number of active set changes */
EXTERNC int acado_getNWSR( void );

/** Get the error string. */
const char* acado_getErrorString( int error );

#endif /* QPOASES_HEADER */
