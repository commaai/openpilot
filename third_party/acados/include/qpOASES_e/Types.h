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
 *	\file include/qpOASES_e/Types.h
 *	\author Hans Joachim Ferreau, Andreas Potschka, Christian Kirches
 *	\version 3.1embedded
 *	\date 2007-2015
 *
 *	Declaration of all non-built-in types (except for classes).
 */


#ifndef QPOASES_TYPES_H
#define QPOASES_TYPES_H

#ifdef USE_ACADOS_TYPES
#include "acados/utils/types.h"
#endif

/* If your compiler does not support the snprintf() function,
 * uncomment the following line and try to compile again. */
/* #define __NO_SNPRINTF__ */


/* Uncomment the following line for setting the __DSPACE__ flag. */
/* #define __DSPACE__ */

/* Uncomment the following line for setting the __XPCTARGET__ flag. */
/* #define __XPCTARGET__ */


/* Uncomment the following line for setting the __NO_FMATH__ flag. */
/* #define __NO_FMATH__ */

/* Uncomment the following line to enable debug information. */
/* #define __DEBUG__ */

/* Uncomment the following line to enable suppress any kind of console output. */
/* #define __SUPPRESSANYOUTPUT__ */


/** Forces to always include all implicitly fixed bounds and all equality constraints
 *  into the initial working set when setting up an auxiliary QP. */
#define __ALWAYS_INITIALISE_WITH_ALL_EQUALITIES__

/* Uncomment the following line to activate the use of an alternative Givens
 * plane rotation requiring only three multiplications. */
/* #define __USE_THREE_MULTS_GIVENS__ */

/* Uncomment the following line to activate the use of single precision arithmetic. */
/* #define __USE_SINGLE_PRECISION__ */

/* The inline keyword is skipped by default as it is not part of the C90 standard.
 * However, by uncommenting the following line, use of the inline keyword can be enforced. */
/* #define __USE_INLINE__ */


/* Work-around for Borland BCC 5.5 compiler. */
#ifdef __BORLANDC__
#if __BORLANDC__ < 0x0561
  #define __STDC__ 1
#endif
#endif


/* Work-around for Microsoft compilers. */
#ifdef _MSC_VER
  #define __NO_SNPRINTF__
  #pragma warning( disable : 4061 4100 4250 4514 4996 )
#endif


/* Apply pre-processor settings when using qpOASES within auto-generated code. */
#ifdef __CODE_GENERATION__
  #define __NO_COPYRIGHT__
  #define __EXTERNAL_DIMENSIONS__
#endif /* __CODE_GENERATION__ */


/* Avoid using static variables declaration within functions. */
#ifdef __NO_STATIC__
  #define myStatic
#else
  #define myStatic static
#endif /* __NO_STATIC__ */


/* Skip inline keyword if not specified otherwise. */
#ifndef __USE_INLINE__
  #define inline
#endif


/* Avoid any printing on embedded platforms. */
#if defined(__DSPACE__) || defined(__XPCTARGET__)
  #define __SUPPRESSANYOUTPUT__
  #define __NO_SNPRINTF__
#endif


#ifdef __NO_SNPRINTF__
  #if (!defined(_MSC_VER)) || defined(__DSPACE__) || defined(__XPCTARGET__)
    /* If snprintf is not available, provide an empty implementation... */
    int snprintf( char* s, size_t n, const char* format, ... );
  #else
	/* ... or substitute snprintf by _snprintf for Microsoft compilers. */
    #define snprintf _snprintf
  #endif
#endif /* __NO_SNPRINTF__ */


/** Macro for switching on/off the beginning of the qpOASES namespace definition. */
#define BEGIN_NAMESPACE_QPOASES

/** Macro for switching on/off the end of the qpOASES namespace definition. */
#define END_NAMESPACE_QPOASES

/** Macro for switching on/off the use of the qpOASES namespace. */
#define USING_NAMESPACE_QPOASES

/** Macro for switching on/off references to the qpOASES namespace. */
#define REFER_NAMESPACE_QPOASES /*::*/


/** Macro for accessing the Cholesky factor R. */
#define RR( I,J )  _THIS->R[(I)+nV*(J)]

/** Macro for accessing the orthonormal matrix Q of the QT factorisation. */
#define QQ( I,J )  _THIS->Q[(I)+nV*(J)]

/** Macro for accessing the triangular matrix T of the QT factorisation. */
#define TT( I,J )  _THIS->T[(I)*nVC_min+(J)]



BEGIN_NAMESPACE_QPOASES


/** Defines real_t for facilitating switching between double and float. */

#ifndef USE_ACADOS_TYPES
#ifndef __CODE_GENERATION__

  #ifdef __USE_SINGLE_PRECISION__
  typedef float real_t;
  #else
  typedef double real_t;
  #endif /* __USE_SINGLE_PRECISION__ */

#endif /* __CODE_GENERATION__ */
#endif /* USE_ACADOS_TYPES */

/** Summarises all possible logical values. */
typedef enum
{
	BT_FALSE = 0,				/**< Logical value for "false". */
	BT_TRUE						/**< Logical value for "true". */
} BooleanType;


/** Summarises all possible print levels. Print levels are used to describe
 *	the desired amount of output during runtime of qpOASES. */
typedef enum
{
	PL_DEBUG_ITER = -2,			/**< Full tabular debugging output. */
	PL_TABULAR,					/**< Tabular output. */
	PL_NONE,					/**< No output. */
	PL_LOW,						/**< Print error messages only. */
	PL_MEDIUM,					/**< Print error and warning messages as well as concise info messages. */
	PL_HIGH						/**< Print all messages with full details. */
} PrintLevel;


/** Defines visibility status of a message. */
typedef enum
{
	VS_HIDDEN,					/**< Message not visible. */
	VS_VISIBLE					/**< Message visible. */
} VisibilityStatus;


/** Summarises all possible states of the (S)QProblem(B) object during the
solution process of a QP sequence. */
typedef enum
{
	QPS_NOTINITIALISED,			/**< QProblem object is freshly instantiated or reset. */
	QPS_PREPARINGAUXILIARYQP,	/**< An auxiliary problem is currently setup, either at the very beginning
								 *   via an initial homotopy or after changing the QP matrices. */
	QPS_AUXILIARYQPSOLVED,		/**< An auxilary problem was solved, either at the very beginning
								 *   via an initial homotopy or after changing the QP matrices. */
	QPS_PERFORMINGHOMOTOPY,		/**< A homotopy according to the main idea of the online active
								 *   set strategy is performed. */
	QPS_HOMOTOPYQPSOLVED,		/**< An intermediate QP along the homotopy path was solved. */
	QPS_SOLVED					/**< The solution of the actual QP was found. */
} QProblemStatus;


/** Summarises all possible types of the QP's Hessian matrix. */
typedef enum
{
	HST_ZERO,				/**< Hessian is zero matrix (i.e. LP formulation). */
	HST_IDENTITY,			/**< Hessian is identity matrix. */
	HST_POSDEF,				/**< Hessian is (strictly) positive definite. */
	HST_POSDEF_NULLSPACE,	/**< Hessian is positive definite on null space of active bounds/constraints. */
	HST_SEMIDEF,			/**< Hessian is positive semi-definite. */
	HST_INDEF,				/**< Hessian is indefinite. */
	HST_UNKNOWN				/**< Hessian type is unknown. */
} HessianType;


/** Summarises all possible types of bounds and constraints. */
typedef enum
{
	ST_UNBOUNDED,		/**< Bound/constraint is unbounded. */
	ST_BOUNDED,			/**< Bound/constraint is bounded but not fixed. */
	ST_EQUALITY,		/**< Bound/constraint is fixed (implicit equality bound/constraint). */
	ST_DISABLED,		/**< Bound/constraint is disabled (i.e. ignored when solving QP). */
	ST_UNKNOWN			/**< Type of bound/constraint unknown. */
} SubjectToType;


/** Summarises all possible states of bounds and constraints. */
typedef enum
{
	ST_LOWER = -1,			/**< Bound/constraint is at its lower bound. */
	ST_INACTIVE,			/**< Bound/constraint is inactive. */
	ST_UPPER,				/**< Bound/constraint is at its upper bound. */
	ST_INFEASIBLE_LOWER,	/**< (to be documented) */
	ST_INFEASIBLE_UPPER,	/**< (to be documented) */
	ST_UNDEFINED			/**< Status of bound/constraint undefined. */
} SubjectToStatus;


/**
 *	\brief Stores internal information for tabular (debugging) output.
 *
 *	Struct storing internal information for tabular (debugging) output
 *	when using the (S)QProblem(B) objects.
 *
 *	\author Hans Joachim Ferreau
 *	\version 3.1embedded
 *	\date 2013-2015
 */
typedef struct
{
	int idxAddB;		/**< Index of bound that has been added to working set. */
	int idxRemB;		/**< Index of bound that has been removed from working set. */
	int idxAddC;		/**< Index of constraint that has been added to working set. */
	int idxRemC;		/**< Index of constraint that has been removed from working set. */
	int excAddB;		/**< Flag indicating whether a bound has been added to working set to keep a regular projected Hessian. */
	int excRemB;		/**< Flag indicating whether a bound has been removed from working set to keep a regular projected Hessian. */
	int excAddC;		/**< Flag indicating whether a constraint has been added to working set to keep a regular projected Hessian. */
	int excRemC;		/**< Flag indicating whether a constraint has been removed from working set to keep a regular projected Hessian. */
} TabularOutput;

/**
 *	\brief Struct containing the variable header for mat file.
 *
 *	Struct storing the header of a variable to be stored in
 *	Matlab's binary format (using the outdated Level 4 variant
 *  for simplictiy).
 *
 *  Note, this code snippet has been inspired from the document
 *  "Matlab(R) MAT-file Format, R2013b" by MathWorks
 *
 *	\author Hans Joachim Ferreau
 *	\version 3.1embedded
 *	\date 2013-2015
 */
typedef struct
{
	long numericFormat;		/**< Flag indicating numerical format. */
	long nRows;				/**< Number of rows. */
	long nCols;				/**< Number of rows. */
	long imaginaryPart;		/**< (to be documented) */
	long nCharName;			/**< Number of character in name. */
} MatMatrixHeader;


END_NAMESPACE_QPOASES


#endif	/* QPOASES_TYPES_H */


/*
 *	end of file
 */
