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
 *	\file include/qpOASES_e/Utils.h
 *	\author Hans Joachim Ferreau, Andreas Potschka, Christian Kirches
 *	\version 3.1embedded
 *	\date 2007-2015
 *
 *	Declaration of some utilities for working with the different QProblem classes.
 */


#ifndef QPOASES_UTILS_H
#define QPOASES_UTILS_H

#include <qpOASES_e/MessageHandling.h>


BEGIN_NAMESPACE_QPOASES


/** Prints a vector.
 * \return SUCCESSFUL_RETURN */
returnValue qpOASES_printV(	const real_t* const v,	/**< Vector to be printed. */
							int n					/**< Length of vector. */
							);

/** Prints a permuted vector.
 * \return SUCCESSFUL_RETURN */
returnValue qpOASES_printPV(	const real_t* const v,		/**< Vector to be printed. */
								int n,						/**< Length of vector. */
								const int* const V_idx		/**< Pemutation vector. */
								);

/** Prints a named vector.
 * \return SUCCESSFUL_RETURN */
returnValue qpOASES_printNV(	const real_t* const v,	/**< Vector to be printed. */
								int n,					/**< Length of vector. */
								const char* name		/** Name of vector. */
								);

/** Prints a matrix.
 * \return SUCCESSFUL_RETURN */
returnValue qpOASES_printM(	const real_t* const M,	/**< Matrix to be printed. */
							int nrow,				/**< Row number of matrix. */
							int ncol				/**< Column number of matrix. */
							);

/** Prints a permuted matrix.
 * \return SUCCESSFUL_RETURN */
returnValue qpOASES_printPM(	const real_t* const M,		/**< Matrix to be printed. */
								int nrow,					/**< Row number of matrix. */
								int ncol	,				/**< Column number of matrix. */
								const int* const ROW_idx,	/**< Row pemutation vector. */
								const int* const COL_idx	/**< Column pemutation vector. */
								);

/** Prints a named matrix.
 * \return SUCCESSFUL_RETURN */
returnValue qpOASES_printNM(	const real_t* const M,	/**< Matrix to be printed. */
								int nrow,				/**< Row number of matrix. */
								int ncol,				/**< Column number of matrix. */
								const char* name		/** Name of matrix. */
								);

/** Prints an index array.
 * \return SUCCESSFUL_RETURN */
returnValue qpOASES_printI(	const int* const _index,	/**< Index array to be printed. */
							int n						/**< Length of index array. */
							);

/** Prints a named index array.
 * \return SUCCESSFUL_RETURN */
returnValue qpOASES_printNI(	const int* const _index,	/**< Index array to be printed. */
								int n,						/**< Length of index array. */
								const char* name			/**< Name of index array. */
								);


/** Prints a string to desired output target (useful also for MATLAB output!).
 * \return SUCCESSFUL_RETURN */
returnValue qpOASES_myPrintf(	const char* s	/**< String to be written. */
								);


/** Prints qpOASES copyright notice.
 * \return SUCCESSFUL_RETURN */
returnValue qpOASES_printCopyrightNotice( );


/** Reads a real_t matrix from file.
 * \return SUCCESSFUL_RETURN \n
 		   RET_UNABLE_TO_OPEN_FILE \n
		   RET_UNABLE_TO_READ_FILE */
returnValue qpOASES_readFromFileM(	real_t* data,				/**< Matrix to be read from file. */
									int nrow,					/**< Row number of matrix. */
									int ncol,					/**< Column number of matrix. */
									const char* datafilename	/**< Data file name. */
									);

/** Reads a real_t vector from file.
 * \return SUCCESSFUL_RETURN \n
 		   RET_UNABLE_TO_OPEN_FILE \n
		   RET_UNABLE_TO_READ_FILE */
returnValue qpOASES_readFromFileV(	real_t* data,				/**< Vector to be read from file. */
									int n,						/**< Length of vector. */
									const char* datafilename	/**< Data file name. */
									);

/** Reads an integer (column) vector from file.
 * \return SUCCESSFUL_RETURN \n
 		   RET_UNABLE_TO_OPEN_FILE \n
		   RET_UNABLE_TO_READ_FILE */
returnValue qpOASES_readFromFileI(	int* data,					/**< Vector to be read from file. */
									int n,						/**< Length of vector. */
									const char* datafilename	/**< Data file name. */
									);


/** Writes a real_t matrix into a file.
 * \return SUCCESSFUL_RETURN \n
 		   RET_UNABLE_TO_OPEN_FILE  */
returnValue qpOASES_writeIntoFileM(	const real_t* const data,	/**< Matrix to be written into file. */
									int nrow,					/**< Row number of matrix. */
									int ncol,					/**< Column number of matrix. */
									const char* datafilename,	/**< Data file name. */
									BooleanType append			/**< Indicates if data shall be appended if the file already exists (otherwise it is overwritten). */
									);

/** Writes a real_t vector into a file.
 * \return SUCCESSFUL_RETURN \n
 		   RET_UNABLE_TO_OPEN_FILE  */
returnValue qpOASES_writeIntoFileV(	const real_t* const data,	/**< Vector to be written into file. */
									int n,						/**< Length of vector. */
									const char* datafilename,	/**< Data file name. */
									BooleanType append			/**< Indicates if data shall be appended if the file already exists (otherwise it is overwritten). */
									);

/** Writes an integer (column) vector into a file.
 * \return SUCCESSFUL_RETURN \n
 		   RET_UNABLE_TO_OPEN_FILE */
returnValue qpOASES_writeIntoFileI(	const int* const integer,	/**< Integer vector to be written into file. */
									int n,						/**< Length of vector. */
									const char* datafilename,	/**< Data file name. */
									BooleanType append			/**< Indicates if integer shall be appended if the file already exists (otherwise it is overwritten). */
									);

/** Writes a real_t matrix/vector into a Matlab binary file.
 * \return SUCCESSFUL_RETURN \n
		   RET_INVALID_ARGUMENTS
 		   RET_UNABLE_TO_WRITE_FILE */
returnValue qpOASES_writeIntoMatFile(	FILE* const matFile,		/**< Pointer to Matlab binary file. */
										const real_t* const data,	/**< Data to be written into file. */
										int nRows,					/**< Row number of matrix. */
										int nCols, 					/**< Column number of matrix. */
										const char* name			/**< Matlab name of matrix/vector to be stored. */
 										);

/** Writes in integer matrix/vector into a Matlab binary file.
 * \return SUCCESSFUL_RETURN \n
		   RET_INVALID_ARGUMENTS
 		   RET_UNABLE_TO_WRITE_FILE */
returnValue qpOASES_writeIntoMatFileI(	FILE* const matFile,		/**< Pointer to Matlab binary file. */
										const int* const data,		/**< Data to be written into file. */
										int nRows,					/**< Row number of matrix. */
										int nCols,					/**< Column number of matrix. */
										const char* name			/**< Matlab name of matrix/vector to be stored. */
 										);


/** Returns the current system time.
 * \return current system time */
real_t qpOASES_getCPUtime( );


/** Returns the N-norm of a vector.
 * \return >= 0.0: successful */
real_t qpOASES_getNorm(	const real_t* const v,	/**< Vector. */
						int n,					/**< Vector's dimension. */
						int type				/**< Norm type, 1: one-norm, 2: Euclidean norm. */
						);

/** Tests whether two real-valued arguments are (numerically) equal.
 * \return	BT_TRUE:  arguments differ not more than TOL \n
		 	BT_FALSE: arguments differ more than TOL */
static inline BooleanType qpOASES_isEqual(	real_t x,	/**< First real number. */
											real_t y,	/**< Second real number. */
											real_t TOL	/**< Tolerance for comparison. */
											);


/** Tests whether a real-valued argument is (numerically) zero.
 * \return	BT_TRUE:  argument differs from 0.0 not more than TOL \n
		 	BT_FALSE: argument differs from 0.0 more than TOL */
static inline BooleanType qpOASES_isZero(	real_t x,			/**< Real number. */
											real_t TOL			/**< Tolerance for comparison. */
											);


/** Returns sign of a real-valued argument.
 * \return	 1.0: argument is non-negative \n
		 	-1.0: argument is negative */
static inline real_t qpOASES_getSign(	real_t arg	/**< real-valued argument whose sign is to be determined. */
										);


/** Returns maximum of two integers.
 * \return	Maximum of two integers */
static inline int qpOASES_getMaxI(	int x,	/**< First integer. */
									int y	/**< Second integer. */
									);


/** Returns minimum of two integers.
 * \return	Minimum of two integers */
static inline int qpOASES_getMinI(	int x,	/**< First integer. */
									int y	/**< Second integer. */
									);


/** Returns maximum of two reals.
 * \return	Maximum of two reals */
static inline real_t qpOASES_getMax(	real_t x,	/**< First real number. */
										real_t y	/**< Second real number. */
										);


/** Returns minimum of two reals.
 * \return	Minimum of two reals */
static inline real_t qpOASES_getMin(	real_t x,	/**< First real number. */
										real_t y	/**< Second real number. */
										);


/** Returns the absolute value of a real_t-valued argument.
 * \return	Absolute value of a real_t-valued argument */
static inline real_t qpOASES_getAbs(	real_t x	/**< real_t-valued argument. */
										);

/** Returns the square-root of a real number.
 * \return	Square-root of a real number */
static inline real_t qpOASES_getSqrt(	real_t x	/**< Non-negative real number. */
										);


/** Computes the maximum violation of the KKT optimality conditions
 *	of given iterate for given QP data. */
returnValue qpOASES_getKktViolation(	int nV,						/**< Number of variables. */
										int nC,						/**< Number of constraints. */
										const real_t* const H,		/**< Hessian matrix (may be NULL if Hessian is zero or identity matrix). */
										const real_t* const g,		/**< Gradient vector. */
										const real_t* const A,		/**< Constraint matrix. */
										const real_t* const lb,		/**< Lower bound vector (on variables). */
										const real_t* const ub,		/**< Upper bound vector (on variables). */
										const real_t* const lbA,	/**< Lower constraints' bound vector. */
										const real_t* const ubA,	/**< Upper constraints' bound vector. */
										const real_t* const x,		/**< Primal trial vector. */
										const real_t* const y,		/**< Dual trial vector. */
										real_t* const _stat,		/**< Output: maximum value of stationarity condition residual. */
										real_t* const feas,			/**< Output: maximum value of primal feasibility violation. */
										real_t* const cmpl			/**< Output: maximum value of complementarity residual. */
										);

/** Computes the maximum violation of the KKT optimality conditions
 *	of given iterate for given QP data. */
returnValue qpOASES_getKktViolationSB(	int nV,						/**< Number of variables. */
										const real_t* const H,		/**< Hessian matrix (may be NULL if Hessian is zero or identity matrix). */
										const real_t* const g,		/**< Gradient vector. */
										const real_t* const lb,		/**< Lower bound vector (on variables). */
										const real_t* const ub,		/**< Upper bound vector (on variables). */
										const real_t* const x,		/**< Primal trial vector. */
										const real_t* const y,		/**< Dual trial vector. */
										real_t* const _stat,		/**< Output: maximum value of stationarity condition residual. */
										real_t* const feas,			/**< Output: maximum value of primal feasibility violation. */
										real_t* const cmpl			/**< Output: maximum value of complementarity residual. */
										);


/** Writes a value of BooleanType into a string.
 * \return SUCCESSFUL_RETURN */
returnValue qpOASES_convertBooleanTypeToString(	BooleanType value, 		/**< Value to be written. */
												char* const string		/**< Input: String of sufficient size, \n
																			Output: String containing value. */
												);

/** Writes a value of SubjectToStatus into a string.
 * \return SUCCESSFUL_RETURN */
returnValue qpOASES_convertSubjectToStatusToString(	SubjectToStatus value,	/**< Value to be written. */
													char* const string		/**< Input: String of sufficient size, \n
																				Output: String containing value. */
													);

/** Writes a value of PrintLevel into a string.
 * \return SUCCESSFUL_RETURN */
returnValue qpOASES_convertPrintLevelToString(	PrintLevel value, 		/**< Value to be written. */
												char* const string		/**< Input: String of sufficient size, \n
																			Output: String containing value. */
												);


/** Converts a returnValue from an QProblem(B) object into a more 
 *	simple status flag.
 *
 * \return  0: QP problem solved
 *          1: QP could not be solved within given number of iterations
 *         -1: QP could not be solved due to an internal error
 *         -2: QP is infeasible (and thus could not be solved)
 *         -3: QP is unbounded (and thus could not be solved)
 */
int qpOASES_getSimpleStatus(	returnValue returnvalue, 	/**< ReturnValue to be analysed. */
								BooleanType doPrintStatus	/**< Flag indicating whether simple status shall be printed to screen. */
								);

/** Normalises QP constraints.
 * \return SUCCESSFUL_RETURN \n
 *		   RET_INVALID_ARGUMENTS */
returnValue qpOASES_normaliseConstraints(	int nV,			/**< Number of variables. */
											int nC, 		/**< Number of constraints. */
											real_t* A,		/**< Input:  Constraint matrix, \n
																 Output: Normalised constraint matrix. */
											real_t* lbA,	/**< Input:  Constraints' lower bound vector, \n
																 Output: Normalised constraints' lower bound vector. */
											real_t* ubA,	/**< Input:  Constraints' upper bound vector, \n
																 Output: Normalised constraints' upper bound vector. */
											int type		/**< Norm type, 1: one-norm, 2: Euclidean norm. */
											);


#ifdef __DEBUG__
/** Writes matrix with given dimension into specified file. */
void gdb_printmat(	const char *fname,			/**< File name. */
					real_t *M,					/**< Matrix to be written. */
					int n,						/**< Number of rows. */
					int m,						/**< Number of columns. */
					int ldim					/**< Leading dimension. */
					);
#endif /* __DEBUG__ */


#if defined(__DSPACE__) || defined(__XPCTARGET__) 
void __cxa_pure_virtual( void );
#endif /* __DSPACE__ || __XPCTARGET__*/ 



/*
 *   i s E q u a l
 */
static inline BooleanType qpOASES_isEqual(	real_t x,
											real_t y,
											real_t TOL
											)
{
    if ( qpOASES_getAbs(x-y) <= TOL )
		return BT_TRUE;
	else
		return BT_FALSE;
}


/*
 *   i s Z e r o
 */
static inline BooleanType qpOASES_isZero(	real_t x,
											real_t TOL
											)
{
    if ( qpOASES_getAbs(x) <= TOL )
		return BT_TRUE;
	else
		return BT_FALSE;
}


/*
 *   g e t S i g n
 */
static inline real_t qpOASES_getSign(	real_t arg
										)
{
	if ( arg >= 0.0 )
		return 1.0;
	else
		return -1.0;
}



/*
 *   g e t M a x
 */
static inline int qpOASES_getMaxI(	int x,
									int y
									)
{
    return (y<x) ? x : y;
}


/*
 *   g e t M i n
 */
static inline int qpOASES_getMinI(	int x,
									int y
									)
{
    return (y>x) ? x : y;
}


/*
 *   g e t M a x
 */
static inline real_t qpOASES_getMax(	real_t x,
										real_t y
										)
{
	#ifdef __NO_FMATH__
    return (y<x) ? x : y;
	#else
	return (y<x) ? x : y;
	/*return fmax(x,y); seems to be slower */
	#endif
}


/*
 *   g e t M i n
 */
static inline real_t qpOASES_getMin(	real_t x,
										real_t y
										)
{
	#ifdef __NO_FMATH__
    return (y>x) ? x : y;
	#else
	return (y>x) ? x : y;
	/*return fmin(x,y); seems to be slower */
	#endif
}


/*
 *   g e t A b s
 */
static inline real_t qpOASES_getAbs(	real_t x
										)
{
	#ifdef __NO_FMATH__
	return (x>=0.0) ? x : -x;
	#else
	return fabs(x);
	#endif
}

/*
 *   g e t S q r t
 */
static inline real_t qpOASES_getSqrt(	real_t x
										)
{
    #ifdef __NO_FMATH__
	return sqrt(x); /* put your custom sqrt-replacement here */
	#else
	return sqrt(x);
	#endif
}


END_NAMESPACE_QPOASES


#endif	/* QPOASES_UTILS_H */


/*
 *	end of file
 */
