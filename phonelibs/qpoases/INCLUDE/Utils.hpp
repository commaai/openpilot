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
 *	\file INCLUDE/Utils.hpp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Declaration of global utility functions for working with qpOASES.
 */


#ifndef QPOASES_UTILS_HPP
#define QPOASES_UTILS_HPP


#include <MessageHandling.hpp>


#ifdef PC_DEBUG  /* Define print functions only for debugging! */

/** Prints a vector.
 * \return SUCCESSFUL_RETURN */
returnValue print(	const real_t* const v,	/**< Vector to be printed. */
					int n					/**< Length of vector. */
					);

/** Prints a permuted vector.
 * \return SUCCESSFUL_RETURN */
returnValue print(	const real_t* const v,		/**< Vector to be printed. */
					int n,						/**< Length of vector. */
					const int* const V_idx		/**< Pemutation vector. */
					);

/** Prints a named vector.
 * \return SUCCESSFUL_RETURN */
returnValue print(	const real_t* const v,	/**< Vector to be printed. */
					int n,					/**< Length of vector. */
					const char* name		/** Name of vector. */
					);

/** Prints a matrix.
 * \return SUCCESSFUL_RETURN */
returnValue print(	const real_t* const M,	/**< Matrix to be printed. */
					int nrow,				/**< Row number of matrix. */
					int ncol				/**< Column number of matrix. */
					);

/** Prints a permuted matrix.
 * \return SUCCESSFUL_RETURN */
returnValue print(	const real_t* const M,		/**< Matrix to be printed. */
					int nrow,					/**< Row number of matrix. */
					int ncol	,				/**< Column number of matrix. */
					const int* const ROW_idx,	/**< Row pemutation vector. */
					const int* const COL_idx	/**< Column pemutation vector. */
					);

/** Prints a named matrix.
 * \return SUCCESSFUL_RETURN */
returnValue print(	const real_t* const M,	/**< Matrix to be printed. */
					int nrow,				/**< Row number of matrix. */
					int ncol,				/**< Column number of matrix. */
					const char* name		/** Name of matrix. */
					);

/** Prints an index array.
 * \return SUCCESSFUL_RETURN */
returnValue print(	const int* const index,	/**< Index array to be printed. */
					int n					/**< Length of index array. */
					);

/** Prints a named index array.
 * \return SUCCESSFUL_RETURN */
returnValue print(	const int* const index,	/**< Index array to be printed. */
					int n,					/**< Length of index array. */
					const char* name		/**< Name of index array. */
					);


/** Prints a string to desired output target (useful also for MATLAB output!).
 * \return SUCCESSFUL_RETURN */
returnValue myPrintf(	const char* s	/**< String to be written. */
						);


/** Prints qpOASES copyright notice.
 * \return SUCCESSFUL_RETURN */
returnValue printCopyrightNotice( );


/** Reads a real_t matrix from file.
 * \return SUCCESSFUL_RETURN \n
 		   RET_UNABLE_TO_OPEN_FILE \n
		   RET_UNABLE_TO_READ_FILE */
returnValue readFromFile(	real_t* data,				/**< Matrix to be read from file. */
							int nrow,					/**< Row number of matrix. */
							int ncol,					/**< Column number of matrix. */
							const char* datafilename	/**< Data file name. */
							);

/** Reads a real_t vector from file.
 * \return SUCCESSFUL_RETURN \n
 		   RET_UNABLE_TO_OPEN_FILE \n
		   RET_UNABLE_TO_READ_FILE */
returnValue readFromFile(	real_t* data,				/**< Vector to be read from file. */
							int n,						/**< Length of vector. */
							const char* datafilename	/**< Data file name. */
							);

/** Reads an integer (column) vector from file.
 * \return SUCCESSFUL_RETURN \n
 		   RET_UNABLE_TO_OPEN_FILE \n
		   RET_UNABLE_TO_READ_FILE */
returnValue readFromFile(	int* data,					/**< Vector to be read from file. */
							int n,						/**< Length of vector. */
							const char* datafilename	/**< Data file name. */
							);


/** Writes a real_t matrix into a file.
 * \return SUCCESSFUL_RETURN \n
 		   RET_UNABLE_TO_OPEN_FILE  */
returnValue writeIntoFile(	const real_t* const data,	/**< Matrix to be written into file. */
							int nrow,					/**< Row number of matrix. */
							int ncol,					/**< Column number of matrix. */
							const char* datafilename,	/**< Data file name. */
							BooleanType append			/**< Indicates if data shall be appended if the file already exists (otherwise it is overwritten). */
							);

/** Writes a real_t vector into a file.
 * \return SUCCESSFUL_RETURN \n
 		   RET_UNABLE_TO_OPEN_FILE  */
returnValue writeIntoFile(	const real_t* const data,	/**< Vector to be written into file. */
							int n,						/**< Length of vector. */
							const char* datafilename,	/**< Data file name. */
							BooleanType append			/**< Indicates if data shall be appended if the file already exists (otherwise it is overwritten). */
							);

/** Writes an integer (column) vector into a file.
 * \return SUCCESSFUL_RETURN \n
 		   RET_UNABLE_TO_OPEN_FILE */
returnValue writeIntoFile(	const int* const integer,	/**< Integer vector to be written into file. */
							int n,						/**< Length of vector. */
							const char* datafilename,	/**< Data file name. */
							BooleanType append			/**< Indicates if integer shall be appended if the file already exists (otherwise it is overwritten). */
							);

#endif  /* PC_DEBUG */


/** Returns the current system time.
 * \return current system time */
real_t getCPUtime( );


/** Returns the Euclidean norm of a vector.
 * \return 0: successful */
real_t getNorm(	const real_t* const v,	/**< Vector. */
				int n					/**< Vector's dimension. */
				);

/** Returns the absolute value of a real_t.
 * \return Absolute value of a real_t */
inline real_t getAbs(	real_t x		/**< Input argument. */
						);



#include <Utils.ipp>

#endif	/* QPOASES_UTILS_HPP */


/*
 *	end of file
 */
