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
 *	\file include/qpOASES_e/Flipper.h
 *	\author Hans Joachim Ferreau, Andreas Potschka, Christian Kirches
 *	\version 3.1embedded
 *	\date 2007-2015
 *
 *	Declaration of the Options class designed to manage user-specified
 *	options for solving a QProblem.
 */


#ifndef QPOASES_FLIPPER_H
#define QPOASES_FLIPPER_H


#include <qpOASES_e/Bounds.h>
#include <qpOASES_e/Constraints.h>


BEGIN_NAMESPACE_QPOASES


/**
 *	\brief Auxiliary class for storing a copy of the current matrix factorisations.
 *
 *	This auxiliary class stores a copy of the current matrix factorisations. It
 *	is used by the classe QProblemB and QProblem in case flipping bounds are enabled.
 *
 *	\author Hans Joachim Ferreau, Andreas Potschka, Christian Kirches
 *	\version 3.1embedded
 *	\date 2007-2015
 */
typedef struct
{
	Bounds      *bounds;			/**< Data structure for problem's bounds. */
	Constraints *constraints;		/**< Data structure for problem's constraints. */

	real_t *R;						/**< Cholesky factor of H (i.e. H = R^T*R). */
	real_t *Q;						/**< Orthonormal quadratic matrix, A = [0 T]*Q'. */
	real_t *T;						/**< Reverse triangular matrix, A = [0 T]*Q'. */

	unsigned int nV;				/**< Number of variables. */
	unsigned int nC;				/**< Number of constraints. */
} Flipper;

int Flipper_calculateMemorySize( unsigned int nV, unsigned int nC );

char *Flipper_assignMemory( unsigned int nV, unsigned int nC, Flipper **mem, void *raw_memory );

Flipper *Flipper_createMemory( unsigned int nV, unsigned int nC );

/** Constructor which takes the number of bounds and constraints. */
void FlipperCON(	Flipper* _THIS,
					unsigned int _nV,		/**< Number of bounds. */
					unsigned int _nC		/**< Number of constraints. */
					);

/** Copy constructor (deep copy). */
void FlipperCPY(	Flipper* FROM,
					Flipper* TO
					);

/** Initialises object with given number of bounds and constraints.
 *	\return SUCCESSFUL_RETURN \n
 			RET_INVALID_ARGUMENTS */
returnValue Flipper_init(	Flipper* _THIS,
							unsigned int _nV,	/**< Number of bounds. */
							unsigned int _nC	/**< Number of constraints. */
							);


/** Copies current values to non-null arguments (assumed to be allocated with consistent size).
 *	\return SUCCESSFUL_RETURN */
returnValue Flipper_get(	Flipper* _THIS,
							Bounds* const _bounds,				/**< Pointer to new bounds. */
							real_t* const R,					/**< New matrix R. */
							Constraints* const _constraints,	/**< Pointer to new constraints. */
							real_t* const _Q,					/**< New matrix Q. */
							real_t* const _T					/**< New matrix T. */
							);

/** Assigns new values to non-null arguments.
 *	\return SUCCESSFUL_RETURN */
returnValue Flipper_set(	Flipper* _THIS,
							const Bounds* const _bounds,			/**< Pointer to new bounds. */
							const real_t* const _R,					/**< New matrix R. */
							const Constraints* const _constraints,	/**< Pointer to new constraints. */
							const real_t* const _Q,					/**< New matrix Q. */
							const real_t* const _T					/**< New matrix T. */
							);

/** Returns dimension of matrix T.
 *  \return Dimension of matrix T. */
unsigned int Flipper_getDimT( Flipper* _THIS );


END_NAMESPACE_QPOASES


#endif	/* QPOASES_FLIPPER_H */


/*
 *	end of file
 */
