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
 *	\file INCLUDE/Bounds.hpp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Declaration of the Bounds class designed to manage working sets of
 *	bounds within a QProblem.
 */


#ifndef QPOASES_BOUNDS_HPP
#define QPOASES_BOUNDS_HPP


#include <SubjectTo.hpp>



/** This class manages working sets of bounds by storing
 *	index sets and other status information.
 *
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 */
class Bounds : public SubjectTo
{
	/*
	 *	PUBLIC MEMBER FUNCTIONS
	 */
	public:
		/** Default constructor. */
		Bounds( );

		/** Copy constructor (deep copy). */
		Bounds(	const Bounds& rhs	/**< Rhs object. */
				);

		/** Destructor. */
		~Bounds( );

		/** Assignment operator (deep copy). */
		Bounds& operator=(	const Bounds& rhs	/**< Rhs object. */
							);


		/** Pseudo-constructor takes the number of bounds.
		 *	\return SUCCESSFUL_RETURN */
		returnValue init(	int n	/**< Number of bounds. */
							);


		/** Initially adds number of a new (i.e. not yet in the list) bound to
		 *  given index set.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_SETUP_BOUND_FAILED \n
					RET_INDEX_OUT_OF_BOUNDS \n
					RET_INVALID_ARGUMENTS */
		returnValue setupBound(	int _number,					/**< Number of new bound. */
								SubjectToStatus _status			/**< Status of new bound. */
								);

		/** Initially adds all numbers of new (i.e. not yet in the list) bounds to
		 *  to the index set of free bounds; the order depends on the SujectToType
		 *  of each index.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_SETUP_BOUND_FAILED */
		returnValue setupAllFree( );


		/** Moves index of a bound from index list of fixed to that of free bounds.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_MOVING_BOUND_FAILED \n
					RET_INDEX_OUT_OF_BOUNDS */
		returnValue moveFixedToFree(	int _number				/**< Number of bound to be freed. */
										);

		/** Moves index of a bound from index list of free to that of fixed bounds.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_MOVING_BOUND_FAILED \n
					RET_INDEX_OUT_OF_BOUNDS */
		returnValue moveFreeToFixed(	int _number,			/**< Number of bound to be fixed. */
										SubjectToStatus _status	/**< Status of bound to be fixed. */
										);

		/** Swaps the indices of two free bounds within the index set.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_SWAPINDEX_FAILED */
		returnValue swapFree(	int number1,					/**< Number of first constraint or bound. */
								int number2						/**< Number of second constraint or bound. */
								);


		/** Returns number of variables.
		 *	\return Number of variables. */
		inline int getNV( ) const;

		/** Returns number of implicitly fixed variables.
		 *	\return Number of implicitly fixed variables. */
		inline int getNFV( ) const;

		/** Returns number of bounded (but possibly free) variables.
		 *	\return Number of bounded (but possibly free) variables. */
		inline int getNBV( ) const;

		/** Returns number of unbounded variables.
		 *	\return Number of unbounded variables. */
		inline int getNUV( ) const;


		/** Sets number of implicitly fixed variables.
		 *	\return SUCCESSFUL_RETURN */
		inline returnValue setNFV(	int n	/**< Number of implicitly fixed variables. */
									);

		/** Sets number of bounded (but possibly free) variables.
		 *	\return SUCCESSFUL_RETURN */
		inline returnValue setNBV(	int n	/**< Number of bounded (but possibly free) variables. */
									);

		/** Sets number of unbounded variables.
		 *	\return SUCCESSFUL_RETURN */
		inline returnValue setNUV(	int n	/**< Number of unbounded variables */
									);


		/** Returns number of free variables.
		 *	\return Number of free variables. */
		inline int getNFR( );

		/** Returns number of fixed variables.
		 *	\return Number of fixed variables. */
		inline int getNFX( );


		/** Returns a pointer to free variables index list.
		 *	\return Pointer to free variables index list. */
		inline Indexlist* getFree( );

		/** Returns a pointer to fixed variables index list.
		 *	\return Pointer to fixed variables index list. */
		inline Indexlist* getFixed( );


	/*
	 *	PROTECTED MEMBER VARIABLES
	 */
	protected:
		int nV;					/**< Number of variables (nV = nFV + nBV + nUV). */
		int nFV;				/**< Number of implicitly fixed variables. */
		int	nBV;				/**< Number of bounded (but possibly free) variables. */
		int nUV;				/**< Number of unbounded variables. */

		Indexlist free;			/**< Index list of free variables. */
		Indexlist fixed;		/**< Index list of fixed variables. */
};

#include <Bounds.ipp>

#endif	/* QPOASES_BOUNDS_HPP */


/*
 *	end of file
 */
