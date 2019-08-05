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
 *	\file INCLUDE/CyclingManager.hpp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Declaration of the CyclingManager class designed to detect
 *	and handle possible cycling during QP iterations.
 */


#ifndef QPOASES_CYCLINGMANAGER_HPP
#define QPOASES_CYCLINGMANAGER_HPP


#include <Utils.hpp>



/** This class is intended to detect and handle possible cycling during QP iterations.
 *	As cycling seems to occur quite rarely, this class is NOT FULLY IMPLEMENTED YET!
 *
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 */
class CyclingManager
{
	/*
	 *	PUBLIC MEMBER FUNCTIONS
	 */
	public:
		/** Default constructor. */
		CyclingManager( );

		/** Copy constructor (deep copy). */
		CyclingManager(	const CyclingManager& rhs	/**< Rhs object. */
						);

		/** Destructor. */
		~CyclingManager( );

		/** Copy asingment operator (deep copy). */
		CyclingManager& operator=(	const CyclingManager& rhs	/**< Rhs object. */
									);


		/** Pseudo-constructor which takes the number of bounds/constraints.
		 *	\return SUCCESSFUL_RETURN */
		returnValue init(	int _nV,	/**< Number of bounds to be managed. */
							int _nC		/**< Number of constraints to be managed. */
							);


		/** Stores index of a bound/constraint that might cause cycling.
		 *	\return SUCCESSFUL_RETURN \n
					RET_INDEX_OUT_OF_BOUNDS */
		returnValue setCyclingStatus(	int number,				/**< Number of bound/constraint. */
										BooleanType isBound,	/**< Flag that indicates if given number corresponds to a
																 *   bound (BT_TRUE) or a constraint (BT_FALSE). */
										CyclingStatus _status	/**< Cycling status of bound/constraint. */
										);

		/** Returns if bound/constraint might cause cycling.
		 *	\return BT_TRUE: bound/constraint might cause cycling \n
		 			BT_FALSE: otherwise */
		CyclingStatus getCyclingStatus(	int number,			/**< Number of bound/constraint. */
										BooleanType isBound	/**< Flag that indicates if given number corresponds to
															 *   a bound (BT_TRUE) or a constraint (BT_FALSE). */
										) const;


		/** Clears all previous cycling information.
		 *	\return SUCCESSFUL_RETURN */
		returnValue clearCyclingData( );


		/** Returns if cycling was detected.
		 *	\return BT_TRUE iff cycling was detected. */
		inline BooleanType isCyclingDetected( ) const;


	/*
	 *	PROTECTED MEMBER VARIABLES
	 */
	protected:
		int	nV;									/**< Number of managed bounds. */
		int	nC;									/**< Number of managed constraints. */

		CyclingStatus status[NVMAX+NCMAX];		/**< Array to store cycling status of all bounds/constraints. */

		BooleanType cyclingDetected;			/**< Flag if cycling was detected. */
};


#include <CyclingManager.ipp>

#endif	/* QPOASES_CYCLINGMANAGER_HPP */


/*
 *	end of file
 */
