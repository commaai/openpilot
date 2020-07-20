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
 *	\file INCLUDE/Indexlist.hpp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Declaration of the Indexlist class designed to manage index lists of
 *	constraints and bounds within a SubjectTo object.
 */


#ifndef QPOASES_INDEXLIST_HPP
#define QPOASES_INDEXLIST_HPP


#include <Utils.hpp>


/** This class manages index lists.
 *
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 */
class Indexlist
{
	/*
	 *	PUBLIC MEMBER FUNCTIONS
	 */
	public:
		/** Default constructor. */
		Indexlist( );

		/** Copy constructor (deep copy). */
		Indexlist(	const Indexlist& rhs	/**< Rhs object. */
					);

		/** Destructor. */
		~Indexlist( );

		/** Assingment operator (deep copy). */
		Indexlist& operator=(	const Indexlist& rhs	/**< Rhs object. */
								);

		/** Pseudo-constructor.
		 *	\return SUCCESSFUL_RETURN */
		returnValue init( );


		/** Creates an array of all numbers within the index set in correct order.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_INDEXLIST_CORRUPTED */
		returnValue	getNumberArray(	int* const numberarray	/**< Output: Array of numbers (NULL on error). */
									) const;


		/** Determines the index within the index list at with a given number is stored.
		 *	\return >= 0: Index of given number. \n
		 			-1: Number not found. */
		int	getIndex(	int givennumber	/**< Number whose index shall be determined. */
						) const;

		/** Determines the physical index within the index list at with a given number is stored.
		 *	\return >= 0: Index of given number. \n
		 			-1: Number not found. */
		int	getPhysicalIndex(	int givennumber	/**< Number whose physical index shall be determined. */
								) const;

		/** Returns the number stored at a given physical index.
		 *	\return >= 0: Number stored at given physical index. \n
		 			-RET_INDEXLIST_OUTOFBOUNDS */
		int	getNumber(	int physicalindex	/**< Physical index of the number to be returned. */
						) const;


		/** Returns the current length of the index list.
		 *	\return Current length of the index list. */
		inline int getLength( );

		/** Returns last number within the index list.
		 *	\return Last number within the index list. */
		inline int getLastNumber( ) const;


		/** Adds number to index list.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_INDEXLIST_MUST_BE_REORDERD \n
		 			RET_INDEXLIST_EXCEEDS_MAX_LENGTH */
		returnValue addNumber(	int addnumber	/**< Number to be added. */
								);

		/** Removes number from index list.
		 *	\return SUCCESSFUL_RETURN */
		returnValue removeNumber(	int removenumber	/**< Number to be removed. */
									);

		/** Swaps two numbers within index list.
		 *	\return SUCCESSFUL_RETURN */
		returnValue swapNumbers(	int number1,/**< First number for swapping. */
									int number2	/**< Second number for swapping. */
									);

		/** Determines if a given number is contained in the index set.
		 *	\return BT_TRUE iff number is contain in the index set */
		inline BooleanType isMember(	int _number	/**< Number to be tested for membership. */
										) const;


	/*
	 *	PROTECTED MEMBER VARIABLES
	 */
	protected:
		int number[INDEXLISTFACTOR*(NVMAX+NCMAX)];		/**< Array to store numbers of constraints or bounds. */
		int next[INDEXLISTFACTOR*(NVMAX+NCMAX)];		/**< Array to store physical index of successor. */
		int previous[INDEXLISTFACTOR*(NVMAX+NCMAX)];	/**< Array to store physical index of predecossor. */
		int	length;										/**< Length of index list. */
		int	first;										/**< Physical index of first element. */
		int	last;										/**< Physical index of last element. */
		int	lastusedindex;								/**< Physical index of last entry in index list. */
		int	physicallength;								/**< Physical length of index list. */
};


#include <Indexlist.ipp>

#endif	/* QPOASES_INDEXLIST_HPP */


/*
 *	end of file
 */
