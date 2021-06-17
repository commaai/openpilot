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
 *	\file include/qpOASES_e/Indexlist.h
 *	\author Hans Joachim Ferreau, Andreas Potschka, Christian Kirches
 *	\version 3.1embedded
 *	\date 2007-2015
 *
 *	Declaration of the Indexlist class designed to manage index lists of
 *	constraints and bounds within a SubjectTo object.
 */


#ifndef QPOASES_INDEXLIST_H
#define QPOASES_INDEXLIST_H


#include <qpOASES_e/Utils.h>


BEGIN_NAMESPACE_QPOASES


/**
 *	\brief Stores and manages index lists.
 *
 *	This class manages index lists of active/inactive bounds/constraints.
 *
 *	\author Hans Joachim Ferreau
 *	\version 3.1embedded
 *	\date 2007-2015
 */
typedef struct
{
	int *number;		/**< Array to store numbers of constraints or bounds. */
	int *iSort;			/**< Index list to sort vector \a number */

	int	length;			/**< Length of index list. */
	int	first;			/**< Physical index of first element. */
	int	last;			/**< Physical index of last element. */
	int	lastusedindex;	/**< Physical index of last entry in index list. */
	int	physicallength;	/**< Physical length of index list. */
} Indexlist;

int Indexlist_calculateMemorySize( int n);

char *Indexlist_assignMemory(int n, Indexlist **mem, void *raw_memory);

Indexlist *Indexlist_createMemory( int n );

/** Constructor which takes the desired physical length of the index list. */
void IndexlistCON(	Indexlist* _THIS,
					int n	/**< Physical length of index list. */
					);

/** Copies all members from given rhs object.
 *  \return SUCCESSFUL_RETURN */
void IndexlistCPY(	Indexlist* FROM,
					Indexlist* TO
					);

/** Initialises index list of desired physical length.
 *	\return SUCCESSFUL_RETURN \n
 			RET_INVALID_ARGUMENTS */
returnValue Indexlist_init(	Indexlist* _THIS,
							int n		/**< Physical length of index list. */
							);

/** Creates an array of all numbers within the index set in correct order.
 *	\return SUCCESSFUL_RETURN \n
 			RET_INDEXLIST_CORRUPTED */
returnValue Indexlist_getNumberArray(	Indexlist* _THIS,
										int** const numberarray	/**< Output: Array of numbers (NULL on error). */
										);

/** Creates an array of all numbers within the index set in correct order.
 *	\return SUCCESSFUL_RETURN \n
			RET_INDEXLIST_CORRUPTED */
returnValue	Indexlist_getISortArray(	Indexlist* _THIS,
										int** const iSortArray	/**< Output: iSort Array. */
										);


/** Determines the index within the index list at which a given number is stored.
 *	\return >= 0: Index of given number. \n
 			-1: Number not found. */
int Indexlist_getIndex(	Indexlist* _THIS,
						int givennumber	/**< Number whose index shall be determined. */
						);

/** Returns the number stored at a given physical index.
 *	\return >= 0: Number stored at given physical index. \n
 			-RET_INDEXLIST_OUTOFBOUNDS */
static inline int Indexlist_getNumber(	Indexlist* _THIS,
										int physicalindex	/**< Physical index of the number to be returned. */
										);


/** Returns the current length of the index list.
 *	\return Current length of the index list. */
static inline int Indexlist_getLength(	Indexlist* _THIS
										);

/** Returns last number within the index list.
 *	\return Last number within the index list. */
static inline int Indexlist_getLastNumber(	Indexlist* _THIS
											);


/** Adds number to index list.
 *	\return SUCCESSFUL_RETURN \n
 			RET_INDEXLIST_MUST_BE_REORDERD \n
 			RET_INDEXLIST_EXCEEDS_MAX_LENGTH */
returnValue Indexlist_addNumber(	Indexlist* _THIS,
									int addnumber			/**< Number to be added. */
									);

/** Removes number from index list.
 *	\return SUCCESSFUL_RETURN */
returnValue Indexlist_removeNumber(	Indexlist* _THIS,
									int removenumber	/**< Number to be removed. */
									);

/** Swaps two numbers within index list.
 *	\return SUCCESSFUL_RETURN */
returnValue Indexlist_swapNumbers(	Indexlist* _THIS,
									int number1,		/**< First number for swapping. */
									int number2			/**< Second number for swapping. */
									);

/** Determines if a given number is contained in the index set.
 *	\return BT_TRUE iff number is contain in the index set */
static inline BooleanType Indexlist_isMember(	Indexlist* _THIS,
												int _number		/**< Number to be tested for membership. */
												);


/** Find first index j between -1 and length in sorted list of indices
 *  iSort such that numbers[iSort[j]] <= i < numbers[iSort[j+1]]. Uses
 *  bisection.
 *  \return j. */
int Indexlist_findInsert(	Indexlist* _THIS,
							int i
							);



/*
 *	g e t N u m b e r
 */
static inline int Indexlist_getNumber( Indexlist* _THIS, int physicalindex )
{
	/* consistency check */
	if ( ( physicalindex < 0 ) || ( physicalindex > _THIS->length ) )
		return -RET_INDEXLIST_OUTOFBOUNDS;

	return _THIS->number[physicalindex];
}


/*
 *	g e t L e n g t h
 */
static inline int Indexlist_getLength( Indexlist* _THIS )
{
	return _THIS->length;
}


/*
 *	g e t L a s t N u m b e r
 */
static inline int Indexlist_getLastNumber( Indexlist* _THIS )
{
	return _THIS->number[_THIS->length-1];
}


/*
 *	g e t L a s t N u m b e r
 */
static inline BooleanType Indexlist_isMember( Indexlist* _THIS, int _number )
{
	if ( Indexlist_getIndex( _THIS,_number ) >= 0 )
		return BT_TRUE;
	else
		return BT_FALSE;
}


END_NAMESPACE_QPOASES


#endif	/* QPOASES_INDEXLIST_H */


/*
 *	end of file
 */
