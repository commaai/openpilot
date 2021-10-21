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
 *	\file include/qpOASES_e/Bounds.h
 *	\author Hans Joachim Ferreau, Andreas Potschka, Christian Kirches
 *	\version 3.1embedded
 *	\date 2007-2015
 *
 *	Declaration of the Bounds class designed to manage working sets of
 *	bounds within a QProblem.
 */


#ifndef QPOASES_BOUNDS_H
#define QPOASES_BOUNDS_H


#include <qpOASES_e/Indexlist.h>


BEGIN_NAMESPACE_QPOASES


/**
 *	\brief Manages working sets of bounds (= box constraints).
 *
 *	This class manages working sets of bounds (= box constraints)
 *	by storing index sets and other status information.
 *
 *	\author Hans Joachim Ferreau
 *	\version 3.1embedded
 *	\date 2007-2015
 */
typedef struct
{
	Indexlist *freee;					/**< Index list of free variables. */
	Indexlist *fixed;					/**< Index list of fixed variables. */

	Indexlist *shiftedFreee;			/**< Memory for shifting free variables. */
	Indexlist *shiftedFixed;			/**< Memory for shifting fixed variables. */

	Indexlist *rotatedFreee;			/**< Memory for rotating free variables. */
	Indexlist *rotatedFixed;			/**< Memory for rotating fixed variables. */

	SubjectToType   *type; 				/**< Type of bounds. */
	SubjectToStatus *status;			/**< Status of bounds. */

	SubjectToType   *typeTmp;			/**< Temp memory for type of bounds. */
	SubjectToStatus *statusTmp;			/**< Temp memory for status of bounds. */

	BooleanType noLower;	 			/**< This flag indicates if there is no lower bound on any variable. */
	BooleanType noUpper;	 			/**< This flag indicates if there is no upper bound on any variable. */

	int n;								/**< Total number of bounds. */
} Bounds;

int Bounds_calculateMemorySize( int n);

char *Bounds_assignMemory(int n, Bounds **mem, void *raw_memory);

Bounds *Bounds_createMemory( int n );

/** Constructor which takes the number of bounds. */
void BoundsCON(	Bounds* _THIS,
				int _n									/**< Number of bounds. */
				);

/** Copies all members from given rhs object.
 *  \return SUCCESSFUL_RETURN */
void BoundsCPY(	Bounds* FROM,
				Bounds* TO
				);


/** Initialises object with given number of bounds.
 *	\return SUCCESSFUL_RETURN \n
 			RET_INVALID_ARGUMENTS */
returnValue Bounds_init(	Bounds* _THIS,
							int _n					/**< Number of bounds. */
							);


/** Initially adds number of a new (i.e. not yet in the list) bound to
 *  given index set.
 *	\return SUCCESSFUL_RETURN \n
 			RET_SETUP_BOUND_FAILED \n
			RET_INDEX_OUT_OF_BOUNDS \n
			RET_INVALID_ARGUMENTS */
returnValue Bounds_setupBound(	Bounds* _THIS,
								int number,				/**< Number of new bound. */
								SubjectToStatus _status	/**< Status of new bound. */
								);

/** Initially adds all numbers of new (i.e. not yet in the list) bounds to
 *  to the index set of free bounds; the order depends on the SujectToType
 *  of each index.
 *	\return SUCCESSFUL_RETURN \n
 			RET_SETUP_BOUND_FAILED */
returnValue Bounds_setupAllFree(	Bounds* _THIS
									);

/** Initially adds all numbers of new (i.e. not yet in the list) bounds to
 *  to the index set of fixed bounds (on their lower bounds);
 *  the order depends on the SujectToType of each index.
 *	\return SUCCESSFUL_RETURN \n
 			RET_SETUP_BOUND_FAILED */
returnValue Bounds_setupAllLower(	Bounds* _THIS
									);

/** Initially adds all numbers of new (i.e. not yet in the list) bounds to
 *  to the index set of fixed bounds (on their upper bounds);
 *  the order depends on the SujectToType of each index.
 *	\return SUCCESSFUL_RETURN \n
 			RET_SETUP_BOUND_FAILED */
returnValue Bounds_setupAllUpper(	Bounds* _THIS
									);


/** Moves index of a bound from index list of fixed to that of free bounds.
 *	\return SUCCESSFUL_RETURN \n
 			RET_MOVING_BOUND_FAILED \n
			RET_INDEX_OUT_OF_BOUNDS */
returnValue Bounds_moveFixedToFree(	Bounds* _THIS,
									int number				/**< Number of bound to be freed. */
									);

/** Moves index of a bound from index list of free to that of fixed bounds.
 *	\return SUCCESSFUL_RETURN \n
 			RET_MOVING_BOUND_FAILED \n
			RET_INDEX_OUT_OF_BOUNDS */
returnValue Bounds_moveFreeToFixed(	Bounds* _THIS,
									int number,				/**< Number of bound to be fixed. */
									SubjectToStatus _status	/**< Status of bound to be fixed. */
									);

/** Flip fixed bound.
 *	\return SUCCESSFUL_RETURN \n
 			RET_MOVING_BOUND_FAILED \n
			RET_INDEX_OUT_OF_BOUNDS */
returnValue Bounds_flipFixed(	Bounds* _THIS,
								int number
								);

/** Swaps the indices of two free bounds within the index set.
 *	\return SUCCESSFUL_RETURN \n
 			RET_SWAPINDEX_FAILED */
returnValue Bounds_swapFree(	Bounds* _THIS,
								int number1,					/**< Number of first bound. */
								int number2						/**< Number of second bound. */
								);


/** Returns number of variables.
 *	\return Number of variables. */
static inline int Bounds_getNV(	Bounds* _THIS
							);

/** Returns number of implicitly fixed variables.
 *	\return Number of implicitly fixed variables. */
static inline int Bounds_getNFV(	Bounds* _THIS
									);

/** Returns number of bounded (but possibly free) variables.
 *	\return Number of bounded (but possibly free) variables. */
static inline int Bounds_getNBV(	Bounds* _THIS
									);

/** Returns number of unbounded variables.
 *	\return Number of unbounded variables. */
static inline int Bounds_getNUV(	Bounds* _THIS
									);

/** Returns number of free variables.
 *	\return Number of free variables. */
static inline int Bounds_getNFR(	Bounds* _THIS
									);

/** Returns number of fixed variables.
 *	\return Number of fixed variables. */
static inline int Bounds_getNFX(	Bounds* _THIS
									);


/** Returns a pointer to free variables index list.
 *	\return Pointer to free variables index list. */
static inline Indexlist* Bounds_getFree(	Bounds* _THIS
											);

/** Returns a pointer to fixed variables index list.
 *	\return Pointer to fixed variables index list. */
static inline Indexlist* Bounds_getFixed(	Bounds* _THIS
											);


/** Returns number of bounds with given SubjectTo type.
 *	\return Number of bounds with given type. */
static inline int Bounds_getNumberOfType(	Bounds* _THIS,
											SubjectToType _type	/**< Type of bound. */
											);


/** Returns type of bound.
 *	\return Type of bound \n
 			RET_INDEX_OUT_OF_BOUNDS */
static inline SubjectToType Bounds_getType(	Bounds* _THIS,
											int i			/**< Number of bound. */
											);

/** Returns status of bound.
 *	\return Status of bound \n
 			ST_UNDEFINED */
static inline SubjectToStatus Bounds_getStatus(	Bounds* _THIS,
												int i		/**< Number of bound. */
												);


/** Sets type of bound.
 *	\return SUCCESSFUL_RETURN \n
 			RET_INDEX_OUT_OF_BOUNDS */
static inline returnValue Bounds_setType(	Bounds* _THIS,
											int i,				/**< Number of bound. */
											SubjectToType value	/**< Type of bound. */
											);

/** Sets status of bound.
 *	\return SUCCESSFUL_RETURN \n
 			RET_INDEX_OUT_OF_BOUNDS */
static inline returnValue Bounds_setStatus(	Bounds* _THIS,
											int i,					/**< Number of bound. */
											SubjectToStatus value	/**< Status of bound. */
											);


/** Sets status of lower bounds. */
static inline void Bounds_setNoLower(	Bounds* _THIS,
										BooleanType _status		/**< Status of lower bounds. */
										);

/** Sets status of upper bounds. */
static inline void Bounds_setNoUpper(	Bounds* _THIS,
										BooleanType _status		/**< Status of upper bounds. */
										);


/** Returns status of lower bounds.
 *	\return BT_TRUE if there is no lower bound on any variable. */
static inline BooleanType Bounds_hasNoLower(	Bounds* _THIS
												);

/** Returns status of upper bounds.
 *	\return BT_TRUE if there is no upper bound on any variable. */
static inline BooleanType Bounds_hasNoUpper(	Bounds* _THIS
												);


/** Shifts forward type and status of all bounds by a given
 *  offset. This offset has to lie within the range [0,n/2] and has to
 *  be an integer divisor of the total number of bounds n.
 *  Type and status of the first \<offset\> bounds is thrown away,
 *  type and status of the last \<offset\> bounds is doubled,
 *  e.g. for offset = 2: \n
 *  shift( {b1,b2,b3,b4,b5,b6} ) = {b3,b4,b5,b6,b5,b6}
 *	\return SUCCESSFUL_RETURN \n
 			RET_INDEX_OUT_OF_BOUNDS \n
 			RET_INVALID_ARGUMENTS \n
 			RET_SHIFTING_FAILED */
returnValue Bounds_shift(	Bounds* _THIS,
							int offset		/**< Shift offset within the range [0,n/2] and integer divisor of n. */
							);

/** Rotates forward type and status of all bounds by a given
 *  offset. This offset has to lie within the range [0,n].
 *  Example for offset = 2: \n
 *  rotate( {b1,b2,b3,b4,b5,b6} ) = {b3,b4,b5,b6,b1,b2}
 *	\return SUCCESSFUL_RETURN \n
 			RET_INDEX_OUT_OF_BOUNDS \n
 			RET_ROTATING_FAILED */
returnValue Bounds_rotate(	Bounds* _THIS,
							int offset		/**< Rotation offset within the range [0,n]. */
							);


/** Prints information on bounds object
 *  (in particular, lists of free and fixed bounds.
 * \return SUCCESSFUL_RETURN \n
		   RET_INDEXLIST_CORRUPTED */
returnValue Bounds_print(	Bounds* _THIS
							);


/** Initially adds all numbers of new (i.e. not yet in the list) bounds to
 *  to the index set corresponding to the desired status;
 *  the order depends on the SujectToType of each index.
 *	\return SUCCESSFUL_RETURN \n
 			RET_SETUP_BOUND_FAILED */
returnValue Bounds_setupAll(	Bounds* _THIS,
								SubjectToStatus _status	/**< Desired initial status for all bounds. */
								);


/** Adds the index of a new bound to index set.
 *	\return SUCCESSFUL_RETURN \n
 			RET_ADDINDEX_FAILED \n
			RET_INVALID_ARGUMENTS */
returnValue Bounds_addIndex(	Bounds* _THIS,
								Indexlist* const indexlist,	/**< Index list to which the new index shall be added. */
								int newnumber,				/**< Number of new bound. */
								SubjectToStatus newstatus	/**< Status of new bound. */
								);

/** Removes the index of a bound from index set.
 *	\return SUCCESSFUL_RETURN \n
 			RET_REMOVEINDEX_FAILED \n
			RET_INVALID_ARGUMENTS */
returnValue Bounds_removeIndex(	Bounds* _THIS,
								Indexlist* const indexlist,	/**< Index list from which the new index shall be removed. */
								int removenumber			/**< Number of bound to be removed. */
								);

/** Swaps the indices of two constraints or bounds within the index set.
 *	\return SUCCESSFUL_RETURN \n
 			RET_SWAPINDEX_FAILED \n
			RET_INVALID_ARGUMENTS */
returnValue Bounds_swapIndex(	Bounds* _THIS,
								Indexlist* const indexlist,	/**< Index list in which the indices shold be swapped. */
								int number1,				/**< Number of first bound. */
								int number2					/**< Number of second bound. */
								);



/*
 *	g e t N u m b e r O f T y p e
 */
static inline int Bounds_getNumberOfType( Bounds* _THIS, SubjectToType _type )
{
	int i;
	int numberOfType = 0;

	if ( _THIS->type != 0 )
	{
		for( i=0; i<_THIS->n; ++i )
			if ( _THIS->type[i] == _type )
				++numberOfType;
	}

	return numberOfType;
}


/*
 *	g e t T y p e
 */
static inline SubjectToType Bounds_getType( Bounds* _THIS, int i )
{
	if ( ( i >= 0 ) && ( i < _THIS->n ) )
		return _THIS->type[i];

	return ST_UNKNOWN;
}


/*
 *	g e t S t a t u s
 */
static inline SubjectToStatus Bounds_getStatus( Bounds* _THIS, int i )
{
	if ( ( i >= 0 ) && ( i < _THIS->n ) )
		return _THIS->status[i];

	return ST_UNDEFINED;
}


/*
 *	s e t T y p e
 */
static inline returnValue Bounds_setType( Bounds* _THIS, int i, SubjectToType value )
{
	if ( ( i >= 0 ) && ( i < _THIS->n ) )
	{
		_THIS->type[i] = value;
		return SUCCESSFUL_RETURN;
	}
	else
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );
}


/*
 *	s e t S t a t u s
 */
static inline returnValue Bounds_setStatus( Bounds* _THIS, int i, SubjectToStatus value )
{
	if ( ( i >= 0 ) && ( i < _THIS->n ) )
	{
		_THIS->status[i] = value;
		return SUCCESSFUL_RETURN;
	}
	else
		return THROWERROR( RET_INDEX_OUT_OF_BOUNDS );
}


/*
 *	s e t N o L o w e r
 */
static inline void Bounds_setNoLower( Bounds* _THIS, BooleanType _status )
{
	_THIS->noLower = _status;
}


/*
 *	s e t N o U p p e r
 */
static inline void Bounds_setNoUpper( Bounds* _THIS, BooleanType _status )
{
	_THIS->noUpper = _status;
}


/*
 *	h a s N o L o w e r
 */
static inline BooleanType Bounds_hasNoLower( Bounds* _THIS )
{
	return _THIS->noLower;
}


/*
 *	h a s N o U p p p e r
 */
static inline BooleanType Bounds_hasNoUpper( Bounds* _THIS )
{
	return _THIS->noUpper;
}



/*
 *	g e t N V
 */
static inline int Bounds_getNV( Bounds* _THIS )
{
 	return _THIS->n;
}


/*
 *	g e t N F V
 */
static inline int Bounds_getNFV( Bounds* _THIS )
{
 	return Bounds_getNumberOfType( _THIS,ST_EQUALITY );
}


/*
 *	g e t N B V
 */
static inline int Bounds_getNBV( Bounds* _THIS )
{
 	return Bounds_getNumberOfType( _THIS,ST_BOUNDED );
}


/*
 *	g e t N U V
 */
static inline int Bounds_getNUV( Bounds* _THIS )
{
	return Bounds_getNumberOfType( _THIS,ST_UNBOUNDED );
}


/*
 *	g e t N F R
 */
static inline int Bounds_getNFR( Bounds* _THIS )
{
 	return Indexlist_getLength( _THIS->freee );
}


/*
 *	g e t N F X
 */
static inline int Bounds_getNFX( Bounds* _THIS )
{
	return Indexlist_getLength( _THIS->fixed );
}


/*
 *	g e t F r e e
 */
static inline Indexlist* Bounds_getFree( Bounds* _THIS )
{
	return _THIS->freee;
}


/*
 *	g e t F i x e d
 */
static inline Indexlist* Bounds_getFixed( Bounds* _THIS )
{
	return _THIS->fixed;
}


END_NAMESPACE_QPOASES

#endif	/* QPOASES_BOUNDS_H */


/*
 *	end of file
 */
