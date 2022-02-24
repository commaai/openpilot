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
 *	\file include/qpOASES_e/Constraints.h
 *	\author Hans Joachim Ferreau, Andreas Potschka, Christian Kirches
 *	\version 3.1embedded
 *	\date 2007-2015
 *
 *	Declaration of the Constraints class designed to manage working sets of
 *	constraints within a QProblem.
 */


#ifndef QPOASES_CONSTRAINTS_H
#define QPOASES_CONSTRAINTS_H


#include <qpOASES_e/Indexlist.h>


BEGIN_NAMESPACE_QPOASES


/**
 *	\brief Manages working sets of constraints.
 *
 *	This class manages working sets of constraints by storing
 *	index sets and other status information.
 *
 *	\author Hans Joachim Ferreau
 *	\version 3.1embedded
 *	\date 2007-2015
 */
typedef struct
{
	Indexlist *active;					/**< Index list of active constraints. */
	Indexlist *inactive;				/**< Index list of inactive constraints. */

	Indexlist *shiftedActive;			/**< Memory for shifting active constraints. */
	Indexlist *shiftedInactive;			/**< Memory for shifting inactive constraints. */

	Indexlist *rotatedActive;			/**< Memory for rotating active constraints. */
	Indexlist *rotatedInactive;			/**< Memory for rotating inactive constraints. */

	SubjectToType   *type; 				/**< Type of constraints. */
	SubjectToStatus *status;			/**< Status of constraints. */

	SubjectToType   *typeTmp;			/**< Temp memory for type of constraints. */
	SubjectToStatus *statusTmp;			/**< Temp memory for status of constraints. */

	BooleanType noLower;			 	/**< This flag indicates if there is no lower bound on any variable. */
	BooleanType noUpper;	 			/**< This flag indicates if there is no upper bound on any variable. */

	int n;								/**< Total number of constraints. */
} Constraints;

int Constraints_calculateMemorySize( int n);

char *Constraints_assignMemory(int n, Constraints **mem, void *raw_memory);

Constraints *Constraints_createMemory( int n );

/** Constructor which takes the number of constraints. */
void ConstraintsCON(	Constraints* _THIS,
						int _n							/**< Number of constraints. */
						);

/** Copies all members from given rhs object.
 *  \return SUCCESSFUL_RETURN */
void ConstraintsCPY(	Constraints* FROM,
						Constraints* TO
						);


/** Initialises object with given number of constraints.
 *	\return SUCCESSFUL_RETURN \n
 			RET_INVALID_ARGUMENTS */
returnValue Constraints_init(	Constraints* _THIS,
								int _n					/**< Number of constraints. */
								);


/** Initially adds number of a new (i.e. not yet in the list) constraint to
 *  a given index set.
 *	\return SUCCESSFUL_RETURN \n
 			RET_SETUP_CONSTRAINT_FAILED \n
			RET_INDEX_OUT_OF_BOUNDS \n
			RET_INVALID_ARGUMENTS */
returnValue Constraints_setupConstraint(	Constraints* _THIS,
											int number,				/**< Number of new constraint. */
											SubjectToStatus _status	/**< Status of new constraint. */
											);

/** Initially adds all enabled numbers of new (i.e. not yet in the list) constraints to
 *  to the index set of inactive constraints; the order depends on the SujectToType
 *  of each index. Only disabled constraints are added to index set of disabled constraints!
 *	\return SUCCESSFUL_RETURN \n
 			RET_SETUP_CONSTRAINT_FAILED */
returnValue Constraints_setupAllInactive(	Constraints* _THIS
											);

/** Initially adds all enabled numbers of new (i.e. not yet in the list) constraints to
 *  to the index set of active constraints (on their lower bounds); the order depends on the SujectToType
 *  of each index. Only disabled constraints are added to index set of disabled constraints!
 *	\return SUCCESSFUL_RETURN \n
 			RET_SETUP_CONSTRAINT_FAILED */
returnValue Constraints_setupAllLower(	Constraints* _THIS
										);

/** Initially adds all enabled numbers of new (i.e. not yet in the list) constraints to
 *  to the index set of active constraints (on their upper bounds); the order depends on the SujectToType
 *  of each index. Only disabled constraints are added to index set of disabled constraints!
 *	\return SUCCESSFUL_RETURN \n
 			RET_SETUP_CONSTRAINT_FAILED */
returnValue Constraints_setupAllUpper(	Constraints* _THIS
										);


/** Moves index of a constraint from index list of active to that of inactive constraints.
 *	\return SUCCESSFUL_RETURN \n
 			RET_MOVING_CONSTRAINT_FAILED */
returnValue Constraints_moveActiveToInactive(	Constraints* _THIS,
												int number				/**< Number of constraint to become inactive. */
												);

/** Moves index of a constraint from index list of inactive to that of active constraints.
 *	\return SUCCESSFUL_RETURN \n
 			RET_MOVING_CONSTRAINT_FAILED */
returnValue Constraints_moveInactiveToActive(	Constraints* _THIS,
												int number,				/**< Number of constraint to become active. */
												SubjectToStatus _status	/**< Status of constraint to become active. */
												);

/** Flip fixed constraint.
 *	\return SUCCESSFUL_RETURN \n
 			RET_MOVING_CONSTRAINT_FAILED \n
			RET_INDEX_OUT_OF_BOUNDS */
returnValue Constraints_flipFixed(	Constraints* _THIS,
									int number
									);


/** Returns the number of constraints.
 *	\return Number of constraints. */
static inline int Constraints_getNC(	Constraints* _THIS
										);

/** Returns the number of implicit equality constraints.
 *	\return Number of implicit equality constraints. */
static inline int Constraints_getNEC(	Constraints* _THIS
										);

/** Returns the number of "real" inequality constraints.
 *	\return Number of "real" inequality constraints. */
static inline int Constraints_getNIC(	Constraints* _THIS
										);

/** Returns the number of unbounded constraints (i.e. without any bounds).
 *	\return Number of unbounded constraints (i.e. without any bounds). */
static inline int Constraints_getNUC(	Constraints* _THIS
										);

/** Returns the number of active constraints.
 *	\return Number of active constraints. */
static inline int Constraints_getNAC(	Constraints* _THIS
										);

/** Returns the number of inactive constraints.
 *	\return Number of inactive constraints. */
static inline int Constraints_getNIAC(	Constraints* _THIS
										);


/** Returns a pointer to active constraints index list.
 *	\return Pointer to active constraints index list. */
static inline Indexlist* Constraints_getActive(	Constraints* _THIS
												);

/** Returns a pointer to inactive constraints index list.
 *	\return Pointer to inactive constraints index list. */
static inline Indexlist* Constraints_getInactive(	Constraints* _THIS
													);


/** Returns number of constraints with given SubjectTo type.
 *	\return Number of constraints with given type. */
static inline int Constraints_getNumberOfType(	Constraints* _THIS,
												SubjectToType _type	/**< Type of constraints' bound. */
												);


/** Returns type of constraints' bound.
 *	\return Type of constraints' bound \n
 			RET_INDEX_OUT_OF_BOUNDS */
static inline SubjectToType Constraints_getType(	Constraints* _THIS,
													int i			/**< Number of constraints' bound. */
													);

/** Returns status of constraints' bound.
 *	\return Status of constraints' bound \n
 			ST_UNDEFINED */
static inline SubjectToStatus Constraints_getStatus(	Constraints* _THIS,
														int i		/**< Number of constraints' bound. */
														);


/** Sets type of constraints' bound.
 *	\return SUCCESSFUL_RETURN \n
 			RET_INDEX_OUT_OF_BOUNDS */
static inline returnValue Constraints_setType(	Constraints* _THIS,
												int i,				/**< Number of constraints' bound. */
												SubjectToType value	/**< Type of constraints' bound. */
												);

/** Sets status of constraints' bound.
 *	\return SUCCESSFUL_RETURN \n
 			RET_INDEX_OUT_OF_BOUNDS */
static inline returnValue Constraints_setStatus(	Constraints* _THIS,
													int i,					/**< Number of constraints' bound. */
													SubjectToStatus value	/**< Status of constraints' bound. */
													);


/** Sets status of lower constraints' bounds. */
static inline void Constraints_setNoLower(	Constraints* _THIS,
											BooleanType _status		/**< Status of lower constraints' bounds. */
											);

/** Sets status of upper constraints' bounds. */
static inline void Constraints_setNoUpper(	Constraints* _THIS,
											BooleanType _status		/**< Status of upper constraints' bounds. */
											);


/** Returns status of lower constraints' bounds.
 *	\return BT_TRUE if there is no lower constraints' bound on any variable. */
static inline BooleanType Constraints_hasNoLower(	Constraints* _THIS
													);

/** Returns status of upper bounds.
 *	\return BT_TRUE if there is no upper constraints' bound on any variable. */
static inline BooleanType Constraints_hasNoUpper(	Constraints* _THIS
													);


/** Shifts forward type and status of all constraints by a given
 *  offset. This offset has to lie within the range [0,n/2] and has to
 *  be an integer divisor of the total number of constraints n.
 *  Type and status of the first \<offset\> constraints  is thrown away,
 *  type and status of the last \<offset\> constraints is doubled,
 *  e.g. for offset = 2: \n
 *  shift( {c/b1,c/b2,c/b3,c/b4,c/b5,c/b6} ) = {c/b3,c/b4,c/b5,c/b6,c/b5,c/b6}
 *	\return SUCCESSFUL_RETURN \n
 			RET_INDEX_OUT_OF_BOUNDS \n
 			RET_INVALID_ARGUMENTS \n
 			RET_SHIFTING_FAILED */
returnValue Constraints_shift(	Constraints* _THIS,
								int offset		/**< Shift offset within the range [0,n/2] and integer divisor of n. */
								);

/** Rotates forward type and status of all constraints by a given
 *  offset. This offset has to lie within the range [0,n].
 *  Example for offset = 2: \n
 *  rotate( {c1,c2,c3,c4,c5,c6} ) = {c3,c4,c5,c6,c1,c2}
 *	\return SUCCESSFUL_RETURN \n
 			RET_INDEX_OUT_OF_BOUNDS \n
 			RET_ROTATING_FAILED */
returnValue Constraints_rotate(	Constraints* _THIS,
								int offset		/**< Rotation offset within the range [0,n]. */
								);


/** Prints information on constraints object
 *  (in particular, lists of inactive and active constraints.
 * \return SUCCESSFUL_RETURN \n
		   RET_INDEXLIST_CORRUPTED */
returnValue Constraints_print(	Constraints* _THIS
								);


/** Initially adds all numbers of new (i.e. not yet in the list) bounds to
 *  to the index set corresponding to the desired status;
 *  the order depends on the SujectToType of each index.
 *	\return SUCCESSFUL_RETURN \n
 			RET_SETUP_CONSTRAINT_FAILED */
returnValue Constraints_setupAll(	Constraints* _THIS,
									SubjectToStatus _status	/**< Desired initial status for all bounds. */
									);


/** Adds the index of a new constraint to index set.
 *	\return SUCCESSFUL_RETURN \n
 			RET_ADDINDEX_FAILED \n
			RET_INVALID_ARGUMENTS */
returnValue Constraints_addIndex(	Constraints* _THIS,
									Indexlist* const indexlist,	/**< Index list to which the new index shall be added. */
									int newnumber,				/**< Number of new constraint. */
									SubjectToStatus newstatus	/**< Status of new constraint. */
									);

/** Removes the index of a constraint from index set.
 *	\return SUCCESSFUL_RETURN \n
 			RET_REMOVEINDEX_FAILED \n
			RET_INVALID_ARGUMENTS */
returnValue Constraints_removeIndex(	Constraints* _THIS,
										Indexlist* const indexlist,	/**< Index list from which the new index shall be removed. */
										int removenumber			/**< Number of constraint to be removed. */
										);

/** Swaps the indices of two constraints or bounds within the index set.
 *	\return SUCCESSFUL_RETURN \n
 			RET_SWAPINDEX_FAILED \n
			RET_INVALID_ARGUMENTS */
returnValue Constraints_swapIndex(	Constraints* _THIS,
									Indexlist* const indexlist,	/**< Index list in which the indices shold be swapped. */
									int number1,				/**< Number of first constraint. */
									int number2					/**< Number of second constraint. */
									);



/*
 *	g e t N u m b e r O f T y p e
 */
static inline int Constraints_getNumberOfType( Constraints* _THIS, SubjectToType _type )
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
static inline SubjectToType Constraints_getType( Constraints* _THIS, int i )
{
	if ( ( i >= 0 ) && ( i < _THIS->n ) )
		return _THIS->type[i];

	return ST_UNKNOWN;
}


/*
 *	g e t S t a t u s
 */
static inline SubjectToStatus Constraints_getStatus( Constraints* _THIS, int i )
{
	if ( ( i >= 0 ) && ( i < _THIS->n ) )
		return _THIS->status[i];

	return ST_UNDEFINED;
}


/*
 *	s e t T y p e
 */
static inline returnValue Constraints_setType( Constraints* _THIS, int i, SubjectToType value )
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
static inline returnValue Constraints_setStatus( Constraints* _THIS, int i, SubjectToStatus value )
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
static inline void Constraints_setNoLower( Constraints* _THIS, BooleanType _status )
{
	_THIS->noLower = _status;
}


/*
 *	s e t N o U p p e r
 */
static inline void Constraints_setNoUpper( Constraints* _THIS, BooleanType _status )
{
	_THIS->noUpper = _status;
}


/*
 *	h a s N o L o w e r
 */
static inline BooleanType Constraints_hasNoLower( Constraints* _THIS )
{
	return _THIS->noLower;
}


/*
 *	h a s N o U p p p e r
 */
static inline BooleanType Constraints_hasNoUpper( Constraints* _THIS )
{
	return _THIS->noUpper;
}



/*
 *	g e t N C
 */
static inline int Constraints_getNC( Constraints* _THIS )
{
 	return _THIS->n;
}


/*
 *	g e t N E C
 */
static inline int Constraints_getNEC( Constraints* _THIS )
{
	return Constraints_getNumberOfType( _THIS,ST_EQUALITY );
}


/*
 *	g e t N I C
 */
static inline int Constraints_getNIC( Constraints* _THIS )
{
 	return Constraints_getNumberOfType( _THIS,ST_BOUNDED );
}


/*
 *	g e t N U C
 */
static inline int Constraints_getNUC( Constraints* _THIS )
{
 	return Constraints_getNumberOfType( _THIS,ST_UNBOUNDED );
}


/*
 *	g e t N A C
 */
static inline int Constraints_getNAC( Constraints* _THIS )
{
 	return Indexlist_getLength( _THIS->active );
}


/*
 *	g e t N I A C
 */
static inline int Constraints_getNIAC( Constraints* _THIS )
{
	return Indexlist_getLength( _THIS->inactive );
}



/*
 *	g e t A c t i v e
 */
static inline Indexlist* Constraints_getActive( Constraints* _THIS )
{
	return _THIS->active;
}


/*
 *	g e t I n a c t i v e
 */
static inline Indexlist* Constraints_getInactive( Constraints* _THIS )
{
	return _THIS->inactive;
}


END_NAMESPACE_QPOASES


#endif	/* QPOASES_CONSTRAINTS_H */


/*
 *	end of file
 */
