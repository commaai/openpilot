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
 *	\file INCLUDE/SubjectTo.hpp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Declaration of the SubjectTo class designed to manage working sets of
 *	constraints and bounds within a QProblem.
 */


#ifndef QPOASES_SUBJECTTO_HPP
#define QPOASES_SUBJECTTO_HPP


#include <Indexlist.hpp>



/** This class manages working sets of constraints and bounds by storing
 *	index sets and other status information.
 *
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 */
class SubjectTo
{
	/*
	 *	PUBLIC MEMBER FUNCTIONS
	 */
	public:
		/** Default constructor. */
		SubjectTo( );

		/** Copy constructor (deep copy). */
		SubjectTo(	const SubjectTo& rhs	/**< Rhs object. */
					);

		/** Destructor. */
		~SubjectTo( );

		/** Assignment operator (deep copy). */
		SubjectTo& operator=(	const SubjectTo& rhs	/**< Rhs object. */
								);


		/** Pseudo-constructor takes the number of constraints or bounds.
		 *	\return SUCCESSFUL_RETURN */
		returnValue init(	int n 	/**< Number of constraints or bounds. */
							);


		/** Returns type of (constraints') bound.
		 *	\return Type of (constraints') bound \n
		 			RET_INDEX_OUT_OF_BOUNDS */
		inline SubjectToType getType(	int i		/**< Number of (constraints') bound. */
										) const ;

		/** Returns status of (constraints') bound.
		 *	\return Status of (constraints') bound \n
		 			ST_UNDEFINED */
		inline SubjectToStatus getStatus(	int i		/**< Number of (constraints') bound. */
											) const;


		/** Sets type of (constraints') bound.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_INDEX_OUT_OF_BOUNDS */
		inline returnValue setType(	int i,				/**< Number of (constraints') bound. */
									SubjectToType value	/**< Type of (constraints') bound. */
									);

		/** Sets status of (constraints') bound.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_INDEX_OUT_OF_BOUNDS */
		inline returnValue setStatus(	int i,					/**< Number of (constraints') bound. */
										SubjectToStatus value	/**< Status of (constraints') bound. */
										);


		/** Sets status of lower (constraints') bounds. */
		inline void setNoLower(	BooleanType _status		/**< Status of lower (constraints') bounds. */
								);

		/** Sets status of upper (constraints') bounds. */
		inline void setNoUpper(	BooleanType _status		/**< Status of upper (constraints') bounds. */
								);


		/** Returns status of lower (constraints') bounds.
		 *	\return BT_TRUE if there is no lower (constraints') bound on any variable. */
		inline BooleanType isNoLower( ) const;

		/** Returns status of upper bounds.
		 *	\return BT_TRUE if there is no upper (constraints') bound on any variable. */
		inline BooleanType isNoUpper( ) const;


	/*
	 *	PROTECTED MEMBER FUNCTIONS
	 */
	protected:
		/** Adds the index of a new constraint or bound to index set.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_ADDINDEX_FAILED */
		returnValue addIndex(	Indexlist* const indexlist,	/**< Index list to which the new index shall be added. */
								int newnumber,				/**< Number of new constraint or bound. */
								SubjectToStatus newstatus	/**< Status of new constraint or bound. */
								);

		/** Removes the index of a constraint or bound from index set.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_UNKNOWN_BUG */
		returnValue removeIndex(	Indexlist* const indexlist,	/**< Index list from which the new index shall be removed. */
									int removenumber			/**< Number of constraint or bound to be removed. */
									);

		/** Swaps the indices of two constraints or bounds within the index set.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_SWAPINDEX_FAILED */
		returnValue swapIndex(	Indexlist* const indexlist,	/**< Index list in which the indices shold be swapped. */
								int number1,				/**< Number of first constraint or bound. */
								int number2					/**< Number of second constraint or bound. */
								);


	/*
	 *	PROTECTED MEMBER VARIABLES
	 */
	protected:
		SubjectToType type[NVMAX+NCMAX]; 		/**< Type of constraints/bounds. */
		SubjectToStatus status[NVMAX+NCMAX];	/**< Status of constraints/bounds. */

		BooleanType noLower;				 	/**< This flag indicates if there is no lower bound on any variable. */
		BooleanType noUpper;	 				/**< This flag indicates if there is no upper bound on any variable. */


	/*
	 *	PRIVATE MEMBER VARIABLES
	 */
	private:
		int size;
};



#include <SubjectTo.ipp>

#endif	/* QPOASES_SUBJECTTO_HPP */


/*
 *	end of file
 */
