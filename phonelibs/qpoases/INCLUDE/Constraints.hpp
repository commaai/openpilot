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
 *	\file INCLUDE/Constraints.hpp
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 *
 *	Declaration of the Constraints class designed to manage working sets of
 *	constraints within a QProblem.
 */


#ifndef QPOASES_CONSTRAINTS_HPP
#define QPOASES_CONSTRAINTS_HPP


#include <SubjectTo.hpp>



/** This class manages working sets of constraints by storing
 *	index sets and other status information.
 *
 *	\author Hans Joachim Ferreau
 *	\version 1.3embedded
 *	\date 2007-2008
 */
class Constraints : public SubjectTo
{
	/*
	 *	PUBLIC MEMBER FUNCTIONS
	 */
	public:
		/** Default constructor. */
		Constraints( );

		/** Copy constructor (deep copy). */
		Constraints(	const Constraints& rhs	/**< Rhs object. */
						);

		/** Destructor. */
		~Constraints( );

		/** Assignment operator (deep copy). */
		Constraints& operator=(	const Constraints& rhs	/**< Rhs object. */
								);


		/** Pseudo-constructor takes the number of constraints.
		 *	\return SUCCESSFUL_RETURN */
		returnValue init(	int n	/**< Number of constraints. */
							);


		/** Initially adds number of a new (i.e. not yet in the list) constraint to
		 *  a given index set.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_SETUP_CONSTRAINT_FAILED \n
					RET_INDEX_OUT_OF_BOUNDS \n
					RET_INVALID_ARGUMENTS */
		returnValue setupConstraint(	int _number,				/**< Number of new constraint. */
										SubjectToStatus _status		/**< Status of new constraint. */
										);

		/** Initially adds all enabled numbers of new (i.e. not yet in the list) constraints to
		 *  to the index set of inactive constraints; the order depends on the SujectToType
		 *  of each index. Only disabled constraints are added to index set of disabled constraints!
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_SETUP_CONSTRAINT_FAILED */
		returnValue setupAllInactive( );


		/** Moves index of a constraint from index list of active to that of inactive constraints.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_MOVING_CONSTRAINT_FAILED */
		returnValue moveActiveToInactive(	int _number				/**< Number of constraint to become inactive. */
											);

		/** Moves index of a constraint from index list of inactive to that of active constraints.
		 *	\return SUCCESSFUL_RETURN \n
		 			RET_MOVING_CONSTRAINT_FAILED */
		returnValue moveInactiveToActive(	int _number,			/**< Number of constraint to become active. */
											SubjectToStatus _status	/**< Status of constraint to become active. */
											);


		/** Returns the number of constraints.
		 *	\return Number of constraints. */
		inline int getNC( ) const;

		/** Returns the number of implicit equality constraints.
		 *	\return Number of implicit equality constraints. */
		inline int getNEC( ) const;

		/** Returns the number of "real" inequality constraints.
		 *	\return Number of "real" inequality constraints. */
		inline int getNIC( ) const;

		/** Returns the number of unbounded constraints (i.e. without any bounds).
		 *	\return Number of unbounded constraints (i.e. without any bounds). */
		inline int getNUC( ) const;


		/** Sets number of implicit equality constraints.
		 *	\return SUCCESSFUL_RETURN */
		inline returnValue setNEC(	int n	/**< Number of implicit equality constraints. */
							);

		/** Sets number of "real" inequality constraints.
		 *	\return SUCCESSFUL_RETURN */
		inline returnValue setNIC(	int n	/**< Number of "real" inequality constraints. */
									);

		/** Sets number of unbounded constraints (i.e. without any bounds).
		 *	\return SUCCESSFUL_RETURN */
		inline returnValue setNUC(	int n	/**< Number of unbounded constraints (i.e. without any bounds). */
									);


		/** Returns the number of active constraints.
		 *	\return Number of constraints. */
		inline int getNAC( );

		/** Returns the number of inactive constraints.
		 *	\return Number of constraints. */
		inline int getNIAC( );


		/** Returns a pointer to active constraints index list.
		 *	\return Pointer to active constraints index list. */
		inline Indexlist* getActive( );

		/** Returns a pointer to inactive constraints index list.
		 *	\return Pointer to inactive constraints index list. */
		inline Indexlist* getInactive( );


	/*
	 *	PROTECTED MEMBER VARIABLES
	 */
	protected:
		int nC;					/**< Number of constraints (nC = nEC + nIC + nUC). */
		int nEC;				/**< Number of implicit equality constraints. */
		int	nIC;				/**< Number of "real" inequality constraints. */
		int nUC;				/**< Number of unbounded constraints (i.e. without any bounds). */

		Indexlist active;		/**< Index list of active constraints. */
		Indexlist inactive;		/**< Index list of inactive constraints. */
};


#include <Constraints.ipp>

#endif	/* QPOASES_CONSTRAINTS_HPP */


/*
 *	end of file
 */
