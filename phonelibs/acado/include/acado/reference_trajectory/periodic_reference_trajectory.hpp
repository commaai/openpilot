/*
 *    This file is part of ACADO Toolkit.
 *
 *    ACADO Toolkit -- A Toolkit for Automatic Control and Dynamic Optimization.
 *    Copyright (C) 2008-2014 by Boris Houska, Hans Joachim Ferreau,
 *    Milan Vukov, Rien Quirynen, KU Leuven.
 *    Developed within the Optimization in Engineering Center (OPTEC)
 *    under supervision of Moritz Diehl. All rights reserved.
 *
 *    ACADO Toolkit is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    ACADO Toolkit is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with ACADO Toolkit; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


/**
 *	\file include/acado/reference_trajectory/periodic_reference_trajectory.hpp
 *	\author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_PERIODIC_REFERENCE_TRAJECTORY_HPP
#define ACADO_TOOLKIT_PERIODIC_REFERENCE_TRAJECTORY_HPP


#include <acado/reference_trajectory/static_reference_trajectory.hpp> 
#include <acado/variables_grid/variables_grid.hpp>
#include <acado/curve/curve.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Allows to define a static periodic reference trajectory that the ControlLaw aims to track.
 *
 *	\ingroup UserDataStructures
 *
 *  The class PeriodicReferenceTrajectory allows to define a static periodic reference trajectory 
 *	(given beforehand) that the ControlLaw aims to track while computing its control action.
 *
 *	The user-specified static reference trajectory is repeated as often as necessary to yield an
 *	infinitely long periodic reference trajectory.
 *
 *	 \author Hans Joachim Ferreau, Boris Houska
 */
class PeriodicReferenceTrajectory : public StaticReferenceTrajectory
{
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor.
		 */
		PeriodicReferenceTrajectory( );

//		 PeriodicReferenceTrajectory(	const Curve& _yRef
// 										);

		/** Constructor which takes a pre-defined static reference trajectory.
		 *
		 *	@param[in] _yRef			Pre-defined reference trajectory.
		 */
		PeriodicReferenceTrajectory(	const VariablesGrid& _yRef
										);

		/** Constructor which takes a pre-defined static reference trajectory.
		 *
		 *	@param[in] _yRefFileName	Name of file containing the pre-defined reference trajectory.
		 */
		PeriodicReferenceTrajectory(	const char* const _yRefFileName
										);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		PeriodicReferenceTrajectory(	const PeriodicReferenceTrajectory& rhs
										);

		/** Destructor.
		 */
		virtual ~PeriodicReferenceTrajectory( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		PeriodicReferenceTrajectory& operator=(	const PeriodicReferenceTrajectory& rhs
												);

		/** Clone constructor (deep copy).
		 *
		 *	\return Pointer to deep copy of base class type
		 */
		virtual ReferenceTrajectory* clone( ) const;


		/** Returns a piece of the reference trajectory starting and ending at given times.
		 *
		 *	@param[in]  tStart	Start time of reference piece.
		 *	@param[in]  tEnd	End time of reference piece.
		 *	@param[out] _yRef	Desired piece of the reference trajectory.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		virtual returnValue getReference(	double tStart,
											double tEnd,
											VariablesGrid& _yRef
											) const;


	//
	// PROTECTED MEMBER FUNCTIONS:
	//
	protected:



	//
	// DATA MEMBERS:
	//
	protected:

};


CLOSE_NAMESPACE_ACADO



//#include <acado/reference_trajectory/periodic_reference_trajectory.ipp>


#endif  // ACADO_TOOLKIT_PERIODIC_REFERENCE_TRAJECTORY_HPP

/*
 *	end of file
 */
