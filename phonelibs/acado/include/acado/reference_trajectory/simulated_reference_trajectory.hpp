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
 *	\file include/acado/reference_trajectory/simulated_reference_trajectory.hpp
 *	\author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_SIMULATED_REFERENCE_TRAJECTORY_HPP
#define ACADO_TOOLKIT_SIMULATED_REFERENCE_TRAJECTORY_HPP


#include <acado/variables_grid/variables_grid.hpp>
#include <acado/process/process.hpp>
#include <acado/reference_trajectory/adaptive_reference_trajectory.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Allows to define a simulated reference trajectory that the ControlLaw aims to track.
 *
 *	\ingroup UserDataStructures
 *
 *	The class SimulatedReferenceTrajectory allows to define an reference trajectory that is
 *	determined online based on the simulation of a dynamic system. The ControlLaw can aim to 
 *	track this reference trajectory while computing its control action.
 *
 *	 \author Hans Joachim Ferreau, Boris Houska
 */
class SimulatedReferenceTrajectory : public AdaptiveReferenceTrajectory
{
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor. 
		 */
		SimulatedReferenceTrajectory( );

		/** Default constructor. 
		 */
		SimulatedReferenceTrajectory(	const DynamicSystem& _dynamicSystem,
										IntegratorType _integratorType = INT_UNKNOWN
										);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		SimulatedReferenceTrajectory(	const SimulatedReferenceTrajectory& rhs
										);

		/** Destructor. 
		 */
		virtual ~SimulatedReferenceTrajectory( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		SimulatedReferenceTrajectory& operator=(	const SimulatedReferenceTrajectory& rhs
												);

		/** Clone constructor (deep copy).
		 *
		 *	\return Pointer to deep copy of base class type
		 */
		virtual ReferenceTrajectory* clone( ) const;


		/** Initializes the reference trajectory evaluation based on the given inputs.
		 *
		 *	@param[in]  _startTime	Start time.
		 *	@param[in]  _x			Initial value for differential states.
		 *	@param[in]  _xa			Initial value for algebraic states.
		 *	@param[in]  _u			Initial value for controls.
		 *	@param[in]  _p			Initial value for parameters.
		 *	@param[in]  _w			Initial value for disturbances.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue init(	double startTime = 0.0,
									const DVector& _x  = emptyConstVector,
									const DVector& _xa = emptyConstVector,
									const DVector& _u  = emptyConstVector,
									const DVector& _p  = emptyConstVector,
									const DVector& _w  = emptyConstVector
									);


		/** Updates the reference trajectory evaluation based on the given inputs.
		 *
		 *	@param[in]  _currentTime	Start time.
		 *	@param[in]  _y				Current process output.
		 *	@param[in]  _x				Estimated current value for differential states.
		 *	@param[in]  _xa				Estimated current value for algebraic states.
		 *	@param[in]  _u				Estimated current value for controls.
		 *	@param[in]  _p				Estimated current value for parameters.
		 *	@param[in]  _w				Estimated current value for disturbances.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue step(	double _currentTime,
									const DVector& _y,
									const DVector& _x  = emptyConstVector,
									const DVector& _xa = emptyConstVector,
									const DVector& _u  = emptyConstVector,
									const DVector& _p  = emptyConstVector,
									const DVector& _w  = emptyConstVector
									);

		/** Updates the reference trajectory evaluation based on the given inputs.
		 *
		 *	@param[in]  _x				Estimated current value for differential states.
		 *	@param[in]  _u				Estimated current time-varying value for controls.
		 *	@param[in]  _p				Estimated current time-varying value for parameters.
		 *	@param[in]  _w				Estimated current time-varying value for disturbances.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue step(	const DVector& _x,
									const VariablesGrid& _u = emptyConstVariablesGrid,
									const VariablesGrid& _p = emptyConstVariablesGrid,
									const VariablesGrid& _w = emptyConstVariablesGrid
									);


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


		/** Returns dimension of reference trajectory.
		 *
		 *  \return Dimension of reference trajectory
		 */
		virtual uint getDim( ) const;



	//
	// PROTECTED MEMBER FUNCTIONS:
	//
	protected:



	//
	// DATA MEMBERS:
	//
	protected:
		Process* process;
};


CLOSE_NAMESPACE_ACADO



#endif  // ACADO_TOOLKIT_SIMULATED_REFERENCE_TRAJECTORY_HPP

/*
 *	end of file
 */
