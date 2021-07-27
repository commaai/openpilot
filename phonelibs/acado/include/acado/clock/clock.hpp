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
*    \file include/acado/clock/clock.hpp
*    \author Hans Joachim Ferreau, Boris Houska
*/


#ifndef ACADO_TOOLKIT_CLOCK_HPP
#define ACADO_TOOLKIT_CLOCK_HPP


#include <acado/utils/acado_utils.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Base class for all kind of time measurements.
 *
 *	\ingroup BasicDataStructures
 *
 *	The class Clock serves as base class for all kind of time measurements, 
 *	both real and simulated ones.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class Clock
{
	//
	//  PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor. 
		 */
		Clock( );

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		Clock(	const Clock &rhs
				);

		/** Destructor. 
		 */
		virtual ~Clock();

		/** Assignment Operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		Clock& operator=(	const Clock &rhs
							);

		/** Clone constructor (deep copy).
		 *
		 *	\return Pointer to deep copy of base class type
		 */
		virtual Clock* clone( ) const = 0;


		/** Initializes the clock with given initial time.
		 *
		 *	@param[in]  _initialTime	Initial time.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue init(	double _initialTime
									);

		/** Initializes the clock with initial time zero.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue reset( );


		/** Starts time measurement.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_CLOCK_NOT_READY, \n
		 *	        RET_NO_SYSTEM_TIME
		 */
		virtual returnValue start( ) = 0;
	
		/** Shifts measured time by a given offset.
		 *
		 *	@param[in]  _timeShift		Time offset.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_CLOCK_NOT_READY, \n
		 *	        RET_NO_SYSTEM_TIME
		 */
		virtual returnValue step(	double _timeShift
									) = 0;
	
		/** Stops time measurement.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_CLOCK_NOT_READY, \n
		 *	        RET_NO_SYSTEM_TIME
		 */
		virtual returnValue stop ( ) = 0;


		/** Returns elapsed time.
		 *
		 *	@param[in]  _elapsedTime		Elapsed time.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY
		 */
		virtual returnValue getTime(	double& _elapsedTime
										);

		/** Returns elapsed time.
		 *
		 *  \return   >= 0: elapsed time, \n
		 *	        -INFTY: internal error
		 */
		virtual double getTime( );


		/** Returns current status of clock, see documentation of ClockStatus for details.
		 *
		 *  \return Current status of clock
		 */
		inline ClockStatus getStatus( ) const;



	//
	//  PROTECTED MEMBERS:
	//
	protected:
		double elapsedTime;				/**< Elapsed time since last reset. */
		ClockStatus status;				/**< Status of clock. */
};


CLOSE_NAMESPACE_ACADO


#include <acado/clock/clock.ipp>


// collect all remaining headers of clock directory
#include <acado/clock/real_clock.hpp>
#include <acado/clock/simulation_clock.hpp>


#endif	// ACADO_TOOLKIT_CLOCK_HPP


/*
 *	end of file
 */
