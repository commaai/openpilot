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
*    \file include/acado/clock/real_clock.hpp
*    \author Boris Houska, Hans Joachim Ferreau
*/


#ifndef ACADO_TOOLKIT_REAL_CLOCK_HPP
#define ACADO_TOOLKIT_REAL_CLOCK_HPP


#include <acado/clock/clock.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Allows real time measurements based on the system's clock.
 *
 *	\ingroup BasicDataStructures
 *
 *	The class RealClock allows real time measurements based on the
 *	system's clock.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class RealClock : public Clock
{
	//
	//  PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor.
		 */
		RealClock();
	
		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		RealClock(	const RealClock &rhs
					);
	
		/** Destructor.
		 */
		virtual ~RealClock( );
	
		/** Assignment Operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		RealClock& operator=(	const RealClock &rhs
								);

		/** Clone constructor (deep copy).
		 *
		 *	\return Pointer to deep copy of base class type
		 */
		virtual Clock* clone( ) const;


		/** Starts time measurement.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_CLOCK_NOT_READY, \n
		 *	        RET_NO_SYSTEM_TIME
		 */
		virtual returnValue start( );
	
		/** Shifts measured time by a given offset.
		 *
		 *	@param[in]  _timeShift		Time offset.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_CLOCK_NOT_READY, \n
		 *	        RET_NO_SYSTEM_TIME
		 */
		virtual returnValue step(	double _timeShift
									);
	
		/** Stops time measurement.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_CLOCK_NOT_READY, \n
		 *	        RET_NO_SYSTEM_TIME
		 */
		virtual returnValue stop( );



	//
	//  PROTECTED MEMBERS:
	//
	protected:

		double lastTimeInstant;			/**< Last time instant at which start() has been called. */
};


CLOSE_NAMESPACE_ACADO


//#include <acado/clock/real_clock.ipp>


#endif	// ACADO_TOOLKIT_REAL_CLOCK_HPP


/*
 *	end of file
 */
