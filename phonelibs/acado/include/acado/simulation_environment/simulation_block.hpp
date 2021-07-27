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


#ifndef ACADO_TOOLKIT_SIMULATION_BLOCK_HPP
#define ACADO_TOOLKIT_SIMULATION_BLOCK_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/clock/clock.hpp>
#include <acado/user_interaction/user_interaction.hpp>


/**
*    \file include/acado/simulation_environment/simulation_block.hpp
*    \author Boris Houska, Hans Joachim Ferreau
*/


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Base class for building-blocks of the SimulationEnvironment.
 *
 *	\ingroup AuxiliaryFunctionality
 *
 *	The class SimulationBlock serves as base class for all building-blocks of the
 *	SimulationEnvironment. All common functionality, like storing of the sampling time,
 *	should be collected here.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class SimulationBlock : public UserInteraction
{
	//
	//  PUBLIC MEMBER FUNCTIONS:
	//
	public:
		/** Default constructor.
		 */
		SimulationBlock( );

		/** Constructor which takes the name of the block and the sampling time.
		 *
		 *	@param[in] _name			Name of the block, see documentation of BlockName for details.
		 *	@param[in] _samplingTime	Sampling time of the block (has to be positive).
		 */
		SimulationBlock(	BlockName _name,
							double _samplingTime = DEFAULT_SAMPLING_TIME
							);
	
		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		SimulationBlock(	const SimulationBlock& rhs
							);
	
		/** Destructor. 
		 */
		virtual ~SimulationBlock( );
	
		/** Assignment Operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		SimulationBlock& operator=(	const SimulationBlock& rhs
									);


		/** Returns whether the block has been defined (i.e. setup properly).
		 *
		 *  \return BT_TRUE  iff block has been defined, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType isDefined( ) const;


		/** Returns name of the block.
		 *
		 *  \return Name of the block
		 */
		inline BlockName getName( ) const;

		/** Returns sampling time of the block.
		 *
		 *  \return Sampling time of the block
		 */
		inline double getSamplingTime( ) const;


		/** Assigns new name to the block.
		 *
		 *	@param[in]  _name			New name.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		inline returnValue setName(	BlockName _name
									);

		/** Assigns new sampling time to the block.
		 *
		 *	@param[in]  _samplingTime	New sampling time.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		inline returnValue setSamplingTime(	double _samplingTime
											);



	//
	//  PROTECTED MEMBER FUNCTIONS:
	//
	protected:
	
	
	//
	//  PROTECTED MEMBERS:
	//
	protected:
		BlockName name;					/**< Name of the block, see documentation of BlockName for details. */
		double samplingTime;			/**< Sampling time of the block. */

		RealClock realClock;			/**< Clock for real time measurements to be optionally used in derived classes.  */
};


CLOSE_NAMESPACE_ACADO



#include <acado/simulation_environment/simulation_block.ipp>


#endif	// ACADO_TOOLKIT_SIMULATION_BLOCK_HPP


/*
 *	end of file
 */
