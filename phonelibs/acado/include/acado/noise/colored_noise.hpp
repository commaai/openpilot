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
*    \file include/acado/noise/colored_noise.hpp
*    \author Hans Joachim Ferreau, Boris Houska
*/


#ifndef ACADO_TOOLKIT_COLORED_NOISE_HPP
#define ACADO_TOOLKIT_COLORED_NOISE_HPP


#include <acado/noise/noise.hpp>
#include <acado/dynamic_system/dynamic_system.hpp>



BEGIN_NAMESPACE_ACADO


/**
 *	\brief Generates pseudo-random colored noise for simulating the Process.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class ColoredNoise generates pseudo-random colored noise
 *	for simulating the Process within the SimulationEnvironment.
 *
 *	\note NOT YET OPERATIONAL!
 *
 *	 \author Hans Joachim Ferreau, Boris Houska
 */
class ColoredNoise : public Noise
{
	//
	//  PUBLIC MEMBER FUNCTIONS:
	//
	
	public:

		/** Default constructor. 
		 */
		ColoredNoise( );

		/** Constructor that takes dimension of noise vector. */
        ColoredNoise(	const DynamicSystem& _dynamicSystem
						);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		ColoredNoise(	const ColoredNoise& rhs
						);
	
		/** Destructor. 
		 */
		virtual ~ColoredNoise( );

		/** Assignment Operator (deep copy)
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		ColoredNoise& operator=(	const ColoredNoise& rhs
									);

		/** Clone constructor (deep copy).
		 *
		 *	\return Pointer to deep copy of base class type
		 */
        virtual ColoredNoise* clone( ) const;

		/** Clone constructor for a given noise component (deep copy).
		 *
		 *	@param[in] idx		Right-hand side object.
		 *
		 *	\return Pointer to deep copy of base class type
		 */
        virtual ColoredNoise* clone(	uint idx
										) const;


		/** Assigns a new dynamic system for generating the colored noise.
		 *
		 *	\return SUCCESFUL_RETURN
		 */
		returnValue setDynamicSystem(	const DynamicSystem& _dynamicSystem
										);


		/** Initializes noise generation and performs a couple of consistency checks.
		 *	Initialization of the pseudo-random number generator can be based on
		 *	a seed in order to allow exact reproduction of generated noise. If seed
		 *	is not specified (i.e. 0), a seed is obtain from the system clock.
		 *
		 *	@param[in] seed		Seed for pseudo-random number generator.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_NOISE_SETTINGS, \n
		 *	        RET_NO_NOISE_SETTINGS
		 */
		virtual returnValue init(	uint seed = 0
									);
	
		/** Generates a single noise vector based on current internal settings.
		 *
		 *	@param[out] _w		Generated noise vector.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_VECTOR_DIMENSION_MISMATCH
		 */
		virtual returnValue step(	DVector& _w
									);

		/** Generates a noise vector sequence based on current internal settings.
		 *	Noise is generated for each grid point of the VariablesGrid passed.
		 *
		 *	@param[in,out] _w		Generated noise vector sequence.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_VECTOR_DIMENSION_MISMATCH
		 */
		virtual returnValue step(	VariablesGrid& _w
									);



	//
	//  PROTECTED MEMBER FUNCTIONS:
	//
	
	protected:

	//
	//  PROTECTED MEMBERS:
	//
	
	protected:

		DynamicSystem* dynamicSystem;				/**< Dynamic system generating the colored noise. */
};


CLOSE_NAMESPACE_ACADO



#include <acado/noise/colored_noise.ipp>


#endif	// ACADO_TOOLKIT_COLORED_NOISE_HPP


/*
 *	end of file
 */
