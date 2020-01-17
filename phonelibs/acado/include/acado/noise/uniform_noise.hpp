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
*    \file include/acado/noise/uniform_noise.hpp
*    \author Boris Houska, Hans Joachim Ferreau
*/


#ifndef ACADO_TOOLKIT_UNIFORM_NOISE_HPP
#define ACADO_TOOLKIT_UNIFORM_NOISE_HPP


#include <acado/noise/noise.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Generates pseudo-random uniformly distributed noise for simulating the Process.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class UniformNoise generates pseudo-random uniformly distributed noise
 *	for simulating the Process within the SimulationEnvironment.
 *
 *	 \author Hans Joachim Ferreau, Boris Houska
 */
class UniformNoise : public Noise
{
	//
	//  PUBLIC MEMBER FUNCTIONS:
	//
	
	public:

		/** Default constructor. 
		 */
		UniformNoise( );

		/** Constructor which takes lower and upper limits of the random variable.
		 *	The dimension of these limit vector determine the dimension of the 
		 *	random variable.
		 *
		 *	@param[in] _lowerLimit		Lower limit for each component.
		 *	@param[in] _upperLimit		Upper limit for each component.
		 */
        UniformNoise(	const DVector& _lowerLimit,
						const DVector& _upperLimit
						);

		/** Constructor which takes the dimension of the random variable as well as
		 *	as common values for the lower and upper limits of all components.
		 *
		 *	@param[in] _dim				Dimension of random variable.
		 *	@param[in] _lowerLimit		Common lower limit for all components.
		 *	@param[in] _upperLimit		Common upper limit for all components.
		 */
        UniformNoise(	uint _dim,
						double _lowerLimit,
						double _upperLimit
						);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		UniformNoise(	const UniformNoise& rhs
						);
	
		/** Destructor. 
		 */
		virtual ~UniformNoise( );

		/** Assignment Operator (deep copy)
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		UniformNoise& operator=(	const UniformNoise& rhs
									);

		/** Clone constructor (deep copy).
		 *
		 *	\return Pointer to deep copy of base class type
		 */
        virtual UniformNoise* clone( ) const;

		/** Clone constructor for a given noise component (deep copy).
		 *
		 *	@param[in] idx		Right-hand side object.
		 *
		 *	\return Pointer to deep copy of base class type
		 */
        virtual UniformNoise* clone(	uint idx
										) const;


		/** Assigns new lower and upper limits on the random variable.
		 *
		 *	@param[in] _lowerLimit		New lower limits for each component.
		 *	@param[in] _upperLimit		New upper limits for each component.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_VECTOR_DIMENSION_MISMATCH, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue setLimits(	const DVector& _lowerLimit,
								const DVector& _upperLimit
								);

		/** Assigns new lower and upper limits on the random variable.
		 *
		 *	@param[in] _lowerLimit		New common lower limit for all components.
		 *	@param[in] _upperLimit		New common upper limit for all components.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue setLimits(	double _lowerLimit,
								double _upperLimit
								);

		/** Assigns new lower and upper limit on the component of the random variable
		 *	with given index.
		 *
		 *	@param[in] idx				Index of component.
		 *	@param[in] _lowerLimit		New lower limit.
		 *	@param[in] _upperLimit		New upper limit.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue setLimit(	uint idx,
								double _lowerLimit,
								double _upperLimit
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


		/** Returns lower limits of the random variable.
		 *
		 *	\return Lower limits of the random variable
		 */
		inline const DVector& getLowerLimit( ) const;

		/** Returns upper limits of the random variable.
		 *
		 *	\return Upper limits of the random variable
		 */
		inline const DVector& getUpperLimit( ) const;



	//
	//  PROTECTED MEMBER FUNCTIONS:
	//
	protected:


	//
	//  PROTECTED MEMBERS:
	//
	protected:

		DVector lowerLimit;				/**< Lower limit for each component. */
		DVector upperLimit;				/**< Upper limit for each component. */
};


CLOSE_NAMESPACE_ACADO



#include <acado/noise/uniform_noise.ipp>


#endif	// ACADO_TOOLKIT_UNIFORM_NOISE_HPP


/*
 *	end of file
 */
