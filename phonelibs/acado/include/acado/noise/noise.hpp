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
*    \file include/acado/noise/noise.hpp
*    \author Hans Joachim Ferreau, Boris Houska
*/


#ifndef ACADO_TOOLKIT_NOISE_HPP
#define ACADO_TOOLKIT_NOISE_HPP


#include <acado/utils/acado_utils.hpp>

#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/variables_grid/variables_grid.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Base class for generating pseudo-random noise for simulating the Process.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class Noise serves as base class for generating pseudo-random noise
 *	for simulating the Process within the SimulationEnvironment.
 *
 *	 \author Hans Joachim Ferreau, Boris Houska
 */
class Noise
{
	//
	//  PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor. 
		 */
		Noise( );

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		Noise(	const Noise& rhs
				);

		/** Destructor. 
		 */
		virtual ~Noise( );

		/** Assignment Operator (deep copy)
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		Noise& operator=(	const Noise& rhs
							);

		/** Clone constructor (deep copy).
		 *
		 *	\return Pointer to deep copy of base class type
		 */
        virtual Noise* clone( ) const = 0;

		/** Clone constructor for a given noise component (deep copy).
		 *
		 *	@param[in] idx		Right-hand side object.
		 *
		 *	\return Pointer to deep copy of base class type
		 */
        virtual Noise* clone(	uint idx
								) const = 0;


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
									) = 0;

		/** Generates a single noise vector based on current internal settings.
		 *
		 *	@param[out] _w		Generated noise vector.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_VECTOR_DIMENSION_MISMATCH
		 */
		virtual returnValue step(	DVector& _w
									) = 0;

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
									) = 0;


		/** Returns dimension of noise vector.
		 *
		 *  \return Dimension of noise vector
		 */
		inline uint getDim( ) const;

		/** Returns whether noise vector is empty (i.e. has dimension zero).
		 *
		 *  \return BT_TRUE  iff noise vector is empty, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType isEmpty( ) const;


		/** Returns current status of noise block, see documentation of BlockStatus for details.
		 *
		 *  \return Current status of noise block
		 */
		inline BlockStatus getStatus( ) const;



	//
	//  PROTECTED MEMBER FUNCTIONS:
	//
	protected:

		/** Assigns new status to noise block
		 *
		 *	@param[in] _status		New status.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue setStatus(	BlockStatus _status
										);

		/** Returns a pseudo-random number based on a uniform distribution with
		 *	given lower and upper limits.
		 *
		 *	@param[in] _lowerLimit		Lower limit of random variable.
		 *	@param[in] _upperLimit		Lower limit of random variable.
		 *
		 *  \return Uniformly distributed pseudo-random number
		 */
		inline double getUniformRandomNumber(	double _lowerLimit,
												double _upperLimit
												) const;


	//
	//  PROTECTED MEMBERS:
	//
	protected:
		BlockStatus status;				/**< Current status of the noise. */

		VariablesGrid w;				/**< Sequence of most recently generated noise. */
};


CLOSE_NAMESPACE_ACADO


#include <acado/noise/noise.ipp>


#include <acado/noise/uniform_noise.hpp>
#include <acado/noise/gaussian_noise.hpp>
#include <acado/noise/colored_noise.hpp>


#endif	// ACADO_TOOLKIT_NOISE_HPP


/*
 *	end of file
 */
