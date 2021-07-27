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
*    \file include/acado/noise/gaussian_noise.hpp
*    \author Hans Joachim Ferreau, Boris Houska
*/


#ifndef ACADO_TOOLKIT_GAUSSIAN_NOISE_HPP
#define ACADO_TOOLKIT_GAUSSIAN_NOISE_HPP


#include <acado/noise/noise.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Generates pseudo-random Gaussian noise for simulating the Process.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class GaussiaNoise generates pseudo-random Gaussian noise
 *	for simulating the Process within the SimulationEnvironment.
 *
 *	 \author Hans Joachim Ferreau, Boris Houska
 */
class GaussianNoise : public Noise
{
	//
	//  PUBLIC MEMBER FUNCTIONS:
	//
	
	public:

		/** Default constructor. 
		 */
		GaussianNoise( );

		/** Constructor which takes mean value and variance of the random variable.
		 *	The dimension of these limit vector determine the dimension of the 
		 *	random variable.
		 *
		 *	@param[in] _mean			Mean value for each component.
		 *	@param[in] _variance		Variance for each component.
		 */
        GaussianNoise(	const DVector& _mean,
						const DVector& _variance
						);

		/** Constructor which takes the dimension of the random variable as well as
		 *	as common values for the mean value and variance of the random variable.
		 *
		 *	@param[in] _dim				Dimension of random variable.
		 *	@param[in] _mean			Mean value for each component.
		 *	@param[in] _variance		Variance for each component.
		 */
        GaussianNoise(	uint _dim,
						double _mean,
						double _variance
						);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		GaussianNoise(	const GaussianNoise& rhs
						);
	
		/** Destructor.
		 */
		virtual ~GaussianNoise( );

		/** Assignment Operator (deep copy)
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		GaussianNoise& operator=(	const GaussianNoise& rhs
									);

		/** Clone constructor (deep copy).
		 *
		 *	\return Pointer to deep copy of base class type
		 */
        virtual GaussianNoise* clone( ) const;

		/** Clone constructor for a given noise component (deep copy).
		 *
		 *	@param[in] idx		Right-hand side object.
		 *
		 *	\return Pointer to deep copy of base class type
		 */
        virtual GaussianNoise* clone(	uint idx
										) const;


		/** Assigns new mean values to the random variable.
		 *
		 *	@param[in] _mean		New mean value for each component.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_VECTOR_DIMENSION_MISMATCH
		 */
		inline returnValue setMeans(	const DVector& _mean
										);

		/** Assigns new mean values to the random variable.
		 *
		 *	@param[in] _mean		New common mean value for all components.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		inline returnValue setMeans(	double _mean
										);

		/** Assigns new mean value on the component of the random variable
		 *	with given index.
		 *
		 *	@param[in] idx			Index of component.
		 *	@param[in] _mean		New mean value.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		inline returnValue setMean(		uint idx,
										double _mean
										);


		/** Assigns new variances to the random variable.
		 *
		 *	@param[in] _variance	New variances for each component.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_VECTOR_DIMENSION_MISMATCH
		 */
		inline returnValue setVariances(	const DVector& _variance
											);

		/** Assigns new variances to the random variable.
		 *
		 *	@param[in] _variance	New common variance for all components.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		inline returnValue setVariances(	double _variance
											);

		/** Assigns new variance on the component of the random variable
		 *	with given index.
		 *
		 *	@param[in] idx			Index of component.
		 *	@param[in] _variance	New variance.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		inline returnValue setVariance(	uint idx,
										double _variance
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


		/** Returns mean values of the random variable.
		 *
		 *	\return Mean values of the random variable
		 */
		inline const DVector& getMean( ) const;

		/** Returns variances of the random variable.
		 *
		 *	\return Variances of the random variable
		 */
		inline const DVector& getVariance( ) const;


	
	//
	//  PROTECTED MEMBER FUNCTIONS:
	//
	protected:

		/** Returns a pseudo-random number based on a Gaussian distribution with
		 *	given mean and variance.
		 *
		 *	@param[in] _mean		Mean value of Gaussian distribution.
		 *	@param[in] _variance	Variance of Gaussian distribution.
		 *
		 *  \return Gaussian distributed pseudo-random number
		 */
		double getGaussianRandomNumber(	double _mean,
										double _variance
										) const;


	//
	//  PROTECTED MEMBERS:
	//
	protected:

		DVector mean;					/**< Mean value for each component. */
		DVector variance;				/**< Variance for each component. */
};


CLOSE_NAMESPACE_ACADO



#include <acado/noise/gaussian_noise.ipp>


#endif	// ACADO_TOOLKIT_GAUSSIAN_NOISE_HPP


/*
 *	end of file
 */
