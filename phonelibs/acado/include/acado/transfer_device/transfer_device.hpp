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
 *	\file include/acado/transfer_device/transfer_device.hpp
 *	\author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_TRANSFER_DEVICE_HPP
#define ACADO_TOOLKIT_TRANSFER_DEVICE_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/simulation_environment/simulation_block.hpp>

#include <acado/noise/noise.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Base class for simulating Actuator and Sensor behaviour wihtin the Process.
 *
 *	\ingroup UserDataStructures
 *
 *  The class TransferDevive serves as a base class for simulating Actuator
 *	and Sensor behaviour within the Process. It is intended to collect
 *	common features of the Actuator and Sensor.
 *
 *	At the moment, it basically stores an array of additive noise as well as 
 *	the dead times of each component. It also provides a common way to generate
 *	noise.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class TransferDevice : public SimulationBlock
{
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor. 
		 */
		TransferDevice( );

		/** Constructor which takes the dimension, the name and the sampling time 
		 *	of the transfer device. 
		 *
		 *	@param[in] _dim				Dimension of transfer device signal.
		 *	@param[in] _name			Name of the block.
		 *	@param[in] _samplingTime	Sampling time.
		 *
		 *	\note Actuators pass the sum of their control and parameter signal dimensions.
		 */
		TransferDevice( uint _dim,
						BlockName _name = BN_DEFAULT,
						double _samplingTime = DEFAULT_SAMPLING_TIME
						);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		TransferDevice(	const TransferDevice& rhs
						);

		/** Destructor. 
		 */
		virtual ~TransferDevice( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		TransferDevice& operator=(	const TransferDevice& rhs
									);


		/** Returns dimension of transfer device.
		 *
		 *  \return Dimension of transfer device
		 */
		inline uint getDim( ) const;

		/** Returns whether transfer device is empty (i.e. has dimension 0).
		 *
		 *  \return BT_TRUE  iff transfer device has dimension 0, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType isEmpty( ) const;


		/** Returns whether additive noise has been specified.
		 *
		 *  \return BT_TRUE  iff additive noise has been specified, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasNoise( ) const;

		/** Returns whether dead time have been specified.
		 *
		 *  \return BT_TRUE  iff dead time have been specified, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasDeadTime( ) const;



	//
	// PROTECTED MEMBER FUNCTIONS:
	//
	protected:

		/** Initializes all components of the transfer device and the 
		 *	lastSignal member based on the given start information.
		 *
		 *	@param[in]  _startTime		Start time.
		 *	@param[in]  _startValue		Initial value of the transfer device signal.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue init(	double _startTime = 0.0,
									const DVector& _startValue = emptyConstVector
									);

		/** Generates additive noise on the given time interval based on the internal
		 *	noise settings and sampling times.
		 *
		 *	@param[in]  startTime		Start time for noise generation.
		 *	@param[in]  endTime			End time for noise generation.
		 *	@param[out] currentNoise	Generated additive noise on given time interval.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		virtual returnValue generateNoise(	double startTime,
											double endTime,
											VariablesGrid& currentNoise
											) const;



	//
	// DATA MEMBERS:
	//

	protected:

		VariablesGrid lastSignal;					/**< Most recent transfer device signal. */

		Noise** additiveNoise;						/**< Array of additive noise for each component of the transfer device signal. */
		DVector  noiseSamplingTimes;					/**< Noise sampling times for each component of the transfer device signal. */

		DVector  deadTimes;							/**< Dead times for each component of the transfer device signal. */
};


CLOSE_NAMESPACE_ACADO



#include <acado/transfer_device/transfer_device.ipp>


#endif  // ACADO_TOOLKIT_TRANSFER_DEVICE_HPP

/*
 *	end of file
 */
