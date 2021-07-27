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
 *    \file include/acado/transfer_device/sensor.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_SENSOR_HPP
#define ACADO_TOOLKIT_SENSOR_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/transfer_device/transfer_device.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Allows to simulate the behaviour of sensors within the Process.
 *
 *	\ingroup UserDataStructures
 *
 *  The class Sensor allows to simulate the behaviour of sensors within 
 *	the Process.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class Sensor : public TransferDevice
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. 
		 */
        Sensor( );

		/** Constructor which takes the number of process outputs and the sampling time. 
		 *
		 *	@param[in] _nY				Number of proces outputs.
		 *	@param[in] _samplingTime	Sampling time.
		 */
        Sensor(	uint _nY,
				double _samplingTime = DEFAULT_SAMPLING_TIME
				);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
        Sensor(	const Sensor& rhs
				);

        /** Destructor.
		 */
        virtual ~Sensor( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
        Sensor& operator=(	const Sensor& rhs
							);


		/** Assigns new additive noise with given sampling time to all 
		 *	components of the sensor signal.
		 *
		 *	@param[in]  _noise				New additive noise.
		 *	@param[in]  _noiseSamplingTime	New noise sampling time.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue setOutputNoise(	const Noise& _noise,
									double _noiseSamplingTime
									);

		/** Assigns new additive noise with given sampling time to given
		 *	component of the sensor signal.
		 *
		 *	@param[in]  idx					Index of component.
		 *	@param[in]  _noise				New additive noise.
		 *	@param[in]  _noiseSamplingTime	New noise sampling time.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue setOutputNoise(	uint idx,
									const Noise& _noise,
									double _noiseSamplingTime
									);


		/** Assigns new dead times to each component of the sensor signal.
		 *
		 *	@param[in]  _deadTimes			New dead times.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS, \n
		 *	        RET_MEMBER_NOT_INITIALISED
		 */
		returnValue setOutputDeadTimes(	const DVector& _deadTimes
										);

		/** Assigns new dead time to all components of the sensor signal.
		 *
		 *	@param[in]  _deadTime			New dead time.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS, \n
		 *	        RET_MEMBER_NOT_INITIALISED
		 */
		returnValue setOutputDeadTimes(	double _deadTime
										);

		/** Assigns new dead time to given component of the sensor signal.
		 *
		 *	@param[in]  idx					Index of component.
		 *	@param[in]  _deadTimes			New dead time.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS, \n
		 *	        RET_MEMBER_NOT_INITIALISED
		 */
		returnValue setOutputDeadTime(	uint idx,
										double _deadTime
										);


		/** Initializes all components of the sensor and the 
		 *	lastSignal member based on the given start information.
		 *
		 *	@param[in]  _startTime		Start time.
		 *	@param[in]  _startValue		Initial value of the sensor signal.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue init(	double _startTime = 0.0,
									const DVector& _startValue = emptyConstVector
									);


		/** Performs one step of the sensor transforming the given signal
		 *	according to the internal sensor settings.
		 *
		 *	@param[in,out]  _y		Sensor signal to be transformed.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_INVALID_ARGUMENTS, \n
		 *	        RET_SENSOR_STEP_FAILED
		 */
		virtual returnValue step(	VariablesGrid& _y
									);


		/** Returns number of sensor signal components.
		 *
		 *	\return Number of sensor signal components.
		 */
		inline uint getNY( ) const;


		/** Returns pointer to additive noise of given component of the sensor signal.
		 *
		 *	@param[in]  idx			Index of component.
		 *
		 *  \return Pointer to additive noise of given component
		 */
		inline Noise* getOutputNoise(	uint idx
										) const;


		/** Returns dead time of given component of the sensor signal.
		 *
		 *	@param[in]  idx			Index of component.
		 *
		 *  \return Dead time of given component
		 */
		inline double getOutputDeadTime(	uint idx
											) const;

		/** Returns dead times of the sensor signal.
		 *
		 *  \return Dead times of sensor signal
		 */
		inline DVector getOutputDeadTimes( ) const;



    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

		/** Delays given signal according to the internal dead times.
		 *
		 *	@param[in,out]  _y		Sensor signal to be delayed.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_DELAYING_INPUTS_FAILED
		 */
		returnValue delaySensorOutput(	VariablesGrid& _y
										);

		/** Returns time grid for representing the given sensor signal in 
		 *	delayed form.
		 *
		 *	@param[in]  _y			Sensor signal to be delayed.
		 *	@param[out] _yDelayed	Time grid for representing delayed sensor signal.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue getDelayedOutputGrid(	const VariablesGrid& _y,
											VariablesGrid& _yDelayed
											) const;

		/** Adds noise to given signal.
		 *
		 *	@param[in,out]  _y		Sensor signal to be noised.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_GENERATING_NOISE_FAILED
		 */
		returnValue addSensorNoise(	VariablesGrid& _y
									) const;



	//
	// DATA MEMBERS:
	//
	protected:

};


CLOSE_NAMESPACE_ACADO



#include <acado/transfer_device/sensor.ipp>


#endif  // ACADO_TOOLKIT_SENSOR_HPP

/*
 *	end of file
 */
