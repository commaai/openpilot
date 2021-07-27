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
 *	\file include/acado/transfer_device/actuator.hpp
 *	\author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_ACTUATOR_HPP
#define ACADO_TOOLKIT_ACTUATOR_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/transfer_device/transfer_device.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Allows to simulate the behaviour of actuators within the Process.
 *
 *	\ingroup UserDataStructures
 *
 *  The class Actuator allows to simulate the behaviour of actuators within 
 *	the Process.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class Actuator : public TransferDevice
{
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor.
		 */
		Actuator( );

		/** Constructor which takes the number of process inputs and the sampling time. 
		 *
		 *	@param[in] _nU				Number of control inputs to the process.
		 *	@param[in] _nP				Number of parameter inputs to the process.
		 *	@param[in] _samplingTime	Sampling time.
		 */
		Actuator(	uint _nU,
					uint _nP = 0,
					double _samplingTime = DEFAULT_SAMPLING_TIME
					);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		Actuator(	const Actuator& rhs
					);

		/** Destructor.
		 */
		virtual ~Actuator( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		Actuator& operator=(	const Actuator& rhs
								);


		/** Assigns new additive noise with given sampling time to all 
		 *	components of the actuator control signal.
		 *
		 *	@param[in]  _noise				New additive noise.
		 *	@param[in]  _noiseSamplingTime	New noise sampling time.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue setControlNoise(	const Noise& _noise,
										double _noiseSamplingTime
										);

		/** Assigns new additive noise with given sampling time to given
		 *	component of the actuator control signal.
		 *
		 *	@param[in]  idx					Index of component.
		 *	@param[in]  _noise				New additive noise.
		 *	@param[in]  _noiseSamplingTime	New noise sampling time.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue setControlNoise(	uint idx,
										const Noise& _noise,
										double _noiseSamplingTime
										);


		/** Assigns new additive noise with given sampling time to all 
		 *	components of the actuator parameter signal.
		 *
		 *	@param[in]  _noise				New additive noise.
		 *	@param[in]  _noiseSamplingTime	New noise sampling time.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue setParameterNoise(	const Noise& _noise,
										double _noiseSamplingTime
										);

		/** Assigns new additive noise with given sampling time to given
		 *	component of the actuator parameter signal.
		 *
		 *	@param[in]  idx					Index of component.
		 *	@param[in]  _noise				New additive noise.
		 *	@param[in]  _noiseSamplingTime	New noise sampling time.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue setParameterNoise(	uint idx,
										const Noise& _noise,
										double _noiseSamplingTime
										);


		/** Assigns new dead times to each component of the actuator control signal.
		 *
		 *	@param[in]  _deadTimes			New dead times.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS, \n
		 *	        RET_MEMBER_NOT_INITIALISED
		 */
		returnValue setControlDeadTimes(	const DVector& _deadTimes
											);

		/** Assigns new dead time to all components of the actuator control signal.
		 *
		 *	@param[in]  _deadTime			New dead time.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS, \n
		 *	        RET_MEMBER_NOT_INITIALISED
		 */
		returnValue setControlDeadTimes(	double _deadTime
											);

		/** Assigns new dead time to given component of the actuator control signal.
		 *
		 *	@param[in]  idx					Index of component.
		 *	@param[in]  _deadTimes			New dead time.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS, \n
		 *	        RET_MEMBER_NOT_INITIALISED
		 */
		returnValue setControlDeadTime(		uint idx,
											double _deadTime
											);


		/** Assigns new dead times to each component of the actuator parameter signal.
		 *
		 *	@param[in]  _deadTimes			New dead times.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS, \n
		 *	        RET_MEMBER_NOT_INITIALISED
		 */
		returnValue setParameterDeadTimes(	const DVector& _deadTimes
											);

		/** Assigns new dead time to all components of the actuator parameter signal.
		 *
		 *	@param[in]  _deadTime			New dead time.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS, \n
		 *	        RET_MEMBER_NOT_INITIALISED
		 */
		returnValue setParameterDeadTimes(	double _deadTime
											);

		/** Assigns new dead time to given component of the actuator parameter signal.
		 *
		 *	@param[in]  idx					Index of component.
		 *	@param[in]  _deadTimes			New dead time.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS, \n
		 *	        RET_MEMBER_NOT_INITIALISED
		 */
		returnValue setParameterDeadTime(	uint idx,
											double _deadTime
											);


		/** Initializes all components of the actuator and the 
		 *	lastSignal member based on the given start information.
		 *
		 *	@param[in]  _startTime		Start time.
		 *	@param[in]  _startValueU	Initial value of the actuator control signal.
		 *	@param[in]  _startValueP	Initial value of the actuator parameter signal.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue init(	double _startTime = 0.0,
									const DVector& _startValueU = emptyConstVector,
									const DVector& _startValueP = emptyConstVector
									);


		/** Performs one step of the actuator transforming the given signals
		 *	according to the internal actuator settings.
		 *
		 *	@param[in,out]  _u		Actuator control signal to be transformed.
		 *	@param[in,out]  _p		Actuator parameter signal to be transformed.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_BLOCK_NOT_READY, \n
		 *	        RET_INVALID_ARGUMENTS, \n
		 *	        RET_ACTUATOR_STEP_FAILED
		 */
		virtual returnValue step(	VariablesGrid& _u,
									VariablesGrid& _p = emptyVariablesGrid
									);


		/** Returns number of actuator control signal components.
		 *
		 *	\return Number of actuator control signal components.
		 */
		inline uint getNU( ) const;

		/** Returns number of actuator parameter signal components.
		 *
		 *	\return Number of actuator parameter signal components.
		 */
		inline uint getNP( ) const;


		/** Returns pointer to additive noise of given component of the actuator control signal.
		 *
		 *	@param[in]  idx			Index of component.
		 *
		 *  \return Pointer to additive noise of given component
		 */
		inline Noise* getControlNoise(	uint idx
										) const;

		/** Returns pointer to additive noise of given component of the actuator parameter signal.
		 *
		 *	@param[in]  idx			Index of component.
		 *
		 *  \return Pointer to additive noise of given component
		 */
		inline Noise* getParameterNoise(	uint idx
											) const;


		/** Returns dead time of given component of the actuator control signal.
		 *
		 *	@param[in]  idx			Index of component.
		 *
		 *  \return Dead time of given component
		 */
		inline double getControlDeadTime(	uint idx
											) const;

		/** Returns dead times of the actuator control signal.
		 *
		 *  \return Dead times of actuator control signal
		 */
		inline DVector getControlDeadTimes( ) const;


		/** Returns dead time of given component of the actuator parameter signal.
		 *
		 *	@param[in]  idx			Index of component.
		 *
		 *  \return Dead time of given component
		 */
		inline double getParameterDeadTime(	uint idx
											) const;

		/** Returns dead times of the actuator parameter signal.
		 *
		 *  \return Dead times of actuator parameter signal
		 */
		inline DVector getParameterDeadTimes( ) const;



	//
	// PROTECTED MEMBER FUNCTIONS:
	//
	protected:

		/** Checks whether the two actuator signal are valid and compatible.
		 *
		 *	@param[in]  _u		Actuator control signal.
		 *	@param[in]  _p		Actuator parameter signal.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS, \n
		 *	        RET_CONTROL_DIMENSION_MISMATCH, \n
		 *	        RET_PARAMETER_DIMENSION_MISMATCH
		 */
		returnValue checkInputConsistency(	const VariablesGrid& _u,
											const VariablesGrid& _p
											) const;


		/** Delays given signals according to the internal dead times.
		 *
		 *	@param[in,out]  _u		Actuator control signal to be delayed.
		 *	@param[in,out]  _p		Actuator parameter signal to be delayed.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_DELAYING_INPUTS_FAILED
		 */
		returnValue delayActuatorInput(	VariablesGrid& _u,
										VariablesGrid& _p
										);

		/** Returns time grids for representing the given actuator signals in 
		 *	delayed form.
		 *
		 *	@param[in]  _u			Actuator control signal to be delayed.
		 *	@param[in]  _p			Actuator parameter signal to be delayed.
		 *	@param[out] _uDelayed	Time grid for representing delayed actuator control signal.
		 *	@param[out] _pDelayed	Time grid for representing delayed actuator parameter signal.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue getDelayedInputGrids(	const VariablesGrid& _u,
											const VariablesGrid& _p,
											VariablesGrid& _uDelayed,
											VariablesGrid& _pDelayed
											) const;


		/** Adds noise to given signals.
		 *
		 *	@param[in,out]  _u		Actuator control signal to be noised.
		 *	@param[in,out]  _p		Actuator parameter signal to be noised.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_GENERATING_NOISE_FAILED
		 */
		returnValue addActuatorNoise(	VariablesGrid& _u,
										VariablesGrid& _p
										) const;



	//
	// DATA MEMBERS:
	//
	protected:

		uint nU;						/**< Number of actuator control signal components. */
		uint nP;						/**< Number of actuator parameter signal components. */
};


CLOSE_NAMESPACE_ACADO


#include <acado/transfer_device/actuator.ipp>


#endif  // ACADO_TOOLKIT_ACTUATOR_HPP

/*
 *	end of file
 */
