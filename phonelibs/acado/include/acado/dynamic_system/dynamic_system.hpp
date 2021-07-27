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
*	\file include/acado/dynamic_system/dynamic_system.hpp
*	\author Hans Joachim Ferreau, Boris Houska
*/



#ifndef ACADO_TOOLKIT_DYNAMIC_SYSTEM_HPP
#define ACADO_TOOLKIT_DYNAMIC_SYSTEM_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/function/function.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Stores a DifferentialEquation together with an OutputFcn.
 *
 *	\ingroup UserDataStructures
 *
 *  The class DynamicSystem is a data class for storing a DifferentialEquation
 *	together with an OutputFcn. The dynamic system might be of hybrid nature,
 *	i.e. differential equation and output function might switch depending on
 *	a state-dependend switch function.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class DynamicSystem
{
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/**< Default Constructor. 
		 */
		DynamicSystem( );

		/** Constructor which takes differential equation of first stage.
		 *
		 *	@param[in] _diffEqn		Differential equation.
		 */
		 DynamicSystem(	const DifferentialEquation& _diffEqn
						);

		/** Constructor which takes differential equation and output function 
		 *	of first stage.
		 *
		 *	@param[in] _diffEqn		Differential equation.
		 *	@param[in] _outputFcn	Output function.
		 */
		 DynamicSystem(	const DifferentialEquation& _diffEqn,
						const OutputFcn& _outputFcn
						);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		DynamicSystem(	const DynamicSystem &rhs
						);

		/** Destructor.
		 */
		~DynamicSystem( );

		/**< Assignment Operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		DynamicSystem& operator=(	const DynamicSystem& rhs
									);


		/** Adds a new dynamic system stage comprising the given differential equation.
		 *
		 *	@param[in] _diffEqn		Differential equation.
		 *
		 *	\return RET_NOT_YET_IMPLEMENTED, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue addSubsystem(	const DifferentialEquation& _diffEqn
									);

		/** Adds a new dynamic system stage comprising the given differential equation
		 *	and output function.
		 *
		 *	@param[in] _diffEqn		Differential equation.
		 *	@param[in] _outputFcn	Output function.
		 *
		 *	\return RET_NOT_YET_IMPLEMENTED, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue addSubsystem(	const DifferentialEquation& _diffEqn,
									const OutputFcn& _outputFcn
									);


		/** (not yet documented)
		 *
		 *	@param[in] _switchFcn		.
		 *
		 *	\return RET_NOT_YET_IMPLEMENTED
		 */
		returnValue addSwitchFunction(	const Function& _switchFcn
										);

		/** (not yet documented)
		 *
		 *	@param[in] _selectFcn		.
		 *
		 *	\return RET_NOT_YET_IMPLEMENTED
		 */
		returnValue setSelectFunction(	const Function& _selectFcn
										);


		/** Returns dynamic subsystem at given stage.
		 *
		 *	@param[in]  stageIdx		Index of stage.
		 *	@param[out] _diffEqn		Differential equation at given stage.
		 *	@param[out] _outputFcn		Output function at given stage.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		inline returnValue getSubsystem(	uint stageIdx,
											DifferentialEquation& _diffEqn,
											OutputFcn& _outputFcn
											) const;

		/** Returns differential equation at given stage.
		 *
		 *	@param[in]  stageIdx	Index of stage.
		 *
		 *	\return Differential equation at given stage
		 */
		inline const DifferentialEquation& getDifferentialEquation(	uint stageIdx = 0
																) const;

		/** Returns output function at given stage.
		 *
		 *	@param[in]  stageIdx	Index of stage.
		 *
		 *	\return Output function at given stage
		 */
		inline const OutputFcn& getOutputFcn(	uint stageIdx = 0
										) const;

		/** (not yet documented)
		 *
		 *	@param[in] _switchFcn		.
		 *
		 *	\return RET_NOT_YET_IMPLEMENTED
		 */
		inline returnValue getSwitchFunction(	uint idx,
												Function& _switchFcn
												) const;

		/** (not yet documented)
		 *
		 *	@param[in] _selectFcn		.
		 *
		 *	\return RET_NOT_YET_IMPLEMENTED
		 */
		inline returnValue getSelectFunction(	Function& _selectFcn
												) const;


		/** Returns whether dynamic system is an ODE.
		 *
		 *  \return BT_TRUE  iff dynamic system is an ODE, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType isODE( ) const;

		/** Returns whether dynamic system is a DAE.
		 *
		 *  \return BT_TRUE  iff dynamic system is a DAE, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType isDAE( ) const;


		/** Returns whether dynamic system is discretized in time.
		 *
		 *  \return BT_TRUE  iff dynamic system is discretized in time, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType isDiscretized( ) const;

		/** Returns whether dynamic system is continuous in time.
		 *
		 *  \return BT_TRUE  iff dynamic system is continuous in time, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType isContinuous( ) const;

		/** Returns sample time of the dynamic system.
		 *
		 *	\return    > 0: sample time of discretized system, \n
		 *	        -INFTY: system is time-continuous
		 */
		inline double getSampleTime( ) const;


		/** Returns number of dynamic equations of the dynamic system.
		 *
		 *  \return Number of dynamic equations
		 */
		inline uint getNumDynamicEquations( ) const;

		/** Returns number of algebraic equations of the dynamic system.
		 *
		 *  \return Number of algebraic equations
		 */
		inline uint getNumAlgebraicEquations( ) const;

		/** Returns number of outputs of the dynamic system.
		 *
		 *  \return Number of outputs equations
		 */
		inline uint getNumOutputs( ) const;


		/** Returns maximum number of controls of the dynamic system.
		 *
		 *  \return Maximum number of controls
		 */
		inline uint getNumControls( ) const;

		/** Returns maximum number of parameters of the dynamic system.
		 *
		 *  \return Maximum number of parameters
		 */
		inline uint getNumParameters( ) const;

		/** Returns maximum number of disturbances of the dynamic system.
		 *
		 *  \return Maximum number of disturbances
		 */
		inline uint getNumDisturbances( ) const;


		/** Returns number of subsystems (i.e. stages) of the dynamic system.
		 *
		 *  \return Number of subsystems of the dynamic system
		 */
		inline uint getNumSubsystems( ) const;

		/** Returns number of switch functions of the dynamic system.
		 *
		 *  \return Number of switch functions of the dynamic system
		 */
		inline uint getNumSwitchFunctions( ) const;

		/** Returns whether dynamic system has implicit switches.
		 *
		 *	\return BT_TRUE  iff dynamic system has implicit switches, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasImplicitSwitches( ) const;



	//
	// PROTECTED MEMBER FUNCTIONS:
	//
	protected:

		/** Returns whether given differential equation is consistent with the
		 *	existing ones at other stages.
		 *
		 *	@param[in] _diffEqn		Differential equation.
		 *
		 *	\return BT_TRUE  iff  differential equation is consistent, \n
		 *	        BT_FALSE otherwise
		 */
		BooleanType isConsistentDiffEqn(	const DifferentialEquation& _diffEqn
											) const;

		/** Returns whether given output function is consistent with the corresponding 
		 *	differential equation and existing output functions at other stages.
		 *
		 *	@param[in] _outputFcn	Output function.
		 *
		 *	\return BT_TRUE  iff  output function is consistent, \n
		 *	        BT_FALSE otherwise
		 */
		BooleanType isConsistentOutputFcn(	const OutputFcn& _outputFcn
											) const;



	//
	// DATA MEMBERS:
	//
	protected:

		uint nDiffEqn;							/**< Number of differential equations. */
		uint nSwitchFcn;						/**< Number of switch functions. */

		DifferentialEquation** diffEqn;			/**< Differential equation(s) describing the states of the dynamic system. */
		OutputFcn** outputFcn;					/**< Output function(s) for evaluating the output of the dynamic system. */

		Function** switchFcn;					/**< Function(s) for determining switches between different differential equations. */
		Function* selectFcn;					/**< Function for selecting the current differential equation based on the values of the switch function(s). */
};


CLOSE_NAMESPACE_ACADO



#include <acado/dynamic_system/dynamic_system.ipp>


#endif	// ACADO_TOOLKIT_DYNAMIC_SYSTEM_HPP


/*
 *	end of file
 */
