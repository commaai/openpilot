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
 *	\file include/acado/control_law/clipping_functionality.hpp
 *	\author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_CLIPPING_FUNCTIONALITY_HPP
#define ACADO_TOOLKIT_CLIPPING_FUNCTIONALITY_HPP


#include <acado/utils/acado_utils.hpp>

#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/variables_grid/variables_grid.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Allows to transform the output of the ControlLaw before passing it to the Process.
 *
 *	\ingroup AuxiliaryFunctionality
 *
 *  The class ClippingFunctionality allows to limit the output of the 
 *	ControlLaw before passing it as signal to control the Process.
 *
 *	 \author Hans Joachim Ferreau, Boris Houska
 */
class ClippingFunctionality
{
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor. 
		 */
		ClippingFunctionality( );

		/** Constructor which takes dimensions of the signals to be clipped.
		 *
		 *	@param[in] _nU		Number of control signals to be clipped.
		 *	@param[in] _nP		Number of parameter signals to be clipped.
		 */
		ClippingFunctionality(	uint _nU,
								uint _nP = 0
								);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		ClippingFunctionality(	const ClippingFunctionality& rhs
								);

		/** Destructor. 
		 */
		~ClippingFunctionality( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		ClippingFunctionality& operator=(	const ClippingFunctionality& rhs
											);


		/** Assigns new lower limits on control signals.
		 *
		 *	@param[in]  _lowerLimit		New lower limits on control signals.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_VECTOR_DIMENSION_MISMATCH
		 */
		returnValue setControlLowerLimits(	const DVector& _lowerLimit
											);

		/** Assigns new lower limit on given component of the control signal.
		 *
		 *	@param[in]  idx				Index of control signal component.
		 *	@param[in]  _lowerLimit		New lower limit.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		returnValue setControlLowerLimit(	uint idx,
											double _lowerLimit
											);

		/** Assigns new upper limits on control signals.
		 *
		 *	@param[in]  _upperLimit		New upper limits on control signals.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_VECTOR_DIMENSION_MISMATCH
		 */
		returnValue setControlUpperLimits(	const DVector& _upperLimit
											);

		/** Assigns new upper limit on given component of the control signal.
		 *
		 *	@param[in]  idx				Index of control signal component.
		 *	@param[in]  _upperLimit		New upper limit.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		returnValue setControlUpperLimit(	uint idx,
											double _upperLimit
											);


		/** Assigns new lower limits on parameter signals.
		 *
		 *	@param[in]  _lowerLimit		New lower limits on parameter signals.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_VECTOR_DIMENSION_MISMATCH
		 */
		returnValue setParameterLowerLimits(	const DVector& _lowerLimit
												);

		/** Assigns new lower limit on given component of the parameter signal.
		 *
		 *	@param[in]  idx				Index of parameter signal component.
		 *	@param[in]  _lowerLimit		New lower limit.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		returnValue setParameterLowerLimit(		uint idx,
												double _lowerLimit
												);

		/** Assigns new upper limits on parameter signals.
		 *
		 *	@param[in]  _upperLimit		New upper limits on parameter signals.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_VECTOR_DIMENSION_MISMATCH
		 */
		returnValue setParameterUpperLimits(	const DVector& _upperLimit
												);

		/** Assigns new upper limit on given component of the parameter signal.
		 *
		 *	@param[in]  idx				Index of parameter signal component.
		 *	@param[in]  _upperLimit		New upper limit.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		returnValue setParameterUpperLimit(	uint idx,
											double _upperLimit
											);


	//
	// PROTECTED MEMBER FUNCTIONS:
	//
	protected:

		/** Actually clips given control and parameter signals.
		 *
		 *	@param[in,out] _u	Control signal sequence to be clipped.
		 *	@param[in,out] _p	Parameter signal sequence to be clipped.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_VECTOR_DIMENSION_MISMATCH
		 */
		returnValue clipSignals(	VariablesGrid& _u,
									VariablesGrid& _p = emptyVariablesGrid
									);

		/** Actually clips given control and parameter signals.
		 *
		 *	@param[in,out] _u	Control signal to be clipped.
		 *	@param[in,out] _p	Parameter signal to be clipped.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_VECTOR_DIMENSION_MISMATCH
		 */
		returnValue clipSignals(	DVector& _u,
									DVector& _p = emptyVector
									);


		/** Returns number of control signal limits.
		 *
		 *  \return Number of control signal limits
		 */
		inline uint getNumControlLimits( ) const;

		/** Returns number of parameter signal limits.
		 *
		 *  \return Number of parameter signal limits
		 */
		inline uint getNumParameterLimits( ) const;



	//
	// DATA MEMBERS:
	//
	protected:

		DVector lowerLimitControls;					/**< Lower limits on control signals. */
		DVector upperLimitControls;					/**< Upper limits on control signals. */

		DVector lowerLimitParameters;				/**< Lower limits on parameter signals. */
		DVector upperLimitParameters;				/**< Upper limits on parameter signals. */
};


CLOSE_NAMESPACE_ACADO



#include <acado/control_law/clipping_functionality.ipp>


#endif  // ACADO_TOOLKIT_CLIPPING_FUNCTIONALITY_HPP

/*
 *	end of file
 */
