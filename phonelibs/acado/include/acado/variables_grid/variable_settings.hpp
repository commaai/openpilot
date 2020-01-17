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
 *    \file include/acado/variables_grid/variable_settings.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_VARIABLE_SETTINGS_HPP
#define ACADO_TOOLKIT_VARIABLE_SETTINGS_HPP


#include <string.h>

#include <acado/matrix_vector/matrix_vector.hpp>


BEGIN_NAMESPACE_ACADO


const char defaultName[] = "-";
const char defaultUnit[] = "-";

const double defaultScaling = 1.0;
const double defaultUpperBound =  INFTY;
const double defaultLowerBound = -INFTY;

const BooleanType defaultAutoInit = BT_TRUE;


/**
 *	\brief Provides variable-specific settings for vector- or matrix-valued optimization variables (for internal use).
 *
 *	\ingroup BasicDataStructures
 *
 *  The class VariableSettings provides variable-specific settings for 
 *	enhancing a DVector or a DMatrix to vector- or matrix-valued 
 *	optimization variables. It is intended for internal use only.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class VariableSettings
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. 
		 */
        VariableSettings( );

		/** Constructor which takes all settings.
		 *
		 *	@param[in] _dim			Dimension of variable.
		 *	@param[in] _type		Type of the variable.
		 *	@param[in] _names		Array containing name labels for each component of the variable.
		 *	@param[in] _units		Array containing unit labels for each component of the variable.
		 *	@param[in] _scaling		Scaling for each component of the variable.
		 *	@param[in] _lb			Lower bounds for each component of the variable.
		 *	@param[in] _ub			Upper bounds for each component of the variable.
		 *	@param[in] _autoInit	Flag indicating whether variable is to be automatically initialized.
		 */
        VariableSettings(	uint _dim,
							VariableType _type = VT_UNKNOWN,
							const char** const _names = 0,
							const char** const _units = 0,
							const DVector& _scaling = emptyConstVector,
							const DVector& _lb = emptyConstVector,
							const DVector& _ub = emptyConstVector,
							BooleanType _autoInit = defaultAutoInit
							);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		VariableSettings(	const VariableSettings& rhs
							);

		/** Destructor. 
		 */
		~VariableSettings( );

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		VariableSettings& operator=(	const VariableSettings& rhs
										);


		/** Initializes empty object.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue init( );

		/** Initializes object with given dimension and settings.
		 *
		 *	@param[in] _dim			Dimension of variable.
		 *	@param[in] _type		Type of the variable.
		 *	@param[in] _names		Array containing name labels for each component of the variable.
		 *	@param[in] _units		Array containing unit labels for each component of the variable.
		 *	@param[in] _scaling		Array containing the scaling for each component of the variable.
		 *	@param[in] _lb			Array containing lower bounds for each component of the variable.
		 *	@param[in] _ub			Array containing upper bounds for each component of the variable.
		 *	@param[in] _autoInit	Array defining if each component of the variable is to be automatically initialized.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue init(	uint _dim,
							VariableType _type,
							const char** const _names,
							const char** const _units,
							const DVector& _scaling = emptyConstVector,
							const DVector& _lb = emptyConstVector,
							const DVector& _ub = emptyConstVector,
							BooleanType _autoInit = defaultAutoInit
							);


		/** Appends given VariableSettings object as additional components
		 *	to the current one.
		 *
		 *	@param[in] rhs			VariableSettings to be appended.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue appendSettings(	const VariableSettings& rhs
									);

		/** Appends given VariableSettings object as additional components
		 *	to the current one.
		 *
		 *	@param[in] _dim			Dimension of variable.
		 *	@param[in] _names		Array containing name labels for each component of the variable.
		 *	@param[in] _units		Array containing unit labels for each component of the variable.
		 *	@param[in] _scaling		Array containing the scaling for each component of the variable.
		 *	@param[in] _lb			Array containing lower bounds for each component of the variable.
		 *	@param[in] _ub			Array containing upper bounds for each component of the variable.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue appendSettings(	uint _dim,
									const char** const _names,
									const char** const _units,
									const DVector& _scaling = emptyConstVector,
									const DVector& _lb = emptyConstVector,
									const DVector& _ub = emptyConstVector
									);


		/** Returns current variable type.
		 *
		 *  \return Current variable type
		 */
        inline VariableType getType( ) const;

		/** Assigns new variable type.
		 *
		 *	@param[in] _type		New variable type.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
        inline returnValue setType(	VariableType _type
									);


		/** Returns current name label of given component.
		 *
		 *	@param[in]  idx			Index of component.
		 *	@param[out] _name		Name label of given component.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS, \n
		 *	        RET_MEMBER_NOT_INITIALISED
		 */
		returnValue getName(	uint idx,
								char* _name
								) const;

		/** Assigns new name label to given component.
		 *
		 *	@param[in]  idx			Index of component.
		 *	@param[in]  _name		New name label.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS, \n
		 *	        RET_MEMBER_NOT_INITIALISED
		 */
		returnValue setName(	uint idx,
								const char* const _name
								);


		/** Returns current unit label of given component.
		 *
		 *	@param[in]  idx			Index of component.
		 *	@param[out] _unit		Unit label of given component.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS, \n
		 *	        RET_MEMBER_NOT_INITIALISED
		 */
		returnValue getUnit(	uint idx,
								char* _unit
								) const;

		/** Assigns new unit label to given component.
		 *
		 *	@param[in]  idx			Index of component.
		 *	@param[in]  _unit		New unit label.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS, \n
		 *	        RET_MEMBER_NOT_INITIALISED
		 */
		returnValue setUnit(	uint idx,
								const char* const _unit
								);


		/** Returns current scaling.
		 *
		 *  \return Current scaling
		 */
		inline DVector getScaling( ) const;

		/** Assigns new scaling.
		 *
		 *	@param[in] _scaling		New scaling.
		 *
		 *	\note Scaling factors need to be positive.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_VECTOR_DIMENSION_MISMATCH, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		inline returnValue setScaling(	const DVector& _scaling
										);

		/** Returns current scaling of given component.
		 *
		 *	@param[in] idx		Index of component.
		 *
		 *  \return Current scaling of given component
		 */
		inline double getScaling(	uint idx
									) const;

		/** Assigns new scaling to given component.
		 *
		 *	@param[in] idx			Index of component.
		 *	@param[in] _scaling		New scaling.
		 *
		 *	\note Scaling factor needs to be positive.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		inline returnValue setScaling(	uint idx,
										double _scaling
										);


		/** Returns current lower bounds.
		 *
		 *  \return Current lower bounds
		 */
		inline DVector getLowerBounds( ) const;

		/** Assigns new lower bounds.
		 *
		 *	@param[in] _lb		New lower bounds.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_VECTOR_DIMENSION_MISMATCH
		 */
		inline returnValue setLowerBounds(	const DVector& _lb
											);

		/** Returns current lower bound of given component.
		 *
		 *	@param[in] idx		Index of component.
		 *
		 *  \return Current lower bound of given component
		 */
		inline double getLowerBound(	uint idx
										) const;

		/** Assigns new lower bound to given component.
		 *
		 *	@param[in] idx		Index of component.
		 *	@param[in] _lb		New lower bound.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		inline returnValue setLowerBound(	uint idx,
											double _lb
											);


		/** Returns current upper bounds.
		 *
		 *  \return Current upper bounds
		 */
		inline DVector getUpperBounds( ) const;

		/** Assigns new upper bounds.
		 *
		 *	@param[in] _ub		New upper bounds.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_VECTOR_DIMENSION_MISMATCH
		 */
		inline returnValue setUpperBounds(	const DVector& _ub
											);

		/** Returns current upper bound of given component.
		 *
		 *	@param[in] idx		Index of component.
		 *
		 *  \return Current upper bound of given component
		 */
		inline double getUpperBound(	uint idx
										) const;

		/** Assigns new upper bound to given component.
		 *
		 *	@param[in] idx		Index of component.
		 *	@param[in] _ub		New upper bound.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		inline returnValue setUpperBound(	uint idx,
											double _ub
											);


		/** Returns whether automatic initialization is enabled or not.
		 *
		 *  \return BT_TRUE  iff automatic initialization is enabled, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType getAutoInit( ) const;

		/** Assigns new auto initialization flag.
		 *
		 *	@param[in] _autoInit	New auto initialization flag.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue setAutoInit(	BooleanType _autoInit
										);


		/** Returns whether VariableSettings comprises (non-empty) name labels
		 *	(at at least one of its grid points).
		 *
		 *  \return BT_TRUE  iff VariableSettings comprises name labels, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasNames( ) const;

		/** Returns whether VariableSettings comprises (non-empty) unit labels
		 *	(at at least one of its grid points).
		 *
		 *  \return BT_TRUE  iff VariableSettings comprises unit labels, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasUnits( ) const;

		/** Returns whether scaling is set (at at least one grid point).
		 *
		 *  \return BT_TRUE  iff scaling is set, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasScaling( ) const;

		/** Returns whether VariableSettings comprises lower bounds 
		 *	(at at least one of its grid points).
		 *
		 *  \return BT_TRUE  iff VariableSettings comprises lower bounds, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasLowerBounds( ) const;

		/** Returns whether VariableSettings comprises upper bounds
		 *	(at at least one of its grid points).
		 *
		 *  \return BT_TRUE  iff VariableSettings comprises upper bounds, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasUpperBounds( ) const;



    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

		/** Clears all settings.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue clear( );



    //
    // DATA MEMBERS:
    //
    protected:

		uint dim;							/**< Dimension of variable. */
		VariableType type;					/**< Type of the variable. */

		char** names;						/**< Array containing name labels for each component of the variable.. */
		char** units;						/**< Array containing unit labels for each component of the variable.. */

		DVector scaling;			/**< Scaling for each component of the variable. */

		DVector lb;				/**< Lower bounds for each component of the variable. */
		DVector ub;				/**< Upper bounds for each component of the variable. */

        BooleanType autoInit;				/**< Flag indicating whether variable is to be automatically initialized. */
};


CLOSE_NAMESPACE_ACADO


#include <acado/variables_grid/variable_settings.ipp>


#endif  // ACADO_TOOLKIT_VARIABLE_SETTINGS_HPP

/*
 *	end of file
 */
