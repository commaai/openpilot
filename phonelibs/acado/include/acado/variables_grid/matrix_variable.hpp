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
 *    \file include/acado/variables_grid/matrix_variable.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_MATRIX_VARIABLE_HPP
#define ACADO_TOOLKIT_MATRIX_VARIABLE_HPP


#include <acado/variables_grid/variable_settings.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>


BEGIN_NAMESPACE_ACADO



/**
 *	\brief Provides matrix-valued optimization variables.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class MatrixVariable provides matrix-valued optimization variables by
 *	enhancing the DMatrix class with variable-specific settings.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class MatrixVariable : public DMatrix, public VariableSettings
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:
        /** Default constructor. 
		 */
        MatrixVariable( );

		/** Constructor which takes dimensions of the matrix as well as
		 *	all variable settings.
		 *
		 *	@param[in] _nRows		Number of rows of each matrix.
		 *	@param[in] _nCols		Number of columns of each matrix.
		 *	@param[in] _type		Type of the variable.
		 *	@param[in] _names		Array containing name labels for each component of the variable.
		 *	@param[in] _units		Array containing unit labels for each component of the variable.
		 *	@param[in] _scaling		Scaling for each component of the variable.
		 *	@param[in] _lb			Lower bounds for each component of the variable.
		 *	@param[in] _ub			Upper bounds for each component of the variable.
		 *	@param[in] _autoInit	Flag indicating whether variable is to be automatically initialized.
		 */
		MatrixVariable(	uint _nRows,
						uint _nCols,
						VariableType _type = VT_UNKNOWN,
						const char** const _names = 0,
						const char** const _units = 0,
						DVector _scaling = emptyVector,
						DVector _lb = emptyVector,
						DVector _ub = emptyVector,
						BooleanType _autoInit = defaultAutoInit
						);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		MatrixVariable(	const MatrixVariable& rhs
						);

		/** Copy constructor converting a matrix to a MatrixVariable (of given type).
		 *
		 *	@param[in] _matrix	DMatrix to be converted.
		 *	@param[in] _type	Type of the variable.
		 */
        MatrixVariable( const DMatrix& _matrix,
						VariableType _type = VT_UNKNOWN
						);

        /** Destructor. 
		 */
        ~MatrixVariable( );

        /** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
        MatrixVariable& operator=(	const MatrixVariable& rhs
									);

        /** Assignment operator converting a matrix to a MatrixVariable.
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
        MatrixVariable& operator=(	const DMatrix& rhs
									);


		/** Initializes object with given dimensions of the matrix and 
		 *	given variable settings.
		 *
		 *	@param[in] _nRows		Number of rows of each matrix.
		 *	@param[in] _nCols		Number of columns of each matrix.
		 *	@param[in] _type		Type of the variable.
		 *	@param[in] _names		Array containing name labels for each component of the variable.
		 *	@param[in] _units		Array containing unit labels for each component of the variable.
		 *	@param[in] _scaling		Scaling for each component of the variable.
		 *	@param[in] _lb			Lower bounds for each component of the variable.
		 *	@param[in] _ub			Upper bounds for each component of the variable.
		 *	@param[in] _autoInit	Flag indicating whether variable is to be automatically initialized.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_UNKNOWN_BUG
		 */
		returnValue init(	uint _nRows,
							uint _nCols,
							VariableType _type = VT_UNKNOWN,
							const char** const _names = 0,
							const char** const _units = 0,
							DVector _scaling = emptyVector,
							DVector _lb = emptyVector,
							DVector _ub = emptyVector,
							BooleanType _autoInit = defaultAutoInit
							);


		/** Returns matrix containing the numerical values of the MatrixVariable.
		 *
		 *	\return DMatrix containing the numerical values
		 */
		inline DMatrix getMatrix( ) const;


		/** Returns a MatrixVariable containing only the rows between given
		 *	indices while keeping all columns.
		 *
		 *	@param[in] startIdx		Index of first row to be included.
		 *	@param[in] endIdx		Index of last row to be included.
		 *
		 *	\note Is not fully implemented yet!
		 *
		 *	\return DMatrix containing desired rows
		 */
		MatrixVariable getRows(	uint startIdx,
								uint endIdx
								) const;

		/** Returns a MatrixVariable containing only the columns between given
		 *	indices while keeping all rows.
		 *
		 *	@param[in] startIdx		Index of first column to be included.
		 *	@param[in] endIdx		Index of last column to be included.
		 *
		 *	\note Is not fully implemented yet!
		 *
		 *	\return DMatrix containing desired columns
		 */
		MatrixVariable getCols(	uint startIdx,
								uint endIdx
								) const;


    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:


    //
    // DATA MEMBERS:
    //
    protected:

};


CLOSE_NAMESPACE_ACADO



#include <acado/variables_grid/matrix_variable.ipp>


#endif  // ACADO_TOOLKIT_MATRIX_VARIABLE_HPP

/*
 *	end of file
 */
