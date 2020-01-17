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
 *    \file include/acado/code_generation/export_variable.hpp
 *    \author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */


#ifndef ACADO_TOOLKIT_EXPORT_VARIABLE_INTERNAL_HPP
#define ACADO_TOOLKIT_EXPORT_VARIABLE_INTERNAL_HPP

#include <acado/utils/acado_utils.hpp>
#include <acado/code_generation/export_argument_internal.hpp>
#include <acado/code_generation/export_index.hpp>


BEGIN_NAMESPACE_ACADO


class ExportArithmeticStatement;


/**
 *	\brief Defines a matrix-valued variable to be used for exporting code.
 *
 *	\ingroup UserDataStructures
 *
 *	The class ExportVariableInternal defines a matrix-valued variable to be used for exporting
 *	code. Instances of this class can be used similar to usual DMatrix objects
 *	but offer additional functionality, e.g. they allow to export arithmetic
 *	expressions and they can be passed as argument to exported functions. By
 *	default, all entries of a ExportVariableInternal are undefined, but each of its
 *	component can be set to a fixed value if known beforehand.
 *
 *	\author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */

class ExportVariableInternal : public ExportArgumentInternal
{
	//
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

		/** Default constructor.
		 */
        ExportVariableInternal( );

		/** Constructor which takes the name and type string of the variable.
		 *	Moreover, it initializes the variable with the dimensions and the
		 *	values of the given matrix.
		 *
		 *	@param[in] _name			Name of the argument.
		 *	@param[in] _data			DMatrix used for initialization.
		 *	@param[in] _type			Data type of the argument.
		 *	@param[in] _dataStruct		Global data struct to which the argument belongs to (if any).
		 *	@param[in] _callByValue		Flag indicating whether argument it to be called by value.
		 */
		ExportVariableInternal(	const std::string& _name,
								const DMatrixPtr& _data,
								ExportType _type = REAL,
								ExportStruct _dataStruct = ACADO_LOCAL,
								bool _callItByValue = false,
								const std::string& _prefix = std::string()
								);

        /** Destructor.
		 */
		virtual ~ExportVariableInternal( );

		/** Clone constructor (deep copy).
		 *
		 *	\return Pointer to cloned object.
		 */
		virtual ExportVariableInternal* clone() const;

		/** Returns whether given component is set to zero.
		 *
		 *	@param[in] rowIdx		Variable row index of the component.
		 *	@param[in] colIdx		Variable column index of the component.
		 *
		 *	\return true  iff given component is set to zero, \n
		 *	        false otherwise
		 */
		bool isZero( const ExportIndex& rowIdx,
							const ExportIndex& colIdx
							) const;

		/** Returns whether given component is set to one.
		 *
		 *	@param[in] rowIdx		Variable row index of the component.
		 *	@param[in] colIdx		Variable column index of the component.
		 *
		 *	\return true  iff given component is set to one, \n
		 *	        false otherwise
		 */
		bool isOne(	const ExportIndex& rowIdx,
							const ExportIndex& colIdx
							) const;

		/** Returns whether given component is set to a given value.
		 *
		 *	@param[in] rowIdx		Variable row index of the component.
		 *	@param[in] colIdx		Variable column index of the component.
		 *
		 *	\return true  iff given component is set to a given value, \n
		 *	        false otherwise
		 */
		bool isGiven(	const ExportIndex& rowIdx,
								const ExportIndex& colIdx
								) const;

		virtual bool isGiven() const;


		/** Returns string containing the value of a given component. If its
		 *	value is undefined, the string contains the address of the component.
		 *
		 *	@param[in] rowIdx		Variable row index of the component.
		 *	@param[in] colIdx		Variable column index of the component.
		 *
		 *	\return std::string containing the value of a given component
		 */
		const std::string get(	const ExportIndex& rowIdx,
							const ExportIndex& colIdx
							) const;

		/** Returns number of rows of the variable.
		 *
		 *	\return Number of rows of the variable
		 */
		virtual uint getNumRows( ) const;

		/** Returns number of columns of the variable.
		 *
		 *	\return Number of columns of the variable
		 */
		virtual uint getNumCols( ) const;

		/** Returns total dimension of the variable.
		 *
		 *	\return Total dimension of the variable
		 */
		virtual uint getDim( ) const;

		/** Returns a copy of the variable with transposed components.
		 *
		 *	\return Copy of the variable with transposed components
		 */
		ExportVariable getTranspose( ) const;

		/** Returns a new variable containing only the given row of the variable.
		 *
		 *	@param[in] idx			Variable row index.
		 *
		 *	\return New variable containing only the given row of the variable
		 */
		ExportVariable getRow(	const ExportIndex& idx
								) const;

		/** Returns a new variable containing only the given column of the variable.
		 *
		 *	@param[in] idx			Variable column index.
		 *
		 *	\return New variable containing only the given column of the variable
		 */
		ExportVariable getCol(	const ExportIndex& idx
								) const;

		/** Returns a new variable containing only the given rows of the variable.
		 *
		 *	@param[in] idx1			Variable index of first row of new variable.
		 *	@param[in] idx2			Variable index following last row of new variable.
		 *
		 *	\return New variable containing only the given rows of the variable
		 */
		ExportVariable getRows(	const ExportIndex& idx1,
								const ExportIndex& idx2
								) const;

		/** Returns a new variable containing only the given columns of the variable.
		 *
		 *	@param[in] idx1			Variable index of first column of new variable.
		 *	@param[in] idx2			Variable index following last column of new variable.
		 *
		 *	\return New variable containing only the given columns of the variable
		 */
		ExportVariable getCols(	const ExportIndex& idx1,
								const ExportIndex& idx2
								) const;

		/** Returns a new variable containing only the given rows and columns of the variable.
		 *
		 *	@param[in] rowIdx1		Variable index of first row of new variable.
		 *	@param[in] rowIdx2		Variable index following last row of new variable.
		 *	@param[in] colIdx1		Variable index of first column of new variable.
		 *	@param[in] colIdx2		Variable index following last column of new variable.
		 *
		 *	\return New variable containing only the given sub-matrix of the variable
		 */
		ExportVariable getSubMatrix(	const ExportIndex& _rowIdx1,
										const ExportIndex& _rowIdx2,
										const ExportIndex& _colIdx1,
										const ExportIndex& _colIdx2
										) const;


		/** Returns a copy of the variable that is transformed to a row vector.
		 *
		 *	\return Copy of the variable that is transformed to a row vector
		 */
		ExportVariable makeRowVector( ) const;

		/** Returns a copy of the variable that is transformed to a column vector.
		 *
		 *	\return Copy of the variable that is transformed to a column vector
		 */
		ExportVariable makeColVector( ) const;


		/** Returns whether variable is a vector.
		 *
		 *	\return true  iff variable is a vector, \n
		 *	        false otherwise
		 */
		bool isVector( ) const;


		/** Returns the internal data matrix.
		 *
		 *	\return Internal data matrix
		 */
		const DMatrix& getGivenMatrix( ) const;


		/** Prints contents of variable to screen.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue print( ) const;

		/** Check whether the matrix is actually a submatrix. */
		bool isSubMatrix() const;

		/** Check whether the matrix is diagonal. */
		bool isDiagonal() const;

	//
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

		/** Returns column dimension of the variable.
		 *
		 *	\return Column dimension of the variable
		 */
		virtual uint getColDim( ) const;

		/** Returns total index of given component within memory.
		 *
		 *	@param[in] rowIdx		Row index of the component.
		 *	@param[in] colIdx		Column index of the component.
		 *
		 *	\return Total index of given component
		 */
		virtual ExportIndex	getTotalIdx(	const ExportIndex& rowIdx,
											const ExportIndex& colIdx
											) const;

		/** Assigns offsets and dimensions of a sub-matrix. This function is used to
		 *	access only a sub-matrix of the variable without copying its values to
		 *	a new variable.
		 *
		 *	@param[in] _rowOffset		Variable index of first row of sub-matrix.
		 *	@param[in] _colOffset		Variable index of first column of sub-matrix.
		 *	@param[in] _rowDim			Row dimension of variable (as only the submatrix data is stored).
		 *	@param[in] _colDim			Column dimension of variable (as only the submatrix data is stored).
		 *	@param[in] _nRows			Number of rows of sub-matrix.
		 *	@param[in] _nCols			Number of columns of sub-matrix.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS
		 */
		returnValue setSubmatrixOffsets(	const ExportIndex& _rowOffset,
											const ExportIndex& _colOffset,
											unsigned _rowDim = 0,
											unsigned _colDim = 0,
											unsigned _nRows = 0,
											unsigned _nCols = 0
											);


		/** Returns whether given component is set to given value.
		 *
		 *	@param[in] rowIdx		Variable row index of the component.
		 *	@param[in] colIdx		Variable column index of the component.
		 *	@param[in] _value		Value used for comparison.
		 *
		 *	\return true  iff given component is set to given value, \n
		 *	        false otherwise
		 */
		bool hasValue(	const ExportIndex& _rowIdx,
						const ExportIndex& _colIdx,
						double _value
						) const;

	protected:

		bool doAccessTransposed;				/**< Flag indicating whether variable is to be accessed in a transposed manner. */

		ExportIndex rowOffset;						/**< Index of first row of a possible sub-matrix of the variable. */
		ExportIndex colOffset;						/**< Index of first column of a possible sub-matrix of the variable. */
		unsigned rowDim;							/**< Row dimension of variable (as only the submatrix data is stored). */
		unsigned colDim;							/**< Column dimension of variable (as only the submatrix data is stored). */
		unsigned nRows;								/**< Number of rows of a possible sub-matrix of the variable. */
		unsigned nCols;								/**< Number of columns of a possible sub-matrix of the variable. */
};

CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_EXPORT_VARIABLE_INTERNAL_HPP

// end of file.
