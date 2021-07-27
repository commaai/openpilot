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
 *    \authors Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */

#ifndef ACADO_TOOLKIT_EXPORT_VARIABLE_HPP
#define ACADO_TOOLKIT_EXPORT_VARIABLE_HPP

#include <acado/utils/acado_utils.hpp>
#include <acado/code_generation/export_argument.hpp>
#include <acado/code_generation/export_index.hpp>

BEGIN_NAMESPACE_ACADO

class ExportArithmeticStatement;
class ExportVariableInternal;

/** 
 *	\brief Defines a matrix-valued variable to be used for exporting code.
 *
 *	\ingroup UserDataStructures
 *
 *	The class ExportVariable defines a matrix-valued variable to be used for exporting
 *	code. Instances of this class can be used similar to usual DMatrix objects
 *	but offer additional functionality, e.g. they allow to export arithmetic 
 *	expressions and they can be passed as argument to exported functions. By 
 *	default, all entries of a ExportVariable are undefined, but each of its 
 *	component can be set to a fixed value if known beforehand.
 *
 *	\authors Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */

class ExportVariable : public ExportArgument
{
	//
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

		/** Default constructor. */
		ExportVariable();

		/** Constructor which takes the name, type string
		 *	and dimensions of the variable.
		 *
		 *	@param[in] _name			Name of the argument.
		 *	@param[in] _nRows			Number of rows of the argument.
		 *	@param[in] _nCols			Number of columns of the argument.
		 *	@param[in] _type			Data type of the argument.
		 *	@param[in] _dataStruct		Global data struct to which the argument belongs to (if any).
		 *	@param[in] _callByValue		Flag indicating whether argument it to be called by value.
		 */
		ExportVariable(	const std::string& _name,
						uint _nRows,
						uint _nCols,
						ExportType _type = REAL,
						ExportStruct _dataStruct = ACADO_LOCAL,
						bool _callItByValue = false,
						const std::string& _prefix = std::string()
						);

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
		ExportVariable(	const std::string& _name,
						const DMatrix& _data,
						ExportType _type = REAL,
						ExportStruct _dataStruct = ACADO_LOCAL,
						bool _callItByValue = false,
						const std::string& _prefix = std::string(),
						bool _isGiven = true
						);

		/** Constructor which takes the name and type string of the variable.
		 *	Moreover, it initializes the variable with the dimensions and the
		 *	values of the given matrix.
		 *
		 *	@param[in] _name			Name of the argument.
		 *	@param[in] _data			Shared pointer to DMatrix used for initialization.
		 *	@param[in] _type			Data type of the argument.
		 *	@param[in] _dataStruct		Global data struct to which the argument belongs to (if any).
		 *	@param[in] _callByValue		Flag indicating whether argument it to be called by value.
		 */
		ExportVariable(	const std::string& _name,
						const DMatrixPtr& _data,
						ExportType _type = REAL,
						ExportStruct _dataStruct = ACADO_LOCAL,
						bool _callItByValue = false,
						const std::string& _prefix = std::string()
						);

		/** Constructor which takes the name and type string of the variable.
		 *	Moreover, it initializes the variable with the dimensions of the matrix.
		 *
		 *	@param[in] _nRows			Name of the argument.
		 *	@param[in] _nCols			Name of the argument.
		 *	@param[in] _type			Data type of the argument.
		 *	@param[in] _dataStruct		Global data struct to which the argument belongs to (if any).
		 *	@param[in] _callByValue		Flag indicating whether argument it to be called by value.
		 */
		ExportVariable(	unsigned _nRows,
						unsigned _nCols,
						ExportType _type = REAL,
						ExportStruct _dataStruct = ACADO_LOCAL,
						bool _callItByValue = false,
						const std::string& _prefix = std::string()
						);

		/** \name Constructor which converts a given matrix/vector/scalar into an ExportVariable.
		  * @{ */

		template<typename Derived>
		ExportVariable(	const Eigen::MatrixBase<Derived>& _data
						)
		{
			simpleForward(DMatrix( _data ));
		}

		ExportVariable(	const double _data	/**< Scalar used for initialization */
						);
		/** @} */

        /** Destructor.
		 */
		virtual ~ExportVariable( );

		ExportVariable clone() const;

		ExportVariableInternal* operator->();

		const ExportVariableInternal* operator->() const;

		/** Initializes variable with given name, type string
		 *	and dimensions of the variable.
		 *
		 *	@param[in] _name			Name of the argument.
		 *	@param[in] _nRows			Number of rows of the argument.
		 *	@param[in] _nCols			Number of columns of the argument.
		 *	@param[in] _type			Data type of the argument.
		 *	@param[in] _dataStruct		Global data struct to which the argument belongs to (if any).
		 *	@param[in] _callByValue		Flag indicating whether argument it to be called by value.
		 *
		 *	\return Reference to initialized object
		 */
		ExportVariable& setup(	const std::string& _name,
								uint _nRows = 1,
								uint _nCols = 1,
								ExportType _type = REAL,
								ExportStruct _dataStruct = ACADO_LOCAL,
								bool _callItByValue = false,
								const std::string& _prefix = std::string()
								);

		/** Initializes variable with given name and type string of the variable.
		 *	Moreover, the variable is initialized with the dimensions and the 
		 *	values of the given matrix.
		 *
		 *	@param[in] _name			Name of the argument.
		 *	@param[in] _data			DMatrix used for initialization.
		 *	@param[in] _type			Data type of the argument.
		 *	@param[in] _dataStruct		Global data struct to which the argument belongs to (if any).
		 *	@param[in] _callByValue		Flag indicating whether argument it to be called by value.
		 *
		 *	\return Reference to initialized object
		 */
		ExportVariable& setup(	const std::string& _name,
								const DMatrix& _data,
								ExportType _type = REAL,
								ExportStruct _dataStruct = ACADO_LOCAL,
								bool _callItByValue = false,
								const std::string& _prefix = std::string(),
								bool _isGiven = true
								);

		/** Returns value of given component.
		 *
		 *	@param[in] rowIdx		Row index of the component to be returned.
		 *	@param[in] colIdx		Column index of the component to be returned.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		double operator()(	uint rowIdx,
							uint colIdx
							) const;

		/** Returns value of given component.
		 *
		 *	@param[in] totalIdx		Memory location of the component to be returned.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		double operator()(	uint totalIdx
							) const;

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

		/** Returns whether all components of the variable are set to a given value.
		 *
		 *	\return true  iff all components of the variable are set to a given value, \n
		 *	        false otherwise
		 */
		bool isGiven( ) const;


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


		/** Operator for adding two ExportVariables.
		 *
		 *	@param[in] arg		Variable to be added.
		 *
		 *	\return Arithmetic statement containing the addition
		 */
		friend ExportArithmeticStatement operator+(	const ExportVariable& arg1,
													const ExportVariable& arg2
													);

		/** Operator for subtracting an ExportVariable from another.
		 *
		 *	@param[in] arg		Variable to be subtracted.
		 *
		 *	\return Arithmetic statement containing the subtraction
		 */
		friend ExportArithmeticStatement operator-(	const ExportVariable& arg1,
													const ExportVariable& arg2
													);

		/** Operator for add-assigning an ExportVariable to another.
		 *
		 *	@param[in] arg		Variable to be add-assigned.
		 *
		 *	\return Arithmetic statement containing the add-assignment
		 */
		friend ExportArithmeticStatement operator+=(	const ExportVariable& arg1,
														const ExportVariable& arg2
														);

		/** Operator for subtract-assigning an ExportVariables from another.
		 *
		 *	@param[in] arg		Variable to be subtract-assigned.
		 *
		 *	\return Arithmetic statement containing the subtract-assignment
		 */
		friend ExportArithmeticStatement operator-=(	const ExportVariable& arg1,
														const ExportVariable& arg2
														);

		/** Operator for multiplying two ExportVariables.
		 *
		 *	@param[in] arg		Variable to be multiplied from the right.
		 *
		 *	\return Arithmetic statement containing the multiplication
		 */
		friend ExportArithmeticStatement operator*(		const ExportVariable& arg1,
														const ExportVariable& arg2
														);

		/** Operator for multiplying an ExportVariable to the transposed on another.
		 *
		 *	@param[in] arg		Variable to be multiplied from the right.
		 *
		 *	\return Arithmetic statement containing the multiplication with left-hand side variable transposed
		 */
		friend ExportArithmeticStatement operator^(	const ExportVariable& arg1,
													const ExportVariable& arg2
													);

		/** Operator for assigning an ExportVariable to another.
		 *
		 *	@param[in] arg		Variable to be assined.
		 *
		 *	\return Arithmetic statement containing the assignment
		 */
		friend ExportArithmeticStatement operator==(	const ExportVariable& arg1,
														const ExportVariable& arg2
														);

		/** Operator for assigning an arithmetic statement to an ExportVariable.
		 *
		 *	@param[in] arg		Arithmetic statement to be assigned.
		 *
		 *	\return Arithmetic statement containing the assignment
		 */
		ExportArithmeticStatement operator==(	ExportArithmeticStatement arg
												) const;

		/** Operator for adding an arithmetic statement to an ExportVariable.
		 *
		 *	@param[in] arg		Arithmetic statement to be added.
		 *
		 *	\return Arithmetic statement containing the addition
		 */
		ExportArithmeticStatement operator+(	ExportArithmeticStatement arg
												) const;

		/** Operator for subtraction an arithmetic statement from an ExportVariable.
		 *
		 *	@param[in] arg		Arithmetic statement to be subtracted.
		 *
		 *	\return Arithmetic statement containing the subtraction
		 */
		ExportArithmeticStatement operator-(	ExportArithmeticStatement arg
												) const;

		/** Operator for add-assigning an arithmetic statement to an ExportVariable.
		 *
		 *	@param[in] arg		Arithmetic statement to be add-assigned.
		 *
		 *	\return Arithmetic statement containing the add-assignment
		 */
		ExportArithmeticStatement operator+=(	ExportArithmeticStatement arg
												) const;

		/** Operator for subtract-assigning an arithmetic statement from an ExportVariable.
		 *
		 *	@param[in] arg		Arithmetic statement to be subtract-assigned.
		 *
		 *	\return Arithmetic statement containing the subtract-assignment
		 */
		ExportArithmeticStatement operator-=(	ExportArithmeticStatement arg
												) const;

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
		ExportVariable getSubMatrix(	const ExportIndex& rowIdx1,
										const ExportIndex& rowIdx2,
										const ExportIndex& colIdx1,
										const ExportIndex& colIdx2
										) const;

		/** Returns element at position (rowIdx, colIdx).
		 *
		 *	@param[in] rowIdx		Variable row index of the component.
		 *	@param[in] colIdx		Variable column index of the component.
		 *
		 *	\return Element at position (rowIdx, colIdx)
		 */
		ExportVariable getElement(	const ExportIndex& rowIdx,
									const ExportIndex& colIdx
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

		/** Check whether the matrix is actually a submatrix. */
		bool isSubMatrix() const;

		/** Check whether the matrix is diagonal. */
		bool isDiagonal() const;

		/** Prints contents of variable to screen.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue print( ) const;

    private:
		void simpleForward(const DMatrix& _value);
};

static const ExportVariable emptyConstExportVariable;

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_EXPORT_VARIABLE_HPP
