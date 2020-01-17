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
 *    \file include/acado/code_generation/export_argument.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_EXPORT_ARGUMENT_HPP
#define ACADO_TOOLKIT_EXPORT_ARGUMENT_HPP

#include <acado/utils/acado_utils.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/code_generation/export_data.hpp>
#include <acado/code_generation/export_index.hpp>

BEGIN_NAMESPACE_ACADO


class ExportArgumentInternal;


/** 
 *	\brief Defines a matrix-valued variable that can be passed as argument to exported functions.
 *
 *	\ingroup AuxiliaryFunctionality
 *
 *	The class ExportArgument defines a matrix-valued variable that 
 *	can be passed as argument to exported functions. By default, all entries
 *	of an arguments are undefined, but each of its component can be set to 
 *	a fixed value if known beforehand.
 *
 *	\author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */

class ExportArgument : public ExportData
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

		/** Default constructor. 
		 */
        ExportArgument( );

		/** Constructor which takes the name, type
		 *	and dimensions of the argument.
		 *
		 *	@param[in] _name			Name of the argument.
		 *	@param[in] _nRows			Number of rows of the argument.
		 *	@param[in] _nCols			Number of columns of the argument.
		 *	@param[in] _type			Data type of the argument.
		 *	@param[in] _dataStruct		Global data struct to which the argument belongs to (if any).
		 *	@param[in] _callByValue		Flag indicating whether argument it to be called by value.
		 *	@param[in] _addressIdx		If an address index is specified, not the argument itself but 
		 *								a pointer to this address within the memory of the argument is passed.
		 */
		ExportArgument(	const std::string& _name,
						uint _nRows = 1,
						uint _nCols = 1,
						ExportType _type = REAL,
						ExportStruct _dataStruct = ACADO_LOCAL,
						bool _callByValue = false,
						const ExportIndex& _addressIdx = emptyConstExportIndex,
						const std::string& _prefix = std::string()
						);

		/** Constructor which takes the name and type of the argument.
		 *	Moreover, it initializes the argument with the dimensions and the 
		 *	values of the given matrix.
		 *
		 *	@param[in] _name			Name of the argument.
		 *	@param[in] _data			DMatrix used for initialization.
		 *	@param[in] _type			Data type of the argument.
		 *	@param[in] _dataStruct		Global data struct to which the argument belongs to (if any).
		 *	@param[in] _callByValue		Flag indicating whether argument it to be called by value.
		 *	@param[in] _addressIdx		If an address index is specified, not the argument itself but 
		 *								a pointer to this address within the memory of the argument is passed.
		 */
		ExportArgument(	const std::string& _name,
						const DMatrixPtr& _data,
						ExportType _type = REAL,
						ExportStruct _dataStruct = ACADO_LOCAL,
						bool _callByValue = false,
						const ExportIndex& _addressIdx = emptyConstExportIndex,
						const std::string& _prefix = std::string()
						);

		ExportArgument(	const DMatrix& _data
						);

		ExportArgumentInternal* operator->();

		const ExportArgumentInternal* operator->() const;

		/** Returns a copy of the argument with address index set to given location.
		 *
		 *	@param[in] rowIdx		Row index of the adress.
		 *	@param[in] colIdx		Column index of the adress.
		 *
		 *	\return Copy of the argument with address index set to given location
		 */
		ExportArgument getAddress(	const ExportIndex& _rowIdx,
									const ExportIndex& _colIdx = emptyConstExportIndex
									) const;

		/** Returns a string containing the address of the argument to be called. 
		 *	If an address index has been set, the string contains a pointer to the 
		 *	desired location. The string also depends on whether the argument is 
		 *	to be called by value or not.
		 *
		 *	\return String containing the address of the argument
		 */
		const std::string getAddressString(	bool withDataStruct = true
											) const;


		/** Returns number of rows of the argument.
		 *
		 *	\return Number of rows of the argument
		 */
		virtual uint getNumRows( ) const;

		/** Returns number of columns of the argument.
		 *
		 *	\return Number of columns of the argument
		 */
		virtual uint getNumCols( ) const;

		/** Returns total dimension of the argument.
		 *
		 *	\return Total dimension of the argument
		 */
		virtual uint getDim( ) const;


		/** Returns whether all components of the argument are given.
		 *
		 *	\return true  iff all components of the argument have given values, \n
		 *	        false otherwise
		 */
		virtual bool isGiven( ) const;

		/** Returns whether argument is to be called by value.
		 *
		 *	\return true  iff argument is to be called by value, \n
		 *	        false otherwise
		 */
		bool isCalledByValue( ) const;

		/** Specifies to call argument by value.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue callByValue( );



		/** Exports declaration of the argument into given file. Its appearance can 
		 *  can be adjusted by various options.
		 *
		 *	@param[in] file				Name of file to be used to export declaration.
		 *	@param[in] _realString		std::string to be used to declare real variables.
		 *	@param[in] _intString		std::string to be used to declare integer variables.
		 *	@param[in] _precision		Number of digits to be used for exporting real values.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue exportDataDeclaration(	std::ostream& stream,
													const std::string& _realString = "real_t",
													const std::string& _intString = "int",
													int _precision = 16
													) const;
};


static const ExportArgument emptyConstExportArgument;


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_EXPORT_ARGUMENT_HPP

// end of file.
