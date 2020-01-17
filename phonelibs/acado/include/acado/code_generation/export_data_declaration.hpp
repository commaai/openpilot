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
 *    \file include/acado/code_generation/export_data_declaration.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 2010-2011
 */


#ifndef ACADO_TOOLKIT_EXPORT_DATA_DECLARATION_HPP
#define ACADO_TOOLKIT_EXPORT_DATA_DECLARATION_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/code_generation/export_statement.hpp>
#include <acado/code_generation/export_variable.hpp>
#include <acado/code_generation/export_index.hpp>



BEGIN_NAMESPACE_ACADO



/** 
 *	\brief Allows to export code containing variable declarations.
 *
 *	\ingroup AuxiliaryFunctionality
 *
 *	The class ExportDataDeclaration allows to export code containing variable 
 *	declarations of different data types.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class ExportDataDeclaration : public ExportStatement
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

		/** Default constructor. 
		 */
        ExportDataDeclaration( );
		
		/** Constructor taking the variable to be declared.
		 *
		 *	@param[in] _data	Variable to be declared.
		 */
		ExportDataDeclaration(	const ExportVariable& _data
								);

		/** Constructor taking the index variable to be declared.
		 *
		 *	@param[in] _data	Index Variable to be declared.
		 */
		ExportDataDeclaration(	const ExportIndex& _data
								);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
		ExportDataDeclaration(	const ExportDataDeclaration& arg
								);

        /** Destructor. 
		 */
        virtual ~ExportDataDeclaration( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
        ExportDataDeclaration& operator=(	const ExportDataDeclaration& arg
											);

		/** Clone constructor (deep copy).
		 *
		 *	\return Pointer to cloned object.
		 */
		virtual ExportStatement* clone( ) const;


		/** Exports source code of the statement into given file. Its appearance can 
		 *  can be adjusted by various options.
		 *
		 *	@param[in] file				Name of file to be used to export statement.
		 *	@param[in] _realString		std::string to be used to declare real variables.
		 *	@param[in] _intString		std::string to be used to declare integer variables.
		 *	@param[in] _precision		Number of digits to be used for exporting real values.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue exportCode(	std::ostream& stream,
										const std::string& _realString = "real_t",
										const std::string& _intString = "int",
										int _precision = 16
										) const;


	//
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:



    protected:
		
		ExportData data;					/**< Variable to be declared. */

};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_EXPORT_DATA_DECLARATION_HPP

// end of file.
