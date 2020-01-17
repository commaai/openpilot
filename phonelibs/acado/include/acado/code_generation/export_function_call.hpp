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
 *    \file include/acado/code_generation/export_function_call.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 2010-2011
 */


#ifndef ACADO_TOOLKIT_EXPORT_FUNCTION_CALL_HPP
#define ACADO_TOOLKIT_EXPORT_FUNCTION_CALL_HPP

#include <acado/utils/acado_utils.hpp>

#include <acado/code_generation/export_statement.hpp>
#include <acado/code_generation/export_function.hpp>


BEGIN_NAMESPACE_ACADO



/** 
 *	\brief Allows to export code of a function call.
 *
 *	\ingroup AuxiliaryFunctionality
 *
 *	The class ExportFunctionCall allows to export code of a function call.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class ExportFunctionCall : public ExportStatement
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

		/** Default constructor which optionally takes the name of the function
		 *	to be called as well as possible calling arguments.
		 *
		 *	@param[in] _name		Name of the function to be called.
		 *	@param[in] _argument1	Calling argument no. 1.
		 *	@param[in] _argument2	Calling argument no. 2.
		 *	@param[in] _argument3	Calling argument no. 3.
		 *	@param[in] _argument4	Calling argument no. 4.
		 *	@param[in] _argument5	Calling argument no. 5.
		 *	@param[in] _argument6	Calling argument no. 6.
		 *	@param[in] _argument7	Calling argument no. 7.
		 *	@param[in] _argument8	Calling argument no. 8.
		 *	@param[in] _argument9	Calling argument no. 9.
		 */
		ExportFunctionCall(	const std::string& _name = "acadoFcn",
							const ExportArgument& _argument1 = emptyConstExportArgument,
							const ExportArgument& _argument2 = emptyConstExportArgument,
							const ExportArgument& _argument3 = emptyConstExportArgument,
							const ExportArgument& _argument4 = emptyConstExportArgument,
							const ExportArgument& _argument5 = emptyConstExportArgument,
							const ExportArgument& _argument6 = emptyConstExportArgument,
							const ExportArgument& _argument7 = emptyConstExportArgument,
							const ExportArgument& _argument8 = emptyConstExportArgument,
							const ExportArgument& _argument9 = emptyConstExportArgument
							);

		/** Constructor which optionally takes the function to be called
		 *	as well as possible calling arguments.
		 *
		 *	@param[in] _f			Function to be called.
		 *	@param[in] _argument1	Calling argument no. 1.
		 *	@param[in] _argument2	Calling argument no. 2.
		 *	@param[in] _argument3	Calling argument no. 3.
		 *	@param[in] _argument4	Calling argument no. 4.
		 *	@param[in] _argument5	Calling argument no. 5.
		 *	@param[in] _argument6	Calling argument no. 6.
		 *	@param[in] _argument7	Calling argument no. 7.
		 *	@param[in] _argument8	Calling argument no. 8.
		 *	@param[in] _argument9	Calling argument no. 9.
		 */
		ExportFunctionCall(	const ExportFunction& _f,
							const ExportArgument& _argument1 = emptyConstExportArgument,
							const ExportArgument& _argument2 = emptyConstExportArgument,
							const ExportArgument& _argument3 = emptyConstExportArgument,
							const ExportArgument& _argument4 = emptyConstExportArgument,
							const ExportArgument& _argument5 = emptyConstExportArgument,
							const ExportArgument& _argument6 = emptyConstExportArgument,
							const ExportArgument& _argument7 = emptyConstExportArgument,
							const ExportArgument& _argument8 = emptyConstExportArgument,
							const ExportArgument& _argument9 = emptyConstExportArgument
							);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
		ExportFunctionCall(	const ExportFunctionCall& arg
							);

        /** Destructor. 
		 */
		virtual ~ExportFunctionCall( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
		ExportFunctionCall& operator=(	const ExportFunctionCall& arg
										);

		/** Clone constructor (deep copy).
		 *
		 *	\return Pointer to cloned object.
		 */
		virtual ExportStatement* clone( ) const;


		/** Initializes function call with given name of the function and possible calling arguments.
		 *
		 *	@param[in] _name			Name of the function to be called.
		 *	@param[in] _argument1	Calling argument no. 1.
		 *	@param[in] _argument2	Calling argument no. 2.
		 *	@param[in] _argument3	Calling argument no. 3.
		 *	@param[in] _argument4	Calling argument no. 4.
		 *	@param[in] _argument5	Calling argument no. 5.
		 *	@param[in] _argument6	Calling argument no. 6.
		 *	@param[in] _argument7	Calling argument no. 7.
		 *	@param[in] _argument8	Calling argument no. 8.
		 *	@param[in] _argument9	Calling argument no. 9.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue init(	const std::string& _name = "defaultFunctionName",
							const ExportArgument& _argument1 = emptyConstExportArgument,
							const ExportArgument& _argument2 = emptyConstExportArgument,
							const ExportArgument& _argument3 = emptyConstExportArgument,
							const ExportArgument& _argument4 = emptyConstExportArgument,
							const ExportArgument& _argument5 = emptyConstExportArgument,
							const ExportArgument& _argument6 = emptyConstExportArgument,
							const ExportArgument& _argument7 = emptyConstExportArgument,
							const ExportArgument& _argument8 = emptyConstExportArgument,
							const ExportArgument& _argument9 = emptyConstExportArgument
							);

		/** Initializes function call with function to be called and possible calling arguments.
		 *
		 *	@param[in] _f			Function to be called.
		 *	@param[in] _argument1	Calling argument no. 1.
		 *	@param[in] _argument2	Calling argument no. 2.
		 *	@param[in] _argument3	Calling argument no. 3.
		 *	@param[in] _argument4	Calling argument no. 4.
		 *	@param[in] _argument5	Calling argument no. 5.
		 *	@param[in] _argument6	Calling argument no. 6.
		 *	@param[in] _argument7	Calling argument no. 7.
		 *	@param[in] _argument8	Calling argument no. 8.
		 *	@param[in] _argument9	Calling argument no. 9.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_CALL_TO_EXPORTED_FUNCTION
		 */
		returnValue init(	const ExportFunction& _f,
							const ExportArgument& _argument1 = emptyConstExportArgument,
							const ExportArgument& _argument2 = emptyConstExportArgument,
							const ExportArgument& _argument3 = emptyConstExportArgument,
							const ExportArgument& _argument4 = emptyConstExportArgument,
							const ExportArgument& _argument5 = emptyConstExportArgument,
							const ExportArgument& _argument6 = emptyConstExportArgument,
							const ExportArgument& _argument7 = emptyConstExportArgument,
							const ExportArgument& _argument8 = emptyConstExportArgument,
							const ExportArgument& _argument9 = emptyConstExportArgument
							);


		/** Exports source code of the function call into given file. Its appearance can 
		 *  can be adjusted by various options.
		 *
		 *	@param[in] stream			Name of file to be used to export function call.
		 *	@param[in] _realString		std::string to be used to declare real variables.
		 *	@param[in] _intString		std::string to be used to declare integer variables.
		 *	@param[in] _precision		Number of digits to be used for exporting real values.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_MEMBER_NOT_INITIALISED
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

		/** Frees internal dynamic memory to yield an empty function call.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue clear( );


		/** Sets the name of the function to be called.
		 *
		 *	@param[in] _name		New name of the function to be called.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue	setName(	const std::string& _name
								);


    protected:

		std::string name;							/**< Name of function to be called. */
		ExportArgumentList functionArguments;		/**< List of calling arguments. */
};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_EXPORT_STATEMENT_HPP

// end of file.
