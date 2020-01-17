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
*	\file include/acado/code_generation/export_argument_list.hpp
*	\author Hans Joachim Ferreau, Boris Houska
*/



#ifndef ACADO_TOOLKIT_EXPORT_ARGUMENT_LIST_HPP
#define ACADO_TOOLKIT_EXPORT_ARGUMENT_LIST_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/code_generation/export_argument.hpp>

BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Allows to store a list of calling arguments of an ExportFunction.
 *
 *	\ingroup AuxiliaryFunctionality
 *
 *	The class ExportArgumentList allows to store a list of calling 
 *	arguments of an ExportFunction.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class ExportArgumentList
{
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/**< Default Constructor. 
		 */
		ExportArgumentList( );

		/** Constructor which takes up to nine calling arguments.
		 *
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
		ExportArgumentList(	const ExportArgument& _argument1,
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
		 *	@param[in] arg	Right-hand side object.
		 */
		ExportArgumentList(	const ExportArgumentList& arg
							);

		/** Destructor.
		 */
		virtual ~ExportArgumentList( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] arg	Right-hand side object.
		 */
		ExportArgumentList& operator=(	const ExportArgumentList& rhs
										);


		/** Adds up to nine calling arguments to the list.
		 *
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
		returnValue addArgument(	const ExportArgument& _argument1,
									const ExportArgument& _argument2 = emptyConstExportArgument,
									const ExportArgument& _argument3 = emptyConstExportArgument,
									const ExportArgument& _argument4 = emptyConstExportArgument,
									const ExportArgument& _argument5 = emptyConstExportArgument,
									const ExportArgument& _argument6 = emptyConstExportArgument,
									const ExportArgument& _argument7 = emptyConstExportArgument,
									const ExportArgument& _argument8 = emptyConstExportArgument,
									const ExportArgument& _argument9 = emptyConstExportArgument
									);


		/** Return number of calling arguments in list.
		 *
		 *  \return Number of calling arguments
		 */
		uint getNumArguments( ) const;


		/** Exports a code snippet containing all calling arguments of the list.
		 *  Its appearance can can be adjusted by various options.
		 *
		 *	@param[in] stream			Name of file to be used to export function.
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


		/** Removes all calling arguments to yield an empty argument list.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue clear( );
		
		
		/** Specifies to include variable types into calling arguments.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue doIncludeType( );
		
		
		/** Specifies not to include variable types into calling arguments.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue doNotIncludeType( );

		/** Get the list of arguments.
		 *
		 *	\return Argument list
		 */
		const std::vector< ExportArgument >& get( ) const;

	//
	// PROTECTED MEMBER FUNCTIONS:
	//
	protected:

		/** Adds a single calling arguments to the list.
		 *
		 *	@param[in] _argument	Calling argument.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue addSingleArgument(	const ExportArgument& _argument
										);


	//
	// DATA MEMBERS:
	//
	protected:

		/** Array containing all calling arguments. */
		std::vector< ExportArgument > arguments;
		
		/** Flag indicating whether variable types are to be included in calling arguments. */
		bool includeType;
};


CLOSE_NAMESPACE_ACADO



#endif	// ACADO_TOOLKIT_EXPORT_ARGUMENT_LIST_HPP


/*
 *	end of file
 */
