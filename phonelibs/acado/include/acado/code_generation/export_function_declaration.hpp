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
 *    \file include/acado/code_generation/export_function_declaration.hpp
 *    \author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 *    \date 2010 - 2013
 */


#ifndef ACADO_TOOLKIT_EXPORT_FUNCTION_DECLARATION_HPP
#define ACADO_TOOLKIT_EXPORT_FUNCTION_DECLARATION_HPP

#include <acado/code_generation/export_statement.hpp>
#include <acado/code_generation/export_function.hpp>
#include <acado/code_generation/export_acado_function.hpp>

BEGIN_NAMESPACE_ACADO

/** 
*	\brief Allows to export code containing function (forward) declarations.
 *
 *	\ingroup AuxiliaryFunctionality
 *
 *	The class ExportDataDeclaration allows to export code containing function 
 *	(forward) declarations.
 *
 *	\author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */
class ExportFunctionDeclaration : public ExportStatement
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

		/** Constructor taking the function to be declared.
		 *
		 *	@param[in] _f		Function to be declared.
		 */
		ExportFunctionDeclaration(	const ExportFunction& _f
									);

		/** Constructor taking the ODE function to be declared.
		 *
		 *	@param[in] _f		ODE function to be declared.
		 */
		ExportFunctionDeclaration(	const ExportAcadoFunction& _f
									);

        /** Destructor. 
		 */
        virtual ~ExportFunctionDeclaration( );

		/** Clone constructor (deep copy).
		 *
		 *	\return Pointer to cloned object.
		 */
		virtual ExportStatement* clone( ) const;

		/** Exports source code of the statement into given file. Its appearance can 
		 *  can be adjusted by various options.
		 *
		 *	@param[in] stream			Name of file to be used to export statement.
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

    private:
		ExportFunctionDeclaration( );

		const ExportFunction& f;
};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_EXPORT_FUNCTION_DECLARATION_HPP
