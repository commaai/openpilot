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
 *    \file include/acado/code_generation/export_module.hpp
 *    \author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 *    \date 2010 - 2014
 */


#ifndef ACADO_TOOLKIT_EXPORT_MODULE_HPP
#define ACADO_TOOLKIT_EXPORT_MODULE_HPP

#include <acado/user_interaction/user_interaction.hpp>

BEGIN_NAMESPACE_ACADO

class ExportStatementBlock;

/** 
 *	\brief User-interface to automatically generate algorithms for fast model predictive control
 *
 *	\ingroup UserInterfaces
 *
 *  The class ExportModule is a user-interface to automatically generate tailored
 *  algorithms for fast model predictive control. It takes an optimal control 
 *  problem (OCP) formulation and generates code based on given user options.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class ExportModule : public UserInteraction
{
//
// PUBLIC MEMBER FUNCTIONS:
//
public:

	/** Default constructor. */
	ExportModule( );

	/** Destructor.
	 */
	virtual ~ExportModule( );

	/** Exports all files of the auto-generated code into the given directory.
	 *
	 *	@param[in] dirName			Name of directory to be used to export files.
	 *	@param[in] _realString		std::string to be used to declare real variables.
	 *	@param[in] _intString		std::string to be used to declare integer variables.
	 *	@param[in] _precision		Number of digits to be used for exporting real values.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue exportCode(	const std::string& dirName,
									const std::string& _realString = "real_t",
									const std::string& _intString = "int",
									int _precision = 16
									) = 0;

protected:

	/** Exports main header file for using the exported algorithm.
	 *
	 *	@param[in] _dirName			Name of directory to be used to export file.
	 *	@param[in] _fileName		Name of file to be exported.
	 *	@param[in] _realString		std::string to be used to declare real variables.
	 *	@param[in] _intString		std::string to be used to declare integer variables.
	 *	@param[in] _precision		Number of digits to be used for exporting real values.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue exportAcadoHeader(	const std::string& _dirName,
											const std::string& _fileName,
											const std::string& _realString = "real_t",
											const std::string& _intString = "int",
											int _precision = 16
											) const = 0;


	/** Collects all data declarations of the auto-generated sub-modules to given
	 *	list of declarations.
	 *
	 *	@param[in] declarations		List of declarations.
	 *
	 *	\return SUCCESSFUL_RETURN, \n
	 *	        RET_UNABLE_TO_EXPORT_CODE
	 */
	virtual returnValue collectDataDeclarations(	ExportStatementBlock& declarations,
													ExportStruct dataStruct = ACADO_ANY
													) const = 0;


	/** Collects all function (forward) declarations of the auto-generated sub-modules
	 *	to given list of declarations.
	 *
	 *	@param[in] declarations		List of declarations.
	 *
	 *	\return SUCCESSFUL_RETURN, \n
	 *	        RET_UNABLE_TO_EXPORT_CODE
	 */
	virtual returnValue collectFunctionDeclarations(	ExportStatementBlock& declarations
														) const = 0;

	/** Sets-up default options.
	 *
	 *  \return SUCCESSFUL_RETURN
	 */
	returnValue setupOptions( );

	/** Name of common header file. */
	std::string commonHeaderName;
};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_EXPORT_MODULE_HPP
