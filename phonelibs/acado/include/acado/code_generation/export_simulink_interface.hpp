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
 *    \file include/acado/code_generation/export_simulink_interface.hpp
 *    \author Milan Vukov
 *    \date 2013 - 2014
 */

#ifndef ACADO_TOOLKIT_EXPORT_SIMULINK_INTERFACE_HPP
#define ACADO_TOOLKIT_EXPORT_SIMULINK_INTERFACE_HPP

#include <acado/code_generation/export_templated_file.hpp>

BEGIN_NAMESPACE_ACADO

/**
 *	\brief A class for generating the glue code and makefile
 *	       for interfacing generated code and Simulink.
 *
 *	\ingroup AuxiliaryFunctionality
 *
 *	\author Milan Vukov
 */
class ExportSimulinkInterface
{
public:

	/** Default constructor.
	 *
	 *	@param[in] _moduleName		    Module name for customization.
     *	@param[in] _modulePrefix	    Module prefix for customization.
	 *	@param[in] _commonHeaderName	Name of common header file to be included.
	 *	@param[in] _realString			std::string to be used to declare real variables.
	 *	@param[in] _intString			std::string to be used to declare integer variables.
	 *	@param[in] _precision			Number of digits to be used for exporting real values.
	 *	@param[in] _commentString		std::string to be used for exporting comments.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	ExportSimulinkInterface(	const std::string& _makefileName,
								const std::string& _wrapperHeaderFileName,
								const std::string& _wrapperSourceFileName,
								const std::string& _moduleName = "acado",
                                const std::string& _modulePrefix = "ACADO",
								const std::string& _commonHeaderName = "",
								const std::string& _realString = "double",
								const std::string& _intString = "int",
								int _precision = 16,
								const std::string& _commentString = std::string()
								);

	/** Destructor. */
	virtual ~ExportSimulinkInterface()
	{}

	/** Configure the template
	 *
	 *  \return SUCCESSFUL_RETURN
	 */
	returnValue configure(	unsigned N,
							unsigned NX,
							unsigned NDX,
							unsigned NXA,
							unsigned NU,
							unsigned NP,
							unsigned NY,
							unsigned NYN,
							bool _initialStateFixed,
							unsigned _wMatrixType,
							bool _hardcodedConstraints,
							bool _useArrivalCost,
							bool _compCovMatrix,
							std::string _qpSolver
							);

	/** Export the interface. */
	returnValue exportCode();

private:

	ExportTemplatedFile makefile;
	ExportTemplatedFile wrapperSource;
	ExportTemplatedFile wrapperHeader;
	std::string moduleName;
    std::string modulePrefix;
};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_EXPORT_SIMULINK_INTERFACE_HPP
