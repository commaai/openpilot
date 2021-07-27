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
 *    \file include/acado/code_generation/export_qpoases3_interface.hpp
 *    \author Milan Vukov, Joachim Ferreau
 *    \date 2012 - 2015
 */

#ifndef ACADO_TOOLKIT_EXPORT_QPOASES3_INTERFACE_HPP
#define ACADO_TOOLKIT_EXPORT_QPOASES3_INTERFACE_HPP

#include <acado/code_generation/export_templated_file.hpp>
#include <acado/code_generation/export_qpoases_interface.hpp>

BEGIN_NAMESPACE_ACADO

/**
 *	\brief A class for generating the glue code for interfacing qpOASES.
 *
 *	\ingroup AuxiliaryFunctionality
 *
 *	\author Milan Vukov
 *
 *	\note Based on code originally developed by Hans Joachim Ferreau and Boris Houska.
 */
class ExportQpOases3Interface : public ExportQpOasesInterface
{
public:

	/** Default constructor.
	 *
	 *	@param[in] _headerFileName		Name of exported file for header file.
	 *	@param[in] _sourceFileName		Name of exported file for source file.
	 *	@param[in] _commonHeaderName	Name of common header file to be included.
	 *	@param[in] _realString			std::string to be used to declare real variables.
	 *	@param[in] _intString			std::string to be used to declare integer variables.
	 *	@param[in] _precision			Number of digits to be used for exporting real values.
	 *	@param[in] _commentString		std::string to be used for exporting comments.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	ExportQpOases3Interface(	const std::string& _headerFileName,
								const std::string& _sourceFileName,
								const std::string& _commonHeaderName = "",
								const std::string& _realString = "real_t",
								const std::string& _intString = "int",
								int _precision = 16,
								const std::string& _commentString = std::string()
								);

	/** Destructor. */
	virtual ~ExportQpOases3Interface()
	{}

	/** Configure the template
	 *
	 *  \return SUCCESSFUL_RETURN
	 */
	virtual returnValue configure(	const std::string& _prefix,
									const std::string& _solverDefine,
									const int nvmax,
									const int ncmax,
									const int nwsrmax,
									const std::string& _printLevel,
									bool _useSinglePrecision,

									const std::string& _commonHeader,
									const std::string& _namespace,
									const std::string& _primalSolution,
									const std::string& _dualSolution,
									const std::string& _sigma,
									bool _hotstartQP,
									bool _externalCholesky,
									const std::string& _qpH,
									const std::string& _qpR,
									const std::string& _qpg,
									const std::string& _qpA,
									const std::string& _qplb,
									const std::string& _qpub,
									const std::string& _qplbA,
									const std::string& _qpubA
									);

	/** Export the interface. */
	virtual returnValue exportCode();

protected:

};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_EXPORT_QPOASES3_INTERFACE_HPP
