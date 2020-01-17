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
 *    \file include/acado/code_generation/export_split_qpdunes_interface.hpp
 *    \author Milan Vukov, Rien Quirynen
 *    \date 2013 - 2014
 */

#ifndef ACADO_TOOLKIT_EXPORT_SPLIT_QPDUNES_INTERFACE_HPP
#define ACADO_TOOLKIT_EXPORT_SPLIT_QPDUNES_INTERFACE_HPP


#include <acado/code_generation/export_templated_file.hpp>

BEGIN_NAMESPACE_ACADO

/**
 *	\brief Interface generator for the qpDUNES QP solver
 *
 *	\ingroup AuxiliaryFunctionality
 *
 *	\author Milan Vukov
 */
class ExportSplitQpDunesInterface : public ExportTemplatedFile
{
public:
	/** Default constructor.
	 *
	 *	@param[in] _fileName			Name of exported file.
	 *	@param[in] _commonHeaderName	Name of common header file to be included.
	 *	@param[in] _realString			std::string to be used to declare real variables.
	 *	@param[in] _intString			std::string to be used to declare integer variables.
	 *	@param[in] _precision			Number of digits to be used for exporting real values.
	 *	@param[in] _commentstd::string		std::string to be used for exporting comments.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	ExportSplitQpDunesInterface(	const std::string& _fileName,
							const std::string& _commonHeaderName = "",
							const std::string& _realString = "real_t",
							const std::string& _intString = "int",
							int _precision = 16,
							const std::string& _commentString = std::string()
							);

	/** Destructor. */
	virtual ~ExportSplitQpDunesInterface( )
	{}

	/** Configure the template
	 *
	 *  \return SUCCESSFUL_RETURN
	 */
	returnValue configure(	const unsigned _maxIter,
							const unsigned _printLevel,
							const std::string& _HH,
							const std::string& _g,
							const std::string& _gN,
							const std::string& _CC,
							const std::string& _c,
							const std::string& _DD,
							const std::string& _lb0,
							const std::string& _ub0,
							const std::string& _lb,
							const std::string& _ub,
							const std::string& _lbA,
							const std::string& _ubA,
							const std::string& _primal,
							const std::string& _lambda,
							const std::string& _mu,
							const std::vector< unsigned >& conDim,
							const std::string& _initialStateFixed,
							const std::string& _diagH,
							const std::string& _diagHN,
							const unsigned _NI,
							const unsigned _NX,
							const unsigned _NU
							);
};

CLOSE_NAMESPACE_ACADO

#endif // ACADO_TOOLKIT_EXPORT_SPLIT_QPDUNES_INTERFACE_HPP
