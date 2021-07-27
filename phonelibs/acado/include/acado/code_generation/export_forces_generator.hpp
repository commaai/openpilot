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
 *    \file include/acado/code_generation/export_forces_generator.hpp
 *    \author Milan Vukov
 *    \date 2012 - 2014
 */

#ifndef ACADO_TOOLKIT_EXPORT_FORCES_GENERATOR_HPP
#define ACADO_TOOLKIT_EXPORT_FORCES_GENERATOR_HPP

#include <acado/code_generation/export_templated_file.hpp>

BEGIN_NAMESPACE_ACADO

/**
 *	\brief  Generator of the FORCES interface, the to, be called from MATLAB
 *
 *	\ingroup AuxiliaryFunctionality
 *
 *	\author Milan Vukov
 */
class ExportForcesGenerator : public ExportTemplatedFile
{
public:

	/** Default constructor.
	 *
	 *	@param[in] _templateName		Name of a template.
	 *	@param[in] _fileName			Name of exported file.
	 *	@param[in] _commonHeaderName	Name of common header file to be included.
	 *	@param[in] _realString			std::string to be used to declare real variables.
	 *	@param[in] _intString			std::string to be used to declare integer variables.
	 *	@param[in] _precision			Number of digits to be used for exporting real values.
	 *	@param[in] _commentString		std::string to be used for exporting comments.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	ExportForcesGenerator(	const std::string& _templateName,
							const std::string& _fileName,
							const std::string& _commonHeaderName = "",
							const std::string& _realString = "real_t",
							const std::string& _intString = "int",
							int _precision = 16,
							const std::string& _commentString = std::string()
							);

	/** Destructor. */
	virtual ~ExportForcesGenerator( )
	{}

	/** Configure the template
	 *
	 *  \return SUCCESSFUL_RETURN
	 */
	returnValue configure(	const unsigned _nx,
							const unsigned _nu,
							const unsigned _NN,
							const std::vector< std::vector< unsigned > >& _lbIdx,
							const std::vector< std::vector< unsigned > >& _ubIdx,
							const std::vector< unsigned >& _AbDim,
							const bool _constHessian,
							const bool _diagHessian,
							const bool _diagHessianN,
							const bool _fixedInitialState,
							const std::string& _solverName,
							const unsigned _printLevel,
							const unsigned _maxIterations,
							const unsigned _parallel,
							bool matlabGenerator,
							bool warmStart
							);
};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_EXPORT_FORCES_GENERATOR_HPP
