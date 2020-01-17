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
 *    \file include/acado/code_generation/export_cholesky_decomposition.hpp
 *    \author Milan Vukov
 *    \date 2013
 */

#ifndef ACADO_TOOLKIT_EXPORT_CHOLESKY_DECOMPOSITION_HPP
#define ACADO_TOOLKIT_EXPORT_CHOLESKY_DECOMPOSITION_HPP

#include <acado/code_generation/export_algorithm.hpp>
#include <acado/code_generation/export_function.hpp>
#include <acado/code_generation/export_variable.hpp>

BEGIN_NAMESPACE_ACADO

/**
 *	\brief A class for exporting a function for calculation of the
 *	       Cholesky decomposition.
 *
 *	\ingroup NumericalAlgorithms
 *
 *	\author Milan Vukov
 */

class ExportCholeskyDecomposition: public ExportAlgorithm
{
public:

	/** Default constructor.
	 *
	 *	@param[in] _userInteraction		Pointer to corresponding user interface.
	 *	@param[in] _commonHeaderName	Name of common header file to be included.
	 */
	ExportCholeskyDecomposition(	UserInteraction* _userInteraction = 0,
									const std::string& _commonHeaderName = ""
									);

	/** Destructor. */
	virtual ~ExportCholeskyDecomposition()
	{}

	/** Initializes the different parameters of the linear solver that will be exported.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	returnValue init(	const std::string& _name,
						unsigned _dim,
						bool _unrolling = false
						);

	/** Initializes code export into given file.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue setup();

	/** Adds all data declarations of the auto-generated algorithm to given list of declarations.
	 *
	 *	@param[in] declarations		List of declarations.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue getDataDeclarations(	ExportStatementBlock& declarations,
												ExportStruct dataStruct = ACADO_ANY
												) const;

	/** Adds all function (forward) declarations of the auto-generated algorithm to given list of declarations.
	 *
	 *	@param[in] declarations		List of declarations.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue getFunctionDeclarations(	ExportStatementBlock& declarations
													) const;

	/** Exports source code of the auto-generated algorithm into the given directory.
	 *
	 *	@param[in] code				Code block containing the auto-generated algorithm.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue getCode(	ExportStatementBlock& code
									);

	/** Get name of the function that perform the decomposition. */
	const std::string getName();

private:

	ExportVariable A;
	ExportFunction fcn;
	bool unrolling;
};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_EXPORT_CHOLESKY_DECOMPOSITION_HPP
