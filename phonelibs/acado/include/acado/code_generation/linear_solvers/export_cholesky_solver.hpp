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
 *    \file include/acado/code_generation/export_cholesky_solver.hpp
 *    \author Milan Vukov
 *    \date 2014
 */

#ifndef ACADO_TOOLKIT_EXPORT_CHOLESKY_SOLVER_HPP
#define ACADO_TOOLKIT_EXPORT_CHOLESKY_SOLVER_HPP

#include <acado/code_generation/linear_solvers/linear_solver_export.hpp>

BEGIN_NAMESPACE_ACADO

/**
 *	\brief Allows to export linear solver based on Cholesky factorization
 *
 *	\ingroup NumericalAlgorithms
 *
 *	Allows export of linear system solver, where the A matrix is a symmetric
 *	positive definite of dimensions n x n, and matrix B is of dimensions
 *	n x m, m >= 1.
 *
 *	This class configures and exports two functions:
 *
 *	- chol( A ), which computes Cholesky decomposition of matrix A = R' * R
 *	  R is upper triangular. A gets overwritten by R.
 *	- solve(R, B), which at the moment solves the system R' * X = B. B gets
 *	  overwritten by the solution X.
 *
 *	\note Early experimental version.
 *	\note Exported solver code is based on qpOASES code.
 *	\todo Extend the generator to be able to solve RX = B, too.
 *
 *	\author Milan Vukov
 */

class ExportCholeskySolver : public ExportLinearSolver
{
public:

	/** Default constructor.
	 *
	 *	@param[in] _userInteraction		Pointer to corresponding user interface.
	 *	@param[in] _commonHeaderName	Name of common header file to be included.
	 */
	ExportCholeskySolver(	UserInteraction* _userInteraction = 0,
							const std::string& _commonHeaderName = ""
							);

	/** Destructor. */
	virtual ~ExportCholeskySolver( );

	/** Solver initialization routine.
	 *
	 *	\return SUCCESSFUL_RETURN
	 *
	 * */
	returnValue init(	unsigned _dimA,
						unsigned _numColsB,
						const std::string& _id
						);

	/** The setup routine.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue setup( );

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

	/** Appends the names of the used variables to a given stringstream.
	 *
	 *	@param[in] string				The string to which the names of the used variables are appended.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue appendVariableNames( std::stringstream& string );

	/** Get the reference to the Cholesky decomposition routine. */
	const ExportFunction& getCholeskyFunction() const;

	/** Get the reference to the solve function. */
	const ExportFunction& getSolveFunction() const;

private:

	ExportFunction chol;

	ExportVariable B;

	unsigned nColsB;
};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_EXPORT_CHOLESKY_SOLVER_HPP
