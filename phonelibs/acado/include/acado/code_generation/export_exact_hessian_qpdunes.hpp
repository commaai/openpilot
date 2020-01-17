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
 *    \file include/acado/code_generation/export_gauss_newton_qpdunes.hpp
 *    \author Rien Quirynen
 *    \date 2014
 */

#ifndef ACADO_TOOLKIT_EXPORT_EXACT_HESSIAN_QPDUNES_HPP
#define ACADO_TOOLKIT_EXPORT_EXACT_HESSIAN_QPDUNES_HPP

#include <acado/code_generation/export_nlp_solver.hpp>
#include <acado/code_generation/export_gauss_newton_qpdunes.hpp>

BEGIN_NAMESPACE_ACADO

/**
 *	\brief A class for export of an OCP solver using sparse QP solver qpDUNES
 *
 *	\ingroup NumericalAlgorithms
 *
 *	The class ExportExactHessianQpDunes allows export of an OCP solver using
 *	the generalized Exact-Hessian method. The underlying QP is solved using the
 *	structured sparse QP solver qpDUNES.
 *
 *	\author Rien Quirynen
 */
class ExportExactHessianQpDunes : public ExportGaussNewtonQpDunes
{
public:

	/** Default constructor.
	 *
	 *	@param[in] _userInteraction		Pointer to corresponding user interface.
	 *	@param[in] _commonHeaderName	Name of common header file to be included.
	 */
	ExportExactHessianQpDunes(	UserInteraction* _userInteraction = 0,
								const std::string& _commonHeaderName = ""
								);

	/** Destructor.
	*/
	virtual ~ExportExactHessianQpDunes( )
	{}

	/** Initializes export of an algorithm.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue setup( );

	/** Adds all function (forward) declarations of the auto-generated condensing algorithm
	 *	to given list of declarations.
	 *
	 *	@param[in] declarations		List of declarations.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue getFunctionDeclarations(	ExportStatementBlock& declarations
													) const;


	/** Exports source code of the auto-generated condensing algorithm
	 *  into the given directory.
	 *
	 *	@param[in] code				Code block containing the auto-generated condensing algorithm.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue getCode(	ExportStatementBlock& code
									);

protected:

	/** Setting up of an objective evaluation:
	 *   - functions and derivatives evaulation
	 *   - creating Hessians and gradients
	 *
	 *   \return SUCCESSFUL_RETURN
	 */
	virtual returnValue setupObjectiveEvaluation( void );

	/** Setting up of a Hessian regularization routine */
	virtual returnValue setupHessianRegularization( void );

protected:

};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_EXPORT_EXACT_HESSIAN_QPDUNES_HPP
