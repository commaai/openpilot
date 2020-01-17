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
 *    \file include/acado/integrators/lifted_erk_export.hpp
 *    \author Rien Quirynen
 *    \date 2015
 */


#ifndef ACADO_TOOLKIT_LIFTED_ERK_EXPORT_HPP
#define ACADO_TOOLKIT_LIFTED_ERK_EXPORT_HPP

#include <acado/code_generation/integrators/erk_export.hpp>
#include <acado/code_generation/linear_solvers/linear_solver_generation.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Allows to export a tailored explicit Runge-Kutta integrator with a lifted Newton method to efficiently support (implicit) DAE systems for fast model predictive control.
 *
 *	\ingroup NumericalAlgorithms
 *
 *	The class LiftedERKExport allows to export a tailored explicit Runge-Kutta integrator with a lifted Newton method to efficiently support (implicit) DAE systems
 *	for fast model predictive control.
 *
 *	\author Rien Quirynen
 */
class LiftedERKExport : public ExplicitRungeKuttaExport
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //

    public:

		/** Default constructor. 
		 *
		 *	@param[in] _userInteraction		Pointer to corresponding user interface.
		 *	@param[in] _commonHeaderName	Name of common header file to be included.
		 */
        LiftedERKExport(	UserInteraction* _userInteraction = 0,
							const std::string& _commonHeaderName = ""
							);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
        LiftedERKExport(	const LiftedERKExport& arg
							);

        /** Destructor. 
		 */
        virtual ~LiftedERKExport( );


		/** Assigns Differential Equation to be used by the integrator.
		 *
		 *	@param[in] rhs		Right-hand side expression.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */

		virtual returnValue setDifferentialEquation( const Expression& rhs );


		/** Initializes export of a tailored integrator.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue setup( );


		/** Adds all data declarations of the auto-generated integrator to given list of declarations.
		 *
		 *	@param[in] declarations		List of declarations.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue getDataDeclarations(	ExportStatementBlock& declarations,
													ExportStruct dataStruct = ACADO_ANY
													) const;


		/** Exports source code of the auto-generated integrator into the given directory.
		 *
		 *	@param[in] code				Code block containing the auto-generated integrator.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue getCode(	ExportStatementBlock& code
										);


	protected:


		/** Returns the largest global export variable.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		ExportVariable getAuxVariable() const;


    protected:

		ExportAcadoFunction alg_rhs;		/**< Module to export the evaluation of the derivatives of the algebraic equations. */

		ExportLinearSolver* solver;			/**< This is the exported linear solver that is used by the implicit Runge-Kutta method. */

		ExportVariable	rk_A;				/**< Variable containing the matrix of the linear system. */
		ExportVariable	rk_b;				/**< Variable containing the right-hand side of the linear system. */
		ExportVariable  rk_auxSolver;		/**< Variable containing auxiliary values for the exported linear solver. */

		ExportVariable	rk_zzz;				/**< Variable containing the lifted algebraic variables. */
		ExportVariable	rk_zTemp;			/**< Variable containing the evaluation of the algebraic equations */

		ExportVariable	rk_diffZ;			/**< Variable containing the sensitivities of the algebraic variables */
		ExportVariable	rk_delta;			/**< Variable containing the difference of the optimization variables */
		ExportVariable	rk_prev;			/**< Variable containing the previous values of the optimization variables over the horizon */

};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_LIFTED_ERK_EXPORT_HPP

// end of file.
