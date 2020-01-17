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
 *    \file include/acado/integrators/erk_adjoint_export.hpp
 *    \author Rien Quirynen
 *    \date 2014
 */


#ifndef ACADO_TOOLKIT_ERK_ADJOINT_EXPORT_HPP
#define ACADO_TOOLKIT_ERK_ADJOINT_EXPORT_HPP

#include <acado/code_generation/integrators/erk_export.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Allows to export a tailored explicit Runge-Kutta integrator with adjoint first order sensitivity propagation for fast model predictive control.
 *
 *	\ingroup NumericalAlgorithms
 *
 *	The class AdjointERKExport allows to export a tailored explicit Runge-Kutta integrator with adjoint first order sensitivity propagation
 *	for fast model predictive control.
 *
 *	\author Rien Quirynen
 */
class AdjointERKExport : public ExplicitRungeKuttaExport
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
        AdjointERKExport(	UserInteraction* _userInteraction = 0,
							const std::string& _commonHeaderName = ""
							);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
        AdjointERKExport(	const AdjointERKExport& arg
							);

        /** Destructor. 
		 */
        virtual ~AdjointERKExport( );


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


    protected:

		ExportVariable rk_forward_sweep;				/**< Variable containing intermediate results of a forward sweep of the RK integrator. */
//		ExportVariable seed_backward;					/**< Variable containing the seed for a backward propagation of the RK integrator. */

};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_ERK_ADJOINT_EXPORT_HPP

// end of file.
