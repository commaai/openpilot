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
 *    \file include/acado/code_generation/integrators/irk_lifted_symmetric_export.hpp
 *    \author Rien Quirynen
 *    \date 2015
 */


#ifndef ACADO_TOOLKIT_LIFTED_IRK_SYMMETRIC_EXPORT_HPP
#define ACADO_TOOLKIT_LIFTED_IRK_SYMMETRIC_EXPORT_HPP

#include <acado/code_generation/integrators/irk_lifted_forward_export.hpp>


BEGIN_NAMESPACE_ACADO

/** 
 *	\brief Allows to export a tailored lifted implicit Runge-Kutta integrator with symmetric second order sensitivity generation for extra fast model predictive control.
 *
 *	\ingroup NumericalAlgorithms
 *
 *	The class SymmetricLiftedIRKExport allows to export a tailored lifted implicit Runge-Kutta integrator
 *	with symmetric second order sensitivity generation for extra fast model predictive control.
 *
 *	\author Rien Quirynen
 */
class SymmetricLiftedIRKExport : public ForwardLiftedIRKExport
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
        SymmetricLiftedIRKExport(	UserInteraction* _userInteraction = 0,
							const std::string& _commonHeaderName = ""
							);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
        SymmetricLiftedIRKExport(	const SymmetricLiftedIRKExport& arg
							);

        /** Destructor. 
		 */
        virtual ~SymmetricLiftedIRKExport( );


		/** Assignment operator (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
		SymmetricLiftedIRKExport& operator=(	const SymmetricLiftedIRKExport& arg
										);


		/** Initializes export of a tailored integrator.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue setup( );


		/** Assigns Differential Equation to be used by the integrator.
		 *
		 *	@param[in] rhs		Right-hand side expression.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue setDifferentialEquation( const Expression& rhs );
        

		/** Adds all data declarations of the auto-generated integrator to given list of declarations.
		 *
		 *	@param[in] declarations		List of declarations.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue getDataDeclarations(	ExportStatementBlock& declarations,
													ExportStruct dataStruct = ACADO_ANY
													) const;


		/** Adds all function (forward) declarations of the auto-generated integrator to given list of declarations.
		 *
		 *	@param[in] declarations		List of declarations.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue getFunctionDeclarations(	ExportStatementBlock& declarations
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


		virtual returnValue updateImplicitSystem( 	ExportStatementBlock* block,
													const ExportIndex& index1,
													const ExportIndex& index2,
													const ExportIndex& tmp_index  	);


		Expression returnLowerTriangular( const Expression& expr );


		/** Returns the largest global export variable.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		ExportVariable getAuxVariable() const;


    protected:


		ExportAcadoFunction diffs_sweep;		/**< Module to export the evaluation of a forward sweep of the derivatives of the ordinary differential equations. */

		ExportVariable	rk_S_traj;				/**< Variable containing the forward trajectory of the first order sensitivities. */
		ExportVariable	rk_A_traj;				/**< Variable containing the factorized matrix of the linear system over the forward trajectory. */
		ExportVariable	rk_aux_traj;			/**< Variable containing the factorized matrix of the linear system over the forward trajectory. */

		ExportVariable 	rk_kkk_prev;			/**< TODO. */
		ExportVariable 	rk_kkk_delta;			/**< TODO. */
		ExportVariable 	rk_delta_full;			/**< Variable containing the FULL update on all the variables. */
		ExportVariable 	rk_diff_mu;				/**< Sensitivities to update the mu variables. */
		ExportVariable 	rk_diff_lam;			/**< Sensitivities to propagate the lambda variables. */

		ExportVariable 	rk_lambda;				/**< The lambda variables over the shooting interval. */
		ExportVariable 	rk_b_mu;				/**< The lambda variables over the shooting interval. */

};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_LIFTED_IRK_SYMMETRIC_EXPORT_HPP

// end of file.
