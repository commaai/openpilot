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
 *    \file include/acado/code_generation/integrators/irk_lifted_fob_export.hpp
 *    \author Rien Quirynen
 *    \date 2015
 */


#ifndef ACADO_TOOLKIT_LIFTED_IRK_FORWARD_BACKWARD_EXPORT_HPP
#define ACADO_TOOLKIT_LIFTED_IRK_FORWARD_BACKWARD_EXPORT_HPP

#include <acado/code_generation/integrators/irk_lifted_forward_export.hpp>


BEGIN_NAMESPACE_ACADO

/** 
 *	\brief Allows to export a tailored lifted implicit Runge-Kutta integrator with forward-over-adjoint second order sensitivity generation for extra fast model predictive control.
 *
 *	\ingroup NumericalAlgorithms
 *
 *	The class ForwardBackwardLiftedIRKExport allows to export a tailored lifted implicit Runge-Kutta integrator
 *	with forward-over-adjoint second order sensitivity generation for extra fast model predictive control.
 *
 *	\author Rien Quirynen
 */
class ForwardBackwardLiftedIRKExport : public ForwardLiftedIRKExport
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
        ForwardBackwardLiftedIRKExport(	UserInteraction* _userInteraction = 0,
							const std::string& _commonHeaderName = ""
							);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
        ForwardBackwardLiftedIRKExport(	const ForwardBackwardLiftedIRKExport& arg
							);

        /** Destructor. 
		 */
        virtual ~ForwardBackwardLiftedIRKExport( );


		/** Assignment operator (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
		ForwardBackwardLiftedIRKExport& operator=(	const ForwardBackwardLiftedIRKExport& arg
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


		/** Exports the evaluation of the states at all stages.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *	@param[in] Ah				The matrix A of the IRK method, multiplied by the step size h.
		 *	@param[in] index			The loop index, defining the stage.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue evaluateAllStatesImplicitSystem( 	ExportStatementBlock* block,
											const ExportIndex& k_index,
											const ExportVariable& Ah,
											const ExportVariable& C,
											const ExportIndex& stage,
											const ExportIndex& i,
											const ExportIndex& tmp_index );


		virtual returnValue updateImplicitSystem( 	ExportStatementBlock* block,
													const ExportIndex& index1,
													const ExportIndex& index2,
													const ExportIndex& tmp_index  	);


		Expression returnLowerTriangular( const Expression& expr );


		returnValue updateHessianTerm( ExportStatementBlock* block, const ExportIndex& index1, const ExportIndex& index2 );


		virtual returnValue allSensitivitiesImplicitSystem( 	ExportStatementBlock* block,
													const ExportIndex& index1,
													const ExportIndex& index2,
													const ExportIndex& index3,
													const ExportIndex& tmp_index1,
													const ExportIndex& tmp_index2,
													const ExportIndex& tmp_index3,
													const ExportIndex& k_index,
													const ExportVariable& Bh,
													bool update );


		/** Exports the code needed to compute the sensitivities of the states defined by the nonlinear, fully implicit system.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *	@param[in] Ah				The variable containing the internal coefficients of the RK method, multiplied with the step size.
		 *	@param[in] Bh				The variable containing the weights of the RK method, multiplied with the step size.
		 *	@param[in] det				The variable that holds the determinant of the matrix in the linear system.
		 *	@param[in] STATES			True if the sensitivities with respect to a state are needed, false otherwise.
		 *	@param[in] number			This number defines the stage of the state with respect to which the sensitivities are computed.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue evaluateRhsInexactSensitivities( 	ExportStatementBlock* block,
													const ExportIndex& index1,
													const ExportIndex& index2,
													const ExportIndex& index3,
													const ExportIndex& tmp_index1,
													const ExportIndex& tmp_index2,
													const ExportIndex& tmp_index3,
													const ExportIndex& k_index,
													const ExportVariable& Ah );


		/** Returns the largest global export variable.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		ExportVariable getAuxVariable() const;


    protected:


		ExportAcadoFunction diffs_sweep;		/**< Module to export the evaluation of a forward sweep of the derivatives of the ordinary differential equations. */
		ExportAcadoFunction adjoint_sweep;		/**< Module to export the evaluation of a forward sweep of the derivatives of the ordinary differential equations. */

		ExportVariable  rk_b_trans;

		ExportVariable  rk_adj_diffs_tmp;

		ExportVariable  rk_Khat_traj;
		ExportVariable  rk_Xhat_traj;

		ExportVariable	rk_xxx_traj;			/**< Variable containing the forward trajectory of the state values. */
		ExportVariable	rk_adj_traj;			/**< Variable containing the adjoint trajectory of the lambda_hat values. */
		ExportVariable	rk_S_traj;				/**< Variable containing the forward trajectory of the first order sensitivities. */
		ExportVariable	rk_A_traj;				/**< Variable containing the factorized matrix of the linear system over the forward trajectory. */

		ExportVariable  rk_hess_tmp1;
		ExportVariable  rk_hess_tmp2;

};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_LIFTED_IRK_FORWARD_BACKWARD_EXPORT_HPP

// end of file.
