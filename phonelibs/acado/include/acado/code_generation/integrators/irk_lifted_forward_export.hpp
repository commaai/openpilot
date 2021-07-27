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
 *    \file include/acado/code_generation/integrators/irk_lifted_forward_export.hpp
 *    \author Rien Quirynen
 *    \date 2014
 */


#ifndef ACADO_TOOLKIT_LIFTED_IRK_FORWARD_EXPORT_HPP
#define ACADO_TOOLKIT_LIFTED_IRK_FORWARD_EXPORT_HPP

#include <acado/code_generation/integrators/irk_forward_export.hpp>


BEGIN_NAMESPACE_ACADO

/** 
 *	\brief Allows to export a tailored lifted implicit Runge-Kutta integrator with forward sensitivity generation for extra fast model predictive control.
 *
 *	\ingroup NumericalAlgorithms
 *
 *	The class ForwardLiftedIRKExport allows to export a tailored lifted implicit Runge-Kutta integrator
 *	with forward sensitivity generation for extra fast model predictive control.
 *
 *	\author Rien Quirynen
 */
class ForwardLiftedIRKExport : public ForwardIRKExport
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
        ForwardLiftedIRKExport(	UserInteraction* _userInteraction = 0,
							const std::string& _commonHeaderName = ""
							);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
        ForwardLiftedIRKExport(	const ForwardLiftedIRKExport& arg
							);

        /** Destructor. 
		 */
        virtual ~ForwardLiftedIRKExport( );


		/** Assignment operator (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
		ForwardLiftedIRKExport& operator=(	const ForwardLiftedIRKExport& arg
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


		/** Precompute as much as possible for the linear input system and export the resulting definitions.
		 *
		 *	@param[in] code			The block to which the code will be exported.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue prepareInputSystem(	ExportStatementBlock& code );


		/** Precompute as much as possible for the linear output system and export the resulting definitions.
		 *
		 *	@param[in] code			The block to which the code will be exported.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue prepareOutputSystem( ExportStatementBlock& code );


		/** Exports the code needed to solve the system of collocation equations for the nonlinear, fully implicit system.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *	@param[in] Ah				The variable containing the internal coefficients of the RK method, multiplied with the step size.
		 *	@param[in] det				The variable that holds the determinant of the matrix in the linear system.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue solveImplicitSystem( 	ExportStatementBlock* block,
											const ExportIndex& index1,
											const ExportIndex& index2,
											const ExportIndex& index3,
											const ExportIndex& tmp_index,
											const ExportIndex& k_index,
											const ExportVariable& Ah,
											const ExportVariable& C,
											const ExportVariable& det,
											bool DERIVATIVES = false );


		/** Exports the evaluation of the inexact matrix for the linear system.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *	@param[in] index1			The loop index of the outer loop.
		 *	@param[in] index2			The loop index of the inner loop.
		 *	@param[in] tmp_index		A temporary index to be used.
		 *	@param[in] Ah				The matrix A of the IRK method, multiplied by the step size h.
		 *	@param[in] evaluateB		True if the right-hand side of the linear system should also be evaluated, false otherwise.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue evaluateInexactMatrix( 	ExportStatementBlock* block,
										const ExportIndex& index1,
										const ExportIndex& index2,
										const ExportIndex& tmp_index,
										const ExportIndex& k_index,
										const ExportVariable& _rk_A,
										const ExportVariable& Ah,
										const ExportVariable& C,
										bool evaluateB,
										bool DERIVATIVES );


		/** Exports the code needed to compute the sensitivities of the states, defined by the linear input system.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *	@param[in] Bh				The variable containing the weights of the RK method, multiplied with the step size.
		 *	@param[in] STATES			True if the sensitivities with respect to a state are needed, false otherwise.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue sensitivitiesInputSystem( 	ExportStatementBlock* block,
														const ExportIndex& index1,
														const ExportIndex& index2,
														const ExportVariable& Bh,
														bool STATES  	);


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


		/** Exports the evaluation of the states at a specific stage.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *	@param[in] Ah				The matrix A of the IRK method, multiplied by the step size h.
		 *	@param[in] index			The loop index, defining the stage.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue evaluateStatesImplicitSystem( 	ExportStatementBlock* block,
											const ExportIndex& k_index,
											const ExportVariable& Ah,
											const ExportVariable& C,
											const ExportIndex& stage,
											const ExportIndex& i,
											const ExportIndex& tmp_index );


		/** Exports the evaluation of the right-hand side of the linear system at a specific stage.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *	@param[in] index			The loop index, defining the stage.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue evaluateRhsImplicitSystem( 	ExportStatementBlock* block,
														const ExportIndex& k_index,
														const ExportIndex& stage );


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


		virtual returnValue evaluateRhsSensitivities( 	ExportStatementBlock* block,
													const ExportIndex& index1,
													const ExportIndex& index2,
													const ExportIndex& index3,
													const ExportIndex& tmp_index1,
													const ExportIndex& tmp_index2 );


		/** Exports the code needed to update the sensitivities of the states defined by the nonlinear, fully implicit system.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue updateImplicitSystem( 	ExportStatementBlock* block,
													const ExportIndex& index1,
													const ExportIndex& index2,
													const ExportIndex& tmp_index  	);


		/** Exports the code needed to compute the sensitivities of the states, defined by the linear output system.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *	@param[in] Ah				The variable containing the internal coefficients of the RK method, multiplied with the step size.
		 *	@param[in] Bh				The variable containing the weights of the RK method, multiplied with the step size.
		 *	@param[in] STATES			True if the sensitivities with respect to a state are needed, false otherwise.
		 *	@param[in] number			This number defines the stage of the state with respect to which the sensitivities are computed.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue sensitivitiesOutputSystem( 	ExportStatementBlock* block,
												const ExportIndex& index1,
												const ExportIndex& index2,
												const ExportIndex& index3,
												const ExportIndex& index4,
												const ExportIndex& tmp_index1,
												const ExportIndex& tmp_index2,
												const ExportVariable& Ah,
												const ExportVariable& Bh,
												bool STATES,
												uint number 		);


		/** Exports the computation of the sensitivities for the continuous output.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *	@param[in] tmp_meas			The number of measurements in the current integration step (in case of an online grid).
		 *	@param[in] rk_tPrev			The time point, defining the beginning of the current integration step (in case of an online grid).
		 *	@param[in] time_tmp			A variable used for time transformations (in case of an online grid).
		 *	@param[in] STATES			True if the sensitivities with respect to a state are needed, false otherwise.
		 *	@param[in] base				The number of states in stages with respect to which the sensitivities have already been computed.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue sensitivitiesOutputs( 	ExportStatementBlock* block,
											const ExportIndex& index0,
											const ExportIndex& index1,
											const ExportIndex& index2,
											const ExportIndex& tmp_index1,
											const ExportIndex& tmp_index2,
											const ExportIndex& tmp_index3,
											const ExportVariable& tmp_meas,
											const ExportVariable& time_tmp,
											bool STATES,
											uint base			);


		/** Exports the propagation of the sensitivities for the continuous output.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *	@param[in] tmp_meas			The number of measurements in the current integration step (in case of an online grid).
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue propagateOutputs(	ExportStatementBlock* block,
												const ExportIndex& index,
												const ExportIndex& index0,
												const ExportIndex& index1,
												const ExportIndex& index2,
												const ExportIndex& index3,
												const ExportIndex& tmp_index1,
												const ExportIndex& tmp_index2,
												const ExportIndex& tmp_index3,
												const ExportIndex& tmp_index4,
												const ExportVariable& tmp_meas );


		/** Returns the largest global export variable.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		ExportVariable getAuxVariable() const;


    protected:

		ExportAcadoFunction forward_sweep;		/**< Module to export the evaluation of a forward sweep of the derivatives of the ordinary differential equations. */
		ExportAcadoFunction adjoint_sweep;		/**< Module to export the evaluation of an adjoint sweep of the derivatives of the ordinary differential equations. */
		ExportVariable  rk_b_trans;
		ExportVariable	rk_adj_traj;			/**< Variable containing the adjoint trajectory of the lambda_hat values. */
		ExportVariable  rk_adj_diffs_tmp;
		ExportVariable 	rk_seed2;
		ExportVariable	rk_xxx_traj;			/**< Variable containing the forward trajectory of the state values. */

		ExportVariable 	rk_diffSweep;
		ExportVariable 	rk_I;

		ExportVariable 	rk_seed;				/**< Variable containing the forward seed. */
		ExportVariable 	rk_stageValues;			/**< Variable containing the evaluated stage values. */

		ExportVariable 	rk_Xprev;				/**< Variable containing the full previous state trajectory. */
		ExportVariable 	rk_Uprev;				/**< Variable containing the previous control trajectory. */

		ExportVariable 	rk_delta;				/**< Variable containing the update on the optimization variables. */

		ExportVariable  rk_xxx_lin;
		ExportVariable  rk_Khat_traj;
		ExportVariable  rk_Xhat_traj;

		ExportVariable 	rk_diffK_local;

};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_LIFTED_IRK_FORWARD_EXPORT_HPP

// end of file.
