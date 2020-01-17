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
 *    \file include/acado/code_generation/integrators/dirk_export.hpp
 *    \author Rien Quirynen
 *    \date 2013
 */


#ifndef ACADO_TOOLKIT_DIRK_EXPORT_HPP
#define ACADO_TOOLKIT_DIRK_EXPORT_HPP

#include <acado/code_generation/integrators/irk_export.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Allows to export a tailored diagonally implicit Runge-Kutta integrator for fast model predictive control.
 *
 *	\ingroup NumericalAlgorithms
 *
 *	The class DiagonallyImplicitRKExport allows to export a tailored diagonally implicit Runge-Kutta integrator
 *	for fast model predictive control.
 *
 *	\author Rien Quirynen
 */
class DiagonallyImplicitRKExport : public ForwardIRKExport
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
        DiagonallyImplicitRKExport(	UserInteraction* _userInteraction = 0,
							const std::string& _commonHeaderName = ""
							);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
        DiagonallyImplicitRKExport(	const DiagonallyImplicitRKExport& arg
							);

        /** Destructor. 
		 */
        virtual ~DiagonallyImplicitRKExport( );


		/** Assignment operator (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
		DiagonallyImplicitRKExport& operator=(	const DiagonallyImplicitRKExport& arg
										);


		/** Exports the code needed to solve the system of collocation equations for the linear input system.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *	@param[in] A1				A constant matrix defining the equations of the linear input system.
		 *	@param[in] B1				A constant matrix defining the equations of the linear input system.
		 *	@param[in] Ah				The variable containing the internal coefficients of the RK method, multiplied with the step size.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue solveInputSystem( 	ExportStatementBlock* block,
										const ExportIndex& index1,
										const ExportIndex& index2,
										const ExportIndex& index3,
										const ExportIndex& tmp_index,
										const ExportVariable& Ah );


		/** Precompute as much as possible for the linear input system and export the resulting definitions.
		 *
		 *	@param[in] code			The block to which the code will be exported.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue prepareInputSystem(	ExportStatementBlock& code );


		/** Exports the code needed to solve the system of collocation equations for the linear output system.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *	@param[in] Ah				The variable containing the internal coefficients of the RK method, multiplied with the step size.
		 *	@param[in] A3				A constant matrix defining the equations of the linear output system.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue solveOutputSystem( 	ExportStatementBlock* block,
										const ExportIndex& index1,
										const ExportIndex& index2,
										const ExportIndex& index3,
										const ExportIndex& tmp_index,
										const ExportVariable& Ah,
										bool DERIVATIVES = false );


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


		/** Precompute as much as possible for the linear output system and export the resulting definitions.
		 *
		 *	@param[in] code			The block to which the code will be exported.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue prepareOutputSystem( ExportStatementBlock& code );


		/** Forms a constant linear system matrix for the collocation equations, given a constant jacobian and mass matrix.
		 *
		 *	@param[in] jacobian			given constant Jacobian matrix
		 *	@param[in] mass				given constant mass matrix
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual DMatrix formMatrix( const DMatrix& mass, const DMatrix& jacobian );


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
											const ExportVariable& Ah,
											const ExportVariable& C,
											const ExportVariable& det,
											bool DERIVATIVES = false  	);


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
		virtual returnValue sensitivitiesImplicitSystem( 	ExportStatementBlock* block,
													const ExportIndex& index1,
													const ExportIndex& index2,
													const ExportIndex& index3,
													const ExportIndex& tmp_index1,
													const ExportIndex& tmp_index2,
													const ExportVariable& Ah,
													const ExportVariable& Bh,
													const ExportVariable& det,
													bool STATES,
													uint number 		);


		/** Exports the evaluation of the matrix of the linear system.
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
		virtual returnValue evaluateMatrix( 	ExportStatementBlock* block,
										const ExportIndex& index1,
										const ExportIndex& index2,
										const ExportIndex& tmp_index,
										const ExportVariable& _rk_A,
										const ExportVariable& Ah,
										const ExportVariable& C,
										bool evaluateB,
										bool DERIVATIVES );


		/** Exports the evaluation of the states at a specific stage.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *	@param[in] Ah				The matrix A of the IRK method, multiplied by the step size h.
		 *	@param[in] index			The loop index, defining the stage.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue evaluateStatesImplicitSystem( 	ExportStatementBlock* block,
											const ExportVariable& Ah,
											const ExportVariable& C,
											const ExportIndex& stage,
											const ExportIndex& i,
											const ExportIndex& j );


		/** Exports the evaluation of the right-hand side of the linear system at a specific stage.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *	@param[in] index			The loop index, defining the stage.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue evaluateRhsImplicitSystem( 	ExportStatementBlock* block,
														const ExportIndex& stage );


		/** Initializes export of a tailored integrator.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue setup( );


	protected:
		


    protected:
    

};


//
// Create the integrator
//
inline DiagonallyImplicitRKExport* createDiagonallyImplicitRKExport(	UserInteraction* _userInteraction,
		const std::string &_commonHeaderName	)
{
	int sensGen;
	_userInteraction->get( DYNAMIC_SENSITIVITY, sensGen );
	if ( (ExportSensitivityType)sensGen == FORWARD ) {
		return new DiagonallyImplicitRKExport(_userInteraction, _commonHeaderName);
	}
	else {
		ACADOERROR( RET_INVALID_OPTION );
		return new DiagonallyImplicitRKExport(_userInteraction, _commonHeaderName);
	}
}


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_DIRK_EXPORT_HPP

// end of file.
