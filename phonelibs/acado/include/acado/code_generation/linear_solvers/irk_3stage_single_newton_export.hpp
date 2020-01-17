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
 *    \file include/acado/code_generation/irk_3stage_single_newton_export.hpp
 *    \author Rien Quirynen
 */


#ifndef ACADO_TOOLKIT_EXPORT_IRK_3STAGE_SINGLE_SOLVER_HPP
#define ACADO_TOOLKIT_EXPORT_IRK_3STAGE_SINGLE_SOLVER_HPP

#include <acado/code_generation/linear_solvers/gaussian_elimination_export.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Allows to export a tailored IRK solver based on Gaussian elimination of specific dimensions.
 *
 *	\ingroup NumericalAlgorithms
 *
 *	The class ExportIRK3StageSolver allows to export a tailored IRK solver
 *	based on Gaussian elimination of specific dimensions.
 *
 *	\author Rien Quirynen
 */

class ExportIRK3StageSingleNewton : public ExportGaussElim
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
        ExportIRK3StageSingleNewton(	UserInteraction* _userInteraction = 0,
							const std::string& _commonHeaderName = ""
							);

        /** Destructor. */
        virtual ~ExportIRK3StageSingleNewton( );


		/** Initializes code export into given file.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue setup( );

		/** This routine sets the transformation matrices, defined by the inverse of the AA matrix. */
        returnValue setTransformations( const double _tau, const DVector& _low_tria, const DMatrix& _transf1, const DMatrix& _transf2, const DMatrix& _transf1_T, const DMatrix& _transf2_T );

		/** This routine sets the step size used in the IRK method. */
        returnValue setStepSize( double _stepsize );


		/** Exports source code of the auto-generated algorithm into the given directory.
		 *
		 *	@param[in] code				Code block containing the auto-generated algorithm.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue performTransformation(	ExportStatementBlock& code, const ExportVariable& from, const ExportVariable& to, const ExportVariable& transf, const ExportIndex& index );


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


		returnValue setImplicit( BooleanType _implicit );


	//
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:


		const std::string getNameSubSolveFunction();
		const std::string getNameSubSolveReuseFunction();
		const std::string getNameSubSolveTransposeReuseFunction();


    protected:

		BooleanType implicit;
		double stepsize;
		double tau;
		DVector low_tria;
		DMatrix transf1;
		DMatrix transf2;
		DMatrix transf1_T;
		DMatrix transf2_T;

		// DEFINITION OF THE EXPORTVARIABLES
		ExportVariable A_full;						/**< Variable containing the matrix for the complete linear system. */
		ExportVariable I_full;						/**< Variable containing the matrix for the complete linear system. */
		ExportVariable b_full;						/**< Variable containing the right-hand side of the complete linear system and it will also contain the solution. */
		ExportVariable rk_perm_full;				/**< Variable containing the order of the rows. */

		ExportFunction solve_full;					/**< Function that solves the complete linear system. */
		ExportFunction solveReuse_full;				/**< Function that solves a complete linear system with the same matrix, reusing previous results. */

		ExportFunction solveReuseTranspose_full;	/**< Function that solves a complete linear system with the same matrix, reusing previous results. */
		ExportVariable b_full_trans;				/**< Variable containing the right-hand side of the complete linear system and it will also contain the solution. */
		ExportVariable b_mem_trans;					/**< Variable containing the right-hand side for the linear subsystems. */

		ExportVariable A_mem;					/**< Variable containing the factorized matrix of the linear subsystems. */
		ExportVariable b_mem;					/**< Variable containing the right-hand side for the linear subsystems. */

};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_EXPORT_IRK_3STAGE_SINGLE_SOLVER_HPP

// end of file.
