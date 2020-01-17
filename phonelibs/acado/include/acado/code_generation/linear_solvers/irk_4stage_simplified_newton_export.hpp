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
 *    \file include/acado/code_generation/irk_4stage_simplified_newton_export.hpp
 *    \author Rien Quirynen
 */


#ifndef ACADO_TOOLKIT_EXPORT_IRK_4STAGE_SIMPLIFIED_SOLVER_HPP
#define ACADO_TOOLKIT_EXPORT_IRK_4STAGE_SIMPLIFIED_SOLVER_HPP

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

class ExportIRK4StageSimplifiedNewton : public ExportGaussElim
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
        ExportIRK4StageSimplifiedNewton(	UserInteraction* _userInteraction = 0,
							const std::string& _commonHeaderName = ""
							);

        /** Destructor. */
        virtual ~ExportIRK4StageSimplifiedNewton( );


		/** Initializes code export into given file.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue setup( );


		/** This routine sets the eigenvalues of the inverse of the AA matrix. */
        returnValue setEigenvalues( const DMatrix& _eig );

		/** This routine sets the transformation matrices, defined by the inverse of the AA matrix. */
        returnValue setTransformations( const DMatrix& _transf1, const DMatrix& _transf2, const DMatrix& _transf1_T, const DMatrix& _transf2_T );

		/** This routine sets the step size used in the IRK method. */
        returnValue setStepSize( double _stepsize );


		/** Exports source code of the auto-generated algorithm into the given directory.
		 *
		 *	@param[in] code				Code block containing the auto-generated algorithm.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue transformRightHandSide(	ExportStatementBlock& code, const ExportVariable& b_mem1, const ExportVariable& b_mem2, const ExportVariable& b_full_, const ExportVariable& transf_, const ExportIndex& index, const bool transpose );


		/** Exports source code of the auto-generated algorithm into the given directory.
		 *
		 *	@param[in] code				Code block containing the auto-generated algorithm.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue transformSolution(	ExportStatementBlock& code, const ExportVariable& b_mem1, const ExportVariable& b_mem2, const ExportVariable& b_full_, const ExportVariable& transf_, const ExportIndex& index, const bool transpose );


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
		returnValue appendVariableNames( std::stringstream& string );


		returnValue setImplicit( BooleanType _implicit );


	//
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:


		const std::string getNameSolveComplexFunction();
		const std::string getNameSolveComplexReuseFunction();
		const std::string getNameSolveComplexTransposeReuseFunction();


    protected:

		BooleanType implicit;
		double stepsize;
		DMatrix eig;
		DMatrix transf1;
		DMatrix transf2;
		DMatrix transf1_T;
		DMatrix transf2_T;

		// DEFINITION OF THE EXPORTVARIABLES
		ExportVariable determinant_complex;			/**< Variable containing the matrix determinant. */
		ExportVariable rk_swap_complex;				/**< Variable that is used to swap rows for pivoting. */
		ExportVariable rk_bPerm_complex;			/**< Variable containing the reordered right-hand side. */
		ExportVariable rk_bPerm_complex_trans;		/**< Variable containing the reordered right-hand side. */

		ExportVariable A_complex;					/**< Variable containing the matrix of the complex linear system. */
		ExportVariable b_complex;					/**< Variable containing the right-hand side of the complex linear system and it will also contain the solution. */
		ExportVariable b_complex_trans;				/**< Variable containing the right-hand side of the complex linear system and it will also contain the solution. */
		ExportVariable rk_perm_complex;			/**< Variable containing the order of the rows. */
		
		ExportFunction solve_complex;				/**< Function that solves the complex linear system. */
		ExportFunction solveReuse_complex;			/**< Function that solves a complex linear system with the same matrix, reusing previous results. */
		ExportFunction solveReuse_complexTranspose;	/**< Function that solves a complex linear system with the same matrix, reusing previous results. */


		ExportVariable A_full;						/**< Variable containing the matrix for the complete linear system. */
		ExportVariable I_full;						/**< Variable containing the matrix for the complete linear system. */
		ExportVariable b_full;						/**< Variable containing the right-hand side of the complete linear system and it will also contain the solution. */
		ExportVariable b_full_trans;				/**< Variable containing the right-hand side of the complete linear system and it will also contain the solution. */
		ExportVariable rk_perm_full;				/**< Variable containing the order of the rows. */

		ExportFunction solve_full;					/**< Function that solves the complete linear system. */
		ExportFunction solveReuse_full;				/**< Function that solves a complete linear system with the same matrix, reusing previous results. */
		ExportFunction solveReuseTranspose_full;	/**< Function that solves a complete linear system with the same matrix, reusing previous results. */


		ExportVariable A_mem_complex1;				/**< Variable containing the factorized matrix of the complex linear system. */
		ExportVariable b_mem_complex1;				/**< Variable containing the right-hand side for the complex linear system. */
		ExportVariable A_mem_complex2;				/**< Variable containing the factorized matrix of the complex linear system. */
		ExportVariable b_mem_complex2;				/**< Variable containing the right-hand side for the complex linear system. */

		ExportVariable b_mem_complex1_trans;		/**< Variable containing the right-hand side for the complex linear system. */
		ExportVariable b_mem_complex2_trans;		/**< Variable containing the right-hand side for the complex linear system. */

};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_EXPORT_IRK_4STAGE_SIMPLIFIED_SOLVER_HPP

// end of file.
