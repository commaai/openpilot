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
 *    \file include/acado/code_generation/linear_solver_export.hpp
 *    \author Rien Quirynen
 */


#ifndef ACADO_TOOLKIT_EXPORT_LINEAR_SOLVER_HPP
#define ACADO_TOOLKIT_EXPORT_LINEAR_SOLVER_HPP

#include <acado/code_generation/export_acado_function.hpp>
#include <acado/code_generation/export_variable.hpp>
#include <acado/code_generation/export_algorithm.hpp>
#include <acado/code_generation/export_arithmetic_statement.hpp>
#include <acado/code_generation/export_function_call.hpp>
#include <acado/code_generation/export_for_loop.hpp>
#include <acado/function/function.hpp>

#include <acado/user_interaction/options.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Allows to export automatically generated algorithms for solving linear systems of specific dimensions.
 *
 *	\ingroup NumericalAlgorithms
 *
 *	The class ExportLinearSolver allows to export automatically generated 
 *	algorithms for solving linear systems of specific dimensions.
 *
 *	\author Rien Quirynen
 */

class ExportLinearSolver : public ExportAlgorithm
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
        ExportLinearSolver(	UserInteraction* _userInteraction = 0,
							const std::string& _commonHeaderName = ""
							);

        /** Destructor. */
        virtual ~ExportLinearSolver( );

		/** Initializes code export into given file.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue setup( ) = 0;


		/** Initializes the different parameters of the linear solver that will be exported.
		 *
		 * 	@param[in] newDim		The dimensions of the linear system.
		 * 	@param[in] reuse		A boolean that is true when more than one system of linear equations with the same
		 * 							matrix needs to be solved. This means that an algorithm will be exported expecially 
		 * 							for this case, with extra optimizations by reusing as many intermediate results as possible.
		 * 	@param[in] unrolling	A boolean that is true when the exported code for the linear solver needs to be unrolled
		 * 							completely.
		 * 
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue init(	const uint newDim,
							const bool& reuse = true,
							const bool& unrolling = false
							);

		returnValue init(	const uint newDim,
							const uint _nRightHandSides,
							const bool& reuse = true,
							const bool& unrolling = false
							);


		/** Initializes the different parameters of the linear solver that will be exported.
		 *
		 * 	@param[in] newDim		The dimensions of the linear system.
		 * 	@param[in] reuse		A boolean that is true when more than one system of linear equations with the same
		 * 							matrix needs to be solved. This means that an algorithm will be exported expecially 
		 * 							for this case, with extra optimizations by reusing as many intermediate results as possible.
		 * 	@param[in] unrolling	A boolean that is true when the exported code for the linear solver needs to be unrolled
		 * 							completely.
		 * 	@param[in] newId		The new identifier for this linear solver to be exported.
		 * 
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue init(	const uint newDim,
							const bool& reuse,
							const bool& unrolling,
							const std::string& newId
							);

		/** \todo DOC */
		returnValue init(	unsigned _nRows,
							unsigned _nCols,
							unsigned _nBacksolves,
							bool _reuse,
							bool _unroll,
							const std::string& _id
							);

		/** \todo DOC */
		returnValue init(	unsigned _nRows,
							unsigned _nCols,
							unsigned _nBacksolves,
							unsigned _nRightHandSides,
							bool _reuse,
							bool _unroll,
							const std::string& _id
							);

		/** Adds all data declarations of the auto-generated algorithm to given list of declarations.
		 *
		 *	@param[in] declarations		List of declarations.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue getDataDeclarations(	ExportStatementBlock& declarations,
													ExportStruct dataStruct = ACADO_ANY
													) const = 0;


		/** Adds all function (forward) declarations of the auto-generated algorithm to given list of declarations.
		 *
		 *	@param[in] declarations		List of declarations.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue getFunctionDeclarations(	ExportStatementBlock& declarations
														) const = 0;


		/** Exports source code of the auto-generated algorithm into the given directory.
		 *
		 *	@param[in] code				Code block containing the auto-generated algorithm.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue getCode(	ExportStatementBlock& code
										) = 0;


		/** Appends the names of the used variables to a given stringstream.
		 *
		 *	@param[in] string				The string to which the names of the used variables are appended.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue appendVariableNames( std::stringstream& string ) = 0;


		/** Returns the dimensions of the linear system.
		 *
		 *  \return The dimensions of the linear system.
		 */
		uint getDim() const;


		/** Returns the dimension of the auxiliary variables for the linear solver.
		 *
		 *  \return The dimension of the auxiliary variables for the linear solver.
		 */
		virtual ExportVariable getGlobalExportVariable( const uint factor ) const;
		
		
		/** Returns a boolean that is true when an extra algorithm will be exported for reuse.
		 *
		 *  \return A boolean that is true when an extra algorithm will be exported for reuse.
		 */
		bool getReuse() const;
		
		
		/** Sets the boolean that is true when an extra algorithm will be exported for reuse.
		 * 
		 * 	@param[in] reuse		The new value of this boolean.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue setReuse( const bool& reuse );
		
		
		/** Returns a boolean that is true when an extra algorithm will be exported for solving a transposed linear system based on reuse.
		 *
		 *  \return A boolean that is true when an extra algorithm will be exported for solving a transposed linear system based on reuse.
		 */
		bool getTranspose() const;


		/** Sets the boolean that is true when an extra algorithm will be exported for solving a transposed linear system based on reuse.
		 *
		 * 	@param[in] transpose		The new value of this boolean.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue setTranspose( const bool& transpose );


		/** Returns a boolean that is true when the exported code for the linear solver needs to be unrolled
		 * 	completely.
		 *
		 *  \return A boolean that is true when the exported code for the linear solver needs to be unrolled
		 * 			completely.
		 */
		bool getUnrolling() const;
		
		
		/** Sets the boolean that is true when the exported code for the linear solver needs to be unrolled
		 * 	completely.
		 * 
		 * 	@param[in] unrolling	The new value of this boolean.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue setUnrolling( const bool& unrolling );
		
		
		/** Returns a string containing the name of the function which is exported to solve the linear system.
		 *
		 *  \return A string containing the name of the function which is exported to solve the linear system.
		 */
		const std::string getNameSolveFunction(); 
		
		
		/** Returns a string containing the name of the function which is exported to solve the linear system with
		 * 	the reuse of previous results.
		 *
		 *  \return A string containing the name of the function which is exported to solve the linear system with
		 * 			the reuse of previous results.
		 */
		const std::string getNameSolveReuseFunction(); 
		const std::string getNameSolveTransposeReuseFunction();

	//
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:


    protected:
    
		std::string identifier;							/**< This identifier allows the use of more than one exported linear solver. */
    
		bool UNROLLING;						/**< The boolean that defines the unrolling. */
		bool REUSE;							/**< The boolean that defines the reuse. */
		bool TRANSPOSE;
		uint dim;									/**< The dimensions of the linear system. */
		
		unsigned nRows;								/**< Number of rows of matrix A. */
		unsigned nCols;								/**< Number of columns of matrix A. */
		unsigned nBacksolves;						/**< Number of back-solves. */
		unsigned nRightHandSides;					/**< Number of back-solves. */

		// DEFINITION OF THE EXPORTVARIABLES
		ExportVariable A;							/**< Variable containing the matrix of the linear system. */
		ExportVariable b;							/**< Variable containing the right-hand side of the linear system and it will also contain the solution. */
		
		ExportFunction solve;						/**< Function that solves the linear system. */
		ExportFunction solveTriangular;				/**< Function that solves the upper-triangular system. */
		ExportFunction solveReuse;					/**< Function that solves a linear system with the same matrix, reusing previous results. */
		ExportFunction solveReuseTranspose;			/**< Function that solves a transposed linear system with the same matrix, reusing previous results. */

		ExportVariable determinant;					/**< Variable containing the matrix determinant. */
};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_EXPORT_LINEAR_SOLVER_HPP

// end of file.
