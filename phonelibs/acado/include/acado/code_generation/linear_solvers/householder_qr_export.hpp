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
 *    \file include/acado/code_generation/householder_qr_export.hpp
 *    \author Rien Quirynen
 */


#ifndef ACADO_TOOLKIT_EXPORT_HOUSE_QR_HPP
#define ACADO_TOOLKIT_EXPORT_HOUSE_QR_HPP

#include <acado/code_generation/linear_solvers/linear_solver_export.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Allows to export Householder QR Factorization for solving linear systems of specific dimensions.
 *
 *	\ingroup NumericalAlgorithms
 *
 *	The class ExportHouseholderQR allows to export Householder QR Factorization
 * 	for solving linear systems of specific dimensions.
 *
 *	\author Rien Quirynen
 */

class ExportHouseholderQR : public ExportLinearSolver
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
        ExportHouseholderQR(	UserInteraction* _userInteraction = 0,
							const std::string& _commonHeaderName = ""
							);

        /** Destructor. 
		 */
        virtual ~ExportHouseholderQR( );

		/** Initializes code export into given file.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue setup( );


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


		/** Returns the dimension of the auxiliary variables for the linear solver.
		 *
		 *  \return The dimension of the auxiliary variables for the linear solver.
		 */
		virtual ExportVariable getGlobalExportVariable( const uint factor ) const;


	//
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:


    protected:
		
		// DEFINITION OF THE EXPORTVARIABLES
		ExportVariable rk_temp;						/**< Variable that is used to store intermediate results that can be reused. */

};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_EXPORT_HOUSE_QR_HPP

// end of file.
