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
 *    \file include/acado/integrator/export_algorithm.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 2010-2011
 */


#ifndef ACADO_TOOLKIT_EXPORT_ALGORITHM_HPP
#define ACADO_TOOLKIT_EXPORT_ALGORITHM_HPP

#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/algorithmic_base.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>

#include <acado/code_generation/export_variable.hpp>
#include <acado/code_generation/export_function.hpp>
#include <acado/code_generation/export_acado_function.hpp>
#include <acado/code_generation/export_arithmetic_statement.hpp>
#include <acado/code_generation/export_function_call.hpp>
#include <acado/code_generation/export_for_loop.hpp>

BEGIN_NAMESPACE_ACADO

/** 
 *	\brief Allows to export automatically generated algorithms for fast model predictive control
 *
 *	\ingroup NumericalAlgorithms
 *
 *	The class ExportAlgorithm allows to export automatically generated 
 *	algorithms for fast model predictive control.
 *
 *	\author Hans Joachim Ferreau, Milan Vukov, Boris Houska
 */
class ExportAlgorithm : public AlgorithmicBase
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
        ExportAlgorithm(	UserInteraction* _userInteraction = 0,
							const std::string& _commonHeaderName = std::string()
							);

        /** Destructor. */
        virtual ~ExportAlgorithm( );

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


		/** Sets the variables dimensions (ODE).
		 *
		 *	@param[in] _NX		New number of differential states.
		 *	@param[in] _NU		New number of control inputs.
		 *	@param[in] _NP		New number of parameters.
		 *	@param[in] _NI		New number of control intervals. (using _N resulted in a strange error when compiling with cygwin!)
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue setDimensions(	uint _NX = 0,
									uint _NU = 0,
									uint _NP = 0,
									uint _NI = 0,
									uint _NOD = 0
									);


		/** Sets the variables dimensions (DAE).
		 *
		 *	@param[in] _NX		New number of differential states.
		 *	@param[in] _NDX		New number of differential states derivatives.
		 *	@param[in] _NXA		New number of algebraic states.
		 *	@param[in] _NU		New number of control inputs.
		 *	@param[in] _NP		New number of parameters.
		 *	@param[in] _NI		New number of control intervals.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue setDimensions(	uint _NX,
									uint _NDX,
									uint _NXA,
									uint _NU,
									uint _NP,
									uint _NI,
									uint _NOD
									);


		/** Returns number of differential states.
		 *
		 *  \return Number of differential states
		 */
		uint getNX( ) const;
		
		/** Returns number of algebraic states.
		 *
		 *  \return Number of algebraic states
		 */
		uint getNXA( ) const;

		/** Returns the number of differential states derivatives.
		 * 
		 *  \return The requested number of differential state derivatives
		 */
		uint getNDX( ) const;

		/** Returns number of control inputs.
		 *
		 *  \return Number of control inputs
		 */
		uint getNU( ) const;

		/** Returns number of parameters.
		 *
		 *  \return Number of parameters
		 */
		uint getNP( ) const;

		/** Returns number of parameters.
		 *
		 *  \return Number of parameters
		 */
		uint getNOD( ) const;

		/** Returns number of control intervals.
		 *
		 *  \return Number of control intervals
		 */
		uint getN( ) const;
		
		void setNY( uint NY_ );
		uint getNY( ) const;

		void setNYN( uint NYN_ );
		uint getNYN( ) const;

    protected:

		uint NX;							/**< Number of differential states. */
		uint NDX;							/**< Number of differential states derivatives. */
		uint NXA;							/**< Number of algebraic states. */
		uint NU;							/**< Number of control inputs. */
		uint NP;							/**< Number of parameters. */
		uint NOD;							/**< Number of "online data" values. */
		uint N;								/**< Number of control intervals. */

		uint NY;							/**< Number of references/measurements, nodes 0,..., N - 1. */
		uint NYN;							/**< Number of references/measurements, node N. */

		std::string commonHeaderName;		/**< Name of common header file. */
};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_EXPORT_ALGORITHM_HPP

// end of file.
