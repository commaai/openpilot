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
 *    \file include/acado/code_generation/integrators/narx_export.hpp
 *    \author Rien Quirynen
 *    \date 2013
 */


#ifndef ACADO_TOOLKIT_NARX_EXPORT_HPP
#define ACADO_TOOLKIT_NARX_EXPORT_HPP

#include <acado/code_generation/integrators/discrete_export.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Allows to export a tailored polynomial NARX integrator for fast model predictive control.
 *
 *	\ingroup NumericalAlgorithms
 *
 *	The class NARXExport allows to export a tailored polynomial NARX integrator
 *	for fast model predictive control.
 *
 *	\author Rien Quirynen
 */
class NARXExport : public DiscreteTimeExport
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
        NARXExport(	UserInteraction* _userInteraction = 0,
							const std::string& _commonHeaderName = ""
							);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
        NARXExport(	const NARXExport& arg
							);

        /** Destructor. 
		 */
        virtual ~NARXExport( );


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


		/** Assigns the model to be used by the integrator.
		 *
		 *	@param[in] _rhs				Name of the function, evaluating the right-hand side.
		 *	@param[in] _diffs_rhs		Name of the function, evaluating the derivatives of the right-hand side.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */

		returnValue setModel( const std::string& _rhs, const std::string& _diffs_rhs );


		/** Adds all data declarations of the auto-generated integrator to given list of declarations.
		 *
		 *	@param[in] declarations		List of declarations.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue getDataDeclarations(	ExportStatementBlock& declarations,
													ExportStruct dataStruct = ACADO_ANY
													) const;


		/** Exports the code needed to update the sensitivities of the states, defined by the linear input system.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue updateInputSystem( 	ExportStatementBlock* block,
										const ExportIndex& index1,
										const ExportIndex& index2,
										const ExportIndex& tmp_index  	);


		/** Exports the code needed to update the sensitivities of the states defined by the nonlinear part.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue updateImplicitSystem( 	ExportStatementBlock* block,
											const ExportIndex& index1,
											const ExportIndex& index2,
											const ExportIndex& tmp_index  	);


		/** Exports the code needed to update the sensitivities of the states, defined by the linear output system.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue updateOutputSystem( 	ExportStatementBlock* block,
													const ExportIndex& index1,
													const ExportIndex& index2,
													const ExportIndex& tmp_index  	);


		/** Exports the code needed to propagate the sensitivities of the states, defined by the linear input system.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue propagateInputSystem( 	ExportStatementBlock* block,
											const ExportIndex& index1,
											const ExportIndex& index2,
											const ExportIndex& index3,
											const ExportIndex& tmp_index  	);


		/** Exports the code needed to propagate the sensitivities of the states defined by the nonlinear part.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue propagateImplicitSystem( 	ExportStatementBlock* block,
												const ExportIndex& index1,
												const ExportIndex& index2,
												const ExportIndex& index3,
												const ExportIndex& tmp_index  	);


		/** Exports the code needed to propagate the sensitivities of the states, defined by the linear output system.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue propagateOutputSystem( 	ExportStatementBlock* block,
													const ExportIndex& index1,
													const ExportIndex& index2,
													const ExportIndex& index3,
													const ExportIndex& tmp_index  	);


		/** Sets a polynomial NARX model to be used by the integrator.
		 *
		 *	@param[in] delay		The delay for the states in the NARX model.
		 *	@param[in] parms		The parameters defining the polynomial NARX model.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */

		returnValue setNARXmodel( const uint _delay, const DMatrix& _parms );


		/** .
		 *
		 *	@param[in] 		.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue setLinearOutput( const DMatrix& M3, const DMatrix& A3, const Expression& rhs );


		/** .
		 *
		 *	@param[in] 		.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue setLinearOutput( const DMatrix& M3, const DMatrix& A3, const std::string& _rhs3, const std::string& _diffs_rhs3 );


	protected:


		/** Prepares a function that evaluates the complete right-hand side.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue prepareFullRhs( );


		/** ..
		 *
		 */
		returnValue formNARXpolynomial( const uint num, const uint order, uint& base, const uint index, IntermediateState& result );


    protected:

		uint delay;
		DMatrix parms;

};

IntegratorExport* createNARXExport(	UserInteraction* _userInteraction,
													const std::string &_commonHeaderName);


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_NARX_EXPORT_HPP

// end of file.
