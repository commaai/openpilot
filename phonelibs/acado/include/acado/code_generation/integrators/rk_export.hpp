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
 *    \file include/acado/integrator/rk_export.hpp
 *    \author Rien Quirynen
 *    \date 2012
 */


#ifndef ACADO_TOOLKIT_RK_EXPORT_HPP
#define ACADO_TOOLKIT_RK_EXPORT_HPP

#include <acado/code_generation/integrators/integrator_export.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Allows to export a tailored Runge-Kutta integrator for fast model predictive control.
 *
 *	\ingroup NumericalAlgorithms
 *
 *	The class RungeKuttaExport allows to export a tailored Runge-Kutta integrator
 *	for fast model predictive control.
 *
 *	\author Rien Quirynen
 */
class RungeKuttaExport : public IntegratorExport
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
        RungeKuttaExport(	UserInteraction* _userInteraction = 0,
							const std::string& _commonHeaderName = ""
							);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
        RungeKuttaExport(	const RungeKuttaExport& arg
							);

        /** Destructor. 
		 */
        virtual ~RungeKuttaExport( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
		RungeKuttaExport& operator=(	const RungeKuttaExport& arg
										);


		/** Initializes export of a tailored integrator.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue setup( ) = 0;


		/** This routine initializes the matrices AA, bb and cc which
		 * 	form the Butcher Tableau. */
		returnValue initializeButcherTableau( const DMatrix& _AA, const DVector& _bb, const DVector& _cc );


		/** This routine checks the symmetry of the cc vector from the Butcher Tableau. */
		BooleanType checkSymmetry( const DVector& _cc );


		/** Assigns Differential Equation to be used by the integrator.
		 *
		 *	@param[in] rhs		Right-hand side expression.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		
		virtual returnValue setDifferentialEquation( const Expression& rhs ) = 0;


		/** Sets a polynomial NARX model to be used by the integrator.
		 *
		 *	@param[in] delay		The delay for the states in the NARX model.
		 *	@param[in] parms		The parameters defining the polynomial NARX model.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue setNARXmodel( const uint delay, const DMatrix& parms );


		/** Adds all data declarations of the auto-generated integrator to given list of declarations.
		 *
		 *	@param[in] declarations		List of declarations.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue getDataDeclarations(	ExportStatementBlock& declarations,
													ExportStruct dataStruct = ACADO_ANY
													) const = 0;


		/** Adds all function (forward) declarations of the auto-generated integrator to given list of declarations.
		 *
		 *	@param[in] declarations		List of declarations.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue getFunctionDeclarations(	ExportStatementBlock& declarations
														) const = 0;



		/** Exports source code of the auto-generated integrator into the given directory.
		 *
		 *	@param[in] code				Code block containing the auto-generated integrator.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue getCode(	ExportStatementBlock& code
										) = 0;
        
        
        /** This routine returns the number of stages of the Runge-Kutta integrator that will be exported.
         */
        uint getNumStages();
							
        
        /** Sets up the output with the grids for the different output functions.									\n
		*                                                                      										\n
		*  \param outputGrids_	  	The vector containing a grid for each output function.			  				\n
		*  \param rhs 	  	  		The expressions corresponding the output functions.								\n
		*                                                                      										\n
		*  \return SUCCESSFUL_RETURN
		*/
		virtual returnValue setupOutput(  const std::vector<Grid> outputGrids_,
									  	  const std::vector<Expression> rhs ) = 0;



	protected:

		/** Copies all class members from given object.
		 *
		 *	@param[in] arg		Right-hand side object.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue copy(	const RungeKuttaExport& arg
							);


    protected:
        
		ExportVariable rk_kkk;				/**< Variable containing intermediate results of the RK integrator. */

		DMatrix AA;							/**< This matrix defines the Runge-Kutta method to be exported. */
		DVector bb, cc;						/**< These vectors define the Runge-Kutta method to be exported. */
		
		BooleanType is_symmetric;			/**< Boolean defining whether a certain RK method is symmetric or not, which is important for backward sensitivity propagation. */

		uint numStages;						/**< This is the number of stages for the Runge-Kutta method. */
};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_RK_EXPORT_HPP

// end of file.
