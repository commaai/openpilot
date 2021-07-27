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
 *    \file include/acado/code_generation/integrators/irk_export.hpp
 *    \author Rien Quirynen
 *    \date 2012
 */


#ifndef ACADO_TOOLKIT_IRK_EXPORT_HPP
#define ACADO_TOOLKIT_IRK_EXPORT_HPP

#include <acado/code_generation/integrators/rk_export.hpp>
#include <acado/code_generation/linear_solvers/linear_solver_generation.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Allows to export a tailored implicit Runge-Kutta integrator for fast model predictive control.
 *
 *	\ingroup NumericalAlgorithms
 *
 *	The class ImplicitRungeKuttaExport allows to export a tailored implicit Runge-Kutta integrator
 *	for fast model predictive control.
 *
 *	\author Rien Quirynen
 */
class ImplicitRungeKuttaExport : public RungeKuttaExport
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
        ImplicitRungeKuttaExport(	UserInteraction* _userInteraction = 0,
							const std::string& _commonHeaderName = ""
							);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
        ImplicitRungeKuttaExport(	const ImplicitRungeKuttaExport& arg
							);

        /** Destructor. 
		 */
        virtual ~ImplicitRungeKuttaExport( );


		/** Assignment operator (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
		ImplicitRungeKuttaExport& operator=(	const ImplicitRungeKuttaExport& arg
										);


		/** Initializes export of a tailored integrator.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue setup( );


		/** This routine sets the eigenvalues of the inverse of the AA matrix. */
        returnValue setEigenvalues( const DMatrix& _eig );

		/** This routine sets the transformation matrices, defined by the inverse of the AA matrix. */
        returnValue setSimplifiedTransformations( const DMatrix& _transf1, const DMatrix& _transf2 );
        returnValue setSimplifiedTransformations( const DMatrix& _transf1, const DMatrix& _transf2, const DMatrix& _transf1_T, const DMatrix& _transf2_T );

		/** This routine sets the transformation matrices, defined by the inverse of the AA matrix. */
        returnValue setSingleTransformations( const double _tau, const DVector& _low_tria, const DMatrix& _transf1, const DMatrix& _transf2 );
        returnValue setSingleTransformations( const double _tau, const DVector& _low_tria, const DMatrix& _transf1, const DMatrix& _transf2, const DMatrix& _transf1_T, const DMatrix& _transf2_T );


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
							
        
        /** Sets up the output with the grids for the different output functions.									\n
		*                                                                      										\n
		*  \param outputGrids_	  	The vector containing a grid for each output function.			  				\n
		*  \param rhs 	  	  		The expressions corresponding the output functions.								\n
		*                                                                      										\n
		*  \return SUCCESSFUL_RETURN
		*/
		virtual returnValue setupOutput( const std::vector<Grid> outputGrids_,
									  const std::vector<Expression> rhs );


		/** Sets up the output with the grids for the different output functions.									\n
		 *                                                                      										\n
		 *  \param outputGrids_	  		The vector containing a grid for each output function.			  			\n
		 *  \param _outputNames 	  		The names of the output functions.									  		\n
		 *  \param _diffs_outputNames 	The names of the functions, evaluating the derivatives of the outputs.		\n
		 *  \param _dims_output 			The dimensions of the output functions.										\n
		 *                                                                      										\n
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue setupOutput(  const std::vector<Grid> outputGrids_,
									  	  const std::vector<std::string> _outputNames,
									  	  const std::vector<std::string> _diffs_outputNames,
										  const std::vector<uint> _dims_output );


		/** Sets up the output with the grids for the different output functions.										\n
		 *                                                                      										\n
		 *  \param outputGrids_	  			The vector containing a grid for each output function.			  			\n
		 *  \param _outputNames 	  		The names of the output functions.									  		\n
		 *  \param _diffs_outputNames 		The names of the functions, evaluating the derivatives of the outputs.		\n
		 *  \param _dims_output 			The dimensions of the output functions.										\n
		 *  \param _outputDependencies		A separate dependency matrix for each output.								\n
		 *                                                                      										\n
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue setupOutput(  const std::vector<Grid> outputGrids_,
									  	  const std::vector<std::string> _outputNames,
									  	  const std::vector<std::string> _diffs_outputNames,
										  const std::vector<uint> _dims_output,
										  const std::vector<DMatrix> _outputDependencies );
        

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


		/** Initializes the matrix DD, which is used to extrapolate the variables of the IRK method to the next step.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue initializeDDMatrix( );
		
		
		/** Initializes the matrix coeffs, containing coefficients of polynomials that are used to evaluate the 
		 * 	continuous output (see evaluatePolynomial).
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue initializeCoefficients( );
		
		
		/** Recursive function that helps with the computation of the coefficients of polynomials that are used to evaluate the 
		 * 	continuous output (see initializeCoefficients), by computing the correct combinations of elements of the vector
		 * 	cc from the Butcher table.
		 *
		 * 	@param[in] cVec			The vector with all the elements of the vector cc from the Butcher table, of which combinations
		 * 							are computed in a recursive way.
		 * 	@param[in] index		An index of the vector cVec which denotes the relevant part for this invocation.
		 * 	@param[in] numEls		The number of elements in the combination.
		 * 
		 *	\return SUCCESSFUL_RETURN
		 */
		DVector computeCombinations( const DVector& cVec, uint index, uint numEls );
		
		
		/** Returns the coefficients of the polynomial, representing the continuous output of the integrator.
		 *
		 *	@param[in] time				The point in the interval (0,1] for which the coefficients are returned.
		 *
		 *	\return Coefficients of the polynomial, corresponding the given grid point
		 */
		DVector evaluatePolynomial( double time );
		
		
		/** Returns the coefficients of the derived polynomial, representing the derivative of the continuous output with respect to time.
		 *
		 *	@param[in] time				The point in the interval (0,1] for which the coefficients are returned.
		 *
		 *	\return Coefficients of the polynomial, corresponding the given grid point
		 */
		DVector evaluateDerivedPolynomial( double time );


		/** Exports the evaluation of the coefficients of the polynomial, representing the continuous output of the integrator.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *	@param[in] variable			The variable containing the coefficients of the polynomial.
		 *	@param[in] grid				The variable containing the grid point for the specific output.
		 *	@param[in] h				The integration step size.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue evaluatePolynomial( ExportStatementBlock& block, 
										const ExportVariable& variable, 
										const ExportVariable& grid, 
										const std::string& h );


		/** Exports the evaluation of the coefficients of the derived polynomial, representing the derivative of the continuous output with respect to time.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *	@param[in] variable			The variable containing the coefficients of the polynomial.
		 *	@param[in] grid				The variable containing the grid point for the specific output.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue evaluateDerivedPolynomial( ExportStatementBlock& block,
										const ExportVariable& variable,
										const ExportVariable& grid );
		
		
		/** Returns the coefficients of the polynomial for the complete grid of the output, corresponding a certain index.
		 *
		 *	@param[in] index	The index of the continuous output for which the coefficients are returned.
		 *
		 *	\return Coefficients of the polynomial, corresponding the given continuous output
		 */
		DMatrix evaluatePolynomial( uint index );


		/** Returns the coefficients of the derived polynomial for the complete grid of the output, corresponding a certain index.
		 *
		 *	@param[in] index	The index of the continuous output for which the coefficients are returned.
		 *
		 *	\return Coefficients of the derived polynomial, corresponding the given continuous output
		 */
		DMatrix evaluateDerivedPolynomial( uint index );


		/** Divide the total number of measurements over the different integration steps.
		 *
		 *	@param[in] index	The index of the continuous output for which the division of measurements is returned.
		 *
		 *	\return The division of measurements over the integration steps, corresponding the given continuous output.
		 */
		DVector divideMeasurements( uint index );


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


		/** Forms a constant linear system matrix for the collocation equations, given a constant jacobian and mass matrix.
		 *
		 *	@param[in] jacobian			given constant Jacobian matrix
		 *	@param[in] mass				given constant mass matrix
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual DMatrix formMatrix( const DMatrix& mass, const DMatrix& jacobian );


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


		/** Exports the evaluation of the states at a specific stage.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *	@param[in] Ah				The matrix A of the IRK method, multiplied by the step size h.
		 *	@param[in] index			The loop index, defining the stage.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue evaluateStatesOutputSystem( 	ExportStatementBlock* block,
													const ExportVariable& Ah,
													const ExportIndex& stage );


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
										const ExportIndex& k_index,
										const ExportVariable& _rk_A,
										const ExportVariable& Ah,
										const ExportVariable& C,
										bool evaluateB,
										bool DERIVATIVES );


		/** Prepares the structures to evaluate the continuous output and exports the resulting definitions.
		 *
		 *	@param[in] code				The block to which the code will be exported.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue prepareOutputEvaluation( 	ExportStatementBlock& code );


		/** Exports the necessary code for the computation of the continuous output.
		 *
		 *	@param[in] block			The block to which the code will be exported.
		 *	@param[in] tmp_meas			The number of measurements in the current integration step (in case of an online grid).
		 *	@param[in] rk_tPrev			The time point, defining the beginning of the current integration step (in case of an online grid).
		 *	@param[in] time_tmp			A variable used for time transformations (in case of an online grid).
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		returnValue generateOutput( 	ExportStatementBlock* block,
										const ExportIndex& index0,
										const ExportIndex& index1,
										const ExportIndex& tmp_index1,
										const ExportIndex& tmp_index2,
										const ExportVariable& tmp_meas,
										const ExportVariable& time_tmp,
										const uint directions );


		/** Copies all class members from given object.
		 *
		 *	@param[in] arg		Right-hand side object.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue copy(	const ImplicitRungeKuttaExport& arg
							);
		
		
		/** Returns the performed number of Newton iterations.
		 * 
		 * 	\return The performed number of Newton iterations.
		 */
		uint getNumIts() const;
		
		
		/** Returns the performed number of Newton iterations for the initialization of the first step.
		 * 
		 * 	\return The performed number of Newton iterations for the initialization of the first step.
		 */
        uint getNumItsInit() const;


		/** Returns the largest global export variable.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
        virtual ExportVariable getAuxVariable() const;


    protected:
    
		bool REUSE;						/**< This boolean is true when the IFTR method is used instead of the IFT method. */
		bool CONTINUOUS_OUTPUT;			/**< This boolean is true when continuous output needs to be provided. */

		uint numIts;							/**< This is the performed number of Newton iterations. */
		uint numItsInit;						/**< This is the performed number of Newton iterations for the initialization of the first step. */

		uint NDX2;
		uint NVARS2;

		uint NVARS3;

		ExportLinearSolver* solver;				/**< This is the exported linear solver that is used by the implicit Runge-Kutta method. */

		DMatrix DD;								/**< This matrix is used for the initialization of the variables for the next integration step. */
		DMatrix coeffs;							/**< This matrix contains coefficients of polynomials that are used to evaluate the continuous output (see evaluatePolynomial). */

		DVector numDX_output;
		DVector numXA_output;
		DVector numVARS_output;

        
        // DEFINITION OF THE EXPORTVARIABLES
		ExportVariable 	rk_rhsOutputTemp;		/**< Variable containing intermediate results of evaluations of the right-hand side expression of an output function. */
		ExportVariable  rk_diffsOutputTemp;		/**< Variable containing intermediate results of evaluations of the derivatives of an output function. */
		ExportVariable 	rk_outH;				/**< Variable that is used for the evaluations of the continuous output. */
		ExportVariable 	rk_out;					/**< Variable that is used for the evaluations of the continuous output. */
		ExportVariable 	polynEvalVar;			/**< Local variable that is used for the evaluations of the continuous output. */
		
		ExportVariable stepsH;					/**< Variable defining the different integration step sizes in case of a non equidistant grid. */

		std::vector<ExportVariable> gridVariables;	/**< This vector contains an ExportVariable for the grid of each continuous output. */
		std::vector<uint> totalMeas;				/**< This vector contains the total number of measurements per output (per shooting or integration interval, depending on grid type). */

		std::vector<ExportVariable> rk_outputs;			/**< Variables containing the evaluations of the continuous output from the integrator. */
		std::vector<ExportVariable> polynVariables;		/**< Variables containing the coefficients for the polynomial. */
		std::vector<ExportVariable> polynDerVariables;	/**< Variables containing the coefficients for the derived polynomial. */
		std::vector<ExportVariable> numMeasVariables;	/**< Variables containing the number of measurements per integration interval. */
		std::vector<ExportIndex> numMeas;				/**< Indices containing the number of measurements that are already computed. */

		ExportVariable	rk_mat1;
		ExportVariable 	rk_dk1;

		ExportVariable	rk_A;					/**< Variable containing the matrix of the linear system. */
		ExportVariable	rk_b;					/**< Variable containing the right-hand side of the linear system. */
		ExportVariable  rk_auxSolver;			/**< Variable containing auxiliary values for the exported linear solver. */
		ExportVariable 	rk_rhsTemp;				/**< Variable containing intermediate results of evaluations of the right-hand side expression. */

		ExportAcadoFunction lin_output;
		ExportVariable	rk_mat3;
		ExportVariable 	rk_dk3;
		ExportVariable  rk_diffsTemp3;

		ExportVariable 	rk_diffK;
		ExportVariable	debug_mat;

		DMatrix eig;
		DMatrix simplified_transf1;
		DMatrix simplified_transf2;
		DMatrix simplified_transf1_T;
		DMatrix simplified_transf2_T;

		double tau;
		DVector low_tria;
		DMatrix single_transf1;
		DMatrix single_transf2;

		DMatrix single_transf1_T;
		DMatrix single_transf2_T;

};


CLOSE_NAMESPACE_ACADO


#include <acado/code_generation/integrators/irk_lifted_adjoint_export.hpp>
#include <acado/code_generation/integrators/irk_lifted_symmetric_export.hpp>
#include <acado/code_generation/integrators/irk_lifted_fob_export.hpp>
#include <acado/code_generation/integrators/irk_lifted_forward_export.hpp>
#include <acado/code_generation/integrators/irk_lifted_feedback_export.hpp>
#include <acado/code_generation/integrators/irk_symmetric_export.hpp>
#include <acado/code_generation/integrators/irk_forward_export.hpp>
#include <acado/code_generation/integrators/irk_export.ipp>


#endif  // ACADO_TOOLKIT_IRK_EXPORT_HPP

// end of file.
