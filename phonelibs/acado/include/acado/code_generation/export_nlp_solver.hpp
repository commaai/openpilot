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
 *    \file include/acado/code_generation/export_nlp_solver.hpp
 *    \author Milan Vukov, Rien Quirynen
 *    \date 2012 - 2013
 */

#ifndef ACADO_TOOLKIT_EXPORT_NLP_SOLVER_HPP
#define ACADO_TOOLKIT_EXPORT_NLP_SOLVER_HPP

#include <acado/utils/acado_utils.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>

#include <acado/code_generation/export_algorithm_factory.hpp>
#include <acado/code_generation/integrators/integrator_export.hpp>

#include <acado/code_generation/export_cholesky_decomposition.hpp>
#include <acado/code_generation/linear_solvers/householder_qr_export.hpp>

BEGIN_NAMESPACE_ACADO

class OCP;
class Objective;

/** 
 *	\brief Base class for export of NLP/OCP solvers.
 *
 *	\ingroup NumericalAlgorithms
 *
 *	This base class is basically used to extract information from an OCP
 *	object and prepare low level structures. Later, a derived class is
 *	actually building the solver to solve an OCP problem.
 * 
 *	\author Milan Vukov
 *
 *	\note Based on code originally developed by Boris Houska and Hand Joachim Ferreau.
 */
class ExportNLPSolver : public ExportAlgorithm
{
public:

	/** Default constructor.
	 *
	 *	@param[in] _userInteraction		Pointer to corresponding user interface.
	 *	@param[in] _commonHeaderName	Name of common header file to be included.
	 */
	ExportNLPSolver(	UserInteraction* _userInteraction = 0,
						const std::string& _commonHeaderName = ""
						);

	/** Destructor. */
	virtual ~ExportNLPSolver( )
	{}

	/** Initializes export of an algorithm.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue setup( ) = 0;


	/** Assigns module for exporting a tailored integrator.
	 *
	 *	@param[in] _integrator	Integrator module.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	returnValue setIntegratorExport(	IntegratorExportPtr const _integrator
										);

	/** Assigns new constant for Levenberg-Marquardt regularization.
	 *
	 *	@param[in] _levenbergMarquardt		Non-negative constant for Levenberg-Marquardt regularization.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	returnValue setLevenbergMarquardt(	double _levenbergMarquardt
										);


	/** Adds all data declarations of the auto-generated condensing algorithm
	 *	to given list of declarations.
	 *
	 *	@param[in] declarations		List of declarations.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue getDataDeclarations(	ExportStatementBlock& declarations,
												ExportStruct dataStruct = ACADO_ANY
												) const;

	/** Adds all function (forward) declarations of the auto-generated condensing algorithm
	 *	to given list of declarations.
	 *
	 *	@param[in] declarations		List of declarations.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue getFunctionDeclarations(	ExportStatementBlock& declarations
													) const = 0;


	/** Exports source code of the auto-generated condensing algorithm
	 *  into the given directory.
	 *
	 *	@param[in] code				Code block containing the auto-generated condensing algorithm.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue getCode(	ExportStatementBlock& code
									) = 0;


	/** Returns number of variables in underlying QP.
	 *
	 *  \return Number of variables in underlying QP
	 */
	virtual unsigned getNumQPvars( ) const = 0;

	/** Returns whether a single shooting state discretization is used.
	 *
	 *	\return true  iff single shooting state discretization is used, \n
	 *	        false otherwise
	 */
	bool performsSingleShooting( ) const;

	/** Set objective function
	 *  \return SUCCESSFUL_RETURN, \n
	 *          RET_INITIALIZE_FIRST, \n
	 *          RET_INVALID_OBJECTIVE_FOR_CODE_EXPORT, \n
	 *			RET_INVALID_ARGUMENTS
	 * */
	returnValue setObjective(const Objective& _objective);
	returnValue setLSQObjective(const Objective& _objective);
	returnValue setGeneralObjective(const Objective& _objective);

	/** Set the "complex" path and point constraints
	 *  \return SUCCESSFUL_RETURN
	 * */
	returnValue setConstraints(const OCP& _ocp);

	/** Get the number of complex constraints - path + point constraints.
	 *  \return Number of complex constraints
	 * */
	unsigned getNumComplexConstraints( void );

    /** Get the number of path constraints.
     *  \return Number of path constraints
     * */
    unsigned getNumPathConstraints( void );

	/** Return type of weighting matrices.
	 *  \return Type of weighting matrices. */
	unsigned weightingMatricesType( void ) const;

	/** Indicates whether initial state is fixed. */
	bool initialStateFixed( ) const;

	/** Indicates whether linear terms in the objective are used. */
	bool usingLinearTerms() const;

protected:

	/** Setting up of a model simulation:
	 *   - model integration
	 *   - sensitivity generation
	 *
	 *  \return SUCCESSFUL_RETURN
	 */
	virtual returnValue setupSimulation( void );

	/** Initialization of member variables.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue setupVariables( ) = 0;

	/** Exports source code containing the multiplication routines of the algorithm.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue setupMultiplicationRoutines( ) = 0;

	/** Exports source code containing the evaluation routines of the algorithm. */
	virtual returnValue setupEvaluation( ) = 0;

	/** Setup of functions for evaluation of constraints. */
	virtual returnValue setupConstraintsEvaluation( ) = 0;

	/** Setup of functions for evaluation of auxiliary functions. */
	returnValue setupAuxiliaryFunctions();

	/** Setup the function for evaluating the actual objective value. */
	virtual returnValue setupGetObjective();

	/** Setup the function for evaluating the actual LSQ objective value. */
	virtual returnValue setupGetLSQObjective();

	/** Setup the function for evaluating the actual objective value. */
	virtual returnValue setupGetGeneralObjective();

	/** Setup of functions and variables for evaluation of arrival cost. */
	returnValue setupArrivalCostCalculation();

	/** Setup main initialization code for the solver */
	virtual returnValue setupInitialization();

protected:

	/** \name Evaluation of model dynamics. */
	/** @{ */

	/** Module for exporting a tailored integrator. */
	IntegratorExportPtr integrator;

	ExportFunction modelSimulation;

	ExportVariable state;
	ExportVariable x;
	ExportVariable z;
	ExportVariable u;
	ExportVariable od;
	ExportVariable d;

	ExportVariable evGx; // stack of sensitivities w.r.t. x
	ExportVariable evGu; // stack of sensitivities w.r.t. u

	/** @} */

	/** \name Evaluation of objective */
	/** @{ */

	/** Non-negative constant for Levenberg-Marquardt regularization. */
	double levenbergMarquardt;

	ExportVariable y, yN, Dy, DyN;

	// lagrange multipliers
	ExportVariable mu;

	ExportVariable objg, objS, objSEndTerm;
	ExportVariable objEvFx, objEvFu, objEvFxEnd; // aliasing
	ExportVariable objEvFxx, objEvFxu, objEvFuu, objEvFxxEnd; // aliasing

	ExportVariable objAuxVar, objValueIn, objValueOut;
	ExportAcadoFunction evaluateStageCost;
	ExportAcadoFunction evaluateTerminalCost;

	ExportVariable Q1, Q2;
	ExportVariable R1, R2;
	ExportVariable S1;
	ExportVariable QN1, QN2;

	ExportVariable objSlx, objSlu;

	bool diagonalH, diagonalHN;

	/** @} */

	/** \name Evaluation of box constraints*/
	/** @{ */
	VariablesGrid uBounds;
	VariablesGrid xBounds;
	/** @} */

	/** \name Evaluation of path constraints */
	/** @{ */
	unsigned dimPacH;
	ExportAcadoFunction evaluatePathConstraints;
	ExportVariable conAuxVar;
	ExportVariable conValueIn;
	ExportVariable conValueOut;

	DVector lbPathConValues, ubPathConValues;

	ExportVariable pacEvH;
	ExportVariable pacEvHx, pacEvHu, pacEvHxd;
	ExportVariable pacEvDDH;
	/** @} */

	/** \name Evaluation of point constraints */
	/** @{ */
	unsigned dimPocH;
	std::vector< std::shared_ptr< ExportAcadoFunction > > evaluatePointConstraints;
	DVector lbPointConValues, ubPointConValues;

	std::vector< DVector > pocLbStack, pocUbStack;

	ExportVariable pocEvH;
	ExportVariable pocEvHx, pocEvHu, pocEvHxd;
	/** @} */

	/** \name Auxiliary functions */
	/**  @{ */

	/** Main initialization function for the solver. */
	ExportFunction initialize;

	ExportFunction shiftStates;
	ExportFunction shiftControls;
	ExportFunction getObjective;
	ExportFunction initializeNodes;
	/** @} */

	/** \name Arrival cost related */
	/**  @{ */
	ExportFunction updateArrivalCost;

	ExportCholeskyDecomposition cholObjS;
	ExportCholeskyDecomposition cholSAC;

	ExportHouseholderQR acSolver;

	ExportVariable acA, acb, acP, acTmp;
	// acWL and acVL are assumed to be upper triangular matrices
	ExportVariable acWL, acVL, acHx, acHu, acXx, acXu, acXTilde, acHTilde;

	// Older stuff; TODO make this more unique
	ExportVariable SAC, xAC, DxAC;

	ExportFunction regularizeHessian;
	ExportFunction regularization;
	/** @} */

private:
	returnValue setupResidualVariables();
	returnValue setupObjectiveLinearTerms(const Objective& _objective);
};

/** Types of NLP/OCP solvers. */
enum ExportNLPType
{
	GAUSS_NEWTON_CONDENSED,
	GAUSS_NEWTON_CN2,
	GAUSS_NEWTON_BLOCK_QPDUNES,
	GAUSS_NEWTON_BLOCK_FORCES,
	GAUSS_NEWTON_CN2_FACTORIZATION,
	GAUSS_NEWTON_FORCES,
	GAUSS_NEWTON_QPDUNES,
	GAUSS_NEWTON_HPMPC,
    GAUSS_NEWTON_GENERIC,
	EXACT_HESSIAN_CN2,
	EXACT_HESSIAN_QPDUNES
};

/** Factory for creation of exported NLP/OCP solvers. */
typedef ExportAlgorithmFactory<ExportNLPSolver, ExportNLPType> NLPSolverFactory;

/** Shared pointer to an NLP solver. */
typedef std::shared_ptr< ExportNLPSolver > ExportNLPSolverPtr;

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_EXPORT_NLP_SOLVER_HPP
