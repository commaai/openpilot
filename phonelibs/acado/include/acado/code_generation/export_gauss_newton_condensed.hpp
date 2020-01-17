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
 *    \file include/acado/code_generation/export_gauss_newton_condensed.hpp
 *    \authors Boris Houska, Hans Joachim Ferreau, Milan Vukov
 *    \date 2010 - 2014
 */

#ifndef ACADO_TOOLKIT_EXPORT_GAUSS_NEWTON_CONDENSED_HPP
#define ACADO_TOOLKIT_EXPORT_GAUSS_NEWTON_CONDENSED_HPP

#include <acado/code_generation/export_nlp_solver.hpp>
#include <acado/code_generation/linear_solvers/export_cholesky_solver.hpp>

BEGIN_NAMESPACE_ACADO

/**
 *	\brief A class for export of Gauss-Newton condensed OCP solver
 *
 *	\ingroup NumericalAlgorithms
 *
 *	The class ExportGaussNewtonCondensed allows to export an OCP solver
 *	using the generalized Gauss-Newton method. The sparse QP is condensed
 *	and solved with qpOASES QP solver.
 *
 *	\authors Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */
class ExportGaussNewtonCondensed : public ExportNLPSolver
{
public:

	/** Default constructor.
	 *
	 *	@param[in] _userInteraction		Pointer to corresponding user interface.
	 *	@param[in] _commonHeaderName	Name of common header file to be included.
	 */
	ExportGaussNewtonCondensed(	UserInteraction* _userInteraction = 0,
								const std::string& _commonHeaderName = ""
								);

	/** Destructor.
	*/
	virtual ~ExportGaussNewtonCondensed( )
	{}

	/** Initializes export of an algorithm.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue setup( );

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
													) const;


	/** Exports source code of the auto-generated condensing algorithm
	 *  into the given directory.
	 *
	 *	@param[in] code				Code block containing the auto-generated condensing algorithm.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue getCode(	ExportStatementBlock& code
									);


	/** Returns number of variables in underlying QP.
	 *
	 *  \return Number of variables in underlying QP
	 */
	unsigned getNumQPvars( ) const;

	/** Returns number of bounds on differential states.
	 *
	 *  \return Number of bounds on differential states
	 */
	virtual unsigned getNumStateBounds( ) const;

protected:

	/** Setting up of an objective evaluation:
	 *   - functions and derivatives evaulation
	 *   - creating Hessians and gradients
	 *
	 *   \return SUCCESSFUL_RETURN
	 */
	virtual returnValue setupObjectiveEvaluation( void );

	/** Set-up evaluation of constraints
	 *   - box constraints on states and controls
	 *
	 *  \return SUCCESSFUL_RETURN
	 */
	virtual returnValue setupConstraintsEvaluation( void );

	/** Initialization of all member variables.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue setupVariables( );

	/** Exports source code containing the multiplication routines of the algorithm.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue setupMultiplicationRoutines( );

	/** Exports source code containing the evaluation routines of the algorithm.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue setupEvaluation( );

	/** Setup qpOASES interface. */
	virtual returnValue setupQPInterface( );

	/** Setup condensing routine variables and functions. */
	virtual returnValue setupCondensing( );

	/** Indicator for full condensing. */
	bool performFullCondensing( ) const;

private:

	/** Current state feedback. */
	ExportVariable x0;
	/** Current state feedback deviation. */
	ExportVariable Dx0;

	/** \name Objective evaluation */
	/** @{ */
	ExportFunction evaluateObjective;
	ExportFunction setObjQ1Q2;
	ExportFunction setObjR1R2;
	ExportFunction setObjQN1QN2;
	/** @} */

	/** \name Condensing functions and variables */
	/** @{ */
	ExportFunction condensePrep;
	ExportFunction condenseFdb;
	ExportFunction expand;

	ExportVariable T, E, QE, QGx, QDy, Qd;

	ExportVariable H00, H10, H11;
	ExportVariable g0, g1;

	ExportCholeskySolver cholSolver;

	std::vector< unsigned > xBoundsIdx;
	ExportVariable lbValues, ubValues;
	ExportVariable lbAValues, ubAValues;
	/** @} */

	/** \name Helper functions */
	/** @{ */
	ExportFunction multGxd;
	ExportFunction moveGxT;
	ExportFunction multGxGx;
	ExportFunction multGxGu;
	ExportFunction moveGuE;
	ExportFunction setBlockH11;
	ExportFunction setBlockH11_R1;
	ExportFunction zeroBlockH11;
	ExportFunction copyHTH;
	ExportFunction multQ1d;
	ExportFunction multQN1d;
	ExportFunction multRDy;
	ExportFunction multQDy;
	ExportFunction multEQDy;
	ExportFunction multQETGx;
	ExportFunction zeroBlockH10;
	ExportFunction multEDu;
	ExportFunction multQ1Gx;
	ExportFunction multQN1Gx;
	ExportFunction multQ1Gu;
	ExportFunction multQN1Gu;
	ExportFunction zeroBlockH00;
	ExportFunction multCTQC;

	ExportFunction macCTSlx;
	ExportFunction macETSlu;

	ExportFunction multHxC;
	ExportFunction multHxE;
	ExportFunction macHxd;
	/** @} */

	/** \name Contraint evaluation variables */
	/** @{ */
	ExportVariable A10;
	ExportVariable A20;
	ExportVariable pacA01Dx0;
	ExportVariable pocA02Dx0;
	/** @} */

	/** \name RTI related */
	/** @{ */
	ExportFunction preparation;
	ExportFunction feedback;

	ExportFunction getKKT;
	/** @} */

	/** \name Covariance calculation varibables and functions */
	/** @{ */
	ExportVariable CEN, sigmaTmp, sigma, sigmaN;
	ExportFunction calculateCovariance;
	/** @} */

	/** \name qpOASES interface variables */
	/** @{ */
	/** Variable containing the QP Hessian matrix. */
	ExportVariable H;
	/** Variable containing factorization of the QP Hessian matrix; R' * R = H. */
	ExportVariable R;
	/** Variable containing the QP constraint matrix. */
	ExportVariable A;
	/** Variable containing the QP gradient. */
	ExportVariable g;
	/** Variable containing the lower limits on QP variables. */
	ExportVariable lb;
	/** Variable containing the upper limits on QP variables. */
	ExportVariable ub;
	/** Variable containing lower limits on QP constraints. */
	ExportVariable lbA;
	/** Variable containing upper limits on QP constraints. */
	ExportVariable ubA;
	/** Variable containing the primal QP variables. */
	ExportVariable xVars;
	/** Variable containing the dual QP variables. */
	ExportVariable yVars;
	/** @} */
};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_EXPORT_GAUSS_NEWTON_CONDENSED_HPP
