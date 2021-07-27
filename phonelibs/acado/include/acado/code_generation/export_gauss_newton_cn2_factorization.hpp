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
 *    \file include/acado/code_generation/export_gauss_newton_cn2_factorization.hpp
 *    \authors Milan Vukov
 *    \date 2013
 */

#ifndef ACADO_TOOLKIT_EXPORT_GAUSS_NEWTON_CN2_FACTORIZATION_HPP
#define ACADO_TOOLKIT_EXPORT_GAUSS_NEWTON_CN2_FACTORIZATION_HPP

#include <acado/code_generation/export_nlp_solver.hpp>
#include <acado/code_generation/linear_solvers/export_cholesky_solver.hpp>

BEGIN_NAMESPACE_ACADO

/**
 *	\brief TBD
 *
 *	\ingroup NumericalAlgorithms
 *
 *	\authors Milan Vukov
 *
 *	\note Early experimental implementation
 */
class ExportGaussNewtonCn2Factorization : public ExportNLPSolver
{
public:

	/** Default constructor.
	 *
	 *	@param[in] _userInteraction		Pointer to corresponding user interface.
	 *	@param[in] _commonHeaderName	Name of common header file to be included.
	 */
	ExportGaussNewtonCn2Factorization(	UserInteraction* _userInteraction = 0,
										const std::string& _commonHeaderName = ""
										);

	/** Destructor. */
	virtual ~ExportGaussNewtonCn2Factorization( )
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

	virtual returnValue setupQPInterface( );

	virtual returnValue setupCondensing( );

	bool performFullCondensing( ) const;

protected:

	ExportFunction evaluateObjective;

	ExportVariable x0, Dx0;

	ExportFunction setObjQ1Q2;
	ExportFunction setObjR1R2;
	ExportFunction setObjQN1QN2;

	/** Variable containing the QP Hessian matrix. */
	ExportVariable H, H00, H10, H11;
	/** Variable containing factorization of the QP Hessian matrix; R' * R = H. */
	ExportVariable U;
	/** Variable containing the QP constraint matrix. */
	ExportVariable A;

	/** Variable containing the QP gradient. */
	ExportVariable g, g0, g1;

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

	std::vector< unsigned > xBoundsIdxRev, xBoundsIdx;
	ExportVariable lbValues, ubValues;
	ExportVariable lbAValues, ubAValues;

	ExportVariable Qd;

	ExportFunction condensePrep;
	ExportFunction condenseFdb;
	ExportFunction expand;

	ExportVariable E, QE, QGx, QDy;

	ExportFunction multGxd;
	ExportFunction moveGxT;
	ExportFunction multGxGx;
	ExportFunction multGxGu;
	ExportFunction moveGuE;
	ExportFunction setBlockH11;
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

	ExportVariable A10;
	ExportVariable A20;
	ExportVariable pacA01Dx0;
	ExportVariable pocA02Dx0;
	ExportFunction multHxC;
	ExportFunction multHxE;
	ExportFunction macHxd;

	ExportFunction macCTSlx;
	ExportFunction macETSlu;

	ExportFunction preparation;
	ExportFunction feedback;

	ExportFunction getKKT;

	// N2 condensing related
	ExportVariable W1, W2;
	ExportVariable sbar, w1, w2;

	ExportFunction mult_H_W2T_W3, mac_H_W2T_W3_R, mac_W3_G_W1T_G;

	ExportFunction multBTW1, macBTW1_R1, multGxTGu, macQEW2;
	ExportFunction macATw1QDy, macBTw1, macQSbarW2, macASbar, macASbarD2;
	ExportFunction expansionStep;

	ExportCholeskySolver cholSolver;

	ExportFunction mac_R_T2_B_D;
	ExportFunction move_D_U;
	ExportFunction mult_L_E_U;
	ExportFunction updateQ;
	ExportFunction mul_T2_A_L;
	ExportFunction mult_BT_T1_T2;

	ExportVariable D, L;

	ExportVariable T1, T2, T3, F;

	ExportFunction mac_R_BT_F_D, mult_FT_A_L;
	ExportFunction updateQ2;
	ExportFunction mac_W1_T1_E_F;
	ExportFunction move_GxT_T3;
};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_EXPORT_GAUSS_NEWTON_CN2_FACTORIZATION_HPP
