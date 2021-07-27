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
 *    \file include/acado/code_generation/export_gauss_newton_forces.hpp
 *    \author Milan Vukov
 *    \date 2012 - 2014
 */

#ifndef ACADO_TOOLKIT_EXPORT_GAUSS_NEWTON_FORCES_HPP
#define ACADO_TOOLKIT_EXPORT_GAUSS_NEWTON_FORCES_HPP

#include <acado/code_generation/export_nlp_solver.hpp>

BEGIN_NAMESPACE_ACADO

class ExportForcesInterface;
class ExportForcesGenerator;

/** 
 *	\brief A class for export of an OCP solver using sparse QP solver FORCES
 *
 *	\ingroup NumericalAlgorithms
 *
 *	The class ExportGaussNewtonForces allows export of and OCP solver using
 *	the generalized Gauss-Newton method. The underlying QP is solved using the
 *	structured sparse QP solver FORCES.
 * 
 *	\author Milan Vukov
 */
class ExportGaussNewtonForces : public ExportNLPSolver
{
public:

	/** Default constructor.
	 *
	 *	@param[in] _userInteraction		Pointer to corresponding user interface.
	 *	@param[in] _commonHeaderName	Name of common header file to be included.
	 */
	ExportGaussNewtonForces(	UserInteraction* _userInteraction = 0,
								const std::string& _commonHeaderName = ""
								);

	/** Destructor.
	*/
	virtual ~ExportGaussNewtonForces( )
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

	unsigned getNumLowerBounds( ) const;

	unsigned getNumUpperBounds( ) const;

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

private:
	/** Current state feedback. */
	ExportVariable x0;

	/** \name Objective evaluation */
	/** @{ */
	ExportFunction evaluateObjective;

	std::vector< ExportVariable > objHessians;
	std::vector< ExportVariable > objGradients;

	ExportFunction setStageH;
	ExportFunction setStagef;

	ExportFunction setObjQ1Q2;
	ExportFunction setObjR1R2;
	ExportFunction setObjQN1QN2;
	ExportFunction setObjS1;

	bool diagH;
	bool diagHN;
	/** @} */

	/** \name Constraint evaluation */
	/** @{ */
	std::vector< ExportVariable > conLB;
	std::vector< ExportVariable > conUB;
	ExportVariable lbValues, ubValues;

	unsigned numLB;
	unsigned numUB;

	std::vector< std::vector< unsigned > > conLBIndices, conUBIndices;
	std::vector< unsigned > conABDimensions;
	std::vector< std::vector< double > > conLBValues, conUBValues;

	ExportFunction evaluateConstraints;
	ExportFunction conSetGxGu;
	ExportVariable conStageC;

	std::vector< ExportVariable > conC;
	std::vector< ExportVariable > cond;
	ExportFunction conSetd;
	/** @} */

	/** \name RTI related */
	/** @{ */
	ExportFunction preparation;
	ExportFunction feedback;

	ExportFunction getKKT;
	/** @} */

	/** \name Helper functions */
	/** @{ */
	ExportFunction acc;
	/** @} */

	/** \name QP interface */
	/** @{ */
	std::string qpModuleName;
	std::string qpObjPrefix;

	std::shared_ptr< ExportForcesInterface > qpInterface;
	std::shared_ptr< ExportForcesGenerator > qpGenerator;
	/** @} */
};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_EXPORT_GAUSS_NEWTON_FORCES_HPP
