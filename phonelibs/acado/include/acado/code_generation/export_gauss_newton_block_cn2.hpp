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
 *    \file include/acado/code_generation/export_gauss_newton_block_cn2.hpp
 *    \authors Rien Quirynen
 *    \date 2014
 */

#ifndef ACADO_TOOLKIT_EXPORT_GAUSS_NEWTON_BLOCK_CN2_HPP
#define ACADO_TOOLKIT_EXPORT_GAUSS_NEWTON_BLOCK_CN2_HPP

#include <acado/code_generation/export_gauss_newton_cn2.hpp>

BEGIN_NAMESPACE_ACADO

class ExportQpDunesInterface;

/**
 *	\brief An OCP solver based on the block N^2 condensing algorithm.
 *
 *	\ingroup NumericalAlgorithms
 *
 *	\authors Rien Quirynen
 *
 *	\note Still a limited experimental version
 */
class ExportGaussNewtonBlockCN2 : public ExportGaussNewtonCN2
{
public:

	/** Default constructor.
	 *
	 *	@param[in] _userInteraction		Pointer to corresponding user interface.
	 *	@param[in] _commonHeaderName	Name of common header file to be included.
	 */
	ExportGaussNewtonBlockCN2(	UserInteraction* _userInteraction = 0,
							const std::string& _commonHeaderName = ""
							);

	/** Destructor. */
	virtual ~ExportGaussNewtonBlockCN2( )
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
									) = 0;


	/** Returns number of variables in underlying QP.
	 *
	 *  \return Number of variables in underlying QP
	 */
	unsigned getNumQPvars( ) const;

	uint getBlockSize( ) const;

	uint getNumberOfBlocks( ) const;

	uint getNumBlockVariables( ) const;

	virtual unsigned getNumStateBoundsPerBlock( ) const;


protected:

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
	virtual returnValue setupEvaluation( ) = 0;

	virtual returnValue setupQPInterface( ) = 0;

	virtual returnValue setupCondensing( );

protected:

	ExportIndex blockI;

	std::vector< unsigned > qpConDim;

	ExportVariable qpgN;

	ExportVariable qpH;
	ExportVariable qpC;
	ExportVariable qpc;
	ExportVariable qpLb0, qpUb0;

	ExportVariable qpLambda, qpMu;

	ExportFunction cleanup;
	ExportFunction shiftQpData;

	ExportFunction evaluateConstraints;

};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_EXPORT_GAUSS_NEWTON_BLOCK_CN2_HPP
