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
 *    \file include/acado/code_generation/export_gauss_newton_block_forces.hpp
 *    \authors Rien Quirynen
 *    \date 2014
 */

#ifndef ACADO_TOOLKIT_EXPORT_GAUSS_NEWTON_BLOCK_FORCES_HPP
#define ACADO_TOOLKIT_EXPORT_GAUSS_NEWTON_BLOCK_FORCES_HPP

#include <acado/code_generation/export_gauss_newton_block_cn2.hpp>
#include <acado/code_generation/export_forces_interface.hpp>
#include <acado/code_generation/export_forces_generator.hpp>

BEGIN_NAMESPACE_ACADO

class ExportQpDunesInterface;

/**
 *	\brief An OCP solver based on the block N^2 condensing algorithm, in combination with qpDUNES as the QP solver.
 *
 *	\ingroup NumericalAlgorithms
 *
 *	\authors Rien Quirynen
 *
 *	\note Still a limited experimental version
 */
class ExportGaussNewtonBlockForces : public ExportGaussNewtonBlockCN2
{
public:

	/** Default constructor.
	 *
	 *	@param[in] _userInteraction		Pointer to corresponding user interface.
	 *	@param[in] _commonHeaderName	Name of common header file to be included.
	 */
	ExportGaussNewtonBlockForces(	UserInteraction* _userInteraction = 0,
							const std::string& _commonHeaderName = ""
							);

	/** Destructor. */
	virtual ~ExportGaussNewtonBlockForces( )
	{}

	/** Initializes export of an algorithm.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue setup( );


	/** Exports source code of the auto-generated condensing algorithm
	 *  into the given directory.
	 *
	 *	@param[in] code				Code block containing the auto-generated condensing algorithm.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue getCode(	ExportStatementBlock& code
									);


protected:

	virtual returnValue setupCondensing( );

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

	/** Exports source code containing the evaluation routines of the algorithm.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue setupEvaluation( );

	virtual returnValue setupQPInterface( );

protected:

	/** \name QP interface */
	/** @{ */
	std::string qpModuleName;
	std::string qpObjPrefix;
	/** @} */

	std::vector< ExportVariable > objHessians;
	std::vector< ExportVariable > objGradients;

	std::vector< ExportVariable > conLB;
	std::vector< ExportVariable > conUB;
	std::vector< std::vector< unsigned > > conLBIndices, conUBIndices;

	std::vector< ExportVariable > conA, conAB;
	ExportFunction evaluateAffineConstraints;
	std::vector< unsigned > conABDimensions;

	std::vector< ExportVariable > conC;
	std::vector< ExportVariable > cond;

	/** FORCES interface object. */
	std::shared_ptr< ExportForcesInterface > qpInterface;
	std::shared_ptr< ExportForcesGenerator > qpGenerator;

};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_EXPORT_GAUSS_NEWTON_BLOCK_QPDUNES_HPP
