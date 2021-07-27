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
 *    \file include/acado/code_generation/ocp_export.hpp
 *    \authors Hans Joachim Ferreau, Boris Houska, Milan Vukov
 *    \date 2010 - 2014
 */

#ifndef ACADO_TOOLKIT_OCP_EXPORT_HPP
#define ACADO_TOOLKIT_OCP_EXPORT_HPP

#include <acado/code_generation/export_module.hpp>
#include <acado/ocp/ocp.hpp>

BEGIN_NAMESPACE_ACADO

class IntegratorExport;
class ExportNLPSolver;

/** \brief A user class for auto-generation of OCP solvers.
 *
 *	\ingroup UserInterfaces
 *
 *	The class OCPexport is a user-interface to automatically generate tailored
 *  algorithms for fast model predictive control and moving horizon estimation.
 *  It takes an optimal control problem (OCP) formulation and generates code
 *  based on given user options, e.g specifying the number of integrator steps
 *  or the online QP solver.
 *
 *	\authors Boris Houska, Hans Joachim Ferreau, Milan Vukov
 *
 *	\note Based on the old mpc_export class.
 */
class OCPexport : public ExportModule
{
public:

	/** Default constructor.
	 */
	OCPexport();

	/** Constructor which takes OCP formulation.
	 *
	 *	@param[in] _ocp		OCP formulation for code export.
	 */
	OCPexport(const OCP& _ocp);

	/** Destructor. */
	virtual ~OCPexport()
	{}

	/** Exports all files of the auto-generated code into the given directory.
	 *
	 *	@param[in] dirName			Name of directory to be used to export files.
	 *	@param[in] _realString		std::string to be used to declare real variables.
	 *	@param[in] _intString		std::string to be used to declare integer variables.
	 *	@param[in] _precision		Number of digits to be used for exporting real values.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue exportCode(	const std::string& dirName,
									const std::string& _realString = "real_t",
									const std::string& _intString = "int",
									int _precision = 16);

	/** Prints dimensions (i.e. number of variables and constraints)
	 *  of underlying QP.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	returnValue printDimensionsQP();

protected:

	/** Sets-up code export and initializes underlying export modules.
	 *
	 *	\return SUCCESSFUL_RETURN, \n
	 *	        RET_INVALID_OPTION, \n
	 *	        RET_INVALID_OBJECTIVE_FOR_CODE_EXPORT, \n
	 *	        RET_ONLY_ODE_FOR_CODE_EXPORT, \n
	 *	        RET_NO_DISCRETE_ODE_FOR_CODE_EXPORT, \n
	 *	        RET_ONLY_STATES_AND_CONTROLS_FOR_CODE_EXPORT, \n
	 *	        RET_ONLY_EQUIDISTANT_GRID_FOR_CODE_EXPORT, \n
	 *	        RET_ONLY_BOUNDS_FOR_CODE_EXPORT, \n
	 *	        RET_UNABLE_TO_EXPORT_CODE
	 */
	returnValue setup();

	/** Checks whether OCP formulation is compatible with code export capabilities.
	 *
	 *	\return SUCCESSFUL_RETURN, \n
	 *	        RET_INVALID_OBJECTIVE_FOR_CODE_EXPORT, \n
	 *	        RET_ONLY_ODE_FOR_CODE_EXPORT, \n
	 *	        RET_NO_DISCRETE_ODE_FOR_CODE_EXPORT, \n
	 *	        RET_ONLY_STATES_AND_CONTROLS_FOR_CODE_EXPORT, \n
	 *	        RET_ONLY_EQUIDISTANT_GRID_FOR_CODE_EXPORT, \n
	 *	        RET_ONLY_BOUNDS_FOR_CODE_EXPORT
	 */
	returnValue checkConsistency() const;

	/** Collects all data declarations of the auto-generated sub-modules to given
	 *	list of declarations.
	 *
	 *	@param[in] declarations		List of declarations.
	 *
	 *	\return SUCCESSFUL_RETURN, \n
	 *	        RET_UNABLE_TO_EXPORT_CODE
	 */
	returnValue collectDataDeclarations(	ExportStatementBlock& declarations,
											ExportStruct dataStruct = ACADO_ANY) const;

	/** Collects all function (forward) declarations of the auto-generated sub-modules
	 *	to given list of declarations.
	 *
	 *	@param[in] declarations		List of declarations.
	 *
	 *	\return SUCCESSFUL_RETURN, \n
	 *	        RET_UNABLE_TO_EXPORT_CODE
	 */
	returnValue collectFunctionDeclarations(ExportStatementBlock& declarations) const;

	/** Exports main header file for using the exported MHE algorithm.
	 *
	 *	@param[in] _dirName			Name of directory to be used to export file.
	 *	@param[in] _fileName		Name of file to be exported.
	 *	@param[in] _realString		std::string to be used to declare real variables.
	 *	@param[in] _intString		std::string to be used to declare integer variables.
	 *	@param[in] _precision		Number of digits to be used for exporting real values.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	returnValue exportAcadoHeader(	const std::string& _dirName,
									const std::string& _fileName,
									const std::string& _realString = "real_t",
									const std::string& _intString = "int",
									int _precision = 16) const;

	/** Shared pointer to a tailored integrator. */
	std::shared_ptr< IntegratorExport > integrator;

	/** Shared pointer to an NLP solver. */
	std::shared_ptr< ExportNLPSolver > solver;

	/** Internal copy of the OCP object. */
	OCP ocp;
};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_OCP_EXPORT_HPP
