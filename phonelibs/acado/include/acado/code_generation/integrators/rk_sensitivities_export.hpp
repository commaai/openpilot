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
 *    \file include/acado/code_generation/integrators/rk_sensitivities_export.hpp
 *    \author Rien Quirynen
 *    \date 2013
 */


#ifndef ACADO_TOOLKIT_RK_SENSITIVITIES_EXPORT_HPP
#define ACADO_TOOLKIT_RK_SENSITIVITIES_EXPORT_HPP



BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Allows to export a tailored Runge-Kutta sensitivity propagation for fast model predictive control.
 *
 *	\ingroup NumericalAlgorithms
 *
 *	The class RKSensitivitiesExport allows to export a tailored Runge-Kutta sensitivity propagation
 *	for fast model predictive control.
 *
 *	\author Rien Quirynen
 */
class RKSensitivitiesExport
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //

    public:




	protected:


			/** Exports the code needed to compute the sensitivities of the states, defined by the linear input system.
			 *
			 *	@param[in] block			The block to which the code will be exported.
			 *	@param[in] Bh				The variable containing the weights of the RK method, multiplied with the step size.
			 *	@param[in] STATES			True if the sensitivities with respect to a state are needed, false otherwise.
			 *
			 *	\return SUCCESSFUL_RETURN
			 */
			virtual returnValue sensitivitiesInputSystem( 	ExportStatementBlock* block,
															const ExportIndex& index1,
															const ExportIndex& index2,
															const ExportVariable& Bh,
															bool STATES  	) = 0;


			/** Exports the code needed to compute the sensitivities of the states defined by the nonlinear, fully implicit system.
			 *
			 *	@param[in] block			The block to which the code will be exported.
			 *	@param[in] Ah				The variable containing the internal coefficients of the RK method, multiplied with the step size.
			 *	@param[in] Bh				The variable containing the weights of the RK method, multiplied with the step size.
			 *	@param[in] det				The variable that holds the determinant of the matrix in the linear system.
			 *	@param[in] STATES			True if the sensitivities with respect to a state are needed, false otherwise.
			 *	@param[in] number			This number defines the stage of the state with respect to which the sensitivities are computed.
			 *
			 *	\return SUCCESSFUL_RETURN
			 */
			virtual returnValue sensitivitiesImplicitSystem( 	ExportStatementBlock* block,
														const ExportIndex& index1,
														const ExportIndex& index2,
														const ExportIndex& index3,
														const ExportIndex& tmp_index1,
														const ExportIndex& tmp_index2,
														const ExportVariable& Ah,
														const ExportVariable& Bh,
														const ExportVariable& det,
														bool STATES,
														uint number 		) = 0;


			/** Exports the code needed to compute the sensitivities of the states, defined by the linear output system.
			 *
			 *	@param[in] block			The block to which the code will be exported.
			 *	@param[in] Ah				The variable containing the internal coefficients of the RK method, multiplied with the step size.
			 *	@param[in] Bh				The variable containing the weights of the RK method, multiplied with the step size.
			 *	@param[in] STATES			True if the sensitivities with respect to a state are needed, false otherwise.
			 *	@param[in] number			This number defines the stage of the state with respect to which the sensitivities are computed.
			 *
			 *	\return SUCCESSFUL_RETURN
			 */
			virtual returnValue sensitivitiesOutputSystem( 	ExportStatementBlock* block,
													const ExportIndex& index1,
													const ExportIndex& index2,
													const ExportIndex& index3,
													const ExportIndex& index4,
													const ExportIndex& tmp_index1,
													const ExportIndex& tmp_index2,
													const ExportVariable& Ah,
													const ExportVariable& Bh,
													bool STATES,
													uint number 		) = 0;


			/** Exports the computation of the sensitivities for the continuous output.
			 *
			 *	@param[in] block			The block to which the code will be exported.
			 *	@param[in] tmp_meas			The number of measurements in the current integration step (in case of an online grid).
			 *	@param[in] rk_tPrev			The time point, defining the beginning of the current integration step (in case of an online grid).
			 *	@param[in] time_tmp			A variable used for time transformations (in case of an online grid).
			 *	@param[in] STATES			True if the sensitivities with respect to a state are needed, false otherwise.
			 *	@param[in] base				The number of states in stages with respect to which the sensitivities have already been computed.
			 *
			 *	\return SUCCESSFUL_RETURN
			 */
			virtual returnValue sensitivitiesOutputs( 	ExportStatementBlock* block,
					const ExportIndex& index0,
					const ExportIndex& index1,
					const ExportIndex& index2,
					const ExportIndex& tmp_index1,
					const ExportIndex& tmp_index2,
					const ExportIndex& tmp_index3,
					const ExportVariable& tmp_meas,
					const ExportVariable& time_tmp,
					bool STATES,
					uint base			) = 0;


	protected:

};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_RK_SENSITIVITIES_EXPORT_HPP

// end of file.
