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
 *    \file include/acado_controller.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 */



#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/user_interaction.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/variables_grid/variables_grid.hpp>
#include <acado/index_list/index_list.hpp>
#include <acado/symbolic_expression/symbolic_expression.hpp>
#include <acado/function/function.hpp>
#include <acado/integrator/integrator.hpp>
#include <acado/sparse_solver/sparse_solver.hpp>

#include <acado/dynamic_system/dynamic_system.hpp>
#include <acado/dynamic_discretization/dynamic_discretization.hpp>
#include <acado/dynamic_discretization/integration_algorithm.hpp>
#include <acado/nlp_solver/nlp_solver.hpp>
#include <acado/nlp_solver/scp_method.hpp>
#include <acado/ocp/ocp.hpp>
#include <acado/optimization_algorithm/optimization_algorithm.hpp>
#include <acado/optimization_algorithm/parameter_estimation_algorithm.hpp>
#include <acado/optimization_algorithm/multi_objective_algorithm.hpp>

#include <acado/curve/curve.hpp>
#include <acado/controller/controller.hpp>
#include <acado/estimator/estimator.hpp>
#include <acado/control_law/control_law.hpp>
#include <acado/control_law/pid_controller.hpp>
#include <acado/control_law/dynamic_feedback_law.hpp>
#include <acado/control_law/linear_state_feedback.hpp>
#include <acado/control_law/feedforward_law.hpp>
#include <acado/reference_trajectory/reference_trajectory.hpp>
