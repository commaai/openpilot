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
 *    \file include/acado/integrator/integrator_fwd.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *    \date 2008
 */


#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/options.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/variables_grid/variables_grid.hpp>
#include <acado/symbolic_expression/symbolic_expression.hpp>
#include <acado/function/function.hpp>
#include <acado/clock/real_clock.hpp>

#include <acado/variables_grid/grid.hpp>
#include <acado/sparse_solver/sparse_solver.hpp>
#include <acado/function/differential_equation.hpp>
#include <acado/function/discretized_differential_equation.hpp>


BEGIN_NAMESPACE_ACADO


// FORWARD DEKLARATIONS:
// ---------------------

    class Integrator               ;
    class IntegratorDAE            ;
    class IntegratorODE            ;
    class IntegratorRK             ;
    class IntegratorRK12           ;
    class IntegratorRK23           ;
    class IntegratorRK45           ;
    class IntegratorRK78           ;
    class IntegratorDiscretizedODE ;
    class IntegratorBDF            ;


CLOSE_NAMESPACE_ACADO


