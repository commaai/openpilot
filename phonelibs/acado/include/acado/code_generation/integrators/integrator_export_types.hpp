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
 *    \file include/acado/integrator/integrator_export_types.hpp
 *    \author Hans Joachim Ferreau, Boris Houska, Rien Quirynen, Milan Vukov
 *    \date 2010-2014
 */

#ifndef ACADO_TOOLKIT_INTEGRATOR_EXPORT_TYPES_HPP
#define ACADO_TOOLKIT_INTEGRATOR_EXPORT_TYPES_HPP

#include <acado/utils/acado_namespace_macros.hpp>

BEGIN_NAMESPACE_ACADO

/** Summarizes all available integrators for code generation.  */
enum ExportIntegratorType{

     INT_EX_EULER,         	/**< Explicit Euler method.           */
     INT_RK2,         	 	/**< Explicit Runge-Kutta integrator of order 2.           */
     INT_RK3,         	 	/**< Explicit Runge-Kutta integrator of order 3.           */
     INT_RK4,         	 	/**< Explicit Runge-Kutta integrator of order 4.           */
     INT_IRK_GL2,			/**< Gauss-Legendre integrator of order 2 (Continuous output Implicit Runge-Kutta). */
     INT_IRK_GL4,			/**< Gauss-Legendre integrator of order 4 (Continuous output Implicit Runge-Kutta). */
     INT_IRK_GL6,			/**< Gauss-Legendre integrator of order 6 (Continuous output Implicit Runge-Kutta). */
     INT_IRK_GL8,			/**< Gauss-Legendre integrator of order 8 (Continuous output Implicit Runge-Kutta). */

     INT_IRK_RIIA1,			/**< Radau IIA integrator of order 1 (Continuous output Implicit Runge-Kutta). */
     INT_IRK_RIIA3,			/**< Radau IIA integrator of order 3 (Continuous output Implicit Runge-Kutta). */
     INT_IRK_RIIA5,			/**< Radau IIA integrator of order 5 (Continuous output Implicit Runge-Kutta). */

     INT_DIRK3,				/**< Diagonally Implicit 2-stage Runge-Kutta integrator of order 3 (Continuous output). */
     INT_DIRK4,				/**< Diagonally Implicit 3-stage Runge-Kutta integrator of order 4 (Continuous output). */
     INT_DIRK5,				/**< Diagonally Implicit 5-stage Runge-Kutta integrator of order 5 (Continuous output). */

     INT_DT,				/**< An algorithm which handles the simulation and sensitivity generation for a discrete time state-space model. */
     INT_NARX				/**< An algorithm which handles the simulation and sensitivity generation for a NARX model. */
};

/**  Summarizes all possible sensitivity generation types for exported integrators.  */
enum ExportSensitivityType{

	NO_SENSITIVITY, 				/**< No sensitivities are computed, if possible. 		  					 */
    FORWARD,    					/**< Sensitivities are computed in forward mode.                             */
    BACKWARD,    					/**< Sensitivities are computed in backward mode.                            */
    FORWARD_OVER_BACKWARD,         	/**< Sensitivities (first and second order) are computed.					 */
    SYMMETRIC,         				/**< Sensitivities (first and second order) are computed.					 */
    SYMMETRIC_FB,         				/**< Sensitivities (first and second order) are computed.				 */
    INEXACT         				/**< Inexact sensitivities are computed by Newton iterations.				 */
};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_INTEGRATOR_EXPORT_TYPES_HPP
