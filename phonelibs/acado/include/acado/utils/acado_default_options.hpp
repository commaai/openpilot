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
 *    \file include/acado/utils/acado_default_options.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 26.04.2010
 *
 *    This file collects default values for all options within the ACADO Toolkit.
 */

#ifndef ACADO_TOOLKIT_ACADO_DEFAULT_OPTIONS_HPP
#define ACADO_TOOLKIT_ACADO_DEFAULT_OPTIONS_HPP

#include <acado/utils/acado_types.hpp>

BEGIN_NAMESPACE_ACADO

// OptimizationAlgorithm
const int 		defaultMaxNumIterations = 200;										/**< Default value for maximum number of iterations of the NLP solver (possible values: any non-negative integer). */
const double 	defaultKKTtolerance = 1.0e-6;										/**< Default value for the KKT tolerance used as termination criterium by the NLP solver (possible values: any positive real number). */
const double 	defaultKKTtoleranceSafeguard = 1.0;									/**< Default value for safeguarding the KKT tolerance as termination criterium for the NLP solver (possible values: any non-negative real number). */
const double 	defaultLevenbergMarguardt = 0.0;									/**< Default value for Levenberg-Marquardt regularization (possible values: any non-negative real number). */
const double 	defaultHessianProjectionFactor = 1.0;								/**< Default value for projecting semi-definite Hessians to positive definite part (possible values: any positive real number). */
const int 		defaultHessianApproximation = BLOCK_BFGS_UPDATE;					/**< Default value for approximating the Hessian within the NLP solver (possible values: CONSTANT_HESSIAN, GAUSS_NEWTON, FULL_BFGS_UPDATE, BLOCK_BFGS_UPDATE, GAUSS_NEWTON_WITH_BLOCK_BFGS, EXACT_HESSIAN, DEFAULT_HESSIAN_APPROXIMATION). */
const int 		defaultDynamicHessianApproximation = DEFAULT_HESSIAN_APPROXIMATION;	/**< Default value for approximating the Hessian of the dynamic equations within the NLP solver (possible values: CONSTANT_HESSIAN, GAUSS_NEWTON, FULL_BFGS_UPDATE, BLOCK_BFGS_UPDATE, GAUSS_NEWTON_WITH_BLOCK_BFGS, EXACT_HESSIAN, DEFAULT_HESSIAN_APPROXIMATION). */
const int 		defaultDynamicSensitivity = BACKWARD_SENSITIVITY;					/**< Default value for generating sensitivities of the dynamic equations (possible values: FORWARD_SENSITIVITY, BACKWARD_SENSITIVITY). */
const int 		defaultObjectiveSensitivity = BACKWARD_SENSITIVITY;					/**< Default value for generating sensitivities of the objective function (possible values: FORWARD_SENSITIVITY, BACKWARD_SENSITIVITY). */
const int 		defaultConstraintSensitivity = BACKWARD_SENSITIVITY;				/**< Default value for generating sensitivities of the constraints (possible values: FORWARD_SENSITIVITY, BACKWARD_SENSITIVITY). */
const int 		defaultDiscretizationType = MULTIPLE_SHOOTING;						/**< Default value for specifying how to discretize the OCP in time (possible values: SINGLE_SHOOTING, MULTIPLE_SHOOTING, COLLOCATION). */
const int 		defaultSparseQPsolution = CONDENSING;								/**< Default value for specifying how to solve the sparse sub-QP (possible values: SPARSE_SOLVER, CONDENSING, FULL_CONDENSING). */
const int 		defaultGlobalizationStrategy = GS_LINESEARCH;						/**< Default value for specifying which globablization strategy is used within the NLP solver (possible values: GS_FULLSTEP, GS_LINESEARCH). */
const double 	defaultLinesearchTolerance = 1.0e-5;								/**< Default value for the tolerance of the line-search globalization (possible values: any positive real number). */
const double 	defaultMinLinesearchParameter = 0.5;								/**< Default value for the minimum stepsize of the line-search globalization (possible values: any positive real number). */
const int 		defaultMaxNumQPiterations = 10000;									/**< Default value for maximum number of iterations of the (underlying) QP solver (possible values: any positive integer). */
const int 		defaultHotstartQP = BT_FALSE;										/**< Default value for specifying whether the underlying QP shall be hotstarted or not (possible values: BT_TRUE, BT_FALSE). */
const double 	defaultInfeasibleQPrelaxation = 1.0e-8;								/**< Default value for the amount constraints are relaxed in case of an infeasible sub-QP (possible values: ). */
const int 		defaultInfeasibleQPhandling = IQH_RELAX_L2;							/**< Default value for specifying the strategy to handle infeasible sub-QPs (possible values: IQH_STOP, IQH_IGNORE, IQH_RELAX_L2). */
const int 		defaultUseRealtimeIterations = BT_FALSE;							/**< Default value for specifying whether real-time iterations shall be used (possible values: BT_TRUE, BT_FALSE). */
const int 		defaultUseRealtimeShifts = BT_FALSE;								/**< Default value for specifying whether shifted real-time iterations shall be used (possible values: BT_TRUE, BT_FALSE). */
const int 		defaultUseImmediateFeedback = BT_FALSE;								/**< Default value for specifying whether immediate feedback shall be used (possible values: BT_TRUE, BT_FALSE). */
const int 		defaultTerminateAtConvergence = BT_TRUE;							/**< Default value for specifying whether to stop iterations at convergence (possible values: BT_TRUE, BT_FALSE). */
const int 		defaultUseReferencePrediction = BT_TRUE;							/**< Default value for specifying whether the prediction of the reference trajectory shall be known the control law (possible values: BT_TRUE, BT_FALSE). */
const int 		defaultPrintlevel = MEDIUM;											/**< Default value for the printlevel determining the quatity of output given by the optimization algorithm (possible values: HIGH, MEDIUM, LOW, NONE). */
const int 		defaultPrintCopyright = BT_TRUE;									/**< Default value for specifying whether the ACADO copyright notice is printed or not (possible values: BT_TRUE, BT_FALSE). */
const int 		defaultprintSCPmethodProfile = BT_FALSE;							/**< Default value for printing the profile of the SCP method (possible values: BT_FALSE, BT_TRUE). */

// DynamicDiscretization
const int 		defaultFreezeIntegrator = BT_TRUE;							/**< Default value for specifying whether integrator should freeze all intermediate results (possible values: BT_TRUE, BT_FALSE). */
const int 		defaultIntegratorType = INT_RK45;							/**< Default value for integrator type (possible values: INT_RK12, INT_RK23, INT_RK45, INT_RK78, INT_BDF). */
const int 		defaultFeasibilityCheck = BT_FALSE;							/**< Default value for specifying whether infeasibilty shall be checked (possible values: BT_TRUE, BT_FALSE). */
const int 		defaultPlotResoltion = LOW;									/**< Default value for specifying the plot resolution (possible values: HIGH, MEDIUM, LOW). */

// Integrator
const int 		defaultMaxNumSteps = 1000;									/**< Default value for maximum number of integrator steps (possible values: any positive integer). */
const double 	defaultIntegratorTolerance = 1.0e-6;						/**< Default value for the (relative) integrator tolerance (possible values: any positive real number). */
const double 	defaultAbsoluteTolerance = 1.0e-8;							/**< Default value for the absolute integrator tolerance (possible values: any positive real number). */
const double 	defaultInitialStepsize = 1.0e-3;							/**< Default value for the intial stepsize of the integrator (possible values: any positive real number). */
const double 	defaultMinStepsize = 1.0e-8;								/**< Default value for the minimum stepsize of the integrator (possible values: any positive real number). */
const double 	defaultMaxStepsize = 1.0e+8;								/**< Default value for the maximum stepsize of the integrator (possible values: any positive real number). */
const double 	defaultStepsizeTuning = 0.5;								/**< Default value for the factor adapting the integrator stepsize (possible values: any positive real smaller than one). */
const double 	defaultCorrectorTolerance = 1.0e-14;						/**< Default value for the corrector tolerance of implicit integrators (possible values: any positive real number). */
const int 		defaultIntegratorPrintlevel = LOW;							/**< Default value for for the printlevel determining the quatity of output given by the integrator (possible values: HIGH, MEDIUM, LOW, NONE). */
const int 		defaultLinearAlgebraSolver = HOUSEHOLDER_METHOD;			/**< Default value for specifying how the linear systems are solved within the integrator (possible values: HOUSEHOLDER_METHOD, SPARSE_LU). */
const int 		defaultAlgebraicRelaxation = ART_ADAPTIVE_POLYNOMIAL;		/**< Default value for specifying how algebraic equations are relaxed within the integrator (possible values: ART_EXPONENTIAL, ART_ADAPTIVE_POLYNOMIAL). */
const double	defaultRelaxationParameter = 0.5;							/**< Default value for the amount algebraic equations are relaxed within the integrator (possible values: any positive real number). */
const int       defaultprintIntegratorProfile = BT_FALSE;					/**< Default value for specifying whether a runtime profile of the integrator shall be printed (possible values: BT_TRUE, BT_FALSE). */

// MultiObjectiveAlgorithm
const int 		defaultParetoFrontDiscretization = 21;						/**< Default value for the number of points of the pareto front (possible values: any postive integer). */
const int 		defaultParetoFrontGeneration = PFG_WEIGHTED_SUM;			/**< Default value for specifying the scalarization method (possible values: PFG_FIRST_OBJECTIVE, PFG_SECOND_OBJECTIVE, PFG_WEIGHTED_SUM, PFG_NORMALIZED_NORMAL_CONSTRAINT, PFG_NORMAL_BOUNDARY_INTERSECTION, PFG_ENHANCED_NORMALIZED_NORMAL_CONSTRAINT, PFG_EPSILON_CONSTRAINT). */
const int 		defaultParetoFrontHotstart = BT_TRUE;						/**< Default value for specifying whether hotstarts are to be used within the multi-objective optimization (possible values: BT_TRUE, BT_FALSE). */

// SimulationEnvironment
const int 		defaultSimulateComputationalDelay = BT_FALSE;				/**< Default value for specifying whether computational delays shall be simulated or not (possible values: BT_TRUE, BT_FALSE). */
const double 	defaultComputationalDelayFactor = 1.0;						/**< Default value for the factor scaling the actual computation time for simulating the computational delay (possible values: any non-negative real number). */
const double 	defaultComputationalDelayOffset = 0.0;						/**< Default value for the offset correcting the actual computation time for simulating the computational delay (possible values: any non-negative real number). */

// Process
const int 		defaultSimulationAlgorithm = SIMULATION_BY_INTEGRATION;		/**< Default value for specifying the simulation algorithm used within the process (possible values: SIMULATION_BY_INTEGRATION). */
const int 		defaultControlPlotting = PLOT_REAL;							/**< Default value for specifying how to plot controls within the process (possible values: PLOT_NOMINAL, PLOT_REAL). */
const int 		defaultParameterPlotting = PLOT_REAL;						/**< Default value for specifying how to plot parameters within the process (possible values: PLOT_NOMINAL, PLOT_REAL). */
const int 		defaultOutputPlotting = PLOT_REAL;							/**< Default value for specifying how to plot outputs within the process (possible values: PLOT_NOMINAL, PLOT_REAL). */

CLOSE_NAMESPACE_ACADO

#endif	// ACADO_TOOLKIT_ACADO_DEFAULT_OPTIONS_HPP

/*
 *	end of file
 */
