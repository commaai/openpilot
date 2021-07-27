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
 *    \file include/acado/utils/acado_types.hpp
 *    \author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 *
 *    This file collects all declarations of all non-built-in types (except for classes).
 */

#ifndef ACADO_TOOLKIT_ACADO_TYPES_HPP
#define ACADO_TOOLKIT_ACADO_TYPES_HPP

#include <acado/utils/acado_namespace_macros.hpp>

BEGIN_NAMESPACE_ACADO

/** Short-cut for unsigned integer. */
typedef unsigned int uint;

/** Boolean type aliasing. */
typedef bool BooleanType;
/** Aliasing true. */
#define BT_TRUE true
/** Aliasing false. */
#define BT_FALSE false
/** Aliasing yes. */
#define YES true
/** Aliasing no. */
#define NO false

/** Function pointer type for functions given as C source code. */
typedef void (*cFcnPtr)( double* x, double* f, void *userData );

/** Function pointer type for derivatives given as C source code. */
typedef void (*cFcnDPtr)( int number, double* x, double* seed, double* f, double* df, void *userData );

/** Defines the Neutral Elements ZERO and ONE as well as the default
 *  NEITHER_ONE_NOR_ZERO
 */
enum NeutralElement{

    NE_ZERO,
    NE_ONE,
    NE_NEITHER_ONE_NOR_ZERO
};

/** Defines the names of all implemented symbolic operators. */
enum OperatorName{

    ON_SIN,
    ON_COS,
    ON_TAN,
    ON_ASIN,
    ON_ACOS,
    ON_ATAN,
    ON_LOGARITHM,
    ON_EXP,
    ON_ADDITION,
    ON_SUBTRACTION,
    ON_POWER,
    ON_POWER_INT,
    ON_PRODUCT,
    ON_QUOTIENT,
    ON_VARIABLE,
    ON_DOUBLE_CONSTANT,
    ON_DIFFERENTIAL_STATE,
    ON_CEXPRESSION
};

/** Defines the names of all implemented variable types. */
enum VariableType{

    VT_DIFFERENTIAL_STATE,
    VT_ALGEBRAIC_STATE,
    VT_CONTROL,
    VT_INTEGER_CONTROL,
    VT_PARAMETER,
    VT_INTEGER_PARAMETER,
    VT_DISTURBANCE,
    VT_TIME,
    VT_INTERMEDIATE_STATE,
    VT_DDIFFERENTIAL_STATE,
    VT_OUTPUT,
    VT_VARIABLE,
    VT_ONLINE_DATA,
    VT_UNKNOWN
};


/** Defines all possible methods of merging variables grids in case a grid point 
 *	exists in both grids. */
enum MergeMethod
{
	MM_KEEP,			/**< Keeps original values. */
	MM_REPLACE,			/**< Replace by new values. */
	MM_DUPLICATE		/**< Duplicate grid point (i.e. keeping old and adding new). */
};



/** Defines all possible sub-block matrix types. */
enum SubBlockMatrixType{

    SBMT_ZERO,
    SBMT_ONE,
    SBMT_DENSE,
    SBMT_UNKNOWN
};


/** Defines all possible relaxation type used in DAE integration routines */
enum AlgebraicRelaxationType{

    ART_EXPONENTIAL,
    ART_ADAPTIVE_POLYNOMIAL,
    ART_UNKNOWN
};


/** Defines all possible linear algebra solvers. */
enum LinearAlgebraSolver{

	HOUSEHOLDER_QR,
	GAUSS_LU,
	SIMPLIFIED_IRK_NEWTON,
	SINGLE_IRK_NEWTON,
    HOUSEHOLDER_METHOD,
	SPARSE_LU,
    LAS_UNKNOWN
};


/** Defines the mode of the exported implicit integrator. */
enum ImplicitIntegratorMode{

	IFTR,			/**< With the reuse of the matrix evaluation and factorization from the previous step (1 evaluation and factorization per integration step). */
	IFT,				/**< Without the reuse of the matrix from the previous step (2 evaluations and factorizations per integration step). */
	LIFTED,
	LIFTED_FEEDBACK
};


/** Defines all possible monotonicity types. */
enum MonotonicityType{

    MT_CONSTANT,
    MT_NONDECREASING,
    MT_NONINCREASING,
    MT_NONMONOTONIC,
    MT_UNKNOWN
};


/** Defines all possible curvature types. */
enum CurvatureType{

    CT_CONSTANT,                     /**< constant expression        */
    CT_AFFINE,                       /**< affine expression          */
    CT_CONVEX,                       /**< convex expression          */
    CT_CONCAVE,                      /**< concave expression         */
    CT_NEITHER_CONVEX_NOR_CONCAVE,   /**< neither convex nor concave */
    CT_UNKNOWN                       /**< unknown                    */
};


enum DifferentialEquationType{

    DET_ODE,                         /**< ordinary differential equation  */
    DET_DAE,                         /**< differential algebraic equation */
    DET_UNKNOWN                      /**< unknown                         */
};



/** Summarises all possible ways of discretising the system's states. */
enum StateDiscretizationType{

    SINGLE_SHOOTING,        /**< Single shooting discretisation.   */
    MULTIPLE_SHOOTING,      /**< Multiple shooting discretisation. */
    COLLOCATION,            /**< Collocation discretisation.       */
    UNKNOWN_DISCRETIZATION  /**< Discretisation type unknown.      */
};


/** Summarises all possible ways of discretising the system's states. */
enum ControlParameterizationType{

    CPT_CONSTANT,           /**< piece wise constant parametrization */
    CPT_LINEAR,             /**< piece wise linear   parametrization */
    CPT_LINEAR_CONTINUOUS,  /**< contious linear     parametrization */
    CPT_CUSTOMIZED,         /**< costumized                          */
    CPT_UNKNOWN             /**< unknown                             */
};


/** Summarises all possible ways of globablizing NLP steps. */
enum GlobalizationStrategy
{
	GS_FULLSTEP,					/**< Full step. */
	GS_LINESEARCH,					/**< Linesearch. */
	GS_UNKNOWN						/**< Unknown. */
};


/** Summarises all possible interpolation modes for VariablesGrids, Curves and the like. */
enum InterpolationMode
{
	IM_CONSTANT,					/**< Piecewise constant interpolation (not continous). */
	IM_LINEAR,						/**< Linear interpolation. */
	IM_QUADRATIC,					/**< Quadratic interpolation. */
	IM_CUBIC,						/**< Cubic interpolation. */
	IM_UNKNOWN						/**< Unknown interpolation mode. */
};


/** Summarizes all possible states of aggregation. (e.g. a mesh for
 *  an integration routine can be freezed, unfreezed, etc.)
 */
enum StateOfAggregation{

    SOA_FREEZING_MESH,             /**< freeze the mesh during next evaluation   */
    SOA_FREEZING_ALL,              /**< freeze everything during next evaluation */
    SOA_MESH_FROZEN,               /**< the mesh is frozen                       */
    SOA_MESH_FROZEN_FREEZING_ALL,  /**< the mesh is frozen, freeze also trajectory during next evaluation */
    SOA_EVERYTHING_FROZEN,         /**< everything is frozen                     */
    SOA_UNFROZEN,                  /**< everything is unfrozed                   */
    SOA_UNKNOWN                    /**< unknown                                  */
};


/** Summarizes all available integrators in standard ACADO. */
enum IntegratorType{

     INT_RK12,             	/**< Explicit Runge-Kutta integrator of order 1/2          */
     INT_RK23,             	/**< Explicit Runge-Kutta integrator of order 2/3          */
     INT_RK45,             	/**< Explicit Runge-Kutta integrator of order 4/5          */
     INT_RK78,            	/**< Explicit Runge-Kutta integrator of order 7/8          */
     INT_BDF,             	/**< Implicit backward differentiation formula integrator. */
     INT_DISCRETE,        	/**< Discrete time integrator                              */
     INT_LYAPUNOV45,        /**< Explicit Runge-Kutta integrator of order 4/5  with Lyapunov structure exploiting        */
     INT_UNKNOWN           	/**< unkown.                                               */
};


/** The available options for providing the grid of measurements.
 */
enum MeasurementGrid{

	OFFLINE_GRID,       	/**< An equidistant grid specified independent of the integration grid.         */
	ONLINE_GRID         	/**< A random grid, provided online by the user.           						*/
};


/** Unrolling option.
 */
enum UnrollOption{

     UNROLL,
     NO_UNROLL,
     HEURISTIC_UNROLL
};



/** Summarises all possible print levels. Print levels are used to describe
 *  the desired amount of output during runtime of ACADO Toolkit.
 */
enum PrintLevel
{
    NONE,        /**< No output.                                                         */
    LOW,         /**< Print error messages only.                                         */
    MEDIUM,      /**< Print error and warning messages as well as concise info messages. */
    HIGH,        /**< Print all messages with full details.                              */
    DEBUG        /**< Print all messages with full details as well                       *
                     *   all ugly messages that might be helpful for                        *
                     *   debugging the code.                                                */
};

enum LogRecordItemType{

    LRT_ENUM,
    LRT_VARIABLE,
    LRT_UNKNOWN
};


/** Defines logging frequency.
 */
enum LogFrequency{

    LOG_AT_START,
    LOG_AT_END,
    LOG_AT_EACH_ITERATION
};


/** Defines all possibilities to print LogRecords.
 */
enum LogPrintMode
{
	PRINT_ITEM_BY_ITEM,		/**< Print all numerical values of one item and continue with next item. */
	PRINT_ITER_BY_ITER,		/**< Print all numerical values of all items at one time instant (or iteration) and continue with next time instant. */
	PRINT_LAST_ITER			/**< Print all numerical values of all items at last time instant (or iteration) only. */
};


enum OptionsName
{
	CG_FORCE_DIAGONAL_HESSIAN,					/**< Force diagonal (stage) Hessian during the code export phase.*/
	CG_CONDENSED_HESSIAN_CHOLESKY,				/**< Type of the Cholesky decomposition of the condensed Hessian. \sa CondensedHessianCholeskyDecomposition */
	CG_MODULE_NAME,								/**< Name of the module, used as a prefix for the file-names and functions (shall be all lowercase). */
    CG_MODULE_PREFIX,                           /**< Prefix used for all global variables (shall be all uppercase). */
	CG_EXPORT_FOLDER_NAME,						/**< Export folder name. */
	CG_USE_ARRIVAL_COST,						/**< Enable interface for arival cost calculation. */
	CG_USE_OPENMP,								/**< Use OpenMP for parallelization in multiple shooting. */
	CG_USE_VARIABLE_WEIGHTING_MATRIX,			/**< Use variable weighting matrix S on first N shooting nodes. */
	CG_USE_C99,									/**< Code generation is allowed (or not) to export C-code that conforms C99 standard. */
	CG_COMPUTE_COVARIANCE_MATRIX,				/**< Enable computation of the variance-covariance matrix for the last estimate. */
	CG_HARDCODE_CONSTRAINT_VALUES,				/**< Enable/disable hard-coding of the constraint values. */
	IMPLICIT_INTEGRATOR_MODE,					/**< This determines the mode of the implicit integrator (see enum ImplicitIntegratorMode). */
//	LIFTED_INTEGRATOR_MODE,						/**< This determines the mode of lifting of the implicit integrator. */
	LIFTED_GRADIENT_UPDATE,						/**< This determines whether the gradient will be updated, based on the lifted implicit integrator. */
	IMPLICIT_INTEGRATOR_NUM_ITS,				/**< This is the performed number of Newton iterations in the implicit integrator. */
	IMPLICIT_INTEGRATOR_NUM_ITS_INIT,			/**< This is the performed number of Newton iterations in the implicit integrator for the initialization of the first step. */
	UNROLL_LINEAR_SOLVER,						/**< This option of the boolean type determines the unrolling of the linear solver (no unrolling recommended for larger systems). */
	CONDENSING_BLOCK_SIZE,						/**< Defines the block size used in a block based condensing approach for code generated RTI. */
	INTEGRATOR_DEBUG_MODE,
	OPT_UNKNOWN,
	MAX_NUM_INTEGRATOR_STEPS,
	NUM_INTEGRATOR_STEPS,
	INTEGRATOR_TOLERANCE,
	MEX_ITERATION_STEPS,						/**< The number of real-time iterations performed in the auto generated mex function. */
	MEX_VERBOSE,
	ABSOLUTE_TOLERANCE,
	INITIAL_INTEGRATOR_STEPSIZE,
	MIN_INTEGRATOR_STEPSIZE,
	MAX_INTEGRATOR_STEPSIZE,
	STEPSIZE_TUNING,
	CORRECTOR_TOLERANCE,
	INTEGRATOR_PRINTLEVEL,
	LINEAR_ALGEBRA_SOLVER,
	ALGEBRAIC_RELAXATION,
	RELAXATION_PARAMETER,
	PRINT_INTEGRATOR_PROFILE,
	FEASIBILITY_CHECK,
	MAX_NUM_ITERATIONS,
	KKT_TOLERANCE,
	KKT_TOLERANCE_SAFEGUARD,
	LEVENBERG_MARQUARDT,
	PRINTLEVEL,
	PRINT_COPYRIGHT,
	HESSIAN_APPROXIMATION,
	HESSIAN_REGULARIZATION,
	DYNAMIC_HESSIAN_APPROXIMATION,
	HESSIAN_PROJECTION_FACTOR,
	DYNAMIC_SENSITIVITY,
	OBJECTIVE_SENSITIVITY,
	CONSTRAINT_SENSITIVITY,
	DISCRETIZATION_TYPE,
	LINESEARCH_TOLERANCE,
	MIN_LINESEARCH_PARAMETER,
	QP_SOLVER,
	MAX_NUM_QP_ITERATIONS,
	HOTSTART_QP,
	INFEASIBLE_QP_RELAXATION,
	INFEASIBLE_QP_HANDLING,
	USE_REALTIME_ITERATIONS,
	USE_REALTIME_SHIFTS,
	USE_IMMEDIATE_FEEDBACK,
	TERMINATE_AT_CONVERGENCE,
	USE_REFERENCE_PREDICTION,
	FREEZE_INTEGRATOR,
	INTEGRATOR_TYPE,
	MEASUREMENT_GRID,
	SAMPLING_TIME,
	SIMULATE_COMPUTATIONAL_DELAY,
	COMPUTATIONAL_DELAY_FACTOR,
	COMPUTATIONAL_DELAY_OFFSET,
	PARETO_FRONT_DISCRETIZATION,
	PARETO_FRONT_GENERATION,
	PARETO_FRONT_HOTSTART,
	SIMULATION_ALGORITHM,
	CONTROL_PLOTTING,
	PARAMETER_PLOTTING,
	OUTPUT_PLOTTING,
	SPARSE_QP_SOLUTION,
	GLOBALIZATION_STRATEGY,
	CONIC_SOLVER_MAXIMUM_NUMBER_OF_STEPS,
	CONIC_SOLVER_TOLERANCE,
	CONIC_SOLVER_LINE_SEARCH_TUNING,
	CONIC_SOLVER_BARRIER_TUNING,
	CONIC_SOLVER_MEHROTRA_CORRECTION,
	CONIC_SOLVER_PRINT_LEVEL,
	PRINT_SCP_METHOD_PROFILE,
	PLOT_RESOLUTION,
	FIX_INITIAL_STATE,
	GENERATE_TEST_FILE,
	GENERATE_MAKE_FILE,
	GENERATE_SIMULINK_INTERFACE,
	GENERATE_MATLAB_INTERFACE,
	OPERATING_SYSTEM,
	USE_SINGLE_PRECISION
};


/** Defines possible logging output
 */
enum LogName
{
    LOG_NOTHING,
	// 1
	LOG_NUM_NLP_ITERATIONS, /**< Log number of NLP interations */
	LOG_NUM_SQP_ITERATIONS, /**< Log number of SQP interations */
	LOG_NUM_QP_ITERATIONS, /**< Log number of QP iterations */
	LOG_KKT_TOLERANCE,   /**< Log KKT tolerances */
	LOG_OBJECTIVE_VALUE, /**< Log values objective function */
	LOG_MERIT_FUNCTION_VALUE, /**< Log Merit function value*/
	LOG_LINESEARCH_STEPLENGTH, /**< Steplength of the line search routine (if used) */
	LOG_NORM_LAGRANGE_GRADIENT, /**< Log norm of Lagrange gradient*/
	LOG_IS_QP_RELAXED, /**< Log whether the QP is relaxed or not */
	// 10
	LOG_DUAL_RESIDUUM,
	LOG_PRIMAL_RESIDUUM,
	LOG_SURROGATE_DUALITY_GAP,
	LOG_NUM_INTEGRATOR_STEPS,
	LOG_TIME_SQP_ITERATION,
	LOG_TIME_CONDENSING,
	LOG_TIME_QP,
	LOG_TIME_RELAXED_QP,
	LOG_TIME_EXPAND,
	LOG_TIME_EVALUATION,
	// 20
	LOG_TIME_HESSIAN_COMPUTATION,
	LOG_TIME_GLOBALIZATION,
	LOG_TIME_SENSITIVITIES,
	LOG_TIME_LAGRANGE_GRADIENT,
	LOG_TIME_PROCESS,
	LOG_TIME_CONTROLLER,
	LOG_TIME_ESTIMATOR,
	LOG_TIME_CONTROL_LAW,
	LOG_DIFFERENTIAL_STATES, /**< Log all differential states in the order of occurrence*/
	LOG_ALGEBRAIC_STATES, /**< Log all algebraic states in the order of occurrence*/
	LOG_PARAMETERS, /**< Log all parameters in the order of occurrence*/
	LOG_CONTROLS, /**< Log all controls in the order of occurrence*/
	LOG_DISTURBANCES, /**< Log all disturbances in the order of occurrence*/
	LOG_INTERMEDIATE_STATES, /**< Log all intermediate states in the order of occurrence*/
	// 30
	LOG_DISCRETIZATION_INTERVALS, /**< Log discretization intervals*/
	LOG_STAGE_BREAK_POINTS,
	LOG_FEEDBACK_CONTROL,
	LOG_NOMINAL_CONTROLS,
	LOG_NOMINAL_PARAMETERS,
	LOG_SIMULATED_DIFFERENTIAL_STATES,
	LOG_SIMULATED_ALGEBRAIC_STATES,
	LOG_SIMULATED_CONTROLS,
	LOG_SIMULATED_PARAMETERS,
	LOG_SIMULATED_DISTURBANCES,
	// 40
	LOG_SIMULATED_INTERMEDIATE_STATES,
	LOG_SIMULATED_OUTPUT,
	LOG_PROCESS_OUTPUT,
    LOG_NUMBER_OF_INTEGRATOR_STEPS,
    LOG_NUMBER_OF_INTEGRATOR_REJECTED_STEPS,
    LOG_NUMBER_OF_INTEGRATOR_FUNCTION_EVALUATIONS,
    LOG_NUMBER_OF_BDF_INTEGRATOR_JACOBIAN_EVALUATIONS,
    LOG_TIME_INTEGRATOR,
    LOG_TIME_INTEGRATOR_FUNCTION_EVALUATIONS,
    LOG_TIME_BDF_INTEGRATOR_JACOBIAN_EVALUATION,
	// 50
    LOG_TIME_BDF_INTEGRATOR_JACOBIAN_DECOMPOSITION
};


enum PlotFrequency
{
	PLOT_AT_START,
	PLOT_AT_END,
	PLOT_AT_EACH_ITERATION,
	PLOT_IN_ANY_CASE,
	PLOT_NEVER
};


enum PlotName
{
	PLOT_NOTHING,
	// 1
// 	PLOT_DIFFERENTIAL_STATES,
// 	PLOT_ALGEBRAIC_STATES,
// 	PLOT_CONTROLS,
// 	PLOT_PARAMETERS,
// 	PLOT_DISTURBANCES,
// 	PLOT_INTERMEDIATE_STATES,
	PLOT_KKT_TOLERANCE,
	PLOT_OBJECTIVE_VALUE,
	PLOT_MERIT_FUNCTION_VALUE,
	PLOT_LINESEARCH_STEPLENGTH,
	PLOT_NORM_LAGRANGE_GRADIENT
};


enum ProcessPlotName
{
	PLOT_NOMINAL,
	PLOT_REAL
};


/** Defines all possible plot formats.
 */
enum PlotFormat
{
	PF_PLAIN,						/**< Plot with linear x- and y-axes. */
	PF_LOG,							/**< Plot with linear x-axis and logarithmic y-axis. */
	PF_LOG_LOG,						/**< Plot with logarithmic x- and y-axes. */
	PF_UNKNOWN						/**< Plot format unknown. */
};



/** Defines all possible plot modes.
 */
enum PlotMode
{
	PM_LINES,						/**< Plot data points linearly interpolated with lines. */
	PM_POINTS,						/**< Plot data points as single points. */
	PM_UNKNOWN						/**< Plot mode unknown. */
};



/** Defines all possible sub-plot types.
 */
enum SubPlotType
{
	SPT_VARIABLE,
	SPT_VARIABLE_VARIABLE,
	SPT_VARIABLE_EXPRESSION,
	SPT_VARIABLES_GRID,
	SPT_EXPRESSION,
	SPT_EXPRESSION_EXPRESSION,
	SPT_EXPRESSION_VARIABLE,
	SPT_ENUM,
	SPT_UNKNOWN
};


/** Defines possible printing types used in logging.
 */
enum PrintScheme
{
	PS_DEFAULT,			/**< Default printing, each row starts with [ and ends with ] and a newline. Colums are separated with a space. */
	PS_PLAIN,			/**< Plain printing, rows are separated with a newline and columns with a space */
	PS_MATLAB,			/**< Matlab style output. List starts with [ and ends with ]. Rows are separated by ; and columns by ,. */
	PS_MATLAB_BINARY	/**< Outputs a binary data file that can be read by Matlab. */
};


/**  Summarizes all possible sensitivity types */
enum SensitivityType{

    FORWARD_SENSITIVITY       ,    /**< Sensitivities are computed in forward mode                              */
    FORWARD_SENSITIVITY_LIFTED,    /**< Sensitivities are computed in forward mode using "lifting" if possible. */
    BACKWARD_SENSITIVITY      ,    /**< Sensitivities are computed in backward mode                             */
    UNKNOWN_SENSITIVITY            /**< unknown                                                                 */
};


/**  Condensing type */
enum CondensingType{

    CT_LIFTING,                 /**< Sensitivities are lifted                    */
    CT_SPARSE                   /**< Sensitivities are sparse                    */
};


/** Summarizes all possible algorithms for simulating the process. */
enum ProcessSimulationAlgorithm
{
	SIMULATION_BY_INTEGRATION,		/**< Simulation by using an integrator. */
	SIMULATION_BY_COLLOCATION		/**< Simulation by using a collocation scheme. */
};



/** Definition of several Hessian approximation modes. */
enum HessianApproximationMode{

    CONSTANT_HESSIAN,
	// 1
    GAUSS_NEWTON,
    FULL_BFGS_UPDATE,
    BLOCK_BFGS_UPDATE,
    GAUSS_NEWTON_WITH_BLOCK_BFGS,
    EXACT_HESSIAN,
    DEFAULT_HESSIAN_APPROXIMATION
};



/** Definition of several Hessian regularization modes. */
enum HessianRegularizationMode{

    BLOCK_REG, // = 0
	CONDENSED_REG
};


enum QPSolverName
{
	QP_QPOASES,
	QP_QPOASES3,
	QP_FORCES,
	QP_QPDUNES,
	QP_HPMPC,
    QP_GENERIC,
	QP_NONE
};


/** Summarises all possible states of the Conic Solver. */
enum ConicSolverStatus{

    CSS_NOTINITIALISED,			/**< The ConicSolver object is freshly instantiated or reset.          */
    CSS_INITIALIZED,			/**< The ConicSolver object is initialised                             */
    CSS_SOLVED,					/**< The solution of the actual Convex Optimization Problem was found. */
    CSS_UNKNOWN					/**< Status unknown.                                                   */
};



/** Summarizes all available strategies for handling infeasible QPs within
 *	an SQP-type NLP solver. */
enum InfeasibleQPhandling
{
	IQH_STOP,					/**< Stop solution. */
	IQH_IGNORE,					/**< Ignore infeasibility and continue solution. */
	IQH_RELAX_L1,				/**< Re-solve relaxed QP using a L1 penalty. */
	IQH_RELAX_L2,				/**< Re-solve relaxed QP using a L2 penalty. */
	IQH_UNDEFINED				/**< No infeasibility handling strategy defined. */
};


/**  Summarizes all possible states of a QP problem. */
enum QPStatus
{
    QPS_NOT_INITIALIZED,		/**< QP problem has not been initialised yet. */
    QPS_INITIALIZED,			/**< QP problem has been initialised.         */
    QPS_SOLVING,				/**< QP problem is being solved.              */
    QPS_SOLVED,					/**< QP problem successfully solved.          */
    QPS_RELAXED,				/**< QP problem has been relaxed.             */
    QPS_SOLVING_RELAXATION,		/**< A relaxed QP problem is being solved.    */
    QPS_SOLVED_RELAXATION,		/**< A relaxed QP problem has been solved.    */
    QPS_INFEASIBLE,				/**< QP problem is infeasible.                */
    QPS_UNBOUNDED,				/**< QP problem is unbounded.                 */
    QPS_NOTSOLVED				/**< QP problem could not been solved.        */
};


/**  Summarizes all possible states of the condensing used within condensing based CP solvers. */
enum CondensingStatus
{
    COS_NOT_INITIALIZED,		/**< Condensing has not been initialised yet. */
    COS_INITIALIZED,			/**< Condensing has been initialised, banded CP ready for condensing. */
    COS_CONDENSED,				/**< Banded CP has been condensed.            */
    COS_FROZEN					/**< Banded CP has been condensed and is frozen in this status. */
};


/**  Summarizes all possible block names. */
enum BlockName
{
    BN_DEFAULT,
    BN_SIMULATION_ENVIRONMENT,
    BN_PROCESS,
    BN_ACTUATOR,
    BN_SENSOR,
    BN_CONTROLLER,
    BN_ESTIMATOR,
    BN_REFERENCE_TRAJECTORY,
    BN_CONTROL_LAW
};


/**  Summarizes all possible states of a block or algorithmic module. */
enum BlockStatus
{
	BS_UNDEFINED,				/**< Status is undefined. */
	BS_NOT_INITIALIZED,			/**< Block/algorithm has been instantiated but not initialized. */
	BS_READY,					/**< Block/algorithm has been initialized and is ready to run. */
	BS_RUNNING					/**< Block/algorithm is running. */
};


/**  Summarizes all possible states of a clock. */
enum ClockStatus
{
	CS_NOT_INITIALIZED,			/**< Clock has not been initialized. */
	CS_RUNNING,					/**< Clock is running. */
	CS_STOPPED					/**< Clock has been initialized and stopped. */
};


/** Defines the time horizon start and end. */
enum TimeHorizonElement
{
	AT_TRANSITION = -3,
    AT_START,
    AT_END
};


/** Defines the pareto front generation options. \n
 */
enum ParetoFrontGeneration{

    PFG_FIRST_OBJECTIVE,
    PFG_SECOND_OBJECTIVE,
    PFG_WEIGHTED_SUM,
    PFG_NORMALIZED_NORMAL_CONSTRAINT,
    PFG_NORMAL_BOUNDARY_INTERSECTION,
    PFG_ENHANCED_NORMALIZED_NORMAL_CONSTRAINT,
    PFG_EPSILON_CONSTRAINT,
    PFG_UNKNOWN
};



/** Defines . \n
 */
enum SparseQPsolutionMethods
{
	SPARSE_SOLVER,
	CONDENSING,
	FULL_CONDENSING,
	FULL_CONDENSING_N2,
	CONDENSING_N2,
	BLOCK_CONDENSING_N2,
	FULL_CONDENSING_N2_FACTORIZATION
};



/** Defines . \n
 */
enum ExportStatementOperator
{
	ESO_ADD,
	ESO_SUBTRACT,
	ESO_ADD_ASSIGN,
	ESO_SUBTRACT_ASSIGN,
	ESO_MULTIPLY,
	ESO_MULTIPLY_TRANSPOSE,
	ESO_DIVIDE,
	ESO_MODULO,
	ESO_ASSIGN,
	ESO_UNDEFINED
};


enum OperatingSystem
{
	OS_DEFAULT,
	OS_UNIX,
	OS_WINDOWS
};


enum ExportType
{
	INT,
	REAL,
	COMPLEX,
	STATIC_CONST_INT,
	STATIC_CONST_REAL
};

enum ExportStruct
{
	ACADO_VARIABLES,
	ACADO_WORKSPACE,
	ACADO_PARAMS,
	ACADO_VARS,
	ACADO_LOCAL,
	ACADO_ANY,
	FORCES_PARAMS,
	FORCES_OUTPUT,
	FORCES_INFO
};

enum CondensedHessianCholeskyDecomposition
{
	EXTERNAL,		/**< External, performed within a QP solver. */
	INTERNAL_N3,	/**< n-cube version, performed within the exported code and passed to a QP solver. */
	INTERNAL_N2		/**< n-square version, performed within the exported code, and passed to a QP solver. */
};

/**
 *	\brief Defines all symbols for global return values.
 *
 *	\ingroup BasicDataStructures
 *
 *  The enumeration returnValueType defines all symbols for global return values.
 *	Important: All return values are assumed to be nonnegative!
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
enum returnValueType
{
TERMINAL_LIST_ELEMENT = -1,						/**< Terminal list element, internal usage only! */
/* miscellaneous */
SUCCESSFUL_RETURN = 0,							/**< Successful return. */
RET_DIV_BY_ZERO,									/**< Division by zero. */
RET_INDEX_OUT_OF_BOUNDS,						/**< Index out of bounds. */
RET_INVALID_ARGUMENTS,							/**< At least one of the arguments is invalid. */
RET_ERROR_UNDEFINED,							/**< Error number undefined. */
RET_WARNING_UNDEFINED,							/**< Warning number undefined. */
RET_INFO_UNDEFINED,								/**< Info number undefined. */
RET_EWI_UNDEFINED,								/**< Error/warning/info number undefined. */
RET_AVAILABLE_WITH_LINUX_ONLY,					/**< This function is available under Linux only. */
RET_UNKNOWN_BUG,								/**< The error occured is not yet known. */
RET_PRINTLEVEL_CHANGED,							/**< Print level changed. */
RET_NOT_YET_IMPLEMENTED,						/**< Requested function is not yet implemented. */
RET_NOT_IMPLEMENTED_YET,						/**< Requested function is not yet implemented. */
RET_NOT_IMPLEMENTED_IN_BASE_CLASS,				/**< Requested function is not implemented within this class. */
RET_ASSERTION,									/**< An assertion has been violated. */
RET_MEMBER_NOT_INITIALISED, 					/**< Requested member has not been initialised. */
RET_ABSTRACT_BASE_CLASS,						/**< Invalid call to member function of abstract base class. */
RET_NO_DATA_FOUND,								/**< There has no data been found. */
RET_INPUT_DIMENSION_MISMATCH,					/**< The dimensions of the input are wrong. */
RET_STRING_EXCEEDS_LENGTH,						/**< String exceeds maximum length. */

/* IO utils: */
RET_FILE_NOT_FOUND,								/**< The file has not been found.*/
RET_FILE_CAN_NOT_BE_OPENED,						/**< The file can not be opened. */
RET_CAN_NOT_WRITE_INTO_FILE,					/**< The routine has no write access or writing into the file failed. */
RET_FILE_CAN_NOT_BE_CLOSED,						/**< The file could not be closed. */
RET_FILE_HAS_NO_VALID_ENTRIES,					/**< The file has no valid entries. */
RET_DOES_DIRECTORY_EXISTS,						/**< Could not open file, check if given directory exists. */

/* DMatrix/DVector */
RET_VECTOR_DIMENSION_MISMATCH,					/**< Incompatible vector dimensions. */
RET_DIFFERENTIAL_STATE_DIMENSION_MISMATCH,		/**< Incompatible differential state vector dimensions. */
RET_ALGEBRAIC_STATE_DIMENSION_MISMATCH,			/**< Incompatible algebraic state vector dimensions. */
RET_CONTROL_DIMENSION_MISMATCH,					/**< Incompatible control vector dimensions. */
RET_PARAMETER_DIMENSION_MISMATCH,				/**< Incompatible parameter vector dimensions. */
RET_DISTURBANCE_DIMENSION_MISMATCH,				/**< Incompatible disturbance vector dimensions. */
RET_OUTPUT_DIMENSION_MISMATCH,					/**< Incompatible output vector dimensions. */
RET_MATRIX_NOT_SQUARE,							/**< Operation requires square matrix. */

/* Sparse Solver */
RET_LINEAR_SYSTEM_NUMERICALLY_SINGULAR,			/**< Linear system could not be solved with required accuracy. Check whether the system is singular or ill-conditioned. */

/* Grid */
RET_GRIDPOINT_SETUP_FAILED,						/**< Failed to setup grid point. */
RET_GRIDPOINT_HAS_INVALID_TIME,					/**< Unable to setup a grid point with this time. */
RET_CONFLICTING_GRIDS,							/**< Conflicting grids detected. */
RET_TIME_INTERVAL_NOT_VALID,					/**< The time interval is not valid / not in range. */
RET_INVALID_TIME_POINT,							/**< A time point is not in its permissible range. */

/* Options */
RET_OPTION_ALREADY_EXISTS,						/**< An option with this name already exists. */
RET_OPTION_DOESNT_EXIST,						/**< An option with this name does not exist. */
RET_OPTIONS_LIST_CORRUPTED,						/**< Internal options list is corrupted. */
RET_INVALID_OPTION,								/**< A user-defined option has an invalid value. */

/* Plotting */
RET_PLOTTING_FAILED,							/**< Unable to plot current window. */
RET_EMPTY_PLOT_DATA,							/**< Unable to plot subplot as plot data is empty. */
RET_PLOT_WINDOW_CORRUPTED,						/**< PlotWindow has corrupted list of subplots. */
RET_PLOT_COLLECTION_CORRUPTED,					/**< PlotCollection has corrupted list of plot windows. */

/* Logging */
RET_LOG_RECORD_CORRUPTED,						/**< LogRecord has corrupted list of items. */
RET_LOG_ENTRY_DOESNT_EXIST,						/**< An log entry with this name does not exist. */
RET_LOG_COLLECTION_CORRUPTED,					/**< LogCollection has corrupted list of records. */

/* SimulationEnvironment */
RET_BLOCK_DIMENSION_MISMATCH,					/**< Blocks with incompatible dimensions. */
RET_NO_PROCESS_SPECIFIED,						/**< No process has been specified. */
RET_NO_CONTROLLER_SPECIFIED,					/**< No controller has been specified. */
RET_ENVIRONMENT_INIT_FAILED,					/**< Unable to initialize simulation environment. */
RET_ENVIRONMENT_STEP_FAILED,					/**< Unable to perform simulation environment step. */
RET_COMPUTATIONAL_DELAY_TOO_BIG,				/**< Simulation stops as computational delay is too big. */
RET_COMPUTATIONAL_DELAY_NOT_SUPPORTED,			/**< Simulation of computational delay is not yet supported. */

/* Block */
RET_BLOCK_NOT_READY,							/**< Block is not ready. */

/* Time */
RET_NO_SYSTEM_TIME,								/**< No system time available. */
RET_CLOCK_NOT_READY,							/**< Unable to start the clock as it is not ready. */

/* Process */
RET_PROCESS_INIT_FAILED,						/**< Unable to initialize process. */
RET_PROCESS_STEP_FAILED,						/**< Unable to perform process step. */
RET_PROCESS_STEP_FAILED_DISTURBANCE,			/**< Unable to perform process step due to error in disturbance evaluation. */
RET_PROCESS_RUN_FAILED,							/**< Unable to run process simulation. */
RET_NO_DYNAMICSYSTEM_SPECIFIED,					/**< No dynamic system has been specified. */
RET_NO_INTEGRATIONALGORITHM_SPECIFIED,			/**< No integration algorithm has been specified. */
RET_NO_DISCRETE_TIME_SYSTEMS_SUPPORTED,			/**< Discrete-time systems are not yet supported. */
RET_WRONG_DISTURBANCE_HORIZON,					/**< Process disturbance is defined over an incompatible time horizon. */

/* Actuator / Sensor */
RET_ACTUATOR_INIT_FAILED,						/**< Unable to initialize actuator. */
RET_ACTUATOR_STEP_FAILED,						/**< Unable to perform actuator step. */
RET_SENSOR_INIT_FAILED,							/**< Unable to initialize sensor. */
RET_SENSOR_STEP_FAILED,							/**< Unable to perform sensor step. */
RET_GENERATING_NOISE_FAILED,					/**< Unable to generate noise. */
RET_DELAYING_INPUTS_FAILED,						/**< Unable to delay inputs. */
RET_DELAYING_OUTPUTS_FAILED,					/**< Unable to delay outputs. */
RET_INCOMPATIBLE_ACTUATOR_SAMPLING_TIME,		/**< Actuator sampling time has to be an integer multiple of dynamic system sample time. */
RET_INCOMPATIBLE_SENSOR_SAMPLING_TIME,			/**< Sensor sampling time has to be an integer multiple of dynamic system sample time. */
RET_NO_DIFFERENT_NOISE_SAMPLING_FOR_DISCRETE,	/**< When using discrete-time systems, noise has to be sampled equally for all components. */

/* Noise */
RET_NO_NOISE_GENERATED,							/**< No noise has been generated. */
RET_NO_NOISE_SETTINGS,							/**< No noise settings has been defined. */
RET_INVALID_NOISE_SETTINGS,						/**< Specified noise settings are invalid. */

/* Controller */
RET_CONTROLLER_INIT_FAILED,						/**< Unable to initialize controller. */
RET_CONTROLLER_STEP_FAILED,						/**< Unable to perform controller step. */
RET_NO_ESTIMATOR_SPECIFIED,						/**< No estimator has been specified. */
RET_NO_CONTROLLAW_SPECIFIED,					/**< No control law has been specified. */
RET_NO_REALTIME_MODE_AVAILABLE,					/**< Control law does not support real-time mode. */

/* DynamicControlUnit / Estimator / Optimizer */
RET_DCU_INIT_FAILED,							/**< Unable to initialize dynamic control unit. */
RET_DCU_STEP_FAILED,							/**< Unable to perform step of dynamic control unit. */
RET_ESTIMATOR_INIT_FAILED,						/**< Unable to initialize estimator. */
RET_ESTIMATOR_STEP_FAILED,						/**< Unable to perform estimator step. */
RET_OPTIMIZER_INIT_FAILED,						/**< Unable to initialize optimizer. */
RET_OPTIMIZER_STEP_FAILED,						/**< Unable to perform optimizer step. */
RET_NO_OCP_SPECIFIED,							/**< No optimal control problem has been specified. */
RET_NO_SOLUTIONALGORITHM_SPECIFIED,				/**< No solution algorithm has been specified. */

/* ControlLaw */
RET_CONTROLLAW_INIT_FAILED,						/**< Unable to initialize feedback law. */
RET_CONTROLLAW_STEP_FAILED,						/**< Unable to perform feedback law step. */
RET_NO_OPTIMIZER_SPECIFIED,						/**< No optimizer has been specified. */
RET_INVALID_PID_OUTPUT_DIMENSION,				/**< Invalid output dimension of PID controller, reset to 1. */

/* RealTimeAlgorithm */
RET_IMMEDIATE_FEEDBACK_ONE_ITERATION,			/**< Resetting maximum number of iterations to 1 as required for using immediate feedback. */

/* OutputTransformator */
RET_OUTPUTTRANSFORMATOR_INIT_FAILED,			/**< Unable to initialize output transformator. */
RET_OUTPUTTRANSFORMATOR_STEP_FAILED,			/**< Unable to perform output transformator step. */

/* Function */
RET_INVALID_USE_OF_FUNCTION,					/**< Invalid use of the class function. */
RET_INFEASIBLE_CONSTRAINT,						/**< Infeasible Constraints detected. */
RET_ONLY_SUPPORTED_FOR_SYMBOLIC_FUNCTIONS,		/**< This routine is for symbolic functions only. */
RET_INFEASIBLE_ALGEBRAIC_CONSTRAINT,			/**< Infeasible algebraic constraints are not allowed and will be ignored. */
RET_ILLFORMED_ODE,								/**< ODE needs to depend on all differential states. */

/* Expression */
RET_PRECISION_OUT_OF_RANGE,						/**< the requested precision is out of range. */
RET_ERROR_WHILE_PRINTING_A_FILE,				/**< An error has occured while printing a file. */
RET_INDEX_OUT_OF_RANGE,							/**< An index was not in the range. */
RET_INTERMEDIATE_STATE_HAS_NO_ARGUMENT,			/**< The intermediate state has no argument. */
RET_DIMENSION_NOT_SPECIFIED,					/**< The dimension of a array was not specified. */

/* Modeling Tools */
RET_DDQ_DIMENSION_MISMATCH,						/**< ddq argument must be of size 3x1 */
RET_CAN_ONLY_SOLVE_2ND_ORDER_KINVECS,			/**< can only solve 2nd order KinVecs */


/* OBJECTIVE */
RET_GAUSS_NEWTON_APPROXIMATION_NOT_SUPPORTED,	/**< The objective does not support Gauss-Newton Hessian approximations. */
RET_REFERENCE_SHIFTING_WORKS_FOR_LSQ_TERMS_ONLY,	/**< The reference shifting works only for LSQ objectives. */

/* Integrator */
RET_TRIVIAL_RHS,								/**< the dimension of the rhs is zero. */
RET_MISSING_INPUTS,								/**< the integration routine misses some inputs. */
RET_TO_SMALL_OR_NEGATIVE_TIME_INTERVAL,			/**< the time interval was too small or negative.*/
RET_FINAL_STEP_NOT_PERFORMED_YET,				/**< the integration routine is not ready. */
RET_ALREADY_FROZEN,								/**< the integrator is already freezing or frozen. */
RET_MAX_NUMBER_OF_STEPS_EXCEEDED,				/**< the maximum number of steps has been exceeded. */
RET_WRONG_DEFINITION_OF_SEEDS,					/**< the seeds are not set correctly or in the wrong order. */
RET_NOT_FROZEN,									/**< the mesh is not frozen and/or forward results not stored. */
RET_TO_MANY_DIFFERENTIAL_STATES,				/**< there are to many differential states. */
RET_TO_MANY_DIFFERENTIAL_STATE_DERIVATIVES,		/**< there are to many diff. state derivatives. */
RET_RK45_CAN_NOT_TREAT_DAE,						/**< An explicit Runge-Kutta solver cannot treat DAEs. */
RET_CANNOT_TREAT_DAE,							/**< The algorithm cannot treat DAEs. */
RET_INPUT_HAS_WRONG_DIMENSION,					/**< At least one of the inputs has a wrong dimension. */
RET_INPUT_OUT_OF_RANGE,							/**< One of the inputs is out of range. */
RET_THE_DAE_INDEX_IS_TOO_LARGE,					/**< The index of the DAE is larger than 1. */
RET_UNSUCCESSFUL_RETURN_FROM_INTEGRATOR_RK45,	/**< the integration routine stopped due to a problem during the function evaluation. */
RET_UNSUCCESSFUL_RETURN_FROM_INTEGRATOR_BDF,	/**< the integration routine stopped as the required accuracy can not be obtained. */
RET_CANNOT_TREAT_DISCRETE_DE,					/**< This integrator cannot treat discrete-time differential equations. */
RET_CANNOT_TREAT_CONTINUOUS_DE,					/**< This integrator cannot treat time-continuous differential equations. */
RET_CANNOT_TREAT_IMPLICIT_DE,					/**< This integrator cannot treat differential equations in implicit form. */
RET_CANNOT_TREAT_EXPLICIT_DE,					/**< This integrator cannot treat differential equations in explicit form. */

/* DynamicDiscretization */
RET_TO_MANY_DIFFERENTIAL_EQUATIONS,				/**< The number of differential equations is too large. */
RET_BREAK_POINT_SETUP_FAILED,					/**< The break point setup failed. */
RET_WRONG_DEFINITION_OF_STAGE_TRANSITIONS,		/**< The definition of stage transitions is wrong. */
RET_TRANSITION_DEPENDS_ON_ALGEBRAIC_STATES,		/**< A transition should never depend on algebraic states.*/

/* OPTIMIZATION_ALGORITHM: */
RET_NO_VALID_OBJECTIVE,							/**< No valid objective found. */
RET_INCONSISTENT_BOUNDS,						/**< The bounds are inconsistent. */
RET_INCOMPATIBLE_DIMENSIONS,					/**< Incopatible dimensions detected. */
RET_GRID_SETUP_FAILED,							/**< Discretization of the OCP failed. */
RET_OPTALG_INIT_FAILED, 						/**< Initialization of optimization algorithm failed. */
RET_OPTALG_STEP_FAILED, 						/**< Step of optimization algorithm failed. */
RET_OPTALG_FEEDBACK_FAILED, 					/**< Feedback step of optimization algorithm failed. */
RET_OPTALG_PREPARE_FAILED, 						/**< Preparation step of optimization algorithm failed. */
RET_OPTALG_SOLVE_FAILED, 						/**< Problem could not be solved with given optimization algorithm. */
RET_REALTIME_NO_INITIAL_VALUE, 					/**< No initial value has been specified. */

/* INTEGRATION_ALGORITHM: */
RET_INTALG_INIT_FAILED, 						/**< Initialization of integration algorithm failed. */
RET_INTALG_INTEGRATION_FAILED, 					/**< Integration algorithm failed to integrate dynamic system. */
RET_INTALG_NOT_READY, 							/**< Integration algorithm has not been initialized. */

/* PLOT WINDOW */
RET_PLOT_WINDOW_CAN_NOT_BE_OPEN,				/**< ACADO was not able to open the plot window. */

/* NLP SOLVER */
CONVERGENCE_ACHIEVED,							/**< convergence achieved. */
CONVERGENCE_NOT_YET_ACHIEVED,					/**< convergence not yet achieved. */
RET_NLP_INIT_FAILED, 							/**< Initialization of NLP solver failed. */
RET_NLP_STEP_FAILED, 							/**< Step of NLP solver failed. */
RET_NLP_SOLUTION_FAILED,						/**< NLP solution failed. */
RET_INITIALIZE_FIRST, 							/**< Object needs to be initialized first. */
RET_SOLVER_NOT_SUTIABLE_FOR_REAL_TIME_MODE,		/**< The specified NLP solver is not designed for a real-time mode. */
RET_ILLFORMED_HESSIAN_MATRIX,					/**< Hessian matrix is too ill-conditioned to continue. */
RET_NONSYMMETRIC_HESSIAN_MATRIX,				/**< Hessian matrix is not symmetric, proceeding with symmetrized Hessian. */
RET_UNABLE_TO_EVALUATE_OBJECTIVE,				/**< Evaluation of objective function failed. */
RET_UNABLE_TO_EVALUATE_CONSTRAINTS,				/**< Evaluation of constraints failed. */
RET_UNABLE_TO_INTEGRATE_SYSTEM,					/**< Integration of dynamic system failed. Try to adjust integrator tolerances using set( ABSOLUTE_TOLERANCE,<double> ) and set( INTEGRATOR_TOLERANCE,<double> ). */
RET_NEED_TO_ACTIVATE_RTI,						/**< Feedback step requires real-time iterations to be activated. Use set( USE_REALTIME_ITERATIONS,YES ) to do so. */

/* CONIC SOLVER */
RET_CONIC_PROGRAM_INFEASIBLE,					/**< The optimization problem is infeasible. */
RET_CONIC_PROGRAM_SOLUTION_FAILED,				/**< Conic Program solution failed. The optimization problem might be infeasible. */
RET_CONIC_PROGRAM_NOT_SOLVED,					/**< The Conic Program has not been solved successfully. */

/* CP SOLVER */
RET_UNABLE_TO_CONDENSE,							/**< Unable to condense banded CP. */
RET_UNABLE_TO_EXPAND,							/**< Unable to expand condensed CP. */
RET_NEED_TO_CONDENSE_FIRST,						/**< Condensing cannot be frozen as banded CP needs to be condensed first. */
RET_BANDED_CP_INIT_FAILED,						/**< Initialization of banded CP solver failed. */
RET_BANDED_CP_SOLUTION_FAILED,					/**< Solution of banded CP failed. */

/* OCP */
RET_TRANSITION_NOT_DEFINED,						/**> No transition function found. */

/* QP SOLVER */
RET_QP_INIT_FAILED,								/**< QP initialization failed. */
RET_QP_SOLUTION_FAILED,							/**< QP solution failed. */
RET_QP_SOLUTION_REACHED_LIMIT,					/**< QP solution stopped as iteration limit is reached. */
RET_QP_INFEASIBLE, 								/**< QP solution failed due to infeasibility. */
RET_QP_UNBOUNDED, 								/**< QP solution failed due to unboundedness. */
RET_QP_NOT_SOLVED,								/**< QP has not been solved. */
RET_RELAXING_QP,								/**< QP needs to be relaxed due to infeasibility. */
RET_COULD_NOT_RELAX_QP, 						/**< QP could not be relaxed. */
RET_QP_SOLVER_CAN_ONLY_SOLVE_QP,				/**< The QP solver can not deal with general conic problems. */
RET_QP_HAS_INCONSISTENT_BOUNDS,					/**< QP cannot be solved due to inconsistent bounds. */
RET_UNABLE_TO_HOTSTART_QP,						/**< Unable to hotstart QP with given solver. */

/* MPC SOLVER */
RET_NONPOSITIVE_WEIGHT, 						/**< Weighting matrices must be positive semi-definite. */
RET_INITIAL_CHOLESKY_FAILED, 					/**< Setting up initial Cholesky decompostion failed. */
RET_HOMOTOPY_STEP_FAILED,						/**< Unable to perform homotopy step. */
RET_STEPDIRECTION_DETERMINATION_FAILED,			/**< Determination of step direction failed. */
RET_STEPDIRECTION_FAILED_CHOLESKY,				/**< Abnormal termination due to Cholesky factorisation. */
RET_STEPLENGTH_DETERMINATION_FAILED,			/**< Determination of step direction failed. */
RET_OPTIMAL_SOLUTION_FOUND,						/**< Optimal solution of neighbouring QP found. */
RET_MAX_NWSR_REACHED,							/**< Maximum number of working set recalculations performed. */
RET_MATRIX_NOT_SPD,								/**< DMatrix is not positive definite. */

/* CODE EXPORT */
RET_CODE_EXPORT_SUCCESSFUL,						/**< Code generation successful. */
RET_UNABLE_TO_EXPORT_CODE,						/**< Unable to generate code. */
RET_INVALID_OBJECTIVE_FOR_CODE_EXPORT,			/**< Only standard LSQ objective supported for code generation. */
RET_NO_DISCRETE_ODE_FOR_CODE_EXPORT,			/**< No discrete-time ODEs supported for code generation. */
RET_ONLY_ODE_FOR_CODE_EXPORT,					/**< Only ODEs supported for code generation. */
RET_ONLY_STATES_AND_CONTROLS_FOR_CODE_EXPORT,	/**< No parameters, disturbances or algebraic states supported for code generation. */
RET_ONLY_EQUIDISTANT_GRID_FOR_CODE_EXPORT,		/**< Only equidistant evaluation grids supported for  code generation. */
RET_ONLY_BOUNDS_FOR_CODE_EXPORT,				/**< Only state and control bounds supported for code generation. */
RET_QPOASES_EMBEDDED_NOT_FOUND,					/**< Embedded qpOASES code not found. */
RET_UNABLE_TO_EXPORT_STATEMENT,					/**< Unable to export statement due to incomplete definition. */
RET_INVALID_CALL_TO_EXPORTED_FUNCTION,			/**< Invalid call to export functions (check number of calling arguments). */


/* EXPORTED INTEGRATORS */
RET_INVALID_LINEAR_OUTPUT_FUNCTION				/**< Invalid definition of the nonlinear function in a linear output system. */
};

/** Defines the importance level of the message */
enum returnValueLevel
{
	LVL_DEBUG = 0,		///< Lowest level, the debug level.
	LVL_FATAL,			///< Returned value is a fatal error, assert like use, aborts execution is unhandled
	LVL_ERROR,			///< Returned value is a error
	LVL_WARNING,		///< Returned value is a warning
	LVL_INFO			///< Returned value is a information
};

/**
 *  \brief Allows to pass back messages to the calling function.
 *
 *	\ingroup BasicDataStructures
 *
 *	An instance of the class returnValue is returned by all ACADO functions for
 *	passing back messages to the calling function.
 *
 *  \author Martin Lauko, Hans Joachim Ferreau, Boris Houska
 */
class returnValue
{
public:

	/** Construct default returnValue.
	 *
	 */
	returnValue();

	/** Construct returnValue only from typedef.
	 *
	 */
	returnValue(returnValueType _type);

	/** Construct returnValue from int, for compatibility
	 *
	 */
	returnValue(int _type);

	/** Copy constructor with minimum performance cost.
	 *  Newly constructed instance takes ownership of data.
	 */
	returnValue(const returnValue& old);

	/** Constructor used by the ACADOERROR and similar macros.
	 *
	 */
	returnValue(const char* msg, returnValueLevel level = LVL_ERROR, returnValueType type = RET_UNKNOWN_BUG);

	/** Constructor used by the ACADOERROR and similar macros. Special case.
	 *  Constructs returnValue from old one, changing level and adding message
	 */
	returnValue(const char* msg, returnValueLevel level,const returnValue& old);

	/** Adds another message to the end of messages list.
	 *
	 */
	returnValue& addMessage(const char* msg);

	/** Change the importance level of the returned value
	 *
	 */
	returnValue& changeLevel(returnValueLevel level);

	/** Change the type of the returned message
	 *
	 */
	returnValue& changeType(returnValueType type);

	returnValueLevel getLevel() const;

	/** Prints all messages to the standard output.
	 *
	 */
	void print();

	/** Prints only the most basic information, no messages, to the standard output.
	 *
	 */
	void printBasic();

	/** Destroys data instance only if it owns it
	 *
	 */
	~returnValue();

	/** Compares the returnValue type to its enum
	 *
	 */
	bool operator!=(returnValueType cmp_type) const;

	/** Compares the returnValue type to its enum
	 *
	 */
	bool operator==(returnValueType cmp_type) const;

	/** Returns true if return value type is not SUCCESSFUL_RETURN
	 *
	 */
	bool operator!() const;

	/** Assignment operator.
	 *  Left hand side instance takes ownership of data.
	 */
	returnValue& operator=(const returnValue& old);

	/** Compatibility function, allows returnValue to be used as a number, similar to a enum.
     *
     */
	operator int();

private:
	returnValueType type;
	returnValueLevel level;
	int status;

	class returnValueData;
	returnValueData* data;
};

CLOSE_NAMESPACE_ACADO

#endif	// ACADO_TOOLKIT_ACADO_TYPES_HPP

/*
 *    end of file
 */
