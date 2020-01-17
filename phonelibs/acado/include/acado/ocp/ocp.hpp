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
 *    \file include/acado/ocp/ocp.hpp
 *    \authors Boris Houska, Hans Joachim Ferreau, Milan Vukov, Rien Quirynen
 *    \date 2008 - 2014
 */

#ifndef ACADO_TOOLKIT_OCP_HPP
#define ACADO_TOOLKIT_OCP_HPP

#include <acado/ocp/multi_objective_functionality.hpp>
#include <acado/ocp/model_container.hpp>

BEGIN_NAMESPACE_ACADO

class Grid;
class Objective;
class Constraint;

/** 
 *	\brief Data class for defining optimal control problems.
 *
 *	\ingroup BasicDataStructures
 *
 *	The class OCP is a data class for defining optimal control problems.
 *  In the most easiest an optimal control problem can consists of an
 *  objective only - i.e. in principle we can set up NLP's as well if no
 *  dynamic system is specified. However, in general the objective
 *  functional is optimized subject to a dynamic equation and different
 *  kind of constraints. \n
 *  \n
 *  Note that the OCP class is only designed for the user and collects
 *  data only. In order to solve an OCP this class must be setup and
 *  passed to an appropriate OptimizationAlgorithm.\n
 *  \n
 *  For setting up an optimal control problem (OCP), we should first specify
 *  the time horizon on which the OCP is defined. Note that there are
 *  several constructors available which allow to construct an OCP directly
 *  with the corresponding time interval. Here, the interval can consist of
 *  of given bounds, but in another variant a parameter can be passed in
 *  order to allow the setup of optimal control problems for which the end
 *  time is optimized, too.\n
 *  \n
 *  Constraints can be specified with the "subjectTo" syntax. Please note
 *  that every parameter, state, control etc which is not fixed via a
 *  constraint will be regarded as an optimization variable. In particular,
 *  initial value or boundary constraints appear in many OCP formulations
 *  and of course all these constraints should all be set explicitly.
 *  Moreover, the dynamic equations (model) is regarded as a constraint, too.\n
 *  \n
 *  Please note that the OCP class only collects the formulation of the
 *  problem. If initial values for non-linear problems should be specified,
 *  this needs to be done on the algorithm dealing with the OCP.
 *  (\sa OptimizationAlgorithm)\n
 *  \n
 *  For advanced users and developers it might be important to know that the
 *  class OCP inherits the MultiObjectiveFunctionality which is needed if
 *  more than one objective should be specified.\n
 *  \n
 *  Please check the tutorial examples as well as the class reference below,
 *  to learn about the usage of this class in more detail.\n
 *  \n
 */

class OCP: public MultiObjectiveFunctionality, public ModelContainer
{
public:

	/** Default Constructor which can optionally set the time-horizon of the problem.
	 */
	OCP(	const double &tStart_ = 0.0,  /**< start of the time horizon of the OCP */
			const double &tEnd_   = 1.0,  /**< end   of the time horizon of the OCP */
			const int    &N_      = 20    /**< number of discretization intervals   */ );


	/** Constructor which can set a non equidistant time-horizon of the problem.
	 */
	OCP( 	const double &tStart_,  		/**< start of the time horizon of the OCP */
			const double &tEnd_,  			/**< end   of the time horizon of the OCP */
			const DVector& _numSteps   		/**< number of integration steps in each discretization interval   */ );


	/** Constructor that takes a parametric version of the time horizon. This contructor
	 *  should be used if the end time should be optimized, too.
	 */
	OCP( 	const double    &tStart_,  /**< start of the time horizon of the OCP */
			const Parameter &tEnd_,    /**< end   of the time horizon of the OCP */
			const int       &N_ = 20   /**< number of discretization intervals   */ );


	/** Constructor that takes the time horizon in form of a Grid.
	 */
	OCP(	const Grid &grid_  /**< discretization grid  */ );


	/** Destructor (deletes everything). */
	virtual ~OCP( );

	/** Adds an expression as a the Mayer term to be minimized.
	 *  \return SUCCESSFUL_RETURN
	 */
	returnValue minimizeMayerTerm( const Expression& arg );


	/** Adds an expression as a the Mayer term to be minimized. In this version
	 *  the number of the objective can be specified as well. This functionality
	 *  is needed for multi-objective optimal control problems.
	 *  \return SUCCESSFUL_RETURN
	 */
	returnValue minimizeMayerTerm( const int &multiObjectiveIdx,  const Expression& arg );


	/** Adds an expression as a the Mayer term to be maximized.
	 *  Note that this function is introduced for convenience only.
	 *  A call of the form maximizeMayerTerm( arg ) is equivalent to
	 *  calling minimizeMayerTerm( -arg ).\n
	 *
	 *  \return SUCCESSFUL_RETURN
	 */
	returnValue maximizeMayerTerm( const Expression& arg );


	/** Adds an expression as a the Lagrange term to be minimized.
	 *  \return SUCCESSFUL_RETURN
	 */
	returnValue minimizeLagrangeTerm( const Expression& arg );


	/** Adds an expression as a the Lagrange term to be maximized.
	 *  \return SUCCESSFUL_RETURN
	 */
	returnValue maximizeLagrangeTerm( const Expression& arg );

	/** \name Least Squares terms.
	 *
	 *  Adds a Least Square term of the form
	 *
	 *  \f{equation*}{
	 *    \frac{1}{2} \sum_i \Vert h(t_i,x(t_i),u(t_i),p(t_i),...) - r \Vert^2_{S_{i}}
	 *  \f}
	 *
	 *  Here the sum is over all grid points of the objective grid. The
	 *  DMatrix \f$ S \f$ is assumed to be symmetric and positive (semi-) definite.
	 *  The Function \f$ r \f$ is called reference and can be
	 *  specified by the user. The function \f$ h \f$ is a standard Function.
	 *
	 *  \sa Function
	 *
	 *  \return SUCCESSFUL_RETURN
	 *
	 *  @{ */
	returnValue minimizeLSQ(	const DMatrix   &S,   /**< a weighting matrix */
								const Function  &h,   /**< the LSQ-Function   */
								const DVector   &r    /**< the reference      */ );

	returnValue minimizeLSQ(	const Function &h,   /**< the LSQ-Function   */
								const DVector  &r    /**< the reference      */ );

	returnValue minimizeLSQ(	const Function &h    /**< the LSQ-Function   */ );

	returnValue minimizeLSQ(	const MatrixVariablesGrid &S,   /**< a weighting matrix */
								const Function            &h,   /**< the LSQ-Function   */
								const VariablesGrid       &r    /**< the reference      */ );

	returnValue minimizeLSQ(	const DMatrix       &S,   /**< a weighting matrix */
								const Function      &h,   /**< the LSQ-Function   */
								const VariablesGrid &r    /**< the reference      */ );

	returnValue minimizeLSQ(	const Function      &h,	/**< the LSQ-Function   */
								const VariablesGrid &r	/**< the reference      */ );

	returnValue minimizeLSQ(	const MatrixVariablesGrid &S,	/**< a weighting matrix */
								const Function            &h,	/**< the LSQ-Function   */
								const char*        rFilename	/**< filename where the reference is stored */ );

	returnValue minimizeLSQ(	const DMatrix        &S,/**< a weighting matrix */
								const Function      &h,	/**< the LSQ-Function   */
								const char*  rFilename	/**< filename where the reference is stored */ );

	returnValue minimizeLSQ(	const Function      &h,   /**< the LSQ-Function */
								const char*  rFilename    /**< filename where the reference is stored */ );

	/** \note Applicable only for automatic code generation. */
	returnValue minimizeLSQ(	const DMatrix& S,	/**< a weighting matrix */
								const Function& h	/**< the LSQ-Function   */ );

	/** Pass the sparsity pattern of the weighting matrix to the code generator.
	 *  \note Applicable only for automatic code generation. */
	returnValue minimizeLSQ(	const BMatrix& S,	/**< a weighting matrix */
								const Function& h	/**< the LSQ-Function   */ );

	/** \note Applicable only for automatic code generation. */
	returnValue minimizeLSQ(	const DMatrix& S,		/**< a weighting matrix */
								const std::string& h	/**< the externally defined LSQ-Function   */ );

	/** Pass the sparsity pattern of the weighting matrix to the code generator.
	 *  \note Applicable only for automatic code generation. */
	returnValue minimizeLSQ(	const BMatrix& S,		/**< a weighting matrix */
								const std::string& h	/**< the externally defined LSQ-Function   */ );

	/** @} */

	/** \name Least Squares end terms.
	 *
	 *  Adds an Least Square term that is only evaluated at the end:
	 *
	 *  \f{equation*}{
	 *        \frac{1}{2} \Vert( m(T,x(T),u(T),p(T),...) - r )\Vert^2_S
	 *  \f}
	 *
	 *  where  \f$ S \f$ is a weighting matrix, \f$ r \f$ a reference vector
	 *  and \f$ T \f$ the time at the last objective grid point. The function
	 *  \f$ m \f$ is a standard Function.
	 *
	 *  \return SUCCESSFUL_RETURN
	 *
	 *  @{ */
	returnValue minimizeLSQEndTerm( const DMatrix  & S,	/**< a weighting matrix */
									const Function & m,	/**< the LSQ-Function   */
									const DVector  & r	/**< the reference      */ );

	returnValue minimizeLSQEndTerm( const Function & m,	/**< the LSQ-Function   */
									const DVector  & r	/**< the reference      */ );

	/** \note Applicable only for automatic code generation. */
	returnValue minimizeLSQEndTerm(	const DMatrix& S,		/**< a weighting matrix */
									const Function& m		/**< the LSQ-Function   */ );

	/** \note Applicable only for automatic code generation. */
	returnValue minimizeLSQEndTerm(	const DMatrix& S,		/**< a weighting matrix */
									const std::string& m	/**< the externally defined LSQ-Function   */ );

	/** Pass the sparsity pattern of the weighting matrix to the code generator.
	 *  \note Applicable only for automatic code generation. */
	returnValue minimizeLSQEndTerm(	const BMatrix& S,		/**< a weighting matrix */
									const Function& m		/**< the LSQ-Function   */ );

	/** Pass the sparsity pattern of the weighting matrix to the code generator.
	 *  \note Applicable only for automatic code generation. */
	returnValue minimizeLSQEndTerm(	const BMatrix& S,		/**< a weighting matrix */
									const std::string& m	/**< the externally defined LSQ-Function   */ );

	/** @} */

	/** \name Set linear terms in the LSQ formulation.
	 *  @{ */

	/** Applicable only for automatic code generation.
	 *  \note Experimental. */
	returnValue minimizeLSQLinearTerms(	const DVector& Slx,	/**< a weighting vector for differential states. */
										const DVector& Slu	/**< a weighting vector for controls. */ );

	/** Applicable only for automatic code generation.
	 *  \note Experimental. */
	returnValue minimizeLSQLinearTerms(	const BVector& Slx,	/**< a weighting vector for differential states. */
										const BVector& Slu	/**< a weighting vector for controls. */ );
	/** @} */

	/** Adds an differential equation (as a continuous equality constraint). \n
	 *                                                                      \n
	 *  \param differentialEquation_ the differential equation to be added  \n
	 *  \param n_                    the number of control intervals        \n
	 *                                                                      \n
	 *  \return SUCCESSFUL_RETURN
	 */
	returnValue subjectTo( const DifferentialEquation& differentialEquation_ );


	/**  Adds a (continuous) constraint.
	 *  \return SUCCESSFUL_RETURN                     \n
	 *          RET_INFEASIBLE_CONSTRAINT             \n
	 */
	returnValue subjectTo( const ConstraintComponent& component );


	/**< Adds a discrete, point, constraint.
	 *  \return SUCCESSFUL_RETURN                     \n
	 *          RET_INFEASIBLE_CONSTRAINT             \n
	 */
	returnValue subjectTo( int index_, const ConstraintComponent& component );


	/** Add a coupled boundary constraint.
	 *
	 *  Coupled boundary constraints of general form
	 *
	 *  \f{equation*}{
	 *  	\text{lb} \leq  h_1( t_0,x(t_0),u(t_0),p,... ) + h_2( t_e,x(t_e),u(t_e),p,... ) \leq \text{ub}
	 *  \f}
	 *
	 *  where \f$ t_0 \f$ is the first and \f$ t_e \f$ the last time point in the grid.
	 *  Adds a constraint of the form  \verbatim lb <= arg1(0) + arg2(T) <= ub \endverbatim
	 *  with constant lower and upper bounds.
	 *
	 *   \return  SUCCESSFUL_RETURN \n
	 *            RET_INFEASIBLE_CONSTRAINT
	 */
	returnValue subjectTo(	const double lb_,
							const Expression& arg1,
							const Expression& arg2,
							const double ub_ );


	/** Add a custom constraint.
	 *
	 *  Adds a constraint of the form
	 *
	 *  \f{equation*}{
	 *    \text{lb} <= \sum_i h_i(t_i, x(t_i), u(t_i), p, ...) <= \text{ub}
	 *  \f}
	 *
	 *   with constant lower and upper bounds.
	 *
	 *   \return  SUCCESSFUL_RETURN \n
	 *            RET_INFEASIBLE_CONSTRAINT
	 *
	 * @{
	 */
	returnValue subjectTo( const double lb_, const Expression *arguments, const double ub_ );

	/** \name Add a custom constraint to the OCP formulation.
	 *
	 *  Adds a constraint of the form
	 *
	 *  \f{equation*}{
	 *    \text{lb} <= h_i(t_i, x(t_i), u(t_i), p, ...) <= \text{ub}
	 *  \f}
	 *
	 *   with constant lower and upper bounds.
	 *
	 * @{
	 */
	returnValue subjectTo( const DVector& _lb, const Expression& _expr, const DVector& _ub );

	returnValue subjectTo( int _index, const DVector& _lb, const Expression& _expr, const DVector& _ub );

	/** @} */


	/** \name Helper functions.
	 *
	 *  @{ */
	BooleanType hasObjective           () const;
	BooleanType hasConstraint          () const;

	returnValue getGrid                ( Grid&      grid_                               ) const;
	returnValue getObjective           ( Objective& objective_                          ) const;
	returnValue getObjective           ( const int &multiObjectiveIdx, Expression **arg ) const;

	returnValue getConstraint( Constraint& constraint_ ) const;

	returnValue setObjective ( const Objective & objective_  );
	returnValue setConstraint( const Constraint& constraint_ );
	returnValue setNumberIntegrationSteps( const uint numSteps );

	/** Returns whether the ocp grid is equidistant.
	 *
	 * \return true  iff the OCP grid is equidistant, false otherwise.
	 */
	virtual BooleanType hasEquidistantGrid( ) const;

	double getStartTime ( ) const;
	double getEndTime( ) const;
	/** @} */

protected:

	void setupGrid( double tStart, double tEnd, int N );
	void setupGrid( const DVector& times );

	/** Common discretization grid. */
	std::shared_ptr<Grid> grid;
	/** The Objective. */
	std::shared_ptr<Objective> objective;
	/** The Constraints. */
	std::shared_ptr<Constraint> constraint;
};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_OCP_HPP
