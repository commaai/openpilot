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
 *    \file include/acado/dynamic_discretization/dynamic_discretization.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 */


#ifndef ACADO_TOOLKIT_DYNAMIC_DISCRETIZATION_HPP
#define ACADO_TOOLKIT_DYNAMIC_DISCRETIZATION_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/algorithmic_base.hpp>

#include <acado/dynamic_system/dynamic_system.hpp>
#include <acado/variables_grid/grid.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/symbolic_expression/symbolic_expression.hpp>
#include <acado/variables_grid/variables_grid.hpp>
#include <acado/integrator/integrator.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Base class for discretizing a DifferentialEquation for use in optimal control algorithms.
 *
 *	\ingroup NumericalAlgorithms
 *
 *  The class DynamicDiscretization serves as a base class for discretizing 
 *	a DifferentialEquation for use in optimal control algorithms.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class DynamicDiscretization : public AlgorithmicBase
{
	//
	// PUBLIC MEMBER FUNCTIONS:
	//

	public:

		/** Default constructor. */
		DynamicDiscretization( );

		DynamicDiscretization(	UserInteraction* _userInteraction
								);

		/** Copy constructor (deep copy). */
		DynamicDiscretization( const DynamicDiscretization& rhs );

		/** Destructor. */
		virtual ~DynamicDiscretization( );

		/** Assignment operator (deep copy). */
		DynamicDiscretization& operator=( const DynamicDiscretization& rhs );

		/** Clone constructor (deep copy). */
		virtual DynamicDiscretization* clone() const = 0;



        /** Set the Differential Equations stage by stage. */
        virtual returnValue addStage( const DynamicSystem  &dynamicSystem_,
                                      const Grid           &stageIntervals,
                                      const IntegratorType &integratorType_ = INT_UNKNOWN ) = 0;

		/** Set the Transition stages. */
		virtual returnValue addTransition( const Transition& transition_ ) = 0;


        /** Deletes all stages and transitions and resets the DynamicDiscretization. */
        virtual returnValue clear() = 0;



		/** Define a forward seed in form of a block matrix.   \n
		*  Here, the block matrix should have N block rows,   \n
		*  where N is the number of points of the union grid. \n
		*  The i-th row is associated with the i-th grid      \n
		*  point in the union grid. Note that the direction   \n
		*  can itself be organized in sub-blocks as long as   \n
		*  all dimensions fit together.                       \n
		*                                                     \n
		*  \return SUCCESFUL RETURN                           \n
		*          RET_INPUT_OUT_OF_RANGE                     \n
		*/
		virtual returnValue setForwardSeed( const BlockMatrix &xSeed_,   /**< the seed in x-direction */
											const BlockMatrix &pSeed_,   /**< the seed in p-direction */
											const BlockMatrix &uSeed_,   /**< the seed in u-direction */
											const BlockMatrix &wSeed_    /**< the seed in w-direction */ );


		/**  Defines the first order forward seed to be         \n
		*   the unit-directions matrix.                        \n
		*                                                      \n
		*   \return SUCCESFUL_RETURN                           \n
		*           RET_INPUT_OUT_OF_RANGE                     \n
		*/
		virtual returnValue setUnitForwardSeed();



		/**  Define a backward seed in form of a block matrix.  \n
		*  Here, the block matrix should have N block columns, \n
		*  where N is the number of points of the union grid.  \n
		*  The i-th column is associated with the i-th grid    \n
		*  point in the union grid. Note that the directions   \n
		*  can itself be organized in sub-blocks as long as    \n
		*  all dimensions fit together.                        \n
		*                                                      \n
		*   \return SUCCESFUL_RETURN                           \n
		*           RET_INPUT_OUT_OF_RANGE                     \n
		*/
		virtual returnValue setBackwardSeed( const BlockMatrix &seed    /**< the seed matrix       */ );



		/**  Defines the first order backward seed to be        \n
		*   a unit matrix.                                     \n
		*                                                      \n
		*   \return SUCCESFUL_RETURN                           \n
		*           RET_INPUT_OUT_OF_RANGE                     \n
		*/
		virtual returnValue setUnitBackwardSeed();



		/** Deletes all seeds that have been set with the methods above.          \n
		*  This function will also give the corresponding memory free.           \n
		*  \return SUCCESSFUL_RETURN                                             \n
		*          RET_NO_SEED_ALLOCATED                                         \n
		*/
		virtual returnValue deleteAllSeeds();



		/** Evaluates the descretized DifferentialEquation at a specified     \n
		*  VariablesGrid. The results are written into the residuum of the   \n
		*  type VariablesGrid. This routine is for a simple evaluation only. \n
		*  If sensitivities are needed use one of the routines below         \n
		*  instead.                                                          \n
		*                                                                    \n
		*  \return SUCCESSFUL_RETURN                                         \n
		*          RET_INVALID_ARGUMENTS                                     \n
		*          or a specific error message form an underlying            \n
		*          discretization instance.                                  \n
		*/
		virtual returnValue evaluate( OCPiterate &iter ) = 0;



		/** Evaluates the sensitivities.                                      \n
		*                                                                    \n
		*  \return SUCCESSFUL_RETURN                                         \n
		*          RET_NOT_FROZEN                                            \n
		*/
		virtual returnValue evaluateSensitivities( ) = 0;



		/** Evaluates the sensitivities and the hessian.  \n
		*                                                \n
		*  \return SUCCESSFUL_RETURN                     \n
		*          RET_NOT_FROZEN                        \n
		*/
		virtual returnValue evaluateSensitivities( const BlockMatrix &seed, BlockMatrix &hessian ) = 0;



		/** Evaluates the sensitivities.                                      \n
		*                                                                    \n
		*  \return SUCCESSFUL_RETURN                                         \n
		*          RET_NOT_FROZEN                                            \n
		*/
		virtual returnValue evaluateSensitivitiesLifted( ) = 0;




		/** Returns the result for the residuum. \n
		*                                       \n
		*  \return SUCCESSFUL_RETURN            \n
		*/
		virtual returnValue getResiduum( BlockMatrix &residuum_ /**< the residuum */ ) const;



		/** Returns the result for the forward sensitivities in BlockMatrix form.       \n
		*                                                                               \n
		*  \return SUCCESSFUL_RETURN                                                    \n
		*          RET_INPUT_OUT_OF_RANGE                                               \n
		*/
		virtual returnValue getForwardSensitivities( BlockMatrix &D  /**< the result for the
																	  *   forward sensitivi-
																	  *   ties               */ ) const;



		/** Returns the result for the backward sensitivities in BlockMatrix form.       \n
		*                                                                               \n
		*  \return SUCCESSFUL_RETURN                                                    \n
		*          RET_INPUT_OUT_OF_RANGE                                               \n
		*/
		virtual returnValue getBackwardSensitivities( BlockMatrix &D  /**< the result for the
																		*   forward sensitivi-
																		*   ties              */ ) const;



		/**< Returns the total number of intervals. */
		inline int getNumberOfIntervals( ) const;



		virtual BooleanType isAffine( ) const = 0;


        virtual returnValue unfreeze( ) = 0;


	//
	// PROTECTED MEMBER FUNCTIONS:
	//

	protected:
		virtual returnValue setupOptions( );
		virtual returnValue setupLogging( );

		void copy( const DynamicDiscretization& rhs );
		void initializeVariables();

        uint getNumEvaluationPoints() const;


	//
	// PROTECTED MEMBERS:
	//

	protected:

		Grid          unionGrid       ;  /**< the union grids on the stages                */
		int           N               ;  /**< total number of grid points                  */
		PrintLevel    printLevel      ;  /**< the print level                              */
		BooleanType   freezeTraj      ;  /**< whether the trajectory should be frozen      */


		// DIMENSIONS:
		// -----------
		int        nx;
		int        na;
		int        np;
		int        nu;
		int        nw;


		// INPUT STORAGE:
		// ------------------------
		BlockMatrix       xSeed   ;   /**< the 1st order forward seed in x-direction */
		BlockMatrix       pSeed   ;   /**< the 1st order forward seed in p-direction */
		BlockMatrix       uSeed   ;   /**< the 1st order forward seed in u-direction */
		BlockMatrix       wSeed   ;   /**< the 1st order forward seed in w-direction */

		BlockMatrix       bSeed   ;   /**< the 1st order backward seed */


		// RESULTS:
		// ------------------------
		VariablesGrid    residuum ;   /**< the residuum vectors                 */
		BlockMatrix      dForward ;   /**< the first order forward  derivatives */
		BlockMatrix      dBackward;   /**< the first order backward derivatives */

};


CLOSE_NAMESPACE_ACADO



#include <acado/dynamic_discretization/dynamic_discretization.ipp>

//#include <acado/dynamic_discretization/simulation_algorithm.hpp>
//#include <acado/dynamic_discretization/simulation_by_integration.hpp>
//#include <acado/dynamic_discretization/simulation_by_collocation.hpp>
#include <acado/dynamic_discretization/shooting_method.hpp>
#include <acado/dynamic_discretization/collocation_method.hpp>


#endif  // ACADO_TOOLKIT_DYNAMIC_DISCRETIZATION_HPP


// end of file
