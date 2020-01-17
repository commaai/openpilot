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
 *    \file include/acado/dynamic_discretization/shooting_method.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */

#ifndef ACADO_TOOLKIT_SHOOTING_METHOD_HPP
#define ACADO_TOOLKIT_SHOOTING_METHOD_HPP


#include <acado/dynamic_discretization/dynamic_discretization.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Discretizes a DifferentialEquation by means of single or multiple shooting.
 *
 *	\ingroup NumericalAlgorithms
 *
 *  The class ShootingMethod allows to discretize a DifferentialEquation 
 *	for use in optimal control algorithms by means of an (online) integrator
 *	using either single or multiple shooting.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class ShootingMethod : public DynamicDiscretization
{
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor. */
		ShootingMethod();

		ShootingMethod(	UserInteraction* _userInteraction
						);

		/** Copy constructor (deep copy). */
		ShootingMethod( const ShootingMethod& rhs );

		/** Destructor. */
		virtual ~ShootingMethod( );

		/** Assignment operator (deep copy). */
		ShootingMethod& operator=( const ShootingMethod& rhs );

		/** Clone constructor (deep copy). */
		virtual DynamicDiscretization* clone() const;


        /** Set the Differential Equations stage by stage. */
        virtual returnValue addStage( const DynamicSystem  &dynamicSystem_,
                                      const Grid           &stageIntervals,
                                      const IntegratorType &integratorType_ = INT_UNKNOWN );

		/** Set the Transition stages. */
		virtual returnValue addTransition( const Transition& transition_ );


        /** Deletes all stages and transitions and resets the DynamicDiscretization. */
        virtual returnValue clear();



		/** Evaluates the discretized DifferentialEquation at a specified     \n
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
		virtual returnValue evaluate(	OCPiterate &iter
										);


		/** Deletes all seeds that have been set with the methods above.          \n
		*  This function will also give the corresponding memory free.           \n
		*  \return SUCCESSFUL_RETURN                                             \n
		*          RET_NO_SEED_ALLOCATED                                         \n
		*/
		virtual returnValue deleteAllSeeds( );


		/** Evaluates the sensitivities.                                      \n
		*                                                                    \n
		*  \return SUCCESSFUL_RETURN                                         \n
		*          RET_NOT_FROZEN                                            \n
		*/
		virtual returnValue evaluateSensitivities( );


		/** Evaluates the sensitivities.                                      \n
		*                                                                    \n
		*  \return SUCCESSFUL_RETURN                                         \n
		*          RET_NOT_FROZEN                                            \n
		*/
		virtual returnValue evaluateSensitivitiesLifted( );


		/** Evaluates the sensitivities and the hessian.  \n
		*                                                \n
		*  \return SUCCESSFUL_RETURN                     \n
		*          RET_NOT_FROZEN                        \n
		*/
		virtual returnValue evaluateSensitivities(	const BlockMatrix &seed,
													BlockMatrix &hessian
													);


        virtual returnValue unfreeze( );


		virtual BooleanType isAffine( ) const;


		//
		// PROTECTED MEMBER FUNCTIONS:
		//

		protected:

			returnValue deleteAll( );

			void copy( const ShootingMethod &arg );

            returnValue allocateIntegrator( uint idx, IntegratorType type_ );


            returnValue differentiateBackward( const int    &idx ,
                                               const DMatrix &seed,
                                                     DMatrix &Gx  ,
                                                     DMatrix &Gp  ,
                                                     DMatrix &Gu  ,
                                                     DMatrix &Gw    );

            returnValue differentiateForward(  const int     &idx,
                                               const DMatrix  &dX ,
                                               const DMatrix  &dP ,
                                               const DMatrix  &dU ,
                                               const DMatrix  &dW ,
                                                     DMatrix  &D    );


            returnValue differentiateForwardBackward( const int     &idx ,
                                                      const DMatrix  &dX  ,
                                                      const DMatrix  &dP  ,
                                                      const DMatrix  &dU  ,
                                                      const DMatrix  &dW  ,
                                                      const DMatrix  &seed,
                                                            DMatrix  &D   ,
                                                            DMatrix  &ddX ,
                                                            DMatrix  &ddP ,
                                                            DMatrix  &ddU ,
                                                            DMatrix  &ddW   );


            returnValue update( DMatrix &G, const DMatrix &A, const DMatrix &B );


			/**< Writes the continous integrator output to the logging object, if this     \n
			*   is requested. Please note, that this routine converts the VariablesGrids  \n
			*   from the integration routine into a large matrix. Consequently, the break \n
			*   points need to be logged, too.                                            \n
			*                                                                             \n
			*   \return SUCCESSFUL_RETURN                                                 \n
			*/
			returnValue logTrajectory( const OCPiterate &iter );

			returnValue rescale(	VariablesGrid* trajectory,
									double tEndNew,
									double newIntervalLength
									) const;

        //
        // PROTECTED MEMBERS:
        //

        protected:

            Integrator **integrator;
            DMatrix       breakPoints;
};


CLOSE_NAMESPACE_ACADO



#include <acado/dynamic_discretization/shooting_method.ipp>


#endif  // ACADO_TOOLKIT_SHOOTING_METHOD_HPP

// end of file
