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
 *    \file include/acado/optimization_algorithm/optimization_algorithm.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 */


#ifndef ACADO_TOOLKIT_OPTIMIZATION_ALGORITHM_HPP
#define ACADO_TOOLKIT_OPTIMIZATION_ALGORITHM_HPP

#include <acado/user_interaction/user_interaction.hpp>
#include <acado/optimization_algorithm/optimization_algorithm_base.hpp>

BEGIN_NAMESPACE_ACADO


/** 
 *	\brief User-interface to formulate and solve optimal control problems and static NLPs.
 *
 *	\ingroup UserInterfaces
 *
 *	The class OptimizationAlgorithm serves as a user-interface to formulate and
 *	solve optimal control problems and static nonlinear programming (NLP) problems.
 *
 *	\note Time is normalised to [0,1] when doing time optimal control.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class OptimizationAlgorithm : public OptimizationAlgorithmBase, public UserInteraction
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        OptimizationAlgorithm();

        /** Default constructor. */
        OptimizationAlgorithm( const OCP& ocp_ );

        /** Copy constructor (deep copy). */
        OptimizationAlgorithm( const OptimizationAlgorithm& arg );

        /** Destructor. */
        virtual ~OptimizationAlgorithm( );

        /** Assignment operator (deep copy). */
        OptimizationAlgorithm& operator=( const OptimizationAlgorithm& arg );


		/** Initializes the (internal) optimization algorithm part of the RealTimeAlgorithm.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_OPTALG_INIT_FAILED
		 */
		virtual returnValue init( );

        /** Starts execution. */
        virtual returnValue solve( );



    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

		/** Sets-up default options.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue setupOptions( );

		/** Sets-up default logging information.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue setupLogging( );

        virtual returnValue allocateNlpSolver(	Objective *F,
												DynamicDiscretization *G,
												Constraint *H
												);

        virtual returnValue initializeNlpSolver(	const OCPiterate& _userInit
													);

        virtual returnValue initializeObjective(	Objective* F
													);


    //
    // DATA MEMBERS:
    //
    protected:
};


CLOSE_NAMESPACE_ACADO



//#include <acado/optimization_algorithm/optimization_algorithm.ipp>


#endif  // ACADO_TOOLKIT_OPTIMIZATION_ALGORITHM_HPP

/*
 *   end of file
 */
