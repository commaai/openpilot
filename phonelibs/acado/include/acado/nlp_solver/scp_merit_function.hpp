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
 *    \file include/acado/nlp_solver/scp_merit_function.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_SCP_MERIT_FUNCTION_HPP
#define ACADO_TOOLKIT_SCP_MERIT_FUNCTION_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/algorithmic_base.hpp>

#include <acado/function/ocp_iterate.hpp>
#include <acado/conic_program/banded_cp.hpp>
#include <acado/conic_solver/banded_cp_solver.hpp>

#include <acado/nlp_solver/scp_evaluation.hpp>




BEGIN_NAMESPACE_ACADO


/**
 *	\brief Allows to evaluate a merit function within an SCPmethod for solving NLPs.
 *
 *	\ingroup NumericalAlgorithms
 *
 *  The class SCPmeritFunction allows to evaluate a merit function within
 *	SCPmethods for solving nonlinear programming problems.
 *
 *  \author Boris Houska, Hans Joachim Ferreau
 */
class SCPmeritFunction : public AlgorithmicBase
{

    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        SCPmeritFunction( );

		SCPmeritFunction(	UserInteraction* _userInteraction
							);

        /** Copy constructor (deep copy). */
        SCPmeritFunction( const SCPmeritFunction& rhs );

        /** (virtual) destructor. */
        virtual ~SCPmeritFunction( );

        /** Assignment operator (deep copy). */
        SCPmeritFunction& operator=( const SCPmeritFunction& rhs );

        virtual SCPmeritFunction* clone( ) const;


        /** Evaluates the merit function M(alpha) := T( x_k + alpha * Delta x_k ) \n
         *  where the step size parameter "alpha" can be specified.               \n
         *                                                                        \n
         *  \return The value M(alpha) of the merit function at alpha or INFTY if \n
         *          the function evaluation was not successful.                   \n
         */
        virtual returnValue evaluate(	double alpha,
        								const OCPiterate& iter,
        								BandedCP& cp,
										SCPevaluation& eval,
										double& result
        								);


		//virtual updateWeights( ) = 0;


    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:



    //
    // DATA MEMBERS:
    //
    protected:

    	//double functionWeight;
       	//double dynamicWeight;
    	//double equalityWeight;
    	//double boundWeight;
    	//double constraintWeight;


};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_SCP_MERIT_FUNCTION_HPP

/*
 *  end of file
 */
