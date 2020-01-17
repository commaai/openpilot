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
 *    \file include/acado/optimization_algorithm/parameter_estimation_algorithm.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_PARAMETER_ESTIMATION_ALGORITHM_HPP
#define ACADO_TOOLKIT_PARAMETER_ESTIMATION_ALGORITHM_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/variables_grid/variables_grid.hpp>
#include <acado/ocp/ocp.hpp>
#include <acado/nlp_solver/nlp_solver.hpp>
#include <acado/optimization_algorithm/optimization_algorithm.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief User-interface to formulate and solve parameter estimation problems.
 *
 *	\ingroup UserInterfaces
 *
 *	The class ParameterEstimationAlgorithm serves as a user-interface to formulate and
 *  solve parameter estimation problems.
 *
 *  \author Boris Houska, Hans Joachim Ferreau
 */
class ParameterEstimationAlgorithm : public OptimizationAlgorithm {

    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        ParameterEstimationAlgorithm();

        /** Default constructor. */
        ParameterEstimationAlgorithm( const OCP& ocp_ );

        /** Copy constructor (deep copy). */
        ParameterEstimationAlgorithm( const ParameterEstimationAlgorithm& arg );

        /** Destructor. */
        virtual ~ParameterEstimationAlgorithm( );

        /** Assignment operator (deep copy). */
        ParameterEstimationAlgorithm& operator=( const ParameterEstimationAlgorithm& arg );



        /** Method to obtain the variance-coveriance matrix in the optimal solution  \n
         *  (with respect to the parameters)                                         \n
         *  \return SUCCESSFUL_RETURN
         */
        returnValue getParameterVarianceCovariance( DMatrix &pVar );


        /** Method to obtain the variance-coveriance matrix in the optimal solution  \n
         *  (with respect to the parameters)                                         \n
         *  \return SUCCESSFUL_RETURN
         */
        returnValue getDifferentialStateVarianceCovariance( DMatrix &xVar );


        /** Method to obtain the variance-coveriance matrix in the optimal solution  \n
         *  (with respect to the parameters)                                         \n
         *  \return SUCCESSFUL_RETURN
         */
        returnValue getAlgebraicStateVarianceCovariance( DMatrix &xaVar );


        /** Method to obtain the variance-coveriance matrix in the optimal solution  \n
         *  (with respect to the parameters)                                         \n
         *  \return SUCCESSFUL_RETURN
         */
        returnValue getControlCovariance( DMatrix &uVar );


        /** Method to obtain the variance-coveriance matrix in the optimal solution  \n
         *  (with respect to the parameters)                                         \n
         *  \return SUCCESSFUL_RETURN
         */
        returnValue getDistubanceVarianceCovariance( DMatrix &wVar );


        /** Method to obtain the variance-coveriance matrix in the optimal solution  \n
         *                                                                           \n
         *  \return SUCCESSFUL_RETURN
         */
        returnValue getVarianceCovariance( DMatrix &var );




    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

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



#include <acado/optimization_algorithm/parameter_estimation_algorithm.ipp>


#endif  // ACADO_TOOLKIT_PARAMETER_ESTIMATION_ALGORITHM_HPP

/*
 *   end of file
 */
