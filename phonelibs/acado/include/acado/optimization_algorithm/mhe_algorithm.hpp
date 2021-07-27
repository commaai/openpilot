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
 *    \file include/acado/optimization_algorithm/mhe_algorithm.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_MHE_ALGORITHM_HPP
#define ACADO_TOOLKIT_MHE_ALGORITHM_HPP


#include <acado/optimization_algorithm/optimization_algorithm.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief User-interface to formulate and solve moving horizon estimation problems.
 *
 *	\ingroup UserInterfaces
 *
 *	The class MHEalgorithm serves as a user-interface to formulate and
 *  solve moving horizon estimation problems.
 *
 *  \author Boris Houska, Hans Joachim Ferreau
 */
class MHEalgorithm : public OptimizationAlgorithm {

    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        MHEalgorithm();

        /** Default constructor. */
        MHEalgorithm( const OCP& ocp_ );

        /** Copy constructor (deep copy). */
        MHEalgorithm( const MHEalgorithm& arg );

        /** Destructor. */
        virtual ~MHEalgorithm( );

        /** Assignment operator (deep copy). */
        MHEalgorithm& operator=( const MHEalgorithm& arg );


        /** Initializes the MHE Algorithm.                                    \n
         *                                                                    \n
         *  \param  eta   the initial measurement                             \n
         *  \param  S     the variance-covariance of the initial measurement  \n
         *                                                                    \n
         *  \return SUCCESSFUL_RETURN                                         \n
         */
        virtual returnValue init( const DVector &eta,
                                  const DMatrix &S    );


        /** Executes next single step                                         \n
         *                                                                    \n
         *  \param  eta   the current measurement                             \n
         *  \param  S     the variance-covariance of the current measurement  \n
         *                                                                    \n
         *  \return SUCCESSFUL_RETURN                                         \n
         */
        virtual returnValue step( const DVector &eta,
                                  const DMatrix &S    );


        /** Shifts the data for the preparation of the next step.
         */
        virtual returnValue shift( );


        /** Solves current problem.                                           \n
         *                                                                    \n
         *  \param  eta   the current measurement                             \n
         *  \param  S     the variance-covariance of the current measurement  \n
         *                                                                    \n
         *  \return SUCCESSFUL_RETURN                                         \n
         */
        virtual returnValue solve( const DVector &eta,
                                   const DMatrix &S    );


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

        DVector *eta;  // deep copy of the latest initial value.
        DMatrix *S  ;  // deep copy of the latest parameter.
};


CLOSE_NAMESPACE_ACADO



//#include <acado/optimization_algorithm/mhe_algorithm.ipp>


#endif  // ACADO_TOOLKIT_MHE_ALGORITHM_HPP

/*
 *   end of file
 */
