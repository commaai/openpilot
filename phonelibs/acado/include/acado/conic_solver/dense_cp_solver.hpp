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
 *    \file include/acado/conic_solver/dense_cp_solver.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_DENSE_CP_SOLVER_HPP
#define ACADO_TOOLKIT_DENSE_CP_SOLVER_HPP

#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/algorithmic_base.hpp>

#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/conic_program/dense_cp.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Base class for algorithms solving conic programs.
 *
 *	\ingroup AlgorithmInterfaces
 *
 *  The class Dense CP Solver is a base class for all
 *  conic solvers that are able to solve conic problems.
 *
 *  \author Boris Houska, Hans Joachim Ferreau
 */

class DenseCPsolver : public AlgorithmicBase
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        DenseCPsolver( );

        DenseCPsolver(	UserInteraction* _userInteraction
						);

        /** Copy constructor (deep copy). */
        DenseCPsolver( const DenseCPsolver& rhs );

        /** Destructor. */
        virtual ~DenseCPsolver( );

        /** Assignment operator (deep copy). */
        DenseCPsolver& operator=( const DenseCPsolver& rhs );


        virtual DenseCPsolver* clone( ) const = 0;


        /** initializes the dense conic solver */
        virtual returnValue init( const DenseCP *cp_ ) = 0;


        /** Solves the CP */
        virtual returnValue solve( DenseCP *cp_ ) = 0;


        /** Returns a variance-covariance estimate if possible or an error message otherwise.
         *
         *  \return SUCCESSFUL_RETURN
         *          RET_MEMBER_NOT_INITIALISED
         */
        virtual returnValue getVarianceCovariance( DMatrix &var ) = 0;


        /** Returns a variance-covariance estimate if possible or an error message otherwise.
         *
         *  \return SUCCESSFUL_RETURN
         *          RET_MEMBER_NOT_INITIALISED
         */
        virtual returnValue getVarianceCovariance( DMatrix &H, DMatrix &var ) = 0;



        /** Returns number of iterations performed at last QP solution. \n
         *                                                              \n
         *  \return Number of iterations performed at last QP solution  \n
         */
        virtual uint getNumberOfIterations( ) const = 0;


		
    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

		virtual returnValue setupOptions( );
		virtual returnValue setupLogging( );

};


CLOSE_NAMESPACE_ACADO


//#include <acado/conic_solver/dense_cp_solver.ipp>


#endif  // ACADO_TOOLKIT_DENSE_CP_SOLVER_HPP

/*
 *  end of file
 */
