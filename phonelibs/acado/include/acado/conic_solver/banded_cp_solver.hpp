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
 *    \file include/acado/conic_solver/banded_cp_solver.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_BANDED_CP_SOLVER_HPP
#define ACADO_TOOLKIT_BANDED_CP_SOLVER_HPP

#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/algorithmic_base.hpp>

#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/conic_program/banded_cp.hpp>




BEGIN_NAMESPACE_ACADO


/**
 *	\brief Base class for algorithms solving banded conic programs arising in optimal control.
 *
 *	\ingroup AlgorithmInterfaces
 *
 *  The class Banded CP Solver is a base class for all
 *  conic solvers that are able to deal with the specific
 *  band structure that arises in the optimal control context.
 *
 *  \author Boris Houska, Hans Joachim Ferreau
 */



class BandedCPsolver: public AlgorithmicBase
{

    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        BandedCPsolver( );

		BandedCPsolver(	UserInteraction* _userInteraction
						);
		
        /** Copy constructor (deep copy). */
        BandedCPsolver( const BandedCPsolver& rhs );

        /** Destructor. */
        virtual ~BandedCPsolver( );

        /** Assignment operator (deep copy). */
        BandedCPsolver& operator=( const BandedCPsolver& rhs );


        /** Assignment operator (deep copy). */
        virtual BandedCPsolver* clone() const = 0;


        /** Initializes the banded conic solver */
        virtual returnValue init( const OCPiterate &iter_ ) = 0;



        /** Solves a given banded conic program */
        virtual returnValue prepareSolve(	BandedCP& cp
											);

		/** Solves a given banded conic program in feedback mode:                   \n
         *                                                                          \n
         *  \param cp     the banded conic program to be solved                     \n
         *  \param DeltaX difference between state estimate and previous prediction \n
         *  \param DeltaP difference between current and previous parameter value   \n
         *                                                                          \n
         *  \return SUCCESSFUL_RETURN   (if successful)                             \n
         *          or a specific error message from the dense CP solver.           \n
         */
        virtual returnValue solve(	BandedCP& cp
									) = 0;

        /** Solves a given banded conic program */
        virtual returnValue finalizeSolve(	BandedCP& cp
											);


		virtual returnValue getParameters        ( DVector        &p_  ) const = 0;
		virtual returnValue getFirstControl      ( DVector        &u0_ ) const = 0;


        /** Returns a variance-covariance estimate if possible or an error message otherwise.
         *
         *  \return SUCCESSFUL_RETURN
         *          RET_MEMBER_NOT_INITIALISED
         */
        virtual returnValue getVarianceCovariance( DMatrix &var );


		virtual returnValue setRealTimeParameters(	const DVector& DeltaX,
													const DVector& DeltaP = emptyConstVector
													);


		virtual returnValue freezeCondensing( );

		virtual returnValue unfreezeCondensing( );



	protected:

		virtual returnValue setupOptions( );
		virtual returnValue setupLogging( );

};


CLOSE_NAMESPACE_ACADO


//#include <acado/conic_solver/banded_cp_solver.ipp>


#endif  // ACADO_TOOLKIT_BANDED_CP_SOLVER_HPP

/*
 *  end of file
 */
