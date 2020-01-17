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
 *    \file include/acado/nlp_derivative_approximation/gauss_newton_approximation.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_GAUSS_NEWTON_APPROXIMATION_HPP
#define ACADO_TOOLKIT_GAUSS_NEWTON_APPROXIMATION_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/nlp_derivative_approximation/nlp_derivative_approximation.hpp>


BEGIN_NAMESPACE_ACADO



/** 
 *	\brief Implements a Gauss-Newton approximation as second-order derivatives within NLPsolvers.
 *
 *	\ingroup NumericalAlgorithms
 *
 *  The class GaussNewtonApproximation implements a Gauss-Newton approximation as second-order 
 *	derivative information within iterative NLPsolvers.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class GaussNewtonApproximation : public NLPderivativeApproximation
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        GaussNewtonApproximation( );
		
		/** Default constructor. */
        GaussNewtonApproximation(	UserInteraction* _userInteraction
									);

        /** Copy constructor (deep copy). */
        GaussNewtonApproximation( const GaussNewtonApproximation& rhs );

        /** Destructor. */
        virtual ~GaussNewtonApproximation( );

        /** Assignment operator (deep copy). */
        GaussNewtonApproximation& operator=( const GaussNewtonApproximation& rhs );


		virtual NLPderivativeApproximation* clone( ) const;



        virtual returnValue initHessian(	BlockMatrix& B, 	    /**< matrix to be initialised */
											uint N,                 /**< number of intervals      */
											const OCPiterate& iter  /**< current iterate          */
											);

        virtual returnValue initScaling(	BlockMatrix& B, /**< matrix to be updated */
											const BlockMatrix& x, /**< direction x          */
											const BlockMatrix& y  /**< residuum             */
											);


        virtual returnValue apply(       BlockMatrix &B, /**< matrix to be updated */
                                   const BlockMatrix &x, /**< direction x          */
                                   const BlockMatrix &y  /**< residuum             */ );



    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:



    //
    // PROTECTED DATA MEMBERS:
    //
    protected:

};


CLOSE_NAMESPACE_ACADO


//#include <acado/nlp_derivative_approximation/gauss_newton_approximation.ipp>


// collect remaining headers
#include <acado/nlp_derivative_approximation/gauss_newton_approximation_bfgs.hpp>



#endif  // ACADO_TOOLKIT_GAUSS_NEWTON_APPROXIMATION_HPP

/*
 *  end of file
 */
