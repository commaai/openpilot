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
 *    \file include/acado/nlp_derivative_approximation/constant_hessian.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_CONSTANT_HESSIAN_HPP
#define ACADO_TOOLKIT_CONSTANT_HESSIAN_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/nlp_derivative_approximation/nlp_derivative_approximation.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Implements a constant Hessian as approximation of second-order derivatives within NLPsolvers.
 *
 *	\ingroup NumericalAlgorithms
 *
 *  The class ConstantHessian implements a constant Hessian as approximation of 
 *	second-order derivatives within iterative NLPsolvers.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class ConstantHessian : public NLPderivativeApproximation
{

    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        ConstantHessian( );
		
		/** Default constructor. */
        ConstantHessian(	UserInteraction* _userInteraction
							);

        /** Copy constructor (deep copy). */
        ConstantHessian( const ConstantHessian& rhs );

        /** Destructor. */
        virtual ~ConstantHessian( );

        /** Assignment operator (deep copy). */
        ConstantHessian& operator=( const ConstantHessian& rhs );

		virtual NLPderivativeApproximation* clone( ) const;



        virtual returnValue initHessian(	BlockMatrix& B, 	    /**< matrix to be initialised */
											uint N,                 /**< number of intervals      */
											const OCPiterate& iter  /**< current iterate          */
											);

        virtual returnValue initScaling(	BlockMatrix& B, /**< matrix to be updated */
											const BlockMatrix& x, /**< direction x          */
											const BlockMatrix& y  /**< residuum             */
											);


		virtual returnValue apply(	BlockMatrix &B, /**< matrix to be updated */
									const BlockMatrix &x, /**< direction x          */
									const BlockMatrix &y  /**< residuum             */
									);



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

//#include <acado/nlp_derivative_approximation/constant_hessian.ipp>


// collect remaining headers
#include <acado/nlp_derivative_approximation/bfgs_update.hpp>

#endif  // ACADO_TOOLKIT_CONSTANT_HESSIAN_HPP

/*
 *  end of file
 */
