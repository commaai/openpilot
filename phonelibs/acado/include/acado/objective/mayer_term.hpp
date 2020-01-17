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
 *    \file include/acado/objective/mayer_term.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_MAYER_TERM_HPP
#define ACADO_TOOLKIT_MAYER_TERM_HPP


#include <acado/objective/objective_element.hpp>



BEGIN_NAMESPACE_ACADO



/** 
 *	\brief Stores and evaluates Mayer terms within optimal control problems.
 *
 *	\ingroup BasicDataStructures
 *
 *	The class MayerTerm allows to manage and evaluate Mayer terms
 *  within optimal control problems.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class MayerTerm : public ObjectiveElement{


    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        MayerTerm( );

        /** Default constructor. */
        MayerTerm( const Grid &grid_,
                   const Expression& arg );

        /** Default constructor. */
        MayerTerm( const Grid &grid_,
                   const Function& arg );

        /** Copy constructor (deep copy). */
        MayerTerm( const MayerTerm& rhs );

        /** Destructor. */
        virtual ~MayerTerm( );

        /** Assignment operator (deep copy). */
        MayerTerm& operator=( const MayerTerm& rhs );



// =======================================================================================
//
//                                 INITIALIZATION ROUTINES
//
// =======================================================================================


        inline returnValue init( const Grid &grid_, const Expression& arg );



// =======================================================================================
//
//                                   EVALUATION ROUTINES
//
// =======================================================================================


        returnValue evaluate( const OCPiterate &x );


        /** Evaluates the objective gradient contribution from this term \n
         *  and computes the corresponding exact hessian if hessian != 0 \n
         *                                                               \n
         *  \return SUCCESSFUL_RETURN                                    \n
         */
        returnValue evaluateSensitivities( BlockMatrix *hessian );



// =======================================================================================


    //
    // DATA MEMBERS:
    //
    protected:

};


CLOSE_NAMESPACE_ACADO


#include <acado/objective/mayer_term.ipp>

#endif  // ACADO_TOOLKIT_MAYER_TERM_HPP

/*
 *    end of file
 */
