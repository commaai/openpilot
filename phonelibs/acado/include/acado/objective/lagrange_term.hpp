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
 *    \file include/acado/objective/lagrange_term.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_LAGRANGE_TERM_HPP
#define ACADO_TOOLKIT_LAGRANGE_TERM_HPP


#include <acado/variables_grid/variables_grid.hpp>
#include <acado/function/function.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Stores and evaluates Lagrange terms within optimal control problems.
 *
 *	\ingroup BasicDataStructures
 *
 *	The class LagrangeTerm is object that is introduced as a kind of   \n
 *  temporary storage containter of the objective to store lagrange    \n
 *  terms that are defined by the user. As the objective does later    \n
 *  reformulate the Lagrange term into an Mayer term, this class has   \n
 *  no algorithmic functionality - it is just a data class.            \n
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */


class LagrangeTerm{


    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        LagrangeTerm();

        /** Copy constructor (deep copy). */
        LagrangeTerm( const LagrangeTerm& rhs );

        /** Destructor. */
        virtual ~LagrangeTerm( );

        /** Assignment operator (deep copy). */
        LagrangeTerm& operator=( const LagrangeTerm& rhs );


        /**  Sets the discretization grid.   \n
         *                                   \n
         *   \return SUCCESSFUL_RETURN       \n
         */
        returnValue init( const Grid &grid_ );


        /** Adds an expression for the Lagrange term.
         *  \return SUCCESSFUL_RETURN
         */
        inline returnValue addLagrangeTerm( const Expression& arg );


        /** Adds an expression for the Lagrange term.
         *  \return SUCCESSFUL_RETURN
         */
        inline returnValue addLagrangeTerm( const Expression& arg,
                                            const int& stageNumber );



        /** returns the objective grid */
        inline const Grid& getGrid() const;


    //
    // DATA MEMBERS:
    //
    protected:

        Grid         grid          ;  /**< the objective grid.      */
        int          nLagrangeTerms;  /**< number of lagrange terms */
        Expression **lagrangeFcn   ;  /**< the Lagrange function.   */
};


CLOSE_NAMESPACE_ACADO



#include <acado/objective/lagrange_term.ipp>


#endif  // ACADO_TOOLKIT_LAGRANGE_TERM_HPP

/*
 *    end of file
 */
