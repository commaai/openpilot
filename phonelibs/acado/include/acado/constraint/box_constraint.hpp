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
 *    \file include/acado/constraint/box_constraint.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_BOX_CONSTRAINT_HPP
#define ACADO_TOOLKIT_BOX_CONSTRAINT_HPP


#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/variables_grid/variables_grid.hpp>
#include <acado/function/ocp_iterate.hpp>


BEGIN_NAMESPACE_ACADO



/** 
 *	\brief Stores and evaluates box constraints within optimal control problems.
 *
 *	\ingroup BasicDataStructures
 *
 *	The class BoxConstraint allows to manage and evaluate box (path) constraints
 *	(simple upper or lower bounds) on the optimization variables within 
 *	optimal control problems.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class BoxConstraint{

    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

    /** Default constructor. */
    BoxConstraint( );

    /** Copy constructor (deep copy). */
    BoxConstraint( const BoxConstraint& rhs );

    /** Destructor. */
    virtual ~BoxConstraint( );

    /** Assignment operator (deep copy). */
    BoxConstraint& operator=( const BoxConstraint& rhs );


    /** Initialization Routine. */
    returnValue init( const Grid& grid_ );


    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:


    /** Protected destructor. */
    void deleteAll();


    returnValue evaluateBounds( const OCPiterate& iter );


    /** Writes a special copy of the bounds that is needed within the
     *  OptimizationAlgorithm into the optimization variables.
     */
    virtual returnValue getBounds( const OCPiterate& iter );





    //
    // DATA MEMBERS:
    //
    protected:

    Grid             grid  ;   /**< the grid                    */

    // BOUNDS:
    // ----------------------
    int              nb    ;   /**< counts the number of bounds */
    VariableType    *var   ;   /**< variable types              */
    int             *index ;   /**< component of the variable   */
    DVector         **blb   ;   /**< lower bounds                */
    DVector         **bub   ;   /**< upper bounds                */

    DMatrix     *residuumXL ;   /**< residuum of the differential states to the lower bound */
    DMatrix     *residuumXU ;   /**< residuum of the differential states to the upper bound */
    DMatrix     *residuumXAL;   /**< residuum of the algebraic states    to the lower bound */
    DMatrix     *residuumXAU;   /**< residuum of the algebraic states    to the upper bound */
    DMatrix     *residuumPL ;   /**< residuum of the parameters          to the lower bound */
    DMatrix     *residuumPU ;   /**< residuum of the parameters          to the upper bound */
    DMatrix     *residuumUL ;   /**< residuum of the controls            to the lower bound */
    DMatrix     *residuumUU ;   /**< residuum of the controls            to the upper bound */
    DMatrix     *residuumWL ;   /**< residuum of the disturbances        to the lower bound */
    DMatrix     *residuumWU ;   /**< residuum of the disturbances        to the upper bound */
};


CLOSE_NAMESPACE_ACADO



#include <acado/constraint/box_constraint.ipp>


#endif  // ACADO_TOOLKIT_BOX_CONSTRAINT_HPP

/*
 *    end of file
 */
