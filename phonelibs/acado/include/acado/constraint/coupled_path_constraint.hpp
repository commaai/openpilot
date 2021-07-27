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
 *    \file include/acado/constraint/coupled_path_constraint.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_COUPLED_PATH_CONSTRAINT_HPP
#define ACADO_TOOLKIT_COUPLED_PATH_CONSTRAINT_HPP


#include <acado/constraint/constraint_element.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Stores and evaluates coupled path constraints within optimal control problems.
 *
 *	\ingroup BasicDataStructures
 *
 *	The class CoupledPathConstraint allows to manage and evaluate constraints
 *	along the whole horizon within optimal control problems that are coupled 
 *  across control intervals. If the contraints are decoupled, the class 
 *	PathConstraint should be used.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class CoupledPathConstraint : public ConstraintElement{

    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        CoupledPathConstraint( );

        /** Default constructor. */
        CoupledPathConstraint( const Grid& grid_ );

        /** Copy constructor (deep copy). */
        CoupledPathConstraint( const CoupledPathConstraint& rhs );

        /** Destructor. */
        virtual ~CoupledPathConstraint( );

        /** Assignment operator (deep copy). */
        CoupledPathConstraint& operator=( const CoupledPathConstraint& rhs );


        /**  Adds a coupled-path-constraint component.
         *   \return SUCCESSFUL_RETURN
         *           RET_MEMBER_NOT_INITIALISED  (if the ConstraintElement::fcn is not initialized)
         */
        inline returnValue add( const double lb_, const Expression* arg, const double ub_ );



// =======================================================================================
//
//                                   EVALUATION ROUTINES
//
// =======================================================================================


        /** Evaluates all components in this constraint and stores the \n
          * residuum.                                                  \n
          *                                                            \n
          * \return SUCESSFUL_RETURN                                   \n
          */
        returnValue evaluate( const OCPiterate& iter );


        /** Evaluates the sensitivities of all components in this      \n
          * constraint. Note that the seed can be defined via the base \n
          * class ConstraintElement.                                   \n
          *                                                            \n
          * \return SUCESSFUL_RETURN                                   \n
          */
        returnValue evaluateSensitivities();



        /** Evaluates the sensitivities and Hessian.                   \n
          *                                                            \n
          * \return SUCESSFUL_RETURN                                   \n
          */
        returnValue evaluateSensitivities( const DMatrix &seed, BlockMatrix &hessian );



//  =========================================================================

        /** returns the number of constraints */
        inline int getNC() const;


	protected:

};


CLOSE_NAMESPACE_ACADO



#include <acado/constraint/coupled_path_constraint.ipp>


#endif  // ACADO_TOOLKIT_COUPLED_PATH_CONSTRAINT_HPP
/*
 *    end of file
 */
