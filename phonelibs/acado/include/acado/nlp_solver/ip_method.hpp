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
 *    \file include/acado/nlp_solver/ip_method.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_IP_METHOD_HPP
#define ACADO_TOOLKIT_IP_METHOD_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/nlp_solver/nlp_solver.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Implements different interior-point methods for solving NLPs.
 *
 *	\ingroup NumericalAlgorithms
 *
 *  The class IPmethod implements different interior-point methods 
 *  for solving nonlinear programming problems.
 *
 *	 \author Boris Houska, Hans Joachim Ferreau
 */
class IPmethod : public NLPsolver
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        IPmethod( );

        IPmethod(	UserInteraction* _userInteraction
					);
					
        /** Default constructor. */
        IPmethod( const NLP& nlp_ );

        /** Copy constructor (deep copy). */
        IPmethod( const IPmethod& rhs );

        /** Destructor. */
        ~IPmethod( );

        /** Assignment operator (deep copy). */
        IPmethod& operator=( const IPmethod& rhs );


        /** Initialization. */
        virtual returnValue init(VariablesGrid    *xd,
                                 VariablesGrid    *xa,
                                 VariablesGrid    *p,
                                 VariablesGrid    *u,
                                 VariablesGrid    *w   );


        /** Starts execution. */
        virtual returnValue solve( int maxNumSteps );


    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:




    //
    // DATA MEMBERS:
    //
    protected:
};


CLOSE_NAMESPACE_ACADO



#include <acado/nlp_solver/ip_method.ipp>


#endif  // ACADO_TOOLKIT_IP_METHOD_HPP

/*
 *	end of file
 */
