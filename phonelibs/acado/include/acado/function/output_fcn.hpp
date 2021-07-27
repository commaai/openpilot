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
 *    \file include/acado/function/output_fcn.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_OUTPUT_FCN_HPP
#define ACADO_TOOLKIT_OUTPUT_FCN_HPP


#include <acado/function/function.hpp>


BEGIN_NAMESPACE_ACADO



/** 
 *	\brief Allows to setup and evaluate output functions based on SymbolicExpressions.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class OutputFcn allows to setup and evaluate output functions
 *	based on SymbolicExpressions.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class OutputFcn : public Function
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:
        /** Default constructor. */
        OutputFcn( );

        /** Copy constructor (deep copy). */
        OutputFcn( const OutputFcn& rhs );

        /** Destructor. */
        virtual ~OutputFcn( );

        /** Assignment operator (deep copy). */
        OutputFcn& operator=( const OutputFcn& rhs );


        Output operator()( uint componentIdx );


        /** Evaluates the function
         *  \return SUCCESSFUL_RETURN                  \n
         *          RET_NAN                            \n
         * */
        returnValue evaluate( double *x         /**< the input variable x */,
                              double *_result    /**< the result           */  );



        /** Evaluates the output function on a variables grid  \n
         *                                                     \n
         *  \return SUCCESSFUL_RETURN                          \n
         */
        returnValue evaluate( const VariablesGrid *x ,
                              const VariablesGrid *xa,
                              const VariablesGrid *p ,
                              const VariablesGrid *u ,
                              const VariablesGrid *w ,
                              VariablesGrid       *_result );



        /** Evaluates the function.                \n
         *                                         \n
         *  \param x       the evaluation point    \n
         *  \param number  the storage position    \n
         *                                         \n
         *  \return The result of the evaluation.  \n
         */
        DVector evaluate( const EvaluationPoint &x         ,
                         const int             &number = 0  );


		inline BooleanType isDefined( ) const;

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



#include <acado/function/output_fcn.ipp>


#endif  // ACADO_TOOLKIT_OUTPUT_FCN_HPP

/*
 *	end of file
 */
