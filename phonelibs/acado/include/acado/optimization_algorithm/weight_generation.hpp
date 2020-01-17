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
 *    \file include/acado/optimization_algorithm/weight_generation.hpp
 *    \author Boris Houska, Filip Logist, Hans Joachim Ferreau
 */


#ifndef ACADO_TOOLKIT_WEIGHT_GENERATION_HPP
#define ACADO_TOOLKIT_WEIGHT_GENERATION_HPP

#include <acado/matrix_vector/matrix_vector.hpp>

BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Generates weights for solving OCPs having multiple objectives.
 *
 *	\ingroup AuxiliaryFunctionality
 *
 *	Auxiliary class for generating weights for solving optimal control problems
 *  having multiple objectives.
 *
 *  \author Boris Houska, Filip Logist, Hans Joachim Ferreau
 */
class WeightGeneration{


    public:

        /** Default constructor. */
        WeightGeneration();

        /** Copy constructor (deep copy). */
        WeightGeneration( const WeightGeneration& arg );

        /** Destructor. */
        virtual ~WeightGeneration( );

        /** Assignment operator (deep copy). */
        WeightGeneration& operator=( const WeightGeneration& arg );



        /** Generates weights and closest points.                   \n
         *                                                          \n
         *  Input:                                                  \n
         *  \param  m           number of objectives                \n
         *  \param  pnts        number of points in 1 direction     \n
         *  \param  weightsLB   lower bound on weights              \n
         *  \param  weightsUB   upper bound on weights              \n
         *                                                          \n
         *  Output:                                                 \n
         *  \param  Weights     final weights                       \n
         *  \param  formers     closest points                      \n
         */
        returnValue getWeights( const int    &m        ,
                                const int    &pnts     ,
                                const DVector &weightsLB,
                                const DVector &weightsUB,
                                DMatrix       &Weights  ,
                                DVector       &formers    ) const;



    protected:


         /**  Recursive weight generation routine.                       \n
          *                                                              \n
          *   \param    n           dimension of the current layer       \n
          *   \param    weight      current set of weights to be added.  \n
          *   \param    Weights     tentative set of final weights.      \n
          *   \param    weightsLB   the lower bounds for the weights.    \n
          *   \param    weightsUB   the upper bounds for the weights.    \n
          *   \param    formers     tentative list of closest points.    \n
          *   \param    layer       the upper layer (equal to # of Obj.) \n
          *   \param    lastOne     counter for the closest points       \n
          *   \param    currentOne  counter for the closest points       \n
          *   \param    step        step size in one direction.          \n
          *
          */
        returnValue generateWeights( const  int    &n         ,
                                     DVector        &weight    ,
                                     DMatrix        &Weights   ,
                                     const  DVector &weightsLB ,
                                     const  DVector &weightsUB ,
                                     DVector        &formers   ,
                                     const int     &layer     ,
                                     int           &lastOne   ,
                                     int           &currentOne,
                                     double        &step
                                   ) const;
};





CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_MULTI_OBJECTIVE_ALGORITHM_HPP

/*
 *   end of file
 */
