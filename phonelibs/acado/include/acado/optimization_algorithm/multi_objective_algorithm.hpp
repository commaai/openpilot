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
 *    \file include/acado/optimization_algorithm/multi_objective_algorithm.hpp
 *    \author Boris Houska, Filip Logist, Hans Joachim Ferreau, Milan Vukov
 */


#ifndef ACADO_TOOLKIT_MULTI_OBJECTIVE_ALGORITHM_HPP
#define ACADO_TOOLKIT_MULTI_OBJECTIVE_ALGORITHM_HPP


#include <acado/optimization_algorithm/optimization_algorithm.hpp>
#include <acado/optimization_algorithm/weight_generation.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief User-interface to formulate and solve optimal control problems with multiple objectives.
 *
 *	\ingroup UserInterfaces
 *
 *	The class MultiObjectiveAlgorithm serves as a user-interface to formulate and
 *  solve optimal control problems with multiple objectives.
 *
 *  \author Boris Houska, Hans Joachim Ferreau
 */
class MultiObjectiveAlgorithm : public OptimizationAlgorithm
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        MultiObjectiveAlgorithm();

        /** Default constructor. */
        MultiObjectiveAlgorithm( const OCP& ocp_ );

        /** Copy constructor (deep copy). */
        MultiObjectiveAlgorithm( const MultiObjectiveAlgorithm& arg );

        /** Destructor. */
        virtual ~MultiObjectiveAlgorithm( );

        /** Assignment operator (deep copy). */
        MultiObjectiveAlgorithm& operator=( const MultiObjectiveAlgorithm& arg );



        /** Starts execution. */
        virtual returnValue solve( );


        /** Starts the optimization of one objective specified by the  \n
         *  the number.                                                \n
         *                                                             \n
         *  \param number  The number of the objective to be optimized \n
         *                                                             \n
         *  \return SUCCESSFUL_RETURN                                  \n
         *          or an error message from the optimization          \n
         *          algorithms                                         \n
         */
        virtual returnValue solveSingleObjective( const int &number );



        /** Defines the descretization of the Pareto front   \n
         *  (for the 2-dimensional case)                     \n
         *                                                   \n
         *  \param N the number of descretization intervals  \n
         *                                                   \n
         *  \return SUCCESSFUL_RETURN                        \n
         */
        inline returnValue setParetoFrontDiscretization( const int &N_ );



        /** Returns the Pareto front                         \n
         *                                                   \n
         *  \param paretoFront The pareto front in form of a \n
         *                     VariablesGrid.                \n
         *                                                   \n
         *  \return SUCCESSFUL_RETURN                        \n
         */
        inline returnValue getParetoFront( VariablesGrid &paretoFront ) const;



        /** Returns the filtered Pareto front                \n
         *                                                   \n
         *  \param paretoFront The pareto front in form of a \n
         *                     VariablesGrid.                \n
         *                                                   \n
         *  \return SUCCESSFUL_RETURN                        \n
         */
        inline returnValue getParetoFrontWithFilter( VariablesGrid &paretoFront ) const;



        /** Returns the pay-off matrix.                      \n
         *                                                   \n
         *  \return the pay-off matrix.                      \n
         */
        inline DMatrix getPayOffMatrix( ) const;


        /** Returns the normalized pay-off matrix.           \n
         *                                                   \n
         *  \return the pay-off matrix.                      \n
         */
        inline DMatrix getNormalizedPayOffMatrix( ) const;



        /** Returns the utopia vector.                       \n
         *                                                   \n
         *  \return the utopia vector.                       \n
         */
        inline DVector getUtopiaVector( ) const;



        /** Returns the nadir vector.                        \n
         *                                                   \n
         *  \return the nadir vector.                        \n
         */
        inline DVector getNadirVector( ) const;



        /** Returns the normalisation vector                     \n
         *  = difference between nadir vector and utopia vector. \n
         *                                                       \n
         *  \return the normalization vector.                    \n
         */
        inline DVector getNormalizationVector( ) const;



        /** Returns the utopia plane vectors                     \n
         *                                                       \n
         *  \return the utopia plane vector (stored column-wise).\n
         */
        inline DMatrix getUtopiaPlaneVectors( ) const;



        /** Prints general Info (statistics)   \n
         *                                     \n
         *  \return SUCCESSFUL_RETURN          \n
         */
        inline returnValue printInfo();


        /** Returns the weights           \n
         *                                \n
         *  \return SUCCESSFUL_RETURN     \n
         */
        inline DMatrix getWeights() const;


        /** Prints the weights into a file (with pre-ordering) \n
         *                                                     \n
         *  \return SUCCESSFUL_RETURN                          \n
         */
        inline returnValue getWeights( const char*fileName ) const;


        /** Prints the weights into a file (with pre-ordering) \n
         *  and after applying the Pareto filter.              \n
         *                                                     \n
         *  \return SUCCESSFUL_RETURN                          \n
         */
        inline returnValue getWeightsWithFilter( const char*fileName ) const;


        inline returnValue getAllDifferentialStates( const char*fileName ) const;
        inline returnValue getAllAlgebraicStates   ( const char*fileName ) const;
        inline returnValue getAllParameters        ( const char*fileName ) const;
        inline returnValue getAllControls          ( const char*fileName ) const;
        inline returnValue getAllDisturbances      ( const char*fileName ) const;



    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

		virtual returnValue setupOptions( );

        virtual returnValue initializeNlpSolver(	const OCPiterate& _userInit
													);

        virtual returnValue initializeObjective(	Objective* F
													);


        /** Reformulates the OCP problem for the multi-objective case.  \n
         *                                                              \n
         *  \return SUCCESSFUL_RETURN                                   \n
         */
        returnValue formulateOCP( double      *idx ,
                                  OCP         *ocp_,
                                  Expression **arg   );


        /**  Evaluates the objectives.                                  \n
         *                                                              \n
         *  \return SUCCESSFUL_RETURN                                   \n
         */
        returnValue evaluateObjectives( VariablesGrid    &xd_ ,
                                        VariablesGrid    &xa_ ,
                                        VariablesGrid    &p_  ,
                                        VariablesGrid    &u_  ,
                                        VariablesGrid    &w_  ,
                                        Expression      **arg1  );




        inline returnValue printAuxiliaryRoutine( const char*fileName, VariablesGrid *x_ ) const;


    //
    // DATA MEMBERS:
    //
    protected:

        int            N           ;   // number of discretization intervals
        int            m           ;   // number of objectives
        DMatrix         vertices    ;   // result for the objective values at the
                                       // vertices of the simplex.

        DMatrix         result      ;   // the result stored in a matrix
        int            count       ;   // counter for the results being stored


        VariablesGrid *xResults    ;
        VariablesGrid *xaResults   ;
        VariablesGrid *pResults    ;
        VariablesGrid *uResults    ;
        VariablesGrid *wResults    ;


     private:

        int     totalNumberOfSQPiterations;
        double  totalCPUtime              ;
};




CLOSE_NAMESPACE_ACADO


#include <acado/optimization_algorithm/multi_objective_algorithm.ipp>


#endif  // ACADO_TOOLKIT_MULTI_OBJECTIVE_ALGORITHM_HPP

/*
 *   end of file
 */
