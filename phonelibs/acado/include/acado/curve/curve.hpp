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
 *    \file include/acado/curve/curve.hpp
 *    \author Boris Houska, Hans Joachim Ferreau 
 */


#ifndef ACADO_TOOLKIT_CURVE_HPP
#define ACADO_TOOLKIT_CURVE_HPP

#include <acado/variables_grid/variables_grid.hpp>
#include <acado/function/function.hpp>

BEGIN_NAMESPACE_ACADO


/** 
  *	\brief Allows to work with piecewise-continous function defined over a scalar time interval.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class Curve allows to setup and evaluate piecewise-continous functions that 
 *  are defined over a scalar time interval and map into an Vectorspace of given dimension.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class Curve{

    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        Curve();

        /** Copy constructor (deep copy). */
        Curve( const Curve& arg );

        /** Destructor. */
        ~Curve( );

        /** Assignment operator (deep copy). */
        Curve& operator=( const Curve& arg );

		
		Curve operator()(	uint idx
							) const;

        /** Adds a constant piece to the curve. If other pieces of the curve have previously \n
         *  been added, the start time  "tStart" of the time interval [tStart,tEnd] should   \n
         *  contain the last time-point of the curve piece which has been added before.      \n
         *  The curve is not required to be continous at the boundaries of its intervals.    \n
         *  Note that the dimension of the constant input vector should have the same        \n
         *  dimensions as previously added curve pieces. If no other piece has been added    \n
         *  before, only  tStart < tEnd is required, while the dimension of the curve will   \n
         *  be set to the dimension of the vector, which is added.                           \n
         *                                                                                   \n
         *  \param tStart    start of the time interval of the curve piece to be added.      \n
         *  \param tEnd      end of the time interval to be added.                           \n
         *  \param constant  the constant value of the curve on the interval [tStart,tEnd].  \n
         *                                                                                   \n
         *  \return SUCCESSFUL_RETURN                                                        \n
         *          RET_TIME_INTERVAL_NOT_VALID                                              \n
         *          RET_INPUT_DIMENSION_MISMATCH                                             \n
         */
        returnValue add( double tStart,
                         double tEnd,
                         const DVector constant );


        /** Adds new curve pieces, which are given in a sampled form (VariablesGrid). In     \n
         *  order to store the curve pieces as continous functions the sampled data will be  \n
         *  interpolated depending on an interpolation mode. Note that the number of         \n
         *  intervals, which are added, will depend on this interpolation mode. The default  \n
         *  interpolation mode is "IM_LINEAR", i.e. linear interpolation. In this case the   \n
         *  number of added intervals is coinciding with the number of intervals of the      \n
         *  VariablesGrid, which is passed. For the dimension of the variables grid as well  \n
         *  the grid's start- and endpoint the same policy as for the other "add" functions  \n
         *  applies, i.e. the dimension should be equal to previously added pieces and the   \n
         *  time intervals should fit together.                                              \n
         *                                                                                   \n
         *  \param sampledData the data in sampled form to be interpolated and added.        \n
         *  \param mode        the interploation mode (default: linear interpolation)        \n
         *                                                                                   \n
         *  \return SUCCESSFUL_RETURN                                                        \n
         *          RET_TIME_INTERVAL_NOT_VALID                                              \n
         *          RET_INPUT_DIMENSION_MISMATCH                                             \n
         */
        returnValue add( const VariablesGrid& sampledData, InterpolationMode mode = IM_LINEAR );


        /** Adds a new piece to the curve, which is defined on the time interval              \n
         *  [tStart,tEnd] to be added. The function, which is passed as an argument of this   \n
         *  routine, should be the parameterization of the piece of curve to be added. Note   \n
         *  that the input function "parameterization" is only allowed to depend on the time. \n
         *  If the parameterization depends e.g. on DifferentialStates, Controls etc. an      \n
         *  error message will be returned. As for the other "add" routines the dimension of  \n
         *  function as well as the time interval should fit to the previously added curve    \n
         *  pieces.                                                                           \n
         *                                                                                    \n
         *  \param tStart             start of the interval to be added.                      \n
         *  \param tEnd               end of the time interval to be added.                   \n
         *  \param parameterization_  the parameterization of the curve on this interval.     \n
         *                                                                                    \n
         *  \return SUCCESSFUL_RETURN            (if successful)                              \n
         *          RET_TIME_INTERVAL_NOT_VALID  (if the time interval is not valid)          \n
         *          RET_INPUT_DIMENSION_MISMATCH (if the dimension or dependencies are wrong) \n
         */
        returnValue add( double tStart,
                         double tEnd,
                         const Function& parameterization_ );


        /** Returns the dimension of the curve. Please note that this routine will return     \n
         *   -1  for the case that the curve is empty (cf. the routine isEmpty() ).           \n
         *                                                                                    \n
         *  \return the dimension of the curve or -1.                                         \n
         */
        inline int getDim( ) const;


        /** Returns whether the curve is empty.      \n
         *                                           \n
         *  \return BT_TRUE   if the curve is empty. \n
         *          BT_FALSE  otherwise.             \n
         */
        inline BooleanType isEmpty( ) const;



        /** Returns whether the curve is continous.      \n
         *                                               \n
         *  \return BT_TRUE   if the curve is continous. \n
         *          BT_FALSE  otherwise.                 \n
         */
        inline BooleanType isContinuous( ) const;



        /** Returns the number of intervals, on which the pieces of the curve  \n
         *  are defined.                                                       \n
         *                                                                     \n
         *  \return the number if intervals.                                   \n
         */
        inline int getNumIntervals( ) const;



        /** Evaluates the curve at a given time point. This routine will store          \n
         *  the result of the evaluation into the double *result. Note that             \n
         *  this double pointer must be allocated by the user. Otherwise,               \n
         *  segmentation faults might occur, which can not be prevented by              \n 
         *  this routine. For not time critical operations it is recommended to         \n
         *  use the routine                                                             \n
         *                                                                              \n
         *  evaluate( const double t, const DVector &result )                           \n
         *                                                                              \n
         *  instead, which will throw  an error if a dimension mismatch occurs.         \n 
         *  However, the routine based on double* is slightly more efficient.           \n
         *  In order to make the allocation correct, use the routines isEmpty()         \n
         *  and getDim() first to obtain (or check) the dimension.                      \n
         *                                                                              \n
         *  \param  t      (input) the time at which the curve should be evaluated.     \n
         *  \param  result (output) the result of the evaluation.                       \n
         *                                                                              \n
         *                                                                              \n
         *  \return SUCCESSFUL_RETURN           (if the evaluation was successful.)     \n
         *          RET_INVALID_ARGUMENTS       (if the double* result is NULL.)        \n
         *          RET_INVALID_TIME_POINT      (if the time point t is out of range.)  \n
         *          RET_MEMBER_NOT_INITIALISED  (if the curve is empty)                 \n
         */
        returnValue evaluate( double t, double *result ) const;


         /** Evaluates the curve at a given time point. This routine will store            \n
          *  the result of the evaluation into the DVector &result.                         \n
          *                                                                                \n
          *  \param  t      (input) the time at which the curve should be evaluated.       \n
          *  \param  result (output) the result of the evaluation.                         \n
          *                                                                                \n
          *                                                                                \n
          *  \return SUCCESSFUL_RETURN              (if the evaluation was successful.)    \n
          *          RET_INVALID_TIME_POINT         (if the time point t is out of domain.)\n
          *          RET_MEMBER_NOT_INITIALISED     (if the curve is empty)                \n
          */
         returnValue evaluate( double t, DVector &result ) const;


	  /** Evaluates the curve at a given time interval. This routine will store        \n
          *  the result of the evaluation into the VariablesGrid &result.                  \n
	  *  Returns as entries in result exactly those nodes of the curve for which the   \n
	  * node times are contained in the interval [tStart,tEnd]                         \n
	  *                                                                                \n
	  * No interpolation is used.                                                       \n
          *                                                                                \n
          *  \param  tStart (input) the start time at which the curve should be evaluated. \n
          *  \param  tEnd   (input) the end time at which the curve should be evaluated.   \n
          *  \param  result (output) the result of the evaluation.                         \n
          *                                                                                \n
          *                                                                                \n
          *  \return SUCCESSFUL_RETURN              (if the evaluation was successful.)    \n
          *          RET_INVALID_TIME_POINT         (if the time point t is out of domain.)\n
          *          RET_MEMBER_NOT_INITIALISED     (if the curve is empty)                \n
          */

         returnValue evaluate( double tStart, double tEnd, VariablesGrid &result ) const;



         /** Evaluates the curve at specified grid points and stores the result in form of a             \n
          *  VariablesGrid. Note that all time points of the grid, at which the curve should be          \n
          *  evaluated, must be contained in the domain of the curve. This domain can be                 \n 
          *  obtained with the routine  "getTimeDomain( double tStart, double tEnd )".                 \n
          *                                                                                              \n
          *  \param  discretizationGrid  (input) the grid points at which the curve should be evaluated. \n
          *  \param  result              (output) the result of the evaluation.                          \n
          *                                                                                              \n
          *                                                                                              \n
          *  \return SUCCESSFUL_RETURN           (if the evaluation was successful.)                     \n
          *          RET_INVALID_TIME_POINT      (if at least one of the grid points is out of domain.)  \n
          *          RET_MEMBER_NOT_INITIALISED  (if the curve is empty)                                 \n
          */
         returnValue discretize( const Grid &discretizationGrid, VariablesGrid &result ) const;



         /** Returns the time domain of the curve, i.e. the time points tStart and tEnd between    \n
          *  which the curve is defined.                                                           \n
          *                                                                                        \n
          *  \return SUCCESSFUL_RETURN                                                             \n
          *          RET_MEMBER_NOT_INITIALISED  (if the curve is empty.)                          \n
          */
         returnValue getTimeDomain( double tStart, double tEnd ) const;


         /** Returns the time domain of a piece of the curve, i.e. the interrval [tStart,tEnd] on     \n
          *  which the piece with number "idx" is defined.                                            \n
          *                                                                                           \n
          *  \param idx    (input)  the index of the curve piece for which the time domain is needed. \n
          *  \param tStart (output) the start time of the requested interval.                         \n
          *  \param tEnd   (output) the end   time of the requested interval.                         \n
          *                                                                                           \n
          *  \return SUCCESSFUL_RETURN           (if the domain is successfully returned.)            \n
          *          RET_INDEX_OUT_OF_BOUNDS     (if the index "idx" is larger than getNumIntervals())\n
          *          RET_MEMBER_NOT_INITIALISED  (if the curve is empty.)                             \n
          */
         returnValue getTimeDomain( const uint &idx, double tStart, double tEnd ) const;



         /** Returns the time domain of the curve, i.e. the time points tStart and tEnd between    \n
          *  which the curve is defined.                                                           \n
          *                                                                                        \n
          *  \return SUCCESSFUL_RETURN                                                             \n
          *          RET_MEMBER_NOT_INITIALISED  (if the curve is empty.)                          \n
          */
         BooleanType isInTimeDomain(	double t
										) const;





    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:


    //
    // DATA MEMBERS:
    //
    protected:

        uint                 nIntervals      ;   // number of intervals of the curve.
        uint                 dim             ;   // the dimension of the curve.
        Function           **parameterization;   // the parameterizations of the curve pieces
        Grid                *grid            ;   // the grid points associated with the intervals of the curve.
};


CLOSE_NAMESPACE_ACADO


#include <acado/curve/curve.ipp>


#endif  // ACADO_TOOLKIT_CURVE_HPP

/*
 *    end of file
 */
