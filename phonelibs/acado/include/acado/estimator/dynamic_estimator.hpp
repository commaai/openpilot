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
 *    \file include/acado/estimator/dynamic_estimator.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_DYNAMIC_ESTIMATOR_HPP
#define ACADO_TOOLKIT_DYNAMIC_ESTIMATOR_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/estimator/estimator.hpp>
#include <acado/optimization_algorithm/real_time_algorithm.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Implements an online state/parameter estimator based on dynamic optimization.
 *
 *	\ingroup UserInterfaces
 *
 *  The class DynamicEstimator implements an online state/parameter estimators
 *	based on dynamic optimization.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class DynamicEstimator : public Estimator
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:
        /** Default constructor. */
        DynamicEstimator( );

		/** Constructor taking minimal sub-block configuration. */
        DynamicEstimator(	const RealTimeAlgorithm& _realTimeAlgorithm,	/**< Dynamic optimizer. */
							double _samplingTime = DEFAULT_SAMPLING_TIME
							);

        /** Copy constructor (deep copy). */
        DynamicEstimator( const DynamicEstimator& rhs );

        /** Destructor. */
        virtual ~DynamicEstimator( );

        /** Assignment operator (deep copy). */
        DynamicEstimator& operator=( const DynamicEstimator& rhs );

		virtual Estimator* clone( ) const;


        /** Initialization. */
        virtual returnValue init(	double startTime = 0.0,
									const DVector &x0_ = emptyConstVector,
									const DVector &p_  = emptyConstVector
									);

        /** Executes next single step. */
        virtual returnValue step(	double currentTime,
									const DVector& _y
									);


   //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:



    //
    // DATA MEMBERS:
    //
    protected:
		RealTimeAlgorithm* realTimeAlgorithm;	/**< Optimization algorithm for online OCPs. */
};


CLOSE_NAMESPACE_ACADO



#include <acado/estimator/dynamic_estimator.ipp>


#endif  // ACADO_TOOLKIT_DYNAMIC_ESTIMATOR_HPP

/*
 *	end of file
 */
