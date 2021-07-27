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
 *    \file include/acado/estimator/estimator.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_ESTIMATOR_HPP
#define ACADO_TOOLKIT_ESTIMATOR_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/simulation_environment/simulation_block.hpp>

#include <acado/function/function.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Base class for interfacing online state/parameter estimators.
 *
 *	\ingroup UserInterfaces
 *
 *  The class Estimator serves as a base class for interfacing online 
 *	state/parameter estimators.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class Estimator : public SimulationBlock
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:
        /** Default constructor. */
        Estimator( );

		/** Constructor taking minimal sub-block configuration. */
        Estimator(	double _samplingTime
					);

        /** Copy constructor (deep copy). */
        Estimator( const Estimator& rhs );

        /** Destructor. */
        virtual ~Estimator( );

        /** Assignment operator (deep copy). */
        Estimator& operator=( const Estimator& rhs );

		virtual Estimator* clone( ) const = 0;


        /** Initialization. */
        virtual returnValue init(	double startTime = 0.0,
									const DVector &x0_ = emptyConstVector,
									const DVector &p_  = emptyConstVector
									);

        /** Executes next single step. */
        virtual returnValue step(	double currentTime,
									const DVector& _y
									) = 0;


		/** Returns all estimator outputs. */
        inline returnValue getOutputs(	DVector& _x,			/**< Estimated differential states. */
										DVector& _xa,		/**< Estimated algebraic states. */
										DVector& _u,			/**< Estimated previous controls. */
										DVector& _p,			/**< Estimated parameters. */
										DVector& _w			/**< Estimated disturbances. */
										) const;

		/** Returns estimated differential states. */
        inline returnValue getX(	DVector& _x	/**< OUTPUT: estimated differential states. */
									) const;

		/** Returns estimated algebraic states. */
        inline returnValue getXA(	DVector& _xa	/**< OUTPUT: estimated algebraic states. */
									) const;

		/** Returns estimated previous controls. */
        inline returnValue getU(	DVector& _u	/**< OUTPUT: estimated previous controls. */
									) const;

		/** Returns estimated parameters. */
        inline returnValue getP(	DVector& _p	/**< OUTPUT: estimated parameters. */
									) const;

		/** Returns estimated disturbances. */
        inline returnValue getW(	DVector& _w	/**< OUTPUT: estimated disturbances. */
									) const;


		/** Returns number of estimated differential states.
		 *  \return Number of estimated differential states */
		inline uint getNX( ) const;

		/** Returns number of estimated algebraic states.
		 *  \return Number of estimated algebraic states */
		inline uint getNXA( ) const;

		/** Returns number of estimated previous controls.
		 *  \return Number of estimated previous controls */
		inline uint getNU( ) const;

		/** Returns number of estimated parameters.
		 *  \return Number of estimated parameters */
		inline uint getNP( ) const;

		/** Returns number of estimated disturbances.
		 *  \return Number of estimated disturbances */
		inline uint getNW( ) const;

		/** Returns number of process outputs.
		 *  \return Number of process outputs */
		inline uint getNY( ) const;


    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:



    //
    // DATA MEMBERS:
    //
    protected:
		DVector x;			/**< Estimated differential state. */
		DVector xa;			/**< Estimated algebraic state. */
		DVector u;			/**< Estimated previous controls. */
		DVector p;			/**< Estimated parameters. */
		DVector w;			/**< Estimated disturbances. */
};


CLOSE_NAMESPACE_ACADO



#include <acado/estimator/estimator.ipp>


#endif  // ACADO_TOOLKIT_ESTIMATOR_HPP

/*
 *	end of file
 */
