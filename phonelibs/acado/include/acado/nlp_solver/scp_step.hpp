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
 *    \file include/acado/nlp_solver/scp_step.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *
 */


#ifndef ACADO_TOOLKIT_SCP_STEP_HPP
#define ACADO_TOOLKIT_SCP_STEP_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/algorithmic_base.hpp>

#include <acado/function/ocp_iterate.hpp>
#include <acado/conic_program/banded_cp.hpp>
#include <acado/conic_solver/banded_cp_solver.hpp>

#include <acado/nlp_solver/scp_merit_function.hpp>



BEGIN_NAMESPACE_ACADO


/**
 *	\brief Base class for different ways to perform a step of an SCPmethod for solving NLPs.
 *
 *	\ingroup NumericalAlgorithms
 *
 *  The class SCPstep serves as a base class for different ways to perform a 
 *  (globalized) step of an SCPmethod for solving nonlinear programming problems.
 *
 *	 \author Boris Houska, Hans Joachim Ferreau
 */
class SCPstep : public AlgorithmicBase
{
// 	friend class SCPmethod;

    //
    // PUBLIC MEMBER FUNCTIONS:
    //
    public:

        /** Default constructor. */
        SCPstep( );

        SCPstep(	UserInteraction* _userInteraction
					);
		
        /** Copy constructor (deep copy). */
        SCPstep( const SCPstep& rhs );

        /** Destructor. */
        virtual ~SCPstep( );

        /** Assignment operator (deep copy). */
        SCPstep& operator=( const SCPstep& rhs );

        virtual SCPstep* clone() const = 0;


        virtual returnValue performStep(	OCPiterate& iter,
        									BandedCP& cp,
        									SCPevaluation* eval
        									) = 0;


    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:
		
		virtual returnValue setupOptions( );
		virtual returnValue setupLogging( );

		/** 
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
        virtual returnValue applyStep(	OCPiterate& iter,
        								BandedCP& cp,
        								double alpha
		        						) const;

// 		virtual returnValue getUpdatedFirstControl(	const OCPiterate& iter,
// 													const BandedCP& cp,
// 													double alpha,
// 													DVector& _u
// 													) const;


    //
    // DATA MEMBERS:
    //
    protected:

		SCPmeritFunction* meritFcn;

};


CLOSE_NAMESPACE_ACADO



//#include <acado/nlp_solver/scp_step.ipp>


#endif  // ACADO_TOOLKIT_SCP_STEP_HPP

/*
 *  end of file
 */
