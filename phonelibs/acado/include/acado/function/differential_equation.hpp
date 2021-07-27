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
 *    \file include/acado/function/differential_equation.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 */


#ifndef ACADO_TOOLKIT_DIFFERENTIAL_EQUATION_HPP
#define ACADO_TOOLKIT_DIFFERENTIAL_EQUATION_HPP


#include <acado/function/function_fwd.hpp>
#include <acado/symbolic_expression/lyapunov.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Allows to setup and evaluate differential equations (ODEs and DAEs) based on SymbolicExpressions.
 *
 *	\ingroup BasicDataStructures
 *
 *  The class DifferentialEquation allows to setup and evaluate 
 *	differential equations (ODEs and DAEs) based on SymbolicExpressions.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class DifferentialEquation : public Function{

	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:


		/** Default constructor. */
		DifferentialEquation( );

		/** Default constructor. */
		DifferentialEquation( const double &tStart, const double &tEnd );

		/** Default constructor. */
		DifferentialEquation( const double &tStart, const Parameter &tEnd );

		/** Default constructor. */
		DifferentialEquation( const Parameter &tStart, const double &tEnd  );

		/** Default constructor. */
		DifferentialEquation( const Parameter &tStart, const Parameter &tEnd  );


		/** Copy constructor (deep copy). */
		DifferentialEquation( const DifferentialEquation& arg );

		/** Destructor. */
		virtual ~DifferentialEquation( );

		/** Assignment operator (deep copy). */
		DifferentialEquation& operator=( const DifferentialEquation& arg );


		/** Clone constructor (deep copy). */
		virtual DifferentialEquation* clone() const;


		/** Loading Expressions (deep copy). */
		DifferentialEquation& operator<<( const Expression& arg );

		/** Loading Expressions (deep copy). */
		DifferentialEquation& operator<<( const int& arg );

		/** Loading Expressions (deep copy). */
		DifferentialEquation& operator==( const Expression& arg );

		/** Loading Expressions (deep copy). */
		DifferentialEquation& operator==( const double &arg );

		/** Loading Symbolic DVector (deep copy). */
		DifferentialEquation& operator==( const DVector& arg );

		/** Loading Symbolic DMatrix (deep copy). */
		DifferentialEquation& operator==( const DMatrix& arg );


		/** Ask whether the differential equation is a DAE. */
		BooleanType isDAE( ) const;

		/** Ask whether the differential equation is a ODE. */
		BooleanType isODE( ) const;

		/** Ask whether the differential equation is given in implicit form. */
		inline BooleanType isImplicit( ) const;


		/** Adds DifferentialStateDerivative if the differential  \n
		*  equation is an ODE.                                   \n
		*  \return  BT_TRUE   if the ODE is made implicit        \n
		*           BT_FALSE  if the differential equation is no \n
		*                     ODE.
		*/
		virtual BooleanType makeImplicit();



		/** Returns the number of dynamic equations   \n
		*/
		inline int getNumDynamicEquations() const;


		/** Returns the number of algebraic equations   \n
		*/
		inline int getNumAlgebraicEquations() const;


		/** Ask whether the differential equation is discretized (in time). */
		virtual BooleanType isDiscretized( ) const;


		/** Returns the start time of the horizon of the differential equation \n
		*  or -INFTY for the case that the start time is not defined.         \n
		*                                                                     \n
		*  \return the start time.                                            \n
		*/
		inline double getStartTime() const;


		/** Returns the end time of the horizon of the differential equation   \n
		*  or +INFTY for the case that the start time is not defined.         \n
		*                                                                     \n
		*  \return the end time.                                              \n
		*/
		inline double getEndTime() const;


		/** Returns the index of the parameter associated with the start time  \n
		*  of the time horizon of the differential equation or -1 for the     \n
		*  that the start time is constant                                    \n
		*                                                                     \n
		*  \return the start time.                                            \n
		*/
		inline int getStartTimeIdx() const;


		/** Returns the index of the parameter associated with the start time  \n
		*  of the time horizon of the differential equation or -1 for the     \n
		*  that the start time is constant                                    \n
		*                                                                     \n
		*  \return the start time.                                            \n
		*/
		inline int getEndTimeIdx() const;



		/** Always returns -INFTY in the time-continuous case.
		*/
		virtual double getStepLength() const;


		inline int getStateEnumerationIndex( int index_ );


		inline DVector getDifferentialStateComponents() const;



		/** Loading Expressions (deep copy). */
		DifferentialEquation& addDifferential( const Expression& arg );

		/** Loading Expressions (deep copy). */
		DifferentialEquation& addDifferential( const double &arg );

		/** Loading Expressions (deep copy). */
		DifferentialEquation& addAlgebraic( const Expression& arg );

		/** Loading Expressions (deep copy). */
		DifferentialEquation& addAlgebraic( const double &arg );

		/** Loading Symbolic DVector (deep copy). */
		DifferentialEquation& addDifferential( const DVector& arg );

		/** Loading Symbolic DVector (deep copy). */
		DifferentialEquation& addAlgebraic( const DVector& arg );

		/** Loading Symbolic DMatrix (deep copy). */
		DifferentialEquation& addDifferential( const DMatrix& arg );

		/** Loading Symbolic DMatrix (deep copy). */
		DifferentialEquation& addAlgebraic( const DMatrix& arg );

                DifferentialEquation& operator==(const Lyapunov& arg );

                Lyapunov getLyapunovObject( ) const;

                BooleanType hasLyapunovEquation( ) const;


		/** Returning an Expression that contains a Taylor expansion              \n
		  * of the ODE-solution trajectory.                                       \n
		  *                                                                       \n
		  * \param order the order of Taylor expansion. 
		  *
		  * \return A matrix valued expression C of dimension                     \n
		  *         n_x times (order+2) ,                                         \n
		  *         whose rows contain the Taylor coefficients.                   \n
		  *         C = [ x,f(x),f'(x)f(x),f''(x)f(x)f(x)+f'(x)f'(x)f(x), ... ].  \n
		  **/
		Expression getODEexpansion( const int &order ) const;
		
		
		inline int* getComponents() const;
		
		
		
	// PROTECTED MEMBER FUNCTIONS:
	// ---------------------------
	protected:


		/** Setup routine (for internal use only) */
		void setup();


        /** Protected version of the copy constructor. */
        void copy( const DifferentialEquation &arg );





	// PROTECTED MEMBERS:
	// ------------------

	protected:

		DifferentialEquationType           det;
		BooleanType                is_implicit;
		BooleanType             is_discretized;

		int  counter  ;
		int *component;

		Parameter  *T1, *T2;
		double      t1,  t2;

		double stepLength;  // for discretized dynamic equations only
  Lyapunov lyap;
};


CLOSE_NAMESPACE_ACADO



#include <acado/function/differential_equation.ipp>


#endif  // ACADO_TOOLKIT_DIFFERENTIAL_EQUATION_HPP

// end of file.
