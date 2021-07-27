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
 *    \file include/acado/validated_integrator/ellipsoidal_integrator.hpp
 *    \author Boris Houska, Mario Villanueva, Benoit Chachuat
 */


#ifndef ACADO_TOOLKIT_ELLIPSOIDAL_INTEGRATOR_HPP
#define ACADO_TOOLKIT_ELLIPSOIDAL_INTEGRATOR_HPP

#include <iostream>
#include <iomanip>
#include <acado/clock/clock.hpp>
#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/algorithmic_base.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/variables_grid/variables_grid.hpp>
#include <acado/symbolic_expression/symbolic_expression.hpp>
#include <acado/function/function.hpp>
#include <acado/set_arithmetics/set_arithmetics.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief Validated integrator for ODEs based on Taylor models with ellipsoidal remainder term.
 *
 *	\author Mario Villanueva, Boris Houska
 */



class EllipsoidalIntegrator : public AlgorithmicBase
{
 
  public: 
  
    /** Default constructor. */
    EllipsoidalIntegrator( );

    /** Constructor that takes the differential equation and the order (default 3) as an argument.
	 *
	 *  \param rhs_  The differential equation.
	 *  \param N_    The order of the intergrator (default = 3).
	 */
    EllipsoidalIntegrator( const DifferentialEquation &rhs_, const int &N_ = 3 );

    /** Copy constructor (deep copy). */
    EllipsoidalIntegrator( const EllipsoidalIntegrator& arg );

    /** Destructor. */
    virtual ~EllipsoidalIntegrator( );

    /** Assignment operator (deep copy). */
    virtual EllipsoidalIntegrator& operator=( const EllipsoidalIntegrator& arg );
  
	
	
	
	Tmatrix<Interval> integrate( double t0, double tf, int M, const Tmatrix<Interval> &x );
	
	Tmatrix<Interval> integrate( double t0, double tf, int M, const Tmatrix<Interval> &x, const Tmatrix<Interval> &p );
	
	Tmatrix<Interval> integrate( double t0, double tf, int M, const Tmatrix<Interval> &x,
								 const Tmatrix<Interval> &p, const Tmatrix<Interval> &w );
	

	template <typename T> returnValue integrate( double t0, double tf,
												 Tmatrix<T> *x, Tmatrix<T> *p = 0, Tmatrix<T> *w = 0 );

    template <typename T> double step( const double &t, const double &tf,
									   Tmatrix<T> *x, Tmatrix<T> *p = 0, Tmatrix<T> *w = 0 );

	returnValue init( const DifferentialEquation &rhs_, const int &N_ = 3 );

	Tmatrix<Interval> boundQ() const;
	
	template <typename T> Tmatrix<Interval> getStateBound( const Tmatrix<T> &x ) const;
	
	
	
	
// PRIVATE FUNCTIONS:
// ----------------------------------------------------------
	
private:
	
	
	virtual returnValue setupOptions( );
	
	
	void copy( const EllipsoidalIntegrator& arg );
	
	template <typename T> void phase0( double t,
									   Tmatrix<T> *x, Tmatrix<T> *p, Tmatrix<T> *w,
									   Tmatrix<T> &coeff, Tmatrix<double> &C );
	
	template <typename T> double phase1(	double t, double tf,
											Tmatrix<T> *x, Tmatrix<T> *p, Tmatrix<T> *w,
											Tmatrix<T> &coeff,
											Tmatrix<double> &C );
	
	template <typename T> void phase2(	double t, double h,
										Tmatrix<T> *x, Tmatrix<T> *p, Tmatrix<T> *w,
										Tmatrix<T> &coeff,
										Tmatrix<double> &C );

	template <typename T> Tmatrix<T> evaluate(	Function &f, double t,
												Tmatrix<T> *x, Tmatrix<T> *p, Tmatrix<T> *w ) const;


	template <typename T> Tmatrix<T> evaluate(	Function &f, Interval t,
												Tmatrix<T> *x, Tmatrix<T> *p, Tmatrix<T> *w ) const;

	
	template <typename T> Tmatrix<T> phi( const Tmatrix<T> &coeff, const double &h ) const;
	
	
	
	template <typename T> Tmatrix<double> hat( const Tmatrix<T> &x ) const;
	
	Tmatrix<Interval> evalC ( const Tmatrix<double> &C, double h ) const;
	Tmatrix<double>   evalC2( const Tmatrix<double> &C, double h ) const;
	
	double scale( const         Interval  &E, const   Interval  &X ) const;
	double norm ( const Tmatrix<Interval> &E, Tmatrix<Interval> &X ) const;
	
	
	BooleanType isIncluded( const Tmatrix<Interval> &A, const Tmatrix<Interval> &B ) const;

	template <typename T> Tmatrix<Interval> bound( const Tmatrix<T> &x ) const;
	
	template <typename T> Tmatrix<Interval> getRemainder( const Tmatrix<T> &x ) const;
	
	template <typename T> Tmatrix<T> getPolynomial( const Tmatrix<T> &x ) const;
	
	template <typename T> void center( Tmatrix<T> &x ) const;
	
	void updateQ( Tmatrix<double> C, Tmatrix<Interval> R );
	
	void setInfinity();
	
	
	
// PRIVATE MEMBERS:
// ----------------------------------------------------------
	
private:
  
	int       nx;   // number of differential states.
	int       N ;   // the order of the integrator.
	
	Function g  ;   // Taylor expansion of the solution trajectory
	Function gr ;   // Remainder term associated with g 
	Function dg ;   // Jacobian of the function g : g(t,x,...)/dx
	Function ddg;   // Directional derivative of dg: (d^2 g(t,x,...)/d^2x)*r*r

    Tmatrix<double>  Q;  // Ellipsoidal remainder matrix
    
    RealClock totalTime ;
    RealClock Phase0Time;
    RealClock Phase1Time;
};

CLOSE_NAMESPACE_ACADO


#include <acado/validated_integrator/ellipsoidal_integrator.ipp>

#endif  // ACADO_TOOLKIT_ELLIPSOIDAL_INTEGRATOR_HPP

// end of file.
