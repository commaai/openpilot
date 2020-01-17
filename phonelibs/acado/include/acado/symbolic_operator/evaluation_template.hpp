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
 *    \file include/acado/symbolic_operator/evaluation_template.hpp
 *    \author Boris Houska
 */


#ifndef ACADO_TOOLKIT_EVALUATION_TEMPLATE_HPP
#define ACADO_TOOLKIT_EVALUATION_TEMPLATE_HPP

#include <acado/symbolic_operator/symbolic_operator_fwd.hpp>
#include <acado/symbolic_operator/evaluation_base.hpp>


BEGIN_NAMESPACE_ACADO

/**
 *	\brief Templated class for operator evaluation.
 *
 *	\ingroup BasicDataStructures
 *
 *	\author Boris Houska
 */

template <typename T>
class EvaluationTemplate : public EvaluationBase{

public:

	/** Default constructor. */
	EvaluationTemplate();
	EvaluationTemplate( Tmatrix<T> *_val );
	
	virtual ~EvaluationTemplate();

	virtual void addition   ( Operator &arg1, Operator &arg2 );
	virtual void subtraction( Operator &arg1, Operator &arg2 );
	virtual void product    ( Operator &arg1, Operator &arg2 );
	virtual void quotient   ( Operator &arg1, Operator &arg2 );
	virtual void power      ( Operator &arg1, Operator &arg2 );
	virtual void powerInt   ( Operator &arg1, int      &arg2 );

	virtual void project    ( int      &idx );
	virtual void set        ( double   &arg );
	virtual void Acos       ( Operator &arg );
	virtual void Asin       ( Operator &arg );
	virtual void Atan       ( Operator &arg );
	virtual void Cos        ( Operator &arg );
	virtual void Exp        ( Operator &arg );
	virtual void Log        ( Operator &arg );
	virtual void Sin        ( Operator &arg );
	virtual void Tan        ( Operator &arg );
	
	Tmatrix<T> *val;
	T           res;
	
};



CLOSE_NAMESPACE_ACADO

#include <acado/symbolic_operator/operator.hpp>

BEGIN_NAMESPACE_ACADO



template <typename T> EvaluationTemplate<T>::EvaluationTemplate():EvaluationBase(){ val = 0; }
template <typename T> EvaluationTemplate<T>::EvaluationTemplate( Tmatrix<T> *_val ):EvaluationBase()
{ val = _val; }
template <typename T> EvaluationTemplate<T>::~EvaluationTemplate(){}

template <typename T> void EvaluationTemplate<T>::addition( Operator &arg1, Operator &arg2 ){
	
	EvaluationTemplate<T> r(val);
	arg1.evaluate( this );
	arg2.evaluate( &r );
	res += r.res;
}

template <typename T> void EvaluationTemplate<T>::subtraction( Operator &arg1, Operator &arg2 ){
 
	EvaluationTemplate<T> r(val);
	arg1.evaluate( this );
	arg2.evaluate( &r );
	res -= r.res;
}

template <typename T> void EvaluationTemplate<T>::product( Operator &arg1, Operator &arg2 ){
 
	EvaluationTemplate<T> r(val);
	arg1.evaluate( this );
	arg2.evaluate( &r );
	res *= r.res;
}

template <typename T> void EvaluationTemplate<T>::quotient( Operator &arg1, Operator &arg2 ){

	EvaluationTemplate<T> r(val);
	arg1.evaluate( this );
	arg2.evaluate( &r );
	res /= r.res;
}

template <typename T> void EvaluationTemplate<T>::power( Operator &arg1, Operator &arg2 ){
 
	EvaluationTemplate<T> r(val);
	arg1.evaluate( this );
	arg2.evaluate( &r );
	res = pow(res,r.res);
}

template <typename T> void EvaluationTemplate<T>::powerInt( Operator &arg1, int &arg2 ){
 
	arg1.evaluate( this );
	res = pow( res, arg2 );
}


template <typename T> void EvaluationTemplate<T>::project( int &idx ){

	res = val->operator()(idx);
}


template <typename T> void EvaluationTemplate<T>::set( double &arg ){

	res = arg;
}


template <typename T> void EvaluationTemplate<T>::Acos( Operator &arg ){

	arg.evaluate( this );
	res = acos( res );
}

template <typename T> void EvaluationTemplate<T>::Asin( Operator &arg ){

	arg.evaluate( this );
	res = asin( res );
}

template <typename T> void EvaluationTemplate<T>::Atan( Operator &arg ){

	arg.evaluate( this );
	res = atan( res );
}

template <typename T> void EvaluationTemplate<T>::Cos( Operator &arg ){

	arg.evaluate( this );
	res = cos( res );
}

template <typename T> void EvaluationTemplate<T>::Exp( Operator &arg ){

	arg.evaluate( this );
	res = exp( res );
}

template <typename T> void EvaluationTemplate<T>::Log( Operator &arg ){

	arg.evaluate( this );
	res = log( res );
}

template <typename T> void EvaluationTemplate<T>::Sin( Operator &arg ){

	arg.evaluate( this );
	res = sin( res );
}

template <typename T> void EvaluationTemplate<T>::Tan( Operator &arg ){

	arg.evaluate( this );
	res = tan( res );
}

CLOSE_NAMESPACE_ACADO

#endif
