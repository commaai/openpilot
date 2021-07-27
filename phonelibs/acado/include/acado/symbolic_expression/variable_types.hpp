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
*    \file include/acado/symbolic_expression/parameter.hpp
*    \author Boris Houska, Hans Joachim Ferreau, Milan Vukov
*    \date 2008 - 2013
*/

#ifndef ACADO_TOOLKIT_VARIABLE_TYPES_HPP
#define ACADO_TOOLKIT_VARIABLE_TYPES_HPP

#include <acado/symbolic_expression/expression.hpp>

BEGIN_NAMESPACE_ACADO

/** Algebraic variable. */
class AlgebraicState : public ExpressionType<AlgebraicState, VT_ALGEBRAIC_STATE>
{
	typedef ExpressionType<AlgebraicState, VT_ALGEBRAIC_STATE> Base;

public:
	AlgebraicState() : Base() {}

	AlgebraicState(const std::string& _name, unsigned _nRows, unsigned _nCols)
		: Base(_name, _nRows, _nCols)
	{}
};

/** Control variable. */
class Control : public ExpressionType<Control, VT_CONTROL>
{
	typedef ExpressionType<Control, VT_CONTROL> Base;

public:
	Control() : Base() {}

	Control(const std::string& _name, unsigned _nRows, unsigned _nCols)
		: Base(_name, _nRows, _nCols)
	{}
};

/** Differential state derivative variable. */
class DifferentialStateDerivative : public ExpressionType<DifferentialStateDerivative, VT_DDIFFERENTIAL_STATE>
{
	typedef ExpressionType<DifferentialStateDerivative, VT_DDIFFERENTIAL_STATE> Base;

public:
	DifferentialStateDerivative() : Base() {}

	DifferentialStateDerivative(const std::string& _name, unsigned _nRows, unsigned _nCols)
		: Base(_name, _nRows, _nCols)
	{}
};

/** Differential state variable. */
class DifferentialState: public ExpressionType<DifferentialState, VT_DIFFERENTIAL_STATE>
{
	typedef ExpressionType<DifferentialState, VT_DIFFERENTIAL_STATE> Base;

public:
	DifferentialState() : Base() {}

	DifferentialState(const std::string& _name, unsigned _nRows, unsigned _nCols)
		: Base(_name, _nRows, _nCols)
	{}
};

/** Disturbance variable. */
class Disturbance : public ExpressionType<Disturbance, VT_DISTURBANCE>
{
	typedef ExpressionType<Disturbance, VT_DISTURBANCE> Base;

public:
	Disturbance() : Base() {}

	Disturbance(const std::string& _name, unsigned _nRows, unsigned _nCols)
		: Base(_name, _nRows, _nCols)
	{}
};

/** Integer control variable. */
class IntegerControl : public ExpressionType<IntegerControl, VT_INTEGER_CONTROL>
{
	typedef ExpressionType<IntegerControl, VT_INTEGER_CONTROL> Base;

public:
	IntegerControl() : Base() {}

	IntegerControl(const std::string& _name, unsigned _nRows, unsigned _nCols)
		: Base(_name, _nRows, _nCols)
	{}
};

/** Integer parameter variable. */
class IntegerParameter : public ExpressionType<IntegerParameter, VT_INTEGER_PARAMETER>
{
	typedef ExpressionType<IntegerParameter, VT_INTEGER_PARAMETER> Base;

public:
	IntegerParameter() : Base() {}

	IntegerParameter(const std::string& _name, unsigned _nRows, unsigned _nCols)
		: Base(_name, _nRows, _nCols)
	{}
};

/** Online data variable. */
class OnlineData : public ExpressionType<OnlineData, VT_ONLINE_DATA>
{
	typedef ExpressionType<OnlineData, VT_ONLINE_DATA> Base;

public:
	OnlineData() : Base() {}

	OnlineData(const std::string& _name, unsigned _nRows, unsigned _nCols)
		: Base(_name, _nRows, _nCols)
	{}
};

/** Output variable. */
class Output : public ExpressionType<Output, VT_OUTPUT>
{
	typedef ExpressionType<Output, VT_OUTPUT> Base;

public:
	Output() : Base() {}

	Output(const std::string& _name, unsigned _nRows, unsigned _nCols)
		: Base(_name, _nRows, _nCols)
	{}

	/** A constructor from an expression. */
	Output(const Expression& _expression, unsigned _componentIdx = 0)
	    : Base(_expression, _componentIdx)
	{}
};

/** Parameter variable. */
class Parameter : public ExpressionType<Parameter, VT_PARAMETER>
{
	typedef ExpressionType<Parameter, VT_PARAMETER> Base;

public:
	Parameter() : Base() {}

	Parameter(const std::string& _name, unsigned _nRows, unsigned _nCols)
		: Base(_name, _nRows, _nCols)
	{}
};

/** A time variable. */
class TIME : public ExpressionType<TIME, VT_TIME, false>
{
	typedef ExpressionType<TIME, VT_TIME, false> Base;

public:
	TIME() : Base() {}
};

/** Intermediate variable. */
class IntermediateState : public ExpressionType<IntermediateState, VT_INTERMEDIATE_STATE>
{
	typedef ExpressionType<IntermediateState, VT_INTERMEDIATE_STATE> Base;

public:
	IntermediateState() : Base() {}

	/** Default constructor */
	explicit IntermediateState(const std::string& _name, uint _nRows, uint _nCols)
		: Base(_name, _nRows, _nCols)
	{}

	/** Default constructor */
	explicit IntermediateState(const std::string& _name)
		: Base(_name, 1, 1)
	{}

	/** Default constructor */
	explicit IntermediateState( unsigned _nRows, unsigned _nCols = 1)
		: Base("", _nRows, _nCols)
	{}

	/** Default constructor */
	explicit IntermediateState( int _nRows, int _nCols = 1)
		: Base("", _nRows, _nCols)
	{}

	/** Copy constructor (deep copy). */
	IntermediateState( const double& _arg )
		: Base( )
	{
		assignmentSetup( _arg );
	}

	IntermediateState( const DVector& _arg )
		: Base( )
	{
		assignmentSetup( _arg );
	}

	IntermediateState( const DMatrix& _arg )
		: Base( )
	{
		assignmentSetup( _arg );
	}

	IntermediateState( const Operator& _arg )
		: Base( )
	{
		assignmentSetup( _arg );
	}

	IntermediateState( const Expression& _arg )
		: Base( )
	{
		assignmentSetup( _arg );
	}
};

CLOSE_NAMESPACE_ACADO

#endif // ACADO_TOOLKIT_VARIABLE_TYPES_HPP
