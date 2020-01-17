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
 *    \file include/acado/user_interaction/log_record.ipp
 *    \author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 *    \date 2008 - 2013
 */

//
// PUBLIC MEMBER FUNCTIONS:
//

BEGIN_NAMESPACE_ACADO

inline returnValue LogRecord::getAll(	LogName _name,
										MatrixVariablesGrid& values
										) const
{
	return getAll( (uint)_name,LRT_ENUM,values );
}

inline returnValue LogRecord::getAll(	const Expression& _name,
										MatrixVariablesGrid& values
										) const
{
	return getAll( _name.getComponent( 0 ),LRT_VARIABLE,values );
}


inline returnValue LogRecord::getFirst(	LogName _name,
										DMatrix& firstValue
										) const
{
	return getFirst( (uint)_name,LRT_ENUM,firstValue );
}

inline returnValue LogRecord::getFirst(	const Expression& _name,
										DMatrix& firstValue
										) const
{
	return getFirst( _name.getComponent( 0 ),LRT_VARIABLE,firstValue );
}

inline returnValue LogRecord::getFirst(	LogName _name,
										VariablesGrid& firstValue
										) const
{
	DMatrix tmp;
	getFirst( (uint)_name,LRT_ENUM,tmp );
	firstValue = tmp;

	return SUCCESSFUL_RETURN;
}

inline returnValue LogRecord::getFirst(	const Expression& _name,
										VariablesGrid& firstValue
										) const
{
	DMatrix tmp;
	getFirst( _name.getComponent( 0 ),LRT_VARIABLE,tmp );
	firstValue = tmp;
	firstValue.setType( _name.getVariableType() );

	return SUCCESSFUL_RETURN;
}



inline returnValue LogRecord::getLast(	LogName _name,
										DMatrix& lastValue
										) const
{
	return getLast( (uint)_name,LRT_ENUM,lastValue );
}


inline returnValue LogRecord::getLast(	const Expression& _name,
										DMatrix& lastValue
										) const
{
	return getLast( _name.getComponent( 0 ),LRT_VARIABLE,lastValue );
}


inline returnValue LogRecord::getLast(	LogName _name,
										VariablesGrid& lastValue
										) const
{
	DMatrix tmp;
	getLast( (uint)_name,LRT_ENUM,tmp );
	lastValue = tmp;

	return SUCCESSFUL_RETURN;
}


inline returnValue LogRecord::getLast(	const Expression& _name,
										VariablesGrid& lastValue
										) const
{
	DMatrix tmp;
	getLast( _name.getComponent( 0 ),LRT_VARIABLE,tmp );
	lastValue = tmp;

	return SUCCESSFUL_RETURN;
}



inline returnValue LogRecord::setAll(	LogName _name,
										const MatrixVariablesGrid& values
										)
{
	return setAll( (uint)_name,LRT_ENUM,values );
}


inline returnValue LogRecord::setAll(	const Expression& _name,
										const MatrixVariablesGrid& values
										)
{	
	return setAll( _name.getComponent( 0 ),LRT_VARIABLE,values );
}



inline returnValue LogRecord::setLast(	LogName _name,
										const DMatrix& value,
										double time
										)
{
	return setLast( (uint)_name,LRT_ENUM,value,time );
}


inline returnValue LogRecord::setLast(	const Expression& _name,
										const DMatrix& value,
										double time
										)
{	
	return setLast( _name.getComponent( 0 ),LRT_VARIABLE,value,time );
}


inline returnValue LogRecord::setLast(	LogName _name,
										VariablesGrid& value,
										double time
										)
{
	DMatrix tmp( value );
	return setLast( _name,tmp,time );
}


inline returnValue LogRecord::setLast(	const Expression& _name,
										VariablesGrid& value,
										double time
										)
{
	DMatrix tmp( value );
	return setLast( _name,tmp,time );
}

inline uint LogRecord::getNumItems( ) const
{
	return items.size();
}

inline BooleanType LogRecord::isEmpty( ) const
{
	return (items.size() == 0);
}


inline LogFrequency LogRecord::getLogFrequency( ) const
{
	return frequency;
}


inline PrintScheme LogRecord::getPrintScheme( ) const
{
	return printScheme;
}


inline returnValue LogRecord::setLogFrequency(	LogFrequency _frequency
												)
{
	frequency = _frequency;
	return SUCCESSFUL_RETURN;
}


inline returnValue LogRecord::setPrintScheme(	PrintScheme _printScheme
												)
{
	printScheme = _printScheme;
	return SUCCESSFUL_RETURN;
}

inline BooleanType LogRecord::hasItem(	LogName _name
										) const
{
	if (items.count(std::make_pair(_name, LRT_ENUM)))
		return true;
	
	return false;
}


inline BooleanType LogRecord::hasItem(	const Expression& _name
										) const
{
	if (items.count(std::make_pair(_name.getComponent( 0 ), LRT_VARIABLE)))
		return true;
	
	return false;
}


inline BooleanType LogRecord::hasNonEmptyItem(	LogName _name
												) const
{
	LogRecordItems::const_iterator it;
	it = items.find(std::make_pair(_name, LRT_ENUM));
	if (it == items.end())
		return false;
	if (it->second.values.isEmpty( ) == false)
		return true;
		
	return false;
}


inline BooleanType LogRecord::hasNonEmptyItem(	const Expression& _name
												) const
{
	LogRecordItems::const_iterator it;
	it = items.find(std::make_pair(_name.getComponent( 0 ), LRT_VARIABLE));
	if (it == items.end())
		return false;
	if (it->second.values.isEmpty( ) == false)
		return true;
		
	return false;
}

inline uint LogRecord::getNumDoubles( ) const
{
	LogRecordItems::const_iterator it;
	unsigned nDoubles = 0;

	for (it = items.begin(); it != items.end(); ++it)
		nDoubles += it->second.values.getDim();

	return nDoubles;
}

inline returnValue LogRecord::enableWriteProtection(        LogName _name
                                                                                                                )
{
	LogRecordItems::iterator it;
	it = items.find(std::make_pair(_name, LRT_ENUM));
	if (it == items.end())
		return SUCCESSFUL_RETURN;
	
	it->second.writeProtection = true;	
	
	return SUCCESSFUL_RETURN;
}


inline returnValue LogRecord::enableWriteProtection(        const Expression& _name
                                                                                                                )
{
	LogRecordItems::iterator it;
	it = items.find(std::make_pair(_name.getComponent( 0 ), LRT_VARIABLE));
	if (it == items.end())
		return SUCCESSFUL_RETURN;
	
	it->second.writeProtection = true;	
	
	return SUCCESSFUL_RETURN;
}


inline returnValue LogRecord::disableWriteProtection(        LogName _name
                                                                                                                )
{
	LogRecordItems::iterator it;
	it = items.find(std::make_pair(_name, LRT_ENUM));
	if (it == items.end())
		return SUCCESSFUL_RETURN;
	
	it->second.writeProtection = false;	
	
	return SUCCESSFUL_RETURN;
}


inline returnValue LogRecord::disableWriteProtection(        const Expression& _name
                                                                                                                )
{
	LogRecordItems::iterator it;
	it = items.find(std::make_pair(_name.getComponent( 0 ), LRT_VARIABLE));
	if (it == items.end())
		return SUCCESSFUL_RETURN;
	
	it->second.writeProtection = false;	
	
	return SUCCESSFUL_RETURN;
}



//
// PROTECTED INLINED MEMBER FUNCTIONS:
//

inline BooleanType LogRecord::hasItem(	uint _name,
										LogRecordItemType _type
										) const
{
	if (items.count(std::make_pair(_name, _type)))
		return true;
	
	return false;
}

CLOSE_NAMESPACE_ACADO

/*
 *	end of file
 */
