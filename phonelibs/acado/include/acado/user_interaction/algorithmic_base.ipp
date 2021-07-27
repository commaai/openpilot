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
 *    \file include/acado/user_interaction/algorithmic_base.ipp
 *    \author Hans Joachim Ferreau, Boris Houska
 */



BEGIN_NAMESPACE_ACADO

//
// PUBLIC MEMBER FUNCTIONS:
//


inline returnValue AlgorithmicBase::getAll(	LogName _name,
											MatrixVariablesGrid& values
											) const
{
	return userInteraction->getAll( _name,values );
}


inline returnValue AlgorithmicBase::getFirst(	LogName _name,
												DMatrix& firstValue
												) const
{
	return userInteraction->getFirst( _name,firstValue );
}


inline returnValue AlgorithmicBase::getFirst(	LogName _name,
												VariablesGrid& firstValue
												) const
{
	return userInteraction->getFirst( _name,firstValue );
}


inline returnValue AlgorithmicBase::getLast(	LogName _name,
												DMatrix& lastValue
												) const
{
	return userInteraction->getLast( _name,lastValue );
}


inline returnValue AlgorithmicBase::getLast(	LogName _name,
												VariablesGrid& lastValue
												) const
{
	return userInteraction->getLast( _name,lastValue );
}


//
// PROTECTED MEMBER FUNCTIONS:
//


inline returnValue AlgorithmicBase::get(	OptionsName name,
											int& value
											) const
{
	return userInteraction->get( name,value );
}


inline returnValue AlgorithmicBase::get(	OptionsName name,
											double& value
											) const
{
	return userInteraction->get( name,value );
}

inline returnValue AlgorithmicBase::get(	OptionsName name,
											std::string& value
											) const
{
	return userInteraction->get( name,value );
}


inline returnValue AlgorithmicBase::get(	uint idx,
											OptionsName name,
											int& value
											) const
{
	return userInteraction->get( idx,name,value );
}


inline returnValue AlgorithmicBase::get(	uint idx,
											OptionsName name,
											double& value
											) const
{
	return userInteraction->get( idx,name,value );
}



inline returnValue AlgorithmicBase::addOption(	OptionsName name,
												int value
												)
{
	return userInteraction->addOption( name,value );
}


inline returnValue AlgorithmicBase::addOption(	OptionsName name,
												double value
												)
{
	return userInteraction->addOption( name,value );
}


inline returnValue AlgorithmicBase::addOption(	uint idx,
												OptionsName name,
												int value
												)
{
	return userInteraction->addOption( idx,name,value );
}


inline returnValue AlgorithmicBase::addOption(	uint idx,
												OptionsName name,
												double value
												)
{
	return userInteraction->addOption( idx,name,value );
}



inline BooleanType AlgorithmicBase::haveOptionsChanged( ) const
{
	return userInteraction->haveOptionsChanged( );
}


inline BooleanType AlgorithmicBase::haveOptionsChanged(	uint idx
														) const
{
	return userInteraction->haveOptionsChanged( idx );
}



inline returnValue AlgorithmicBase::setAll(	LogName _name,
											const MatrixVariablesGrid& values
											)
{
	return userInteraction->setAll(_name, values);
}



inline returnValue AlgorithmicBase::setLast(	LogName _name,
												int value,
												double time
												)
{
	return userInteraction->setLast( _name,DMatrix((double)value),time );
}


inline returnValue AlgorithmicBase::setLast(	LogName _name,
												double value,
												double time
												)
{
	return userInteraction->setLast( _name, DMatrix(value), time );
}


inline returnValue AlgorithmicBase::setLast(	LogName _name,
												const DVector& value,
												double time
												)
{
	return userInteraction->setLast( _name, value,time );
}


inline returnValue AlgorithmicBase::setLast(	LogName _name,
												const DMatrix& value,
												double time
												)
{
	return userInteraction->setLast( _name, value,time );	
}


inline returnValue AlgorithmicBase::setLast(	LogName _name,
												const VariablesGrid& value,
												double time
												)
{
	return userInteraction->setLast( _name, value,time );
}



inline int AlgorithmicBase::addLogRecord(	LogRecord& _record
											)
{
	return userInteraction->addLogRecord( _record );
}


inline returnValue AlgorithmicBase::printLogRecord(	std::ostream& _stream,
													int idx,
													LogPrintMode _mode
													) const
{
	if ( ( idx < 0 ) || ( idx >= (int)userInteraction->logCollection.size( ) ) )
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );
		
	return userInteraction->logCollection[ idx ].print(_stream, _mode);
}


inline returnValue AlgorithmicBase::plot(	PlotFrequency _frequency
											)
{
	return userInteraction->plot( _frequency );
}


inline returnValue AlgorithmicBase::replot(	PlotFrequency _frequency
											)
{
	return userInteraction->replot( _frequency );
}


CLOSE_NAMESPACE_ACADO


/*
 *	end of file
 */
