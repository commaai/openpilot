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
 *    \file include/acado/user_interaction/plot_window.ipp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 12.06.2008
 */


//
// PUBLIC MEMBER FUNCTIONS:
//


BEGIN_NAMESPACE_ACADO


inline PlotWindowSubplot& PlotWindow::operator()(	uint idx
												)
{
	ASSERT( idx < getNumSubplots( ) );

	PlotWindowSubplot* current = first;
	for( uint i=0; i<idx; ++i )
		current = current->getNext( );	

	return *current;
}


inline PlotWindowSubplot PlotWindow::operator()(	uint idx
											) const
{
	ASSERT( idx < getNumSubplots( ) );

	PlotWindowSubplot* current = first;
	for( uint i=0; i<idx; ++i )
		current = current->getNext( );	

	return *current;
}


inline PlotFrequency PlotWindow::getPlotFrequency( ) const
{
	return frequency;
}


inline uint PlotWindow::getNumSubplots( ) const
{
	return number;
}


inline BooleanType PlotWindow::isEmpty( ) const
{
	if ( getNumSubplots( ) == 0 )
		return BT_TRUE;
	else
		return BT_FALSE;
}



inline returnValue PlotWindow::getPlotDataRecord(	LogRecord& _record
													) const
{
	_record = plotDataRecord;
	return SUCCESSFUL_RETURN;
}


inline returnValue PlotWindow::setPlotDataRecord(	LogRecord& _record
													)
{
	plotDataRecord = _record;
	return SUCCESSFUL_RETURN;
}


inline returnValue PlotWindow::setPlotData(	const Expression& _name,
											VariablesGrid& value
											)
{
	plotDataRecord.setLast( _name,value );
	plotDataRecord.enableWriteProtection( _name );
	
	return SUCCESSFUL_RETURN;
}


inline returnValue PlotWindow::setPlotData(	LogName _name,
											VariablesGrid& value
											)
{
	plotDataRecord.setLast( _name,value );
	plotDataRecord.enableWriteProtection( _name );
	
	return SUCCESSFUL_RETURN;
}


inline BooleanType PlotWindow::isAlias( ) const
{
	if ( aliasIdx < 0 )
		return BT_FALSE;
	else
		return BT_TRUE;
}


inline int PlotWindow::getAliasIdx( ) const
{
	return aliasIdx;
}



//
// PROTECTED INLINED MEMBER FUNCTIONS:
//


inline returnValue PlotWindow::setNext( PlotWindow* const _next )
{
	next = _next;
	return SUCCESSFUL_RETURN;
}


inline PlotWindow* PlotWindow::getNext( ) const
{
	return next;
}



inline returnValue PlotWindow::setAliasIdx(	int _aliasIdx
											)
{
	aliasIdx = _aliasIdx;
	return SUCCESSFUL_RETURN;
}



inline returnValue PlotWindow::enableNominalControls( )
{
	shallPlotNominalControls = BT_TRUE;
	return SUCCESSFUL_RETURN;
}


inline returnValue PlotWindow::disableNominalControls( )
{
	shallPlotNominalControls = BT_FALSE;
	return SUCCESSFUL_RETURN;
}



inline returnValue PlotWindow::enableNominalParameters( )
{
	shallPlotNominalParameters = BT_TRUE;
	return SUCCESSFUL_RETURN;
}


inline returnValue PlotWindow::disableNominalParameters( )
{
	shallPlotNominalParameters = BT_FALSE;
	return SUCCESSFUL_RETURN;
}



inline returnValue PlotWindow::enableNominalOutputs( )
{
	shallPlotNominalOutputs = BT_TRUE;
	return SUCCESSFUL_RETURN;
}


inline returnValue PlotWindow::disableNominalOutputs( )
{
	shallPlotNominalOutputs = BT_FALSE;
	return SUCCESSFUL_RETURN;
}


CLOSE_NAMESPACE_ACADO


/*
 *	end of file
 */
