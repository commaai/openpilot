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
 *    \file include/acado/user_interaction/plot_window_subplot.ipp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 12.06.2008
 */


//
// PUBLIC MEMBER FUNCTIONS:
//


BEGIN_NAMESPACE_ACADO


inline returnValue PlotWindowSubplot::setTitle(	const std::string& _title
												)
{
	title = _title;
	return SUCCESSFUL_RETURN;
}

inline returnValue PlotWindowSubplot::setXLabel(	const std::string& _xLabel
													)
{
	xLabel = _xLabel;

	return SUCCESSFUL_RETURN;
}


inline returnValue PlotWindowSubplot::setYLabel(	const std::string& _yLabel
													)
{
	yLabel = _yLabel;

	return SUCCESSFUL_RETURN;
}


inline returnValue PlotWindowSubplot::setPlotMode(	PlotMode _plotMode
													)
{
	plotMode = _plotMode;
	return SUCCESSFUL_RETURN;
}


inline returnValue PlotWindowSubplot::setPlotFormat(	PlotFormat _plotFormat
														)
{
	plotFormat = _plotFormat;
	return SUCCESSFUL_RETURN;
}


inline returnValue PlotWindowSubplot::setRanges(	double _xRangeLowerLimit,
													double _xRangeUpperLimit,
													double _yRangeLowerLimit,
													double _yRangeUpperLimit
													)
{
	if ( _xRangeLowerLimit > _xRangeUpperLimit )
		return ACADOERROR( RET_INVALID_ARGUMENTS );

	if ( _yRangeLowerLimit > _yRangeUpperLimit )
		return ACADOERROR( RET_INVALID_ARGUMENTS );

	xRangeLowerLimit = _xRangeLowerLimit;
	xRangeUpperLimit = _xRangeUpperLimit;
	yRangeLowerLimit = _yRangeLowerLimit;
	yRangeUpperLimit = _yRangeUpperLimit;

	return SUCCESSFUL_RETURN;
}


inline returnValue PlotWindowSubplot::getTitle(	std::string& _title
												)
{
	_title = title;
	return SUCCESSFUL_RETURN; 
}

inline returnValue PlotWindowSubplot::getXLabel(	std::string& _xLabel
													)
{
	_xLabel = xLabel;
	return SUCCESSFUL_RETURN;
}


inline returnValue PlotWindowSubplot::getYLabel(	std::string& _yLabel
													)
{
	_yLabel = yLabel;
	return SUCCESSFUL_RETURN;
}


inline PlotMode PlotWindowSubplot::getPlotMode( ) const
{
	return plotMode;
}


inline PlotFormat PlotWindowSubplot::getPlotFormat( ) const
{
	return plotFormat;
}


inline returnValue PlotWindowSubplot::getRanges(	double& _xRangeLowerLimit,
													double& _xRangeUpperLimit,
													double& _yRangeLowerLimit,
													double& _yRangeUpperLimit
													) const
{
	_xRangeLowerLimit = xRangeLowerLimit;
	_xRangeUpperLimit = xRangeUpperLimit;
	_yRangeLowerLimit = yRangeLowerLimit;
	_yRangeUpperLimit = yRangeUpperLimit;

	return SUCCESSFUL_RETURN;
}


inline uint PlotWindowSubplot::getNumLines( ) const
{
	return nLines;
}


inline uint PlotWindowSubplot::getNumData( ) const
{
	return nData;
}


inline SubPlotType PlotWindowSubplot::getSubPlotType( ) const
{
	if ( plotVariableY != 0 )
	{
		if ( plotVariableX != 0 )
			return SPT_VARIABLE_VARIABLE;

		if ( plotExpressionX != 0 )
			return SPT_EXPRESSION_VARIABLE;

		return SPT_VARIABLE;
	}

	if ( plotVariablesGrid != 0 )
		return SPT_VARIABLES_GRID;

	if ( plotExpressionY != 0 )
	{
		if ( plotExpressionX != 0 )
			return SPT_EXPRESSION_EXPRESSION;

		if ( plotVariableX != 0 )
			return SPT_VARIABLE_EXPRESSION;

		return SPT_EXPRESSION;
	}

	if ( plotEnum != PLOT_NOTHING )
		return SPT_ENUM;

	return SPT_UNKNOWN;
}


inline VariableType PlotWindowSubplot::getXVariableType( ) const
{
	if ( plotVariableX == 0 )
		return VT_TIME;
	else
		return plotVariableX->getVariableType( );
}


inline VariableType PlotWindowSubplot::getYVariableType( ) const
{
	if ( plotVariableY == 0 )
		if ( plotVariablesGrid == 0 )
			return VT_UNKNOWN;
		else
			return plotVariablesGrid->getType( );
	else
		return plotVariableY->getVariableType( );
}


inline Expression* PlotWindowSubplot::getXPlotExpression( ) const
{
	return plotExpressionX;
}


inline Expression* PlotWindowSubplot::getYPlotExpression( ) const
{
	return plotExpressionY;
}


inline PlotName PlotWindowSubplot::getPlotEnum( ) const
{
	return plotEnum;
}


inline returnValue PlotWindowSubplot::setNext( PlotWindowSubplot* const _next )
{
	next = _next;
	return SUCCESSFUL_RETURN;
}


inline PlotWindowSubplot* PlotWindowSubplot::getNext( ) const
{
	return next;
}





CLOSE_NAMESPACE_ACADO


/*
 *	end of file
 */
