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
 *    \file include/acado/user_interaction/plotting.ipp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 12.05.2009
 */


BEGIN_NAMESPACE_ACADO



//
// PUBLIC MEMBER FUNCTIONS:
//

inline int Plotting::operator<<(	PlotWindow& _window
								)
{
	return addPlotWindow( _window );
}


inline int Plotting::addPlotWindow(	PlotWindow& _window
									)
{
	return plotCollection.addPlotWindow( _window );
}


inline returnValue Plotting::getPlotWindow(	uint idx,
											PlotWindow& _window
											) const
{
	if (idx >= getNumPlotWindows( ))
		return ACADOERROR( RET_INDEX_OUT_OF_BOUNDS );

	_window = plotCollection( idx );

	return getPlotDataFromMemberLoggings( _window );
}


inline returnValue Plotting::getPlotWindow(	PlotWindow& _window
											) const
{
	return getPlotWindow( _window.getAliasIdx( ),_window );
}


inline uint Plotting::getNumPlotWindows( ) const
{
	return plotCollection.getNumPlotWindows( );
}



CLOSE_NAMESPACE_ACADO


/*
 *	end of file
 */
