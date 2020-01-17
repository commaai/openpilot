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
 *    \file external_packages/acado_gnuplot/gnuplot_window.hpp
 *    \author Boris Houska, Hans Joachim Ferreau, Milan Vukov
 *    \date   2009-2013
 */


#ifndef ACADO_TOOLKIT_ACADO_2_GNUPLOT_HPP
#define ACADO_TOOLKIT_ACADO_2_GNUPLOT_HPP


#include <acado/user_interaction/plot_window.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Provides an interface to Gnuplot for plotting algorithmic outputs.
 *
 *	\ingroup ExternalFunctionality
 *
 *	The acado2gnuplot interface provides the functionality to easiliy
 *	plot data which is available in the ACADO Toolkit format. The methods
 *	that are implemented in this interface convert ACADO sturctures into
 *	a format that can be read by the program Gnuplot.
 *
 *	\author Boris Houska, Hans Joachim Ferreau, Milan Vukov
 */


class GnuplotWindow : public PlotWindow
{

    // PUBLIC FUNCTIONS:
    // -----------------

    public:

        /** Default constructor. */
        GnuplotWindow( );

		/** Constructor which takes the plot frequency. 
		 *
		 *	@param[in] _frequency	Frequency determining at which time instants the window is to be plotted.
		 */
		GnuplotWindow(	PlotFrequency _frequency
						);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] arg	Right-hand side object.
		 */
        GnuplotWindow(	const GnuplotWindow& arg
						);

        /** Destructor. */
        virtual ~GnuplotWindow( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] arg	Right-hand side object.
		 */
        GnuplotWindow& operator=(	const GnuplotWindow& arg
									);


		/** Clone operator returning a base class pointer to a deep copy
		 *	of respective class instance.
		 *
		 *	\return Base class pointer to a deep copy of respective class instance
		 */
		virtual PlotWindow* clone( ) const;


        /** Initializes the Gnuplot-thread.
		 *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_PLOT_WINDOW_CAN_NOT_BE_OPEN
		 */
        virtual returnValue init( );


        /** Actually sets-up and plots the data in a Gnuplot window.
		 *
		 *	@param[in] _frequency	Frequency determining at which time instants the window is to be plotted.
         *
         *	\return SUCCESSFUL_RETURN, \n
         *	        RET_PLOTTING_FAILED, \n
		 *	        RET_INVALID_ARGUMENTS, \n
		 *	        RET_PLOT_WINDOW_CAN_NOT_BE_OPEN
         */
		virtual returnValue replot(	PlotFrequency _frequency = PLOT_IN_ANY_CASE
									);


        /** Runs the Gnuplot window in waiting mode until a mouse event
         *  occurs.
		 *
		 *	\return SUCCESSFUL_RETURN
         */
        returnValue waitForMouseEvents( );

        /** Returns whether a mouse event has occured.
		 *
		 *	@param[out] mouseX	X coordinate of mouse click.
		 *	@param[out] mouseX	Y coordinate of mouse click.
         *
         *	\return BT_TRUE  iff mouse event occured, \n
         *	        BT_FALSE otherwise
         */
        BooleanType getMouseEvent(	double& mouseX,
									double& mouseY
									);

        /** Waits until a mouse event occurs.
		 *
		 *	@param[out] mouseX	X coordinate of mouse click.
		 *	@param[out] mouseX	Y coordinate of mouse click.
         *
		 *	\return SUCCESSFUL_RETURN, \n
		 *	        RET_NOT_YET_IMPLEMENTED
         */
        returnValue waitForMouseEvent(	double& mouseX,
										double& mouseY
										);



    // PROTECTED FUNCTIONS:
    // --------------------

    protected:
        /** Sets-up and plots the data in a Gnuplot window.
         *
         *	\return SUCCESSFUL_RETURN, \n
         *	        RET_PLOTTING_FAILED, \n
		 *	        RET_INVALID_ARGUMENTS, \n
		 *	        RET_PLOT_WINDOW_CAN_NOT_BE_OPEN
         */
		returnValue sendDataToGnuplot( );


        /** Generates string in Gnuplot syntax for plotting in given plot mode.
		 *
		 *	@param[in]  plotMode		Plot mode whose string needs to be generated, see the PlotMode documentation for details.
		 *	@param[out] plotModeString	String in Gnuplot syntax for plotting in given plot mode.
         *
		 *	\return SUCCESSFUL_RETURN
         */
		returnValue getPlotModeString(	PlotMode plotMode,
										std::string& plotModeString
										) const;

        /** Generates string in Gnuplot syntax for plotting in given plot style.
		 *
		 *	@param[in]  _type				Type of variable to be plotted.
		 *	@param[out] plotStyleString		String in Gnuplot syntax for plotting in given plot style.
         *
		 *	\return SUCCESSFUL_RETURN
         */
		returnValue getPlotStyleString(	VariableType _type,
										std::string& plotStyleString
										) const;

        /** Generates string in Gnuplot syntax for plotting given data grid.
		 *
		 *	@param[in]  _dataGrid			Date grid to be plotted.
		 *	@param[out] _plotDataString		String in Gnuplot syntax for plotting given data grid.
         *
		 *	\return SUCCESSFUL_RETURN
         */
		returnValue obtainPlotDataString(	VariablesGrid& _dataGrid,
											std::string& _plotDataString
											) const;
	


    // PROTECTED DATA MEMBERS:
    // -----------------------

    protected:

        FILE* gnuPipe;							/**< Pipe to Gnuplot. */

        BooleanType mouseEvent;					/**< Flag indicating whether window should wait for mouse events. */

        static int counter;						/**< Static counter for counting the number of GnuplotWindows. */
};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_ACADO_2_GNUPLOT_HPP

/*
 *	end of file
 */

