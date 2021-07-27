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
 *    \file include/acado/user_interaction/plotting.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_PLOTTING_HPP
#define ACADO_TOOLKIT_PLOTTING_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/plot_collection.hpp>
#include <acado/user_interaction/plot_window.hpp>



BEGIN_NAMESPACE_ACADO


static PlotWindow emptyPlotWindow;


/**
 *	\brief Provides a generic way to plot algorithmic outputs during runtime.
 *
 *	\ingroup AuxiliaryFunctionality
 *	
 *  The class Plotting provides a generic way to plot algorithmic outputs
 *	during runtime. This class is part of the UserInterface class, i.e. all classes 
 *	that are intended to interact with the user inherit the public functionality 
 *	of the Plotting class.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class Plotting
{
	friend class AlgorithmicBase;
	
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor.
		 */
		Plotting( );

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		Plotting(	const Plotting& rhs
					);

		/** Destructor. 
		 */
		virtual ~Plotting( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		Plotting& operator=(	const Plotting& rhs
							);


		/** Adds a window to the plot collection.
		 *
		 *	@param[in] _window	Window to be added.
		 *
		 *	\note This function is doing the same as the corresponding 
		 *	      addPlotWindow member function and is introduced for syntax reasons only.
		 *
		 *  \return >= 0: index of added record, \n
		 *	        -RET_PLOT_COLLECTION_CORRUPTED 
		 */
		inline int operator<<(	PlotWindow& _window
								);

		/** Adds a window to the plot collection.
		 *
		 *	@param[in] _window	Window to be added.
		 *
		 *  \return >= 0: index of added record, \n
		 *	        -RET_PLOT_COLLECTION_CORRUPTED 
		 */
		inline int addPlotWindow(	PlotWindow& _window
									);


		/** Returns the window with given index from the plot collection.
		 *
		 *	@param[in]  idx			Index of desired window.
		 *	@param[out] _window		Desired window.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS 
		 */
		inline returnValue getPlotWindow(	uint idx,
											PlotWindow& _window
											) const;

		/** Returns the window with certain index from the plot collection. 
		 *	This index is not provided when calling the function, but 
		 *	rather obtained by using the alias index of the window. If the
		 *	window is no alias window, the error RET_INDEX_OUT_OF_BOUNDS is thrown.
		 *
		 *	@param[out] _window		Desired window.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS 
		 */
		inline returnValue getPlotWindow(	PlotWindow& _window
											) const;


        /** Plots all windows of the plot collection, each one into a new figure.
		 *
		 *	@param[in] _frequency	Frequency determining at which time instants the window is to be plotted.
         *
         *	\return SUCCESSFUL_RETURN
         */
		virtual returnValue plot(	PlotFrequency _frequency = PLOT_IN_ANY_CASE
									);

        /** Plots all windows of the plot collection, each one into the 
		 *	corresponding existing figure, if possible.
		 *
		 *	@param[in] _frequency	Frequency determining at which time instants the window is to be plotted.
         *
         *	\return SUCCESSFUL_RETURN
         */
		virtual returnValue replot(	PlotFrequency _frequency = PLOT_IN_ANY_CASE
									);


		/** Returns number of windows contained in the plot collection.
		 *
		 *  \return Number of windows
		 */
		inline uint getNumPlotWindows( ) const;



    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

		/** Copies all collected logging information required to plot a given window.
		 *
		 *	@param[in] _window	Window to be plotted.
		 *
		 *	\note This function is overloaded within the UserInterface class to
		 *	      syncronize the logging information collected elsewhere within the
		 *	      algorithm with the one stored within the given window for plotting.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue getPlotDataFromMemberLoggings(	PlotWindow& _window
															) const;


    //
    // DATA MEMBERS:
    //
	protected:
		PlotCollection plotCollection;				/**< Plot collection containing a singly-linked list of plot windows. */
};


CLOSE_NAMESPACE_ACADO


#include <acado/user_interaction/plotting.ipp>


#endif	// ACADO_TOOLKIT_PLOTTING_HPP


/*
 *	end of file
 */
