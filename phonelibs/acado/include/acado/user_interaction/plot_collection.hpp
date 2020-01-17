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
 *    \file include/acado/user_interaction/plot_collection.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_PLOT_COLLECTION_HPP
#define ACADO_TOOLKIT_PLOT_COLLECTION_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/symbolic_expression/symbolic_expression.hpp>

#include <acado/user_interaction/plot_window.hpp>


BEGIN_NAMESPACE_ACADO


/**
 *	\brief Provides a generic list of plot windows (for internal use).
 *
 *	\ingroup AuxiliaryFunctionality
 *
 *  The class PlotCollection manages a basic singly-linked list of plot windows 
 *  that allows to plot algorithmic outputs during runtime. It is intended for 
 *	internal use only, as all user-functionality is encapsulated within the 
 *	classes Plotting and PlotWindow.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class PlotCollection
{
	friend class Plotting;

	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:
		/** Default constructor. */
		PlotCollection( );

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		PlotCollection(	const PlotCollection& rhs
						);

		/** Destructor. */
		~PlotCollection( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		PlotCollection& operator=(	const PlotCollection& rhs
									);


		/** Returns the window of the singly-linked list with given index.
		 *
		 *	@param[in] idx	Index of desired window.
		 *
		 *  \return Window with given index. 
		 */
		inline PlotWindow& operator()(	uint idx
										);

		/** Returns the window of the singly-linked list with given index (const version).
		 *
		 *	@param[in] idx	Index of desired window.
		 *
		 *  \return Window with given index. 
		 */
		inline PlotWindow operator()(	uint idx
										) const;


		/** Adds a window to the singly-linked list.
		 *
		 *	@param[in] window	Window to be added.
		 *
		 *	\note This function is doing the same as the corresponding 
		 *	      addPlotWindow member function and is introduced for syntax reasons only.
		 *
		 *  \return >= 0: index of added window, \n
		 *	        -RET_PLOT_COLLECTION_CORRUPTED 
		 */
		int operator<<(	PlotWindow& window
						);

		/** Adds a window to the singly-linked list.
		 *
		 *	@param[in] window	Window to be added.
		 *
		 *  \return >= 0: index of added window, \n
		 *	        -RET_PLOT_COLLECTION_CORRUPTED 
		 */
		int addPlotWindow(	PlotWindow& window
							);


		/** Clears all windows from the singly-linked list.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue clearAllWindows( );


		/** Returns number of windows contained in the plot collection.
		 *
		 *  \return Number of windows
		 */
		inline uint getNumPlotWindows( ) const;


		returnValue enableNominalControls( );
		
		returnValue disableNominalControls( );


		returnValue enableNominalParameters( );
		
		returnValue disableNominalParameters( );


		returnValue enableNominalOutputs( );
		
		returnValue disableNominalOutputs( );


    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:


    //
    // DATA MEMBERS:
    //
	protected:
		PlotWindow* first;				/**< Pointer to first window of the singly-linked list. */
		PlotWindow* last;				/**< Pointer to last window of the singly-linked list. */

		uint number;					/**< Total number of windows within the singly-linked list of the collection. */
};


CLOSE_NAMESPACE_ACADO


#include <acado/user_interaction/plot_collection.ipp>


#endif	// ACADO_TOOLKIT_PLOT_COLLECTION_HPP


/*
 *	end of file
 */
