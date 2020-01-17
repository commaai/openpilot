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
 *    \file include/acado/user_interaction/user_interaction.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_USER_INTERACTION_HPP
#define ACADO_TOOLKIT_USER_INTERACTION_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/options.hpp>
#include <acado/user_interaction/logging.hpp>
#include <acado/user_interaction/plotting.hpp>



BEGIN_NAMESPACE_ACADO


/**
 *	\brief Encapsulates all user interaction for setting options, logging data and plotting results.
 *
 *	\ingroup AuxiliaryFunctionality
 *	
 *  The class UserInteraction encapsulates all user interaction for 
 *	setting options, logging data and plotting results.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class UserInteraction : public Options, public Logging, public Plotting
{
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor. */
		UserInteraction( );

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		UserInteraction(	const UserInteraction& rhs
							);

		/** Destructor. */
		virtual ~UserInteraction( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		UserInteraction& operator=(	const UserInteraction& rhs
									);


		/** Adds a window to the plot collection. This function is overloaded
		 *	here in order to add the plotDataRecord required to plot the window
		 *	to the log collection.
		 *
		 *	@param[in] _window	Window to be added.
		 *
		 *	\note This function is doing the same as the corresponding 
		 *	      addPlotWindow member function and is introduced for syntax reasons only.
		 *
		 *  \return >= 0: index of added record, \n
		 *	        -RET_PLOT_COLLECTION_CORRUPTED 
		 */
		virtual int operator<<(	PlotWindow& _window
								);

		/** Adds a window to the plot collection. This function is overloaded
		 *	here in order to add the plotDataRecord required to plot the window
		 *	to the log collection.
		 *
		 *	@param[in] _window	Window to be added.
		 *
		 *  \return >= 0: index of added record, \n
		 *	        -RET_PLOT_COLLECTION_CORRUPTED 
		 */
		virtual int addPlotWindow(	PlotWindow& _window
									);

		/** Adds a record to the log collection.
		 *
		 *	@param[in] record	Record to be added.
		 *
		 *	\note This function tunnels the corresponding function of 
		 *	      the Logging class. It is introduced to avoid syntax ambiguity only.
		 *
		 *  \return >= 0: index of added record, \n
		 *	        -RET_LOG_COLLECTION_CORRUPTED 
		 */
		virtual int operator<<(	LogRecord& _record
								);


    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

		/** Copies all collected logging information required to plot a given window.
		 *	This function is overloaded within the UserInterface class to syncronize 
		 *	the logging information collected elsewhere within the algorithm with the 
		 *	one stored within the given window for plotting.
		 *
		 *	@param[in] _window	Window to be plotted.
		 *
         *	\return SUCCESSFUL_RETURN
         */
		virtual returnValue getPlotDataFromMemberLoggings(	PlotWindow& _window
															) const;


		/** Gets current status of user interface.
		 *
		 *  \return Current status of user interface
		 */
		BlockStatus getStatus( ) const;

		/** Sets status of user interface.
		 *
		 *	@param[in]  _status		New status of user interface
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue setStatus(	BlockStatus _status
								);


    //
    // DATA MEMBERS:
    //
	protected:

		BlockStatus status;					/**< Current status of the user interface, see documentation of BlockStatus for details. */
};


CLOSE_NAMESPACE_ACADO


//#include <acado/user_interaction/user_interaction.ipp>


#endif	// ACADO_TOOLKIT_USER_INTERACTION_HPP


/*
 *	end of file
 */
