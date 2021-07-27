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
 *    \file include/acado/user_interaction/algorithmic_base.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 */


#ifndef ACADO_TOOLKIT_ALGORITHMIC_BASE_HPP
#define ACADO_TOOLKIT_ALGORITHMIC_BASE_HPP


#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/user_interaction.hpp>



BEGIN_NAMESPACE_ACADO


/**
 *	\brief Base class for all algorithmic modules within the ACADO Toolkit providing some basic functionality.
 *
 *	\ingroup AuxiliaryFunctionality
 *	
 *  The class AlgorithmicBase is a base class for all algorithmic modules
 *	within the ACADO Toolkit. It provides a number of basic functionality
 *	such as routines to cope with options, logging and plotting objects.
 *
 *	The class AlgorithmicBase only tunnels this functionality to a pointer 
 *	of the UserInteraction class. There are two different possibilities for 
 *	using this class: \n
 *
 *	(i)  Using this class stand-alone: if no pointer to an instance of the 
 *	     UserInteraction class is provided, an object of this class is 
 *	     automatically allocated (with default options and logging information
 *	     as specified within each algorithmic module). \n
 *
 *	(ii) Using this class as (possibly nested) member of a top-level user interface:
 *	     Usually an user interacts with a top-level user interface derived from
 *	     the class UserInteraction. This interface does always have algorithmic 
 *	     modules as members for doing the actual computations. For ensuring that 
 *	     options, logging and plotting information specified by the user is actually
 *	     handled by the algorithmic modules, they are provided with a pointer 
 *	     to this top-level user interface. \n
 *
 *	\note This class tunnels parts of the public functionality of the 
 *	Options/OptionsList, Logging/LogCollection and Plotting/PlotCollection 
 *	classes, respectively. In case their public functionality is modified, 
 *	the AlgorithmicBase class has to be adapted accordingly.
 *
 *	\author Hans Joachim Ferreau, Boris Houska
 */
class AlgorithmicBase
{
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor for using the algorithmic module stand-alone.
		 */
		AlgorithmicBase( );

		/** Constructor which takes a pointer to an instance of the
		 *	UserInterface class. If this pointer is NULL, this constructor 
		 *	is equivalent to the default constructor.
		 *
		 *	@param[in] _userInteraction		Pointer to top-level user interface.
		 */
		AlgorithmicBase(	UserInteraction* _userInteraction
							);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		AlgorithmicBase(	const AlgorithmicBase& rhs
							);

		/** Destructor. */
		virtual ~AlgorithmicBase( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		AlgorithmicBase& operator=(	const AlgorithmicBase& rhs
									);


		/** Adds an additional OptionsList to internal array.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue addOptionsList( );


		/** Sets value of an existing option item of integer type to a given value.
		 *
		 *	@param[in] name		Name of option item.
		 *	@param[in] value	New value of option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_DOESNT_EXISTS, \n
		 *          RET_OPTIONS_LIST_CORRUPTED
		 */
		returnValue set(	OptionsName name,
							int value
							);

		/** Sets value of an existing option item of double type to a given value.
		 *
		 *	@param[in] name		Name of option item.
		 *	@param[in] value	New value of option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_DOESNT_EXISTS, \n
		 *          RET_OPTIONS_LIST_CORRUPTED
		 */
		returnValue set(	OptionsName name,
							double value
							);

		/** Sets value of an existing option item of double type to a string value.
		 *
		 *	@param[in] name		Name of option item.
		 *	@param[in] value	New value of option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_DOESNT_EXISTS, \n
		 *          RET_OPTIONS_LIST_CORRUPTED
		 */
		returnValue set(	OptionsName name,
							const std::string& value
							);

		/** Sets value of an existing option item of integer type 
		 *	within the option list of given index to a given value.
		 *
		 *	@param[in]  idx		Index of option list.
		 *	@param[in] name		Name of option item.
		 *	@param[in] value	New value of option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_DOESNT_EXISTS, \n
		 *          RET_OPTIONS_LIST_CORRUPTED, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		returnValue set(	uint idx,
							OptionsName name,
							int value
							);

		/** Sets value of an existing option item of double type 
		 *	within the option list of given index to a given value.
		 *
		 *	@param[in]  idx		Index of option list.
		 *	@param[in] name		Name of option item.
		 *	@param[in] value	New value of option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_DOESNT_EXISTS, \n
		 *          RET_OPTIONS_LIST_CORRUPTED, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		returnValue set(	uint idx,
							OptionsName name,
							double value
							);


        /** Assigns a given Options object to this object.
		 *
		 *	@param[in] arg		New Options object to be assigned.
		 *
		 *	\note This routine is introduced only for convenience and
         *	      is equivalent to the assignment operator.
		 *
         *  \return SUCCESSFUL_RETURN
         */
        returnValue setOptions(	const Options &arg
								);

        /** Assigns the option list with given index of a given Options object 
		 *	to option list with given index of this object.
		 *
		 *	@param[in] idx		Index of option list.
		 *	@param[in] arg		Options object containing the option list to be assigned.
		 *
         *  \return SUCCESSFUL_RETURN
         */
		returnValue setOptions(	uint idx,
								const Options &arg
								);

        /** Returns an Options object containing exactly the option list with given index.
		 *
		 *	@param[in] idx		Index of option list.
		 *
         *  \return Options object containing exactly the option list with given index
         */
		Options getOptions(	uint idx
							) const;


		/** Gets all numerical values at all time instants of the item
		 *	with given name. If this item exists in more than one record,
		 *	the first one is choosen as they are expected to have identical
		 *	values anyhow.
		 *
		 *	@param[in]  _name	Internal name of item.
		 *	@param[out] values	All numerical values at all time instants of given item.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST
		 */
		inline returnValue getAll(	LogName _name,
									MatrixVariablesGrid& values
									) const;


		/** Gets numerical value at first time instant of the item
		 *	with given name. If this item exists in more than one record,
		 *	the first one is choosen as they are expected to have identical
		 *	values anyhow.
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[out] firstValue	Numerical value at first time instant of given item.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST
		 */
		inline returnValue getFirst(	LogName _name,
										DMatrix& firstValue
										) const;

		/** Gets numerical value at first time instant of the item
		 *	with given name (converts internally used DMatrix into VariablesGrid).
		 *	If this item exists in more than one record, the first one is choosen
		 *	as they are expected to have identical values anyhow.
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[out] firstValue	Numerical value at first time instant of given item.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST
		 */
		inline returnValue getFirst(	LogName _name,
										VariablesGrid& firstValue
										) const;


		/** Gets numerical value at last time instant of the item
		 *	with given name. If this item exists in more than one record,
		 *	the first one is choosen as they are expected to have identical
		 *	values anyhow.
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[out] lastValue	Numerical value at last time instant of given item.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST
		 */
		inline returnValue getLast(	LogName _name,
									DMatrix& lastValue
									) const;

		/** Gets numerical value at last time instant of the item
		 *	with given name (converts internally used DMatrix into VariablesGrid).
		 *	If this item exists in more than one record, the first one is choosen
		 *	as they are expected to have identical values anyhow.
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[out] lastValue	Numerical value at last time instant of given item.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST
		 */
		inline returnValue getLast(	LogName _name,
									VariablesGrid& lastValue
									) const;


		/** Returns value of an existing option item of integer type.
		 *
		 *	@param[in]  name	Name of option item.
		 *	@param[out] value	Value of option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_DOESNT_EXISTS
		 */
		inline returnValue get(	OptionsName name,
								int& value
								) const;

		/** Returns value of an existing option item of double type.
		 *
		 *	@param[in]  name	Name of option item.
		 *	@param[out] value	Value of option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_DOESNT_EXISTS
		 */
		inline returnValue get(	OptionsName name,
								double& value
								) const;

		/** Returns value of an existing option item of string type.
		 *
		 *	@param[in]  name	Name of option item.
		 *	@param[out] value	Value of option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_DOESNT_EXISTS
		 */
		inline returnValue get(	OptionsName name,
								std::string& value
								) const;

		/** Returns value of an existing option item of integer type 
		 *	within the option list of given index.
		 *
		 *	@param[in]  idx		Index of option list.
		 *	@param[in]  name	Name of option item.
		 *	@param[out] value	Value of option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_DOESNT_EXISTS, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		inline returnValue get(	uint idx,
								OptionsName name,
								int& value
								) const;

		/** Returns value of an existing option item of double type 
		 *	within the option list of given index.
		 *
		 *	@param[in]  idx		Index of option list.
		 *	@param[in]  name	Name of option item.
		 *	@param[out] value	Value of option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_DOESNT_EXISTS, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		inline returnValue get(	uint idx,
								OptionsName name,
								double& value
								) const;


		/** Add an option item with a given integer default value to the all option lists.
		 *
		 *	@param[in] name		Name of new option item.
		 *	@param[in] value	Default value of new option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_ALREADY_EXISTS, \n
		 *          RET_OPTIONS_LIST_CORRUPTED
		 */
		inline returnValue addOption(	OptionsName name,
										int value
										);

		/** Add an option item with a given double default value to the all option lists.
		 *
		 *	@param[in] name		Name of new option item.
		 *	@param[in] value	Default value of new option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_ALREADY_EXISTS, \n
		 *          RET_OPTIONS_LIST_CORRUPTED
		 */
		inline returnValue addOption(	OptionsName name,
										double value
										);

		/** Add an option item with a given integer default value to option list with given index.
		 *
		 *	@param[in] idx		Index of option list.
		 *	@param[in] name		Name of new option item.
		 *	@param[in] value	Default value of new option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_ALREADY_EXISTS, \n
		 *          RET_OPTIONS_LIST_CORRUPTED, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		inline returnValue addOption(	uint idx,
										OptionsName name,
										int value
										);

		/** Add an option item with a given double default value to option list with given index.
		 *
		 *	@param[in] idx		Index of option list.
		 *	@param[in] name		Name of new option item.
		 *	@param[in] value	Default value of new option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_ALREADY_EXISTS, \n
		 *          RET_OPTIONS_LIST_CORRUPTED, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS
		 */
		inline returnValue addOption(	uint idx,
										OptionsName name,
										double value
										);


		/** Determines whether options of at least one option list have been modified.
		 *
		 *	\return BT_TRUE  iff options have been modified, \n
		 *	        BT_FALSE otherwise 
		 */
		inline BooleanType haveOptionsChanged( ) const;

		/** Determines whether options of option list with given index have been modified.
		 *
		 *	@param[in] idx		Index of option list.
		 *
		 *	\return BT_TRUE  iff options have been modified, \n
		 *	        BT_FALSE otherwise 
		 */
		inline BooleanType haveOptionsChanged(	uint idx
												) const;


		/** Sets all numerical values at all time instants of all items
		 *	with given name within all records.
		 *
		 *	@param[in]  _name	Internal name of item.
		 *	@param[in]  values	All numerical values at all time instants of given item.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_COLLECTION_CORRUPTED
		 */
		inline returnValue setAll(	LogName _name,
									const MatrixVariablesGrid& values
									);


		/** Sets numerical value at last time instant of all items
		 *	with given name within all records.
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[in]  lastValue	Numerical value at last time instant of given item.
		 *	@param[in]  time		Time label of the instant.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST
		 */
		inline returnValue setLast(	LogName _name,
									int lastValue,
									double time = -INFTY
									);

		/** Sets numerical value at last time instant of all items
		 *	with given name within all records.
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[in]  lastValue	Numerical value at last time instant of given item.
		 *	@param[in]  time		Time label of the instant.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST
		 */
		inline returnValue setLast(	LogName _name,
									double lastValue,
									double time = -INFTY
									);

		/** Sets numerical value at last time instant of all items
		 *	with given name within all records.
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[in]  lastValue	Numerical value at last time instant of given item.
		 *	@param[in]  time		Time label of the instant.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST
		 */
		inline returnValue setLast(	LogName _name,
									const DVector& lastValue,
									double time = -INFTY
									);

		/** Sets numerical value at last time instant of all items
		 *	with given name within all records.
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[in]  lastValue	Numerical value at last time instant of given item.
		 *	@param[in]  time		Time label of the instant.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST
		 */
		inline returnValue setLast(	LogName _name,
									const DMatrix& lastValue,
									double time = -INFTY
									);

		/** Sets numerical value at last time instant of all items
		 *	with given name within all records.
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[in]  lastValue	Numerical value at last time instant of given item.
		 *	@param[in]  time		Time label of the instant.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST
		 */
		inline returnValue setLast(	LogName _name,
									const VariablesGrid& lastValue,
									double time = -INFTY
									);


		/** Adds a record to the log collection.
		 *
		 *	@param[in] record	Record to be added.
		 *
		 *  \return >= 0: index of added record, \n
		 *	        -RET_LOG_COLLECTION_CORRUPTED
		 */
		inline int addLogRecord(	LogRecord& _record
									);


		/** Prints whole record with specified index;
		 *	all items are printed according to the output format settings.
		 *
		 *	@param[in]  idx			Index of record to be printed.
		 *	@param[in]  _mode		Print mode: see documentation of LogPrintMode of details.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS, \n
		 *	        RET_INVALID_ARGUMENTS, \n
		 *	        RET_UNKNOWN_BUG
		 */
		inline returnValue printLogRecord(	std::ostream& _stream,
											int idx,
											LogPrintMode _mode = PRINT_ITEM_BY_ITEM
											) const;


        /** Plots all windows of the plot collection, each one into a new figure.
		 *
		 *	@param[in] _frequency	Frequency determining at which time instants the window is to be plotted.
         *
         *	\return SUCCESSFUL_RETURN
         */
		inline returnValue plot(	PlotFrequency _frequency = PLOT_IN_ANY_CASE
									);

        /** Plots all windows of the plot collection, each one into the 
		 *	corresponding existing figure, if possible.
		 *
		 *	@param[in] _frequency	Frequency determining at which time instants the window is to be plotted.
         *
         *	\return SUCCESSFUL_RETURN
         */
		inline returnValue replot(	PlotFrequency _frequency = PLOT_IN_ANY_CASE
									);


    //
    // DATA MEMBERS:
    //
	protected:
		UserInteraction* userInteraction;				/**< Pointer to top-level user interface.. */
		BooleanType useModuleStandalone;				/**< Flag indicating whether algorithmic module is used stand-alone. */
		
		int outputLoggingIdx;							/**< Index of log record for algorithmic standard output to be optionally used within derived classes. */
};


CLOSE_NAMESPACE_ACADO


#include <acado/user_interaction/algorithmic_base.ipp>


#endif	// ACADO_TOOLKIT_ALGORITHMIC_BASE_HPP


/*
 *	end of file
 */
