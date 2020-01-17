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
 *    \file include/acado/user_interaction/options.hpp
 *    \author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */


#ifndef ACADO_TOOLKIT_OPTIONS_HPP
#define ACADO_TOOLKIT_OPTIONS_HPP

#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/options_list.hpp>

BEGIN_NAMESPACE_ACADO

/**
 *	\brief Provides a generic way to set and pass user-specified options.
 *
 *	\ingroup AuxiliaryFunctionality
 *	
 *  The class Options provides a generic way to set and pass user-specified
 *	options. This class is part of the UserInterface class, i.e. all classes 
 *	that are intended to interact with the user inherit the public functionality 
 *	of the Options class.
 *
 *	The class Options holds an array of OptionsLists, where only one OptionsList 
 *	is allocated by default. This list contains all available options and their
 *	respective values. The array can be extended by calling the addOptionsList()
 *	member function. This allows to handle more than one options list, that can
 *	be accessed via its index. This feature is, e.g., intended to manage a 
 *	seperate options list for each stage of a multi-stage optimal control problem.
 *
 *	\note Parts of the functionality of the Options class are tunnelled into the 
 *	AlgorithmicBase class to be used in derived developer classes. In case this 
 *	functionality is modified or new functionality is added to this class, the 
 *	AlgorithmicBase class has to be adapted accordingly.
 *
 *	\author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */
class Options
{
	friend class AlgorithmicBase;
	
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor.
		 */
		Options( );

		/** Constructor which takes an initialization for first option list.
		 *
		 *	@param[in] _optionsList		Initialization for first option list
		 */
		Options(	const OptionsList& _optionsList
					);

		/** Destructor.
		 */
		virtual ~Options( );

		/** Adds an additional OptionsList to internal array.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue addOptionsList( );


		/** Returns value of an existing option item of integer type.
		 *
		 *	@param[in]  name	Name of option item.
		 *	@param[out] value	Value of option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_DOESNT_EXISTS
		 */
		returnValue get(	OptionsName name,
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
		returnValue get(	OptionsName name,
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
		returnValue get(	OptionsName name,
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
		returnValue get(	uint idx,
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
		returnValue get(	uint idx,
							OptionsName name,
							double& value
							) const;

		/** Returns value of an existing option item of string type
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
		returnValue get(	uint idx,
							OptionsName name,
							std::string& value
							) const;

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

		/** Sets value of an existing option item of string type to a given value.
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

		/** Sets value of an existing option item of string type
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
							const std::string& value
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


		/** Returns total number of option lists.
		 *
		 *  \return Total number of option lists
		 */
		uint getNumOptionsLists( ) const;


		/** Prints a list of all available options of all option lists.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue printOptionsList( ) const;

		/** Prints a list of all available options of option list with given index.
		 *
		 *	@param[in] idx		Index of option list.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue printOptionsList(	uint idx
										) const;



    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

		/** Prototype member function for setting-up the option list(s)
		 *	at initialization of derived classes.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue setupOptions( );


		/** Clears all option lists from array.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue clearOptionsList( );


		/** Determines whether options of at least one option list have been modified.
		 *
		 *	\return BT_TRUE  iff options have been modified, \n
		 *	        BT_FALSE otherwise 
		 */
		BooleanType haveOptionsChanged( ) const;

		/** Determines whether options of option list with given index have been modified.
		 *
		 *	@param[in] idx		Index of option list.
		 *
		 *	\return BT_TRUE  iff options have been modified, \n
		 *	        BT_FALSE otherwise 
		 */
		BooleanType haveOptionsChanged(	uint idx
										) const;


		/** Declares all options of all option lists to be unchanged.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue declareOptionsUnchanged( );
		
		/** Declares all options of option list with given index to be unchanged.
		 *
		 *	@param[in] idx		Index of option list.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue declareOptionsUnchanged(	uint idx
												);


		/** Add an option item with a given integer default value to the all option lists.
		 *
		 *	@param[in] name		Name of new option item.
		 *	@param[in] value	Default value of new option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_ALREADY_EXISTS, \n
		 *          RET_OPTIONS_LIST_CORRUPTED
		 */
		returnValue addOption(	OptionsName name,
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
		returnValue addOption(	OptionsName name,
								double value
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
		returnValue addOption(	OptionsName name,
								const std::string& value
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
		returnValue addOption(	uint idx,
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
		returnValue addOption(	uint idx,
								OptionsName name,
								double value
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
		returnValue addOption(	uint idx,
								OptionsName name,
								const std::string& value
								);

    //
    // DATA MEMBERS:
    //
	protected:

		/** A list consisting of OptionsLists. */
		std::vector< OptionsList > lists;
};

CLOSE_NAMESPACE_ACADO

#endif	// ACADO_TOOLKIT_OPTIONS_HPP

/*
 *	end of file
 */
