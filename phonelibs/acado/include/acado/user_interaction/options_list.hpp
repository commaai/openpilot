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
 *    \file include/acado/user_interaction/options_list.hpp
 *    \author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */

#ifndef ACADO_TOOLKIT_OPTIONS_LIST_HPP
#define ACADO_TOOLKIT_OPTIONS_LIST_HPP

#include <acado/utils/acado_utils.hpp>

#include <map>
#include <memory>

BEGIN_NAMESPACE_ACADO

/** Summarises all possible types of OptionItems. */
enum OptionsItemType
{
	OIT_UNKNOWN	= -1,	/**< Option item comprising a value of unknown type. */
	OIT_INT,			/**< Option item comprising a value of integer type. */
	OIT_DOUBLE,			/**< Option item comprising a value of double type.  */
	OIT_STRING			/**< Option item comprising a value of std::string type.  */
};

/**
 *	\brief Provides a generic list of options (for internal use).
 *
 *	\ingroup AuxiliaryFunctionality
 *
 *  The class OptionsList provides a generic options list that allows to dynamically 
 *  setup and extend option lists. It is intended for internal use only, as all 
 *	user-functionality is encapsulated within the class Options.
 *
 *	\note Parts of the public functionality of the OptionsList class are tunnelled 
 *	via the Options class into the AlgorithmicBase class to be used in derived classes. 
 *	In case public functionality is modified or added to this class, the Options class
 *	as well as the AlgorithmicBase class have to be adapted accordingly.
 *
 *	\author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */
class OptionsList
{
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:



		/** Default constructor.
		 */
		OptionsList( );

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		OptionsList(	const OptionsList& rhs
						);

		/** Destructor.
		 */
		~OptionsList( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] rhs	Right-hand side object.
		 */
		OptionsList& operator=(	const OptionsList& rhs
								);

		/** Add an option item with a given value.
		 *
		 *  @tparam    T		Option data type.
		 *	@param[in] name		Name of new option item.
		 *	@param[in] value	Default value of new option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_ALREADY_EXISTS, \n
		 *          RET_OPTIONS_LIST_CORRUPTED
		 */
		template< typename T >
		inline returnValue add(	OptionsName name,
								const T& value );

		/** Returns value of an existing option item of integer type.
		 *
		 *  @tparam     T		Option data type.
		 *	@param[in]  name	Name of option item.
		 *	@param[out] value	Value of option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_DOESNT_EXISTS
		 */
		template< typename T >
		inline returnValue get(	OptionsName name,
								T& value
								) const;

		/** Sets value of an existing option item of integer type to a given value.
		 *
		 * @tparam     T		Option data type.
		 *	@param[in] name		Name of option item.
		 *	@param[in] value	New value of option.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *          RET_OPTION_DOESNT_EXISTS, \n
		 *          RET_OPTIONS_LIST_CORRUPTED
		 */
		template< typename T >
		inline returnValue set(	OptionsName name,
								const T& value
								);

		/** Returns total number of option items in list.
		 *
		 *  \return Total number of options in list
		 */
		inline uint getNumber( ) const;


		/** Determines whether a given option exists or not.
		 *
		 *	@param[in] name		Name of option item.
		 *	@param[in] type		Internal type of option item.
		 *
		 *  \return BT_TRUE  iff option item exists, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasOption(	OptionsName name,
										OptionsItemType type
										) const;


		/** Determines whether options have been modified.
		 *
		 *	\return BT_TRUE  iff options have been modified, \n
		 *	        BT_FALSE otherwise 
		 */
		inline BooleanType haveOptionsChanged( ) const;

		/** Declares all options to be unchanged.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue declareOptionsUnchanged( );


		/** Prints a list of all available options.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue printOptionsList( ) const;

    //
    // DATA MEMBERS:
    //
	private:

		/** Flag indicating whether the value of at least one option item has been changed. */
		BooleanType optionsHaveChanged;

		/** Value base type */
		struct OptionValueBase
		{
			virtual void print( std::ostream& stream ) = 0;
		};

		/** Value type */
		template< typename T >
		struct OptionValue : public OptionValueBase
		{
			OptionValue( const T& _value )
				: value( _value )
			{}

			virtual void print( std::ostream& stream )
			{
				stream << value;
			}

			T value;
		};

		/** Type for options items. */
		typedef std::map<std::pair<OptionsName, OptionsItemType>, std::shared_ptr< OptionValueBase > > OptionItems;
		/** Option items. */
		OptionItems items;
		/** A helper function to determine type of an option. */
		template< typename T >
		inline OptionsItemType getType() const;
};

template< typename T >
inline OptionsItemType OptionsList::getType() const
{ return OIT_UNKNOWN; }

template<>
inline OptionsItemType OptionsList::getType< int >() const
{ return OIT_INT; }

template<>
inline OptionsItemType OptionsList::getType< double >() const
{ return OIT_DOUBLE; }

template<>
inline OptionsItemType OptionsList::getType< std::string >() const
{ return OIT_STRING; }

template< typename T >
inline returnValue OptionsList::add(	OptionsName name,
										const T& value
										)
{
	if (getType< T >() == OIT_UNKNOWN)
		return ACADOERROR( RET_NOT_IMPLEMENTED_YET );

	items[ std::make_pair(name, getType< T >()) ] =
			std::shared_ptr< OptionValue< T > > (new OptionValue< T >( value ));

	return SUCCESSFUL_RETURN;
}

template< typename T >
inline returnValue OptionsList::get(	OptionsName name,
										T& value
										) const
{
	if (getType< T >() == OIT_UNKNOWN)
		return ACADOERROR( RET_NOT_IMPLEMENTED_YET );

	OptionItems::const_iterator it = items.find(std::make_pair(name, getType< T >()));
	if (it != items.end())
	{
		std::shared_ptr< OptionValue< T > > ptr;
		ptr = std::static_pointer_cast< OptionValue< T > >(it->second);
		value = ptr->value;
		return SUCCESSFUL_RETURN;
	}

	return ACADOERROR( RET_OPTION_DOESNT_EXIST );
}

template< typename T >
inline returnValue OptionsList::set(	OptionsName name,
										const T& value
										)
{
	if (getType< T >() == OIT_UNKNOWN)
		return ACADOERROR( RET_NOT_IMPLEMENTED_YET );

	OptionItems::const_iterator it = items.find(std::make_pair(name, getType< T >()));
	if (it != items.end())
	{
		items[ std::make_pair(name, getType< T >()) ] =
				std::shared_ptr< OptionValue< T > > (new OptionValue< T >( value ));

		optionsHaveChanged = BT_TRUE;

		return SUCCESSFUL_RETURN;
	}

	return ACADOERROR( RET_OPTION_DOESNT_EXIST );
}

CLOSE_NAMESPACE_ACADO

#include <acado/user_interaction/options_list.ipp>

#endif	// ACADO_TOOLKIT_OPTIONS_LIST_HPP

/*
 *	end of file
 */
