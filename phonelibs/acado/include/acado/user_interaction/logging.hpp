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
 *    \file include/acado/user_interaction/logging.hpp
 *    \author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */


#ifndef ACADO_TOOLKIT_LOGGING_HPP
#define ACADO_TOOLKIT_LOGGING_HPP

#include <acado/utils/acado_utils.hpp>
#include <acado/user_interaction/log_record.hpp>

BEGIN_NAMESPACE_ACADO

static LogRecord emptyLogRecord;

/**
 *	\brief Provides a generic way to store algorithmic information during runtime.
 *
 *	\ingroup AuxiliaryFunctionality
 *	
 *  The class Logging provides a generic way to store algorithmic information 
 *	during runtime. This class is part of the UserInterface class, i.e. all classes 
 *	that are intended to interact with the user inherit the public functionality 
 *	of the Logging class.
 *
 *	\author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */
class Logging
{
	friend class AlgorithmicBase;
	
	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Default constructor. */
		Logging( );

		/** Destructor. */
		virtual ~Logging( );

		/** Adds a record to the log collection.
		 *
		 *	@param[in] record	Record to be added.
		 *
		 *	\note This function is doing the same as the corresponding 
		 *	      addLogRecord member function and is introduced for syntax reasons only.
		 *
		 *  \return >= 0: index of added record, \n
		 *	        -RET_LOG_COLLECTION_CORRUPTED 
		 */
		int operator<<(	LogRecord& record
						);

		/** Adds a record to the log collection.
		 *
		 *	@param[in] record	Record to be added.
		 *
		 *  \return >= 0: index of added record, \n
		 *	        -RET_LOG_COLLECTION_CORRUPTED 
		 */
		int addLogRecord(	LogRecord& record
							);

		/** Returns the record with certain index from the log collection. 
		 *	This index is not provided when calling the function, but 
		 *	rather obtained by using the alias index of the record. If the
		 *	record is no alias record, the error RET_INDEX_OUT_OF_BOUNDS is thrown.
		 *
		 *	@param[out] _record		Desired record.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INDEX_OUT_OF_BOUNDS 
		 */
		returnValue getLogRecord(	LogRecord& _record
									) const;

		/** Updates all items with the record given as argument. In doing so,
		 *	it is checked for each item whether it is contained within one of
		 *	of the records of the collection; and if so, the numerical values
		 *	are copied into the argument record.
		 *
		 *	@param[in,out]  _record		Record to be updated
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue updateLogRecord(	LogRecord& _record
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

		/** Sets all numerical values at all time instants of the item
		 *	with given name.
		 *
		 *	@param[in]  _name	Internal name of item.
		 *	@param[in]  values	All numerical values at all time instants of given item.
		 *
		 *	\note All public setAll member functions make use of the <em>protected</em> setAll function.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_RECORD_CORRUPTED
		 */
		inline returnValue setAll(	LogName _name,
									const MatrixVariablesGrid& values
									);


		/** Sets numerical value at last time instant of the item
		 *	with given name.
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[in]  lastValue	Numerical value at last time instant of given item.
		 *	@param[in]  time		Time label of the instant.
		 *
		 *	\note All public setLast member functions make use of the <em>protected</em> setLast function.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST
		 */
		inline returnValue setLast(	LogName _name,
									const DMatrix& value,
									double time = -INFTY
									);

		/** Sets numerical value at last time instant of the item
		 *	with given name.
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[in]  lastValue	Numerical value at last time instant of given item.
		 *	@param[in]  time		Time label of the instant.
		 *
		 *	\note All public setLast member functions make use of the <em>protected</em> setLast function.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST
		 */
		inline returnValue setLast(	LogName _name,
									VariablesGrid& value,
									double time = -INFTY
									);

		/** Returns number of records contained in the log collection.
		 *
		 *  \return Number of records
		 */
		uint getNumLogRecords( ) const;


		/** Prints information on all records and their items on screen.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue printLoggingInfo( ) const;

		returnValue printNumDoubles( ) const;

    //
    // PROTECTED MEMBER FUNCTIONS:
    //
    protected:

		/** Prototype member function for setting-up the logging information
		 *	at initialization of derived classes.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		virtual returnValue setupLogging( );

    //
    // DATA MEMBERS:
    //
	protected:
		/** Log collection containing a singly-linked list of log records. */
		std::vector< LogRecord > logCollection;
		/** Index of a certain log record to be optionally used within derived classes. */
		int logIdx;
};

CLOSE_NAMESPACE_ACADO

#include <acado/user_interaction/logging.ipp>

#endif	// ACADO_TOOLKIT_LOGGING_HPP

/*
 *	end of file
 */
