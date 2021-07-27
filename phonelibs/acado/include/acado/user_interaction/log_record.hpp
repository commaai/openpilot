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
 *    \file include/acado/user_interaction/log_record.hpp
 *    \author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */

#ifndef ACADO_TOOLKIT_LOG_RECORD_HPP
#define ACADO_TOOLKIT_LOG_RECORD_HPP

#include <acado/utils/acado_utils.hpp>
#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/symbolic_expression/symbolic_expression.hpp>
#include <acado/variables_grid/variables_grid.hpp>

#include <map>
#include <iterator>

BEGIN_NAMESPACE_ACADO

class Logging;

/**
 *	\brief Allows to setup and store user-specified log records of algorithmic information.
 *
 *	\ingroup UserDataStructures
 *	
 *  The class LogRecord allows to setup and store user-specified log records of algorithmic 
 *	information consisting of numerical values during runtime.
 *
 *	A log record comprises arbitrarily many LogRecordItems stored in a simple 
 *	singly-linked list. Within these items, the actual numerical values of the 
 *	algorithmic to be logged as well as settings defining their output format is
 *	stored. Several commonly-used output formats are pre-defined within so-called 
 *	PrintSchemes.
 *
 *	Additionally, a log record stores two important settings: (i) the LogFrequency
 *	defining whether the information is stored at each iteration or only at the 
 *	start or end, respectively; (ii) the file, e.g. a file or the screen,
 *	to which the whole log record including all its items can be printed.
 *
 *	Finally, it is interesting to know that LogRecords can be setup by the user and
 *	flushed to UserInterface classes. Internally, LogRecords are stored as basic 
 *	singly-linked within a LogCollection.
 *
 *	\author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */
class LogRecord
{
	friend class Logging;

	//
	// PUBLIC MEMBER FUNCTIONS:
	//
	public:

		/** Constructor which takes information in the log frequency and 
		 *	the general output file. 
		 *
		 *	@param[in] _frequency	Frequency determining at which time instants the numerical values are to be stored.
		 *	@param[in] _printScheme	Print scheme defining the output format of the information.
		 */
		LogRecord(	LogFrequency _frequency = LOG_AT_EACH_ITERATION,
					PrintScheme _printScheme = PS_DEFAULT
					);

		/** Destructor. */
		~LogRecord( );

		/** Adds an item of given name to the singly-linked list.
		 *
		 *	@param[in] _name	Internal name of item to be added.
		 *
		 *	\note This function is doing the same as the corresponding 
		 *	      addItem member function and is introduced for syntax reasons only.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue operator<<(	LogName _name
								);

		/** Adds an item of given internal name to the singly-linked list.
		 *
		 *	@param[in] _name	Internal name of item to be added.
		 *
		 *	\note This function is doing the same as the corresponding 
		 *	      addItem member function and is introduced for syntax reasons only.
		 *
		 *  \return SUCCESSFUL_RETURN 
		 */
		returnValue operator<<(	const Expression& _name
								);

		/** Adds an item of given internal name to the singly-linked list.
		 *	In addition, label of the item can be specified.
		 *
		 *	@param[in] _name	Internal name of item to be added.
		 *	@param[in] _label	Label of item to be added.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_RECORD_CORRUPTED 
		 */
		returnValue addItem(	LogName _name,
								const char* const _label = DEFAULT_LABEL
								);

		/** Adds an item of given internal name to the singly-linked list.
		 *	In addition, label of the item can be specified.
		 *
		 *	@param[in] _name	Internal name of item to be added.
		 *	@param[in] _label	Label of item to be added.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_RECORD_CORRUPTED 
		 */
		returnValue addItem(	const Expression& _name,
								const char* const _label = DEFAULT_LABEL
								);

		/** Gets all numerical values at all time instants of the item
		 *	with given name.
		 *
		 *	@param[in]  _name	Internal name of item.
		 *	@param[out] values	All numerical values at all time instants of given item.
		 *
		 *	\note All public getAll member functions make use of the <em>protected</em> getAll function.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST 
		 */
		inline returnValue getAll(	LogName _name,
									MatrixVariablesGrid& values
									) const;

		/** Gets all numerical values at all time instants of the item
		 *	with given name.
		 *
		 *	@param[in]  _name	Internal name of item.
		 *	@param[out] values	All numerical values at all time instants of given item.
		 *
		 *	\note All public getAll member functions make use of the <em>protected</em> getAll function.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST 
		 */
		inline returnValue getAll(	const Expression& _name,
									MatrixVariablesGrid& values
									) const;


		/** Gets numerical value at first time instant of the item
		 *	with given name.
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[out] firstValue	Numerical value at first time instant of given item.
		 *
		 *	\note All public getFirst member functions make use of the <em>protected</em> getFirst function.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST 
		 */
		inline returnValue getFirst(	LogName _name,
										DMatrix& firstValue
										) const;

		/** Gets numerical value at first time instant of the item
		 *	with given name.
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[out] firstValue	Numerical value at first time instant of given item.
		 *
		 *	\note All public getFirst member functions make use of the <em>protected</em> getFirst function.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST 
		 */
		inline returnValue getFirst(	const Expression& _name,
										DMatrix& firstValue
										) const;

		/** Gets numerical value at first time instant of the item
		 *	with given name (converts internally used DMatrix into VariablesGrid).
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[out] firstValue	Numerical value at first time instant of given item.
		 *
		 *	\note All public getFirst member functions make use of the <em>protected</em> getFirst function.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST 
		 */
		inline returnValue getFirst(	LogName _name,
										VariablesGrid& firstValue
										) const;

		/** Gets numerical value at first time instant of the item
		 *	with given name (converts internally used DMatrix into VariablesGrid).
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[out] firstValue	Numerical value at first time instant of given item.
		 *
		 *	\note All public getFirst member functions make use of the <em>protected</em> getFirst function.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST 
		 */
		inline returnValue getFirst(	const Expression& _name,
										VariablesGrid& firstValue
										) const;


		/** Gets numerical value at last time instant of the item
		 *	with given name.
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[out] lastValue	Numerical value at last time instant of given item.
		 *
		 *	\note All public getLast member functions make use of the <em>protected</em> getLast function.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST 
		 */
		inline returnValue getLast(	LogName _name,
									DMatrix& lastValue
									) const;

		/** Gets numerical value at last time instant of the item
		 *	with given name.
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[out] lastValue	Numerical value at last time instant of given item.
		 *
		 *	\note All public getLast member functions make use of the <em>protected</em> getLast function.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST 
		 */
		inline returnValue getLast(	const Expression& _name,
									DMatrix& lastValue
									) const;

		/** Gets numerical value at last time instant of the item
		 *	with given name (converts internally used DMatrix into VariablesGrid).
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[out] lastValue	Numerical value at last time instant of given item.
		 *
		 *	\note All public getLast member functions make use of the <em>protected</em> getLast function.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST 
		 */
		inline returnValue getLast(	LogName _name,
									VariablesGrid& lastValue
									) const;

		/** Gets numerical value at last time instant of the item
		 *	with given name (converts internally used DMatrix into VariablesGrid).
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[out] lastValue	Numerical value at last time instant of given item.
		 *
		 *	\note All public getLast member functions make use of the <em>protected</em> getLast function.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST 
		 */
		inline returnValue getLast(	const Expression& _name,
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
		inline returnValue setAll(	const Expression& _name,
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
		inline returnValue setLast(	const Expression& _name,
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
		inline returnValue setLast(	const Expression& _name,
									VariablesGrid& value,
									double time = -INFTY
									);

		/** Prints whole record into a stream;
		 *	all items are printed according to the output format settings.
		 *
		 *	@param[in] _stream      Stream to print the record.
		 *	@param[in] _mode		Print mode: see documentation of LogPrintMode of details.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_INVALID_ARGUMENTS, \n
		 *	        RET_UNKNOWN_BUG
		 */
		returnValue print(	std::ostream& _stream = std::cout,
							LogPrintMode _mode = PRINT_ITEM_BY_ITEM
							) const;

		/** Prints information on the record and its items on screen.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue printInfo( ) const;

		/** Returns maximum number of matrices per item. Remember that the
		 *	numerical values of each item is stored in a MatrixVariablesGrid
		 *	and that the number of matrices might be different from one item
		 *	to the other.
		 *
		 *  \return Maximum number of matrices per item
		 */
		uint getMaxNumMatrices( ) const;

		/** Returns number of items contained in the record.
		 *
		 *  \return Number of items
		 */
		inline uint getNumItems( ) const;

		/** Returns whether the record is empty (i.e. contains no items) or not.
		 *
		 *  \return BT_TRUE  iff record is empty, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType isEmpty( ) const;

		/** Returns current log frequency determining at which time instants the 
		 *	numerical values are to be stored.
		 *
		 *  \return Current log frequency
		 */
		inline LogFrequency getLogFrequency( ) const;

		/** Returns current print scheme defining the output format of the information.
		 *
		 *  \return Current print scheme
		 */
		inline PrintScheme getPrintScheme( ) const;

		/** Sets the log frequency determining at which time instants the 
		 *	numerical values are to be stored.
		 *
		 *	@param[in]  _frequency	New log frequency
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue setLogFrequency(	LogFrequency _frequency
												);

		/** Sets the print scheme defining the output format of the information.
		 *
		 *	@param[in]  _printScheme	New print scheme
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue setPrintScheme(	PrintScheme _printScheme
											);

		/** Returns whether an (possibly empty) item with given internal name 
		 *	exists or not.
		 *
		 *	@param[in] _name	Internal name of item.
		 *
		 *  \return BT_TRUE  iff item exists, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasItem(	LogName _name
									) const;

		/** Returns whether an (possibly empty) item with given internal name 
		 *	exists or not.
		 *
		 *	@param[in] _name	Internal name of item.
		 *
		 *  \return BT_TRUE  iff item exists, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasItem(	const Expression& _name
									) const;

		/** Returns whether a non-empty item with given internal name exists or not.
		 *
		 *	@param[in] _name	Internal name of item.
		 *
		 *  \return BT_TRUE  iff non-empty item exists, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasNonEmptyItem(	LogName _name
											) const;

		/** Returns whether a non-empty item with given internal name exists or not.
		 *
		 *	@param[in] _name	Internal name of item.
		 *
		 *  \return BT_TRUE  iff non-empty item exists, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasNonEmptyItem(	const Expression& _name
											) const;


		inline uint getNumDoubles( ) const;

		/** Gets all numerical values at all time instants of the item
		 *	with given internal name and internal type.
		 *
		 *	@param[in]  _name	Internal name of item.
		 *	@param[in]  _type	Internal type of item.
		 *	@param[out] values	All numerical values at all time instants of given item.
		 *
		 *	\note All <em>public</em> getAll member functions make use of this protected function.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST 
		 */
		returnValue getAll(	uint _name,
							LogRecordItemType _type,
							MatrixVariablesGrid& values
							) const;

		/** Gets numerical value at first time instant of the item
		 *	with given internal name and internal type.
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[in]  _type		Internal type of item.
		 *	@param[out] firstValue	Numerical value at first time instant of given item.
		 *
		 *	\note All <em>public</em> getFirst member functions make use of this protected function.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST 
		 */
		returnValue getFirst(	uint _name,
								LogRecordItemType _type,
								DMatrix& firstValue
								) const;

		/** Gets numerical value at last time instant of the item
		 *	with given internal name and internal type.
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[in]  _type		Internal type of item.
		 *	@param[out] lastValue	Numerical value at last time instant of given item.
		 *
		 *	\note All <em>public</em> getLast member functions make use of this protected function.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST 
		 */
		returnValue getLast(	uint _name,
								LogRecordItemType _type,
								DMatrix& lastValue
								) const;

		/** Sets all numerical values at all time instants of the item
		 *	with given internal name and internal type.
		 *
		 *	@param[in]  _name	Internal name of item.
		 *	@param[in]  _type	Internal type of item.
		 *	@param[in]  values	All numerical values at all time instants of given item.
		 *
		 *	\note All <em>public</em> setAll member functions make use of this protected function.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_RECORD_CORRUPTED 
		 */
		returnValue setAll(	uint _name,
							LogRecordItemType _type,
							const MatrixVariablesGrid& values
							);

		/** Sets numerical value at last time instant of the item
		 *	with given internal name and internal type.
		 *
		 *	@param[in]  _name		Internal name of item.
		 *	@param[in]  _type		Internal type of item.
		 *	@param[in]  lastValue	Numerical value at last time instant of given item.
		 *	@param[in]  time		Time label of the instant.
		 *
		 *	\note All <em>public</em> setLast member functions make use of this protected function.
		 *
		 *  \return SUCCESSFUL_RETURN, \n
		 *	        RET_LOG_ENTRY_DOESNT_EXIST 
		 */
		returnValue setLast(	uint _name,
								LogRecordItemType _type,
								const DMatrix& value,
								double time = -INFTY
								);


		/** Returns whether an (possibly empty) item with given internal name 
		 *	and internal type exists or not.
		 *
		 *	@param[in] _name	Internal name of item.
		 *	@param[in] _type	Internal type of item.
		 *
		 *  \return BT_TRUE  iff item exists, \n
		 *	        BT_FALSE otherwise
		 */
		inline BooleanType hasItem(	uint _name,
									LogRecordItemType _type
									) const;

		/** Enables write protection of item with given name. As long as
		 *  write protection is enabled, numerical values of this item cannot be modified.
		 *
		 *  @param[in]  _name                Internal name of item.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue enableWriteProtection(LogName _name);

		/** Enables write protection of item with given name. As long as
		 *  write protection is enabled, numerical values of this item cannot be modified.
	 	 *
	 	 *  @param[in]  _name                Internal name of item.
	 	 *
	 	 *  \return SUCCESSFUL_RETURN
	 	 */
		inline returnValue enableWriteProtection(const Expression& _name);

		/** Disables write protection of item with given name. As long as
		 *  write protection is enabled, numerical values of this item cannot be modified.
		 *
		 *  @param[in]  _name                Internal name of item.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue disableWriteProtection(LogName _name);

		/** Disables write protection of item with given name. As long as
		 *  write protection is enabled, numerical values of this item cannot be modified.
		 *
		 *  @param[in]  _name                Internal name of item.
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		inline returnValue disableWriteProtection(const Expression& _name);

		/** Update an existing record. */
		returnValue updateLogRecord(	LogRecord& _record
										) const;

    //
    // DATA MEMBERS:
    //
	protected:
		/** Alias index of record (-1 = no alias record). */
		int aliasIdx;
		/** Frequency determining at which time instants the numerical values are to be stored. */
		LogFrequency frequency;
		/** Print scheme defining the output format of the information. */
		PrintScheme printScheme;

		/** Log record item data. */
		struct LogRecordData
		{
			LogRecordData()
				: label( DEFAULT_LABEL ), writeProtection( false )
			{}

			LogRecordData(	const std::string& _label
							)
				: label( _label ), writeProtection( false )
			{}

			LogRecordData(	const MatrixVariablesGrid& _values,
							const std::string& _label,
							bool _writeProtection
							)
				: values( _values ), label( _label ), writeProtection( _writeProtection )
			{}

			MatrixVariablesGrid values;
			std::string label;
			bool writeProtection;
		};

		/** Type definition for Log record items. */
		typedef std::map<std::pair<int, LogRecordItemType>, LogRecordData> LogRecordItems;
		/** Log record items. */
		LogRecordItems items;
};

CLOSE_NAMESPACE_ACADO

#include <acado/user_interaction/log_record.ipp>

#endif	// ACADO_TOOLKIT_LOG_RECORD_HPP

/*
 *	end of file
 */
