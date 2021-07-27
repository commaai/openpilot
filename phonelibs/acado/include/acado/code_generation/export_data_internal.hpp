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
#ifndef ACADO_TOOLKIT_EXPORT_DATA_INTERNAL_HPP
#define ACADO_TOOLKIT_EXPORT_DATA_INTERNAL_HPP

#include <casadi/symbolic/shared_object.hpp>
#include <acado/utils/acado_utils.hpp>

BEGIN_NAMESPACE_ACADO

class ExportDataInternal : public CasADi::SharedObjectNode
{
public:

	/** Default constructor which optionally takes name and type string
	 *	of the data object.
	 *
	 *	@param[in] _name			Name of the data object.
	 *	@param[in] _type			Data type of the data object.
	 *	@param[in] _dataStruct		Global data struct to which the data object belongs to (if any).
	 *	@param[in] _prefix			Optional prefix that will be put in front of the name.
	 */
	explicit ExportDataInternal(	const std::string& _name = std::string(),
									ExportType _type = REAL,
									ExportStruct _dataStruct = ACADO_LOCAL,
									const std::string& _prefix = std::string()
									);

	/** Destructor.
	 */
	virtual ~ExportDataInternal( );

	virtual ExportDataInternal* clone() const = 0;

	/** Sets the name of the data object.
	 *
	 *	@param[in] _name			New name of the data object.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	returnValue	setName(	const std::string& _name
							);

	/** Sets the data type of the data object.
	 *
	 *	@param[in] _type			New data type of the data object.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	returnValue	setType(	ExportType _type
							);

	/** Sets the global data struct to which the data object belongs to.
	 *
	 *	@param[in] _dataStruct		New global data struct to which the data object belongs to.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	returnValue	setDataStruct(	ExportStruct _dataStruct
								);

	/** Sets the prefix which is placed before the structure name.
	 *
	 *  @param[in] _prefix Prefix name.
	 *
	 *  \return SUCCESSFUL_RETURN
	 */
	returnValue setPrefix(	const std::string& _prefix
							);
	/** Returns the name of the data object.
	 *
	 *	\return Name of the data object
	 */
	std::string getName( ) const;

	/** Returns the data type of the data object.
	 *
	 *	\return Data type of the data object
	 */
	ExportType getType( ) const;

	/** Returns a string containing the data type of the data object.
	 *
	 *	@param[in] _realString		std::string to be used to declare real variables.
	 *	@param[in] _intString		std::string to be used to declare integer variables.
	 *
	 *	\return std::string containing the data type of the data object.
	 */
	std::string getTypeString(	const std::string& _realString = "real_t",
								const std::string& _intString = "int"
								) const;

	/** Returns the global data struct to which the data object belongs to.
	 *
	 *	\return Global data struct to which the data object belongs to
	 */
	ExportStruct getDataStruct( ) const;

	/** Returns a string containing the global data struct to which the data object belongs to.
	 *
	 *	\return String containing the global data struct to which the data object belongs to.
	 */
	std::string getDataStructString( ) const;

	/** Returns a string which contains a prefix name.
	 *
	 *  \return Prefix name
	 */
	std::string getPrefix( ) const;

	/** Returns the full name of the data object including the possible prefix
	 *	of the global data struct.
	 *
	 *	\return Full name of the data object
	 */
	std::string getFullName( ) const;


	/** Exports declaration of the index variable. Its appearance can
	 *  can be adjusted by various options.
	 *
	 *	@param[in] file				Name of file to be used to export function.
	 *	@param[in] _realString		std::string to be used to declare real variables.
	 *	@param[in] _intString		std::string to be used to declare integer variables.
	 *	@param[in] _precision		Number of digits to be used for exporting real values.
	 *
	 *	\return SUCCESSFUL_RETURN
	 */
	virtual returnValue exportDataDeclaration(	std::ostream& stream,
												const std::string& _realString = "real_t",
												const std::string& _intString = "int",
												int _precision = 16
												) const = 0;

	/** Returns whether the index is set to a given value.
	 *
	 *	\return true  iff index is set to a given value, \n
	 *	        false otherwise
	 */
	virtual bool isGiven( ) const = 0;

	virtual returnValue setDoc( const std::string& _doc );
	virtual std::string getDoc( ) const;

	//
	// PROTECTED MEMBER FUNCTIONS:
	//
protected:

	returnValue setFullName( void );

protected:

	/** Name of the data object. */
	std::string name;

	/** Data type of the data object. */
	ExportType type;

	/** Prefix, which is added before the structure name*/
	std::string prefix;

	/** Global data struct to which the data object belongs to (if any). */
	ExportStruct dataStruct;

	/** Full name of the data object including the possible prefix of the global data struct. */
	std::string fullName;

	/** Description of the variable */
	std::string description;
    
    
public:
    static std::string fcnPrefix;
};

CLOSE_NAMESPACE_ACADO

#endif // ACADO_TOOLKIT_EXPORT_DATA_INTERNAL_HPP
