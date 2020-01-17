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
 *    \file include/acado/code_generation/export_index_node.hpp
 *    \author Milan Vukov
 */


#ifndef ACADO_TOOLKIT_EXPORT_INDEX_NODE_HPP
#define ACADO_TOOLKIT_EXPORT_INDEX_NODE_HPP

#include <acado/utils/acado_utils.hpp>
#include <acado/code_generation/export_data_internal.hpp>
#include <acado/code_generation/export_index.hpp>
#include <acado/code_generation/export_index_node.hpp>

BEGIN_NAMESPACE_ACADO

enum ExportVariableType
{
	EVT_VARIABLE,
	EVT_VALUE,
	EVT_BINARY_OPERATOR
};

class ExportIndexNode : public ExportDataInternal
{
public:
	ExportIndexNode(	const std::string& _name,
						const std::string& _prefix,
						const int _factor = 1,
						const int _offset = 0)
	:	ExportDataInternal(_name, INT, ACADO_LOCAL, _prefix)
	{
		if ( _factor )
		{
			varType = EVT_VARIABLE;
			value = 0;
			factor = _factor;
			offset = _offset;
			op = ESO_UNDEFINED;
		}
		else
		{
			varType = EVT_VALUE;
			value = _offset;
			factor = 1;
			offset = 0;
			op = ESO_UNDEFINED;
		}
	}

	explicit ExportIndexNode(	const int _value )
	:	ExportDataInternal("defaultIndexName", INT, ACADO_LOCAL, ""),
	 	varType( EVT_VALUE ), value( _value ), factor( 1 ), offset( 0 ), op( ESO_UNDEFINED )
	{}

	ExportIndexNode(	ExportStatementOperator _op,
						const ExportIndex& _arg1,
						const ExportIndex& _arg2
						)
	:	ExportDataInternal("defaultIndexName", INT, ACADO_LOCAL, ""),
	 	varType( EVT_BINARY_OPERATOR ), value( 0 ), factor( 1 ), offset( 0 ),
	 	op( _op ), left( _arg1 ), right( _arg2 )
	{}

	virtual ~ExportIndexNode()
	{}

	virtual ExportIndexNode* clone() const
	{
		return new ExportIndexNode( *this );
	}

	virtual returnValue exportDataDeclaration(	std::ostream& stream,
												const std::string& _realString = "real_t",
												const std::string& _intString = "int",
												int _precision = 16
												) const;

	/// Returns a string containing the value of the index.
	const std::string get( ) const;


	/** Returns the given value of the index (if defined).
	 *
	 *	\return Given value of the index or 0 in case the index is undefined.
	 */
	const int getGivenValue( ) const;


	/** Returns whether the index is set to a given value.
	 *
	 *	\return true  iff index is set to a given value, \n
	 *	        false otherwise
	 */
	virtual bool isGiven( ) const;

	bool isBinary() const
	{
		if (varType == EVT_BINARY_OPERATOR)
			return true;

		return false;
	}

	bool isVariable() const
	{
		if (varType == EVT_VARIABLE)
			return true;

		return false;
	}

	const int getFactor( ) const
	{
		return factor;
	}

	const int getOffset( ) const
	{
		return offset;
	}

private:
	ExportVariableType varType;
	int value;
	int factor;
	int offset;

	int op;
	ExportIndex left;
	ExportIndex right;
};

CLOSE_NAMESPACE_ACADO

#endif // ACADO_TOOLKIT_EXPORT_INDEX_NODE_HPP

