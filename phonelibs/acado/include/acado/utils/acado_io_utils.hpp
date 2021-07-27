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
 *    \file include/acado/utils/acado_io_utils.hpp
 *    \author Boris Houska, Hans Joachim Ferreau, Milan Vukov
 *    \date 2008 - 2013
 *
 *    This file declares several utility functions that are useful for low-level
 *    file reading and printing.
 *
 */

#ifndef ACADO_TOOLKIT_ACADO_IO_UTILS_HPP
#define ACADO_TOOLKIT_ACADO_IO_UTILS_HPP

#include <cstdlib>
#include <cstring>

#include <string>
#include <sstream>

#include <fstream>
#include <vector>
#include <iterator>

#include <acado/utils/acado_message_handling.hpp>

BEGIN_NAMESPACE_ACADO

/** Global definition of the default line separation symbol. This constant should
 *  be used for all text-file interaction.
 */
const char LINE_SEPARATOR = '\n';

/** Global definition of the default text separation symbol. This constant
 *  should be used for all text-files interactions.
 */
const char TEXT_SEPARATOR = ' ';

/** Global definition of the default text separation symbol. This constant
 *  should be used for all text-files interactions.
 */
const char NOT_A_NUMBER[3] = { 'n', 'a', 'n' };

const char  DEFAULT_LABEL         [1]   = { '\0'                              };
const char  DEFAULT_START_STRING  [3]   = { '[' , '\t', '\0'                  };
const char  DEFAULT_END_STRING    [4]   = { '\t', ']' , '\n', '\0'            };
const uint  DEFAULT_WIDTH               = 22;
const uint  DEFAULT_PRECISION           = 16;
const char  DEFAULT_COL_SEPARATOR [2]   = { '\t', '\0'                        };
const char  DEFAULT_ROW_SEPARATOR [6]   = { '\t', ']' , '\n', '[', '\t', '\0' };

/** Get global string definitions. */
returnValue getGlobalStringDefinitions(	PrintScheme _printScheme,
										char** _startString,
										char** _endString,
										uint& _width,
										uint& _precision,
										char** _colSeparator,
										char** _rowSeparator
										);

/** A function for file copying. */
returnValue acadoCopyFile(	const std::string& source,
							const std::string& destination,
							const std::string& commentString,
							bool printCodegenNotice = false
							);

/** A function for copying of template files. */
returnValue acadoCopyTemplateFile(	const std::string& source,
									const std::string& destination,
									const std::string& commentString,
									bool printCodegenNotice = false
									);

/** A function to create a folder. */
returnValue acadoCreateFolder(const std::string& name);

/** Prints ACADO Toolkit's copyright notice.
 *
 *  \return SUCCESSFUL_RETURN */
returnValue acadoPrintCopyrightNotice( );

/** Prints ACADO Toolkit's copyright notice for sub-packages.
 *
 *  \return SUCCESSFUL_RETURN */
returnValue acadoPrintCopyrightNotice(	const std::string& subpackage
										);

/** Prints ACADO Toolkit's copyright notice for auto generated code.
 *
 *  \return SUCCESSFUL_RETURN */
returnValue acadoPrintAutoGenerationNotice(	std::ofstream& stream,
											const std::string& commentString
											);

/** Returns the current system time.
 *
 *  \return current system time
 */
double acadoGetTime( );

CLOSE_NAMESPACE_ACADO

namespace std
{

/** Print vector data to output stream. */
template< class T >
ostream& operator<<(	ostream& stream,
						vector<T>& array
						)
{
	copy(array.begin(), array.end(), ostream_iterator< T >(stream, "\t"));
	return stream;
}

/** Print matrix data to output stream. */
template< class T >
ostream& operator<<(	ostream& stream,
						vector< vector<T> >& array
						)
{
	typename vector< vector< T > >::iterator it;
	for (it = array.begin(); it != array.end(); ++it)
	{
		copy(it->begin(), it->end(), ostream_iterator< T >(stream, "\t"));
		stream << endl;
	}
	return stream;
}

/** Read vector data from a text file. */
template< class T >
istream& operator>>(	istream& stream,
						vector< T >& array
						)
{
	while (stream.good() == true)
	{
		string line;
		getline(stream, line);
		// Skip lines beginning with "/" or "#" or empty lines
		if (line[ 0 ] == '/' || line[ 0 ] == '#' || line.empty())
			continue;

		stringstream ss( line );
		T val;
		ss >> val;
		array.push_back( val );
	}

	return stream;
}

/** Read matrix data from a text file. */
template< class T >
istream& operator>>(	istream& stream,
						vector< vector< T > >& array
						)
{
	while (stream.good() == true)
	{
		string line;
		vector< T > data;
		getline(stream, line);
		// Skip lines beginning with "/" or "#" or empty lines
		if (line[ 0 ] == '/' || line[ 0 ] == '#' || line.empty())
			continue;

		stringstream ss( line );
		copy(istream_iterator< T >( ss ), istream_iterator< T >(), back_inserter( data ));
		// Skip lines where we could not read anything
		if (data.size())
			array.push_back( data );
	}

	return stream;
}

/** "To string " conversion. */
template< typename T >
string toString(T const& value)
{
    stringstream ss;
    ss << value;
    return ss.str();
}

/** I/O formatter */
struct IoFormatter
{
	IoFormatter( ostream& _stream )
		: stream( _stream ), precision( _stream.precision() ), width( _stream.width() ), flags( _stream.flags() )
	{}

	void set( streamsize _precision, streamsize _width, ios_base::fmtflags _flags)
	{
		stream.precision( _precision );
		stream.width( _width );
		stream.setf( _flags );
	}

	void reset()
	{
		stream.precision( precision );
		stream.width( width );
		stream.setf( flags );
	}

	ostream& stream;
	streamsize precision;
	streamsize width;
	ios_base::fmtflags flags;
};

} // namespace std

#endif	// ACADO_TOOLKIT_ACADO_IO_UTILS_HPP

/*
 *	end of file
 */
