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
 *	\file include/acado/utils/acado_message_handling.hpp
 *	\author Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */

#ifndef ACADO_TOOLKIT_ACADO_MESSAGE_HANDLING_HPP
#define ACADO_TOOLKIT_ACADO_MESSAGE_HANDLING_HPP

#include <acado/utils/acado_namespace_macros.hpp>
#include <acado/utils/acado_types.hpp>

#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

#ifdef __MATLAB__
 #include <mex.h>
 #include <iostream>
#endif

BEGIN_NAMESPACE_ACADO

/** Defines visibility status of a message. */
enum VisibilityStatus
{
	VS_VISIBLE,	/**< Message visible. */
	VS_HIDDEN	/**< Message not visible. */
};

/** Colored/formatted terminal output */
#ifndef __MATLAB__

#define COL_DEBUG		"\033[1;34m"
#define COL_FATAL		"\033[0;31m"
#define COL_ERROR		"\033[1;31m"
#define COL_WARNING		"\033[1;33m"
#define COL_INFO		"\033[0m"

#else

#define COL_DEBUG		""
#define COL_FATAL		""
#define COL_ERROR		""
#define COL_WARNING		""
#define COL_INFO		""

#endif /* __MATLAB__ */

/** Defines whether user has handled the returned value */
enum returnValueStatus {
	STATUS_UNHANDLED, ///< returnValue was not yet handled by user
	STATUS_HANDLED    ///< returnValue was handled by user
};

/** Converts returnValueLevel enum type to a const char* */
const char* returnValueLevelToString(returnValueLevel level);

/** Converts returnValueType enum type to a const char* */
const char* returnValueTypeToString(returnValueType type);

// define DEBUG macros if necessary
#ifndef __FUNCTION__
	#define __FUNCTION__ 0
#endif

#ifndef __FILE__
	#define __FILE__ 0
#endif

#ifndef __LINE__
	#define __LINE__ 0
#endif

/** Macro to quote macro values as strings, e.g. __LINE__ number to string, used in other macros */
#define QUOTE_(x) #x
#define QUOTE(x) QUOTE_(x)

/** Macro to return a error */
#define ACADOERROR(retval) \
		returnValue("Code: ("#retval") \n  File: " __FILE__ "\n  Line: " QUOTE(__LINE__), LVL_ERROR, retval)

/** Macro to return a error, with user message */
#define ACADOERRORTEXT(retval, text) \
		returnValue("Message: "#text"\n  Code:    ("#retval") \n  File:    " __FILE__ "\n  Line:    " QUOTE(__LINE__), LVL_ERROR, retval)

/** Macro to return a fatal error */
#define ACADOFATAL(retval) \
		returnValue("Code: ("#retval") \n  File: " __FILE__ "\n  Line: " QUOTE(__LINE__), LVL_FATAL, retval)

/** Macro to return a fatal error, with user message */
#define ACADOFATALTEXT(retval, text) \
		returnValue("Message: "#text"\n  Code:    ("#retval") \n  File:    " __FILE__ "\n  Line:    " QUOTE(__LINE__), LVL_FATAL, retval)

/** Macro to return a warning */
#define ACADOWARNING(retval) \
		returnValue("Code: ("#retval") \n  File: " __FILE__ "\n  Line: " QUOTE(__LINE__), LVL_WARNING, retval)

/** Macro to return a warning, with user message */
#define ACADOWARNINGTEXT(retval,text) \
		returnValue("Message: "#text"\n  Code:    ("#retval") \n  File:    " __FILE__ "\n  Line:    " QUOTE(__LINE__), LVL_WARNING, retval)

/** Macro to return a information */
#define ACADOINFO(retval) \
		returnValue("", LVL_INFO, retval)

/** Macro to return a information, with user message */
#define ACADOINFOTEXT(retval,text) \
		returnValue("Message: "#text"\n  Code:    ("#retval") \n  File:    " __FILE__ "\n  Line:    " QUOTE(__LINE__), LVL_INFO, retval)


/** Executes the statement X and handles returned message.
 *  This is the default message handler. Statement X must return type returnValue.
 *  If message is not equal to SUCCESSFUL_RETURN a message is added informing where and what this statement is and imediately returned.
 *  Example: ACADO_TRY( func() );
 *  Example 2, extended use: ADACO_TRY( func() ).addMessage( "func() failed" );
 */
#define ACADO_TRY(X) for(returnValue ACADO_R = X; !ACADO_R;) return ACADO_R

/**
 *  \brief A very simple logging class.
 *  \ingroup BasicDataStructures
 *  \author Milan Vukov
 *  \date 2013.
 */
class Logger
{
public:
	/** Get an instance of the logger. */
	static Logger& instance()
	{
		static Logger instance;
		return instance;
	}

	/** Set the log level. */
	Logger& setLogLevel(returnValueLevel level)
	{
		logLevel = level;

		return *this;
	}

	/** Get log level. */
	returnValueLevel getLogLevel()
	{
		return logLevel;
	}

	/** Get a reference to the output stream. */
	std::ostream& get(returnValueLevel level);

private:
	returnValueLevel logLevel;

	Logger()
		: logLevel( LVL_FATAL )
	{}

	Logger(const Logger&);
	Logger& operator=(const Logger&);
	~Logger()
	{}
};

/// Just define a handy macro for getting the logger
#define LOG( level ) \
		if (level < Logger::instance().getLogLevel()); \
		else Logger::instance().get( level )



#ifdef __MATLAB__
// Writes character stream to MATLAB console.
class MatlabConsoleStreamBuf : public std::basic_streambuf<char>
{
protected:
    int_type overflow( int_type ch = traits_type::eof() )
    {
        if (!traits_type::eq_int_type(ch, traits_type::eof()))
            return mexPrintf("%c", traits_type::to_char_type(ch)) > 0 ? 0 : traits_type::eof();

        return 0;
    }
};

// Automatically restores previous rdbuf() when goes out of scope.
class RedirectStream
{
public:
    RedirectStream(std::ostream& stream, std::streambuf& new_streambuf)
    :   _stream(stream)
    ,   _old_streambuf(stream.rdbuf())
    {
        _stream.rdbuf(&new_streambuf);
    }

    ~RedirectStream()
    {
        _stream.rdbuf(_old_streambuf);
    }

private:
    std::ostream& _stream;
    std::streambuf * _old_streambuf;
};
#endif

CLOSE_NAMESPACE_ACADO



#endif	// ACADO_TOOLKIT_ACADO_MESSAGE_HANDLING_HPP

/*
 *	end of file
 */
