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
 *    \file include/acado/user_interaction/logging.ipp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 12.05.2009
 */



BEGIN_NAMESPACE_ACADO


//
// PUBLIC MEMBER FUNCTIONS:
//

inline returnValue Logging::getAll(	LogName _name,
									MatrixVariablesGrid& _values
									) const
{
	for (unsigned it = 0; it < logCollection.size(); ++it)
		if (logCollection[ it ].hasNonEmptyItem( _name ) == true)
			return logCollection[ it ].getAll(_name, _values); 

	return ACADOERROR( RET_LOG_ENTRY_DOESNT_EXIST );
}


inline returnValue Logging::getFirst(	LogName _name,
										DMatrix& _firstValue
										) const
{
	for (unsigned it = 0; it < logCollection.size(); ++it)
		if (logCollection[ it ].hasNonEmptyItem( _name ) == true)
			return logCollection[ it ].getFirst(_name, _firstValue); 

	return ACADOERROR( RET_LOG_ENTRY_DOESNT_EXIST );
}


inline returnValue Logging::getFirst(	LogName _name,
										VariablesGrid& _firstValue
										) const
{
	for (unsigned it = 0; it < logCollection.size(); ++it)
		if (logCollection[ it ].hasNonEmptyItem( _name ) == BT_TRUE)
			return logCollection[ it ].getFirst(_name, _firstValue); 

	return ACADOERROR( RET_LOG_ENTRY_DOESNT_EXIST );
}


inline returnValue Logging::getLast(	LogName _name,
										DMatrix& _lastValue
										) const
{
	for (unsigned it = 0; it < logCollection.size(); ++it)
		if (logCollection[ it ].hasNonEmptyItem( _name ) == BT_TRUE)
			return logCollection[ it ].getLast(_name, _lastValue); 

	return ACADOERROR( RET_LOG_ENTRY_DOESNT_EXIST );
}


inline returnValue Logging::getLast(	LogName _name,
										VariablesGrid& _lastValue
										) const
{
	for (unsigned it = 0; it < logCollection.size(); ++it)
		if (logCollection[ it ].hasNonEmptyItem( _name ) == BT_TRUE)
			return logCollection[ it ].getLast(_name, _lastValue); 

	return ACADOERROR( RET_LOG_ENTRY_DOESNT_EXIST );
}

		
inline returnValue Logging::setAll(	LogName _name,
									const MatrixVariablesGrid& values
									)
{
	for (unsigned it = 0; it < logCollection.size(); ++it)
		if (logCollection[ it ].hasItem( _name ) == true)
			return logCollection[ it ].setAll(_name, values); 

	return SUCCESSFUL_RETURN;
}


		
inline returnValue Logging::setLast(	LogName _name,
										const DMatrix& value,
										double time
										)
{
	for (unsigned it = 0; it < logCollection.size(); ++it)
		if (logCollection[ it ].hasItem( _name ) == true)
			return logCollection[ it ].setLast(_name, value, time); 

	return SUCCESSFUL_RETURN;
}

inline returnValue Logging::setLast(	LogName _name,
										VariablesGrid& value,
										double time
										)
{
	for (unsigned it = 0; it < logCollection.size(); ++it)
		if (logCollection[ it ].hasItem( _name ) == true)
			return logCollection[ it ].setLast(_name, value, time);

	return SUCCESSFUL_RETURN;
}

CLOSE_NAMESPACE_ACADO

/*
 *	end of file
 */
