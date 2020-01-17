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
 *    \file include/acado/utils/acado_message_handling.ipp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 31.05.2008
 */



//
// PUBLIC MEMBER FUNCTIONS:
//

/* Construct default returnValue.
 *
 */
inline returnValue::returnValue() : data(0) {}

/* Construct returnValue only from typedef.
 *
 */
inline returnValue::returnValue(returnValueType _type) : type(_type), level(LVL_ERROR), status(STATUS_UNHANDLED), data(0) {}

/* Construct returnValue from int, for compatibility
 *
 */
inline returnValue::returnValue(int _type) : level(LVL_ERROR), status(STATUS_UNHANDLED), data(0) {
	type = returnValueType(_type);
}

/* Copy constructor with minimum performance cost.
 *  Newly constructed instance takes ownership of data.
 */
inline returnValue::returnValue(const returnValue& old) {
	// Copy data
	if (old.data) {
		data = old.data;
		data->owner = this;
	} else {
		data = 0;
	}
	// Copy details
	type = old.type;
	level = old.level;
	status = old.status;
}

inline returnValueLevel returnValue::getLevel() const {
	return level;
}

/* Compares the returnValue type to its enum
 *
 */
inline bool returnValue::operator!=(returnValueType cmp_type) const {
	return type != cmp_type;
}

/* Compares the returnValue type to its enum
 *
 */
inline bool returnValue::operator==(returnValueType cmp_type) const {
	return type == cmp_type;
}

/* Returns true if return value type is not SUCCESSFUL_RETURN
 *
 */
inline bool returnValue::operator!() const {
	return type != SUCCESSFUL_RETURN;
}

/*  Assignment operator.
 *  Left hand side instance takes ownership of data.
 */
inline returnValue& returnValue::operator=(const returnValue& old) {
	// Clean up data if already existing
	if (data && (data->owner == this)) delete data;

	// Copy data
	if (old.data) {
		data = old.data;
		data->owner = this;
	} else {
		data = 0;
	}
	// Copy details
	type = old.type;
	level = old.level;
	status = old.status;

	return *this;
}

/* Compatibility function, allows returnValue to be used as a number, similar to a enum.
 *
 */
inline returnValue::operator int() {
	return type;
}


/*
 *	end of file
 */
