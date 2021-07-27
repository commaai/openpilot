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
 *    \file include/acado/ocp/multi_objective_functionality.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 */


#ifndef ACADO_TOOLKIT_MULTI_OBJECTIVE_FUNCTIONALITY_HPP
#define ACADO_TOOLKIT_MULTI_OBJECTIVE_FUNCTIONALITY_HPP

#include <acado/utils/acado_types.hpp>

BEGIN_NAMESPACE_ACADO

class Expression;

/**
 *	\brief Encapsulates functionality for defining OCPs having multiple objectives.
 *
 *	\ingroup AuxiliaryFunctionality
 *
 *	The class MultiObjectiveFunctionality is a data class that encapsulates 
 *	all functionality for defining optimal control problems having multiple 
 *	objectives.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class MultiObjectiveFunctionality
{
public:
	/** Default constructor. */
	MultiObjectiveFunctionality( );

	/** Copy constructor (deep copy). */
	MultiObjectiveFunctionality( const MultiObjectiveFunctionality& rhs );

	/** Destructor. */
	~MultiObjectiveFunctionality( );

	/** Assignment operator (deep copy). */
	MultiObjectiveFunctionality& operator=( const MultiObjectiveFunctionality& rhs );

	/** Adds an expression as a the Mayer term to be minimized, within \n
	 *  a multi-objective context.                                     \n
	 *                                                                 \n
	 *  @param multiObjectiveIdx The index of the objective the        \n
	 *                           expression should be added to.        \n
	 *                                                                 \n
	 *  @param arg The expression to be added as a Mayer term.         \n
	 *                                                                 \n
	 *  \return SUCCESSFUL_RETURN                                      \n
	 */
	returnValue minimizeMayerTerm( const int &multiObjectiveIdx,  const Expression& arg );

	int getNumberOfMayerTerms() const;

	returnValue getObjective( const int &multiObjectiveIdx, Expression **arg ) const;

protected:

	int              nMayer;
	Expression **mayerTerms;
};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_MULTI_OBJECTIVE_FUNCTIONALITY_HPP
