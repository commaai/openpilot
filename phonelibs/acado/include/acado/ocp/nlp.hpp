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
 *    \file include/acado/ocp/nlp.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 */

#ifndef ACADO_TOOLKIT_NLP_HPP
#define ACADO_TOOLKIT_NLP_HPP

#include <acado/ocp/ocp.hpp>

BEGIN_NAMESPACE_ACADO

/** 
 *	\brief Data class for defining static optimization problems.
 *
 *	\ingroup BasicDataStructures
 *
 *	The class NLP is a data class for defining static optimization problems.
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */
class NLP: public OCP
{
public:
    /** Default constructor. */
    NLP();

    /** Destructor. */
    virtual ~NLP();

    returnValue minimize( const Expression& arg );

    returnValue minimize( const int &multiObjectiveIdx,  const Expression& arg );
};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_NLP_HPP
