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
 *    \file include/acado/utils/acado_debugging.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 17.08.2008
 *
 *    This file collects several macros for debugging.
 */


#ifndef ACADO_TOOLKIT_ACADO_DEBUGGING_HPP
#define ACADO_TOOLKIT_ACADO_DEBUGGING_HPP

#ifdef __MATLAB__
	#include <mex.h>
#endif

#if __DEBUG__

    #include "acado_message_handling.hpp"
    #define ASSERT(x)        {if (!(x))        ACADOFATALTEXT(RET_ASSERTION, Assertion failure: #x);}
    #define ASSERT_RETURN(x) {if (!(x)) return ACADOFATALTEXT(RET_ASSERTION, Assertion failure: #x);}

#else

    #define ASSERT( x )
	#define ASSERT_RETURN(x) if (0) returnValue()

#endif

#endif   // ACADO_TOOLKIT_ACADO_DEBUGGING_HPP

/*
 *    end of file
 */
