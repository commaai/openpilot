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
 *    \file include/acado/utils/acado_namespace_macros.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *    \date 22.01.2009
 *
 *    This file collects several macros which can be used to
 *    define the ACADO namespace. By default this namespace
 *    is used. However, as some ugly C++ compiler versions 
 *    complain about namespaces you can turn them off by
 *    using the following compiler option:
 *
 *                -D__WITHOUT_NAMESPACE__
 *
 *    For developers: It is recommended to use the macros
 *    below instead of directly using the standard C++
 *    namespace notation.
 */

#ifndef ACADO_TOOLKIT_ACADO_NAMESPACE_MACROS_HPP
#define ACADO_TOOLKIT_ACADO_NAMESPACE_MACROS_HPP

#ifdef __WITHOUT_NAMESPACE__

    #define BEGIN_NAMESPACE_ACADO
    #define CLOSE_NAMESPACE_ACADO
    #define USING_NAMESPACE_ACADO
    #define REFER_NAMESPACE_ACADO

#else

    #define BEGIN_NAMESPACE_ACADO  namespace ACADO{
    #define CLOSE_NAMESPACE_ACADO  }
    #define USING_NAMESPACE_ACADO  using namespace ACADO;
    #define REFER_NAMESPACE_ACADO  ACADO::

#endif

#endif   // ACADO_TOOLKIT_ACADO_NAMESPACE_MACROS_HPP

// end of file

