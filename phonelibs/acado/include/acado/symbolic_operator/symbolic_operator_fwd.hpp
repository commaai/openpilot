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
 *    \file include/acado/symbolic_operator/symbolic_operator_fwd.hpp
 *    \author Boris Houska, Hans Joachim Ferreau
 *    \date 2008
 */


#include <acado/matrix_vector/matrix_vector.hpp>
#include <acado/variables_grid/variables_grid.hpp>

BEGIN_NAMESPACE_ACADO


// FORWARD DECLARATIONS:
// ---------------------

   class IndexList                   ;
   class CFunction                   ;
   class COperator                   ;
   class SymbolicIndexList           ;

   class Operator                    ;
   class SmoothOperator              ;

   class BinaryOperator              ;
   class Addition                    ;
   class Subtraction                 ;
   class Product                     ;
   class Quotient                    ;
   class Power                       ;
   class Power_Int                   ;

   class UnaryOperator               ;
   class Sin                         ;
   class Cos                         ;
   class Tan                         ;
   class Asin                        ;
   class Acos                        ;
   class Atan                        ;
   class Exp                         ;
   class Power                       ;
   class Logarithm                   ;

   class DoubleConstant              ;
   class Projection                  ;
   class TreeProjection              ;


CLOSE_NAMESPACE_ACADO

// end of file.
