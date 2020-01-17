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
*    \file include/acado/symbolic_expression/parameter.hpp
*    \author Boris Houska, Hans Joachim Ferreau
*    \date 2008
*/


#ifndef ACADO_TOOLKIT_LYAPUNOV_HPP
#define ACADO_TOOLKIT_LYAPUNOV_HPP


#include <acado/symbolic_expression/acado_syntax.hpp>

BEGIN_NAMESPACE_ACADO


/**
 *	\brief Implements a parameter.
 *
 *	\ingroup BasicDataStructures
 *
 *	The class LYAPUNOV implements a Lyapunov  object
 *
 *	\author Boris Houska, Hans Joachim Ferreau
 */

class Lyapunov: public Expression{

public:

    Lyapunov();


    /** Default constructor */

	Lyapunov(const Expression &rhs1, const Expression &A_, const Expression &B_,
			const Expression &P_, const Expression &x1_, const Expression &u_,
			const Expression &p_);

	Lyapunov(const Expression &rhs1, const Expression &rhs2,
			const Expression &A_, const Expression &B_, const Expression &P_,
			const Expression &x1_, const Expression &x2_, const Expression &u_,
			const Expression &p_);

	Lyapunov(const Expression &rhs1, const Expression &rhs2,
			const Expression &A_, const Expression &B_, const Expression &P_,
			const Expression &x1_, const Expression &x2_, const Expression &u_,
			const Expression &p_, const Expression &useed_,
			const Expression &pseed_, const Expression &Yx1_,
			const Expression &Yx2_, const Expression &YP_);

	Lyapunov(const Expression &rhs1, const Expression &A_, const Expression &B_,
			const Expression &P_, const Expression &x1_, const Expression &u_,
			const Expression &p_, const Expression &w_);

	Lyapunov(const Expression &rhs1, const Expression &rhs2,
			const Expression &A_, const Expression &B_, const Expression &P_,
			const Expression &x1_, const Expression &x2_, const Expression &u_,
			const Expression &p_, const Expression &w_);

	Lyapunov(const Expression &rhs1, const Expression &rhs2,
			const Expression &A_, const Expression &B_, const Expression &P_,
			const Expression &x1_, const Expression &x2_, const Expression &u_,
			const Expression &p_, const Expression &w_,
			const Expression &useed_, const Expression &pseed_,
			const Expression &Yx1_, const Expression &Yx2_,
			const Expression &YP_);

    /** Default constructor */
    Lyapunov( const Lyapunov &arg );

    /** Default destructor. */
    virtual ~Lyapunov();

    Lyapunov& operator=( const Lyapunov &arg );


    BooleanType isEmpty() const;


//
//  PROTECTED MEMBERS:
//

public:

     Expression      rhs1;
     Expression      rhs2;
     Expression        A;
     Expression        B;
     Expression        P;
     Expression       x1;
     Expression       x2;
     Expression        u;
     Expression        p;
     Expression        w;
     Expression        pseed;
     Expression        useed;
     Expression       Yx1;
     Expression       Yx2;
     Expression       YP;
};


CLOSE_NAMESPACE_ACADO



#endif
