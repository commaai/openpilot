/*
 *	This file is part of qpOASES.
 *
 *	qpOASES -- An Implementation of the Online Active Set Strategy.
 *	Copyright (C) 2007-2015 by Hans Joachim Ferreau, Andreas Potschka,
 *	Christian Kirches et al. All rights reserved.
 *
 *	qpOASES is free software; you can redistribute it and/or
 *	modify it under the terms of the GNU Lesser General Public
 *	License as published by the Free Software Foundation; either
 *	version 2.1 of the License, or (at your option) any later version.
 *
 *	qpOASES is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *	See the GNU Lesser General Public License for more details.
 *
 *	You should have received a copy of the GNU Lesser General Public
 *	License along with qpOASES; if not, write to the Free Software
 *	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


/**
 *	\file include/qpOASES_e/ConstraintProduct.h
 *	\author Hans Joachim Ferreau, Andreas Potschka, Christian Kirches (thanks to D. Kwame Minde Kufoalor)
 *	\version 3.1embedded
 *	\date 2009-2015
 *
 *	Declaration of the ConstraintProduct interface which allows to specify a
 *	user-defined function for evaluating the constraint product at the 
 *	current iterate to speed-up QP solution in case of a specially structured
 *	constraint matrix.
 */



#ifndef QPOASES_CONSTRAINT_PRODUCT_H
#define QPOASES_CONSTRAINT_PRODUCT_H


BEGIN_NAMESPACE_QPOASES


/** 
 *	\brief Interface for specifying user-defined evaluations of constraint products.
 *
 *	An interface which allows to specify a user-defined function for evaluating the 
 *	constraint product at the current iterate to speed-up QP solution in case 
 *	of a specially structured constraint matrix.
 *
 *	\author Hans Joachim Ferreau (thanks to Kwame Minde Kufoalor)
 *	\version 3.1embedded
 *	\date 2009-2015
 */
typedef int(*ConstraintProduct)( int, const real_t* const, real_t* const );


END_NAMESPACE_QPOASES

#endif	/* QPOASES_CONSTRAINT_PRODUCT_H */
