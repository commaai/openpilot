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
 *	\file include/acado/matrix_vector/acado_mat_file.hpp
 *	\author Carlo Savorgnan, Hans Joachim Ferreau, Boris Houska
 */

#ifndef ACADO_TOOLKIT_ACADO_MAT_FILE_HPP
#define ACADO_TOOLKIT_ACADO_MAT_FILE_HPP

#include <acado/matrix_vector/matrix_vector.hpp>

BEGIN_NAMESPACE_ACADO

/**
 *	\brief Simple class for writing binary data file that are compatible with Matlab.
 *	
 *  Simple class for writing binary data file that are compatible with Matlab.
 *
 *	\author Carlo Savorgnan, Hans Joachim Ferreau, Boris Houska, Milan Vukov
 */
template<typename T>
class MatFile
{
	/*
	 *	INTERNAL DATA STRUCTURES
	 */
	typedef struct
	{
		long type;
		long mrows;
		long ncols;
		long imagf;
		long namelen;
	} Fmatrix;

		
	/*
	 *	PUBLIC MEMBER FUNCTIONS
	 */
	public:

		void write( std::ostream& stream,
					const GenericMatrix< T >& mat,
					const char* name
					);

		void write(	std::ostream& stream,
					const GenericVector< T >& vec,
					const char* name
					);
};

CLOSE_NAMESPACE_ACADO

#endif	// ACADO_TOOLKIT_ACADO_MAT_FILE_HPP
