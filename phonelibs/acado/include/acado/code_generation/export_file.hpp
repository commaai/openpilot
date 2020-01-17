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
 *    \file include/acado/integrator/export_file.hpp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date 2010-2011
 */


#ifndef ACADO_TOOLKIT_EXPORT_FILE_HPP
#define ACADO_TOOLKIT_EXPORT_FILE_HPP

#include <acado/code_generation/export_statement_block.hpp>

BEGIN_NAMESPACE_ACADO

/** 
 *	\brief Allows to export files containing automatically generated algorithms for fast model predictive control
 *
 *	\ingroup NumericalAlgorithms
 *
 *	The class ExportFile allows to export files containing automatically generated 
 *	algorithms for fast model predictive control.
 *
 *	\author Hans Joachim Ferreau, Milan Vukov, Boris Houska
 */
class ExportFile : public ExportStatementBlock
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //

    public:

        /** Default constructor. 
		 */
		ExportFile( );
                
		/** Standard constructor. 
		 *
		 *	@param[in] _fileName			Name of exported file.
		 *	@param[in] _commonHeaderName	Name of common header file to be included.
		 *	@param[in] _realString			std::string to be used to declare real variables.
		 *	@param[in] _intString			std::string to be used to declare integer variables.
		 *	@param[in] _precision			Number of digits to be used for exporting real values.
		 *	@param[in] _commentString		std::string to be used for exporting comments.
		 */
		ExportFile(	const std::string& _fileName,
					const std::string& _commonHeaderName = "",
					const std::string& _realString = "real_t",
					const std::string& _intString = "int",
					int _precision = 16,
					const std::string& _commentString = std::string()
					);

        /** Destructor. */
        virtual ~ExportFile( );

        
        /** Setup routine.
		 *
		 *	@param[in] _fileName			Name of exported file.
		 *	@param[in] _commonHeaderName	Name of common header file to be included.
		 *	@param[in] _realString			std::string to be used to declare real variables.
		 *	@param[in] _intString			std::string to be used to declare integer variables.
		 *	@param[in] _precision			Number of digits to be used for exporting real values.
		 *	@param[in] _commentString		std::string to be used for exporting comments.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue setup(	const std::string& _fileName,
                                    const std::string& _commonHeaderName = "",
                                    const std::string& _realString = "real_t",
                                    const std::string& _intString = "int",
                                    int _precision = 16,
                                    const std::string& _commentString = std::string()
                                    );
        
        
		/** Exports the file containing the auto-generated code.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		virtual returnValue exportCode( ) const;

    protected:

		std::string fileName;					/**< Name of exported file. */
		std::string commonHeaderName;			/**< Name of common header file. */
		
		std::string realString;					/**< std::string to be used to declare real variables. */
		std::string intString;					/**< std::string to be used to declare integer variables. */
		int precision;							/**< Number of digits to be used for exporting real values. */
		std::string commentString;				/**< std::string to be used for exporting comments. */
};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_EXPORT_FILE_HPP

// end of file.
