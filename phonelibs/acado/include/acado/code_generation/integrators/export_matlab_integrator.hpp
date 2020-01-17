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
 *    \file ...
 *    \author Rien Quirynen
 *    \date 2012
 */


#ifndef ACADO_TOOLKIT_EXPORT_MATLAB_INTEGRATOR_HPP
#define ACADO_TOOLKIT_EXPORT_MATLAB_INTEGRATOR_HPP


#include <acado/code_generation/export_templated_file.hpp>


BEGIN_NAMESPACE_ACADO


/** 
 *	\brief ...
 *
 *	\ingroup ...
 *
 *	\author Rien Quirynen
 */
class ExportMatlabIntegrator : public ExportTemplatedFile
{
    //
    // PUBLIC MEMBER FUNCTIONS:
    //

    public:

		/** Default constructor. 
		 *
		 *	@param[in] _templateName		Name of a template.
		 *	@param[in] _fileName			Name of exported file.
		 *	@param[in] _commonHeaderName	Name of common header file to be included.
		 *	@param[in] _realString			std::string to be used to declare real variables.
		 *	@param[in] _intString			std::string to be used to declare integer variables.
		 *	@param[in] _precision			Number of digits to be used for exporting real values.
		 *	@param[in] _commentString		std::string to be used for exporting comments.
		 *
		 *	\return SUCCESSFUL_RETURN
		 */
		ExportMatlabIntegrator(	const std::string& _templateName,
								const std::string& _fileName,
								const std::string& _commonHeaderName = "",
								const std::string& _realString = "real_t",
								const std::string& _intString = "int",
								int _precision = 16,
								const std::string& _commentString = ""
								);

		/** Copy constructor (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
        ExportMatlabIntegrator(	const ExportMatlabIntegrator& arg
								);

        /** Destructor. 
		 */
        virtual ~ExportMatlabIntegrator( );

		/** Assignment operator (deep copy).
		 *
		 *	@param[in] arg		Right-hand side object.
		 */
		ExportMatlabIntegrator& operator=(	const ExportMatlabIntegrator& arg
										);
		
		/** Configure the template
		 *
		 *  \return SUCCESSFUL_RETURN
		 */
		returnValue configure(	const uint firstOrder, const uint online, const uint debugMode, const uint timingCalls, const uint numStages );

	protected:

};


CLOSE_NAMESPACE_ACADO


#endif  // ACADO_TOOLKIT_EXPORT_MATLAB_INTEGRATOR_HPP
