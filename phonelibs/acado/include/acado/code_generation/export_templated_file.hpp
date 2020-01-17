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
 *    \file include/acado/code_generation/export_templated_file.hpp
 *    \author Milan Vukov
 *    \date 2012 - 2013
 */

#ifndef ACADO_TOOLKIT_EXPORT_TEMPLATED_FILE_HPP
#define ACADO_TOOLKIT_EXPORT_TEMPLATED_FILE_HPP

#include <acado/code_generation/export_file.hpp>

#include <map>

BEGIN_NAMESPACE_ACADO

class ExportQpOasesInterface;
class ExportSimulinkInterface;
class ExportAuxiliaryFunctions;

/** 
 *	\brief Allows export of template files.
 *
 *	\ingroup AuxiliaryFunctionality
 *
 *	\author Milan Vukov
 */
class ExportTemplatedFile : public ExportFile
{
public:

	friend class ExportQpOasesInterface;
	friend class ExportQpOases3Interface;
	friend class ExportSimulinkInterface;
	friend class ExportAuxiliaryFunctions;
	friend class ExportHessianRegularization;
	friend class ExportAuxiliarySimFunctions;
    friend class OCPexport;
    friend class SIMexport;

    /** Default constructor.
	 */
	ExportTemplatedFile( );
    
            
	/** Standard constructor.
	 *
	 *	@param[in] _templateName		Name of a template.
	 *	@param[in] _fileName			Name of exported file.
	 *	@param[in] _commonHeaderName	Name of common header file to be included.
	 *	@param[in] _realString			std::string to be used to declare real variables.
	 *	@param[in] _intString			std::string to be used to declare integer variables.
	 *	@param[in] _precision			Number of digits to be used for exporting real values.
	 *	@param[in] _commentString		std::string to be used for exporting comments.
	 */
	ExportTemplatedFile(	const std::string& _templateName,
							const std::string& _fileName,
							const std::string& _commonHeaderName = "",
							const std::string& _realString = "real_t",
							const std::string& _intString = "int",
							int _precision = 16,
							const std::string& _commentString = std::string()
							);

	/** Destructor. */
	virtual ~ExportTemplatedFile( )
	{}

    
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
	virtual returnValue setup(	const std::string& _templateName,
                                const std::string& _fileName,
                                const std::string& _commonHeaderName = "",
                                const std::string& _realString = "real_t",
                                const std::string& _intString = "int",
                                int _precision = 16,
                                const std::string& _commentString = std::string()
                                );

    
	/** Configure the template
	 *
	 *  \return SUCCESSFUL_RETURN
	 */
	virtual returnValue configure(  )
	{
		return fillTemplate( );
	}

protected:

	/** Fill in the template. */
	returnValue fillTemplate( );
	/** Name of the template file. */
	std::string templateName;
	/** Dictionary used to fill in the template file. */
	std::map< std::string, std::string > dictionary;
	/** List of folders where templates are stored. */
	std::string folders;
};

CLOSE_NAMESPACE_ACADO

#endif  // ACADO_TOOLKIT_EXPORT_TEMPLATED_FILE_HPP
