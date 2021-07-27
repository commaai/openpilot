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

/**	\file include/code_generation/export_algorithm_factory.hpp
 * 	\author Milan Vukov
 * 	\date 2012
 * */

#ifndef ACADO_TOOLKIT_EXPORT_ALGORITHM_FACTORY_HPP
#define ACADO_TOOLKIT_EXPORT_ALGORITHM_FACTORY_HPP

#include <acado/code_generation/export_algorithm.hpp>
#include <acado/utils/acado_types.hpp>
#include <acado/utils/acado_message_handling.hpp>

#include <map>

BEGIN_NAMESPACE_ACADO

/**	\brief  Factory for creation of exported algorithms.
 *
 * 	Note that the class is implemented as a singleton
 *
 * 	\author Milan Vukov
 * 	\date   2012 - 2013
 * */
template
<
	/** Base class, derived from ExportAlgorithm class. */
	class	 AlgorithmBase,
	/** Type identifier. */
	typename AlgorithmType
>
class ExportAlgorithmFactory
{
public:
	/** Helper type. */
	typedef AlgorithmBase* (*exportAlgorithmCreator)(UserInteraction* _userInteraction, const std::string &_commonHeaderName);

	/** Static creator function. */
	static ExportAlgorithmFactory& instance()
	{
		static ExportAlgorithmFactory instance;
		return instance;
	}

	/** Function for algorithm registration. */
	bool registerAlgorithm(	const AlgorithmType& id,
									exportAlgorithmCreator creator)
	{
		bool status = associations_.insert(
				typename idToProductMap::value_type(id, creator)).second;

		if ( status == true )
			return true;

		return false;
	}

	/** Function to unregister an algorithm. */
	bool unregisterAlgorithm(	const AlgorithmType& id)
	{
		bool status = associations_.erase( id ) == 1;

		if ( status == true )
			return true;

		return false;
	}

	/** Function to create an instance of an algorithm. */
	AlgorithmBase* createAlgorithm(	UserInteraction* _userInteraction,
									const std::string& _commonHeaderName,
									const AlgorithmType& id)
	{
		typename idToProductMap::const_iterator it = associations_.find( id );
		if (it != associations_.end())
		{
			return (it->second)(_userInteraction, _commonHeaderName);
		}

		LOG( LVL_DEBUG ) << "Algorithm is not registered!" << std::endl;

		return NULL;
	}

private:
	typedef std::map<AlgorithmType, exportAlgorithmCreator> idToProductMap;

	idToProductMap associations_;

	ExportAlgorithmFactory()
	{}

	ExportAlgorithmFactory(const ExportAlgorithmFactory&);

	ExportAlgorithmFactory& operator=(const ExportAlgorithmFactory&);

	~ExportAlgorithmFactory()
	{}
};

CLOSE_NAMESPACE_ACADO

#endif // ACADO_TOOLKIT_EXPORT_ALGORITHM_FACTORY_HPP
