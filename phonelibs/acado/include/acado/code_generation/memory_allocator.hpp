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
 *    \file include/code_generation/memory_allocator.hpp
 *    \author Milan Vukov
 *    \date 2012
 */

#ifndef ACADO_TOLLKIT_MEMORY_ALLOCATOR
#define ACADO_TOLLKIT_MEMORY_ALLOCATOR

#include <acado/utils/acado_utils.hpp>
#include <acado/code_generation/object_pool.hpp>
#include <acado/code_generation/export_index.hpp>

BEGIN_NAMESPACE_ACADO

class MemoryAllocator
{
public:
	MemoryAllocator()
	{}

	~MemoryAllocator()
	{}

	returnValue acquire(ExportIndex& _obj);

	returnValue release(const ExportIndex& _obj);

	returnValue add(const ExportIndex& _obj);

	std::vector< ExportIndex > getPool( void );

private:
	ObjectPool<ExportIndex, ExportIndexComparator> indices;
};

CLOSE_NAMESPACE_ACADO

#endif // ACADO_TOLLKIT_MEMORY_ALLOCATOR

