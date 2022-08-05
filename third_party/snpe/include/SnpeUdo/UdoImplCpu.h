//==============================================================================
//
// Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

// Header to be used by a CPU UDO Implementation library

#ifndef SNPE_UDO_IMPL_CPU_H
#define SNPE_UDO_IMPL_CPU_H

#include <stdio.h>

/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * @brief This struct provides the infrastructure needed by a developer of
 * CPU UDO Implementation library.
 *
 * The framework/runtime which loads the CPU UDO implementation library provides
 * this infrastructure data to the loaded library at the time of op factory creation.
 * as an opaque pointer. It contains hooks for the UDO library to invoke supported
 * functionality at the time of execution
 *
 * @param getData function pointer to retrieve raw tensor data from opaque pointer
 *  passed into the UDO when creating an instance.
 * @param getDataSize function pointer to retrieve tensor data size from opaque pointer
 */

typedef struct
{
   /// function pointer to retrieve raw tensor data from opaque pointer
   /// passed into the UDO when creating an instance.
   float* (*getData)(void*);
   /// function pointer to retrieve tensor data size from opaque pointer
   size_t (*getDataSize) (void*);
} SnpeUdo_CpuInfrastructure_t;

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

#endif // SNPE_UDO_IMPL_CPU_H