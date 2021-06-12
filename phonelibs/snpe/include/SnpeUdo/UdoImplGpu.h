//==============================================================================
//
// Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

// Header to be used by a GPU UDO Implementation library

#ifndef SNPE_UDO_IMPL_GPU_H
#define SNPE_UDO_IMPL_GPU_H

#include "CL/cl.h"
#include "SnpeUdo/UdoBase.h"

/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * This header defines version 0.0.0 of the GPU UDO Infrastructure.
 * It defines the interpretation of the global and per-OpFactory infrastructure pointers
 * as well as the interpretation of tensorData pointers.
 *
 * The per-Operation infrastructure pointer is defined to be null, and should not be used.
 *
 * The SnpeUdoTensorParam_t struct below provides the interpretation for
 * the tensorData opaque pointer for SnpeUdoTensorParams representing inputs or outputs.
 *
 * The tensorData opaque pointer populated in SnpeUdoScalarParam_t structs should be interpreted
 * as a host-readable data pointer.
 *
 */

/**
 * @brief Function to retrieve opencl program from Program Cache repository.
 * @param programCache is opaque pointer to Program Cache repository provided by
 * SNPE GPU UDO runtime.
 * @param programName is name associated with opencl program for UDO.
 * @param program is pointer to opencl program which will be populated with
 * valid opencl program if found in Program Cache repository.
 * @return SnpeUdo_ErrorType_t is error type returned. SNPE_UDO_NO_ERROR is returned
 * on success.
 */
typedef SnpeUdo_ErrorType_t (*SnpeUdo_getProgram_t)
   (void* programCache, const char* programName, cl_program* program);

/**
 * @brief Function to store valid opencl program in Program Cache repository.
 * @param programCache is opaque pointer to Program Cache repository provided by
 * SNPE GPU UDO runtime.
 * @param programName is name associated with opencl program for UDO.
 * @param program is valid opencl program after program is built.
 * @return SnpeUdo_ErrorType_t is error type returned. SNPE_UDO_NO_ERROR is returned
 * on success.
 * */
typedef SnpeUdo_ErrorType_t (*SnpeUdo_storeProgram_t)
   (void* programCache, const char * programName, cl_program program);

/**
 * @brief Global Infrastructure Definition for GPU UDO Implementations.
 */
typedef struct {
   // Infrastructure definition version. This header is 0.0.0
   SnpeUdo_Version_t   gpuInfraVersion;
   SnpeUdo_getProgram_t SnpeUdo_getProgram;
   SnpeUdo_storeProgram_t SnpeUdo_storeProgram;
} SnpeUdo_GpuInfrastructure_t;

/**
 * @brief Per OpFactory Infrastructure Definition for GPU UDO Implementations.
 * @note  This version of the infrastructure definition guarantees that the same
 *        Per OpFactory infrastructure pointer will be provided to all OpFactories
 *        in the same network.
 */
typedef struct
{
   cl_context context;
   cl_command_queue commandQueue;
   void* programCache;
} SnpeUdo_GpuOpFactoryInfrastructure_t;

/**
 * @brief Opaque tensorData definition for operation inputs and outputs.
 *
 * The following is a list of all SnpeUdoTensorLayout_t values supported by the
 * GPU UDO implementation, and how the parameters of the struct should be
 * interpreted in each case:
 *
 * SNPE_UDO_LAYOUT_NHWC:
 *   mem shall be single-element array, pointing to a cl buffer memory object.
 *   the dimensions of this object match the dimensions specified in the encompassing
 *   SnpeUdoTensorParam_t's currDimensions.
 *
 *   memCount shall be 1.
 *
 *   paddedRank and paddedDimensions are undefined and shall be ignored by the UDO
 *   implementation.
 *
 */
typedef struct
{
   cl_mem*   mem;
   uint32_t  memCount;
   uint32_t  paddedRank;
   uint32_t* paddedDimensions;

} SnpeUdo_GpuTensorData_t;

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

#endif // SNPE_UDO_IMPL_GPU_H
