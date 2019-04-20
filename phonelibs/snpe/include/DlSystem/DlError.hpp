//==============================================================================
//
//  Copyright (c) 2016-18 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef _DL_ERROR_HPP_
#define _DL_ERROR_HPP_

#include <stdint.h>
#include <limits> // numeric_limits

#include "DlSystem/ZdlExportDefine.hpp"

namespace zdl {
namespace DlSystem {

// clang and arm gcc different in how ZDL_EXPORT is used with enum class
#if !defined (__clang__)
enum class ErrorCode : uint32_t ZDL_EXPORT {
#else
enum class ZDL_EXPORT ErrorCode : uint32_t {
#endif // ARM64V8A
   NONE = 0,

   // System config errors
   SNPE_CONFIG_MISSING_PARAM                             = 100,
   SNPE_CONFIG_INVALID_PARAM                             = 101,
   SNPE_CONFIG_MISSING_FILE                              = 102,
   SNPE_CONFIG_NNCONFIG_NOT_SET                          = 103,
   SNPE_CONFIG_NNCONFIG_INVALID                          = 104,
   SNPE_CONFIG_WRONG_INPUT_NAME                          = 105,
   SNPE_CONFIG_INCORRECT_INPUT_DIMENSIONS                = 106,
   SNPE_CONFIG_DIMENSIONS_MODIFICATION_NOT_SUPPORTED     = 107,

   SNPE_CONFIG_NNCONFIG_ONLY_TENSOR_SUPPORTED            = 120,
   SNPE_CONFIG_NNCONFIG_ONLY_USER_BUFFER_SUPPORTED       = 121,

   // DlSystem errors
   SNPE_DLSYSTEM_MISSING_BUFFER                          = 200,
   SNPE_DLSYSTEM_TENSOR_CAST_FAILED                      = 201,
   SNPE_DLSYSTEM_FIXED_POINT_PARAM_INVALID               = 202,
   SNPE_DLSYSTEM_SIZE_MISMATCH                           = 203,
   SNPE_DLSYSTEM_NAME_NOT_FOUND                          = 204,
   SNPE_DLSYSTEM_VALUE_MISMATCH                          = 205,
   SNPE_DLSYSTEM_INSERT_FAILED                           = 206,
   SNPE_DLSYSTEM_TENSOR_FILE_READ_FAILED                 = 207,
   SNPE_DLSYSTEM_DIAGLOG_FAILURE                         = 208,
   SNPE_DLSYSTEM_LAYER_NOT_SET                           = 209,
   SNPE_DLSYSTEM_WRONG_NUMBER_INPUT_BUFFERS              = 210,
   SNPE_DLSYSTEM_RUNTIME_TENSOR_SHAPE_MISMATCH           = 211,
   SNPE_DLSYSTEM_TENSOR_MISSING                          = 212,
   SNPE_DLSYSTEM_TENSOR_ITERATION_UNSUPPORTED            = 213,
   SNPE_DLSYSTEM_BUFFER_MANAGER_MISSING                  = 214,
   SNPE_DLSYSTEM_RUNTIME_BUFFER_SOURCE_UNSUPPORTED       = 215,
   SNPE_DLSYSTEM_BUFFER_CAST_FAILED                      = 216,

   SNPE_DLSYSTEM_BUFFERENCODING_UNKNOWN                  = 240,

   // DlContainer errors
   SNPE_DLCONTAINER_MODEL_PARSING_FAILED                 = 300,
   SNPE_DLCONTAINER_UNKNOWN_LAYER_CODE                   = 301,
   SNPE_DLCONTAINER_MISSING_LAYER_PARAM                  = 302,
   SNPE_DLCONTAINER_LAYER_PARAM_NOT_SUPPORTED            = 303,
   SNPE_DLCONTAINER_LAYER_PARAM_INVALID                  = 304,
   SNPE_DLCONTAINER_TENSOR_DATA_MISSING                  = 305,
   SNPE_DLCONTAINER_MODEL_LOAD_FAILED                    = 306,
   SNPE_DLCONTAINER_MISSING_RECORDS                      = 307,
   SNPE_DLCONTAINER_INVALID_RECORD                       = 308,
   SNPE_DLCONTAINER_WRITE_FAILURE                        = 309,
   SNPE_DLCONTAINER_READ_FAILURE                         = 310,
   SNPE_DLCONTAINER_BAD_CONTAINER                        = 311,
   SNPE_DLCONTAINER_BAD_DNN_FORMAT_VERSION               = 312,
   SNPE_DLCONTAINER_UNKNOWN_AXIS_ANNOTATION              = 313,
   SNPE_DLCONTAINER_UNKNOWN_SHUFFLE_TYPE                 = 314,

   // Network errors
   SNPE_NETWORK_EMPTY_NETWORK                            = 400,
   SNPE_NETWORK_CREATION_FAILED                          = 401,
   SNPE_NETWORK_PARTITION_FAILED                         = 402,
   SNPE_NETWORK_NO_OUTPUT_DEFINED                        = 403,
   SNPE_NETWORK_MISMATCH_BETWEEN_NAMES_AND_DIMS          = 404,
   SNPE_NETWORK_MISSING_INPUT_NAMES                      = 405,
   SNPE_NETWORK_MISSING_OUTPUT_NAMES                     = 406,

   // Host runtime errors
   SNPE_HOST_RUNTIME_TARGET_UNAVAILABLE                  = 500,

   // CPU runtime errors
   SNPE_CPU_LAYER_NOT_SUPPORTED                          = 600,
   SNPE_CPU_LAYER_PARAM_NOT_SUPPORTED                    = 601,
   SNPE_CPU_LAYER_PARAM_INVALID                          = 602,
   SNPE_CPU_LAYER_PARAM_COMBINATION_INVALID              = 603,
   SNPE_CPU_BUFFER_NOT_FOUND                             = 604,
   SNPE_CPU_NETWORK_NOT_SUPPORTED                        = 605,

   // CPU fixed-point runtime errors
   SNPE_CPU_FXP_LAYER_NOT_SUPPORTED                      = 700,
   SNPE_CPU_FXP_LAYER_PARAM_NOT_SUPPORTED                = 701,
   SNPE_CPU_FXP_LAYER_PARAM_INVALID                      = 702,

   // GPU runtime errors
   SNPE_GPU_LAYER_NOT_SUPPORTED                          = 800,
   SNPE_GPU_LAYER_PARAM_NOT_SUPPORTED                    = 801,
   SNPE_GPU_LAYER_PARAM_INVALID                          = 802,
   SNPE_GPU_LAYER_PARAM_COMBINATION_INVALID              = 803,
   SNPE_GPU_KERNEL_COMPILATION_FAILED                    = 804,
   SNPE_GPU_CONTEXT_NOT_SET                              = 805,
   SNPE_GPU_KERNEL_NOT_SET                               = 806,
   SNPE_GPU_KERNEL_PARAM_INVALID                         = 807,
   SNPE_GPU_OPENCL_CHECK_FAILED                          = 808,
   SNPE_GPU_OPENCL_FUNCTION_ERROR                        = 809,
   SNPE_GPU_BUFFER_NOT_FOUND                             = 810,
   SNPE_GPU_TENSOR_DIM_INVALID                           = 811,
   SNPE_GPU_MEMORY_FLAGS_INVALID                         = 812,
   SNPE_GPU_UNEXPECTED_NUMBER_OF_IO                      = 813,
   SNPE_GPU_LAYER_PROXY_ERROR                            = 814,
   SNPE_GPU_BUFFER_IN_USE                                = 815,
   SNPE_GPU_BUFFER_MODIFICATION_ERROR                    = 816,

   // DSP runtime errors
   SNPE_DSP_LAYER_NOT_SUPPORTED                          = 900,
   SNPE_DSP_LAYER_PARAM_NOT_SUPPORTED                    = 901,
   SNPE_DSP_LAYER_PARAM_INVALID                          = 902,
   SNPE_DSP_LAYER_PARAM_COMBINATION_INVALID              = 903,
   SNPE_DSP_STUB_NOT_PRESENT                             = 904,
   SNPE_DSP_LAYER_NAME_TRUNCATED                         = 905,
   SNPE_DSP_LAYER_INPUT_BUFFER_NAME_TRUNCATED            = 906,
   SNPE_DSP_LAYER_OUTPUT_BUFFER_NAME_TRUNCATED           = 907,
   SNPE_DSP_RUNTIME_COMMUNICATION_ERROR                  = 908,
   SNPE_DSP_RUNTIME_INVALID_PARAM_ERROR                  = 909,
   SNPE_DSP_RUNTIME_SYSTEM_ERROR                         = 910,

   // Model validataion errors
   SNPE_MODEL_VALIDATION_LAYER_NOT_SUPPORTED             = 1000,
   SNPE_MODEL_VALIDATION_LAYER_PARAM_NOT_SUPPORTED       = 1001,
   SNPE_MODEL_VALIDATION_LAYER_PARAM_INVALID             = 1002,
   SNPE_MODEL_VALIDATION_LAYER_PARAM_MISSING             = 1003,
   SNPE_MODEL_VALIDATION_LAYER_PARAM_COMBINATION_INVALID = 1004,
   SNPE_MODEL_VALIDATION_LAYER_ORDERING_INVALID          = 1005,
   SNPE_MODEL_VALIDATION_INVALID_CONSTRAINT              = 1006,
   SNPE_MODEL_VALIDATION_MISSING_BUFFER                  = 1007,
   SNPE_MODEL_VALIDATION_BUFFER_REUSE_NOT_SUPPORTED      = 1008,

   // UDL errors
   SNPE_UDL_LAYER_EMPTY_UDL_NETWORK                      = 1100,
   SNPE_UDL_LAYER_PARAM_INVALID                          = 1101,
   SNPE_UDL_LAYER_INSTANCE_MISSING                       = 1102,
   SNPE_UDL_LAYER_SETUP_FAILED                           = 1103,
   SNPE_UDL_EXECUTE_FAILED                               = 1104,
   SNPE_UDL_BUNDLE_INVALID                               = 1105,

   // Dependent library errors
   SNPE_STD_LIBRARY_ERROR                                = 1200,

   // Unknown exception (catch (...)), Has no component attached to this
   SNPE_UNKNOWN_EXCEPTION                                = 1210,

   // Storage Errors
   SNPE_STORAGE_INVALID_KERNEL_REPO                      = 1300,

};


/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * Returns the error code of the last error encountered.
 *
 * @return The error code.
 *
 * @note The returned error code is significant only when the return
 *       value of the call indicated an error.
 */
ZDL_EXPORT ErrorCode getLastErrorCode();

/**
 * Returns the error string of the last error encountered.
 *
 * @return The error string.
 *
 * @note The returned error string is significant only when the return
 *       value of the call indicated an error.
 */
ZDL_EXPORT const char* getLastErrorString();

/**
 * Returns the uint32_t representation of the error code enum.
 *
 * @param[in] code The error code to be converted.
 *
 * @return uint32_t representation of the error code.
 */
ZDL_EXPORT uint32_t enumToUInt32(zdl::DlSystem::ErrorCode code);

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

} // DlSystem
} // zdl

#endif // _DL_ERROR_HPP_

