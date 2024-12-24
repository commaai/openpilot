//==============================================================================
//
// Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef SNPE_UDO_REG_H
#define SNPE_UDO_REG_H

#include "SnpeUdo/UdoShared.h"

#ifdef __cplusplus
extern "C"
{
#endif

/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * @brief Initialize the shared library's data structures. Calling any other
 *        library function before this one will result in an error being returned.
 *
 * @return Error code
 */
SnpeUdo_ErrorType_t
SnpeUdo_initRegLibrary(void);

typedef SnpeUdo_ErrorType_t
(*SnpeUdo_InitRegLibraryFunction_t)(void);

/**
 * @brief A function to query the API version of the UDO registration library.
 *        The function populates a SnpeUdo_LibVersion_t struct, which contains a SnpeUdo_Version_t
 *        struct for API version and library version.
 *
 * @param[in, out] version A pointer to struct which contains major, minor, teeny information for
 *                 library and api versions.
 *
 * @return Error code
 */
SnpeUdo_ErrorType_t
SnpeUdo_getRegLibraryVersion(SnpeUdo_LibVersion_t** version);

typedef SnpeUdo_ErrorType_t
(*SnpeUdo_getRegLibraryVersion_t)(SnpeUdo_LibVersion_t** version);

/**
 * @brief Release the shared library's data structures, and invalidate any
 *        handles returned by the library. The behavior of any outstanding
 *        asynchronous calls made to this library when this function is called
 *        are undefined. All library functions (except SnpeUdo_InitRegLibrary) will
 *        return an error after this function has been successfully called.
 *
 *        It should be possible to call SnpeUdo_InitRegLibrary after calling this
 *        function, and re-initialize the library.
 *
 * @return Error code
 */
SnpeUdo_ErrorType_t
SnpeUdo_terminateRegLibrary(void);

typedef SnpeUdo_ErrorType_t
(*SnpeUdo_TerminateRegLibraryFunction_t)(void);


/**
 * @brief A function to query the info on the UDO set.
 *        The function populates a structure which contains information about
 *        the package and operations contained in it.
 *
 * @param[in, out] registrationInfo A struct which contains information on the set of UDOs
 *
 * @return Error code
 *
 */
SnpeUdo_ErrorType_t
SnpeUdo_getRegInfo(SnpeUdo_RegInfo_t** registrationInfo);

typedef SnpeUdo_ErrorType_t
(*SnpeUdo_GetRegInfoFunction_t)(SnpeUdo_RegInfo_t** registrationInfo);

/**
 * @brief A function to validate that a set of params is supported by an operation
 *        The function receives an operation definition struct, and returns if this configuration is
 *        supported (e.g. if an operation can be created using this configuration)
 *
 * @param[in] opDefinition A struct of SnpeUdo_OpDefinition type, containing the information needed to
 *            validate that an operation can be created with this configuration.
 *
 * @return Error code, indicating is the operation can be created on this set or not.
 *
 */
SnpeUdo_ErrorType_t
SnpeUdo_validateOperation(SnpeUdo_OpDefinition_t* opDefinition);

typedef SnpeUdo_ErrorType_t
(*SnpeUdo_ValidateOperationFunction_t)(SnpeUdo_OpDefinition_t* opDefinition);

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

#ifdef __cplusplus
} // extern "C"
#endif

#endif //SNPE_UDO_REG_H
