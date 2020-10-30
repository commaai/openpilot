//==============================================================================
//
// Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef SNPE_UDO_SHARED_H
#define SNPE_UDO_SHARED_H

#include "SnpeUdo/UdoBase.h"

#ifdef __cplusplus
extern "C"
{
#endif

/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * @brief A function to return the various versions as they relate to the UDO
 *        The function returns a struct containing the the following:
 *        libVersion: the version of the implementation library compiled for the UDO. Set by user
 *        apiVersion: the version of the UDO API used in compiling the implementation library.
 *        Set by SNPE
 *
 * @param[in, out] version A pointer to Version struct of type SnpeUdo_LibVersion_t
 *
 * @return Error code
 *
 */
SnpeUdo_ErrorType_t
SnpeUdo_getVersion (SnpeUdo_LibVersion_t** version);

typedef SnpeUdo_ErrorType_t
(*SnpeUdo_GetVersionFunction_t) (SnpeUdo_LibVersion_t** version);

#ifdef __cplusplus
} // extern "C"
#endif

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

#endif // SNPE_UDO_SHARED_H
