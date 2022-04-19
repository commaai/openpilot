//=============================================================================
//
//  Copyright (c) 2016-2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#ifndef _DL_SYSTEM_IUDL_HPP_
#define _DL_SYSTEM_IUDL_HPP_

#include "ZdlExportDefine.hpp"

namespace zdl {
namespace DlSystem {

/**
 * NOTE: DEPRECATED, MAY BE REMOVED IN THE FUTURE.
 *
 * @brief .
 *
 * Base class user concrete UDL implementation.
 *
 * All functions are marked as:
 *
 * - virtual
 * - noexcept
 *
 * User should make sure no exceptions are propagated outside of
 * their module. Errors can be communicated via return values.
 */
class ZDL_EXPORT IUDL {
public:
   /**
    * NOTE: DEPRECATED, MAY BE REMOVED IN THE FUTURE.
    *
    * @brief .
    *
    * Destructor
    */
   virtual ~IUDL() = default;

   /**
    * NOTE: DEPRECATED, MAY BE REMOVED IN THE FUTURE.
    *
    * @brief Sets up the user's environment.
    * This is called by the SNPE framework to allow the user the
    * opportunity to setup anything which is needed for running
    * user defined layers.
    *
    * @param cookie User provided opaque data returned by the SNPE
    *               runtime
    *
    * @param insz How many elements in input size array
    * @param indim Pointer to a buffer that holds input dimension
    *               array
    * @param indimsz Input dimension size  array of the buffer
    *                 'indim'. Corresponds to indim
    *
    * @param outsz How many elements in output size array
    * @param outdim Pointer to a buffer that holds output
    *              dimension array
    * @param outdimsz Output dimension size of the buffer 'oudim'.
    *                  Corresponds to indim
    *
    * @return true on success, false otherwise
    */
   virtual bool setup(void *cookie,
                      size_t insz, const size_t **indim, const size_t *indimsz,
                      size_t outsz, const size_t **outdim, const size_t *outdimsz)  = 0;

   /**
    * NOTE: DEPRECATED, MAY BE REMOVED IN THE FUTURE.
    *
    * @brief Close the instance. Invoked by the SNPE
    * framework to allow the user the opportunity to release any resources
    * allocated during setup.
    *
    * @param cookie - User provided opaque data returned by the SNPE runtime
    */
   virtual void close(void *cookie) noexcept = 0;

   /**
    * NOTE: DEPRECATED, MAY BE REMOVED IN THE FUTURE.
    *
    * @brief Execute the user defined layer
    *
    * @param cookie User provided opaque data returned by the SNPE 
    *               runtime
    *
    * @param input Const pointer to a float buffer that contains
    *               the input
    *
    * @param output Float pointer to a buffer that would hold
    *                 the user defined layer's output. This buffer
    *                 is allocated and owned by SNPE runtime.
    */
   virtual bool execute(void *cookie, const float **input, float **output)  = 0;
};

} // ns DlSystem

} // ns zdl

#endif // _DL_SYSTEM_IUDL_HPP_
