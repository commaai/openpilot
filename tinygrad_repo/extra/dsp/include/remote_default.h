/*-----------------------------------------------------------------------------
   Copyright (c) 2012-2014,2016,2017,2019-2021 QUALCOMM Technologies, Incorporated.
   All Rights Reserved.
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

   3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
-----------------------------------------------------------------------------*/
#ifndef REMOTE_DEFAULT_H
#define REMOTE_DEFAULT_H

#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @file remote.h
 *  @brief Header file with APIs to interface with FastRPC.
 */

typedef uint32_t remote_handle;        /**< Handle used by non-domain modules */
typedef uint64_t remote_handle64;      /**< Handle used by multi-domain modules */
typedef uint64_t fastrpc_async_jobid;  /**< Job ID filled in the remote_handle_invoke_async() call*/

/**
 * @struct remote_buf
 * @brief remote buffer for arguments.
 */
typedef struct {
   void *pv;    /**< Buffer pointer for input/output arguments */
   size_t nLen; /**< Buffer size for input/output arguments */
} remote_buf;

/**
 * @struct remote_dma_handle
 * @brief remote handle  structure for passing fd.
 */
typedef struct {
   int32_t fd;       /**< File Desecripter for buffer */
   uint32_t offset;  /**< Buffer offset */
} remote_dma_handle;

/**
 * @union remote_arg
 * @brief union of different types of arguments that are passed to the remote invocation call
 */
typedef union {
   remote_buf     buf;    /**< Buffer holds pointer and size for given argument */
   remote_handle    h;    /**< Handle for non-domains modules */
   remote_handle64 h64;   /**< Handle for multi-omains modules */
   remote_dma_handle dma; /**< DMA Handle that holds buffer fd and offset */
} remote_arg;

/**
 * @enum fastrpc_async_notify_type
 * @brief Types of Asynchronous complete notifications
 */
enum fastrpc_async_notify_type {
   FASTRPC_ASYNC_NO_SYNC = 0,   /**< synchronous call */
   FASTRPC_ASYNC_CALLBACK,      /**< asynchronous call with callback response */
   FASTRPC_ASYNC_POLL,          /**< asynchronous call with polling response */
   FASTRPC_ASYNC_TYPE_MAX,      /**< reserved */
};

/**
 * @struct fastrpc_async_callback
 * @brief call back function and context for asynchronous complete notification
 */
typedef struct fastrpc_async_callback {
   void (*fn)(fastrpc_async_jobid jobid, void* context, int result); /**< call back function */
   void *context;                                                    /**< unique context filled by user */
}fastrpc_async_callback_t;

/**
 * @struct fastrpc_async_descriptor
 * @brief  descriptor for defining the complete notification mechanism for given remote invocation call.
 */

typedef struct fastrpc_async_descriptor {
   enum fastrpc_async_notify_type type; /**< asynchronous notification type */
   fastrpc_async_jobid jobid;           /**< Job ID returned in async remote invocation call */
   union {
      fastrpc_async_callback_t cb;      /**< call back function filled by user */
   };
}fastrpc_async_descriptor_t;


/**
 * Retrieves method attribute from the scalars parameter
 */
#define REMOTE_SCALARS_METHOD_ATTR(dwScalars)   (((dwScalars) >> 29) & 0x7)

/**
 * Retrieves method index from the scalars parameter
 */
#define REMOTE_SCALARS_METHOD(dwScalars)        (((dwScalars) >> 24) & 0x1f)

/**
 * Retrieves number of input buffers from the scalars parameter
 */
#define REMOTE_SCALARS_INBUFS(dwScalars)        (((dwScalars) >> 16) & 0x0ff)

/**
 * Retrieves number of output buffers from the scalars parameter
 */
#define REMOTE_SCALARS_OUTBUFS(dwScalars)       (((dwScalars) >> 8) & 0x0ff)

/**
 * Retrieves number of input handles from the scalars parameter
 */
#define REMOTE_SCALARS_INHANDLES(dwScalars)     (((dwScalars) >> 4) & 0x0f)

/**
 * Retrieves number of output handles from the scalars parameter
 */
#define REMOTE_SCALARS_OUTHANDLES(dwScalars)    ((dwScalars) & 0x0f)

/**
 * Internal scalars parameter construction for given nAttr, nMethod, nIn, nOut, noIn and noOut
 */
#define REMOTE_SCALARS_MAKEX(nAttr,nMethod,nIn,nOut,noIn,noOut) \
          ((((uint32_t)   (nAttr) &  0x7) << 29) | \
           (((uint32_t) (nMethod) & 0x1f) << 24) | \
           (((uint32_t)     (nIn) & 0xff) << 16) | \
           (((uint32_t)    (nOut) & 0xff) <<  8) | \
           (((uint32_t)    (noIn) & 0x0f) <<  4) | \
            ((uint32_t)   (noOut) & 0x0f))

/**
 * Internal scalars parameter construction for given nAttr, nMethod, nIn, nOut
 */
#define REMOTE_SCALARS_MAKE(nMethod,nIn,nOut)  REMOTE_SCALARS_MAKEX(0,nMethod,nIn,nOut,0,0)

/**
 * Internal total number of different arguments for given scalars parameter
 */
#define REMOTE_SCALARS_LENGTH(sc) (REMOTE_SCALARS_INBUFS(sc) +\
                                   REMOTE_SCALARS_OUTBUFS(sc) +\
                                   REMOTE_SCALARS_INHANDLES(sc) +\
                                   REMOTE_SCALARS_OUTHANDLES(sc))

/**
 * Internal macro for function definition
 */
#ifndef __QAIC_REMOTE

#define __QAIC_REMOTE(ff) ff
#endif //__QAIC_REMOTE

/**
 * Internal macro for function declaration
 */
#ifndef __QAIC_REMOTE_EXPORT
#ifdef _WIN32
#define __QAIC_REMOTE_EXPORT __declspec(dllexport)
#else //_WIN32
#define __QAIC_REMOTE_EXPORT
#endif //_WIN32
#endif //__QAIC_REMOTE_EXPORT

/**
 * Internal macro for function declaration
 */
#ifndef __QAIC_REMOTE_ATTRIBUTE
#define __QAIC_REMOTE_ATTRIBUTE
#endif

/**
 * Number of Remote Domains
 */
#define NUM_DOMAINS 4

/**
 * Number of sessions for given process
 */
#define NUM_SESSIONS 2

/**
 * Internal Mask to retrieve domain
 */
#define DOMAIN_ID_MASK 3

/**
 * Default remote process domain
 */
#ifndef DEFAULT_DOMAIN_ID
#define DEFAULT_DOMAIN_ID 0
#endif

/**
 * Internal remote process domain ID definition for ADSP
 */
#define ADSP_DOMAIN_ID    0

/**
 * Internal remote process domain ID definition MDSP
 */
#define MDSP_DOMAIN_ID    1

/**
 * Internal remote process domain ID definition for SDSP
 */
#define SDSP_DOMAIN_ID    2

/**
 * Internal remote process domain ID definition for CDSP
 */
#define CDSP_DOMAIN_ID    3

/**
 * Internal ADSP remote process definition to check remote process domain from uri
 */
#define ADSP_DOMAIN "&_dom=adsp"

/**
 * Internal MDSP remote process definition to check remote process domain from uri
 */
#define MDSP_DOMAIN "&_dom=mdsp"

/**
 * Internal SDSP remote process definition to check remote process domain from uri
 */
#define SDSP_DOMAIN "&_dom=sdsp"

/**
 * Internal CDSP remote process definition to check remote process domain from uri
 */
#define CDSP_DOMAIN "&_dom=cdsp"

/**Process Types
 * Return values for FASTRPC_REMOTE_PROCESS_TYPE in enum session_control_req_id for remote_handle_control
 * Return values denote the type of process on remote subsystem
 */
enum fastrpc_process_type {
  PROCESS_TYPE_SIGNED   = 0,
  PROCESS_TYPE_UNSIGNED = 1,
};

/**
 * Loads the shared object in remote process and returns the handle to it
 * This API used for non-domains use case
 *
 * @param[in] name pointer to name of the shared object
 * @param[out] ph pointer to handle
 *
 * @return 0 for success and non-zero for failure.\n
 * Expected error code due to incorrect input arguments is AEE_EBADPARM. Other than this error code,
 * treated as returned from fastRPC framework issues.
*/
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(remote_handle_open)(const char* name, remote_handle *ph) __QAIC_REMOTE_ATTRIBUTE;

/**
 * Loads the shared object in remote process and returns the handle to it
 * This API used for domains use case
 *
 * @param[in] name pointer to name of the shared object
 * @param[out] ph pointer to handle
 *
 * @return 0 for success and non-zero for failure.\n
 * Expected error code due to incorrect input arguments is AEE_EBADPARM. Other than this error code,
 * treated as returned from fastRPC framework issues.
*/
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(remote_handle64_open)(const char* name, remote_handle64 *ph) __QAIC_REMOTE_ATTRIBUTE;

/**
 * Makes remote call for given handle synchronously
 * This API used for non-domains use case
 *
 * @param[in]  h successful handle returned via #remote_handle_open()
 * @param[in]  pra pointer to union #remote_arg \n
 * A sequence of arguments in the following order inbufs, outbufs, inhandles, outhandles.
 * @param[in]  dwScalars integer packs Method ID, number of inputs, outputs, inhandles and outhandles
 * @return Integer value. Zero for success and non-zero for failure. For failure, returns error code.\n
 * Expected error code due to incorrect input arguments are AEE_EINVHANDLE, AEE_EBADPARM. Other than these error codes,
 * treated as returned from fastRPC framework issues.
 */
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(remote_handle_invoke)(remote_handle h, uint32_t dwScalars, remote_arg *pra) __QAIC_REMOTE_ATTRIBUTE;

/**
 * Makes remote call for given handle synchronously
 * This API used for domains use case
 *
 * @param[in]  h successful handle returned via #remote_handle_open()
 * @param[in]  pra pointer to union #remote_arg \n
 * A sequence of arguments in the following order inbufs, outbufs, inhandles, outhandles.
 * @param[in]  dwScalars integer packs Method ID, number of inputs, outputs, inhandles and outhandles
 * @return Integer value. Zero for success and non-zero for failure. For failure, returns error code.\n
 * Expected error code due to incorrect input arguments are AEE_EINVHANDLE, AEE_EBADPARM. Other than these error codes,
 * treated as returned from fastRPC framework issues.
 */
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(remote_handle64_invoke)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra) __QAIC_REMOTE_ATTRIBUTE;

/**
 * Makes remote call for given handle asynchronously
 * This API used for non-domains use case
 *
 * @param[in]  h successful handle returned via #remote_handle_open()
 * @param[in]  desc pointer to structure #fastrpc_async_descriptor_t
 * @param[in]  pra pointer to union #remote_arg \n
 * sequence of arguments in following order input buffers, output\n
 * buffers, in handles and out handles. The number of each type of argument\n
 * given in dwScalars below.
 * @param[in]  dwScalars \n
 * Sequence of the following integers related to invocation: The method ID\
 * the number of input buffers in pra, the number of out buffers in pra,
 * the number of in handles in pra and the number out handles in pra.
 * @return Integer value. Zero for success and non-zero for failure. For failure, returns error code.\n
 * Expected error codes due to incorrect input arguments are AEE_EINVHANDLE, AEE_EBADPARM. Other than these error codes,
 * treated as returned from fastRPC framework issues.
 */
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(remote_handle_invoke_async)(remote_handle h, fastrpc_async_descriptor_t *desc, uint32_t dwScalars, remote_arg *pra) __QAIC_REMOTE_ATTRIBUTE;

/**
 * Makes remote call for given handle asynchronously
 * This API used for domains use case
 *
 * @param[in]  h successful handle returned via #remote_handle_open()
 * @param[in]  desc pointer to structure #fastrpc_async_descriptor_t
 * @param[in]  pra pointer to union #remote_arg \n
 * sequence of arguments in following order input buffers, output\n
 * buffers, in handles and out handles. The number of each type of argument\n
 * given in dwScalars below.
 * @param[in]  dwScalars \n
 * Sequence of the following integers related to invocation: The method ID\
 * the number of input buffers in pra, the number of out buffers in pra,
 * the number of in handles in pra and the number out handles in pra.
 * @return Integer value. Zero for success and non-zero for failure. For failure, returns error code.\n
 * Expected error codes due to incorrect input arguments are AEE_EINVHANDLE, AEE_EBADPARM. Other than these error codes,
 * treated as returned from fastRPC framework issues.
 */
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(remote_handle64_invoke_async)(remote_handle64 h, fastrpc_async_descriptor_t *desc, uint32_t dwScalars, remote_arg *pra) __QAIC_REMOTE_ATTRIBUTE;

/**
 * Unloads the shared object for given handle
 * This API used for non-domains use case
 *
 * @param[in]  h successful handle returned via #remote_handle_open()
 * @return Integer value. Zero for success and non-zero for failure. For failure, returns error code.\n
 * Expected error code due to incorrect input arguments is AEE_EINVHANDLE. Other than this error code,
 * treated as returned from fastRPC framework issues.
 */
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(remote_handle_close)(remote_handle h) __QAIC_REMOTE_ATTRIBUTE;

/**
 * Unloads the shared object for given handle
 * This API used for domains use case
 *
 * @param[in]  h successful handle returned via #remote_handle_open()
 * @return Integer value. Zero for success and non-zero for failure. For failure, returns error code.\n
 * Expected error code due to incorrect input arguments is AEE_EINVHANDLE. Other than this error code,
 * treated as returned from fastRPC framework issues.
 */
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(remote_handle64_close)(remote_handle64 h) __QAIC_REMOTE_ATTRIBUTE;

/**
 * @enum handle_control_req_id
 * @brief different types of requests that are passed to the remote handle control API
 */
enum handle_control_req_id {
    DSPRPC_CONTROL_LATENCY  =   1, /**< fastrpc latency control */
    DSPRPC_GET_DSP_INFO     =   2, /**< Query DSP capabilities */
    DSPRPC_CONTROL_WAKELOCK =   3, /**< Enable or disable wake lock  */
    DSPRPC_GET_DOMAIN       =   4, /**< getting the domain */
};

/**
 * @enum remote_rpc_latency_flags
 * @brief Types of QOS allowed with request ID DSPRPC_CONTROL_LATENCY

 */
typedef enum remote_rpc_latency_flags {
   RPC_DISABLE_QOS = 0,   /**< Disable PM QOS. */
   RPC_PM_QOS = 1,        /**< Enable PM QOS */
   RPC_ADAPTIVE_QOS = 2,  /**< Enable Adaptive QOS */
   RPC_POLL_QOS = 3,      /**< Enable Poll QOS */
} remote_rpc_control_latency_t;

/**
 * @struct remote_rpc_control_latency
 * @brief Argument for setting latency with request ID DSPRPC_CONTROL_LATENCY\n
 * PM QoS control cpu low power modes based on RPC activity in 100 ms window. Recommended for latency sensitive use cases.\n
 * Adaptive QoS, DSP driver predicts completion time of a method and send CPU wake up signal to reduce wake up latency
 * Recommended for moderate latency sensitive use cases. It is more power efficient compared to pm_qos control.
 * Poll QoS, After sending invocation to DSP, CPU will enter polling mode instead of waiting for a glink response.
 * This will boost fastrpc performance by reducing the CPU wakeup and scheduling times.
 */
struct remote_rpc_control_latency {
   remote_rpc_control_latency_t enable;           /**< Enable type of QOS one of enum #remote_rpc_latency_flags*/
   uint32_t latency;          /**< Latency is the time between CPU application making the remote call and beginning of remote DSP method execution.
                                   When used with poll QoS, user needs to pass the expected execution time of method on DSP. CPU will poll for a DSP
                                   response for that specified duration after which it will timeout and fall back to waiting for a glink response.  */
};

/**
 * @enum remote_dsp_attributes
 * @brief Different types of DSP capabilities queried via remote_handle_control
 */
enum remote_dsp_attributes {
   DOMAIN_SUPPORT              = 0,                               /**< Domains feature support*/
   UNSIGNED_PD_SUPPORT         = 1,                               /**< Unsigned PD feature support*/
   HVX_SUPPORT_64B             = 2,                               /**< Number of HVX 64B support*/
   HVX_SUPPORT_128B            = 3,                               /**< Number of HVX 128B support*/
   VTCM_PAGE                   = 4,                               /**< Max page size allocation possible in VTCM */
   VTCM_COUNT                  = 5,                               /**< Number of page_size blocks available */
   ARCH_VER                    = 6,                               /**< Hexagon processor architecture version */
   HMX_SUPPORT_DEPTH           = 7,                               /**< HMX Support Depth */
   HMX_SUPPORT_SPATIAL         = 8,                               /**< HMX Support Spatial */
   ASYNC_FASTRPC_SUPPORT       = 9,                               /**< Async FastRPC Support */
   STATUS_NOTIFICATION_SUPPORT = 10,                              /**< DSP User PD status notification Support */
   FASTRPC_MAX_DSP_ATTRIBUTES  = STATUS_NOTIFICATION_SUPPORT + 1, /**< Number of attributes supported */
};

/**
 * @struct remote_dsp_capability
 * @brief Argument to query DSP capability with request ID DSPRPC_GET_DSP_INFO
 */
typedef struct remote_dsp_capability {
   uint32_t domain;       /**<  Remote domain ID */
   uint32_t attribute_ID; /**<  One of the DSP capabilities from enum #remote_dsp_attributes */
   uint32_t capability;   /**<  Output is capability and possible values vary based on attribute_ID
   For DOMAIN_SUPPORT, output values are one, zero.
   For UNSIGNED_PD_SUPPORT, output values are one, zero
   For, HVX_SUPPORT_64B, output is number of HVX 64B supported
   For, HVX_SUPPORT_128B, output is number HVX 128B supported
   For VTCM_PAGE, output is Max page size allocation possible in VTCM
   For VTCM_COUNT, output is Number of page_size blocks available
   For ARCH_VER, output is Hexagon processor architecture version
   For HMX_SUPPORT, output is one or zero
   For ASYNC_FASTRPC_SUPPORT, output is one or zero*/
} fastrpc_capability;

/** Macro for backward compatbility. Clients can compile wakelock request code
 * in their app only when this is defined */
#define FASTRPC_WAKELOCK_CONTROL_SUPPORTED 1

/**
 * @struct remote_rpc_control_wakelock
 * @brief Argument for enable/disable wake lock with request ID DSPRPC_CONTROL_WAKELOCK\n
 * CPU can go into suspend mode anytime. For clients who want to keep the CPU awake until they get a response for their
 * remote invocation call, recommended to use wake lock feature.

 */
struct remote_rpc_control_wakelock {
    uint32_t enable;    /**< enable control of wake lock */
};

/**
 * @struct remote_rpc_get_domain
 * @brief Argument to get domain from handle with request ID DSPRPC_GET_DOMAIN
 */
typedef struct remote_rpc_get_domain {
   int domain;         /**< Domain ID*/
 } remote_rpc_get_domain_t;

/**
 * remote_handle64_control API allows to enable features based on handle
 *
 * @param[in]  req  one of enum from #handle_control_req_id\n
 * Req ID #DSPRPC_CONTROL_LATENCY is used for enable PM QoS, adaptive QoS\n
 * Req ID #DSPRPC_GET_DSP_INFO is used for query types of features supported\n
 * Req ID #DSPRPC_CONTROL_WAKELOCK is used for enable/disable wake lock\n
 * Req ID #DSPRPC_GET_DOMAIN is used for getting the domain ID for given handle\n
 *
 * @param[in,out] data  void pointer to struct for input params\n
 * Inputs\n
 * struct #remote_rpc_control_latency is used for enable PM QoS, adaptive QoS \n
 * struct #remote_dsp_capability is used for query features supported or not. Listed in #remote_dsp_attributes \n
 * struct #remote_rpc_control_wakelock is used for enable/disable wake lock \n
 * struct #remote_rpc_get_domain is used for getting the domain for given handle \n
 * outputs\n
 * For req #DSPRPC_CONTROL_LATENCY, output is same as API return value \n
 * For req #DSPRPC_GET_DSP_INFO, output is capability argument from #remote_dsp_capability \n
 * For req #DSPRPC_CONTROL_WAKELOCK, output is  same as API return value \n
 * For req #DSPRPC_GET_DOMAIN, output is domain argument from #remote_rpc_get_domain \n
 *
 * @param[in] datalen Length of the struct defintion used for input params
 *
 * @return Integer value. Zero for success and non-zero for failure. For failure, returns error code.\n
 * Expected error codes due to incorrect input arguments are AEE_EBADPARM, AEE_EUNSUPPORTED.
 * Other than these error codes, treated as returned from fastRPC framework issues.
 *
 * For example:
 * @code
 * struct remote_rpc_control_latency data;
 * data.enable = 1;
 * data.latency = 100;
 *
 * if (remote_handle64_control)
 *    remote_handle64_control(h, DSPRPC_CONTROL_LATENCY, (void*)&data, sizeof(data));
 * else
 *    printf("remote_handle64_control not available on this target");
 * @endcode
*/
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(remote_handle_control)(uint32_t req, void* data, uint32_t datalen) __QAIC_REMOTE_ATTRIBUTE;

/**
 * remote_handle64_control API allows to enable features based on handle
 * This API used for domains use case
 *
 * @param[in] h successful handle returned via #remote_handle_open() \n
 * @param[in] req one of enum from #handle_control_req_id \n
 * Req ID #DSPRPC_CONTROL_LATENCY is used for enable PM QoS, adaptive QoS\n
 * Req ID #DSPRPC_GET_DSP_INFO is used for query types of features supported\n
 * Req ID #DSPRPC_CONTROL_WAKELOCK is used for enable/disable wake lock\n
 * Req ID #DSPRPC_GET_DOMAIN is used for getting the domain ID for given handle\n
 *
 * @param[in,out] data  void pointer to struct for input params\n
 * Inputs\n
 * struct #remote_rpc_control_latency is used for enable PM QoS, adaptive QoS \n
 * struct #remote_dsp_capability is used for query features supported or not. Listed in #remote_dsp_attributes \n
 * struct #remote_rpc_control_wakelock is used for enable/disable wake lock \n
 * struct #remote_rpc_get_domain is used for getting the domain for given handle \n
 * outputs\n
 * For req #DSPRPC_CONTROL_LATENCY, output is same as API return value \n
 * For req #DSPRPC_GET_DSP_INFO, output is capability argument from #remote_dsp_capability \n
 * For req #DSPRPC_CONTROL_WAKELOCK, output is  same as API return value \n
 * For req #DSPRPC_GET_DOMAIN, output is domain argument from #remote_rpc_get_domain \n
 *
 * @param[in] datalen Length of the struct defintion used for input params
 *
 * @return Integer value. Zero for success and non-zero for failure. For failure, returns error code.\n
 * Expected error codes due to incorrect input arguments are AEE_EBADPARM, AEE_EINVHANDLE, AEE_EUNSUPPORTED
 * Other than these error codes, treated as returned from fastRPC framework issues.
 *
 * For example:
 * @code
 * struct remote_rpc_control_latency data;
 * data.enable = 1;
 * data.latency = 100;
 *
 * if (remote_handle64_control)
 *    remote_handle64_control(h, DSPRPC_CONTROL_LATENCY, (void*)&data, sizeof(data));
 * else
 *    printf("remote_handle64_control not available on this target");
 * @endcode
*/
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(remote_handle64_control)(remote_handle64 h, uint32_t req, void* data, uint32_t datalen) __QAIC_REMOTE_ATTRIBUTE;

/**
 * @enum session_control_req_id
 * @brief different types of request ID that are passed to the remote session control
 */
enum session_control_req_id {
  FASTRPC_THREAD_PARAMS             = 1,      /**< thread params */
  DSPRPC_CONTROL_UNSIGNED_MODULE    = 2,      /**< unsigned PD */
  FASTRPC_RELATIVE_THREAD_PRIORITY  = 4,      /**< relative thread priority */
  FASTRPC_REMOTE_PROCESS_KILL       = 6,      /**< Kill remote process */
  FASTRPC_SESSION_CLOSE             = 7,      /**< Close all open handles of requested domain */
  FASTRPC_CONTROL_PD_DUMP           = 8,      /**< Enable PD dump feature */
  FASTRPC_REMOTE_PROCESS_EXCEPTION  = 9,      /**< Trigger Exception in the remote process */
  FASTRPC_REMOTE_PROCESS_TYPE       = 10,     /**< Query type of process defined by enum fastrpc_process_type */
  FASTRPC_REGISTER_STATUS_NOTIFICATIONS = 11, /**< Enable DSP User process status notifications */
};

/**
 * @struct remote_rpc_thread_params
 * @brief Argument for setting threads params with request ID FASTRPC_THREAD_PARAMS
 */
struct remote_rpc_thread_params {
    int domain;         /**< Remote subsystem domain ID, pass -1 to set params for all domains */
    int prio;           /**< user thread priority (1 to 255), pass -1 to use default */
    int stack_size;     /**< user thread stack size in bytes, pass -1 to use default */
};

/**
 * @struct remote_rpc_control_unsigned_module
 * @brief Argument for setting the unsigned PD with req ID:DSPRPC_CONTROL_UNSIGNED_MODULE
 */
struct remote_rpc_control_unsigned_module {
   int domain;              /**<  Remote subsystem domain ID, -1 to set params for all domains */
   int enable;              /**<  Non-zero value to set unsigned PD */
};

/**
 * @struct remote_rpc_relative_thread_priority
 * @brief Argument for setting the relative thread priority with request id:FASTRPC_RELATIVE_THREAD_PRIORITY
 */
struct remote_rpc_relative_thread_priority {
    int domain;                     /**< Remote subsystem domain ID, pass -1 to update priority for all domains */
    int relative_thread_priority;   /**< Relative thread priority increased w.r.t to default priority. Negative value will increase priority. Positive value will decrease priority */
};

/**
 * @struct remote_rpc_process_clean_params
 * @brief Argument for cleaning remote session on DSP with request id:FASTRPC_REMOTE_PROCESS_KILL
 */
struct remote_rpc_process_clean_params {
   int domain;          /**< Remote subsystem domain ID, domain ID of process to recover */
};

/**
 * @struct remote_rpc_session_close
 * @brief Argument for closing all handles for all domains with request id:FASTRPC_SESSION_CLOSE
 */
struct remote_rpc_session_close {
   int domain;          /**< Remote subsystem domain ID, pass -1 to close all handles for all domains */
};

/**
 * @struct remote_rpc_control_pd_dump
 * @brief Argument for enabling PD dump on User PD of DSP with request id:FASTRPC_CONTROL_PD_DUMP
 */
struct remote_rpc_control_pd_dump {
   int domain;          /**< Remote subsystem domain ID, pass -1 to enable PD dump on all domains */
   int enable;          /**<  1 to enable PD dump, 0 to disable PD dump */
};

/**
 * @struct remote_process_type
 * Structure for remote_session_control, used with FASTRPC_REMOTE_PROCESS_TYPE request ID to query the type of PD runs on, defined by enum fastrpc_process_type
 * @param[in] : Domain of process
 * @param[out]: process_type belonging to enum fastrpc_process_type
 */
struct remote_process_type {
   int domain;
   int process_type;
};

/**
 * @struct remote_rpc_process_clean_params
 * @brief Argument to trigger exception in the User PD of DSP with request id:FASTRPC_REMOTE_PROCESS_EXCEPTION
 */
typedef struct remote_rpc_process_clean_params remote_rpc_process_exception;

/**
 * @enum remote_rpc_status_flags
 * @brief different types of DSP User PD status notification flags
 */
typedef enum remote_rpc_status_flags {
   FASTRPC_USER_PD_UP          = 0,    /**< DSP user process is up */
   FASTRPC_USER_PD_EXIT        = 1,    /**< DSP user process exited */
   FASTRPC_USER_PD_FORCE_KILL  = 2,    /**< DSP user process forcefully killed. Happens when DSP resources needs to be freed. */
   FASTRPC_USER_PD_EXCEPTION   = 3,    /**< Exception in the user process of DSP. */
   FASTRPC_DSP_SSR             = 4,    /**< Subsystem restart of the DSP, where user process is running. */
} remote_rpc_status_flags_t;

/**
 * Notification call back function
 * @param[in] context used in the registration
 * @param[in] domain of the user process
 * @param[in] session id of user process
 * @param[in] status of user process
 * @return Integer value. Zero for success and non-zero for failure.
 */
typedef int (*fastrpc_notif_fn_t)(void *context, int domain, int session, remote_rpc_status_flags_t status);

/**
 * @struct remote_rpc_notif_register
 * @brief Argument to receive status notifications of user process on DSP, with request ID DSPRPC_REGISTER_STATUS_NOTIFICATIONS
 */
typedef struct remote_rpc_notif_register {
    void *context;                  /**< Context of the client */
    int domain;                     /**< Remote domain ID */
    fastrpc_notif_fn_t notifier_fn; /**< Notification call back function pointer */
} remote_rpc_notif_register_t;

/**
 *  remote_session_control API allows configure the remote session params

 * @param[in]  req is one of enum from #session_control_req_id \n
 * Req FASTRPC_THREAD_PARAMS is used for setting the theread stack size and priority \n
 * Req DSPRPC_CONTROL_UNSIGNED_MODULE used for setting the unsigned PD \n
 * Req FASTRPC_RELATIVE_THREAD_PRIORITY used for setting the relative thread priority \n
 *
 * @param[in]  data void pointer to struct for input params \n
 * struct #remote_rpc_thread_params is used for setting the thread priority and stack size \n
 * struct #remote_rpc_control_unsigned_module is used for setting the unsigned PD \n
 * struct #remote_rpc_relative_thread_priority is used for setting the relative thread priority \n
 *
 * @param[in] datalen is length of the struct for input params \n
 *
 * @return Integer value. Zero for succesfully setting the remote session params. Non-zero for failure. \n
 * Expected error codes due to incorrect input arguments are AEE_EBADPARM, AEE_EUNSUPPORTED
 * Other than this error codes, treated as returned from fastRPC framework issues.
 * For example:
 * @code
 * if (remote_session_control)
 * {
 *    struct remote_rpc_control_unsigned_module data;
 *
 *    data.enable = 1;
 *    data.domain = CDSP_DOMAIN_ID;
 *    retVal = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void*)&data, sizeof(data));
 *    printf("remote_session_control returned %d for configuring unsigned PD", retVal);
 * }
 * else{
 *    printf("Unsigned PD not supported on this device.\n");
 * }
 * @endcode
*/
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(remote_session_control)(uint32_t req, void *data, uint32_t datalen) __QAIC_REMOTE_ATTRIBUTE;

/**
 *  remote_mmap API maps the pages on remote domain
 *
 * @param[in] fd  file descriptor for given buffer
 * @param[in] flags  one of enum from #remote_mem_map_flags.
 * @param[in] vaddrin HLOS address for given buffer
 * @param[in] size Length of the given buffer
 * @param[out] vaddrout virtual address on remote domain
 * @return Integer value. Zero for success and non-zero for failure. For failure, returns error code.\n
 * Expected error code due to incorrect input arguments is AEE_EBADPARM.
 * Other than these error codes, treated as returned from fastRPC framework issues.
 * For example:
 * @code
 * int nErr = 0
 * if (remote_mem_map) {
 *    nErr = remote_mem_map(domain, fd, flags, buf, size, remoteVirtAddr);
 * } else {
 *    // symbol not found. Fall back to old API
 *    nErr = remote_mmap64(fd, flags, buf, size, remoteVirtAddr);
 * }
 * return nErr;
 * @endcode
*/
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(remote_mmap)(int fd, uint32_t flags, uint32_t vaddrin, int size, uint32_t* vaddrout) __QAIC_REMOTE_ATTRIBUTE;

 /**
 * remote_unmap API unmaps the pages on remote domain
 * @param[in] vaddrout virtual address returned via #remote_mmap()
 * @param[in] size Length of the given buffer
 * @return Integer value. Zero for success and non-zero for failure. For failure, returns error code.\n
 * Expected error code due to incorrect input arguments is AEE_EBADPARM.
 * Other than these error codes, treated as returned from fastRPC framework issues.
*/
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(remote_munmap)(uint32_t vaddrout, int size) __QAIC_REMOTE_ATTRIBUTE;

/**
 * @enum remote_mem_map_flags
 * @brief Different set of flags for mapping the buffer on remote process domain\n
 * Create static memory map on remote process with default cache configuration (writeback).
 * Same remoteVirtAddr will be assigned on remote process when fastrpc call made with local virtual address.
 * Map scope:
 * Life time of this mapping is until user unmap using remote_mem_unmap or session close.
 * No reference counts are used. Behavior of mapping multiple times without unmap is undefined.
 * Cache maintenance:
 * Driver clean caches when virtual address passed through RPC calls defined in IDL as a pointer.
 * User is responsible for cleaning cache when remoteVirtAddr shared to DSP and accessed out of fastrpc method invocations on DSP.
 * recommended usage:
 * Map buffers which are reused for long time or until session close. This helps to reduce fastrpc latency.
 * Memory shared with remote process and accessed only by DSP.
 */

enum remote_mem_map_flags {
   REMOTE_MAP_MEM_STATIC      = 0,
   REMOTE_MAP_MAX_FLAG        = REMOTE_MAP_MEM_STATIC + 1,
};

/**
 * Map memory to the remote process on a selected DSP domain
 * @param[in] domain DSP domain ID. Use -1 for using default domain\n
 *          Default domain is selected based on library lib(a/m/s/c)dsprpc.so library linked to application.
 * @param[in] fd file descriptor of memory
 * @param[in] flags one of enum from #remote_mem_map_flags
 * @param[in] virtAddr virtual address of buffer
 * @param[in] size buffer length
 * @param[out] remoteVirtAddr remote process virtual address
 * @return Integer value. Zero for success and non-zero for failure. For failure, returns error code.\n
 * Expected error code due to incorrect input arguments is AEE_EBADPARM
 * Other than these error codes, treated as returned from fastRPC framework issues.
 */
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(remote_mem_map)(int domain, int fd, int flags, uint64_t virtAddr, size_t size, uint64_t* remoteVirtAddr) __QAIC_REMOTE_ATTRIBUTE;

/**
 * Unmap memory to the remote process on a selected DSP domain
 * @param[in] domain DSP domain ID. Use -1 for using default domain. Get domain from multi-domain handle if required.
 * @param[in] remoteVirtAddr virtual address returned via #remote_mem_map()
 * @param[in] size buffer length
 * @return Integer value. Zero for success and non-zero for failure. For failure, returns error code.\n
 * Expected error code due to incorrect input arguments is AEE_EBADPARM
 * Other than this error code, treated as returned from fastRPC framework issues.
 */
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(remote_mem_unmap)(int domain, uint64_t remoteVirtAddr, size_t size) __QAIC_REMOTE_ATTRIBUTE;

/**
 * @enum remote_buf_attributes
 * @brief Types of buffer attributes
 */
enum remote_buf_attributes {
   FASTRPC_ATTR_NON_COHERENT        = 2,    /**< Attribute to map a buffer as DMA non-coherent Driver perform cache maintenance. */
   FASTRPC_ATTR_COHERENT            = 4,    /**< Attribute to map a buffer as DMA coherent Driver skips cache maintenenace.
                                                 It will be ignored if a device is marked as DMA-coherent in device tree. */
   FASTRPC_ATTR_KEEP_MAP            = 8,    /**< Attribute to keep the buffer persistant until unmap is called explicitly */
   FASTRPC_ATTR_NOMAP               = 16,   /**< Attribute for secure buffers to skip  smmu mapping in fastrpc driver */
   FASTRPC_ATTR_FORCE_NOFLUSH       = 32,   /**< Attribute to map buffer such that flush by driver is skipped for that particular buffer.
                                                 Client has to perform cache maintenance. */
   FASTRPC_ATTR_FORCE_NOINVALIDATE  = 64,   /**< Attribute to map buffer such that invalidate by driver is skipped for that particular buffer.
                                                 Client has to perform cache maintenance. */
   FASTRPC_ATTR_TRY_MAP_STATIC      = 128,  /**< Attribute for persistent mapping a buffer to remote DSP
                                                 process during buffer registration with the FastRPC driver.
                                                 This buffer will be automatically mapped during session
                                                 open and unmapped either at session close or buffer
                                                 unregistration. The FastRPC library treis to map buffers
                                                 and ignores error in case of failure. Pre-mapping a buffer    reduces the FastRPC latency. This flag is recommended only
                                                 for buffers used with latency-critical FastRPC calls. */
};

/** Register a file descriptor for a user allocated buffer(Other than rpcmem_alloc). This is only valid on
 * Android with ION allocated memory. Users of fastRPC should register
 * a buffer allocated with ION to enable sharing that buffer to the
 * DSP via the SMMU. If user allocated buffers not registered via this API
 * remote calls are expected to take more latency where these user buffers are passed.
 * Some versions of libadsprpc.so lack this function, so users should set
 * this symbol as weak.\n
 * `pragma weak  remote_register_buf_attr`\n
 * @param[in] buf virtual address of the buffer
 * @param[in] size size of the buffer
 * @param[in] fd file descriptor, callers can use -1 to deregister.
 * @param[in] attr map buffer as coherent or non-coherent \n
 * Expected error code due to incorrect input arguments is AEE_EBADPARM. Other than this error
 * codes treated as returned from fastRPC framework issues.
 */

 __QAIC_REMOTE_EXPORT void __QAIC_REMOTE(remote_register_buf_attr)(void* buf, int size, int fd, int attr) __QAIC_REMOTE_ATTRIBUTE;

/** Register a file descriptor for a user allocated buffer(Other than rpcmem_alloc). This is only valid on
 * Android with ION allocated memory. Users of fastRPC should register
 * a buffer allocated with ION to enable sharing that buffer to the
 * DSP via the SMMU. If user allocated buffers not registered via this API
 * remote calls are expected to take more latency where these user buffers are passed.
 * Some versions of libadsprpc.so lack this function, so users should set
 * this symbol as weak.\n
 * pragma weak  remote_register_buf\n
 * @param[in] buf virtual address of the buffer
 * @param[in] size size of the buffer
 * @param[in] fd file descriptor, callers can use -1 to deregister \n
 * Expected error code due to incorrect input arguments is AEE_EBADPARM. Other than this error
 * codes treated as returned from fastRPC framework issues.
 */
__QAIC_REMOTE_EXPORT void __QAIC_REMOTE(remote_register_buf)(void* buf, int size, int fd) __QAIC_REMOTE_ATTRIBUTE;

/** Register a DMA handle with fastrpc.  This is only valid on
 * android with ION allocated memory.  Users of fastrpc should register
 * a file descriptor allocated with ION to enable sharing that memory to the
 * dsp via the smmu.  Some versions of libadsprpc.so lack this
 * function, so users should set this symbol as weak.\n
 *
 * `pragma weak  remote_register_dma_handle`\n
 * `pragma weak  remote_register_dma_handle_attr`\n
 * @param[in] fd file descriptor, callers can use -1 to deregister.
 * @param[in] len size of the buffer
 * @param[in]  attr map buffer as coherent or non-coherent or no-map\n
 * Expected error codes due to incorrect input arguments are AEE_EBADPARM, AEE_EINVHANDLE. Other than these error codes,
 * treated as returned from fastRPC framework issues.
 */
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(remote_register_dma_handle_attr)(int fd, uint32_t len, uint32_t attr) __QAIC_REMOTE_ATTRIBUTE;

/** Register a DMA handle with fastrpc.  This is only valid on
 * android with ION allocated memory.  Users of fastrpc should register
 * a file descriptor allocated with ION to enable sharing that memory to the
 * dsp via the smmu.  Some versions of libadsprpc.so lack this
 * function, so users should set this symbol as weak.\n
 *
 * pragma weak  remote_register_dma_handle\n
 * @param[in] fd file descriptor, callers can use -1 to deregister.
 * @param[in] len size of the buffer \n
 * Expected error codes due to incorrect input arguments are AEE_EBADPARM, AEE_EINVHANDLE. Other than these error codes,
 * treated as returned from fastRPC framework issues.
 */
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(remote_register_dma_handle)(int fd, uint32_t len) __QAIC_REMOTE_ATTRIBUTE;

/** Register a file descriptor.  This can be used when users do not have
 * a mapping to pass to the RPC layer.  The generated address is a mapping
 * with PROT_NONE, any access to this memory will fail, so it should only
 * be used as an ID to identify this file descriptor to the RPC layer.
 *
 * To deregister use remote_register_buf(addr, size, -1).
 *
 * `pragma weak  remote_register_fd`\n
 * @param[in] fd is the file descriptor.
 * @param[in] size length the buffer
 * @return (void*)-1 on failure, address on success.\n
 * All error codes treated as returned from fastRPC framework issues.
 */
__QAIC_REMOTE_EXPORT void *__QAIC_REMOTE(remote_register_fd)(int fd, int size) __QAIC_REMOTE_ATTRIBUTE;

/** Get status of Async job.  This can be used to query the status of a Async job
 *
 * @param[in] jobid  returned during Async job submission via #remote_handle_invoke_async()
 * @param[in] timeout_us in micro seconds \n
 *                    timeout = 0, returns immediately with status/result\n
 *                    timeout > 0, waits for specified time and then returns with status/result \n
 *                    timeout < 0. waits indefinetely until job completes \n
 * @param[out] result ouptput pointer \n
 *                0 on success\n
 *                error code on failure\n
 * @return 0 on job completion\n
 *          if job status is pending, it is not returned from DSP\n
 * Expected error code due to incorrect input arguments is AEE_EBADPARM. Other than this error code
 * treated as returned from fastRPC framework issues.
 */
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(fastrpc_async_get_status)(fastrpc_async_jobid jobid, int timeout_us, int *result);

/** Release Async job.  Release async job after receiving status either through callback/poll
 *
 * @param[in] jobid returned during Async job submission via #remote_handle_invoke_async()
 * @return  Zero for success. Non-zero for failure.\n
 *          AEE_EBUSY, if job status is pending and is not returned from DSP\n
 *          AEE_EBADPARM, if job id is invalid \n
 */
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(fastrpc_release_async_job)(fastrpc_async_jobid jobid);

/**
 * By default, Driver runs Parallel mode
 */
#define REMOTE_MODE_PARALLEL  0

/**
 * When operating in SERIAL mode the driver will invalidate output buffers
 * before calling into the dsp.  This mode should be used when output
 * buffers have been written to somewhere besides the aDSP.
 */
#define REMOTE_MODE_SERIAL    1

/**
 * Internal transport prefix
 */
#define ITRANSPORT_PREFIX "'\":;./\\"

/** remote_set_mode API will be  deprecated in near future. It is discouraged to use this API.\n
 * Set the mode of operation.\n
 * This is the default mode for the driver.  While the driver is in parallel
 * mode it will try to invalidate output buffers after it transfers control
 * to the dsp.  This allows the invalidate operations to overlap with the
 * dsp processing the call.  This mode should be used when output buffers
 * are only read on the application processor and only written on the aDSP.
 * Some versions of libadsprpc.so lack this function, so users should set
 * this symbol as weak.\n
 * `pragma weak  remote_set_mode`
 * @param[in] mode serial/parallel operation
 * @return Integer value. Zero for success and non-zero for failure. For failure, returns error code.\n
 * All error codes, treated as returned from fastRPC framework issues.
 */
__QAIC_REMOTE_EXPORT int __QAIC_REMOTE(remote_set_mode)(uint32_t mode) __QAIC_REMOTE_ATTRIBUTE;

/**
 * @enum fastrpc_map_flags
 * @brief Types of maps with cache maintenance
 */
enum fastrpc_map_flags {
    /**
     * Map memory pages with RW- permission and CACHE WRITEBACK.
     * Driver will clean cache when buffer passed in a FastRPC call.
     * Same remote virtual address will be assigned for subsequent
     * FastRPC calls.
     */
    FASTRPC_MAP_STATIC = 0,

    /* Reserved for compatibility with deprecated flag */
    FASTRPC_MAP_RESERVED = 1,

    /**
     * Map memory pages with RW- permission and CACHE WRITEBACK.
     * Mapping tagged with a file descriptor. User is responsible for
     * maintenance of CPU and DSP caches for the buffer. Get virtual address
     * of buffer on DSP using HAP_mmap_get() and HAP_mmap_put() functions.
     */
    FASTRPC_MAP_FD = 2,

    /**
     * Mapping delayed until user calls HAP_mmap() and HAP_munmap()
     * functions on DSP. User is responsible for maintenance of CPU and DSP
     * caches for the buffer. Delayed mapping is useful for users to map
     * buffer on DSP with other than default permissions and cache modes
     * using HAP_mmap() and HAP_munmap() functions.
     */
    FASTRPC_MAP_FD_DELAYED = 3,

    /**
     * MAX enum is used internally by the FastRPC library for checking range of valid flags.
     * Add new flags above it. Leave unassigned and as last enum value.
     */
    FASTRPC_MAP_MAX,
};

/**
 * Creates a mapping on remote process for an ION buffer with file descriptor. A new FastRPC session
 * will be opened if not already opened for the domain.
 *
 * @param domain DSP domain ID of a FastRPC session
 * @param fd ION memory file descriptor
 * @param addr buffer virtual address on cpu
 * @param offset offset from the beginining of the buffer
 * @param length size of buffer in bytes
 * @param flags controls mapping functionality on DSP. Refer fastrpc_map_flags enum definition for more information.
 *
 * @return  0 on success, error code on failure.
 *          - AEE_EBADPARM Bad parameters
 *          - AEE_EFAILED Failed to map buffer
 *          - AEE_ENOMEMORY Out of memory (internal error)
 *          - AEE_EUNSUPPORTED Unsupported API on the target
 */
int fastrpc_mmap(int domain, int fd, void *addr, int offset, size_t length, enum fastrpc_map_flags flags);

/**
 * Removes a mapping associated with file descriptor.
 *
 * @param domain DSP domain ID of a FastRPC session
 * @param fd file descriptor
 * @param addr buffer virtual address used for mapping creation
 * @param length size of buffer mapped in bytes
 *
 * @return  0 on success, error code on failure.
 *          - AEE_EBADPARM Bad parameters e.g. Mapping not found for specified fd
 *          - AEE_EFAILED Failed to map buffer
 *          - AEE_EUNSUPPORTED Unsupported API on the target
 */
int fastrpc_munmap(int domain, int fd, void *addr, size_t length);

#ifdef __cplusplus
}
#endif

#endif // REMOTE_DEFAULT_H
