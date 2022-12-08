//==============================================================================
//
// Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef SNPE_UDO_IMPL_H
#define SNPE_UDO_IMPL_H

#include <stdbool.h>

#include "SnpeUdo/UdoShared.h"

#ifdef __cplusplus
extern "C"
{
#endif

/** @addtogroup c_plus_plus_apis C++
@{ */

typedef struct _SnpeUdo_OpFactory_t* SnpeUdo_OpFactory_t;
typedef struct _SnpeUdo_Operation_t* SnpeUdo_Operation_t;

typedef SnpeUdo_OpFactory_t Udo_OpFactory_t;
typedef SnpeUdo_Operation_t Udo_Operation_t;

/**
 * @brief Initialize the shared library's data structures. Calling any other
 *        library function before this one will result in error.
 *
 * @param[in] globalInfrastructure Global core-specific infrastructure to be
 *            used by operations created in this library. The definition and
 *            semantics of this object will be defined in the corresponding
 *            implementation header for the core type.
 * @return Error code
 */
SnpeUdo_ErrorType_t
SnpeUdo_initImplLibrary(void* globalInfrastructure);

typedef SnpeUdo_ErrorType_t
(*SnpeUdo_InitImplLibraryFunction_t)(void*);

/**
 * @brief A function to query the API version of the UDO implementation library.
 *        The function populates a SnpeUdo_LibVersion_t struct, which contains a SnpeUdo_Version_t
 *        struct for API version and library version.
 *
 * @param[in, out] version A pointer to struct which contains major, minor, teeny information for
 *                 library and api versions.
 *
 * @return Error code
 */
SnpeUdo_ErrorType_t
SnpeUdo_getImplVersion(SnpeUdo_LibVersion_t** version);

typedef SnpeUdo_ErrorType_t
(*SnpeUdo_getImplVersion_t)(SnpeUdo_LibVersion_t** version);

/**
 * @brief Release the shared library's data structures, and invalidate any
 *        handles returned by the library. The behavior of any outstanding
 *        asynchronous calls made to this library when this function is called
 *        are undefined. All library functions (except SnpeUdo_initImplLibrary) will
 *        return an error after this function has been successfully called.
 *
 *        It should be possible to call SnpeUdo_initImplLibrary after calling this
 *        function, and re-initialize the library.
 *
 * @return Error code
 */
SnpeUdo_ErrorType_t
SnpeUdo_terminateImplLibrary(void);

typedef SnpeUdo_ErrorType_t
(*SnpeUdo_TerminateImplLibraryFunction_t)(void);


/**
 * @brief A function to query info on the UDO implementation library.
 *        The function populates a structure which contains information about
 *        operations that are part of this library
 *
 * @param[in, out] implementationInfo A pointer to struct which contains information
 *                 on the operations
 *
 * @return error code
 *
 */
SnpeUdo_ErrorType_t
SnpeUdo_getImpInfo(SnpeUdo_ImpInfo_t** implementationInfo);

typedef SnpeUdo_ErrorType_t
(*SnpeUdo_GetImpInfoFunction_t)(SnpeUdo_ImpInfo_t** implementationInfo);

typedef SnpeUdo_GetImpInfoFunction_t Udo_GetImpInfoFunction_t;

/**
 * @brief A function to create an operation factory.
 *        The function receives the operation type, and an array of static parameters,
 *        and returns operation factory handler
 *
 * @param[in] udoCoreType The Core type to create the operation on. An error will
 *            be returned if this does not match the core type of the library.
 *
 * @param[in] perFactoryInfrastructure CreateOpFactory infrastructure appropriate to this
 *            core type. The definition and semantics of this object will be defined
 *            in the corresponding implementation header for the core type.
 *
 * @param[in] operationType A string containing Operation type. for example "MY_CONV"
 *
 * @param[in] numOfStaticParams The number of static parameters.
 *
 * @param[in] staticParams Array of static parameters
 *
 * @param[in,out] opFactory Handler to Operation Factory, to be used when creating operations
 *
 * @return Error Code
 */
SnpeUdo_ErrorType_t
SnpeUdo_createOpFactory(SnpeUdo_CoreType_t    udoCoreType,
                        void*                perFactoryInfrastructure,
                        SnpeUdo_String_t      operationType,
                        uint32_t             numOfStaticParams,
                        SnpeUdo_Param_t*      staticParams,
                        SnpeUdo_OpFactory_t*  opFactory);

typedef SnpeUdo_ErrorType_t
(*SnpeUdo_CreateOpFactoryFunction_t)(SnpeUdo_CoreType_t,
                                     void*,
                                     SnpeUdo_String_t,
                                     uint32_t,
                                     SnpeUdo_Param_t*,
                                     SnpeUdo_OpFactory_t*);

typedef SnpeUdo_CreateOpFactoryFunction_t Udo_CreateOpFactoryFunction_t;

/**
 * @brief A function to release the resources allocated for an operation factory
 *        created by this library.
 *
 * @param[in] factory The operation factory to release. Upon success this handle will be invalidated.
 *
 * @return Error Code
 */
SnpeUdo_ErrorType_t
SnpeUdo_releaseOpFactory(SnpeUdo_OpFactory_t opFactory);

typedef SnpeUdo_ErrorType_t
(*SnpeUdo_ReleaseOpFactoryFunction_t)(SnpeUdo_OpFactory_t);

typedef SnpeUdo_ReleaseOpFactoryFunction_t Udo_ReleaseOpFactoryFunction_t;

/**
 * @brief A function to create an operation from the factory.
 *        The function receives array of inputs and array of outputs, and creates an operation
 *        instance, returning the operation instance handler.
 *
 * @param[in] opFactory OpFactory instance containing the parameters for this operation.
 *
 * @param[in] perOpInfrastructure Per-Op infrastructure for this operation. The definition
 *            and semantics of this object will be defined in the implementation header
 *            appropriate to this core type.
 *
 * @param[in] numOfInputs The number of input tensors this operation will receive.
 *
 * @param[in] inputs Array of input tensors, providing both the sizes and initial
 *            location of the data.
 *
 * @param[in] numOfOutputs Number of output tensors this operation will produce.
 *
 * @param[in] outputs Array of output tensors, providing both the sizes and
 *            initial location of the data.
 *
 * @param[in,out] operation Handle for newly created operation instance.
 *
 * @return Error Code
 */
SnpeUdo_ErrorType_t
SnpeUdo_createOperation(SnpeUdo_OpFactory_t    opFactory,
                        void*                 perOpInfrastructure,
                        uint32_t              numOfInputs,
                        SnpeUdo_TensorParam_t* inputs,
                        uint32_t              numOfOutputs,
                        SnpeUdo_TensorParam_t* outputs,
                        SnpeUdo_Operation_t*   operation);

typedef SnpeUdo_ErrorType_t
(*SnpeUdo_CreateOperationFunction_t)(SnpeUdo_OpFactory_t,
                                     void*,
                                     uint32_t,
                                     SnpeUdo_TensorParam_t*,
                                     uint32_t,
                                     SnpeUdo_TensorParam_t*,
                                     SnpeUdo_Operation_t*);

typedef SnpeUdo_CreateOperationFunction_t Udo_CreateOperationFunction_t;

/**
 * @brief A pointer to notification function.
 *
 *        The notification function supports the non-blocking (e.g. asynchronous) execution use-case.
 *        In case an "executeUdoOp" function is called with "blocking" set to zero, and a
 *        notify function, this function will be called by the implementation library at the
 *        end of execution. The implementation library will pass the notify function the ID
 *        that was provided to it when "executeUdoOp" was called.
 *
 * @param[in] ID 32-bit value, that was provided to executeUdoOp by the calling entity.
 *            Can be used to track the notifications, in case of multiple execute calls issued.
 *
 * @return Error code
 *
 */
typedef SnpeUdo_ErrorType_t
(*SnpeUdo_ExternalNotify_t)(const uint32_t ID);

typedef SnpeUdo_ExternalNotify_t Udo_ExternalNotify_t;

/**
 * @brief Operation execution function.
 *
 *        Calling this function will run the operation on set of inputs, generating a set of outputs.
 *        The call can be blocking (synchronous) or non-blocking (asynchronous). To support the
 *        non-blocking mode, the calling entity can pass an ID and a notification function.
 *        At the end of the execution this notification function would be called, passing it the ID.
 *        <b> NOTE: Asynchronous execution mode not supported in this release. </b>
 *
 * @param[in] operation handle to the operation on which execute is invoked
 * @param[in] blocking flag to indicate execution mode.
 *            If set, execution is blocking,
 *            e.g SnpeUdo_executeOp call does not return until execution is done.
 *            If not set, SnpeUdo_executeOp returns immediately, and the
 *            library will call the notification function (if set) when execution is done.
 *
 * @param[in] ID 32-bit number that can be used by the calling entity to track execution
 *            in case of non-blocking execution.
 *            For example, it can be a sequence number, increased by one on each call.
 *
 * @param[in] notifyFunc Pointer to notification function. if the pointer is set, and execution is
 *            non-blocking, the library will call this function at end of execution,
 *            passing the number provided as ID
 *
 * @return Error code
 *
 */
SnpeUdo_ErrorType_t
SnpeUdo_executeOp(SnpeUdo_Operation_t operation,
                  bool         blocking,
                  const uint32_t ID,
                  SnpeUdo_ExternalNotify_t notifyFunc);

typedef SnpeUdo_ErrorType_t
(*SnpeUdo_ExecuteOpFunction_t)(SnpeUdo_Operation_t,
                               bool,
                               const uint32_t,
                               SnpeUdo_ExternalNotify_t);

typedef SnpeUdo_ExecuteOpFunction_t Udo_ExecuteOpFunction_t;

/**
 * @brief A function to setting the inputs & outputs. part of SnpeUdo_Operation struct,
 *        returned from creation of a new operation instance.
 *        <b> Not supported in this release. </b>
 *
 *        This function allows the calling entity to change some of the inputs and outputs
 *        between calls to execute.
 *        Note that the change is limited to changing the <b> pointer </b> to the tensor data only.
 *        Any other change may be rejected by the implementation library, causing
 *        immediate invalidation of the operation instance
 *
 * @param[in] operation Operation on which IO tensors are set
 *
 * @param[in] inputs array of tensor parameters. The calling entity may provide a subset of the
 *            operation inputs, providing only those that it wants to change.
 *
 * @param[in] outputs array of tensor parameters. The calling entity may provide a subset of the
 *            operation outputs, providing only those that it wants to change.
 *
 * @return Error code
 *
 */
SnpeUdo_ErrorType_t
SnpeUdo_setOpIO(SnpeUdo_Operation_t operation,
                SnpeUdo_TensorParam_t* inputs,
                SnpeUdo_TensorParam_t* outputs);

typedef SnpeUdo_ErrorType_t
(*SnpeUdo_SetOpIOFunction_t)(SnpeUdo_Operation_t,
                             SnpeUdo_TensorParam_t*,
                             SnpeUdo_TensorParam_t*);

typedef SnpeUdo_SetOpIOFunction_t Udo_SetOpIOFunction_t;

/**
 * @brief A function to return execution times.
 *
 *        This function can be called to query the operation execution times on the IP core
 *        on which the operation is run. The time is provided in micro-seconds
 *
 * @param[in] operation Handle to operation whose execution time is being profiled
 *
 * @param[in,out] executionTime pointer to a uint32 value.This function writes the operation
 *                execution time in usec into this value.
 *
 * @return Error code
 *
 */
SnpeUdo_ErrorType_t
SnpeUdo_profileOp(SnpeUdo_Operation_t operation, uint32_t *executionTime);

typedef SnpeUdo_ErrorType_t
(*SnpeUdo_ProfileOpFunction_t)(SnpeUdo_Operation_t, uint32_t*);

typedef SnpeUdo_ProfileOpFunction_t Udo_ProfileOpFunction_t;

/**
 * @brief A function to release the operation instance
 *        \n When it is called, the implementation library needs to release all resources
 *        allocated for this operation instance.
 *        \n Note that all function pointers which are part of SnpeUdo_Operation become
 *        <b> invalid </b> once releaseUdoOp call returns.
 *
 * @param[in] operation Handle to operation to be released
 * @return Error code
 *
 */
SnpeUdo_ErrorType_t
SnpeUdo_releaseOp(SnpeUdo_Operation_t operation);

typedef SnpeUdo_ErrorType_t
(*SnpeUdo_ReleaseOpFunction_t)(SnpeUdo_Operation_t);

typedef SnpeUdo_ReleaseOpFunction_t Udo_ReleaseOpFunction_t;

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

#ifdef __cplusplus
} // extern "C"
#endif

#endif //SNPE_UDO_IMPL_H
