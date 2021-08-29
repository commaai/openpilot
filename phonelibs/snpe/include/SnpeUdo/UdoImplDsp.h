//==============================================================================
//
// Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

//==============================================================================
/*
 * THIS HEADER FILE IS COPIED FROM HEXAGON-NN PROJECT
 *
 */
//==============================================================================


// Header to be used by a DSP Hexnn UDO Implementation library

#ifndef SNPE_UDO_IMPL_DSP_H
#define SNPE_UDO_IMPL_DSP_H
#include <stdio.h>
#include "SnpeUdo/UdoImpl.h"

/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * @brief A function to validate that a set of params is supported by an operation
 *        This function is HexNN specific, use case is when registration library is not in use.
 *        Optional function.
 *
 * @param[in] operationType Operation type
 * @param[in] numOfStaticParams Number of static params defined by the op
 * @param[in] staticParams Array of static params to the op
 * @return Error code, indicating if the operation can be created on this set of configuration or not.
 *
 */

SnpeUdo_ErrorType_t
SnpeUdo_validateOperation (SnpeUdo_String_t operationType,
                           uint32_t numOfStaticParams,
                           const SnpeUdo_Param_t* staticParams);

typedef SnpeUdo_ErrorType_t (*SnpeUdo_ValidateOperationFunction_t) (SnpeUdo_String_t,
                                                                    uint32_t,
                                                                    const SnpeUdo_Param_t*);


// enum used for indicating input/outout tensor data layouts on DSP, plain vs d32
typedef enum {
        SNPE_UDO_DSP_TENSOR_LAYOUT_PLAIN,
        SNPE_UDO_DSP_TENSOR_LAYOUT_D32
} SnpeUdo_HexNNTensorLayout_t;

/**
 * @brief A function to query numbers of inputs and outputs,
 *        quantization type of each input and each output as arrays,
 *        and data layout (plain vs d32) of each input and each output as arrays
 *        of an operation.
 * inputsQuantTypes and inputsLayouts should point to arrays of size numOfInputs
 * outputsQuantTypes and outputsLayouts should point to arrays of size numOfOutputs
 *
 * Note: inputsLayouts and inputsLayouts can point to NULL, in this case, it is
 * assumed all inputs and/or outputs have plain data layouts, i.e. no D32
 *
 * @param[in] operationType Operation type
 * @param[in] numOfStaticParams Number of static params defined by the op
 * @param[in] staticParams Array of static params to the op
 * @param[in,out] numOfInputs Number of input tensors to the op
 * @param[in,out] inputsQuantTypes Array of Quantization info for each input tensor
 * @param[in,out] inputsLayouts Array of layout type for each input tensor
 * @param[in,out] numOfOutputs Number of output tensors to the op
 * @param[in,out] outputsQuantTypes Array of Quantization info for each output tensor
 * @param[in,out] outputsLayouts Array of layout type for each output tensor
 * @return error code, indicating status of query
 */

SnpeUdo_ErrorType_t
SnpeUdo_queryOperation (SnpeUdo_String_t operationType,
                        uint32_t numOfStaticParams,
                        const SnpeUdo_Param_t* staticParams,
                        uint32_t* numOfInputs,
                        SnpeUdo_QuantizationType_t** inputsQuantTypes,
                        SnpeUdo_HexNNTensorLayout_t** inputsLayouts,
                        uint32_t* numOfOutputs,
                        SnpeUdo_QuantizationType_t** outputsQuantTypes,
                        SnpeUdo_HexNNTensorLayout_t** outputsLayouts);

typedef SnpeUdo_ErrorType_t (*SnpeUdo_QueryOperationFunction_t) (SnpeUdo_String_t,
                                                                 uint32_t,
                                                                 const SnpeUdo_Param_t*,
                                                                 uint32_t*,
                                                                 SnpeUdo_QuantizationType_t**,
                                                                 SnpeUdo_HexNNTensorLayout_t**,
                                                                 uint32_t*,
                                                                 SnpeUdo_QuantizationType_t**,
                                                                 SnpeUdo_HexNNTensorLayout_t**);



// Global infrastructure functions supported by Hexagon-NN v2
typedef void (*workerThread_t) (void* perOpInfrastructure, void* userData);
typedef int (*udoSetOutputTensorSize_t) (void* perOpInfrastructure, uint32_t outIdx, uint32_t size);
typedef int (*udoGetInputD32Paddings_t) (void* perOpInfrastructure, uint32_t inIdx,
                                         uint32_t* heightPadBefore, uint32_t* heightPadAfter,
                                         uint32_t* widthPadBefore, uint32_t* widthPadAfter,
                                         uint32_t* depthPadBefore, uint32_t* depthPadAfter);
typedef int (*udoSetOutputD32ShapeSizePaddings_t) (void* perOpInfrastructure, uint32_t outIdx,
                                                   uint32_t batch,
                                                   uint32_t height, uint32_t heightPadBefore, uint32_t heightPadAfter,
                                                   uint32_t width, uint32_t widthPadBefore, uint32_t widthPadAfter,
                                                   uint32_t depth, uint32_t depthPadBefore, uint32_t depthPadAfter,
                                                   SnpeUdo_DataType_t dataType);
typedef void* (*udoMemalign_t) (size_t n, size_t size);
typedef void* (*udoMalloc_t) (size_t size);
typedef void* (*udoCalloc_t) (size_t n, size_t size);
typedef void (*udoFree_t) (void* ptr);
typedef uint32_t (*udoGetVtcmSize_t) (void* perOpInfrastructure);
typedef void* (*udoGetVtcmPtr_t) (void* perOpInfrastructure);
typedef uint32_t (*udoVtcmIsReal_t) (void* perOpInfrastructure);
typedef void (*udoRunWorkerThreads_t) (void* perOpInfrastructure, uint32_t nThreads, workerThread_t w, void* userData);

typedef struct hexNNv2GlobalInfra {
    udoSetOutputTensorSize_t udoSetOutputTensorSize;
    udoGetInputD32Paddings_t udoGetInputD32Paddings;
    udoSetOutputD32ShapeSizePaddings_t udoSetOutputD32ShapeSizePaddings;
    udoMemalign_t udoMemalign;
    udoMalloc_t udoMalloc;
    udoCalloc_t udoCalloc;
    udoFree_t udoFree;
    udoGetVtcmSize_t udoGetVtcmSize;
    udoGetVtcmPtr_t udoGetVtcmPtr;
    udoVtcmIsReal_t udoVtcmIsReal;
    udoRunWorkerThreads_t udoRunWorkerThreads;
} SnpeUdo_HexNNv2GlobalInfra_t;

// hexnn types
typedef enum hexnnInfraType {
   UDO_INFRA_HEXNN_V2,
   UDO_INFRA_HEXNN_V3   // reserved, do not use
} SnpeUdo_HexNNInfraType_t;


/**
 * @brief Infrastructures needed by a developer of DSP Hexnn UDO Implementation library.
 *
 * The framework/runtime which loads the Hexnn UDO implementation library provides
 * this infrastructure to the loaded library by calling "SnpeUdo_initImplLibrary"
 * function, and passing it (cast to void*). The Hexnn UDO library is expected
 * to cast it back to this structure.
 *
 */
typedef struct dspGlobalInfrastructure {
    SnpeUdo_Version_t   dspInfraVersion;     // api version
    SnpeUdo_HexNNInfraType_t infraType;
    SnpeUdo_HexNNv2GlobalInfra_t hexNNv2Infra;
} SnpeUdo_DspGlobalInfrastructure_t;


/**
 * hexnn v2 per op factory infrastructure
 *
 * The framework/runtime passes per op factory infrastructure as a void pointer
 * to HexNN UDO implementation library by calling function "SnpeUdo_createOpFactory".
 * UDO implementation library is expected to cast it back to this following struct.
 *
 */
typedef struct hexnnv2OpFactoryInfra {
   unsigned long graphId;
} SnpeUdo_HexNNv2OpFactoryInfra_t;


/**
 * hexnn v2 per operation infrastructure
 *
 * The framework/runtime passes per operation infrastructure as a void pointer
 * to HexNN UDO implementation library by calling function "SnpeUdo_createOperation".
 * UDO implementation library is expected to cast it to the following type and save it.
 *
 * This is needed to be passed back into some functions from global infrastructure.
 *
 */
typedef void* SnpeUdo_HexNNv2OpInfra_t;

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

#endif // SNPE_UDO_IMPL_DSP_H
