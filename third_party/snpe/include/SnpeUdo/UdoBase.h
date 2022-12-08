//==============================================================================
//
// Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef SNPE_UDO_BASE_H
#define SNPE_UDO_BASE_H

#include <stdint.h>

// Provide values to use for API version.
#define API_VERSION_MAJOR 1
#define API_VERSION_MINOR 6
#define API_VERSION_TEENY 0

/** @addtogroup c_plus_plus_apis C++
@{ */

// Defines a bitmask of enum values.
typedef uint32_t SnpeUdo_Bitmask_t;
typedef SnpeUdo_Bitmask_t Udo_Bitmask_t;

// A string of characters, rather than an array of bytes.
// Assumed to be UTF-8.
typedef char* SnpeUdo_String_t;
typedef SnpeUdo_String_t Udo_String_t;

// The maximum allowable length of a SnpeUdo_String_t in bytes,
// including null terminator. SNPE will truncate strings longer
// than this.
#define SNPE_UDO_MAX_STRING_SIZE 1024

/**
  * An enum which holds the various error types.
  * The error types are divided to classes :
  * 0 - 99    : generic errors
  * 100 - 200 : errors related to configuration
  *
  */
typedef enum
{
   /// No Error
   SNPE_UDO_NO_ERROR                    = 0,          UDO_NO_ERROR                    = 0,
   /// Unsupported value for core type
   SNPE_UDO_WRONG_CORE                  = 1,          UDO_WRONG_CORE                  = 1,
   /// Invalid attribute/argument passed into UDO API
   SNPE_UDO_INVALID_ARGUMENT            = 2,          UDO_INVALID_ARGUMENT            = 2,
   /// Unsupported feature error
   SNPE_UDO_UNSUPPORTED_FEATURE         = 3,          UDO_UNSUPPORTED_FEATURE         = 3,
   /// Error relating to memory allocation
   SNPE_UDO_MEM_ALLOC_ERROR             = 4,          UDO_MEM_ALLOC_ERROR             = 4,
   /* Configuration Specific errors */
   /// No op with given attributes available in library
   SNPE_UDO_WRONG_OPERATION             = 100,        UDO_WRONG_OPERATION             = 100,
   /// Unsupported value for core type in UDO configuration
   SNPE_UDO_WRONG_CORE_TYPE             = 101,        UDO_WRONG_CORE_TYPE             = 101,
   /// Wrong number of params in UDO definition
   SNPE_UDO_WRONG_NUM_OF_PARAMS         = 102,        UDO_WRONG_NUM_OF_PARAMS         = 102,
   /// Wrong number of dimensions for tensor(s) in UDO definition
   SNPE_UDO_WRONG_NUM_OF_DIMENSIONS     = 103,        UDO_WRONG_NUM_OF_DIMENSIONS     = 103,
   /// Wrong number of input tensors in UDO definition
   SNPE_UDO_WRONG_NUM_OF_INPUTS         = 104,        UDO_WRONG_NUM_OF_INPUTS         = 104,
   /// Wrong number of output tensors in UDO definition
   SNPE_UDO_WRONG_NUM_OF_OUTPUTS        = 105,        UDO_WRONG_NUM_OF_OUTPUTS        = 105,
   SNPE_UDO_PROGRAM_CACHE_NOT_FOUND     = 106,        UDO_PROGRAM_CACHE_NOT_FOUND     = 106,
   SNPE_UDO_UNKNOWN_ERROR               = 0xFFFFFFFF, UDO_UNKNOWN_ERROR               = 0xFFFFFFFF
} SnpeUdo_ErrorType_t;

typedef SnpeUdo_ErrorType_t Udo_ErrorType_t;

/**
  * An enum which holds the various data types.
  * Designed to be used as single values or combined into a bitfield parameter
  * (0x1, 0x2, 0x4, etc)
  * \n FIXED_XX types are targeted for data in tensors.
  * \n UINT / INT types are targeted for scalar params
  */
typedef enum
{
   /// data type: 16-bit floating point
   SNPE_UDO_DATATYPE_FLOAT_16       = 0x01,        UDO_DATATYPE_FLOAT_16       = 0x01,
   /// data type: 32-bit floating point
   SNPE_UDO_DATATYPE_FLOAT_32       = 0x02,        UDO_DATATYPE_FLOAT_32       = 0x02,
   /// data type: 4-bit fixed point
   SNPE_UDO_DATATYPE_FIXED_4        = 0x04,        UDO_DATATYPE_FIXED_4        = 0x04,
   /// data type: 8-bit fixed point
   SNPE_UDO_DATATYPE_FIXED_8        = 0x08,        UDO_DATATYPE_FIXED_8        = 0x08,
   /// data type: 16-bit fixed point
   SNPE_UDO_DATATYPE_FIXED_16       = 0x10,        UDO_DATATYPE_FIXED_16       = 0x10,
   /// data type: 32-bit fixed point
   SNPE_UDO_DATATYPE_FIXED_32       = 0x20,        UDO_DATATYPE_FIXED_32       = 0x20,
   /// data type: 8-bit unsigned integer
   SNPE_UDO_DATATYPE_UINT_8         = 0x100,       UDO_DATATYPE_UINT_8         = 0x100,
   /// data type: 16-bit unsigned integer
   SNPE_UDO_DATATYPE_UINT_16        = 0x200,       UDO_DATATYPE_UINT_16        = 0x200,
   /// data type: 32-bit unsigned integer
   SNPE_UDO_DATATYPE_UINT_32        = 0x400,       UDO_DATATYPE_UINT_32        = 0x400,
   /// data type: 8-bit signed integer
   SNPE_UDO_DATATYPE_INT_8          = 0x1000,      UDO_DATATYPE_INT_8          = 0x1000,
   /// data type: 16-bit signed integer
   SNPE_UDO_DATATYPE_INT_16         = 0x2000,      UDO_DATATYPE_INT_16         = 0x2000,
   /// data type: 32-bit signed integer
   SNPE_UDO_DATATYPE_INT_32         = 0x4000,      UDO_DATATYPE_INT_32         = 0x4000,
   SNPE_UDO_DATATYPE_LAST           = 0xFFFFFFFF,  UDO_DATATYPE_LAST           = 0xFFFFFFFF
} SnpeUdo_DataType_t;

typedef SnpeUdo_DataType_t Udo_DataType_t;

/**
  * An enum which holds the various layouts.
  * Designed to be used as single values or combined into a bitfield parameter
  * (0x1, 0x2, 0x4, etc)
  */
typedef enum
{
   /// data layout (4D): NHWC (batch-height-width-channel)
   SNPE_UDO_LAYOUT_NHWC             = 0x01,        UDO_LAYOUT_NHWC             = 0x01,
   /// data layout (4D): NCHW (batch-channel-height-width)
   SNPE_UDO_LAYOUT_NCHW             = 0x02,        UDO_LAYOUT_NCHW             = 0x02,
   /// data layout (5D): NDHWC (batch-dimension-height-width-channel)
   SNPE_UDO_LAYOUT_NDHWC            = 0x04,        UDO_LAYOUT_NDHWC            = 0x04,
   SNPE_UDO_LAYOUT_GPU_OPTIMAL1     = 0x08,        UDO_LAYOUT_GPU_OPTIMAL1     = 0x08,
   SNPE_UDO_LAYOUT_GPU_OPTIMAL2     = 0x10,        UDO_LAYOUT_GPU_OPTIMAL2     = 0x10,
   SNPE_UDO_LAYOUT_DSP_OPTIMAL1     = 0x11,        UDO_LAYOUT_DSP_OPTIMAL1     = 0x11,
   SNPE_UDO_LAYOUT_DSP_OPTIMAL2     = 0x12,        UDO_LAYOUT_DSP_OPTIMAL2     = 0x12,
   // Indicates no data will be allocated for this tensor.
   // Used to specify optional inputs/outputs positionally.
   SNPE_UDO_LAYOUT_NULL             = 0x13,        UDO_LAYOUT_NULL             = 0x13,
   SNPE_UDO_LAYOUT_LAST             = 0xFFFFFFFF,  UDO_LAYOUT_LAST             = 0xFFFFFFFF
} SnpeUdo_TensorLayout_t;

typedef SnpeUdo_TensorLayout_t Udo_TensorLayout_t;

/**
  * An enum which holds the UDO library Core type .
  * Designed to be used as single values or combined into a bitfield parameter
  * (0x1, 0x2, 0x4, etc)
  */
typedef enum
{
   /// Library target IP Core is undefined
   SNPE_UDO_CORETYPE_UNDEFINED   = 0x00,          UDO_CORETYPE_UNDEFINED   = 0x00,
   /// Library target IP Core is CPU
   SNPE_UDO_CORETYPE_CPU         = 0x01,          UDO_CORETYPE_CPU         = 0x01,
   /// Library target IP Core is GPU
   SNPE_UDO_CORETYPE_GPU         = 0x02,          UDO_CORETYPE_GPU         = 0x02,
   /// Library target IP Core is DSP
   SNPE_UDO_CORETYPE_DSP         = 0x04,          UDO_CORETYPE_DSP         = 0x04,
   SNPE_UDO_CORETYPE_LAST        = 0xFFFFFFFF,    UDO_CORETYPE_LAST        = 0xFFFFFFFF
} SnpeUdo_CoreType_t;

typedef SnpeUdo_CoreType_t Udo_CoreType_t;

/**
  * An enum to specify the parameter type : Scalar or Tensor
  */
typedef enum
{
   /// UDO static param type: scalar
   SNPE_UDO_PARAMTYPE_SCALAR = 0x00,         UDO_PARAMTYPE_SCALAR = 0x00,
   /// UDO static param type: string
   SNPE_UDO_PARAMTYPE_STRING = 0x01,         UDO_PARAMTYPE_STRING = 0x01,
   /// UDO static param type: tensor
   SNPE_UDO_PARAMTYPE_TENSOR = 0x02,         UDO_PARAMTYPE_TENSOR = 0x02,
   SNPE_UDO_PARAMTYPE_LAST   = 0xFFFFFFFF,   UDO_PARAMTYPE_LAST   = 0xFFFFFFFF
} SnpeUdo_ParamType_t;

typedef SnpeUdo_ParamType_t Udo_ParamType_t;

/**
  * An enum to specify quantization type
  */
typedef enum
{
   /// Tensor Quantization type: NONE. Signifies unquantized tensor data
   SNPE_UDO_QUANTIZATION_NONE   = 0x00,         UDO_QUANTIZATION_NONE   = 0x00,
   /// Tensor Quantization type: Tensorflow-style
   SNPE_UDO_QUANTIZATION_TF     = 0x01,         UDO_QUANTIZATION_TF     = 0x01,
   SNPE_UDO_QUANTIZATION_QMN    = 0x02,         UDO_QUANTIZATION_QMN    = 0x02,
   SNPE_UDO_QUANTIZATION_LAST   = 0xFFFFFFFF,   UDO_QUANTIZATION_LAST   = 0xFFFFFFFF
} SnpeUdo_QuantizationType_t;

typedef SnpeUdo_QuantizationType_t Udo_QuantizationType_t;

/**
 * @brief A struct which is used to provide a version number using 3 values : major, minor, teeny
 *
 */
typedef struct
{
   /// version field: major - for backward-incompatible changes
   uint32_t major;
   /// version field: minor - for backward-compatible feature updates
   uint32_t minor;
   /// version field: teeny - for minor bug-fixes and clean-up
   uint32_t teeny;
} SnpeUdo_Version_t;

typedef SnpeUdo_Version_t Udo_Version_t;

/**
 * @brief A struct returned from version query, contains the Library version and API version
 *
 */
typedef struct
{
   /// Version of UDO library. Controlled by users
   SnpeUdo_Version_t libVersion;
   /// Version of SNPE UDO API used in compiling library. Determined by SNPE
   SnpeUdo_Version_t apiVersion;
} SnpeUdo_LibVersion_t;

/**
 * @brief A struct returned from version query, contains the package version
 *
 */
typedef struct
{
   /// Version of UDO API used in package.
   Udo_Version_t apiVersion;
} Udo_PkgVersion_t;

/**
 * @brief A union to hold the value of a generic type. Allows defining a parameter struct
 * in a generic way, with a "value" location that holds the data regardless of the type.
 *
 */
typedef union
{
   /// value type: float
   float    floatValue;
   /// value type: unsigned 32-bit integer
   uint32_t uint32Value;
   /// value type: signed 32-bit integer
   int32_t  int32Value;
   /// value type: unsigned 16-bit integer
   uint16_t uint16Value;
   /// value type: signed 16-bit integer
   int16_t  int16Value;
   /// value type: unsigned 8-bit integer
   uint8_t  uint8Value;
   /// value type: signed 8-bit integer
   int8_t   int8Value;
} SnpeUdo_Value_t;

typedef SnpeUdo_Value_t Udo_Value_t;

/**
 * @brief A struct which defines a scalar parameter : name, data type, and union of values
 *
 */
typedef struct
{
   /// The parameter data type : float, int, etc.
   SnpeUdo_DataType_t  dataType;
   /// a union of specified type which holds the data
   SnpeUdo_Value_t dataValue;
} SnpeUdo_ScalarParam_t;

typedef SnpeUdo_ScalarParam_t Udo_ScalarParam_t;

/**
 * @brief A struct which defines the quantization parameters in case of Tensorflow style quantization
 *
 */
typedef struct
{
   /// minimum value of the quantization range of data
   float minValue;
   /// maximum value of the quantization range of data
   float maxValue;
} SnpeUdo_TFQuantize_t;

typedef SnpeUdo_TFQuantize_t Udo_TFQuantize_t;

/**
 * @brief A struct which defines the quantization type, and union of supported quantization structs
 *
 */
typedef struct
{
   /// quantization type (only TF-style currently supported)
   SnpeUdo_QuantizationType_t quantizeType;
   union
   {
     /// TF-style min-max quantization ranges
     SnpeUdo_TFQuantize_t TFParams;
   };
} SnpeUdo_QuantizeParams_t;

typedef SnpeUdo_QuantizeParams_t Udo_QuantizeParams_t;

/**
 * @brief A struct which defines the datatype associated with a specified core-type
 * This should be used to denote the datatypes for a single tensor info, depending
 * on the intended execution core.
 *
 */
typedef struct
{
    /// The IP Core
    SnpeUdo_CoreType_t     coreType;
    /// The associated datatype for this coreType
    SnpeUdo_DataType_t       dataType;
} SnpeUdo_PerCoreDatatype_t;

typedef SnpeUdo_PerCoreDatatype_t Udo_PerCoreDatatype_t;

/**
 * @brief A struct which defines a tensor parameter : name, data type, layout, quantization, more.
 *        Also holds a pointer to the tensor data.
 *
 */
typedef struct
{
   /// The maximum allowable dimensions of the tensor. The memory held in
   /// _tensorData_ is guaranteed to be large enough for this.
   uint32_t*                maxDimensions;
   /// The current dimensions of the tensor. An operation may modify the current
   /// dimensions of its output, to indicate cases where the output has been
   /// "resized".
   /// Note that for static parameters, the current and max dimensions must
   /// match.
   uint32_t*                currDimensions;
   /// Quantization params applicable to the tensor. Currently only supports
   /// Tensorflow quantization style.
   SnpeUdo_QuantizeParams_t quantizeParams;
   /// Number of dimensions to the tensor: 3D, 4D, etc.
   uint32_t                 tensorRank;
   /// The parameter data type: float, int, etc.
   SnpeUdo_DataType_t       dataType;
   /// The tensor layout type: NCHW, NHWC, etc.
   SnpeUdo_TensorLayout_t   layout;
   /// Opaque pointer to tensor data. User may be required to re-interpret the pointer
   /// based on core-specific definitions.
   void*                    tensorData;
} SnpeUdo_TensorParam_t;

typedef SnpeUdo_TensorParam_t Udo_TensorParam_t;

/**
 * @brief A struct which defines tensor information for activation tensors only
 *
 * It describes an activation tensor object using its name, the intended layout and the datatype
 * it will take depending on the intended runtime core. The repeated field indicates that
 * that the tensor info describes several input/output activation tensors, which all share the
 * aforementioned properties.
 */
typedef struct
{
    /// The tensor name
    SnpeUdo_String_t    tensorName;
    /// The tensor layout type: NCHW, NHWC, etc.
    SnpeUdo_TensorLayout_t   layout;
    /// The per core datatype: {SNPE_UDO_DATATYPE, SNPE_UDO_CORE_TYPE}
    SnpeUdo_PerCoreDatatype_t* perCoreDatatype;
    /// A boolean field indicating that this tensorinfo will be repeated e.x for ops such as Concat or Split
    bool repeated;
    /// A boolean field indicating whether input is static or not.
    bool isStatic;
} SnpeUdo_TensorInfo_t;

typedef SnpeUdo_TensorInfo_t Udo_TensorInfo_t;

/**
 * @brief struct which defines a UDO parameter - a union of scalar, tensor and string parameters
 *
 */
typedef struct
{
   /// Type is scalar or tensor
  SnpeUdo_ParamType_t paramType;
  /// The param name, for example : "offset", "activation_type"
  SnpeUdo_String_t    paramName;
  union
  {
    /// scalar param value
    SnpeUdo_ScalarParam_t scalarParam;
    /// tensor param value
    SnpeUdo_TensorParam_t tensorParam;
    /// string param value
    SnpeUdo_String_t      stringParam;
  };
} SnpeUdo_Param_t;

typedef SnpeUdo_Param_t Udo_Param_t;

/**
 * @brief A struct which defines Operation information which is specific for IP core (CPU, GPU, DSP ...)
 *
 */
typedef struct
{
   /// The IP Core
   SnpeUdo_CoreType_t     udoCoreType;
   /// Bitmask, defines supported internal calculation types (like FLOAT_32, etc)
   /// Based on SnpeUdo_DataType
   SnpeUdo_Bitmask_t      operationCalculationTypes;
} SnpeUdo_OpCoreInfo_t;

typedef SnpeUdo_OpCoreInfo_t Udo_OpCoreInfo_t;

/**
 * @brief A struct which defines the common and core-specific Operation information
 *
 */
typedef struct
{
   /// Operation type
   SnpeUdo_String_t      operationType;
   /// A bitmask describing which IP Cores (CPU, GPU, DSP ...) support this operation
   /// Translated based on SnpeUdo_CoreType
   SnpeUdo_Bitmask_t     supportedByCores;
   /// Number of static parameters defined by the op
   uint32_t              numOfStaticParams;
   /// Array of static parameters. Can be scalar or tensor params
   SnpeUdo_Param_t*      staticParams;
   /// Number of input tensors this op receives
   uint32_t              numOfInputs;
   /// Array of input tensor names to this operation
   SnpeUdo_String_t*      inputNames;
   /// Number of output tensors this op receives
   uint32_t              numOfOutputs;
   /// Array of output tensor names to this operation
   SnpeUdo_String_t*      outputNames;
   /// Number of cores that the op can execute on
   uint32_t              numOfCoreInfo;
   /// Array of per-core information entries
   SnpeUdo_OpCoreInfo_t* opPerCoreInfo;
    /// Array of input tensor infos for this operation
   SnpeUdo_TensorInfo_t*     inputInfos;
   /// Array of output tensor infos for this operation
   SnpeUdo_TensorInfo_t*     outputInfos;
} SnpeUdo_OperationInfo_t;

typedef SnpeUdo_OperationInfo_t Udo_OperationInfo_t;

/**
 * @brief A struct which provides the implementation library info : type, name
 *
 */
typedef struct
{
   /// Defines the IP Core that this implementation library is targeting
   SnpeUdo_CoreType_t     udoCoreType;
   /// library name. will be looked at in the standard library path
   SnpeUdo_String_t       libraryName;
} SnpeUdo_LibraryInfo_t;

typedef SnpeUdo_LibraryInfo_t Udo_LibraryInfo_t;

/**
 * @brief A struct returned by the registration library and contains information on the UDO package :
 * name, operations, libraries, etc.
 *
 */
typedef struct
{
   /// A string containing the package name
   SnpeUdo_String_t         packageName;
   /// A bitmask describing supported IP cores (CPU, GPU, DSP ...)
   /// Translated based on SnpeUdo_CoreType
   SnpeUdo_Bitmask_t        supportedCoreTypes;
   /// The number of implementation libraries in the package
   uint32_t                numOfImplementationLib;
   /// Array of implementation libraries names/types
   SnpeUdo_LibraryInfo_t*   implementationLib;
   /// A string containing all operation types separated by space
   SnpeUdo_String_t         operationsString;
   /// Number of supported operations
   uint32_t                numOfOperations;
   /// Array of Operation info structs. Each entry describes one
   /// Operation (name, params, inputs, outputs)
   SnpeUdo_OperationInfo_t* operationsInfo;
} SnpeUdo_RegInfo_t;

typedef SnpeUdo_RegInfo_t Udo_RegInfo_t;

/**
* @brief A struct returned by the implementation library and contains information on the
* specific library: name, IP Core, operations, etc.
*
*/
typedef struct
{
   /// Defines the IP Core that this implementation library is targeting
   SnpeUdo_CoreType_t     udoCoreType;
   /// A string containing the package name
   SnpeUdo_String_t       packageName;
   /// A string containing all operation types separated by space
   SnpeUdo_String_t       operationsString;
   /// Number of supported operations
   uint32_t              numOfOperations;
} SnpeUdo_ImpInfo_t;

typedef SnpeUdo_ImpInfo_t Udo_ImpInfo_t;

/**
 * @brief This struct defines an operation. It is used for validation
 * or creation of an operation.
 * In case of using it for creation, the static params which are tensors
 * contain pointers to the real data (weights, for example), and input/output
 * tensors also include pointers to the buffers used.
 */
typedef struct
{
   /// The IP Core that the operation is defined for - CPU, GPU, DSP...
   SnpeUdo_CoreType_t      udoCoreType;
   /// Operation type
   SnpeUdo_String_t        operationType;
   /// The number of static parameters provided in the staticParams array.
   /// this number has to match the number provided by the UDO Registration library information
   uint32_t               numOfStaticParams;
   /// Array of static parameters
   SnpeUdo_Param_t*        staticParams;
   /// The number of input parameters provided in inputs array.
   /// this number has to match the number provided by the UDO Registration library information
   uint32_t               numOfInputs;
   /// Array of input tensors, providing layout, data type, sizes, etc
   /// When used to create an operation, also contains the initial location of the data
   SnpeUdo_TensorParam_t*  inputs;
   /// The number of output parameters provided in inputs array.
   /// this number has to match the number provided by the UDO Registration library information
   uint32_t               numOfOutputs;
   /// Array of output tensors, providing layout, data type, sizes, etc
   /// When used to create an operation, also contains the initial location of the data
   SnpeUdo_TensorParam_t*  outputs;
} SnpeUdo_OpDefinition_t;

typedef SnpeUdo_OpDefinition_t Udo_OpDefinition_t;

/** @} */ /* end_addtogroup c_plus_plus_apis C++ */

#endif //SNPE_UDO_BASE_H
