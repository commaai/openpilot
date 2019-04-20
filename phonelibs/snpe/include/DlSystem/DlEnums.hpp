//==============================================================================
//
//  Copyright (c) 2014-2018 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef _DL_ENUMS_HPP_
#define _DL_ENUMS_HPP_

#include "DlSystem/ZdlExportDefine.hpp"


namespace zdl {
namespace DlSystem
{
/** @addtogroup c_plus_plus_apis C++
@{ */

/**
 * Enumeration of supported target runtimes.
 */
enum class Runtime_t
{
   /// Run the processing on Snapdragon CPU.
   /// Data: float 32bit
   /// Math: float 32bit
   CPU_FLOAT32  = 0,

   /// Run the processing on the Adreno GPU.
   /// Data: float 16bit
   /// Math: float 32bit
   GPU_FLOAT32_16_HYBRID = 1,

   /// Run the processing on the Hexagon DSP.
   /// Data: 8bit fixed point Tensorflow style format
   /// Math: 8bit fixed point Tensorflow style format
   DSP_FIXED8_TF = 2,

   /// Run the processing on the Adreno GPU.
   /// Data: float 16bit
   /// Math: float 16bit
   GPU_FLOAT16 = 3,

   /// Default legacy enum to retain backward compatibility.
   /// CPU = CPU_FLOAT32
   CPU = CPU_FLOAT32,

   /// Default legacy enum to retain backward compatibility.
   /// GPU = GPU_FLOAT32_16_HYBRID
   GPU = GPU_FLOAT32_16_HYBRID,

   /// Default legacy enum to retain backward compatibility.
   /// DSP = DSP_FIXED8_TF
   DSP = DSP_FIXED8_TF
};

/**
 * Enumeration of various performance profiles that can be requested.
 */
enum class PerformanceProfile_t
{
    /// Run in a standard mode.
    /// This mode will be deprecated in the future and replaced with BALANCED.
    DEFAULT = 0,
    /// Run in a balanced mode.
    BALANCED = 0,

    /// Run in high performance mode
    HIGH_PERFORMANCE,

    /// Run in a power sensitive mode, at the expense of performance.
    POWER_SAVER,

    /// Use system settings.  SNPE makes no calls to any performance related APIs.
    SYSTEM_SETTINGS,

    /// Run in sustained high performance mode
    SUSTAINED_HIGH_PERFORMANCE

};

/**
 * Enumeration of various execution priority hints.
 */
enum class ExecutionPriorityHint_t
{
    /// Normal priority
    NORMAL = 0,

    /// Higher than normal priority
    HIGH,

    /// Lower priority
    LOW

};

/** @} */ /* end_addtogroup c_plus_plus_apis C++*/

/**
 * Enumeration that lists the supported image encoding formats.
 */
enum class ImageEncoding_t
{
   /// For unknown image type. Also used as a default value for ImageEncoding_t.
   UNKNOWN = 0,

   /// The RGB format consists of 3 bytes per pixel: one byte for
   /// Red, one for Green, and one for Blue. The byte ordering is
   /// endian independent and is always in RGB byte order.
   RGB,

   /// The ARGB32 format consists of 4 bytes per pixel: one byte for
   /// Red, one for Green, one for Blue, and one for the alpha channel.
   /// The alpha channel is ignored. The byte ordering depends on the
   /// underlying CPU. For little endian CPUs, the byte order is BGRA.
   /// For big endian CPUs, the byte order is ARGB.
   ARGB32,

   /// The RGBA format consists of 4 bytes per pixel: one byte for
   /// Red, one for Green, one for Blue, and one for the alpha channel.
   /// The alpha channel is ignored. The byte ordering is endian independent
   /// and is always in RGBA byte order.
   RGBA,

   /// The GRAYSCALE format is for 8-bit grayscale.
   GRAYSCALE,

   /// NV21 is the Android version of YUV. The Chrominance is down
   /// sampled and has a subsampling ratio of 4:2:0. Note that this
   /// image format has 3 channels, but the U and V channels
   /// are subsampled. For every four Y pixels there is one U and one V pixel. @newpage
   NV21,

   /// The BGR format consists of 3 bytes per pixel: one byte for
   /// Red, one for Green and one for Blue. The byte ordering is
   /// endian independent and is always BGR byte order.
   BGR
};

}} // namespaces end


#endif
