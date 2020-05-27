/**
 * Copyright (c) 2008 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/**
 * @file OMX_Image.h - OpenMax IL version 1.1.2
 * The structures needed by Image components to exchange parameters and
 * configuration data with the components.
 */
#ifndef OMX_Image_h
#define OMX_Image_h

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


/**
 * Each OMX header must include all required header files to allow the
 * header to compile without errors.  The includes below are required
 * for this header file to compile successfully
 */

#include <OMX_IVCommon.h>

/** @defgroup imaging OpenMAX IL Imaging Domain
 * @ingroup iv
 * Structures for OpenMAX IL Imaging domain
 * @{
 */

/**
 * Enumeration used to define the possible image compression coding.
 */
typedef enum OMX_IMAGE_CODINGTYPE {
    OMX_IMAGE_CodingUnused,      /**< Value when format is N/A */
    OMX_IMAGE_CodingAutoDetect,  /**< Auto detection of image format */
    OMX_IMAGE_CodingJPEG,        /**< JPEG/JFIF image format */
    OMX_IMAGE_CodingJPEG2K,      /**< JPEG 2000 image format */
    OMX_IMAGE_CodingEXIF,        /**< EXIF image format */
    OMX_IMAGE_CodingTIFF,        /**< TIFF image format */
    OMX_IMAGE_CodingGIF,         /**< Graphics image format */
    OMX_IMAGE_CodingPNG,         /**< PNG image format */
    OMX_IMAGE_CodingLZW,         /**< LZW image format */
    OMX_IMAGE_CodingBMP,         /**< Windows Bitmap format */
    OMX_IMAGE_CodingKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_IMAGE_CodingVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_IMAGE_CodingMax = 0x7FFFFFFF
} OMX_IMAGE_CODINGTYPE;


/**
 * Data structure used to define an image path. The number of image paths
 * for input and output will vary by type of the image component.
 *
 *  Input (aka Source) : Zero Inputs, one Output,
 *  Splitter           : One Input, 2 or more Outputs,
 *  Processing Element : One Input, one output,
 *  Mixer              : 2 or more inputs, one output,
 *  Output (aka Sink)  : One Input, zero outputs.
 *
 * The PortDefinition structure is used to define all of the parameters
 * necessary for the compliant component to setup an input or an output
 * image path.  If additional vendor specific data is required, it should
 * be transmitted to the component using the CustomCommand function.
 * Compliant components will prepopulate this structure with optimal
 * values during the OMX_GetParameter() command.
 *
 * STRUCT MEMBERS:
 *  cMIMEType             : MIME type of data for the port
 *  pNativeRender         : Platform specific reference for a display if a
 *                          sync, otherwise this field is 0
 *  nFrameWidth           : Width of frame to be used on port if
 *                          uncompressed format is used.  Use 0 for
 *                          unknown, don't care or variable
 *  nFrameHeight          : Height of frame to be used on port if
 *                          uncompressed format is used. Use 0 for
 *                          unknown, don't care or variable
 *  nStride               : Number of bytes per span of an image (i.e.
 *                          indicates the number of bytes to get from
 *                          span N to span N+1, where negative stride
 *                          indicates the image is bottom up
 *  nSliceHeight          : Height used when encoding in slices
 *  bFlagErrorConcealment : Turns on error concealment if it is supported by
 *                          the OMX component
 *  eCompressionFormat    : Compression format used in this instance of
 *                          the component. When OMX_IMAGE_CodingUnused is
 *                          specified, eColorFormat is valid
 *  eColorFormat          : Decompressed format used by this component
 *  pNativeWindow         : Platform specific reference for a window object if a
 *                          display sink , otherwise this field is 0x0.
 */
typedef struct OMX_IMAGE_PORTDEFINITIONTYPE {
    OMX_STRING cMIMEType;
    OMX_NATIVE_DEVICETYPE pNativeRender;
    OMX_U32 nFrameWidth;
    OMX_U32 nFrameHeight;
    OMX_S32 nStride;
    OMX_U32 nSliceHeight;
    OMX_BOOL bFlagErrorConcealment;
    OMX_IMAGE_CODINGTYPE eCompressionFormat;
    OMX_COLOR_FORMATTYPE eColorFormat;
    OMX_NATIVE_WINDOWTYPE pNativeWindow;
} OMX_IMAGE_PORTDEFINITIONTYPE;


/**
 * Port format parameter.  This structure is used to enumerate the various
 * data input/output format supported by the port.
 *
 * STRUCT MEMBERS:
 *  nSize              : Size of the structure in bytes
 *  nVersion           : OMX specification version information
 *  nPortIndex         : Indicates which port to set
 *  nIndex             : Indicates the enumeration index for the format from
 *                       0x0 to N-1
 *  eCompressionFormat : Compression format used in this instance of the
 *                       component. When OMX_IMAGE_CodingUnused is specified,
 *                       eColorFormat is valid
 *  eColorFormat       : Decompressed format used by this component
 */
typedef struct OMX_IMAGE_PARAM_PORTFORMATTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nIndex;
    OMX_IMAGE_CODINGTYPE eCompressionFormat;
    OMX_COLOR_FORMATTYPE eColorFormat;
} OMX_IMAGE_PARAM_PORTFORMATTYPE;


/**
 * Flash control type
 *
 * ENUMS
 *  Torch : Flash forced constantly on
 */
typedef enum OMX_IMAGE_FLASHCONTROLTYPE {
    OMX_IMAGE_FlashControlOn = 0,
    OMX_IMAGE_FlashControlOff,
    OMX_IMAGE_FlashControlAuto,
    OMX_IMAGE_FlashControlRedEyeReduction,
    OMX_IMAGE_FlashControlFillin,
    OMX_IMAGE_FlashControlTorch,
    OMX_IMAGE_FlashControlKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_IMAGE_FlashControlVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_IMAGE_FlashControlMax = 0x7FFFFFFF
} OMX_IMAGE_FLASHCONTROLTYPE;


/**
 * Flash control configuration
 *
 * STRUCT MEMBERS:
 *  nSize         : Size of the structure in bytes
 *  nVersion      : OMX specification version information
 *  nPortIndex    : Port that this structure applies to
 *  eFlashControl : Flash control type
 */
typedef struct OMX_IMAGE_PARAM_FLASHCONTROLTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_IMAGE_FLASHCONTROLTYPE eFlashControl;
} OMX_IMAGE_PARAM_FLASHCONTROLTYPE;


/**
 * Focus control type
 */
typedef enum OMX_IMAGE_FOCUSCONTROLTYPE {
    OMX_IMAGE_FocusControlOn = 0,
    OMX_IMAGE_FocusControlOff,
    OMX_IMAGE_FocusControlAuto,
    OMX_IMAGE_FocusControlAutoLock,
    OMX_IMAGE_FocusControlKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_IMAGE_FocusControlVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_IMAGE_FocusControlMax = 0x7FFFFFFF
} OMX_IMAGE_FOCUSCONTROLTYPE;


/**
 * Focus control configuration
 *
 * STRUCT MEMBERS:
 *  nSize           : Size of the structure in bytes
 *  nVersion        : OMX specification version information
 *  nPortIndex      : Port that this structure applies to
 *  eFocusControl   : Focus control
 *  nFocusSteps     : Focus can take on values from 0 mm to infinity.
 *                    Interest is only in number of steps over this range.
 *  nFocusStepIndex : Current focus step index
 */
typedef struct OMX_IMAGE_CONFIG_FOCUSCONTROLTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_IMAGE_FOCUSCONTROLTYPE eFocusControl;
    OMX_U32 nFocusSteps;
    OMX_U32 nFocusStepIndex;
} OMX_IMAGE_CONFIG_FOCUSCONTROLTYPE;


/**
 * Q Factor for JPEG compression, which controls the tradeoff between image
 * quality and size.  Q Factor provides a more simple means of controlling
 * JPEG compression quality, without directly programming Quantization
 * tables for chroma and luma
 *
 * STRUCT MEMBERS:
 *  nSize      : Size of the structure in bytes
 *  nVersion   : OMX specification version information
 *  nPortIndex : Port that this structure applies to
 *  nQFactor   : JPEG Q factor value in the range of 1-100. A factor of 1
 *               produces the smallest, worst quality images, and a factor
 *               of 100 produces the largest, best quality images.  A
 *               typical default is 75 for small good quality images
 */
typedef struct OMX_IMAGE_PARAM_QFACTORTYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_U32 nQFactor;
} OMX_IMAGE_PARAM_QFACTORTYPE;

/**
 * Quantization table type
 */

typedef enum OMX_IMAGE_QUANTIZATIONTABLETYPE {
    OMX_IMAGE_QuantizationTableLuma = 0,
    OMX_IMAGE_QuantizationTableChroma,
    OMX_IMAGE_QuantizationTableChromaCb,
    OMX_IMAGE_QuantizationTableChromaCr,
    OMX_IMAGE_QuantizationTableKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_IMAGE_QuantizationTableVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_IMAGE_QuantizationTableMax = 0x7FFFFFFF
} OMX_IMAGE_QUANTIZATIONTABLETYPE;

/**
 * JPEG quantization tables are used to determine DCT compression for
 * YUV data, as an alternative to specifying Q factor, providing exact
 * control of compression
 *
 * STRUCT MEMBERS:
 *  nSize                   : Size of the structure in bytes
 *  nVersion                : OMX specification version information
 *  nPortIndex              : Port that this structure applies to
 *  eQuantizationTable      : Quantization table type
 *  nQuantizationMatrix[64] : JPEG quantization table of coefficients stored
 *                            in increasing columns then by rows of data (i.e.
 *                            row 1, ... row 8). Quantization values are in
 *                            the range 0-255 and stored in linear order
 *                            (i.e. the component will zig-zag the
 *                            quantization table data if required internally)
 */
typedef struct OMX_IMAGE_PARAM_QUANTIZATIONTABLETYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_IMAGE_QUANTIZATIONTABLETYPE eQuantizationTable;
    OMX_U8 nQuantizationMatrix[64];
} OMX_IMAGE_PARAM_QUANTIZATIONTABLETYPE;


/**
 * Huffman table type, the same Huffman table is applied for chroma and
 * luma component
 */
typedef enum OMX_IMAGE_HUFFMANTABLETYPE {
    OMX_IMAGE_HuffmanTableAC = 0,
    OMX_IMAGE_HuffmanTableDC,
    OMX_IMAGE_HuffmanTableACLuma,
    OMX_IMAGE_HuffmanTableACChroma,
    OMX_IMAGE_HuffmanTableDCLuma,
    OMX_IMAGE_HuffmanTableDCChroma,
    OMX_IMAGE_HuffmanTableKhronosExtensions = 0x6F000000, /**< Reserved region for introducing Khronos Standard Extensions */
    OMX_IMAGE_HuffmanTableVendorStartUnused = 0x7F000000, /**< Reserved region for introducing Vendor Extensions */
    OMX_IMAGE_HuffmanTableMax = 0x7FFFFFFF
} OMX_IMAGE_HUFFMANTABLETYPE;

/**
 * JPEG Huffman table
 *
 * STRUCT MEMBERS:
 *  nSize                            : Size of the structure in bytes
 *  nVersion                         : OMX specification version information
 *  nPortIndex                       : Port that this structure applies to
 *  eHuffmanTable                    : Huffman table type
 *  nNumberOfHuffmanCodeOfLength[16] : 0-16, number of Huffman codes of each
 *                                     possible length
 *  nHuffmanTable[256]               : 0-255, the size used for AC and DC
 *                                     HuffmanTable are 16 and 162
 */
typedef struct OMX_IMAGE_PARAM_HUFFMANTTABLETYPE {
    OMX_U32 nSize;
    OMX_VERSIONTYPE nVersion;
    OMX_U32 nPortIndex;
    OMX_IMAGE_HUFFMANTABLETYPE eHuffmanTable;
    OMX_U8 nNumberOfHuffmanCodeOfLength[16];
    OMX_U8 nHuffmanTable[256];
}OMX_IMAGE_PARAM_HUFFMANTTABLETYPE;

/** @} */
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
/* File EOF */
