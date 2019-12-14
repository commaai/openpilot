#ifndef FASTCV_H
#define FASTCV_H

/**=============================================================================

@file
   fastcv.h

@brief
   Public API


Copyright (c) 2011-2013 Qualcomm Technologies Incorporated.
All Rights Reserved Qualcomm Technologies Proprietary

Export of this technology or software is regulated by the U.S.
Government. Diversion contrary to U.S. law prohibited.

All ideas, data and information contained in or disclosed by
this document are confidential and proprietary information of
Qualcomm Technologies Incorporated and all rights therein are expressly reserved.
By accepting this material the recipient agrees that this material
and the information contained therein are held in confidence and in
trust and will not be used, copied, reproduced in whole or in part,
nor its contents revealed in any manner to others without the express
written permission of Qualcomm Technologies Incorporated.

=============================================================================**/

/**=============================================================================
@mainpage FastCV Public API Documentation

@version 1.2.2

@section Overview Overview

FastCV provides two main features to computer vision application developers:
   - First, it provides a library of  frequently used computer vision (CV)
   functions, optimized to run efficiently on mobile devices.
   - Second, it provides a clean processor-agnostic hardware acceleration API,
   under which  chipset vendors can hardware accelerate FastCV functions on
   their hardware.

This initial release (FastCV 1.0) only supports Android mobile developers;
however, we intend to support iOS and Windows devices as soon as possible.
FastCV 1.0 is available for download for free from developer.qualcomm.com.

FastCV 1.0 is released as a unified binary, a single binary containing two
implementations of the library.
   - The first implementation runs on ARM processor, and is referred to as
   the "FastCV for ARM."
   - The second implementation runs only on Qualcomm Snapdragon 
   chipsets, and is called "FastCV for Snapdragon."

Releases are generally motivated for the following reasons:
   - Changes to previously released APIs
   - Addition of new functions
   - Performance improvements and/or bug fixes - also known as implementation 
     modifications

    Each motivation has a varying degree of impact on the user of the library.
    The general release numbering scheme captures this variety of motivations.

    Given release ID: A.B.C

    An increase in "A" indicates that a previously released API has changed, 
    so a developer may encounter compilation issues which require modification 
    of their code in order to adhear to the modified API.  Qualcomm will make 
    every effort to minimize these changes.  Additionally, new functions and 
    implementation modifications may be present.

    An increase in "B" indicates that new functions have been added to the 
    library, so additional functionality is available, however existing APIs 
    have not changed.  Additionally, implementation modifications may be 
    present.

    An increase in "C" indicates that implementation modifications only have 
    been made.

@defgroup math_vector Math / Vector Operations
@details Commonly used vector & math functions

@defgroup image_processing Image processing
@details Image filtering, convolution and scaling operations

@defgroup image_transform Image transformation
@details Warp perspective, affine transformations

@defgroup feature_detection Feature detection
@details Fast corner detection, harris corner detection, canny edge detection, etc.

@defgroup object_detection Object detection
@details Object detection functions such as NCC template match, etc.

@defgroup 3D_reconstruction 3D reconstruction
@details Homography, pose evaluation functions

@defgroup color_conversion Color conversion
@details Commonly used formats supported: e.g., YUV, RGB, YCrCb, etc.

@defgroup clustering_and_search Clustering and search
@details K clusters best fitting of a set of input points

@defgroup Motion_and_Object_Tracking Motion and object tracking
@details Supports and tracking functions
    
@defgroup Structural_Analysis_and_Drawing Shape and drawing
@details Contour and polygon drawing functions
    
@defgroup mem_management Memory Management
@details Functions to allocate and deallocate memory for use with fastCV.

@defgroup misc Miscellaneous 
@details Support functions 
 
**/

//==============================================================================
// Defines
//==============================================================================

#define FASTCV_VERSION  122

#ifdef __GNUC__
   /// Macro to align memory at 4-bytes (32-bits) for GNU-based compilers.
   #define FASTCV_ALIGN32( VAR ) (VAR)  __attribute__ ((aligned(4)))
   /// Macro to align memory at 8-bytes (64-bits) for GNU-based compilers.
   #define FASTCV_ALIGN64( VAR )  (VAR) __attribute__ ((aligned(8)))
   /// Macro to align memory at 16-bytes (128-bits) for GNU-based compilers.
   #define FASTCV_ALIGN128( VAR ) (VAR) __attribute__ ((aligned(16)))
   #ifdef BUILDING_SO
   /// MACRO enables function to be visible in shared-library case.
   #define FASTCV_API __attribute__ ((visibility ("default")))
   #else
   /// MACRO empty for non-shared-library case.
   #define FASTCV_API
   #endif
#else
   /// Macro to align memory at 4-bytes (32-bits) for MSVC compiler.
   #define FASTCV_ALIGN32( VAR ) __declspec(align(4)) (VAR)
   /// Macro to align memory at 8-bytes (64-bits) for MSVC compiler.
   #define FASTCV_ALIGN64( VAR ) __declspec(align(8)) (VAR)
   /// Macro to align memory at 16-bytes (128-bits) for MSVC compiler.
   #define FASTCV_ALIGN128( VAR ) __declspec(align(16)) (VAR)
   #ifdef BUILDING_DLL
   /// MACRO enables function to be visible in shared-library case.
   #define FASTCV_API __declspec(dllexport)
   #else
   /// MACRO empty for non-shared-library case.
   #define FASTCV_API
   #endif
#endif

//==============================================================================
// Included modules
//==============================================================================

#include <stddef.h>

#ifndef FASTCV_STDINT
#define FASTCV_STDINT
   #ifdef _MSC_VER

      #if _MSC_VER <= 1500
         // stdint.h support for VS2008 and older
         #include "stdint_.h"
      #else
         #include <stdint.h>
      #endif

      typedef float  float32_t;
      typedef double float64_t;

   #else

      #ifdef __ARM_NEON__
         #include <arm_neon.h>
      #else
         #include <stdint.h>
         typedef float  float32_t;
         typedef double float64_t;
      #endif

   #endif
#endif

//==============================================================================
// Declarations
//==============================================================================


//------------------------------------------------------------------------------
/// @brief
///    Defines operational mode of interface to allow the end developer to
///    dictate how the target optimized implementation should behave.
//------------------------------------------------------------------------------
typedef enum
{
   /// Target-optimized implementation uses lowest power consuming
   /// implementation.
   FASTCV_OP_LOW_POWER       = 0,

   /// Target-optimized implementation uses higheset performance implementation.
   FASTCV_OP_PERFORMANCE     = 1,

   /// Target-optimized implementation offloads as much of the CPU as possible.
   FASTCV_OP_CPU_OFFLOAD     = 2,

   /// Values >= 0x80000000 are reserved
   FASTCV_OP_RESERVED        = 0x80000000

} fcvOperationMode;


//------------------------------------------------------------------------------
/// @brief
///   Defines a structure to contain points correspondence data.
//------------------------------------------------------------------------------
typedef struct
{
   ///   Tuples of 3 values: xFrom,yFrom,zFrom. Float array which this points to
   ///   must be greater than or equal to 3 * numCorrespondences.
   const float32_t*               from;
   /*~ FIELD fcvCorrespondences.from
       VARRAY LENGTH ( fcvCorrespondences.numCorrespondences * \
       (fcvCorrespondences.fromStride ? fcvCorrespondences.fromStride : 3) ) */

   ///   Tuples of 2 values: xTo,yTo. Float array which this points to
   ///   must be greater than or equal to 2 * numCorrespondences.
   const float32_t*               to;
   /*~ FIELD fcvCorrespondences.to
       VARRAY LENGTH ( fcvCorrespondences.numCorrespondences * \
       (fcvCorrespondences.toStride ? fcvCorrespondences.toStride : 2) ) */

   ///   Distance in bytes between two coordinates in the from array.
   ///   If this parameter is set to 2 or 3, a dense array is assume (stride will
   ///   be sizeof(float) times 2 or 3). The minimum value of fromStride
   ///   should be 2.
   uint32_t                       fromStride;

   ///   Distance in bytes between two coordinates in the to array.
   ///   If this parameter is set to 2, a dense array is assume (stride will
   ///   be 2 * sizeof(float)). The minimum value of toStride
   ///   should be 2.
   uint32_t                       toStride;

   ///   Number of points in points correspondences.
   uint32_t                       numCorrespondences;

   ///   Array of inlier indices for corrs array. Processing will only occur on
   ///   the indices supplied in this array. Array which this points to must be
   ///   at least numIndices long.
   const uint16_t*                indices;
   /*~ FIELD fcvCorrespondences.indices VARRAY LENGTH (fcvCorrespondences.numIndices) */

   ///   Length of indices array.
   uint32_t                       numIndices;
} fcvCorrespondences;


// -----------------------------------------------------------------------------
/// @brief
///   Structure representing an image pyramid level
//------------------------------------------------------------------------------

typedef struct
{
   const void* ptr;
   unsigned int width;
   unsigned int height;
} fcvPyramidLevel ;

// -----------------------------------------------------------------------------
/// @brief
///   Structure describing node of a tree;
///   Assumption is that nodes of all trees are stored in in a single array
///   and all indices refer to this array
/// @remark
///   if indices of both children are negative the node is a leaf
// ----------------------------------------------------------------------------
typedef struct fcvKDTreeNodef32
{
   /// the split value at the node
   float32_t divVal;

   /// dimension at which the split is made;
   /// if this is a leaf (both children equal to -1) then this is
   /// the index of the dataset vector
   int32_t divFeat;

   /// index of the child node with dataset items to the left
   /// of the split value
   int32_t childLeft;

   /// index of the child node with dataset items to the right
   /// of the split value
   int32_t childRight;

} fcvKDTreeNodef32;

// -----------------------------------------------------------------------------
/// @brief
///   structure describing a branch (subtree)
/// @remark
///   branches are stored on the priority queue (heap) for backtracking
// -----------------------------------------------------------------------------
typedef struct fcvKDTreeBranchf32
{
   /// square of minimal distance from query for all nodes below
   float32_t minDistSq;

   /// index of the top node of the branch
   int32_t topNode;

} fcvKDTreeBranchf32;

// -----------------------------------------------------------------------------
/// @brief
///   Structure with KDTrees data
// -----------------------------------------------------------------------------
typedef struct fcvKDTreeDatas8f32
{
   // info about the dataset for which KDTrees are constructed
   /// the dataset of vectors
   const int8_t *dataset;

   /// array with inverse lengths of dataset vectors
   const float32_t* invLen;

   /// number of vectors in the dataset
   int32_t numVectors;

   // info about trees
   /// indice of root nodes of trees
   int32_t* trees;

   /// array of nodes of all trees
   fcvKDTreeNodef32* nodes;

   /// number of all nodes
   int32_t numNodes;

   /// capacity of node array
   int32_t maxNumNodes;

   // info used during lookups
   /// priority queue
   fcvKDTreeBranchf32* heap;

   /// number of branches on the priority queue
   int32_t numBranches;

   /// capactiy of the priority queue
   int32_t maxNumBranches;

   /// array of indices to vectors in the dataset;
   /// during searches used to mark checkID;
   /// should have numVectors capacity
   int32_t* vind;

   /// unique ID for each lookup
   int32_t checkID;

   /// number of nearest neighbors to find
   int32_t numNNs;

} fcvKDTreeDatas8f32;


// -----------------------------------------------------------------------------
/// @brief
///   fixed point kdtrees
///   Structure describing node of tree;
///   Assumption is that nodes of all trees are stored in in a single array
///   and all indices refer to this array 
/// @remark
///   if indices of both children are negative the node is a leaf
// ----------------------------------------------------------------------------
typedef struct fcvKDTreeNodes32
{
   /// the split value at the node
   int32_t divVal;

   /// dimension at which the split is made;
   /// if this is a leaf (both children equal to -1) then this is
   /// the index of the dataset vector
   int32_t divFeat;

   /// index of the child node with dataset items to the left
   /// of the split value
   int32_t childLeft;

   /// index of the child node with dataset items to the right
   /// of the split value
   int32_t childRight;

} fcvKDTreeNodes32;

// -----------------------------------------------------------------------------
/// @brief
///   fixed point kdtrees
///   structure describing a branch (subtree)
/// @remark
///   branches are stored on the priority queue (heap) for backtracking
// -----------------------------------------------------------------------------
typedef struct fcvKDTreeBranchs32
{
   /// square of minimal distance from query for all nodes below
   int32_t minDistSq;

   /// index of the top node of the branch
   int32_t topNode;

} fcvKDTreeBranchs32;

// -----------------------------------------------------------------------------
/// @brief
///   fixed point kdtrees
///   Structure with KDTrees data
// -----------------------------------------------------------------------------
typedef struct fcvKDTreeDatas8s32
{
   // info about the dataset for which KDTrees are constructed
   /// the dataset of vectors
   const int8_t *dataset;

   /// array with inverse lengths of dataset vectors
   const int32_t* invLen;

   /// number of vectors in the dataset
   int32_t numVectors;

   // info about trees
   /// indices of root nodes of all trees
   int32_t* trees;

   /// number of trees used
   int32_t numTrees;

   /// array of nodes of all trees
   fcvKDTreeNodes32* nodes;

   /// number of all nodes
   int32_t numNodes;

   /// capacity of node array
   int32_t maxNumNodes;

   // info used during lookups
   /// priority queue
   fcvKDTreeBranchs32* heap;

   /// number of branches on the priority queue
   int32_t numBranches;

   /// capactiy of the priority queue
   int32_t maxNumBranches;

   /// array of indices to vectors in the dataset;
   /// during searches used to mark checkID;
   /// should have numVectors capacity
   int32_t* vind;

   /// unique ID for each lookup
   int32_t checkID;

   /// number of nearest neighbors to find
   int32_t numNNs;

} fcvKDTreeDatas8s32;

//------------------------------------------------------------------------------
/// @brief
///     Defines a struct of rectangle
//------------------------------------------------------------------------------
typedef struct
{
    ///x-coordinate of the top-left corner
    int32_t x;
    ///y-coordinate of the top-left corner    
    int32_t y;
    ///width of the rectangle
    uint32_t width;
    ///height of the rectangle
    uint32_t height;
} fcvRectangleInt;

//------------------------------------------------------------------------------
/// @brief
///     Defines a struct of termination criteria
//------------------------------------------------------------------------------
typedef struct
{
    /// Maxmimum number of iteration
    int32_t      max_iter;
    /// 
    float32_t    epsilon;
}fcvTermCriteria;

//------------------------------------------------------------------------------
/// @brief
///     Defines a struct of 2D box used for tracking
//------------------------------------------------------------------------------
typedef struct 
{
    // Center of the box
    ///x-coordinate of the 2D point
    int32_t x;
    ///y-coordinate of the 2D point
    int32_t y;
    // The box size
    int32_t    columns;
    int32_t    rows;
    // The orientation of the principal axis
    int32_t orientation;
}fcvBox2D;

//------------------------------------------------------------------------------
/// @brief
///     Defines a struct of code word
//------------------------------------------------------------------------------
typedef struct fcvBGCodeWord
{
    /// Pointer to next codebook element
    struct fcvBGCodeWord* next;

    /// Last update time
    int32_t tLastUpdate;

    /// Longest period of inactivity
    int32_t stale;
    /// Min value of pixel for each channel
    uint8_t min0, min1, min2;

    /// Max value of pixel for each channel
    uint8_t max0, max1, max2;

    /// Min value of learning boundary for each channel
    uint8_t learnLow0, learnLow1, learnLow2;

    /// Max value of learning boundary for each channel
    uint8_t learnHigh0, learnHigh1, learnHigh2;
} fcvBGCodeWord;

//------------------------------------------------------------------------------
/// @brief
///     Defines a struct for circle
//------------------------------------------------------------------------------
typedef struct fcvCircle
{
    int32_t x;
    int32_t y;
    int32_t radius;
} fcvCircle;

//==============================================================================
// UTILITY FUNCTIONS
//==============================================================================

#ifdef __cplusplus
extern "C"
{
#endif

//------------------------------------------------------------------------------
/// @brief
///   Retrieves version of FastCV library.
///
/// @param version
///   Pointer to location to put string.
///
/// @param versionLength
///   Length of storage for version string.
///
/// @ingroup misc
//------------------------------------------------------------------------------

FASTCV_API void
fcvGetVersion( char*        version,
               unsigned int versionLength );


//---------------------------------------------------------------------------
/// @brief
///   Selects HW units for all routines at run-time.  Can be called anytime to
///   update choice.
///
/// @param mode
///   See enum for details.
///
/// @return
///   0 if successful.
///   999 if minmum HW requirement not met.
///   other #'s if unsuccessful.
///
/// @ingroup misc
//---------------------------------------------------------------------------

FASTCV_API int
fcvSetOperationMode( fcvOperationMode mode );


//---------------------------------------------------------------------------
/// @brief
///    Clean up FastCV resources. Must be called before the program exits.
///
/// @ingroup misc
//---------------------------------------------------------------------------

FASTCV_API void
fcvCleanUp( void );


// -----------------------------------------------------------------------------
/// @brief
///   Allocates memory for Pyramid
///
/// @param pyr
///   Pointer to an array of qcvaPyramidLevel
///
/// @param baseWidth
///   Width of the base level: the value assigned to pyr[0].width
///
/// @param baseHeight
///   Height of the base level: the value assigned to pyr[0].height
///
/// @param bytesPerPixel
///   Number of bytes per pixel:
///   \n e.g for uint8_t pyramid,  bytesPerPixel = 1
///   \n for int32_t pyramid, bytesPerPixel = 4
///
/// @param numLevels
///   number of levels in the pyramid
///
/// @param allocateBase
///   \n if set to 1, memory will be allocated for the base level
///   \n if set to 0, memory for the base level is allocated by the callee
///   \n\b WARNING: How this parameter is set will impact how the memory is freed.
///                 Please refer to fcvPyramidDelete for details.
///
/// @ingroup mem_management
//----------------------------------------------------------------------------

FASTCV_API int
fcvPyramidAllocate( fcvPyramidLevel* pyr,
                    unsigned int     baseWidth,
                    unsigned int     baseHeight,
                    unsigned int     bytesPerPixel,
                    unsigned int     numLevels,
                    int              allocateBase );


// -----------------------------------------------------------------------------
/// @brief
///   Deallocates an array of fcvPyramidLevel. Can be used for any
///   type(f32/s8/u8).
///
/// @param pyr
///   pyramid to deallocate
///
/// @param numLevels
///   Number of levels in the pyramid
/// 
/// @param startLevel
///   Start level of the pyramid
///    \n\b WARNING: if pyr was allocated with allocateBase=0 which means baselevel memory
///                  was allocated outside of fcvPyramidAllocate, then startLevel
///                  for fcvPyramidDelete has to be set to 1 (or higher).
///
/// @ingroup mem_management
//----------------------------------------------------------------------------

FASTCV_API void
fcvPyramidDelete( fcvPyramidLevel* pyr,
                  unsigned int     numLevels,
                  unsigned int     startLevel );


//------------------------------------------------------------------------------
/// @brief
///    Allocates aligned memory.
///
/// @param nBytes
///    Number of bytes.
///
/// @param byteAlignment
///    Alignment specified in bytes (e.g., 16 = 128-bit alignment).
///    \n\b WARNING: must be < 255 bytes
///
/// @return
///    SUCCESS: pointer to aligned memory
///    FAILURE: 0
///
/// @ingroup mem_management
//------------------------------------------------------------------------------

FASTCV_API void*
fcvMemAlloc( unsigned int nBytes,
             unsigned int byteAlignment );


//------------------------------------------------------------------------------
/// @brief
///    Frees memory allocated by fcvMemAlloc().
///
/// @param ptr
///    Pointer to memory.
///
/// @ingroup mem_management
//------------------------------------------------------------------------------

FASTCV_API void
fcvMemFree( void* ptr );

#ifdef __cplusplus
}//extern "C"
#endif
//End - Utility functions


//==============================================================================
// FUNCTIONS
//==============================================================================


//------------------------------------------------------------------------------
/// @brief
///   Blurs an image with 3x3 median filter
///
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvFilterMedian3x3u8_v2(). In the 2.0.0 release, 
///   fcvFilterMedian3x3u8_v2 will be renamed to fcvFilterMedian3x3u8
///   and the signature of fcvFilterMedian3x3u8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///   Border values are ignored. The 3x3 mask convolves with the image area
///   | a(1,1)          ,    a12,    ...,   a(1,srcWidth-2)            |\n
///   |       ...       ,    ...,    ...,              ...             |\n
///   | a(srcHeight-2,1),    ...,    ...,   a1(srcHeight-2,srcWidth-2) |\n
///
/// @param srcImg
///   Input 8-bit image. Size of buffer is srcWidth*srcHeight byte.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b NOTE: must be multiple of 8
///
/// @param srcHeight
///   Image height.
///
/// @param dstImg
///   Output 8-bit image. Size of buffer is srcWidth*srcHeight byte.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterMedian3x3u8( const uint8_t* __restrict srcImg,
                      unsigned int              srcWidth,
                      unsigned int              srcHeight,
                      uint8_t* __restrict       dstImg );


//------------------------------------------------------------------------------
/// @brief
///   Blurs an image with 3x3 median filter
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvFilterMedian3x3u8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvFilterMedian3x3u8,
///   \a fcvFilterMedian3x3u8_v2 will be removed, and the current signature
///   for \a fcvFilterMedian3x3u8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvFilterMedian3x3u8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   Border values are ignored. The 3x3 mask convolves with the image area
///   | a(1,1)          ,    a12,    ...,   a(1,srcWidth-2)            |\n
///   |       ...       ,    ...,    ...,              ...             |\n
///   | a(srcHeight-2,1),    ...,    ...,   a1(srcHeight-2,srcWidth-2) |\n
///
/// @param srcImg
///   Input 8-bit image. Size of buffer is srcStride*srcHeight byte.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b NOTE: must be multiple of 8
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcStride
///   Image stride.
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param dstImg
///   Output 8-bit image. Size of buffer is dstStride*srcHeight byte.
///   \n\b NOTE: data should be 128-bit aligned.
/// 
/// @param dstStride
///   Output stride.
///   \n\b NOTE: if 0, dstStride is set as dstWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterMedian3x3u8_v2( const uint8_t* __restrict srcImg,
                         unsigned int              srcWidth,
                         unsigned int              srcHeight,
                         unsigned int              srcStride,
                         uint8_t* __restrict       dstImg,
                         unsigned int              dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Blurs an image with 3x3 Gaussian filter
///
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvFilterGaussian3x3u8_v2(). In the 2.0.0 release, 
///   fcvFilterGaussian3x3u8_v2 will be renamed to fcvFilterGaussian3x3u8
///   and the signature of fcvFilterGaussian3x3u8 as it appears now, 
///   will be removed.
///   \n\n
/// 
/// @details
/// Gaussian kernel:
///   \n 1 2 1
///   \n 2 4 2
///   \n 1 2 1
///
/// @param src
///   Input 8-bit image. Size of buffer is srcWidth*srcHeight byte.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
///
/// @param dst
///   Output 8-bit image. Destination buffer size is srcWidth*srcHeight.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param blurBorder
///   If set to 1, border is blurred by 0-padding adjacent values. If set to 0,
///   borders up to half-kernel width are ignored (e.g. 1 pixel in the 3x3
///   case).
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterGaussian3x3u8( const uint8_t* __restrict src,
                        unsigned int              srcWidth,
                        unsigned int              srcHeight,
                        uint8_t* __restrict       dst,
                        int                       blurBorder );


//------------------------------------------------------------------------------
/// @brief
///   Blurs an image with 3x3 Gaussian filter
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvFilterGaussian3x3u8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvFilterGaussian3x3u8,
///   \a fcvFilterGaussian3x3u8_v2 will be removed, and the current signature
///   for \a fcvFilterGaussian3x3u8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvFilterGaussian3x3u8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   Convolution with 3x3 Gaussian kernel:
///   \n 1 2 1
///   \n 2 4 2
///   \n 1 2 1
///
/// @param src
///   Input 8-bit image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcStride
///   Image stride.
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param dst
///   Output 8-bit image. Size of buffer is dstStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
/// 
/// @param dstStride
///   Output stride.
///   \n\b NOTE: if 0, dstStride is set as dstWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as dstWidth if not 0.
///
/// @param blurBorder
///   If set to 1, border is blurred by 0-padding adjacent values. If set to 0,
///   borders up to half-kernel width are ignored (e.g. 1 pixel in the 3x3
///   case).
///
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterGaussian3x3u8_v2( const uint8_t* __restrict src,
                           unsigned int              srcWidth,
                           unsigned int              srcHeight,
                           unsigned int              srcStride,
                           uint8_t* __restrict       dst,
                           unsigned int              dstStride,
                           int                       blurBorder );


//------------------------------------------------------------------------------
/// @brief
///   Blurs an image with 5x5 Gaussian filter
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvFilterGaussian5x5u8_v2(). In the 2.0.0 release, 
///   fcvFilterGaussian5x5u8_v2 will be renamed to fcvFilterGaussian5x5u8
///   and the signature of fcvFilterGaussian5x5u8 as it appears now, 
///   will be removed.
///   \n\n
/// 
/// @details
///   Convolution with 5x5 Gaussian kernel:
///   \n 1  4  6  4 1
///   \n 4 16 24 16 4
///   \n 6 24 36 24 6
///   \n 4 16 24 16 4
///   \n 1  4  6  4 1
///
/// @param src
///   Input int data (can be sq. of gradient, etc). Size of buffer is srcWidth*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
///
/// @param dst
///   Output 8-bit image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param blurBorder
///   If set to 1, border is blurred by 0-padding adjacent values. If set to 0,
///   borders up to half-kernel width are ignored (e.g. 2 pixel in the 5x5
///   case).
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterGaussian5x5u8( const uint8_t* __restrict src,
                        unsigned int              srcWidth,
                        unsigned int              srcHeight,
                        uint8_t* __restrict       dst,
                        int                       blurBorder );

//------------------------------------------------------------------------------
/// @brief
///   Blurs an image with 5x5 Gaussian filter
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvFilterGaussian5x5u8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvFilterGaussian5x5u8,
///   \a fcvFilterGaussian5x5u8_v2 will be removed, and the current signature
///   for \a fcvFilterGaussian5x5u8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvFilterGaussian5x5u8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   Convolution with 5x5 Gaussian kernel:
///   \n 1  4  6  4 1
///   \n 4 16 24 16 4
///   \n 6 24 36 24 6
///   \n 4 16 24 16 4
///   \n 1  4  6  4 1
///
/// @param src
///   Input int data (can be sq. of gradient, etc).  Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcStride
///   Image stride.
///   \n\b NOTE: if 0, dstStride is set as dstWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param dst
///   Output 8-bit image.  Size of buffer is dstStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
/// 
/// @param dstStride
///   Output stride.
///   \n\b NOTE: if 0, dstStride is set as dstWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as dstWidth if not 0.
///
/// @param blurBorder
///   If set to 1, border is blurred by 0-padding adjacent values. If set to 0,
///   borders up to half-kernel width are ignored (e.g. 2 pixel in the 5x5
///   case).
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterGaussian5x5u8_v2( const uint8_t* __restrict src,
                           unsigned int              srcWidth,
                           unsigned int              srcHeight,
                           unsigned int              srcStride,
                           uint8_t* __restrict       dst,
                           unsigned int              dstStride,
                           int                       blurBorder );


//------------------------------------------------------------------------------
/// @brief
///   Blurs an image with 11x11 Gaussian filter
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvFilterGaussian11x11u8_v2(). In the 2.0.0 release, 
///   fcvFilterGaussian11x11u8_v2 will be renamed to fcvFilterGaussian11x11u8
///   and the signature of fcvFilterGaussian11x11u8 as it appears now, 
///   will be removed.
///   \n\n
///   
/// @details
///   Convolution with 11x11 Gaussian kernel:
///   \n 1     10     45     120     210     252     210     120     45     10     1
///   \n 10    100    450    1200    2100    2520    2100    1200    450    100    10
///   \n 45    450    2025   5400    9450    11340   9450    5400    2025   450    45
///   \n 120   1200   5400   14400   25200   30240   25200   14400   5400   1200   120
///   \n 210   2100   9450   25200   44100   52920   44100   25200   9450   2100   210
///   \n 252   2520   11340  30240   52920   63504   52920   30240   11340  2520   252
///   \n 210   2100   9450   25200   44100   52920   44100   25200   9450   2100   210
///   \n 120   1200   5400   14400   25200   30240   25200   14400   5400   1200   120
///   \n 45    450    2025   5400    9450    11340   9450    5400    2025   450    45
///   \n 10    100    450    1200    2100    2520    2100    1200    450    100    10
///   \n 1     10     45     120     210     252     210     120     45     10 ,   1
///
/// @param src
///   Input 8-bit image.  Size of buffer is srcWidth*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
///
/// @param dst
///   Output 8-bit image.  Size of buffer is srcWidth*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param blurBorder
///   If set to 1, border is blurred by 0-padding adjacent values. If set to 0,
///   borders up to half-kernel width are ignored (e.g. 5 pixel in the 11x11
///   case).
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterGaussian11x11u8( const uint8_t* __restrict src,
                          unsigned int              srcWidth,
                          unsigned int              srcHeight,
                          uint8_t* __restrict       dst,
                          int                       blurBorder );



//------------------------------------------------------------------------------
/// @brief
///   Blurs an image with 11x11 Gaussian filter
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvFilterGaussian11x11u8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvFilterGaussian11x11u8,
///   \a fcvFilterGaussian11x11u8_v2 will be removed, and the current signature
///   for \a fcvFilterGaussian11x11u8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvFilterGaussian11x11u8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   Convolution with 11x11 Gaussian kernel:
///   \n 1     10     45     120     210     252     210     120     45     10     1
///   \n 10    100    450    1200    2100    2520    2100    1200    450    100    10
///   \n 45    450    2025   5400    9450    11340   9450    5400    2025   450    45
///   \n 120   1200   5400   14400   25200   30240   25200   14400   5400   1200   120
///   \n 210   2100   9450   25200   44100   52920   44100   25200   9450   2100   210
///   \n 252   2520   11340  30240   52920   63504   52920   30240   11340  2520   252
///   \n 210   2100   9450   25200   44100   52920   44100   25200   9450   2100   210
///   \n 120   1200   5400   14400   25200   30240   25200   14400   5400   1200   120
///   \n 45    450    2025   5400    9450    11340   9450    5400    2025   450    45
///   \n 10    100    450    1200    2100    2520    2100    1200    450    100    10
///   \n 1     10     45     120     210     252     210     120     45     10 ,   1
///
/// @param src
///   Input 8-bit image.  Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcStride
///   Image stride.
///   \n\b NOTE: if 0, dstStride is set as dstWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param dst
///   Output 8-bit image.  Size of buffer is dstStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
/// 
/// @param dstStride
///   Output stride.
///   \n\b NOTE: if 0, dstStride is set as dstWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param blurBorder
///   If set to 1, border is blurred by 0-padding adjacent values. If set to 0,
///   borders up to half-kernel width are ignored (e.g. 5 pixel in the 11x11
///   case).
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterGaussian11x11u8_v2( const uint8_t* __restrict src,
                             unsigned int              srcWidth,
                             unsigned int              srcHeight,
                             unsigned int              srcStride,
                             uint8_t* __restrict       dst,
                             unsigned int              dstStride,
                             int                       blurBorder );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from YUV (YCrCb) 4:2:0 PesudoPlanar (Interleaved) to RGB 8888.
///      
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvColorYCrCb420PseudoPlanarToRGB8888u8. In the 2.0.0 release, 
///   the signature of fcvColorYUV420toRGB8888u8 as it appears now, 
///   will be removed.
///   \n\n
/// 
/// @param src
///   8-bit image of input YUV 4:2:0 values.
///   \n\b NOTE: must be 128-bit aligned.
///
///   The input are one Y plane followed by one interleaved and 2D (both
///   horizontally and vertically) sub-sampled CrCb plane:
///   Y plane                             : Y00  Y01  Y02  Y03 ...
///                                         Y10  Y11  Y12  Y13 ...
///   Interleaved and 2D sub-sampled plane: Cr0  Cb0  Cr1  Cb1 ...
///
/// @param dst
///   32-bit image of output RGB 8888 values. R is at LSB.
///   \n\b WARNING: size must match input yuv420.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
///
/// 
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYUV420toRGB8888u8( const uint8_t* __restrict src,
                           unsigned int              srcWidth,
                           unsigned int              srcHeight,
                           uint32_t* __restrict      dst );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from YUV (YCrCb) 4:2:0 PesudoPlanar (Interleaved CrCb) to RGB 888.
///
///   \n\b ATTENTION: The name of this function will be changed when the 2.0.0 release 
///   of this library is made.  
///   Until 2.0.0, the developer should use this implementation with the expectation of
///   moving to a different name when transitioning to 2.0.0.
///   \n\n
///  
/// @param src
///   8-bit image of input YUV picture.
///   \n\b NOTE: must be 128-bit aligned.
///
///   The input are one Y plane followed by one interleaved and 2D (both
///   horizontally and vertically) sub-sampled CrCb plane:
///   Y plane                             : Y00  Y01  Y02  Y03 ...
///                                         Y10  Y11  Y12  Y13 ...
///   Interleaved and 2D sub-sampled plane: Cr0  Cb0  Cr1  Cb1 ...
///
/// @param srcWidth
///   Image width.
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcYStride
///   Stride (in bytes) of input image Y component (i.e., number of bytes between 
///   column 0 of row 1 and column 0 of row 2).
///   \n\b WARNING: Must be multiple of 8 (8 * 1-byte values).
///
/// @param srcCStride
///   Stride (in bytes) of input image Chroma component (i.e., number of bytes between 
///   column 0 of row 1 and column 0 of row 2)
///   \n\b WARNING: Must be multiple of 4 (4 * 1-byte values).
/// 
/// @param dst
///   32-bit image of output RGB 8888 values. R in LSB.
///   \n\b WARNING: size must match input yuv420.
///   \n\b NOTE: must be 128-bit aligned.
/// 
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 1 and column 0 of row 2)
///   \n\b WARNING: Must be multiple of 32 (8 * 4-byte values).
///
/// 
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCrCb420PseudoPlanarToRGB8888u8( const uint8_t* __restrict src,
                                         unsigned int              srcWidth,
                                         unsigned int              srcHeight,
                                         unsigned int              srcYStride,
                                         unsigned int              srcCStride,
                                         uint32_t* __restrict      dst,
                                         unsigned int              dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from YUV (YCbCr) 4:2:0 PesudoPlanar (Interleaved CbCr) to RGB 565.
///  
///   \n\b ATTENTION: The name of this function will be changed when the 2.0.0 release 
///   of this library is made.  
///   Until 2.0.0, the developer should use this implementation with the expectation of
///   moving to a different name when transitioning to 2.0.0.
///   \n\n
///   
/// @param src
///   8-bit image of input YUV 4:2:0 values.
///   \n\b NOTE: must be 128-bit aligned.
///
///   The input are one Y plane followed by one interleaved and 2D (both
///   horizontally and vertically) sub-sampled CbCr plane:
///   Y plane                             : Y00  Y01  Y02  Y03 ...
///                                         Y10  Y11  Y12  Y13 ...
///   Interleaved and 2D sub-sampled plane: Cb0  Cr0  Cb1  Cr1 ...
///
/// @param dst
///   16-bit image of output RGB 565 values. R in LSBs.
///   2 pixels are packed into one 32-bit output
///   \n\b WARNING: size must match input yuv420.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: Must be multiple of 4 
///
/// @param srcHeight
///   Image height.
///
/// 
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYUV420toRGB565u8( const uint8_t* __restrict src,
                          unsigned int              srcWidth,
                          unsigned int              srcHeight,
                          uint32_t*  __restrict     dst );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from YCbCr H1V1 to RGB 888.
///
/// @details
///   Color conversion from YCbCr H1V1 (YCbCr 4:4:4 planar) to RGB 888.
///   \n R = Y                    + 1.40200*(Cr-128)
///   \n G = Y - 0.34414*(Cb-128) - 0.71414*(Cr-128)
///   \n B = Y + 1.77200*(CB-128)
///
/// @param src
///   8-bit image of input values. Stored as YCbCr H1V1 planar format in 8x8 blocks for Y,Cb,Cr.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: Must be multiple of 8
///
/// @param srcHeight
///   Image height.
///
/// @param dst
///   8-bit image of output RGB 888 values. R in LSB.
///   \n\b WARNING: size must match input crcb.
///   \n\b NOTE: must be 128-bit aligned.
///
/// 
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCrCbH1V1toRGB888u8( const uint8_t* __restrict src,
                             unsigned int              srcWidth,
                             unsigned int              srcHeight,
                             uint8_t* __restrict       dst );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from YCbCr H2V2 to RGB 888.
///
/// @details
///   Color conversion from YCbCr H2V2 (YCbCr 4:2:0 planar) to RGB 888.
///   \n R = Y                    + 1.40200*(Cr-128)
///   \n G = Y - 0.34414*(Cb-128) - 0.71414*(Cr-128)
///   \n B = Y + 1.77200*(CB-128)
///
/// @param ysrc
///   8-bit input values. Stored as YCbCr H2V2 planar format in 16x16 blocks for Y, 8x8 blocks for Cb, Cr.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: Must be multiple of 8 
///
/// @param srcHeight
///   Image height.
///
/// @param dst
///   8-bit image of output RGB 888 values. R in LSB.
///   \n\b WARNING: size must match input crcb.
///   \n\b NOTE: must be 128-bit aligned.
///
/// 
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCrCbH2V2toRGB888u8( const uint8_t* __restrict ysrc,
                             unsigned int              srcWidth,
                             unsigned int              srcHeight,
                             uint8_t* __restrict       dst );



//------------------------------------------------------------------------------
/// @brief
///   Color conversion from YCbCr H2V1 to RGB 888.
///
/// @details
///   Color conversion from YCbCr H2V1 (YCbCr 4:2:2) to RGB 888.
///   \n R = Y                    + 1.40200*(Cr-128)
///   \n G = Y - 0.34414*(Cb-128) - 0.71414*(Cr-128)
///   \n B = Y + 1.77200*(CB-128)
///
/// @param src
///   8-bit input values. Stored as YCbCr H2V1 planar format in 16x8 blocks for Y, 8x8 blocks for Cb, Cr.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: Must be multiple of 8 
///
/// @param srcHeight
///   Image height.
///
/// @param dst
///   8-bit image of output RGB 888 values. R in LSB.
///   \n\b WARNING: size must match input crcb.
///   \n\b NOTE: must be 128-bit aligned.
///
/// 
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCrCbH2V1toRGB888u8( const uint8_t* __restrict src,
                             unsigned int              srcWidth,
                             unsigned int              srcHeight,
                             uint8_t* __restrict       dst );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from YCbCr H1V2 to RGB 888.
///
/// @details
///   Color conversion from YCbCr H1V2 (YCbCr 4:2:2) to RGB 888.
///   \n R = Y                    + 1.40200*(Cr-128)
///   \n G = Y - 0.34414*(Cb-128) - 0.71414*(Cr-128)
///   \n B = Y + 1.77200*(CB-128)
///
/// @param ysrc
///   8-bit input values. Stored as YCbCr H1V2 planar format in 8x16 blocks for Y, 8x8 blocks for Cb, Cr.
///   \n\b NOTE: must be 128-bit aligned.
///
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: Must be multiple of 8 
///
/// @param srcHeight
///   Image height.
///
/// @param dst
///   8-bit image of output RGB 888 values. R in LSB.
///   \n\b WARNING: size must match input crcb.
///   \n\b NOTE: must be 128-bit aligned.
///
/// 
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCrCbH1V2toRGB888u8( const uint8_t* __restrict ysrc,

                             unsigned int              srcWidth,
                             unsigned int              srcHeight,
                             uint8_t* __restrict       dst );



//------------------------------------------------------------------------------
/// @brief 
///   Color conversion from RGB 888 to YCrCb.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvColorRGB888toYCrCbu8_v2(). In the 2.0.0 release, 
///   fcvColorRGB888toYCrCbu8_v2 will be renamed to fcvColorRGB888toYCrCbu8
///   and the signature of fcvColorRGB888toYCrCbu8 as it appears now, 
///   will be removed.
///   \n\n
/// 
/// @details
///   Color conversion from RGB 888 to YCrCb 4:4:4 interleaved.
///
/// @param src
///   8-bit input values.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: Must be multiple of 8 
///
/// @param srcHeight
///   Image height.
///
/// @param dst
///   8-bit output values. Stored as Y, Cr, Cb interleaved format.
///   \n\b WARNING: size must match input crcb.
///   \n\b NOTE: must be 128-bit aligned.
///
/// 
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB888toYCrCbu8( const uint8_t* __restrict src,
                         unsigned int              srcWidth,
                         unsigned int              srcHeight,
                         uint8_t* __restrict       dst );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB 888 to YCrCb 4:4:4 (Full interleaved, similar to
///   3-channel RGB).
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvColorRGB888toYCrCbu8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvColorRGB888toYCrCbu8,
///   \a fcvColorRGB888toYCrCbu8_v2 will be removed, and the current signature
///   for \a fcvColorRGB888toYCrCbu8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvColorRGB888toYCrCbu8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   Color conversion from RGB 888 to YCrCb 4:4:4 interleaved.
///   
/// @param src
///   8-bit input values.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: Must be multiple of 8.
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcStride
///   Image stride (in bytes).
///   \n\b WARNING: Must be at least 3*srcWidth.
///
/// @param dst
///   8-bit output values. Stored as Y, Cr, Cb interleaved format.
///   \n\b WARNING: size must match input crcb.
///   \n\b NOTE: must be 128-bit aligned.
/// 
/// @param dstStride
///   Output stride (in bytes).
///   \n\b WARNING: Must be at least 3*srcWidth.
///
/// 
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB888toYCrCbu8_v2( const uint8_t* __restrict src,
                            unsigned int              srcWidth,
                            unsigned int              srcHeight,
                            unsigned int              srcStride,
                            uint8_t* __restrict       dst,
                            unsigned int              dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Create a 36-dimension gradient based descriptor on 17x17 patch.
///
/// @details
///
/// @param patch
///   Input luminance data for 17x17 patch to describe.
///
/// @param descriptorChar
///   Output descriptor vector. 36 x 8-bit vector. Normalized and quantized to range [-127, 127]
///
/// @param descriptorNormSq
///   Output squared norm (L2 norm) of the descriptor vector.
///
/// 
///
/// @ingroup object_detection
//------------------------------------------------------------------------------

FASTCV_API int
fcvDescriptor17x17u8To36s8( const uint8_t* __restrict patch,
                            int8_t* __restrict        descriptorChar,
                            int32_t* __restrict       descriptorNormSq );


//---------------------------------------------------------------------------
/// @brief
///   Dot product of two 8-bit vectors.
///
/// @param a
///   Vector.
///
/// @param b
///   Vector.
///
/// @param abSize
///   Number of elements in A and B.
///
/// @return
///   Dot product <A|B>.
///
/// @ingroup math_vector
//---------------------------------------------------------------------------

FASTCV_API int32_t
fcvDotProducts8( const int8_t* __restrict a,
                 const int8_t* __restrict b,
                 unsigned int             abSize );


//------------------------------------------------------------------------------
/// @brief
///   Dot product of two 8-bit vectors.
///
/// @param a
///   Vector A.
///
/// @param b
///   Vector B.
///
/// @param abSize
///   Number of elements in A and B.
///
/// @return
///   Dot product <A|B>.
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API uint32_t
fcvDotProductu8( const uint8_t* __restrict  a,
                 const uint8_t* __restrict  b,
                 unsigned int               abSize );


//---------------------------------------------------------------------------
/// @brief
///   Dot product of two 36-byte vectors.
///
/// @param a
///   Vector.
///
/// @param b
///   Vector.
///
/// @return
///   Dot product <a|b>.
///
/// @ingroup math_vector
//---------------------------------------------------------------------------

FASTCV_API int32_t
fcvDotProduct36x1s8( const int8_t* __restrict a,
                     const int8_t* __restrict b );


//---------------------------------------------------------------------------
/// @brief
///   Dot product of one 36-byte vector against 4 others.
///
/// @details
///   Dot product of 36-byte vector (a) against 4 others (b,c,d,e):\n
///   <a|b>, <a|c>, <a|d>, <a|e>
///
/// @param a
///   Vector.
///
/// @param b
///   Vector.
///
/// @param c
///   Vector.
///
/// @param d
///   Vector.
///
/// @param e
///   Vector.
///
/// @param dotProducts
///   Output of the 4 results { <a|b>, <a|c>, <a|d>, <a|e> }.
///   \n\b WARNING: array must be 128-bit aligned
///
/// @ingroup math_vector
//---------------------------------------------------------------------------

FASTCV_API void
fcvDotProduct36x4s8( const int8_t* __restrict a,
                     const int8_t* __restrict b,
                     const int8_t* __restrict c,
                     const int8_t* __restrict d,
                     const int8_t* __restrict e,
                     int32_t* __restrict      dotProducts );


//------------------------------------------------------------------------------
/// @brief
///   Normalized dot product of one 36-byte vector against 4 others.
///
/// @details
///   Dot product of 36-byte vector (a) against 4 others (b0,b1,b2,b3):\n
///   <a|b0>, <a|b1>, <a|b2>, <a|b3>
///   using their given inverse lengths for normalization.
///
/// @param a
///   Vector.
///
/// @param invLengthA
///   Inverse of vector A.
///
/// @param b0
///   Vector.
///
/// @param b1
///   Vector.
///
/// @param b2
///   Vector.
///
/// @param b3
///   Vector.
///
/// @param invLengthsB
///   Pointer to an array of the inverse values of each B vector.
///   The pointer must point to 4 floating point values.
///   \n\b WARNING: array must be 128-bit aligned
///
/// @param dotProducts
///   Output of the 4 results { <a|b0>, <a|b1>, <a|b2>, <a|b3> }.
///   \n\b WARNING: array must be 128-bit aligned
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API void
fcvDotProductNorm36x4s8( const int8_t* __restrict a,
                         float                    invLengthA,
                         const int8_t* __restrict b0,
                         const int8_t* __restrict b1,
                         const int8_t* __restrict b2,
                         const int8_t* __restrict b3,
                         float* __restrict        invLengthsB,
                         float* __restrict        dotProducts  );


//------------------------------------------------------------------------------
/// @brief
///   Dot product of two 36-byte vectors.
///
/// @param a
///   Vector.
///
/// @param b
///   Vector.
///
/// @return
///   Dot product <a|b>.
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API uint32_t
fcvDotProduct36x1u8( const uint8_t* __restrict a,
                     const uint8_t* __restrict b );


//------------------------------------------------------------------------------
/// @brief
///   Dot product of one 36-byte vector against 4 others.
///
/// @details
///   Dot product of 36-byte vector (a) against 4 others (b,c,d,e):\n
///   <a|b>, <a|c>, <a|d>, <a|e>
///
/// @param a
///   Vector.
///
/// @param b
///   Vector.
///
/// @param c
///   Vector.
///
/// @param d
///   Vector.
///
/// @param e
///   Vector.
///
/// @param dotProducts
///   Output of the 4 results { <a|b>, <a|c>, <a|d>, <a|e> }.
///   \n\b WARNING: array must be 128-bit aligned
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API void
fcvDotProduct36x4u8( const uint8_t* __restrict a,
                     const uint8_t* __restrict b,
                     const uint8_t* __restrict c,
                     const uint8_t* __restrict d,
                     const uint8_t* __restrict e,
                     uint32_t* __restrict      dotProducts );


//------------------------------------------------------------------------------
/// @brief
///   Normalized dot product of one 36-byte vector against 4 others.
///
/// @details
///   Dot product of 36-byte vector (a) against 4 others (b0,b1,b2,b3):\n
///   <a|b0>, <a|b1>, <a|b2>, <a|b3>
///   using their given inverse lengths for normalization.
///
/// @param a
///   Vector.
///
/// @param invLengthA
///   Inverse of vector A.
///
/// @param b0
///   Vector.
///
/// @param b1
///   Vector.
///
/// @param b2
///   Vector.
///
/// @param b3
///   Vector.
///
/// @param invLengthsB
///   Pointer to an array of the inverse values of each B vector.
///   The pointer must point to 4 floating point values.
///   \n\b WARNING: array must be 128-bit aligned
///
/// @param dotProducts
///   Output of the 4 results { <a|b0>, <a|b1>, <a|b2>, <a|b3> }.
///   \n\b WARNING: array must be 128-bit aligned
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API void
fcvDotProductNorm36x4u8( const uint8_t* __restrict  a,
                         float                      invLengthA,
                         const uint8_t* __restrict  b0,
                         const uint8_t* __restrict  b1,
                         const uint8_t* __restrict  b2,
                         const uint8_t* __restrict  b3,
                         float* __restrict          invLengthsB,
                         float* __restrict          dotProducts );


//---------------------------------------------------------------------------
/// @brief
///   Dot product of two 64-byte vectors.
///
/// @param a
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param b
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @return
///   Dot product <a|b>.
///
/// @ingroup math_vector
//---------------------------------------------------------------------------

FASTCV_API int32_t
fcvDotProduct64x1s8( const int8_t* __restrict a,
                     const int8_t* __restrict b );


//---------------------------------------------------------------------------
/// @brief
///   Dot product of one 64-byte vector against 4 others.
///
/// @details
///   Dot product of vector (a) against 4 others (b,c,d,e):\n
///   <a|b>, <a|c>, <a|d>, <a|e>
///
/// @param a
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param b
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param c
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param d
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param e
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param dotProducts
///   Output of the 4 results { <a|b>, <a|c>, <a|d>, <a|e> }.
///   \n\b WARNING: array must be 128-bit aligned
///
/// @ingroup math_vector
//---------------------------------------------------------------------------

FASTCV_API void
fcvDotProduct64x4s8( const int8_t* __restrict a,
                     const int8_t* __restrict b,
                     const int8_t* __restrict c,
                     const int8_t* __restrict d,
                     const int8_t* __restrict e,
                     int32_t* __restrict      dotProducts );


//------------------------------------------------------------------------------
/// @brief
///   Normalized dot product of one 64-byte vector against 4 others.
///
/// @details
///   Dot product of 36-byte vector (a) against 4 others (b0,b1,b2,b3):\n
///   <a|b0>, <a|b1>, <a|b2>, <a|b3>
///   using their given inverse lengths for normalization.
///
/// @param a
///   Vector.
///
/// @param invLengthA
///   Inverse of vector A.
///
/// @param b0
///   Vector.
///
/// @param b1
///   Vector.
///
/// @param b2
///   Vector.
///
/// @param b3
///   Vector.
///
/// @param invLengthsB
///   Pointer to an array of the inverse values of each B vector.
///   The pointer must point to 4 floating point values.
///   \n\b WARNING: array must be 128-bit aligned
///
/// @param dotProducts
///   Output of the 4 results { <a|b0>, <a|b1>, <a|b2>, <a|b3> }.
///   \n\b WARNING: array must be 128-bit aligned
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API void
fcvDotProductNorm64x4s8( const int8_t* __restrict a,
                         float                    invLengthA,
                         const int8_t* __restrict b0,
                         const int8_t* __restrict b1,
                         const int8_t* __restrict b2,
                         const int8_t* __restrict b3,
                         float* __restrict        invLengthsB,
                         float* __restrict        dotProducts  );


//------------------------------------------------------------------------------
/// @brief
///   Dot product of two 64-byte vectors.
///
/// @param a
///   Vector.
///
/// @param b
///   Vector.
///
/// @return
///   Dot product <a|b>.
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API uint32_t
fcvDotProduct64x1u8( const uint8_t* __restrict a,
                     const uint8_t* __restrict b );


//------------------------------------------------------------------------------
/// @brief
///   Dot product of one 64-byte vector against 4 others.
///
/// @details
///   Dot product of 36-byte vector (a) against 4 others (b,c,d,e):\n
///   <a|b>, <a|c>, <a|d>, <a|e>
///
/// @param a
///   Vector.
///
/// @param b
///   Vector.
///
/// @param c
///   Vector.
///
/// @param d
///   Vector.
///
/// @param e
///   Vector.
///
/// @param dotProducts
///   Output of the 4 results { <a|b>, <a|c>, <a|d>, <a|e> }.
///   \n\b WARNING: array must be 128-bit aligned
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API void
fcvDotProduct64x4u8( const uint8_t* __restrict a,
                     const uint8_t* __restrict b,
                     const uint8_t* __restrict c,
                     const uint8_t* __restrict d,
                     const uint8_t* __restrict e,
                     uint32_t* __restrict      dotProducts );


//------------------------------------------------------------------------------
/// @brief
///   Normalized dot product of one 64-byte vector against 4 others.
///
/// @details
///   Dot product of 36-byte vector (a) against 4 others (b0,b1,b2,b3):\n
///   <a|b0>, <a|b1>, <a|b2>, <a|b3>
///   using their given inverse lengths for normalization.
///
/// @param a
///   Vector.
///
/// @param invLengthA
///   Inverse of vector A.
///
/// @param b0
///   Vector.
///
/// @param b1
///   Vector.
///
/// @param b2
///   Vector.
///
/// @param b3
///   Vector.
///
/// @param invLengthsB
///   Pointer to an array of the inverse values of each B vector.
///   The pointer must point to 4 floating point values.
///   \n\b WARNING: array must be 128-bit aligned
///
/// @param dotProducts
///   Output of the 4 results { <a|b0>, <a|b1>, <a|b2>, <a|b3> }.
///   \n\b WARNING: array must be 128-bit aligned
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API void
fcvDotProductNorm64x4u8( const uint8_t* __restrict  a,
                         float                      invLengthA,
                         const uint8_t* __restrict  b0,
                         const uint8_t* __restrict  b1,
                         const uint8_t* __restrict  b2,
                         const uint8_t* __restrict  b3,
                         float* __restrict          invLengthsB,
                         float* __restrict          dotProducts );


//---------------------------------------------------------------------------
/// @brief
///   Dot product of two 128-byte vectors.
///
/// @param a
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param b
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @return
///   Dot product <a|b>.
///
/// @ingroup math_vector
//---------------------------------------------------------------------------

FASTCV_API int32_t
fcvDotProduct128x1s8( const int8_t* __restrict a,
                      const int8_t* __restrict b );


//---------------------------------------------------------------------------
/// @brief
///   Dot product of one 128-byte vector against 4 others.
///
/// @details
///   Dot product of vector (a) against 4 others (b,c,d,e):\n
///   <a|b>, <a|c>, <a|d>, <a|e>
///
/// @param a
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param b
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param c
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param d
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param e
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param dotProducts
///   Output of the 4 results { <a|b>, <a|c>, <a|d>, <a|e> }.
///   \n\b WARNING: array must be 128-bit aligned
///
/// @ingroup math_vector
//---------------------------------------------------------------------------

FASTCV_API void
fcvDotProduct128x4s8( const int8_t* __restrict a,
                      const int8_t* __restrict b,
                      const int8_t* __restrict c,
                      const int8_t* __restrict d,
                      const int8_t* __restrict e,
                      int32_t* __restrict      dotProducts );


//------------------------------------------------------------------------------
/// @brief
///   Normalized dot product of one 128-byte vector against 4 others.
///
/// @details
///   Dot product of vector (a) against 4 others (b0,b1,b2,b3):\n
///   <a|b0>, <a|b1>, <a|b2>, <a|b3>
///   using their given inverse lengths for normalization.
///
/// @param a
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param invLengthA
///   Inverse of vector A.
///
/// @param b0
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param b1
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param b2
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param b3
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param invLengthsB
///   Pointer to an array of the inverse values of each B vector.
///   The pointer must point to 4 floating point values.
///   \n\b WARNING: array must be 128-bit aligned
///
/// @param dotProducts
///   Output of the 4 results { <a|b0>, <a|b1>, <a|b2>, <a|b3> }.
///   \n\b WARNING: array must be 128-bit aligned
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API void
fcvDotProductNorm128x4s8( const int8_t* __restrict a,
                          float                    invLengthA,
                          const int8_t* __restrict b0,
                          const int8_t* __restrict b1,
                          const int8_t* __restrict b2,
                          const int8_t* __restrict b3,
                          float* __restrict        invLengthsB,
                          float* __restrict        dotProducts  );


//------------------------------------------------------------------------------
/// @brief
///   Dot product of two 128-byte vectors.
///
/// @param a
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param b
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @return
///   Dot product <a|b>.
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API uint32_t
fcvDotProduct128x1u8( const uint8_t* __restrict a,
                      const uint8_t* __restrict b );


//------------------------------------------------------------------------------
/// @brief
///   Dot product of one 128-byte vector against 4 others.
///
/// @details
///   Dot product of vector (a) against 4 others (b,c,d,e):\n
///   <a|b>, <a|c>, <a|d>, <a|e>
///
/// @param a
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param b
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param c
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param d
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param e
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param dotProducts
///   Output of the 4 results { <a|b>, <a|c>, <a|d>, <a|e> }.
///   \n\b WARNING: array must be 128-bit aligned
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API void
fcvDotProduct128x4u8( const uint8_t* __restrict a,
                      const uint8_t* __restrict b,
                      const uint8_t* __restrict c,
                      const uint8_t* __restrict d,
                      const uint8_t* __restrict e,
                      uint32_t* __restrict      dotProducts );


//------------------------------------------------------------------------------
/// @brief
///   Normalized dot product of one 128-byte vector against 4 others.
///
/// @details
///   Dot product of vector (a) against 4 others (b0,b1,b2,b3):\n
///   <a|b0>, <a|b1>, <a|b2>, <a|b3>
///   using their given inverse lengths for normalization.
///
/// @param a
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param invLengthA
///   Inverse of vector A.
///
/// @param b0
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param b1
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param b2
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param b3
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param invLengthsB
///   Pointer to an array of the inverse values of each B vector.
///   The pointer must point to 4 floating point values.
///
/// @param dotProducts
///   Output of the 4 results { <a|b0>, <a|b1>, <a|b2>, <a|b3> }.
///   \n\b WARNING: array must be 128-bit aligned
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API void
fcvDotProductNorm128x4u8( const uint8_t* __restrict  a,
                          float                      invLengthA,
                          const uint8_t* __restrict  b0,
                          const uint8_t* __restrict  b1,
                          const uint8_t* __restrict  b2,
                          const uint8_t* __restrict  b3,
                          float* __restrict          invLengthsB,
                          float* __restrict          dotProducts );


//------------------------------------------------------------------------------
/// @brief
///   Dot product of 1 patch (8x8 byte square) with several (n) 8x8 squares
///   along a line of pixels in an image.
///
/// @param patchPixels
///   Pointer to 8-bit patch pixel values linearly laid out in memory.
///
/// @param imagePixels
///   Pointer to 8-bit image pixel values linearly laid out in memory.
///
/// @param imgW
///   Width in pixels of the source image.
///
/// @param imgH
///   Height in pixels of the source image.
///
/// @param nX
///   X location on image of starting search pixel.
///
/// @param nY
///   Y location on image of starting search pixel.
///
/// @param nNum
///   Number of pixels (in X direction) on image to sweep.
///
/// @param dotProducts
///   Output dot product values of nNum pixels.
///   \n\b WARNING: array size must be a multiple of 4 (e.g., 4, 8, 12, ...)
///   \n\b NOTE: array should be 128-bit aligned
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API void
fcvDotProduct8x8u8( const uint8_t* __restrict patchPixels,
                    const uint8_t* __restrict imagePixels,
                    unsigned short            imgW,
                    unsigned short            imgH,
                    int                       nX,
                    int                       nY,
                    unsigned int              nNum,
                    int32_t* __restrict       dotProducts );


//------------------------------------------------------------------------------
/// @brief
///   Dot product of 1 patch (8x8 byte square) with 8x8 squares in 11x12
///   rectangle around the center search pixel (iX,iY).
///
/// @param patchPixels
///   Pointer to 8-bit patch pixel values linearly laid out in memory.
///
/// @param imagePixels
///   Pointer to 8-bit image pixel values linearly laid out in memory.
///
/// @param imgW
///   Width in pixels of the image.
///
/// @param imgH
///   Height in pixels of the image.
///
/// @param iX
///   X location on image of the center of the search window.
///
/// @param iY
///   Y location on image of the center of the search window.
///
/// @param dotProducts
///   Output 11x12 dot product values.
///   \n\b WARNING: array must be 128-bit aligned
///
/// @ingroup math_vector
//---------------------------------------------------------------------------

FASTCV_API void
fcvDotProduct11x12u8( const uint8_t* __restrict patchPixels,
                      const uint8_t* __restrict imagePixels,
                      unsigned short            imgW,
                      unsigned short            imgH,
                      int                       iX,
                      int                       iY,
                      int32_t* __restrict       dotProducts );


//------------------------------------------------------------------------------
/// @brief
///   3x3 Sobel edge filter
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvFilterSobel3x3u8_v2(). In the 2.0.0 release, 
///   fcvFilterSobel3x3u8_v2 will be renamed to fcvFilterSobel3x3u8
///   and the signature of fcvFilterSobel3x3u8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///   This function calculates an image derivative by convolving the image with an appropriate 3x3 kernel.
///   Border values are ignored. The 3x3 mask convolves with the image area
///   | a(1,1)          ,    a12,    ...,   a(1,srcWidth-2)            |\n
///   |       ...       ,    ...,    ...,              ...             |\n
///   | a(srcHeight-2,1),    ...,    ...,   a1(srcHeight-2,srcWidth-2) |\n
///
/// @param src
///   Input 8-bit image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: data must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b NOTE: must be multiple of 8
///
/// @param srcHeight
///   Image height.
///
/// @param dst
///   Output 8-bit image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b NOTE: dst is saturated to 255
///   \n\b WARNING: data must be 128-bit aligned.
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterSobel3x3u8( const uint8_t* __restrict src,
                     unsigned int              srcWidth,
                     unsigned int              srcHeight,
                     uint8_t* __restrict       dst );


//------------------------------------------------------------------------------
/// @brief
///   3x3 Sobel edge filter
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvFilterSobel3x3u8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvFilterSobel3x3u8,
///   \a fcvFilterSobel3x3u8_v2 will be removed, and the current signature
///   for \a fcvFilterSobel3x3u8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvFilterSobel3x3u8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   This function calculates an image derivative by convolving the image with an appropriate 3x3 kernel.
///   Border values are ignored. The 3x3 mask convolves with the image area
///   | a(1,1)          ,    a12,    ...,   a(1,srcWidth-2)            |\n
///   |       ...       ,    ...,    ...,              ...             |\n
///   | a(srcHeight-2,1),    ...,    ...,   a1(srcHeight-2,srcWidth-2) |\n
///
/// @param src
///   Input 8-bit image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b WARNING: data must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b NOTE: must be multiple of 8
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcStride
///   Image stride.
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param dst
///   Output 8-bit image. Size of buffer is dstStride*srcHeight bytes.
///   \n\b NOTE: dst is saturated to 255
///   \n\b WARNING: data must be 128-bit aligned.
/// 
/// @param dstStride
///   Output stride.
///   \n\b NOTE: if 0, dstStride is set as dstWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as dstWidth if not 0.
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterSobel3x3u8_v2( const uint8_t* __restrict src,
                        unsigned int              srcWidth,
                        unsigned int              srcHeight,
                        unsigned int              srcStride,
                        uint8_t* __restrict       dst,
                        unsigned int              dstStride );

//------------------------------------------------------------------------------
/// @brief
///   Canny edge filter
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvFilterCanny3x3u8_v2(). In the 2.0.0 release, 
///   fcvFilterCanny3x3u8_v2 will be renamed to fcvFilterCanny3x3u8
///   and the signature of fcvFilterCanny3x3u8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///   Canny edge detector applied to a 8 bit grayscale image. The min threshold
///   is set to 0 and the max threshold is set to 15. The aperture size used
///   is set to 3. This function will output the edge, since its working with a 
///   3x3 window, it leaves  one row/col of pixels at the corners
///   map stored as a binarized image (0x0 - not an edge, 0xFF - edge). 
///   | a(1,1)          ,    a12,    ...,   a(1,srcWidth-2)            |\n
///   |       ...       ,    ...,    ...,              ...             |\n
///   | a(srcHeight-2,1),    ...,    ...,   a1(srcHeight-2,srcWidth-2) |\n
///
/// @param src
///   Input 8-bit image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: data must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b NOTE: must be multiple of 8
///
/// @param srcHeight
///   Image height.
///
/// @param dst
///   Output 8-bit image containing the edge detection results.
///   Size of buffer is srcWidth*srcHeight bytes.
///
/// @param lowThresh
///   For all the intermediate pixels along the edge, the magnitude of the
///   gradient at the pixel locations should be greater than 'low'
///   (sqrt(gx^2 + gy^2) > low, where gx and gy are X and Y gradient)
///
/// @param highThresh
///   For an edge starting point, i.e. either the first or last
///   pixel of the edge, the magnitude of the gradient at the pixel should be
///   greater than 'high' (sqrt(gx^2 + gy^2) > high, where gx and gy are X and
///   Y gradient).
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterCanny3x3u8( const uint8_t* __restrict src,
                     unsigned int              srcWidth,
                     unsigned int              srcHeight,
                     uint8_t* __restrict       dst,
                     int                       lowThresh,
                     int                       highThresh );

//------------------------------------------------------------------------------
/// @brief
///   Canny edge filter
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvFilterCanny3x3u8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvFilterCanny3x3u8,
///   \a fcvFilterCanny3x3u8_v2 will be removed, and the current signature
///   for \a fcvFilterCanny3x3u8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvFilterCanny3x3u8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   Canny edge detector applied to a 8 bit grayscale image. The Canny edge
///   detector uses min/max threshold to classify an edge. The min threshold
///   is set to 0 and the max threshold is set to 15. The aperture size used
///   in the Canny edge detector will be same as the filter footprint in the
///   Sobel edge detector and is set to 3. This function will output the edge
///   map stored as a binarized image (0x0 - not an edge, 0xFF - edge), since 
///   it works with 3x3 windows, it leaves 1 row/col of pixels at the corners.
///   | a(1,1)          ,    a12,    ...,   a(1,srcWidth-2)            |\n
///   |       ...       ,    ...,    ...,              ...             |\n
///   | a(srcHeight-2,1),    ...,    ...,   a1(srcHeight-2,srcWidth-2) |\n
///
/// @param src
///   Input 8-bit image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b WARNING: data must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b NOTE: must be multiple of 8
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcStride
///   Image stride.
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param dst
///   Output 8-bit image containing the edge detection results.
///   Size of buffer is dstStride*srcHeight bytes.
/// 
/// @param dstStride
///   Output stride.
///   \n\b NOTE: if 0, dstStride is set as dstWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as dstWidth if not 0.
///
/// @param lowThresh
///   For all the intermediate pixels along the edge, the magnitude of the
///   gradient at the pixel locations should be greater than 'low'
///   (sqrt(gx^2 + gy^2) > low, where gx and gy are X and Y gradient)
///
/// @param highThresh
///   For an edge starting point, i.e. either the first or last
///   pixel of the edge, the magnitude of the gradient at the pixel should be
///   greater than 'high' (sqrt(gx^2 + gy^2) > high, where gx and gy are X and
///   Y gradient).
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterCanny3x3u8_v2( const uint8_t* __restrict src,
                        unsigned int              srcWidth,
                        unsigned int              srcHeight,
                        unsigned int              srcStride,
                        uint8_t* __restrict       dst,
                        unsigned int              dstStride,
                        int                       lowThresh,
                        int                       highThresh );

//------------------------------------------------------------------------------
/// @brief
///   Performs image difference by subracting src2 from src1. dst=src1-src2.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvImageDiffu8_v2(). In the 2.0.0 release, 
///   fcvImageDiffu8_v2 will be renamed to fcvImageDiffu8
///   and the signature of fcvImageDiffu8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///   dst[i,j] = (uint8_t) max( min((short)(src1[i,j]-src2[i,j]), 255), 0 );
///   
/// @param src1
///   First source image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param src2
///   Second source image, must be same size as src1.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height.
///
/// @param dst
///   Destination. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b NOTE: Must be same size as src1 and src2.
///   \n\b WARNING: must be 128-bit aligned.
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageDiffu8(   const uint8_t* __restrict src1,
                  const uint8_t* __restrict src2,
                   unsigned int             srcWidth,
                   unsigned int             srcHeight,
                        uint8_t* __restrict dst );

//------------------------------------------------------------------------------
/// @brief
///   Performs image difference by subracting src2 from src1. dst=src1-src2.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvImageDiffu8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvImageDiffu8,
///   \a fcvImageDiffu8_v2 will be removed, and the current signature
///   for \a fcvImageDiffu8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvImageDiffu8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   dst[i,j] = (uint8_t) max( min((short)(src1[i,j]-src2[i,j]), 255), 0 );
///   
/// @param src1
///   First source image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param src2
///   Second source image, must be same size as src1.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcStride
///   Stride of input image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param dst
///   Destination. Size of buffer is dstStride*srcHeight bytes.
///   \n\b NOTE: Must be same size as src1 and src2.
///   \n\b WARNING: must be 128-bit aligned.
/// 
/// @param dstStride
///   Stride of output image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as dstWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as dstWidth if not 0.
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageDiffu8_v2( const uint8_t* __restrict src1,
                   const uint8_t* __restrict src2,
                   unsigned int              srcWidth,
                   unsigned int              srcHeight,
                   unsigned int              srcStride,
                   uint8_t* __restrict       dst,
                   unsigned int              dstStride );


//--------------------------------------------------------------------------
/// @brief
///   Compute image difference src1-src2
///
/// @param src1
///   Input image1 of int16 type. Size of buffer is srcStride*srcHeight*2 bytes.
///   \n\b WARNING: must be 128-bit aligned.
/// 
/// @param src2
///   Input image2, must have same size as src1
///   \n\b WARNING: must be 128-bit aligned.
/// 
/// @param srcWidth
///   Input image width
///   \n\b WARNING: must be multiple of 8
/// 
/// @param srcHeight
///   Input image height
/// 
/// @param srcStride
///   Stride of input image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
/// 
/// @param dst
///   Output image, saturated for int16 type. Size of buffer is dstStride*srcHeight*2 bytes.
///   \n\b WARNING: must be 128-bit aligned.
/// 
/// @param dstStride
///   Stride of output image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, dstStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
/// 
/// @ingroup image_processing
////------------------------------------------------------------------------
FASTCV_API void 
fcvImageDiffs16( const int16_t* __restrict src1,
                 const int16_t* __restrict src2,
                       unsigned int             srcWidth, 
                       unsigned int             srcHeight, 
                       unsigned int             srcStride, 
                            int16_t* __restrict dst,
                       unsigned int             dstStride );

//------------------------------------------------------------------------------
/// @brief
///   Performs image difference by subracting src2 from src1. dst=src1-src2.
///
/// @details
///   
/// @param src1
///   First source image. Size of buffer is srcStride*srcHeight*4 bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param src2
///   Second source image, must be same size as src1.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8
///
/// @param srcHeight
///   Image height.
///
/// @param srcStride
///   Stride of input image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param dst
///   Destination.  Size of buffer is dstStride*srcHeight*4 bytes.
///   \n\b NOTE: Must be same size as src1 and src2.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, dstStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------
FASTCV_API void 
fcvImageDifff32( const float* __restrict src1,
                 const float* __restrict src2,
                unsigned int             srcWidth, 
                unsigned int             srcHeight, 
                unsigned int             srcStride, 
                       float* __restrict dst,
                unsigned int             dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Performs image difference by promoting both src1 and src 2 to 
///   floating point values and then subracting src2 from src1. dst=src1-src2.
///
/// @details
///   
/// @param src1
///   First source image
///
/// @param src2
///   Second source image, must be same size as src1.
///
/// @param srcWidth
///   Image width.
///
/// @param srcHeight
///   Image height.
///
/// @param srcStride
///   Stride of input image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b WARNING: must be multiple of 8.
///
/// @param dst
///   Destination image in float type
///   \n\b NOTE: Must be same size as src1 and src2.
///   \n\b WARNING: must be 128-bit aligned.
/// 
/// @param dstStride
///   Stride of output image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b WARNING: must be multiple of 8.
/// 
/// @ingroup image_processing
//------------------------------------------------------------------------------
FASTCV_API void 
fcvImageDiffu8f32( const uint8_t* __restrict src1,
                   const uint8_t* __restrict src2,
                    unsigned int             srcWidth, 
                    unsigned int             srcHeight, 
                    unsigned int             srcStride, 
                           float* __restrict dst,
                    unsigned int             dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Performs image difference by subracting src2 from src1. 
///   dst = ( src1 >> 1) - ( src2 >> 1).
///
/// @details
///   
/// @param src1
///   First source image
///
/// @param src2
///   Second source image, must be same size as src1.
///
/// @param srcWidth
///   Image width.
///
/// @param srcHeight
///   Image height.
///
/// @param srcStride
///   Stride of input image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b WARNING: must be multiple of 8.
///
/// @param dst
///   Destination image in int8 type
///   \n\b NOTE: Must be same size as src1 and src2.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b WARNING: must be multiple of 8.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------
FASTCV_API void 
fcvImageDiffu8s8( const uint8_t* __restrict src1,
                  const uint8_t* __restrict src2,
                   unsigned int             srcWidth, 
                   unsigned int             srcHeight, 
                   unsigned int             srcStride, 
                         int8_t* __restrict dst,
                    unsigned int             dstStride );

//---------------------------------------------------------------------------
/// @brief
///   Creates 2D gradient from source illuminance data.
///   This function considers only the left/right neighbors
///   for x-gradients and top/bottom neighbors for y-gradients.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvImageGradientInterleaveds16_v2(). In the 2.0.0 release, 
///   fcvImageGradientInterleaveds16_v2 will be renamed to fcvImageGradientInterleaveds16
///   and the signature of fcvImageGradientInterleaveds16 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param  src
///   Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///    Height of src data to create gradient.
///
/// @param  srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param  gradients
///    Buffer to store gradient. Must be 2*(width-1)*(height-1) in size.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageGradientInterleaveds16( const uint8_t* __restrict src,
                                unsigned int              srcWidth,
                                unsigned int              srcHeight,
                                unsigned int              srcStride,
                                int16_t* __restrict       gradients
                              );

//---------------------------------------------------------------------------
/// @brief
///   Creates 2D gradient from source illuminance data.
///   This function considers only the left/right neighbors
///   for x-gradients and top/bottom neighbors for y-gradients.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvImageGradientInterleaveds16() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvImageGradientInterleaveds16,
///   \a fcvImageGradientInterleaveds16_v2 will be removed, and the current signature
///   for \a fcvImageGradientInterleaveds16 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvImageGradientInterleaveds16 when transitioning to 2.0.0.
///   \n\n
///
/// @param  src
///   Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Height of src data to create gradient.
///
/// @param  srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param  gradients
///   Buffer to store gradient. Must be 2*(width-1)*(height-1) in size.
/// 
/// @param gradStride
///   Stride in bytes of the interleaved gradients array.
///   \n\b NOTE: if 0, srcStride is set as 4*(srcWidth-2).
///   \n\b WARNING: must be multiple of 16 ( 8 * 2-byte values ), and at least as much as 4*(srcWidth-2) if not 0.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageGradientInterleaveds16_v2( const uint8_t* __restrict src,
                                   unsigned int              srcWidth,
                                   unsigned int              srcHeight,
                                   unsigned int              srcStride,
                                   int16_t* __restrict       gradients,
                                   unsigned int              gradStride );

//---------------------------------------------------------------------------
///  @brief
///  Function to initialize MSER. To invoke MSER functionality, 3 functions have to be called:
///    fcvMserInit, fcvMseru8, fcvMserRelease.
///  
///   Heris the typical usage:
///  
///    void *mserHandle;
///     if (fcvMserInit (width,........,&mserHandle))
///      {
///          fcvmseru8 (mserHandle,...);
///          fcvRelease(mserHandle);
///      }
/// 
///   
///   @param width          Width of the image  for which MSER has to be done.
///   @param height         Height of the image for which MSER has to be done.
///   @param delta          Delta to be used in MSER algorithm (the difference in grayscale
///                         values within which the region is stable ). 
///                         Typical value range [0.8 8], typical value 2
///   @param minArea        Minimum area (number of pixels) of a mser contour.
///                         Typical value range [10 50], typical value 30
///   @param maxArea        Maximum area (number of pixels) of a  mser contour.
///                         Typical value 14400 or 0.25*width*height
///   @param maxVariation   Maximum variation in grayscale between 2 levels allowed.
///                         Typical value range [0.1 1.0], typical value 0.15
///   @param minDiversity   Minimum diversity in grayscale between 2 levels allowed.
///                         Typical value range [0.1 1.0], typical value 0.2
///   @param mserHandle     Return value: the mserHandle to be used in subsequent calls.
///  
///   @return  int  1 if mserInit is successful, if 0, mserHandle is invalid.
///
///   @ingroup object_detection
//------------------------------------------------------------------------------
FASTCV_API int 
fcvMserInit(const unsigned int width,
                 const unsigned int height, 
                 unsigned int delta, 
                 unsigned int minArea ,
                 unsigned int maxArea , 
                 float maxVariation ,
                 float minDiversity , void ** mserHandle );

//---------------------------------------------------------------------------
/// @brief
///  Function to release  MSER resources.
///  
/// 
///   
///   @param mserHandle   Handle to be used to free up MSER resources.
///  
///   @ingroup object_detection
//------------------------------------------------------------------------------
FASTCV_API void
fcvMserRelease(void *mserHandle);

///---------------------------------------------------------------------------
/// @brief
///  Function to invoke  MSER. 
/// 
///   \n\b ATTENTION: The signature of this function will be changed to reduce complexity
///   and memory usage when the 2.0.0 release of this library is made.  
///   Until 2.0.0, the developer should use this implementation with the expectation of
///   moving to a different signature when transitioning to 2.0.0.
///   \n\n
///    
///   
///   @param mserHandle     The MSER Handle returned by init.
///   @param srcPtr         Pointer to an image array (unsigned char ) for which MSER has to be done.
///   @param srcWidth       Width of the source image.
///   @param srcHeight      Height of the source image.
///   @param srcStride      Stride of the source image.
///   @param maxContours    Maximum contours that will be returned. Must be set to 2x the maximum contours.
///   @param numContours    Output, Number of MSER contours in the region.
///   @param numPointsInContour    Output, Number of points in each contour. This will have values filled up
///                                for the first (*numContours) values. This memory has to be allocated by 
///                                the caller. 
///   @param pointsArraySize Size of the output points Array.
///                          Typical size: (# of pixels in source image) * 30
///   @param pointsArray     Output. This is the points in all the contours. This is a linear array, whose memory
///                          has to be allocated by the caller.
///                          Typical allocation size:  pointArraySize
///                          pointsArray[0...numPointsInContour[0]-1] defines the first MSER region, 
///                          pointsArray[numPointsInContour[0] .. numPointsInContour[1]-1] defines 2nd MSER region
///                          and so on.
/// 
///   @ingroup object_detection
//------------------------------------------------------------------------------  
FASTCV_API void
fcvMseru8( void *mserHandle,
                const uint8_t* __restrict srcPtr,unsigned int srcWidth, 
                unsigned int srcHeight, unsigned int srcStride, 
                unsigned int maxContours,
                unsigned int * __restrict numContours, unsigned int * __restrict numPointsInContour   ,
                unsigned int pointsArraySize,
                unsigned int* __restrict pointsArray
              );

///---------------------------------------------------------------------------
/// @brief
///  Function to invoke  MSER, with additional outputs for each contour.
/// 
///   \n\b ATTENTION: The signature of this function will be changed to reduce complexity
///   and memory usage when the 2.0.0 release of this library is made.  
///   Until 2.0.0, the developer should use this implementation with the expectation of
///   moving to a different signature when transitioning to 2.0.0.
///   \n\n
///   
///   @param mserHandle     The MSER Handle returned by init.
///   @param srcPtr         Pointer to an image array (unsigned char ) for which MSER has to be done.
///   @param srcWidth       Width of the source image.
///   @param srcHeight      Height of the source image.
///   @param srcStride      Stride of the source image.
///   @param maxContours    Maximum contours that will be returned. Need to be set to 2x the maximum contours.
///                         Application dependent. OCR usually requires 100-1000 contours
///                         Segmentation usually requires 50-100
///   @param numContours    Output, Number of MSER contours in the region.
///   @param numPointsInContour    Output, Number of points in each contour. This will have values filled up
///                                for the first (*numContours) values. This memory has to be allocated by 
///                                the caller. 
///   @param pointsArraySize Size of the output points Array.
///                          Typical size: (# of pixels in source image)*30
///   @param pointsArray     Output. This is the points in all the contours. This is a linear array, whose memory
///                          has to be allocated by the caller. 
///                          Typical allocation size:  pointArraySize
///                          pointsArray[0...numPointsInContour[0]-1] defines the first MSER region;
///                          pointsArray[numPointsInContour[0] .. numPointsInContour[1]-1] defines 2nd MSER region
///                          and so on.
///   @param contourVariation    Output, Variation for each contour from previous grey level.
///                                This will have values filled up
///                                for the first (*numContours) values. This memory has to be allocated by 
///                                the caller with size of maxContours.
///   @param contourPolarity     Output, Polarity for each contour. This value is 1 if this is a MSER+ region, 
///                              -1 if this is a MSER- region. . This will have values filled up
///                                for the first (*numContours) values. This memory has to be allocated by 
///                                the caller with size of maxContours. 
///   @param contourNodeId       Output, Node id for each contour.  This will have values filled up
///                                for the first (*numContours) values. This memory has to be allocated by 
///                                the caller with size of maxContours 
///   @param contourNodeCounter    Output, Node counter for each contour. This will have values filled up
///                                for the first (*numContours) values. This memory has to be allocated by 
///                                the caller with size of maxContours. 
/// 
///   @ingroup object_detection
//------------------------------------------------------------------------------  
FASTCV_API void
fcvMserExtu8( void *mserHandle,
                const uint8_t* __restrict srcPtr,unsigned int srcWidth, 
                unsigned int srcHeight, unsigned int srcStride, 
                unsigned int maxContours,
                unsigned int * __restrict numContours, unsigned int * __restrict numPointsInContour   ,
                unsigned int* __restrict pointsArray, unsigned int pointsArraySize,
                unsigned int * __restrict contourVariation,
                int * __restrict contourPolarity,
                unsigned int * __restrict contourNodeId,
                unsigned int * __restrict contourNodeCounter
              );



//---------------------------------------------------------------------------
/// @brief
///   Creates 2D gradient from source illuminance data.
///   This function considers only the left/right neighbors
///   for x-gradients and top/bottom neighbors for y-gradients.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvImageGradientInterleavedf32_v2(). In the 2.0.0 release, 
///   fcvImageGradientInterleavedf32_v2 will be renamed to fcvImageGradientInterleavedf32
///   and the signature of fcvImageGradientInterleavedf32 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param src
///  Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///    Width of src data to create gradient.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///    Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///
/// @param gradients
///    Buffer to store gradient. Must be 2*(width-1)*(height-1) in size.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageGradientInterleavedf32( const uint8_t* __restrict src,
                                unsigned int              srcWidth,
                                unsigned int              srcHeight,
                                unsigned int              srcStride,
                                float* __restrict         gradients );

//---------------------------------------------------------------------------
/// @brief
///   Creates 2D gradient from source illuminance data.
///   This function considers only the left/right neighbors
///   for x-gradients and top/bottom neighbors for y-gradients.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvImageGradientInterleavedf32() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvImageGradientInterleavedf32,
///   \a fcvImageGradientInterleavedf32_v2 will be removed, and the current signature
///   for \a fcvImageGradientInterleavedf32 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvImageGradientInterleavedf32 when transitioning to 2.0.0.
///   \n\n
///
/// @param src
///   Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param gradients
///   Buffer to store gradient. Must be 2*(width-1)*(height-1) in size.
///   \n\b NOTE: data should be 128-bit aligned.
/// 
/// @param gradStride
///   Stride (in bytes) of the interleaved gradients array.
///   \n\b NOTE: if 0, srcStride is set as (srcWidth-2)*2*sizeof(float).
///   \n\b WARNING: must be multiple of 32 ( 8 * 4-byte values ), and at least as much as 8 * srcWidth if not 0.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageGradientInterleavedf32_v2( const uint8_t* __restrict src,
                                   unsigned int              srcWidth,
                                   unsigned int              srcHeight,
                                   unsigned int              srcStride,
                                   float* __restrict         gradients,
                                   unsigned int              gradStride );

//---------------------------------------------------------------------------
/// @brief
///   Creates 2D gradient from source illuminance data.
///   This function considers only the left/right neighbors
///   for x-gradients and top/bottom neighbors for y-gradients.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvImageGradientPlanars16_v2(). In the 2.0.0 release, 
///   fcvImageGradientPlanars16_v2 will be renamed to fcvImageGradientPlanars16
///   and the signature of fcvImageGradientPlanars16 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param src
///   Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///    Width of src data to create gradient.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///    Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///
/// @param dx
///    Buffer to store horizontal gradient. Must be (srcWidth)*(srcHeight) in size.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param dy
///    Buffer to store vertical gradient. Must be (srcWidth)*(srcHeight) in size.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageGradientPlanars16( const uint8_t* __restrict src,
                           unsigned int              srcWidth,
                           unsigned int              srcHeight,
                           unsigned int              srcStride,
                           int16_t* __restrict       dx,
                           int16_t* __restrict       dy );

//---------------------------------------------------------------------------
/// @brief
///   Creates 2D gradient from source illuminance data.
///   This function considers only the left/right neighbors
///   for x-gradients and top/bottom neighbors for y-gradients.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvImageGradientPlanars16() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvImageGradientPlanars16,
///   \a fcvImageGradientPlanars16_v2 will be removed, and the current signature
///   for \a fcvImageGradientPlanars16 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvImageGradientPlanars16 when transitioning to 2.0.0.
///   \n\n
///
/// @param src
///   Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///    Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param dx
///    Buffer to store horizontal gradient. Must be (srcWidth)*(srcHeight) in size.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param dy
///    Buffer to store vertical gradient. Must be (srcWidth)*(srcHeight) in size.
///   \n\b NOTE: data should be 128-bit aligned.
/// 
/// @param dxyStride
///   Stride (in bytes) of 'dx' and 'dy' arrays.
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 16 (8 * 2-bytes per gradient value), and at least as much as srcWidth if not 0.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageGradientPlanars16_v2( const uint8_t* __restrict src,
                              unsigned int              srcWidth,
                              unsigned int              srcHeight,
                              unsigned int              srcStride,
                              int16_t* __restrict       dx,
                              int16_t* __restrict       dy,
                              unsigned int              dxyStride );

//---------------------------------------------------------------------------
/// @brief
///   Creates 2D gradient from source illuminance data.
///   This function considers only the left/right neighbors
///   for x-gradients and top/bottom neighbors for y-gradients.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvImageGradientPlanarf32_v2(). In the 2.0.0 release, 
///   fcvImageGradientPlanarf32_v2 will be renamed to fcvImageGradientPlanarf32
///   and the signature of fcvImageGradientPlanarf32 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param src
///  Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///    Width of src data to create gradient.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///    Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///
/// @param dx
///    Buffer to store horizontal gradient. Must be (width)*(height) in size.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param dy
///    Buffer to store vertical gradient. Must be (width)*(height) in size.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageGradientPlanarf32( const uint8_t* __restrict src,
                           unsigned int              srcWidth,
                           unsigned int              srcHeight,
                           unsigned int              srcStride,
                           float* __restrict         dx,
                           float* __restrict         dy );



//---------------------------------------------------------------------------
/// @brief
///   Creates 2D gradient from source illuminance data.
///   This function considers only the left/right neighbors
///   for x-gradients and top/bottom neighbors for y-gradients.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvImageGradientPlanarf32() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvImageGradientPlanarf32,
///   \a fcvImageGradientPlanarf32_v2 will be removed, and the current signature
///   for \a fcvImageGradientPlanarf32 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvImageGradientPlanarf32 when transitioning to 2.0.0.
///   \n\n
///
/// @param src
///   Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param dx
///   Buffer to store horizontal gradient. Must be (srcWidth)*(srcHeight) in size.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param dy
///   Buffer to store vertical gradient. Must be (srcWidth)*(srcHeight) in size.
///   \n\b NOTE: data should be 128-bit aligned.
/// 
/// @param dxyStride
///   Stride of Gradient values ('dx' and 'dy' arrays) measured in bytes.
///   \n\b NOTE: if 0, srcStride is set as 4*srcWidth.
///   \n\b WARNING: must be multiple of 32 (8 * 4-bytes per gradient value), and at least as much as 4*srcWidth if not 0.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageGradientPlanarf32_v2( const uint8_t* __restrict src,
                              unsigned int              srcWidth,
                              unsigned int              srcHeight,
                              unsigned int              srcStride,
                              float* __restrict         dx,
                              float* __restrict         dy,
                              unsigned int              dxyStride );


//------------------------------------------------------------------------------
/// @brief
///   Extracts FAST corners from the image. This function tests the whole image
///   for corners (apart from the border). FAST-9 looks for continuous segments on the
///   pixel ring of 9 pixels or more.
///
/// @param src
///   Pointer to grayscale image with one byte per pixel
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Input image width
///   \n\b WARNING: must be a multiple of 8.
///   \n\b WARNING: must be <= 2048.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2). If 0 is passed, srcStride is set to width.
///   \n\b WARNING: must be a multiple of 8.
///
/// @param barrier
///  FAST threshold. The threshold is used to compare difference between intensity value of 
///  the central pixel and pixels on a circle surrounding this pixel.
///
/// @param border
///   Number for pixels to ignore from top,bottom,right,left of the image
///
/// @param xy
///   pointer to  the output array containing the interleaved x,y position of the
///   detected corners
///   \n e.g. struct { int x, y; } xy;
///   \n\b WARNING: must be 128-bit aligned.
///   \n\b NOTE: Remember to allocate double the size of @param nCornersMax
///
/// @param nCornersMax
///   Maximum number of corners. The function exits when the maximum number of
///   corners is exceeded
///
/// @param nCorners
///   pointer to an integer storing the number of corners detected
///
/// @return
///   0 if successful.
///
/// 
///
/// @ingroup feature_detection
//------------------------------------------------------------------------------

FASTCV_API void
fcvCornerFast9u8( const uint8_t* __restrict src,
                  unsigned int              srcWidth,
                  unsigned int              srcHeight,
                  unsigned int              srcStride,
                  int                       barrier,
                  unsigned int              border,
                  uint32_t* __restrict      xy,
                  unsigned int              nCornersMax,
                  uint32_t* __restrict      nCorners );


//------------------------------------------------------------------------------
/// @brief
///   Extracts FAST corners from the image. This function takes a bit mask so
///   that only image areas masked with '0' are tested for corners (if these
///   areas are also not part of the border). FAST-9 looks for continuous segments on the
///   pixel ring of 9 pixels or more.
///
/// @param src
///   pointer to grayscale image with one byte per pixel
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   image width
///   \n\b WARNING: must be <= 2048.
///   \n\b WARNING: must be a multiple of 8.
///
/// @param srcHeight
///   image height
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b WARNING: must be a multiple of 8. If left at 0 srcStride is default to srcWidth.
///
/// @param barrier
///  FAST threshold. The threshold is used to compare difference between intensity value of 
///  the central pixel and pixels on a circle surrounding this pixel.
///
/// @param border
///   Number for pixels to ignore from top,bottom,right,left of the image
///
/// @param xy
///   pointer to the output array containing the interleaved x,y position of the
///   detected corners
///   \n\b WARNING: must be 128-bit aligned.
///   \n\b NOTE: Remember to allocate double the size of @param nCornersMax
///
/// @param nCornersMax
///   Maximum number of corners. The function exits when the maximum number of corners
///   is exceeded
///
/// @param nCorners
///   pointer to an integer storing the number of corners detected
///
/// @param mask
///   Per-pixel mask for each pixel represented in input image. 
///   If a bit set to 0, pixel will be a candidate for corner detection. 
///   If a bit set to 1, pixel will be ignored.
///
/// @param maskWidth
///   Width of the mask. Both width and height of the mask must be 'k' times image width and height, 
///   where k = 1/2, 1/4 , 1/8 , 1, 2, 4 and 8.
///
/// @param maskHeight
///   Height of the mask. Both width and height of the mask must be 'k' times image width and height, 
///   where k = 1/2, 1/4 , 1/8 , 1, 2, 4 and 8.
///
/// @return
///   0 if successful.
///
/// 
///
/// @ingroup feature_detection
//------------------------------------------------------------------------------

FASTCV_API void
fcvCornerFast9InMasku8( const uint8_t* __restrict src,
                        unsigned int              srcWidth,
                        unsigned int              srcHeight,
                        unsigned int              srcStride,
                        int                       barrier,
                        unsigned int              border,
                        uint32_t* __restrict      xy,
                        unsigned int              nCornersMax,
                        uint32_t* __restrict      nCorners,
                        const uint8_t* __restrict mask,
                        unsigned int              maskWidth,
                        unsigned int              maskHeight );

//------------------------------------------------------------------------------
/// @brief
///   Extracts FAST corners from the image. This function tests the whole image
///   for corners (apart from the border). FAST-10 looks for continuous segments on the
///   pixel ring of 10 pixels or more.
///
/// @param src
///   Pointer to grayscale image with one byte per pixel
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Input image width
///   \n\b WARNING: must be a multiple of 8.
///   \n\b WARNING: must be <= 2048.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2). If 0 is passed, srcStride is set to width.
///
/// @param barrier
///  FAST threshold. The threshold is used to compare difference between intensity value of 
///  the central pixel and pixels on a circle surrounding this pixel.
///
/// @param border
///   Number for pixels to ignore from top,bottom,right,left of the image
///
/// @param xy
///   pointer to  the output array containing the interleaved x,y position of the
///   detected corners
///   \n e.g. struct { int x, y; } xy;
///   \n\b WARNING: must be 128-bit aligned.
///   \n\b NOTE: Remember to allocate double the size of @param nCornersMax
///
/// @param nCornersMax
///   Maximum number of corners. The function exists when the maximum number of
///   corners is exceeded
///
/// @param nCorners
///   pointer to an integer storing the number of corners detected
///
/// @return
///   0 if successful.
///
/// 
///
/// @ingroup feature_detection
//------------------------------------------------------------------------------

FASTCV_API void
fcvCornerFast10u8( const uint8_t* __restrict src,
                   uint32_t                  srcWidth,
                   uint32_t                  srcHeight,
                   uint32_t                  srcStride,
                   int32_t                   barrier,
                   uint32_t                  border,
                   uint32_t* __restrict      xy,
                   uint32_t                  nCornersMax,
                   uint32_t* __restrict      nCorners);


//------------------------------------------------------------------------------
/// @brief
///   Extracts FAST corners from the image. This function takes a bit mask so
///   that only image areas masked with '0' are tested for corners (if these
///   areas are also not part of the border). FAST-10 looks for continuous segments on the
///   pixel ring of 10 pixels or more.
///
/// @param src
///   pointer to grayscale image with one byte per pixel
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   image width
///   \n\b WARNING: must be <= 2048.
///   \n\b WARNING: must be a multiple of 8.
///
/// @param srcHeight
///   image height
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///
/// @param barrier
///  FAST threshold. The threshold is used to compare difference between intensity value of 
///  the central pixel and pixels on a circle surrounding this pixel.
///
/// @param border
///   Number for pixels to ignore from top,bottom,right,left of the image
///
/// @param xy
///   pointer to the output array containing the interleaved x,y position of the
///   detected corners
///   \n\b WARNING: must be 128-bit aligned.
///   \n\b NOTE: Remember to allocate double the size of @param nCornersMax
///
/// @param nCornersMax
///   Maximum number of corners. The function exists when the maximum number of corners
///   is exceeded
///
/// @param nCorners
///   pointer to an integer storing the number of corners detected
///
/// @param mask
///   Per-pixel mask for each pixel represented in input image. 
///   If a bit set to 0, pixel will be a candidate for corner detection. 
///   If a bit set to 1, pixel will be ignored.
///
/// @param maskWidth
///   Width of the mask. Both width and height of the mask must be 'k' times image width and height, 
///   where k = 1/2, 1/4 , 1/8 , 1, 2, 4 and 8.
///
/// @param maskHeight
///   Height of the mask. Both width and height of the mask must be 'k' times image width and height, 
///   where k = 1/2, 1/4 , 1/8 , 1, 2, 4 and 8.
///
/// @return
///   0 if successful.
///
/// 
///
/// @ingroup feature_detection
//------------------------------------------------------------------------------

FASTCV_API void
fcvCornerFast10InMasku8( const uint8_t* __restrict src,
                         uint32_t                  srcWidth,
                         uint32_t                  srcHeight,
                         uint32_t                  srcStride,
                         int32_t                   barrier,
                         uint32_t                  border,
                         uint32_t* __restrict      xy,
                         uint32_t                  nCornersMax,
                         uint32_t* __restrict      nCorners,
                         const uint8_t* __restrict mask,
                         uint32_t                  maskWidth,
                         uint32_t                  maskHeight );


//------------------------------------------------------------------------------
/// @brief
///   Extracts Harris corners from the image. This function tests the whole
///   image for corners (apart from the border). 
///
/// @param src
///   Pointer to grayscale image with one byte per pixel
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Input image width
///   \n\b WARNING: must be a multiple of 8.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///
/// @param border
///   Number for pixels to ignore from top,bottom,right,left of the image
///
/// @param xy
///   pointer to  the output array containing the interleaved x,y position of the
///   detected corners
///   \n\b WARNING: must be 128-bit aligned.
///   \n\b NOTE: Remember to allocate double the size of @param nCornersMax
///
/// @param nCornersMax
///   Maximum number of corners. The function exits when the maximum number of
///   corners is exceeded
///
/// @param nCorners
///   pointer to an integer storing the number of corners detected
/// 
/// @param threshold
///   Minimum "Harris Score" or "Harris Corner Response" of a pixel for it to be 
///   regarded as a corner.
///
/// @return
///   0 if successful.
///
/// 
///
/// @ingroup feature_detection
//------------------------------------------------------------------------------

FASTCV_API void
fcvCornerHarrisu8( const uint8_t* __restrict src,
                   unsigned int              srcWidth,
                   unsigned int              srcHeight,
                   unsigned int              srcStride,
                   unsigned int              border,
                   uint32_t* __restrict      xy,
                   unsigned int              nCornersMax,
                   uint32_t* __restrict      nCorners,
                   int                       threshold );

//------------------------------------------------------------------------------
/// @brief
///   Local Harris Max applies the Harris Corner algorithm on an 11x11 patch 
///   within an image to determine if a corner is present. 
///
/// @param src
///   Pointer to grayscale image with one byte per pixel
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Input image width
///   \n\b WARNING: must be a multiple of 8.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).  If srcStride == 0, then will use srcWidth.
///
/// @param posX
///   Center X coordinate of the search window
///
/// @param posY
///   Center Y coordinate of the search window
///
/// @param maxX
///   pointer to the X coordinate identified as a corner
///
/// @param maxY
///   pointer to the Y coordinate identified as a corner
///
/// @param maxScore
///   pointer to the Harris score associated with the corner
///
/// @return
///   0 if no corner is found (maxX, maxY, and maxScore are invalid)
///     or if posX and/or posY position the patch outside of the range of
///     the source image.
///   1 if a corner is found (maxX, maxY, and maxScore are valid)
///
/// 
///
/// @ingroup feature_detection
//------------------------------------------------------------------------------

FASTCV_API unsigned int
fcvLocalHarrisMaxu8( const uint8_t* __restrict src,
                     unsigned int              srcWidth,
                     unsigned int              srcHeight,
                     unsigned int              srcStride,
                     unsigned int              posX,
                     unsigned int              posY,
                     unsigned int             *maxX,
                     unsigned int             *maxY,
                     int                      *maxScore);


//------------------------------------------------------------------------------
/// @brief
///   Extracts Harris corners from the image. This function takes a bit mask so
///   that only image areas masked with '0' are tested for corners (if these
///   areas are also not part of the border). 
///
/// @param src
///   pointer to grayscale image with one byte per pixel
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   image width
///   \n\b WARNING: must be a multiple of 8.
///
/// @param srcHeight
///   image height
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///
/// @param border
///   Number for pixels to ignore from top,bottom,right,left of the image
///
/// @param xy
///   pointer to  the output array containing the interleaved x,y position of the
///   detected corners
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param nCornersMax
///   Maximum number of corners. The function exits when the maximum number of corners
///   is exceeded
///
/// @param nCorners
///   pointer to an integer storing the number of corners detected
///
/// @param threshold
///   Minimum "Harris Score" or "Harris Corner Respose" of a pixel for it to be 
///   regarded as a corner.
///
/// @param mask
///   Per-pixel mask for each pixel represented in input image. 
///   If a bit set to 0, pixel will be a candidate for corner detection. 
///   If a bit set to 1, pixel will be ignored.
///
/// @param maskWidth
///   Width of the mask. Both width and height of the mask must be 'k' times image width and height, 
///   where k = 1/2, 1/4 , 1/8 , 1, 2, 4 and 8.
///
/// @param maskHeight
///   Height of the mask. Both width and height of the mask must be 'k' times image width and height, 
///   where k = 1/2, 1/4 , 1/8 , 1, 2, 4 and 8.
///
/// @return
///   0 if successful.
///
/// 
///
/// @ingroup feature_detection
//------------------------------------------------------------------------------

FASTCV_API void
fcvCornerHarrisInMasku8( const uint8_t* __restrict src,
                         unsigned int              srcWidth,
                         unsigned int              srcHeight,
                         unsigned int              srcStride,
                         unsigned int              border,
                         uint32_t* __restrict      xy,
                         unsigned int              nCornersMax,
                         uint32_t* __restrict      nCorners,
                         int                       threshold,
                         const uint8_t* __restrict mask,
                         unsigned int              maskWidth,
                         unsigned int              maskHeight );


//---------------------------------------------------------------------------
/// @brief
///   Computes affine trans. for a given set of corresponding features points
///   using a linear least square colver based on Cholkesky decomposition.
///
/// @param corrs
///  Correspondence data struct containing coords of points in two frames
///
/// @param affine
///  3 x 3 affine matrix (computed best fit affine transformation)
///
/// @ingroup 3D_reconstruction
//---------------------------------------------------------------------------

FASTCV_API void
fcvGeomAffineFitf32( const fcvCorrespondences* __restrict corrs,
                     float* __restrict                    affine );


//------------------------------------------------------------------------------
/// @brief
///   Evaluates specified affine transformation against provided points
///   correspondences. Checks which correspondence members have a projection
///   error that is smaller than the given one (maxSquErr).
///
/// @param corrs
///   Pointer to correspondences structure.
///
/// @param affine
///   Affine matrix representing relationship between ptTo and ptFrom
///   correspondences stored as 3x3 floating point matrix formatted as
///   @todo r0h0, r0h1
///   Pointer storage must be at least a 9-element floating point array.
///
/// @param maxsqerr
///   Maximum error value squared.
///
/// @param inliers
///   Output array for those indices that passed the test - the array MUST
///   be able to store numIndices items.
///
/// @param numinliers
///   Output number of corrs that passed the test.
///
/// @return
///
/// 
///
/// @ingroup 3D_reconstruction
//------------------------------------------------------------------------------

FASTCV_API int
fcvGeomAffineEvaluatef32( const fcvCorrespondences* __restrict corrs,
                          float* __restrict                    affine,
                          float                                maxsqerr,
                          uint16_t* __restrict                 inliers,
                          int32_t*                             numinliers );


//------------------------------------------------------------------------------
/// @brief
///   Performs cholesky homography fitting on specified points correspondences.
///
/// @details
///   Output precision is within 3e-3
///
/// @param corrs
///   Pointer to correspondences structure.
///
/// @param homography
///   3x3 floating point matrix formatted as @todo r0h0, r0h1
///   Pointer storage must be at least a 9-element floating point array.
///
/// 
///
/// @ingroup 3D_reconstruction
//------------------------------------------------------------------------------

FASTCV_API void
fcvGeomHomographyFitf32( const fcvCorrespondences* __restrict corrs,
                         float* __restrict                    homography );


//------------------------------------------------------------------------------
/// @brief
///   Evaluates specified homography against provided points correspondences.
///   Check which correspondence members have a projection error that is
///   smaller than the given one (maxSquErr).
///
/// @param corrs
///   Pointer to correspondences structure.
///
/// @param homography
///   Homography representing relationship between ptTo and ptFrom
///   correspondences stored as 3x3 floating point matrix formatted as
///   @todo r0h0, r0h1
///   Pointer storage must be at least a 9-element floating point array.
///
/// @param maxsqerr
///   Maximum error value squared.
///
/// @param inliers
///   Output array for those indices that passed the test - the array MUST
///   be able to store numIndices items.
///
/// @param numinliers
///   Output number of corrs that passed the test.
///
/// @return
///   0 that error is less than maximum error, -1 greater or equal to maximum
///   error.
///
/// 
///
/// @ingroup 3D_reconstruction
//------------------------------------------------------------------------------

FASTCV_API int
fcvGeomHomographyEvaluatef32( const fcvCorrespondences* __restrict corrs,
                              float* __restrict                    homography,
                              float                                maxsqerr,
                              uint16_t* __restrict                 inliers,
                              int32_t*                             numinliers );


//------------------------------------------------------------------------------
/// @brief
///   Performs cholesky pose fitting on specified points correspondences.
///   Takes a pose and uses the correspondences to refine it using iterative
///   Gauss-Newton optimization.
///
/// @param corrs
///   Pointer to correspondences structure.
///
/// @param minIterations
///   Minimum number of iterations to refine.
///
/// @param maxIterations
///   Maximum number of iterations to refine.
///
/// @param stopCriteria
///   Improvement threshold, iterations stop if improvement is less than this
///   value.
///
/// @param initpose
///   Pose representing initial pose
///   correspondences stored as a
///   3x4 transformation matrix in the form [R|t], where R is a 3x3 rotation
///   matrix and t is the translation vector. The matrix  stored in pose is row
///   major ordering: \n
///   a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34 where the
///   matrix is: \n
///   | a11, a12, a13 , a14|\n
///   | a21, a22, a23, a24 |\n
///   | a31, a32, a33, a34 |\n
///   Pointer storage must be at least a 12-element floating point array.
///
/// @param refinedpose
///   Pose representing refined pose
///   correspondences stored as a
///   3x4 transformation matrix in the form [R|t], where R is a 3x3 rotation
///   matrix and t is the translation vector. The matrix  stored in pose is row
///   major ordering: \n
///   a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34 where the
///   matrix is: \n
///   | a11, a12, a13 , a14|\n
///   | a21, a22, a23, a24 |\n
///   | a31, a32, a33, a34 |\n
///   Pointer storage must be at least a 12-element floating point array.
///
/// @return
///   Final reprojection error.
///
/// 
///
/// @ingroup 3D_reconstruction
//------------------------------------------------------------------------------

FASTCV_API float
fcvGeomPoseRefineGNf32( const fcvCorrespondences* __restrict corrs,
                        short                                minIterations,
                        short                                maxIterations,
                        float                                stopCriteria,
                        float*                               initpose,
                        float*                               refinedpose );

//------------------------------------------------------------------------------
/// @brief
///   Update and compute the differential pose based on the specified points correspondences
///   This function and fcvGeomPoseOptimizeGNf32
///   can be used iteratively to perform poseRefine GN.
///
/// @param projected
///   2D position after projection
///
/// @param reprojErr
///   2D reprojection error in camera coordinates (not in pixels!)
///
/// @param invz
///   Inverse depth (z)
///
/// @param reprojVariance
///   Reprojection variance in camera coordinates
///
/// @param numpts
///    Number of points
///
/// @param pose
///   Pose representing differential pose
///   correspondences stored as a
///   3x4 transformation matrix in the form [R|t], where R is a 3x3 rotation
///   matrix and t is the translation vector. The matrix  stored in pose is row
///   major ordering: \n
///   a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34 where the
///   matrix is: \n
///   | a11, a12, a13 , a14|\n
///   | a21, a22, a23, a24 |\n
///   | a31, a32, a33, a34 |\n
///   Pointer storage must be at least a 12-element floating point array.
///
/// @return
///   0 if successfully clustered, otherwise error code
///
/// 
///
/// @ingroup 3D_reconstruction
//------------------------------------------------------------------------------

FASTCV_API int
fcvGeomPoseUpdatef32(
   const float* __restrict projected,
   const float* __restrict reprojErr,
   const float* __restrict invz,
   const float* __restrict reprojVariance,
   unsigned int                numpts,
   float*       __restrict pose );

//------------------------------------------------------------------------------
/// @brief
///   Update the pose based on the specified points correspondences
///   using Gauss-Newton optimization. This function and fcvGeomPoseEvaluateErrorf32
///   can be used iteratively to perform poseRefine GN.
///
/// @param projected
///   2D position after projection
///
/// @param reprojErr
///   2D reprojection error in camera coordinates (not in pixels!)
///
/// @param invz
///   Inverse depth (z)
///
/// @param reprojVariance
///   Reprojection variance in camera coordinates
///
/// @param numpts
///    Number of points
///
/// @param pose
///   Pose representing updated pose
///   correspondences stored as a
///   3x4 transformation matrix in the form [R|t], where R is a 3x3 rotation
///   matrix and t is the translation vector. The matrix  stored in pose is row
///   major ordering: \n
///   a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34 where the
///   matrix is: \n
///   | a11, a12, a13 , a14|\n
///   | a21, a22, a23, a24 |\n
///   | a31, a32, a33, a34 |\n
///   Pointer storage must be at least a 12-element floating point array.
///
/// @return
///   0 if successfully clustered, otherwise error code
///
/// 
///
/// @ingroup 3D_reconstruction
//------------------------------------------------------------------------------

FASTCV_API int
fcvGeomPoseOptimizeGNf32( const float* __restrict projected,
                          const float* __restrict reprojErr,
                          const float* __restrict invz,
                          const float* __restrict reprojVariance,
                          unsigned int            numpts,
                          float*       __restrict pose );


//------------------------------------------------------------------------------
/// @brief
///   Calculate the reprojection error based on the input pose.
///   This function and fcvGeomPoseOptimizef32 can be used iteratively
///   to perform poseRefine (GN or LM)..
///
/// @param corrs
///   Pointer to correspondences structure.
///
/// @param pose
///   Pose representing updated pose
///   correspondences stored as a
///   3x4 transformation matrix in the form [R|t], where R is a 3x3 rotation
///   matrix and t is the translation vector. The matrix  stored in pose is row
///   major ordering: \n
///   a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34 where the
///   matrix is: \n
///   | a11, a12, a13 , a14|\n
///   | a21, a22, a23, a24 |\n
///   | a31, a32, a33, a34 |\n
///   Pointer storage must be at least a 12-element floating point array.
///
/// @param projected
///   2D position after projection
///
/// @param reprojErr
///   2D reprojection error in camera coordinates (not in pixels!)
///
/// @param invz
///   Inverse depth (z)
///
/// @param reprojVariance
///   Reprojection variance in camera coordinates
///
/// @return
///   Reprojection error.
///
/// 
///
/// @ingroup 3D_reconstruction
//------------------------------------------------------------------------------

FASTCV_API float
fcvGeomPoseEvaluateErrorf32( const fcvCorrespondences* __restrict corrs,
                             const float*              __restrict pose,
                             float*                    __restrict projected,
                             float*                    __restrict reprojErr,
                             float*                    __restrict invz,
                             float*                    __restrict reprojVariance );


//------------------------------------------------------------------------------
/// @brief
///   Checks which members have a projection error that is smaller than the
///   given one.
///
/// @param corrs
///   Pointer to correspondences structure.
///
/// @param pose
///   Pose representing relationship between ptTo and ptFrom
///   correspondences stored as a
///   3x4 transformation matrix in the form [R|t], where R is a 3x3 rotation
///   matrix and t is the translation vector. The matrix  stored in pose is row
///   major ordering: \n
///   a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34 where the
///   matrix is: \n
///   | a11, a12, a13 , a14|\n
///   | a21, a22, a23, a24 |\n
///   | a31, a32, a33, a34 |\n
///   Pointer storage must be at least a 12-element floating point array.
///
/// @param maxSquErr
///   Maximum error value squared.
///
/// @param inliers
///   Output array for those indices that passed the test - the array MUST
///   be able to store numIndices items.
///
/// @param numInliers
///   Output number of corrs that passed the test.
///
/// @return
///   0 that error is less than maximum error, -1 greater or equal to maximum
///   error.
///
/// 
///
/// @ingroup 3D_reconstruction
//------------------------------------------------------------------------------

FASTCV_API int
fcvGeomPoseEvaluatef32( const fcvCorrespondences* __restrict corrs,
                        const float*                         pose,
                        float                                maxSquErr,
                        uint16_t* __restrict                 inliers,
                        uint32_t*                            numInliers );


//------------------------------------------------------------------------------
/// @brief
///   Estimates a 6DOF pose 
///  \n\b NOTE: Given the coordinates of three 3D points (in world reference frame),
///             and their corresponding perspective projections in an image,
///             this algorithm determines the position and orientation of the camera in
///             the world reference frame. The function provides up to four solutions
///             that can be disambiguated using a fourth point.
///             When used in conjunction with RANSAC, this function can perform efficient outlier rejection.
///             Two degenerate cases should be avoided when using this function:
///             - Indeterminate configuration:
///                  When the three points are collinear in space, there will be a family of poses mapping the
///                  three points to the same image points.
///             - Unstable configuration:
///                  The camera center is located on a circular cylinder passing through the three points and
///                  the camera optical axis is perpendicular to the plane derived by the three points.
///                  With this configuration, a small change in the position of the three points will
///                  result in a large change of the estimated pose..
///
/// @param corrs
///  2D-3D correspondence points
///
/// @param pose
///  computed pose (numPoses * 12 data)
///
/// @param numPoses (max = 4)
///
/// @ingroup 3D_reconstruction
//------------------------------------------------------------------------------

FASTCV_API void
fcvGeom3PointPoseEstimatef32( const fcvCorrespondences* __restrict corrs,
                                                 float*            pose,
                                               int32_t*            numPoses );


//------------------------------------------------------------------------------
/// @brief
///   3x3 correlation with non-separable kernel.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvFilterCorr3x3s8_v2(). In the 2.0.0 release, 
///   fcvFilterCorr3x3s8_v2 will be renamed to fcvFilterCorr3x3s8
///   and the signature of fcvFilterCorr3x3s8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param kernel
///   2-D 3x3 kernel.
///   \n\b NOTE: Normalized to Q4.4
///
/// @param src
///   Input image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b NOTE: must be an even number
///
/// @param srcHeight
///   Image height.
///   \n\b NOTE: must be an even number
///
/// @param dst
///   Output convolution. Border values are ignored in this function.
///   Size of buffer is srcWidth*srcHeight bytes.
///   \n\b NOTE: Must be same size as src
///   \n\b WARNING: must be 128-bit aligned.
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterCorr3x3s8( const int8_t* __restrict  kernel,
                    const uint8_t* __restrict src,
                    unsigned int              srcWidth,
                    unsigned int              srcHeight,
                    uint8_t* __restrict       dst );


//------------------------------------------------------------------------------
/// @brief
///   3x3 correlation with non-separable kernel.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvFilterCorr3x3s8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvFilterCorr3x3s8,
///   \a fcvFilterCorr3x3s8_v2 will be removed, and the current signature
///   for \a fcvFilterCorr3x3s8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvFilterCorr3x3s8 when transitioning to 2.0.0.
///   \n\n
///
/// @param kernel
///   2-D 3x3 kernel.
///   \n\b NOTE: Normalized to Q4.4
///
/// @param src
///   Input image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b NOTE: must be an even number
///
/// @param srcHeight
///   Image height.
///   \n\b NOTE: must be an even number
///
/// @param srcStride
///   Image stride.
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
/// 
/// @param dst
///   Output convolution. Size of buffer is dstStride*srcHeight bytes.
///   \n\b NOTE: Must be same size as src
///   \n\b WARNING: must be 128-bit aligned.
/// 
/// @param dstStride
///   Output stride. Border values are ignored in this function.
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterCorr3x3s8_v2( const int8_t* __restrict  kernel,
                       const uint8_t* __restrict src,
                       unsigned int              srcWidth,
                       unsigned int              srcHeight,
                       unsigned int              srcStride,
                       uint8_t* __restrict       dst,
                       unsigned int              dstStride );


//------------------------------------------------------------------------------
/// @brief
///   9x9 correlation with separable kernel.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvFilterCorrSep9x9s16_v2(). In the 2.0.0 release, 
///   fcvFilterCorrSep9x9s16_v2 will be renamed to fcvFilterCorrSep9x9s16
///   and the signature of fcvFilterCorrSep9x9s16 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param kernel
///   1-D kernel in Q15.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param src
///   Input image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///   \n\b WARNING: must be > 8.
///
/// @param srcHeight
///   Image height.
///
/// @param tmp
///   Temporary image buffer used internally.
///   Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: Must be same size as src
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param dst
///   Output correlation. Border values are ignored in this function.
///   Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: Must be same size as src
///   \n\b WARNING: must be 128-bit aligned.
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterCorrSep9x9s16( const int16_t* __restrict kernel,
                        const int16_t* __restrict src,
                        unsigned int              srcWidth,
                        unsigned int              srcHeight,
                        int16_t* __restrict       tmp,
                        int16_t* __restrict       dst );


//---------------------------------------------------------------------------
/// @brief
///   9x9 FIR filter (convolution) with seperable kernel.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvFilterCorrSep9x9s16() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvFilterCorrSep9x9s16,
///   \a fcvFilterCorrSep9x9s16_v2 will be removed, and the current signature
///   for \a fcvFilterCorrSep9x9s16 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvFilterCorrSep9x9s16 when transitioning to 2.0.0.
///   \n\n
///
/// @param kernel
///   1-D kernel.
///
/// @param srcImg
///   Input image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param srcWidth
///   Image tile width.
///
/// @param srcHeight
///   Image tile height.
///
/// @param srcStride
///   source Image width
///
/// @param tmpImg
///   Temporary image scratch space used internally.
///   \n\b NOTE: Size = width * ( height + knlSize - 1 )
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param dstImg
///   Output correlation.  Border values are ignored in this function.
///   Size of buffer is dstStride*srcHeight bytes.
///   \n\b NOTE: Size = width * heigth
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param dstStride
///   dst Image width
/// 
/// @ingroup image_processing
//---------------------------------------------------------------------------
FASTCV_API void
fcvFilterCorrSep9x9s16_v2( const int16_t* __restrict kernel,
                           const int16_t* __restrict srcImg,
                           unsigned int              srcWidth, 
                           unsigned int              srcHeight, 
                           unsigned int              srcStride,
                           int16_t* __restrict       tmpImg,
                           int16_t* __restrict       dstImg, 
                           unsigned int              dstStride );


//------------------------------------------------------------------------------
/// @brief
///   11x11 correlation with separable kernel.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvFilterCorrSep11x11s16_v2(). In the 2.0.0 release, 
///   fcvFilterCorrSep11x11s16_v2 will be renamed to fcvFilterCorrSep11x11s16
///   and the signature of fcvFilterCorrSep11x11s16 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param kernel
///   1-D kernel.
///   \n\b NOTE: array must be >=12 elements with kernel[11]=0
///   \n\b WARNING: must be 128-bit aligned.
///   \n\b NOTE: Normalized to Q1.15
///
/// @param src
///   Input image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///   \n\b WARNING: must be > 8.
///
/// @param srcHeight
///   Image height.
///
/// @param tmpImg
///   Temporary image scratch space used internally.
///   \n\b NOTE: Must be same size as src
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param dst
///   Output correlation.  Border values are ignored in this function.
///   Size of buffer is srcWidth*srcHeight bytes.
///   \n\b NOTE: Must be same size as src
///   \n\b WARNING: must be 128-bit aligned.
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterCorrSep11x11s16( const int16_t* __restrict kernel,
                          const int16_t* __restrict src,
                          unsigned int              srcWidth,
                          unsigned int              srcHeight,
                          int16_t* __restrict       tmpImg,
                          int16_t* __restrict       dst );


//---------------------------------------------------------------------------
/// @brief
///   11x11 FIR filter (convolution) with seperable kernel.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvFilterCorrSep11x11s16() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvFilterCorrSep11x11s16,
///   \a fcvFilterCorrSep11x11s16_v2 will be removed, and the current signature
///   for \a fcvFilterCorrSep11x11s16 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvFilterCorrSep11x11s16 when transitioning to 2.0.0.
///   \n\n
///
/// @param kernel
///   1-D kernel.
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param srcImg
///   Input image. Size of buffer is srStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param srcWidth
///   Image tile width.
///   \n\b WARNING: must be multiple of 8.
///   \n\b WARNING: must be > 8.
///
/// @param srcHeight
///   Image tile height.
///
/// @param srcStride
///   source Image width
///
/// @param tmpImg
///   Temporary image scratch space used internally.
///   \n\b NOTE: Size = width * ( height + knlSize - 1 )
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param dstImg
///   Output correlation.  Border values are ignored in this function.
///   \n\b NOTE: Size = dstStride * srcHeigth
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param dstStride
///   dst Image width
/// 
/// @ingroup image_processing
//---------------------------------------------------------------------------
FASTCV_API void
fcvFilterCorrSep11x11s16_v2( const int16_t* __restrict kernel,
                             const int16_t* __restrict srcImg,
                             unsigned int              srcWidth, 
                             unsigned int              srcHeight, 
                             unsigned int              srcStride,
                             int16_t* __restrict       tmpImg,
                             int16_t* __restrict       dstImg, 
                             unsigned int              dstStride );


//------------------------------------------------------------------------------
/// @brief
///   13x13 correlation with separable kernel.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvFilterCorrSep13x13s16_v2(). In the 2.0.0 release, 
///   fcvFilterCorrSep13x13s16_v2 will be renamed to fcvFilterCorrSep13x13s16
///   and the signature of fcvFilterCorrSep13x13s16 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param kernel
///   1-D kernel.
///   \n\b NOTE: Normalized to Q1.15
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param src
///   Input image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///   \n\b WARNING: must be > 8.
///
/// @param srcHeight
///   Image height.
///
/// @param tmpImg
///   Temporary image scratch space used internally.
///   \n\b NOTE: Must be same size as src
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param dst
///   Output correlation.  Border values are ignored in this function.
///   \n\b NOTE: Must be same size as src
///   \n\b WARNING: must be 128-bit aligned.
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterCorrSep13x13s16( const int16_t* __restrict kernel,
                          const int16_t* __restrict src,
                          unsigned int              srcWidth,
                          unsigned int              srcHeight,
                          int16_t* __restrict       tmpImg,
                          int16_t* __restrict       dst );


//---------------------------------------------------------------------------
/// @brief
///   13x13 FIR filter (convolution) with seperable kernel.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvFilterCorrSep13x13s16() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvFilterCorrSep13x13s16,
///   \a fcvFilterCorrSep13x13s16_v2 will be removed, and the current signature
///   for \a fcvFilterCorrSep13x13s16 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvFilterCorrSep13x13s16 when transitioning to 2.0.0.
///   \n\n
///
/// @param kernel
///   1-D kernel.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcImg
///   Input image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param srcWidth
///   Image tile width.
///   \n\b WARNING: must be multiple of 8.
///   \n\b WARNING: must be > 8.
///
/// @param srcHeight
///   Image tile height.
///
/// @param srcStride
///   source Image width
///
/// @param tmpImg
///   Temporary image scratch space used internally.
///   \n\b NOTE: Size = width * ( height + knlSize - 1 )
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param dstImg
///   Output correlation.  Border values are ignored in this function.
///   \n\b NOTE: Size = dstStride * srcHeigth
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param dstStride
///   dst Image width
/// 
/// @ingroup image_processing
//---------------------------------------------------------------------------
FASTCV_API void
fcvFilterCorrSep13x13s16_v2( const int16_t* __restrict kernel,
                             const int16_t* __restrict srcImg,
                             unsigned int              srcWidth, 
                             unsigned int              srcHeight, 
                             unsigned int              srcStride,
                             int16_t* __restrict       tmpImg,
                             int16_t* __restrict       dstImg, 
                             unsigned int              dstStride );


//------------------------------------------------------------------------------
/// @brief
///   15x15 correlation with separable kernel.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvFilterCorrSep15x15s16_v2(). In the 2.0.0 release, 
///   fcvFilterCorrSep15x15s16_v2 will be renamed to fcvFilterCorrSep15x15s16
///   and the signature of fcvFilterCorrSep15x15s16 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param kernel
///   1-D kernel.
///   \n\b NOTE: array must be 16 elements with kernel[15]=0
///   \n\b NOTE: Normalized to Q1.15
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param src
///   Input image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///   \n\b WARNING: must be > 8.
///
/// @param srcHeight
///   Image height.
///
/// @param tmpImg
///   Temporary image scratch space used internally.
///   \n\b NOTE: Must be same size as src
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param dst
///   Output correlation. Border values are ignored in this function.
///   \n\b NOTE: Must be same size as src
///   \n\b NOTE: data should be 128-bit aligned
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterCorrSep15x15s16( const int16_t* __restrict kernel,
                          const int16_t* __restrict src,
                          unsigned int              srcWidth,
                          unsigned int              srcHeight,
                          int16_t* __restrict       tmpImg,
                          int16_t* __restrict       dst );


//---------------------------------------------------------------------------
/// @brief
///   15x15 FIR filter (convolution) with seperable kernel.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvFilterCorrSep15x15s16() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvFilterCorrSep15x15s16,
///   \a fcvFilterCorrSep15x15s16_v2 will be removed, and the current signature
///   for \a fcvFilterCorrSep15x15s16 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvFilterCorrSep15x15s16 when transitioning to 2.0.0.
///   \n\n
///
/// @param kernel
///   1-D kernel.
///   \n\b NOTE: array must be 16 elements with kernel[15]=0
///   \n\b NOTE: Normalized to Q1.15
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param srcImg
///   Input image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param srcWidth
///   Image tile width.
///   \n\b WARNING: must be multiple of 8.
///   \n\b WARNING: must be > 8.
///
/// @param srcHeight
///   Image tile height.
///
/// @param srcStride
///   source Image width
///
/// @param tmpImg
///   Temporary image scratch space used internally.
///   \n\b NOTE: Size = width * ( height + knlSize - 1 )
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param dstImg
///   Output correlation. Border values are ignored in this function.
///   \n\b NOTE: Size = dstStride * srcHeigth
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param dstStride
///   dst Image width
/// 
/// @ingroup image_processing
//---------------------------------------------------------------------------
FASTCV_API void
fcvFilterCorrSep15x15s16_v2( const int16_t* __restrict kernel,
                             const int16_t* __restrict srcImg,
                             unsigned int              srcWidth, 
                             unsigned int              srcHeight, 
                             unsigned int              srcStride,
                             int16_t* __restrict       tmpImg,
                             int16_t* __restrict       dstImg, 
                             unsigned int              dstStride );


//------------------------------------------------------------------------------
/// @brief
///   17x17 correlation with separable kernel.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvFilterCorrSep17x17s16_v2(). In the 2.0.0 release, 
///   fcvFilterCorrSep17x17s16_v2 will be renamed to fcvFilterCorrSep17x17s16
///   and the signature of fcvFilterCorrSep17x17s16 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param kernel
///   1-D kernel.
///   \n\b NOTE: Normalized to Q1.15
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param src
///   Input image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///   \n\b WARNING: must be > 8.
///
/// @param srcHeight
///   Image height.
///
/// @param tmpImg
///   Temporary image scratch space used internally.
///   \n\b NOTE: Must be same size as src
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param dst
///   Output correlation.. Border values are ignored in this function.
///   \n\b NOTE: Must be same size as src
///   \n\b WARNING: must be 128-bit aligned.
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterCorrSep17x17s16( const int16_t* __restrict kernel,
                          const int16_t* __restrict src,
                          unsigned int              srcWidth,
                          unsigned int              srcHeight,
                          int16_t* __restrict       tmpImg,
                          int16_t* __restrict       dst );



//---------------------------------------------------------------------------
/// @brief
///   17x17 FIR filter (convolution) with seperable kernel.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvFilterCorrSep17x17s16() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvFilterCorrSep17x17s16,
///   \a fcvFilterCorrSep17x17s16_v2 will be removed, and the current signature
///   for \a fcvFilterCorrSep17x17s16 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvFilterCorrSep17x17s16 when transitioning to 2.0.0.
///   \n\n
///
/// @param kernel
///   1-D kernel.
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param srcImg
///   Input image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param srcWidth
///   Image tile width.
///
/// @param srcHeight
///   Image tile height.
///
/// @param srcStride
///   source Image width
///
/// @param tmpImg
///   Temporary image scratch space used internally.
///   \n\b NOTE: Size = width * ( height + knlSize - 1 )
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param dstImg
///   Output correlation. Border values are ignored in this function.
///   \n\b NOTE: Size = dstStride * srcHeigth
///   \n\b NOTE: data should be 128-bit aligned
///
/// @param dstStride
///   dst Image width
/// 
/// @ingroup image_processing
//---------------------------------------------------------------------------
FASTCV_API void
fcvFilterCorrSep17x17s16_v2( const int16_t* __restrict kernel,
                             const int16_t* __restrict srcImg,
                             unsigned int              srcWidth, 
                             unsigned int              srcHeight, 
                             unsigned int              srcStride,
                             int16_t* __restrict       tmpImg,
                             int16_t* __restrict       dstImg, 
                             unsigned int              dstStride );

//------------------------------------------------------------------------------
/// @brief
///   Calculates the mean and variance of intensities of a rectangle in a
///   grayscale image.
///
/// @details
///
/// @param src
///   pointer to 8-bit grayscale image
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of source image
///   \n\b WARNING: must be multiple of 8.
///
/// @param xBegin
///   x coordinate of of top left of rectangle
///
/// @param yBegin
///   y coordinate of of top left of rectangle
///
/// @param recWidth
///   width of rectangular region
///
/// @param recHeight
///   height of rectangular region
///
/// @param mean
///   output of mean of region
///
/// @param variance
///   output of variance of region
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageIntensityStats( const uint8_t* __restrict src,
                        unsigned int              srcWidth,
                        int                       xBegin,
                        int                       yBegin,
                        unsigned int              recWidth,
                        unsigned int              recHeight,
                        float*                    mean,
                        float*                    variance );

//------------------------------------------------------------------------------
/// @brief
///   Creates a histogram of intensities for a rectangular region of a grayscale
///   image. Bins each pixel into a histogram of size 256, depending on the
///   intensity of the pixel (in the range 0 to 255).
///
/// @details
///
/// @param src
///   pointer to 8-bit grayscale image
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of source image
///   \n\b WARNING: must be multiple of 8.
///
/// @param xBegin
///   x coordinate of of top left of rectangle
///
/// @param yBegin
///   y coordinate of of top left of rectangle
///
/// @param recWidth
///   Width of rectangular region
///
/// @param recHeight
///   Height of rectangular region
///
/// @param histogram
///   Array of size 256 for storing the histogram
///   \n\b WARNING: must be 128-bit aligned.
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageIntensityHistogram( const uint8_t* __restrict src,
                            unsigned int              srcWidth,
                            int                       xBegin,
                            int                       yBegin,
                            unsigned int              recWidth,
                            unsigned int              recHeight,
                            int32_t*                  histogram  );


//------------------------------------------------------------------------------
/// @brief
///   Builds an integral image of the incoming 8-bit image and adds an
///   unfilled border on top and to the left.
///   \n NOTE: border usually zero filled elsewhere.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvIntegratePatchu8_v2(). In the 2.0.0 release, 
///   fcvIntegratePatchu8_v2 will be renamed to fcvIntegratePatchu8
///   and the signature of fcvIntegratePatchu8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///   sum (X,Y) = sum_{x<X,y<Y} I(x,y)
///
/// @param src
///   Input image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
///   \n\b NOTE: height must be <= 2048
///
/// @param dst
///   Output integral-image. Should point to a memory of size (width+1)*(height+1).
///   Zero borders for 1st column.
///   \n\b WARNING: must be 128-bit aligned.
///
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvIntegrateImageu8( const uint8_t* __restrict src,
                     unsigned int              srcWidth,
                     unsigned int              srcHeight,
                     uint32_t* __restrict      dst );


//------------------------------------------------------------------------------
/// @brief
///   Builds an integral image of the incoming 8-bit image and adds an
///   unfilled border on top and to the left.
///   \n NOTE: border usually zero filled elsewhere.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvIntegrateImageu8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvIntegrateImageu8,
///   \a fcvIntegrateImageu8_v2 will be removed, and the current signature
///   for \a fcvIntegrateImageu8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvIntegrateImageu8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   sum (X,Y) = sum_{x<X,y<Y} I(x,y)
///
/// @param src
///   Input image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
///   \n\b NOTE: height must be <= 2048
///
/// @param srcStride
///   Stride (in bytes) of the image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b WARNING: must be multiple of 8.
///
/// @param dst
///   Output integral-image. Should point to a memory of size at least (width+1)*(height+1).
///   Zero borders for 1st column.
///   \n\b WARNING: must be 128-bit aligned.
/// 
/// @param dstStride
///   Stride (in bytes) of integral image.
///   \n\b WARNING: must be multiple of 32 (8 * 4-byte values).
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvIntegrateImageu8_v2( const uint8_t* __restrict src,
                        unsigned int              srcWidth,
                        unsigned int              srcHeight,
                        unsigned int              srcStride,
                        uint32_t* __restrict      dst,
                        unsigned int              dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Builds an integral image of the incoming 8-bit patch values and their
///   squares and adds an unfilled border on top and to the left.
///   \n NOTE: border usually zero filled elsewhere.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvIntegratePatchu8_v2(). In the 2.0.0 release, 
///   fcvIntegratePatchu8_v2 will be renamed to fcvIntegratePatchu8
///   and the signature of fcvIntegratePatchu8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///   sum (X,Y) = sum_{x<X,y<Y} I(x,y)
///
/// @param src
///   Input image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
///   \n\b WARNING: height must be <= 2048
///
/// @param patchX
///   Patch location on image of upper-left patch corner.
///
/// @param patchY
///   Patch location on image of upper-left patch corner.
///
/// @param patchW
///   Patch width.
///
/// @param patchH
///   Patch height.
///
/// @param intgrlImgOut
///   Integral image.
///   Zero borders for 1st column.
///   \n\b NOTE: Memory must be > (patchW+1)(patchH+1)
///
/// @param intgrlSqrdImgOut
///   Integral image of squared values.
///   \n\b NOTE: Memory must be > (patchW+1)(patchH+1)
///
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvIntegratePatchu8( const uint8_t* __restrict src,
                     unsigned int              srcWidth,
                     unsigned int              srcHeight,
                     int                       patchX,
                     int                       patchY,
                     unsigned int              patchW,
                     unsigned int              patchH,
                     uint32_t* __restrict      intgrlImgOut,
                     uint32_t* __restrict      intgrlSqrdImgOut );


//------------------------------------------------------------------------------
/// @brief
///   Builds an integral image of the incoming 8-bit patch values and their
///   squares and adds an unfilled border on top and to the left.
///   \n NOTE: border usually zero filled elsewhere.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvIntegratePatchu8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvIntegratePatchu8,
///   \a fcvIntegratePatchu8_v2 will be removed, and the current signature
///   for \a fcvIntegratePatchu8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvIntegratePatchu8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   sum (X,Y) = sum_{x<X,y<Y} I(x,y)
///
/// @param src
///   Input image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
///   \n\b WARNING: height must be <= 2048
/// 
/// @param srcStride
///   Image stride (in bytes).
///   \n\b WARNING: must be multiple of 8 (8 * 1-byte values).
///
/// @param patchX
///   Patch location on image of upper-left patch corner.
///
/// @param patchY
///   Patch location on image of upper-left patch corner.
///
/// @param patchW
///   Patch width.
///   \n\b WARNING: (patchW * patchH) should be less than 66051, to avoid overflow.
///
/// @param patchH
///   Patch height.
///   \n\b WARNING: (patchW * patchH) should be less than 66051, to avoid overflow.
///
/// @param intgrlImgOut
///   Integral image.
///   Zero borders for 1st column.
///   \n\b NOTE: Memory must be > (patchW+1)(patchH+1)
///
/// @param intgrlSqrdImgOut
///   Integral image of squared values.
///   \n\b NOTE: Memory must be > (patchW+1)(patchH+1)
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvIntegratePatchu8_v2( const uint8_t* __restrict src,
                        unsigned int              srcWidth,
                        unsigned int              srcHeight,
                        unsigned int              srcStride,
                        int                       patchX,
                        int                       patchY,
                        unsigned int              patchW,
                        unsigned int              patchH,
                        uint32_t* __restrict      intgrlImgOut,
                        uint32_t* __restrict      intgrlSqrdImgOut );


//---------------------------------------------------------------------------
/// @brief
///   Builds an integral image of the incoming 12x12 8-bit patch values and
///   their squares.  It also adds an unfilled border on top and to the left.
///   \n NOTE: border usually zero filled elsewhere.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvIntegratePatch12x12u8_v2(). In the 2.0.0 release, 
///   fcvIntegratePatch12x12u8_v2 will be renamed to fcvIntegratePatch12x12u8
///   and the signature of fcvIntegratePatch12x12u8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///   sum (X,Y) = sum_{x<X,y<Y} I(x,y)
///
/// @param src
///   Input image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
///
/// @param patchX
///   Patch location on image of upper-left patch corner.
///
/// @param patchY
///   Patch location on image of upper-left patch corner.
///
/// @param intgrlImgOut
///   Integral image.
///   Zero borders for 1st column.
///   \n\b NOTE: Memory must be > (12+1)(12+1)
///
/// @param intgrlSqrdImgOut
///   Integral image of squared values.
///   \n\b NOTE: Memory must be > (12+1)(12+1)
///
/// @ingroup image_processing
//---------------------------------------------------------------------------

FASTCV_API void
fcvIntegratePatch12x12u8( const uint8_t* __restrict src,
                          unsigned int              srcWidth,
                          unsigned int              srcHeight,
                          int                       patchX,
                          int                       patchY,
                          uint32_t* __restrict      intgrlImgOut,
                          uint32_t* __restrict      intgrlSqrdImgOut );


//---------------------------------------------------------------------------
/// @brief
///   Builds an integral image of the incoming 12x12 8-bit patch values and
///   their squares.  It also adds an unfilled border on top and to the left.
///   \n NOTE: border usually zero filled elsewhere.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvIntegratePatch12x12u8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvIntegratePatch12x12u8,
///   \a fcvIntegratePatch12x12u8_v2 will be removed, and the current signature
///   for \a fcvIntegratePatch12x12u8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvIntegratePatch12x12u8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   sum (X,Y) = sum_{x<X,y<Y} I(x,y)
///   \n\b WARNING: do not use - under construction.
///
/// @param src
///   Input image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcStride
///   Image stride (in bytes).
///   \n\b WARNING: must be multiple of 8 (8 * 1-byte values).
///
/// @param patchX
///   Patch location on image of upper-left patch corner.
///
/// @param patchY
///   Patch location on image of upper-left patch corner.
///
/// @param intgrlImgOut
///   Integral image.
///   Zero borders for 1st column.
///   \n\b NOTE: Memory must be > (12+1)(12+1)
///
/// @param intgrlSqrdImgOut
///   Integral image of squared values.
///   \n\b NOTE: Memory must be > (12+1)(12+1)
///
/// @ingroup image_processing
//---------------------------------------------------------------------------

FASTCV_API void
fcvIntegratePatch12x12u8_v2( const uint8_t* __restrict src,
                             unsigned int              srcWidth,
                             unsigned int              srcHeight,
                             unsigned int              srcStride,
                             int                       patchX,
                             int                       patchY,
                             uint32_t* __restrict      intgrlImgOut,
                             uint32_t* __restrict      intgrlSqrdImgOut );


//------------------------------------------------------------------------------
/// @brief
///   Builds an integral image of the incoming 18x18 8-bit patch values and
///   their squares.  It also adds an unfilled border on top and to the left.
///   \n NOTE: border usually zero filled elsewhere.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvIntegratePatch18x18u8_v2(). In the 2.0.0 release, 
///   fcvIntegratePatch18x18u8_v2 will be renamed to fcvIntegratePatch18x18u8
///   and the signature of fcvIntegratePatch18x18u8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///   sum (X,Y) = sum_{x<X,y<Y} I(x,y)
///
/// @param src
///   Input image. Size of buffer is srcWidth*srcHeight bytes.
///
/// @param srcWidth
///   Image srcWidth.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
///
/// @param patchX
///   Patch location on image of upper-left patch corner.
///
/// @param patchY
///   Patch location on image of upper-left patch corner.
///
/// @param intgrlImgOut
///   Integral image.
///   Zero borders for 1st column.
///   \n\b NOTE: Memory must be > (18+1)(18+1)
///
/// @param intgrlSqrdImgOut
///   Integral image of squared values.
///   \n\b NOTE: Memory must be > (18+1)(18+1)
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvIntegratePatch18x18u8( const uint8_t* __restrict src,
                          unsigned int              srcWidth,
                          unsigned int              srcHeight,
                          int                       patchX,
                          int                       patchY,
                          uint32_t* __restrict      intgrlImgOut,
                          uint32_t* __restrict      intgrlSqrdImgOut );


//------------------------------------------------------------------------------
/// @brief
///   Builds an integral image of the incoming 18x18 8-bit patch values and
///   their squares.  It also adds an unfilled border on top and to the left.
///   \n NOTE: border usually zero filled elsewhere.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvIntegratePatch18x18u8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvIntegratePatch18x18u8,
///   \a fcvIntegratePatch18x18u8_v2 will be removed, and the current signature
///   for \a fcvIntegratePatch18x18u8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvIntegratePatch18x18u8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   sum (X,Y) = sum_{x<X,y<Y} I(x,y)
///
/// @param src
///   Input image. Size of buffer is srStride*srcHeight bytes.
///
/// @param srcWidth
///   Image srcWidth.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcStride
///   Image stride (in bytes).
///   \n\b WARNING: must be multiple of 8 (8 * 1-byte values).
///
/// @param patchX
///   Patch location on image of upper-left patch corner.
///
/// @param patchY
///   Patch location on image of upper-left patch corner.
///
/// @param intgrlImgOut
///   Integral image.
///   Zero borders for 1st column.
///   \n\b NOTE: Memory must be > (18+1)(18+1)
///
/// @param intgrlSqrdImgOut
///   Integral image of squared values.
///   \n\b NOTE: Memory must be > (18+1)(18+1)
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvIntegratePatch18x18u8_v2( const uint8_t* __restrict src,
                             unsigned int              srcWidth,
                             unsigned int              srcHeight,
                             unsigned int              srcStride,
                             int                       patchX,
                             int                       patchY,
                             uint32_t* __restrict      intgrlImgOut,
                             uint32_t* __restrict      intgrlSqrdImgOut );


//---------------------------------------------------------------------------
/// @brief
///   Integrates one line of an image or any portion of an image that is
///   contiguous in memory.
///
/// @param src
///   Input image. Size of buffer is srcWidth bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Number of pixels.
///   \n NOTE: bit width enforces numPxls < 2^16
///
/// @param intgrl
///   Sum of values from specified pixels.
///
/// @param intgrlSqrd
///   Sum of squared values from specified pixels.
///
/// @ingroup image_processing
//---------------------------------------------------------------------------

FASTCV_API void
fcvIntegrateImageLineu8( const uint8_t* __restrict src,
                         uint16_t                  srcWidth,
                         uint32_t*                 intgrl,
                         uint32_t*                 intgrlSqrd );


//------------------------------------------------------------------------------
/// @brief
///   Integrates 64 contiguous pixels of an image.
///
/// @param src
///   Input image.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param intgrl
///   Sum of values from specified pixels.
///
/// @param intgrlSqrd
///   Sum of squared values from specified pixels.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvIntegrateImageLine64u8( const uint8_t* __restrict src,
                           uint16_t*                 intgrl,
                           uint32_t*                 intgrlSqrd );


//------------------------------------------------------------------------------
/// @brief
///   compute approximate mean and variance for the range of NFT4 float
///   descriptors where descriptor elements along dimension are treated
///   as random vars
///
/// @param src
///   contiguous block of descriptors of dimension 36
///
/// @param first
///   index of the first descriptor in range array vind for computing mean and var
///
/// @param last
///   index of the last descriptor in range array vind for computing mean and range
///
/// @param vind
///   array of randomized indexes of descriptors
///
/// @param means
///   buffer for approximate means, must be 36 long
///
/// @param vars
///   buffer for approximate variances, must be 36 long
///
/// @param temp
///   bufffer, must be 46 long
///
/// @return
///   0        - success
///   EFAULT   - invalid address
///   EINVAL   - invalid argument
///
/// @remark
///   If descriptor range is > 100 then only
///   100 samples are drawn from the range to compute
///   approximate means and variances.
///
///   Variances computed here do not have to be true variances because their
///   values do not matter in kdtrees. The only thing that matters is that
///   the ordering relation of variances is preserved
///
/// 
///
/// @ingroup object_detection
// -----------------------------------------------------------------------------

FASTCV_API int
fcvDescriptorSampledMeanAndVar36f32( const float* __restrict src,
                                     int                     first,
                                     int                     last,
                                     int32_t*                vind,
                                     float* __restrict       means,
                                     float* __restrict       vars,
                                     float* __restrict       temp );


//------------------------------------------------------------------------------
/// @brief
///   Searches a 8x8 patch within radius around a center pixel for the max NCC.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvNCCPatchOnCircle8x8u8_v2(). In the 2.0.0 release, 
///   fcvNCCPatchOnCircle8x8u8_v2 will be renamed to fcvNCCPatchOnCircle8x8u8
///   and the signature of fcvNCCPatchOnCircle8x8u8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param patch
///   Pointer to 8-bit patch pixel values linearly laid out in memory.
///
/// @param src
///   Pointer to 8-bit image pixel values linearly laid out in memory.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Width in pixels of the image.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Height in pixels of the image.
///
/// @param search_center_x
///   X location of search center in pixels of the image.
///
/// @param search_center_y
///   Y location of search center in pixels of the image.
///
/// @param search_radius
///   Radius of search in pixels. Must be <=5.
///
/// @param best_x
///   Center X location on the image of the best NCC match.  The center X has
///   4 pixels to the left and 3 to the right.
///
/// @param best_y
///   Center Y location on the image of the best NCC match.  The center Y has
///   4 pixels above and 3 pixels below.
///
/// @param bestNCC
///   Largest value of the normalized cross-correlation found in the NCC search.
///   It's quantized to integer value in Q7 (between -128 and 128).
///
/// @param findSubPixel (0 or 1)
///   Use parabolic interpolation of NCC values to find sub-pixel estimates.
///
/// @param subX
///   Sub-pixel estimate for optimal NCC relative to best_x.
///   \n e.g., float x = (float)best_x + subX;
///
/// @param subY
///   Sub-pixel estimate for optimal NCC relative to best_y.
///
/// @return
///   0 = OK \n
///   1 = "search_radius" too large\n
///   2 = invalid "search_center_x,y"\n
///   3 = not found\n
///
/// @ingroup object_detection
//------------------------------------------------------------------------------

FASTCV_API int
fcvNCCPatchOnCircle8x8u8( const uint8_t* __restrict patch,
                          const uint8_t* __restrict src,
                          unsigned short            srcWidth,
                          unsigned short            srcHeight,
                          unsigned short            search_center_x,
                          unsigned short            search_center_y,
                          unsigned short            search_radius,
                          uint16_t*                 best_x,
                          uint16_t*                 best_y,
                          uint32_t*                 bestNCC,
                          int                       findSubPixel,
                          float*                    subX,
                          float*                    subY );


//------------------------------------------------------------------------------
/// @brief
///   Searches a 8x8 patch within radius around a center pixel for the max NCC.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvNCCPatchOnCircle8x8u8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvNCCPatchOnCircle8x8u8,
///   \a fcvNCCPatchOnCircle8x8u8_v2 will be removed, and the current signature
///   for \a fcvNCCPatchOnCircle8x8u8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvNCCPatchOnCircle8x8u8 when transitioning to 2.0.0.
///   \n\n
///
/// @param patch
///   Pointer to 8-bit patch pixel values linearly laid out in memory.
///
/// @param src
///   Pointer to 8-bit image pixel values linearly laid out in memory.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Width in pixels of the image.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Height in pixels of the image.
///
/// @param search_center_x
///   X location of search center in pixels of the image.
///
/// @param search_center_y
///   Y location of search center in pixels of the image.
///
/// @param search_radius
///   Radius of search in pixels. Must be <=5.
///
/// @param filterLowVariance
///   Minimum variance. Used to as threshold to compare against variance of
///   8x8 block of src or patch.    
///
/// @param best_x
///   Center X location on the image of the best NCC match.  The center X has
///   4 pixels to the left and 3 to the right.
///
/// @param best_y
///   Center Y location on the image of the best NCC match.  The center Y has
///   4 pixels above and 3 pixels below.
///
/// @param bestNCC
///   Largest value of the normalized cross-correlation found in the NCC search.
///   It's quantized to integer value in Q7 (between -128 and 128).
///
/// @param findSubPixel (0 or 1)
///   Use parabolic interpolation of NCC values to find sub-pixel estimates.
///
/// @param subX
///   Sub-pixel estimate for optimal NCC relative to best_x.
///   \n e.g., float x = (float)best_x + subX;
///
/// @param subY
///   Sub-pixel estimate for optimal NCC relative to best_y.
///
/// @return
///   0 = OK \n
///   1 = "search_radius" too large\n
///   2 = invalid "search_center_x,y"\n
///   3 = not found\n
///   4 = Patch has too low variance\n
///   5 = Image region has too low variance\n
///
/// @ingroup object_detection
//------------------------------------------------------------------------------

FASTCV_API int
fcvNCCPatchOnCircle8x8u8_v2( const uint8_t* __restrict patch,
                             const uint8_t* __restrict src,
                             unsigned short            srcWidth,
                             unsigned short            srcHeight,
                             unsigned short            search_center_x,
                             unsigned short            search_center_y,
                             unsigned short            search_radius,
                             int                       filterLowVariance,
                             uint16_t*                 best_x,
                             uint16_t*                 best_y,
                             uint32_t*                 bestNCC,
                             int                       findSubPixel,
                             float*                    subX,
                             float*                    subY );




//------------------------------------------------------------------------------
/// @brief
///   Searches a 8x8 patch within square region around a center pixel
///   for the max NCC.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvNCCPatchOnSquare8x8u8_v2(). In the 2.0.0 release, 
///   fcvNCCPatchOnSquare8x8u8_v2 will be renamed to fcvNCCPatchOnSquare8x8u8
///   and the signature of fcvNCCPatchOnSquare8x8u8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param patch
///   Pointer to 8-bit patch pixel values linearly laid out in memory.
///
/// @param src
///   Pointer to 8-bit image pixel values linearly laid out in memory.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Width in pixels of the image.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Height in pixels of the image.
///
/// @param search_center_x
///   Center X coordinate of the search window
///
/// @param search_center_y
///   Center Y coordinate of the search window
///
/// @param search_w
///   Width of search square in pixels
///   \n\b WARNING: must be 11 or less.
///
/// @param best_x
///   Center X location on the image of the best NCC match.  The center X has
///   4 pixels to the left and 3 to the right.
///
/// @param best_y
///   Center Y location on the image of the best NCC match.  The center Y has
///   4 pixels above and 3 pixels below.
/// 
/// @param bestNCC
///   NCC value of the best match block.
///   It's quantized to integer value in Q7 (between -128 and 128).
///
/// @param doSubPixel (0 or 1)
///   Use parabolic interpolation of NCC values to find sub-pixel estimates.
///
/// @param subX
///   Sub-pixel estimate for optimal NCC relative to best_x.
///   \n e.g., float x = (float)best_x + subX;
///
/// @param subY
///   Sub-pixel estimate for optimal NCC relative to best_y.
///
/// @return
///   0 = OK \n
///   1 = "search_radius" too large\n
///   2 = invalid "search_center_x,y"\n
///   3 = not found\n
///
/// @ingroup object_detection
//------------------------------------------------------------------------------

FASTCV_API int
fcvNCCPatchOnSquare8x8u8( const uint8_t* __restrict patch,
                          const uint8_t* __restrict src,
                          unsigned short            srcWidth,
                          unsigned short            srcHeight,
                          unsigned short            search_center_x,
                          unsigned short            search_center_y,
                          unsigned short            search_w,
                          uint16_t*                 best_x,
                          uint16_t*                 best_y,
                          uint32_t*                 bestNCC,
                          int                       doSubPixel,
                          float*                    subX,
                          float*                    subY );


//------------------------------------------------------------------------------
/// @brief
///   Searches a 8x8 patch within square region around a center pixel
///   for the max NCC.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvNCCPatchOnSquare8x8u8 with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvNCCPatchOnSquare8x8u8,
///   \a fcvNCCPatchOnSquare8x8u8_v2 will be removed, and the current signature
///   for \a fcvNCCPatchOnSquare8x8u8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvNCCPatchOnSquare8x8u8 when transitioning to 2.0.0.
///   \n\n
/// 
/// @param patch
///   Pointer to 8-bit patch pixel values linearly laid out in memory.
///
/// @param src
///   Pointer to 8-bit image pixel values linearly laid out in memory.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Width in pixels of the image.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Height in pixels of the image.
///
/// @param search_center_x
///   Center X coordinate of the search window
///
/// @param search_center_y
///   Center Y coordinate of the search window
///
/// @param search_w
///   Width of search square in pixels
///   \n\b WARNING: must be 11 or less.
///
/// @param filterLowVariance
///   Minimum variance. Used to as threshold to compare against variance of
///   8x8 block of src or patch.    
///
/// @param best_x
///   Center X location on the image of the best NCC match.  The center X has
///   4 pixels to the left and 3 to the right.
///
/// @param best_y
///   Center Y location on the image of the best NCC match.  The center Y has
///   4 pixels above and 3 pixels below.
/// 
/// @param bestNCC
///   NCC value of the best match block.
///   It's quantized to integer value in Q7 (between -128 and 128).
///
/// @param doSubPixel (0 or 1)
///   Use parabolic interpolation of NCC values to find sub-pixel estimates.
///
/// @param subX
///   Sub-pixel estimate for optimal NCC relative to best_x.
///   \n e.g., float x = (float)best_x + subX;
///
/// @param subY
///   Sub-pixel estimate for optimal NCC relative to best_y.
///
/// @return
///   0 = OK \n
///   1 = "search_radius" too large\n
///   2 = invalid "search_center_x,y"\n
///   3 = not found\n
///   4 = Patch has too low variance\n
///   5 = Image region has too low variance\n
///
/// @ingroup object_detection
//------------------------------------------------------------------------------

FASTCV_API int
fcvNCCPatchOnSquare8x8u8_v2( const uint8_t* __restrict patch,
                             const uint8_t* __restrict src,
                             unsigned short            srcWidth,
                             unsigned short            srcHeight,
                             unsigned short            search_center_x,
                             unsigned short            search_center_y,
                             unsigned short            search_w,
                             int                       filterLowVariance,
                             uint16_t*                 best_x,
                             uint16_t*                 best_y,
                             uint32_t*                 bestNCC,
                             int                       doSubPixel,
                             float*                    subX,
                             float*                    subY );



//------------------------------------------------------------------------------
/// @brief
///   Sum of absolute differences of an image against an 8x8 template.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvSumOfAbsoluteDiffs8x8u8_v2(). In the 2.0.0 release, 
///   fcvSumOfAbsoluteDiffs8x8u8_v2 will be renamed to fcvSumOfAbsoluteDiffs8x8u8
///   and the signature of fcvSumOfAbsoluteDiffs8x8u8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///   8x8 sum of ||A-B||. The template patch is swept over the entire image and
///   the results are put in dst.
///
/// @param patch
///   8x8 template
///
/// @param src
///   Reference Image.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///    Width of the src image.
///
/// @param srcHeight
///    Height of the src image.
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///
/// @param dst
///   The dst buffer shall be width X height bytes in length.
///   Output of SAD(A,B). dst[4][4] correspondes to the 0,0 pixel of the template
///   aligned with the 0,0 pixel of src. The dst border values not covered by
///   entire 8x8 patch window will remain unmodified by the function. The caller
///   should either initialize these to 0 or ignore.
///   \n\b NOTE: must be 128-bit aligned.
///
/// 
///
/// @ingroup object_detection
//------------------------------------------------------------------------------

FASTCV_API void
fcvSumOfAbsoluteDiffs8x8u8( const uint8_t* __restrict patch,
                            const uint8_t* __restrict src,
                            unsigned int              srcWidth,
                            unsigned int              srcHeight,
                            unsigned int              srcStride,
                            uint16_t* __restrict      dst );


//------------------------------------------------------------------------------
/// @brief
///   Sum of absolute differences of an image against an 8x8 template.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvSumOfAbsoluteDiffs8x8u8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvSumOfAbsoluteDiffs8x8u8,
///   \a fcvSumOfAbsoluteDiffs8x8u8_v2 will be removed, and the current signature
///   for \a fcvSumOfAbsoluteDiffs8x8u8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvSumOfAbsoluteDiffs8x8u8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   8x8 sum of ||A-B||. The template patch is swept over the entire image and
///   the results are put in dst.
///
/// @param patch
///   8x8 template
/// 
/// @param patchStride
///   Stride of the 8x8 template buffer
/// 
/// @param dstStride
///   Stride of the patch (in bytes) - i.e., how many bytes between column 0 of row N 
///   and column 0 of row N+1.
///
/// @param src
///   Reference Image.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///    Width of the src image.
///
/// @param srcHeight
///    Height of the src image.
///
/// @param srcStride
///   Stride of image (in bytes) - i.e., how many bytes between column 0 of row N 
///   and column 0 of row N+1.
///
/// @param dst
///   The dst buffer shall be at least ( width x height ) values in length.
///   Output of SAD(A,B). dst[4][4]correspondes to the 0,0 pixel of the template
///   aligned with the 0,0 pixel of src. The dst border values not covered by
///   entire 8x8 patch window will remain unmodified by the function. The caller
///   should either initialize these to 0 or ignore.
///   \n\b NOTE: must be 128-bit aligned.
/// 
/// @param dstStride
///   Stride of destination (in bytes) - i.e., how many bytes between column 0 of row N 
///   and column 0 of row N+1.
///
/// 
///
/// @ingroup object_detection
//------------------------------------------------------------------------------

FASTCV_API void
fcvSumOfAbsoluteDiffs8x8u8_v2( const uint8_t* __restrict patch,
                               unsigned int              patchStride,
                               const uint8_t* __restrict src,
                               unsigned int              srcWidth,
                               unsigned int              srcHeight,
                               unsigned int              srcStride,
                               uint16_t* __restrict      dst,
                               unsigned int              dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Down-scale the image to half width and height by averaging 2x2 pixels
///   into one.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvScaleDownBy2u8_v2(). In the 2.0.0 release, 
///   fcvScaleDownBy2u8_v2 will be renamed to fcvScaleDownBy2u8
///   and the signature of fcvScaleDownBy2u8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///   A box filter downsampling the next pixel, the pixel below, and the next
///   pixel to the pixel below into one pixel.\n
///   | px00 px01 px02 px03 |\n
///   | px10 px11 px12 px13 |\n
///   to:\n
///   | (px00+px01+px10+px11)/4 (px02+px03+px12+px13)/4 |\n
///
/// @param src
///   Input 8-bit image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b NOTE: data must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
///   \n\b NOTE:must be a multiple of 2
///
/// @param dst
///   Output 8-bit image. Size of buffer is srcWidth*srcHeight/4 bytes.
///   \n\b NOTE: data must be 128-bit aligned.
///
/// 
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API int
fcvScaleDownBy2u8( const uint8_t* __restrict src,
                   unsigned int              srcWidth,
                   unsigned int              srcHeight,
                   uint8_t* __restrict       dst );


//------------------------------------------------------------------------------
/// @brief
///   Down-scale the image to half width and height by averaging 2x2 pixels
///   into one.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvScaleDownBy2u8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvScaleDownBy2u8,
///   \a fcvScaleDownBy2u8_v2 will be removed, and the current signature
///   for \a fcvScaleDownBy2u8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvScaleDownBy2u8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   A box filter downsampling the next pixel, the pixel below, and the next
///   pixel to the pixel below into one pixel.\n
///   | px00 px01 px02 px03 |\n
///   | px10 px11 px12 px13 |\n
///   to:\n
///   | (px00+px01+px10+px11)/4 (px02+px03+px12+px13)/4 |\n
///
/// @param src
///   Input 8-bit image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
///   \n\b NOTE:must be a multiple of 2
/// 
/// @param srcStride
///   Image stride (in bytes).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8 (8 * 1-byte values), and at least as much as srcWidth if not 0.
///
/// @param dst
///   Output 8-bit image. Size of buffer is dstStride*srcHeight/2 bytes.
///   \n\b NOTE: data must be 128-bit aligned.
/// 
/// @param dstStride
///   Output stride (in bytes).
///   \n\b NOTE: if 0, dstStride is set as srcWidth/2.
///   \n\b WARNING: must be multiple of 8 (8 * 1-byte values), and at least as much as srcWidth/2 if not 0.
///
/// 
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API int
fcvScaleDownBy2u8_v2( const uint8_t* __restrict src,
                      unsigned int              srcWidth,
                      unsigned int              srcHeight,
                      unsigned int              srcStride,
                      uint8_t* __restrict       dst,
                      unsigned int              dstStride );

//------------------------------------------------------------------------------
/// @brief
///   Downscale a grayscale image by a factor of two using a 5x5 Gaussian filter kernel
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvScaleDownBy2Gaussian5x5u8_v2(). In the 2.0.0 release, 
///   fcvScaleDownBy2Gaussian5x5u8_v2 will be renamed to fcvScaleDownBy2Gaussian5x5u8
///   and the signature of fcvScaleDownBy2Gaussian5x5u8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///   Downsamples the image using a 5x5 Gaussian filter kernel.
///
/// @param src
///   Input 8-bit image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b NOTE: must be multiple of 8
///
/// @param srcHeight
///   Image height.
///   \n\b NOTE:must be a multiple of 2
///
/// @param dst
///   Output 8-bit downscale image of size (width / 2) x (height / 2).
///   \n\b NOTE: border values have been taken cared w.r.t. the pixel coordinate.
///
/// 
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API void
fcvScaleDownBy2Gaussian5x5u8( const uint8_t* __restrict src,
                              unsigned int              srcWidth,
                              unsigned int              srcHeight,
                              uint8_t* __restrict       dst );


//------------------------------------------------------------------------------
/// @brief
///   Downscale a grayscale image by a factor of two using a 5x5 Gaussian filter kernel
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvScaleDownBy2Gaussian5x5u8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvScaleDownBy2Gaussian5x5u8,
///   \a fcvScaleDownBy2Gaussian5x5u8_v2 will be removed, and the current signature
///   for \a fcvScaleDownBy2Gaussian5x5u8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvScaleDownBy2Gaussian5x5u8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   Downsamples the image using a 5x5 Gaussian filter kernel.
///
/// @param src
///   Input 8-bit image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b NOTE: must be multiple of 8
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcStride
///   Image stride (in bytes).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8 (8 * 1-byte values), and at least as much as srcWidth if not 0.
///
/// @param dst
///   Output 8-bit downscale image of size (width / 2) x (height / 2).
///   \n\b NOTE: border values have been taken cared w.r.t. the pixel coordinate.
/// 
/// @param dstStride
///   Output stride (in bytes).
///   \n\b NOTE: if 0, dstStride is set as srcWidth/2.
///   \n\b WARNING: must be multiple of 8 (8 * 1-byte values), and at least as much as srcWidth/2 if not 0.
///
/// 
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API void
fcvScaleDownBy2Gaussian5x5u8_v2( const uint8_t* __restrict src,
                                 unsigned int              srcWidth,
                                 unsigned int              srcHeight,
                                 unsigned int              srcStride,
                                 uint8_t* __restrict       dst,
                                 unsigned int              dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Downscale the image to quarter width and height by averaging 4x4 pixels
///   into one..
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvScaleDownBy4u8_v2(). In the 2.0.0 release, 
///   fcvScaleDownBy4u8_v2 will be renamed to fcvScaleDownBy4u8
///   and the signature of fcvScaleDownBy4u8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///   A 4x4 downsampling box filter across adjacent pixels is applied.
///
/// @param src
///   Input 8-bit image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b NOTE: must be multiple of 8
///
/// @param srcHeight
///   Image height.
///   \n\b NOTE:must be a multiple of 4
///
/// @param dst
///   Output 8-bit image. Size of buffer is srcWidth*srcHeight/16 bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// 
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API int
fcvScaleDownBy4u8( const uint8_t* __restrict src,
                    unsigned int             srcWidth,
                    unsigned int             srcHeight,
                    uint8_t* __restrict      dst );


//------------------------------------------------------------------------------
/// @brief
///   Downscale the image to quarter width and height by averaging 4x4 pixels
///   into one..
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvScaleDownBy4u8_v2() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvScaleDownBy4u8_v2,
///   \a fcvScaleDownBy4u8_v2 will be removed, and the current signature
///   for \a fcvScaleDownBy4u8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvScaleDownBy4u8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   A 4x4 downsampling box filter across adjacent pixels is applied.
///
/// @param src
///   Input 8-bit image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b NOTE: must be multiple of 8
///
/// @param srcHeight
///   Image height.
///   \n\b NOTE:must be a multiple of 4
/// 
/// @param srcStride
///   Image stride (in bytes).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8 (8 * 1-byte values), and at least as much as srcWidth if not 0.
///
/// @param dst
///   Output 8-bit image. Size of buffer is dstStride*srcHeight/4 bytes.
///   \n\b WARNING: must be 128-bit aligned.
/// 
/// @param dstStride
///   Output stride (in bytes).
///   \n\b NOTE: if 0, dstStride is set as srcWidth/4.
///   \n\b WARNING: must be multiple of 8 (8 * 1-byte values), and at least as much as srcWidth/4 if not 0.
///
/// 
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API int
fcvScaleDownBy4u8_v2( const uint8_t* __restrict src,
                      unsigned int              srcWidth,
                      unsigned int              srcHeight,
                      unsigned int              srcStride,
                      uint8_t* __restrict       dst,
                      unsigned int              dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Downscale the image to 2/3 width and height by averaging 3x3 pixels
///   into one..
///
/// @details
///   A 3x3 downsampling box filter across adjacent pixels is applied.
///
/// @param src
///   Input 8-bit image.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b NOTE: In case of non multiple of 3, it will crop to the closest multiple of 3
///
/// @param srcHeight
///   Image height.
///   \n\b NOTE: In case of non multiple of 3, it will crop to the closest multiple of 3
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2). If 0 is passed, srcStride is set to srcWidth.
/// 
/// @param dst
///   Output 8-bit image.
///   \n\b WARNING: must be 128-bit aligned. 
///   Memory must be pre-allocated at least srcWidth * srcHeight * 2 / 3
///   dstWidth  = srcWidth/3*2
///   dstHeight = srcHeight/3*2
/// 
/// @param dstStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2). If 0 is passed, dstStride is set to dstWidth which is srcWidth *2/3.
/// 
/// @return 0 if successful
/// 
/// 
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API int
fcvScaleDown3To2u8( const uint8_t* __restrict src,
                    unsigned                  srcWidth,
                    unsigned                  srcHeight,
                    unsigned int              srcStride,
                    uint8_t* __restrict       dst,
                    unsigned int              dstStride);

//---------------------------------------------------------------------------
/// @brief
///   Downsample Horizontaly and/or Vertically by an *integer* scale.
///
/// @details
///    Uses Nearest Neighbor method
///
/// @param src
///   Input 8-bit image.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Source Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Source Image height.
/// 
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2). If 0 is passed, srcStride is set to srcWidth.
/// 
/// @param dst
///   Output 8-bit image.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param dstWidth
///   Destination Image width.
///
/// @param dstHeight
///   Destination Image height.
/// 
/// @param dstStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2). If 0 is passed, dstStride is set to dstWidth which is srcWidth *2/3.
/// 
/// @return 0 if successful
///
/// 
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API int
fcvScaleDownNNu8( const uint8_t* __restrict src,
                  unsigned int              srcWidth,
                  unsigned int              srcHeight,
                  unsigned int              srcStride,
                  uint8_t* __restrict       dst,
                  unsigned int              dstWidth,
                  unsigned int              dstHeight,
                  unsigned int              dstStride );

//---------------------------------------------------------------------------
/// @brief
///   Downsample Horizontaly and/or Vertically by an *integer* scale.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvScaleDownu8_v2(). In the 2.0.0 release, 
///   fcvScaleDownu8_v2 will be renamed to fcvScaleDownu8
///   and the signature of fcvScaleDownu8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///    Uses an box averaging filter of size MxN where M is the scale factor
///    in horizontal dimension and N is the scale factor in the vertical
///    dimension.
///    \n \b NOTE: input dimensions should be multiple of output dimensions.
///    \n NOTE: On different processors, some output pixel values may be off by 1
///
/// @param src
///   Input 8-bit image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Source Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Source Image height.
///
/// @param dst
///   Output 8-bit image. Size of buffer is dstWidth*dstHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param dstWidth
///   Destination Image width.
///
/// @param dstHeight
///   Destination Image height.
///
/// 
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API void
fcvScaleDownu8( const uint8_t* __restrict src,
                unsigned int              srcWidth,
                unsigned int              srcHeight,
                uint8_t* __restrict       dst,
                unsigned int              dstWidth,
                unsigned int              dstHeight );


//---------------------------------------------------------------------------
/// @brief
///   Downsample Horizontaly and/or Vertically by an *integer* scale.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvScaleDownu8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvScaleDownu8,
///   \a fcvScaleDownu8_v2 will be removed, and the current signature
///   for \a fcvScaleDownu8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvScaleDownu8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///    Uses an box averaging filter of size MxN where M is the scale factor
///    in horizontal dimension and N is the scale factor in the vertical
///    dimension
///    \n \b NOTE: input dimensions should be multiple of output dimensions.
///    \n NOTE: On different processors, some output pixel values may be off by 1
///
/// @param src
///   Input 8-bit image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Source Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Source Image height.
/// 
/// @param srcStride
///   Image stride (in bytes).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8 (8 * 1-byte values), and at least as much as srcWidth if not 0.
///
/// @param dst
///   Output 8-bit image. Size of buffer is dstStride*dstHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param dstWidth
///   Destination Image width.
///
/// @param dstHeight
///   Destination Image height.
/// 
/// @param dstStride
///   Output stride (in bytes).
///   \n\b NOTE: if 0, dstStride is set as dstWidth.
///   \n\b WARNING: must be multiple of 8 (8 * 1-byte values), and at least as much as dstWidth if not 0.
///
/// 
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API void
fcvScaleDownu8_v2( const uint8_t* __restrict src,
                   unsigned int              srcWidth,
                   unsigned int              srcHeight,
                   unsigned int              srcStride,
                   uint8_t* __restrict       dst,
                   unsigned int              dstWidth,
                   unsigned int              dstHeight,
                   unsigned int              dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Upscale a grayscale image by a factor of two using a 5x5 Gaussian filter kernel
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvScaleUpBy2Gaussian5x5u8_v2(). In the 2.0.0 release, 
///   fcvScaleUpBy2Gaussian5x5u8_v2 will be renamed to fcvScaleUpBy2Gaussian5x5u8
///   and the signature of fcvScaleUpBy2Gaussian5x5u8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///   Upsamples the image using a 5x5 Gaussian filter kernel.
///   /n/b NOTE: border values have been taken care with Gaussion coefficients.
///
/// @param src
///   Input 8-bit image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
///
/// @param dst
///   Output 8-bit upsampled image of size (2*width) x (2*height).
///   \n\b WARNING: must be 128-bit aligned.
///
/// 
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API void
fcvScaleUpBy2Gaussian5x5u8( const uint8_t* __restrict src,
                            unsigned int              srcWidth,
                            unsigned int              srcHeight,
                            uint8_t* __restrict       dst );


//------------------------------------------------------------------------------
/// @brief
///   Upscale a grayscale image by a factor of two using a 5x5 Gaussian filter kernel
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvScaleUpBy2Gaussian5x5u8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvScaleUpBy2Gaussian5x5u8,
///   \a fcvScaleUpBy2Gaussian5x5u8_v2 will be removed, and the current signature
///   for \a fcvScaleUpBy2Gaussian5x5u8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvScaleUpBy2Gaussian5x5u8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   Upsamples the image using a 5x5 Gaussian filter kernel.
///   /n/b NOTE: border values have been taken care with Gaussion coefficients.
///
/// @param src
///   Input 8-bit image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
/// 
/// @param srcStride
///   Image stride (in bytes).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8 (8 * 1-byte values), and at least as much as srcWidth if not 0.
///
/// @param srcHeight
///   Image height.
///
/// @param dst
///   Output 8-bit upsampled image of size (2*dstStride) x (2*srcHeight).
///   \n\b WARNING: must be 128-bit aligned.
/// 
/// @param dstStride
///   Output stride (in bytes).
///   \n\b NOTE: if 0, dstStride is set as srcWidth*2.
///   \n\b WARNING: must be multiple of 8 (8 * 1-byte values), and at least as much as srcWidth*2 if not 0.
///
/// 
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API void
fcvScaleUpBy2Gaussian5x5u8_v2( const uint8_t* __restrict src,
                               unsigned int              srcWidth,
                               unsigned int              srcHeight,
                               unsigned int              srcStride,
                               uint8_t* __restrict       dst,
                               unsigned int              dstStride );


// -----------------------------------------------------------------------------
/// @brief
///   Translate to float and normalize 36 8-bit elements
///
/// @param src
///   Pointer to the first input vector
///
/// @param invLen
///   Pointer to inverse length of the first input vector
///   located right after each 36 element vector
///
/// @param numVecs
///   Number of vectors to translate
///
/// @param reqNorm
///   Required norm
///
/// @param srcStride
///   Step in bytes to data of the next vector
///   Each vector has 36 8-bit elements and 1 float invLen
///
/// @param dst
///   Pointer to contiguous block for output vectors
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param stopBuild
///   Allows other threads to break this function in the middle of processing.
///   When set to 1, the function will exit on the next iteration.
///
/// @return
///   0        - success
///   EFAULT   - invalid address
///   EINVAL   - invalid argument
///
/// @ingroup math_vector
// -----------------------------------------------------------------------------

FASTCV_API int
fcvVecNormalize36s8f32( const int8_t* __restrict src,
                        unsigned int             srcStride,
                        const float*  __restrict invLen,
                        unsigned int             numVecs,
                        float                    reqNorm,
                        float*        __restrict dst,
                        int32_t*                 stopBuild );


//---------------------------------------------------------------------------
/// @brief
///   Sum of squared differences of one 36-byte vector against 4 others.
///
/// @details
///   SSD of one vector (a) against 4 others (b0,b1,b2,b3) using their given
///   inverse lengths for normalization.
///   \n\n SSD(a,b0), SSD(a,b1), SSD(a,b2), SSD(a,b3)
///
/// @param a
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param invLenA
///   Inverse of vector A = 1/|A|
///
/// @param b0
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param b1
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param b2
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param b3
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param invLenB
///   Inverse of vectors b0...b3 = 1/|b0|,... 1/|b3|
///   \n\b WARNING: array must be 128-bit aligned
///
/// @param distances
///   Output of the 4 results { SSD(a,b0), SSD(a,b1), SSD(a,b2), SSD(a,b3) }.
///   \n ACCURACY: 1.0e-6
///   \n\b WARNING: array must be 128-bit aligned
///
/// 
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API void
fcvSumOfSquaredDiffs36x4s8( const int8_t* __restrict a,
                            float                    invLenA,
                            const int8_t* __restrict b0,
                            const int8_t* __restrict b1,
                            const int8_t* __restrict b2,
                            const int8_t* __restrict b3,
                            const float* __restrict  invLenB,
                            float* __restrict        distances );


//---------------------------------------------------------------------------
/// @brief
///   Sum of squared differences of one 36-byte vector against N others.
///
/// @details
///   SSD of one vector (a) against N other 36-byte vectors
///   ( b[0], b[1], ..., b[n-1] )
///   using their given inverse lengths for normalization.
///   \n\n SSD(a,b[0]), SSD(a,b[1]), ..., SSD(a,b[n-1])
///
/// @param a
///   Vector.
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param invLenA
///   Inverse of vector A = 1/|A|
///
/// @param b
///   Vectors b[0]...b[n-1].
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param invLenB
///   Inverse of vectors b[0]...b[n-1]  = 1/|b[0]|,... 1/|b[n-1]|
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param numB
///   Number of B vectors.
///
/// @param distances
///   Output of the N results { SSD(a,b[0]), SSD(a,b[1]), ..., SSD(a,b[n-1]) }.
///   \n ACCURACY: 1.0e-6
///   \n\b WARNING: must be 128-bit aligned.
///
/// 
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API void
fcvSumOfSquaredDiffs36xNs8( const int8_t* __restrict         a,
                            float                            invLenA,
                            const int8_t* const * __restrict b,
                            const float* __restrict          invLenB,
                            unsigned int                     numB,
                            float* __restrict                distances );


//---------------------------------------------------------------------------
/// @brief
///   Sorting of 8 float numbers
///
/// @details
///   Perform sorting of 8 scores in ascending order (output of SumOfSquaredDiffs)
///
/// @param inScores
///   Input 8 element float array
///   \n\b NOTE: array should be 128-bit aligned
///
/// @param outScores
///   Output is 8 element sorted float array
///   \n\b WARNING: array must be 128-bit aligned
///
/// 
///
///   @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API void
fcvSort8Scoresf32( float* __restrict inScores, float* __restrict outScores );

//------------------------------------------------------------------------------
/// @brief
///   Binarizes a grayscale image based on a threshold value.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvFilterThresholdu8_v2(). In the 2.0.0 release, 
///   fcvFilterThresholdu8_v2 will be renamed to fcvFilterThresholdu8
///   and the signature of fcvFilterThresholdu8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///   Sets the pixel to max(255) if it's value is greater than the threshold;
///   else, set the pixel to min(0).
///
/// @param src
///   Input 8-bit image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8
///
/// @param srcHeight
///   Image height.
///
/// @param dst
///   Output 8-bit binarized image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param threshold
///   Threshold value for binarization.
///   \n\b WARNING: must be larger than 0.
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterThresholdu8( const uint8_t* __restrict src,
                      unsigned int              srcWidth,
                      unsigned int              srcHeight,
                      uint8_t* __restrict       dst,
                      unsigned int              threshold );


//------------------------------------------------------------------------------
/// @brief
///   Binarizes a grayscale image based on a threshold value.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvFilterThresholdu8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvFilterThresholdu8,
///   \a fcvFilterThresholdu8_v2 will be removed, and the current signature
///   for \a fcvFilterThresholdu8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvFilterThresholdu8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   Sets the pixel to max(255) if it's value is greater than the threshold;
///   else, set the pixel to min(0).
///
/// @param src
///   Input 8-bit image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcStride
///   Image stride.
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param dst
///   Output 8-bit binarized image. Size of buffer is dstStride*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
/// 
/// @param dstStride
///   Output stride (in bytes).
///   \n\b NOTE: if 0, dstStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8 (8 * 1-byte values), and at least as much as srcWidth if not 0.
///
/// @param threshold
///   Threshold value for binarization.
///   \n\b WARNING: must be larger than 0.
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterThresholdu8_v2( const uint8_t* __restrict src,
                         unsigned int              srcWidth,
                         unsigned int              srcHeight,
                         unsigned int              srcStride,
                         uint8_t* __restrict       dst,
                         unsigned int              dstStride,
                         unsigned int              threshold );


//------------------------------------------------------------------------------
/// @brief
///   Dilate a grayscale image by taking the local maxima of 3x3 neighborhood window.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvFilterDilate3x3u8_v2(). In the 2.0.0 release, 
///   fcvFilterDilate3x3u8_v2 will be renamed to fcvFilterDilate3x3u8
///   and the signature of fcvFilterDilate3x3u8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param src
///   Input 8-bit image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
///
/// @param dst
///   Output 8-bit dilated image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterDilate3x3u8( const uint8_t* __restrict src,
                      unsigned int              srcWidth,
                      unsigned int              srcHeight,
                      uint8_t* __restrict       dst );


//------------------------------------------------------------------------------
/// @brief
///   Dilate a grayscale image by taking the local maxima of 3x3 neighborhood window.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvFilterDilate3x3u8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvFilterDilate3x3u8,
///   \a fcvFilterDilate3x3u8_v2 will be removed, and the current signature
///   for \a fcvFilterDilate3x3u8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvFilterDilate3x3u8 when transitioning to 2.0.0.
///   \n\n
///
/// @param src
///   Input 8-bit image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcStride
///   Image stride.
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param dst
///   Output 8-bit dilated image. Size of buffer is dstStride*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
/// 
/// @param dstStride
///   Stride of output image.
///   \n\b NOTE: if 0, dstStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterDilate3x3u8_v2( const uint8_t* __restrict src,
                         unsigned int              srcWidth,
                         unsigned int              srcHeight,
                         unsigned int              srcStride,
                         uint8_t* __restrict       dst,
                         unsigned int              dstStride );

//------------------------------------------------------------------------------
/// @brief
///   Erode a grayscale image by taking the local minima of 3x3 neighborhood window.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvFilterErode3x3u8_v2(). In the 2.0.0 release, 
///   fcvFilterErode3x3u8_v2 will be renamed to fcvFilterErode3x3u8
///   and the signature of fcvFilterErode3x3u8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param src
///   Input 8-bit image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
///
/// @param dst
///   Output 8-bit eroded image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterErode3x3u8( const uint8_t* __restrict src,
                     unsigned int              srcWidth,
                     unsigned int              srcHeight,
                     uint8_t* __restrict       dst );

//------------------------------------------------------------------------------
/// @brief
///   Erode a grayscale image by taking the local minima of 3x3 nbhd window.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvFilterErode3x3u8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvFilterErode3x3u8,
///   \a fcvFilterErode3x3u8_v2 will be removed, and the current signature
///   for \a fcvFilterErode3x3u8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvFilterErode3x3u8 when transitioning to 2.0.0.
///   \n\n
///
/// @param src
///   Input 8-bit image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcStride
///   Image stride.
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param dst
///   Output 8-bit eroded image. Size of buffer is dstStride*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
/// 
/// @param dstStride
///   Stride of output image.
///   \n\b NOTE: if 0, dstStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvFilterErode3x3u8_v2( const uint8_t* __restrict src,
                        unsigned int              srcWidth,
                        unsigned int              srcHeight,
                        unsigned int              srcStride,
                        uint8_t* __restrict       dst,
                        unsigned int              dstStride );

//---------------------------------------------------------------------------
/// @brief
///   Warps the patch centered at nPos in the input image using the affine
///   transform in nAffine
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvTransformAffine8x8u8_v2(). In the 2.0.0 release, 
///   fcvTransformAffine8x8u8_v2 will be renamed to fcvTransformAffine8x8u8
///   and the signature of fcvTransformAffine8x8u8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param src
///   Input image. Size of buffer is srcWidth*srcHeight bytes.
///
/// @param srcWidth
///   Image width.
///
/// @param srcHeight
///   Image height.
///
/// @param nPos[ 2 ]
///   Position in the image in 32 bit fixed point (Q16)
///   \n\b NOTE: if any 1 coordinates of the warped square are inside the image, return 1 and
///   \n   leave function. Otherwise, return 0.
///   \n\b WARNING: must be 64-bit aligned.
///
/// @param nAffine[ 2 ][ 2 ]
///   Transformation matrix in 32 bit fixed point (Q16). The matrix stored
///    in nAffine is using row major ordering: \n
///    a11, a12, a21, a22 where the matrix is: \n
///    | a11, a12 |\n
///    | a21, a22 |\n
///    
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param nPatch
///   Transformed patch.
///
///
/// @returns 0 if the transformation is valid
///
/// 
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API int
fcvTransformAffine8x8u8( const uint8_t* __restrict src,
                         unsigned int              srcWidth,
                         unsigned int              srcHeight,
                         const int32_t* __restrict nPos,
                         const int32_t* __restrict nAffine,
                         uint8_t* __restrict       nPatch );


//---------------------------------------------------------------------------
/// @brief
///   Warps the patch centered at nPos in the input image using the affine
///   transform in nAffine
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvTransformAffine8x8u8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvTransformAffine8x8u8,
///   \a fcvTransformAffine8x8u8_v2 will be removed, and the current signature
///   for \a fcvTransformAffine8x8u8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvTransformAffine8x8u8 when transitioning to 2.0.0.
///   \n\n
///
/// @param src
///   Input image. Size of buffer is srcStride*srcHeight bytes.
///
/// @param srcWidth
///   Image width.
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcStride
///   Stride of image (in bytes) - i.e., how many bytes between column 0 of row N 
///   and column 0 of row N+1.
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be at least as much as srcWidth if not 0.
///
/// @param nPos[ 2 ]
///   Position in the image in 32 bit fixed point (Q16)
///   \n\b NOTE: if any 1 coordinates of the warped square are inside the image, return 1 and
///   \n   leave function. Otherwise, return 0.
///   \n\b WARNING: must be 64-bit aligned.
///
/// @param nAffine[ 2 ][ 2 ]
///   Transformation matrix in 32 bit fixed point (Q16). The matrix stored
///    in nAffine is using row major ordering: \n
///    a11, a12, a21, a22 where the matrix is: \n
///    | a11, a12 |\n
///    | a21, a22 |\n
///    
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param patch
///   Transformed patch.
/// 
/// @param patchStride
///   Stride of patch (in bytes) - i.e., how many bytes between column 0 of row N 
///   and column 0 of row N+1.
///   \n\b NOTE: if 0, srcStride is set as 8.
///   \n\b WARNING: must be at least as much as 8 if not 0.
///
///
/// @returns 0 if the transformation is valid
///
/// 
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API int
fcvTransformAffine8x8u8_v2( const uint8_t* __restrict src,
                            unsigned int              srcWidth,
                            unsigned int              srcHeight,
                            unsigned int              srcStride,
                            const int32_t* __restrict nPos,
                            const int32_t* __restrict nAffine,
                            uint8_t* __restrict       patch,
                            unsigned int              patchStride );


//------------------------------------------------------------------------------
/// @brief
///   Warps a grayscale image using the a perspective projection transformation
///   matrix (also known as a homography). This type of transformation is an
///   invertible transformation which maps straight lines to straight lines.
///   Bi-linear interpolation is used where applicable.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvWarpPerspectiveu8_v2(). In the 2.0.0 release, 
///   fcvWarpPerspectiveu8_v2 will be renamed to fcvWarpPerspectiveu8
///   and the signature of fcvWarpPerspectiveu8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///   Warps an image taking into consideration the perspective scaling.
///
/// @param src
///   Input 8-bit image. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Input image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Input image height.
///   \n\b WARNING: must be multiple of 8.
///
/// @param dst
///   Warped output image. Size of buffer is dstWidth*dstHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param dstWidth
///   Dst image width.
///   \n\b NOTE: data must be multiple of 8.
///
/// @param dstHeight
///   Dst image height.
///   \n\b NOTE: must be multiple of 8
///
/// @param projectionMatrix
///   3x3 perspective transformation matrix (generally a homography). The
///   matrix stored in homography is row major ordering: \n
///   a11, a12, a13, a21, a22, a23, a31, a32, a33 where the matrix is: \n
///   | a11, a12, a13 |\n
///   | a21, a22, a23 |\n
///   | a31, a32, a33 |\n
///   \n\b WARNING: must be 128-bit aligned.
///
/// 
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API void
fcvWarpPerspectiveu8( const uint8_t* __restrict src,
                      unsigned int              srcWidth,
                      unsigned int              srcHeight,
                      uint8_t* __restrict       dst,
                      unsigned int              dstWidth,
                      unsigned int              dstHeight,
                      float* __restrict         projectionMatrix );


//------------------------------------------------------------------------------
/// @brief
///   Warps a grayscale image using the a perspective projection transformation
///   matrix (also known as a homography). This type of transformation is an
///   invertible transformation which maps straight lines to straight lines.
///   Bi-linear interpolation is used where applicable.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvWarpPerspectiveu8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvWarpPerspectiveu8,
///   \a fcvWarpPerspectiveu8_v2 will be removed, and the current signature
///   for \a fcvWarpPerspectiveu8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvWarpPerspectiveu8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   Warps an image taking into consideration the perspective scaling.
///
/// @param src
///   Input 8-bit image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Input image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Input image height.
///   \n\b WARNING: must be multiple of 8.
/// 
/// @param srcStride
///   Input image stride (in bytes).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8 (8 * 1-byte values), and at least as much as srcWidth if not 0.
///
/// @param dst
///   Warped output image. Size of buffer is dstStride*dstHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param dstWidth
///   Dst image width.
///   \n\b NOTE: data must be multiple of 8.
///
/// @param dstHeight
///   Dst image height.
///   \n\b NOTE: must be multiple of 8
/// 
/// @param dstStride
///   Output image stride (in bytes).
///   \n\b NOTE: if 0, dstStride is set as dstWidth.
///   \n\b WARNING: must be multiple of 8 (8 * 1-byte values), and at least as much as dstWidth if not 0.
///
/// @param projectionMatrix
///   3x3 perspective transformation matrix (generally a homography). The
///   matrix stored in homography is row major ordering: \n
///   a11, a12, a13, a21, a22, a23, a31, a32, a33 where the matrix is: \n
///   | a11, a12, a13 |\n
///   | a21, a22, a23 |\n
///   | a31, a32, a33 |\n
///   \n\b WARNING: must be 128-bit aligned.
///
/// 
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API void
fcvWarpPerspectiveu8_v2( const uint8_t* __restrict src,
                         unsigned int              srcWidth,
                         unsigned int              srcHeight,
                         unsigned int              srcStride,
                         uint8_t* __restrict       dst,
                         unsigned int              dstWidth,
                         unsigned int              dstHeight,
                         unsigned int              dstStride,
                         float* __restrict         projectionMatrix );


//---------------------------------------------------------------------------
/// @brief
///   Warps a 3 color channel image based on a 3x3 perspective projection matrix using
///   bilinear interpolation.
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcv3ChannelWarpPerspectiveu8_v2(). In the 2.0.0 release, 
///   fcv3ChannelWarpPerspectiveu8_v2 will be renamed to fcv3ChannelWarpPerspectiveu8
///   and the signature of fcv3ChannelWarpPerspectiveu8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param src
///   Input image. Size of buffer is srcWidth*srcHeight*3 bytes.
///   \n\b NOTE: data must be 128-bit aligned.
///
/// @param srcWidth
///   Input image width.
///   \n\b NOTE: must be multiple of 8
///
/// @param srcHeight
///   Input image height.
///   \n\b NOTE: must be multiple of 8
///
/// @param dst
///   Warped output image. Size of buffer is dstWidth*dstHeight*3 bytes.
///   \n\b NOTE: data must be 128-bit aligned.
///
/// @param dstWidth
///   Output image width.
///   \n\b NOTE: must be multiple of 8.
///
/// @param dstHeight
///   Output image height.
///   \n\b NOTE: must be multiple of 8.
///
/// @param projectionMatrix
///   3x3 perspective transformation matrix (generally a homography). The
///   matrix stored in homography is row major ordering: \n
///   a11, a12, a13, a21, a22, a23, a31, a32, a33 where the matrix is: \n
///   | a11, a12, a13 |\n
///   | a21, a22, a23 |\n
///   | a31, a32, a33 |\n
///   \n\b WARNING: must be 128-bit aligned.
///
/// @ingroup image_transform
//---------------------------------------------------------------------------

FASTCV_API void
fcv3ChannelWarpPerspectiveu8( const uint8_t* __restrict src,
                              unsigned int              srcWidth,
                              unsigned int              srcHeight,
                              uint8_t* __restrict       dst,
                              unsigned int              dstWidth,
                              unsigned int              dstHeight,
                              float* __restrict         projectionMatrix );


//---------------------------------------------------------------------------
/// @brief
///   Warps a 3 color channel image based on a 3x3 perspective projection 
///   matrix using bilinear interpolation.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcv3ChannelWarpPerspectiveu8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcv3ChannelWarpPerspectiveu8,
///   \a fcv3ChannelWarpPerspectiveu8_v2 will be removed, and the current signature
///   for \a fcv3ChannelWarpPerspectiveu8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcv3ChannelWarpPerspectiveu8 when transitioning to 2.0.0.
///   \n\n
///
/// @param src
///   Input image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b WARNING: data must be 128-bit aligned.
///
/// @param srcWidth
///   Input image width.
///   \n\b WARNING: must be multiple of 8
///
/// @param srcHeight
///   Input image height.
///   \n\b WARNING: must be multiple of 8
/// 
/// @param srcStride
///   Input image stride (in bytes).
///   \n\b NOTE: if 0, srcStride is set as srcWidth*3.
///   \n\b WARNING: must be multiple of 8 (8 * 1-byte values), and at least as much as srcWidth*3 if not 0.
///
/// @param dst
///   Warped output image. Size of buffer is dstStride*dstHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param dstWidth
///   Output image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param dstHeight
///   Output image height.
///   \n\b WARNING: must be multiple of 8.
/// 
/// @param dstStride
///   Output image stride (in bytes).
///   \n\b NOTE: if 0, dstStride is set as dstWidth*3.
///   \n\b WARNING: must be multiple of 8 (8 * 1-byte values), and at least as much as dstWidth*3 if not 0.
///
/// @param projectionMatrix
///   3x3 perspective transformation matrix (generally a homography). The
///   matrix stored in homography is row major ordering: \n
///   a11, a12, a13, a21, a22, a23, a31, a32, a33 where the matrix is: \n
///   | a11, a12, a13 |\n
///   | a21, a22, a23 |\n
///   | a31, a32, a33 |\n
///   \n\b WARNING: must be 128-bit aligned.
///
/// @ingroup image_transform
//---------------------------------------------------------------------------

FASTCV_API void
fcv3ChannelWarpPerspectiveu8_v2( const uint8_t* __restrict src,
                                 unsigned int              srcWidth,
                                 unsigned int              srcHeight,
                                 unsigned int              srcStride,
                                 uint8_t* __restrict       dst,
                                 unsigned int              dstWidth,
                                 unsigned int              dstHeight,
                                 unsigned int              dstStride,
                                 float* __restrict         projectionMatrix );


//---------------------------------------------------------------------------
/// @brief
///   General function for computing cluster centers and cluster bindings
///   for a set of points of dimension dim.
///
/// @param points
///   Array of all points. Array size must be greater than
///   numPoints * dim.
///
/// @param numPoints
///   Number of points in points array.
///   \n\b WARNING: must be > numPoints * dim.
///
/// @param dim
///   dimension, e.g. 36
///
/// @param pointStride
///   Byte distance between adjacent points in array
///
/// @param indices
///   Array of point indices in points array. Processing will only
///   occur on points whose indices are in this array. Each index in array
///   must be smaller numPoints.
///
/// @param numIndices
///   Length of indices array. numIndieces must be <= numPoints.
///
/// @param numClusters
///   Number of cluster centers
///
/// @param clusterCenters
///   current cluster centers;
///   elements are distant by clusterCenterStride
///
/// @param clusterCenterStride
///   byte distance between adjacent cluster centers in array
///
/// @param newClusterCenters
///   array for new cluster centers; should be numClusterCenters long
///
/// @param clusterMemberCounts
///   Element counts for each cluster; should be numClusterCenters long
///
/// @param clusterBindings
///   Output indices of the clusters to which each vector belongs to, array must
///   be numIndices long.
/// 
/// @param sumOfClusterDistances
///   Array for sum of distances of cluster elements to cluster centers;
///   Must be numClusters long
///
/// @return
///   0 if successfully clustered, otherwise error code
///
/// @remark
///   This is general clusterer. There are no assumptions on points other
///   than they belong to a vector space
///
/// 
///
/// @ingroup clustering_and_search
//---------------------------------------------------------------------------

FASTCV_API int
fcvClusterEuclideanf32( const float* __restrict  points,
                        int                      numPoints,  // actually not used but helpful
                        int                      dim,
                        int                      pointStride,
                        const size_t* __restrict indices,
                        int                      numIndices,
                        int                      numClusters,
                        float* __restrict        clusterCenters,
                        int                      clusterCenterStride,
                        float* __restrict        newClusterCenters,
                        size_t* __restrict       clusterMemberCounts,
                        size_t* __restrict       clusterBindings,
                        float*                   sumOfClusterDistances );


//---------------------------------------------------------------------------
/// @brief
///   Function for computing cluster centers and cluster bindings
///   for a set of normalized points of dimension dim. Cluster centers
///   are also normalized (see remark below)
///
/// @param points
///   Array of all points. Array size must be greater than
///   numPoints * dim.
///
/// @param numPoints
///   Number of points in points array.
///
/// @param dim
///   dimension, e.g. 36
///
/// @param pointStride
///   Byte distance between adjacent points in array
///
/// @param indices
///   Array of point indices in points array. Processing will only
///   occur on points whose indices are in this array. Each index in array
///   must be smaller numPoints.
///
/// @param numIndices
///   Length of indices array. numIndieces must be <= numPoints.
/// 
/// @param numClusters
///   Number of cluster centers
///
/// @param clusterCenters
///   current cluster centers;
///   elements are distant by clusterCenterStride
///
/// @param clusterCenterStride
///   byte distance between adjacent cluster centers in array
///
/// @param newClusterCenters
///   array for new cluster centers; should be numClusterCenters long
///
/// @param clusterMemberCounts
///   Element counts for each cluster; should be numClusterCenters long
///
/// @param clusterBindings
///   Output indices of the clusters to which each vector belongs to, a
///   rray must be numIndices long.
///
/// @param sumOfClusterDistances
///   Array for sum of distances of cluster elements to cluster centers;
///   Must be numClusters long
///
/// @return
///   0 if successfully clustered, otherwise error code
///
/// @remark
///   this function assumes that points are normalized (e.g. NFT4
///   descriptors). Cluster centers are also normalized. Normalized points
///   are on a surface of unit sphere which is not a vector space but
///   curved manifold of dimension (dim-1) embeded in Euclidean vector space
///   of dimension dim
///
/// @ingroup clustering_and_search
//---------------------------------------------------------------------------

FASTCV_API int
fcvClusterEuclideanNormedf32( const float* __restrict  points,
                              int                      numPoints,
                              int                      dim,
                              int                      pointStride,
                              const size_t* __restrict indices,
                              int                      numIndices,
                              int                      numClusters,
                              float* __restrict        clusterCenters,
                              int                      clusterCenterStride,
                              float* __restrict        newClusterCenters,
                              size_t* __restrict       clusterMemberCounts,
                              size_t* __restrict       clusterBindings,
                              float*                   sumOfClusterDistances );


//---------------------------------------------------------------------------
/// @brief
///   Function for computing cluster centers and cluster bindings
///   for a set of normalized points of dimension 36. Cluster centers
///   are also normalized (see remark below)
///
/// @param points
///   Array of all points. Array size must be greater than
///   numPoints * 36.
///
/// @param numPoints
///   Number of points in points array.
///
/// @param pointStride
///   Byte distance between adjacent points in array
///
/// @param indices
///   Array of point indices in points array. Processing will only
///   occur on points whose indices are in this array. Each index in array
///   must be smaller numPoints.
///
/// @param numIndices
///   Length of indices array. numIndieces must be <= numPoints.
///
/// @param numClusters
///   Number of cluster centers
///
/// @param clusterCenters
///   current cluster centers;
///   elements are distant by clusterCenterStride
///
/// @param clusterCenterStride
///   byte distance between adjacent cluster centers in array
///
/// @param newClusterCenters
///   array for new cluster centers; should be numClusterCenters long
///
/// @param clusterMemberCounts
///   Element counts for each cluster; should be numClusterCenters long
///
/// @param clusterBindings
///   Output indices of the clusters to which each vector belongs to, a
///   rray must be numIndices long.
/// 
/// @param sumOfClusterDistances
///   Array for sum of distances of cluster elements to cluster centers;
///   Must be numClusters long
///
/// @return
///   0 if successfully clustered, otherwise error code
///
/// @remark
///   this function assumes that points are normalized (e.g. NFT4
///   descriptors). Cluster centers are also normalized. Normalized points
///   are on a surphace of unit sphere which is not a vector space but
///   curved manifold of dimension (dim-1) embeded in Euclidean vector space
///   of dimension dim
///
/// @ingroup clustering_and_search
//---------------------------------------------------------------------------

FASTCV_API int
fcvClusterEuclideanNormed36f32( const float* __restrict  points,
                                int                      numPoints,
                                int                      pointStride,
                                const size_t* __restrict indices,
                                int                      numIndices,
                                int                      numClusters,
                                float* __restrict        clusterCenters,
                                int                      clusterCenterStride,
                                float* __restrict        newClusterCenters,
                                size_t* __restrict       clusterMemberCounts,
                                size_t* __restrict       clusterBindings,
                                float*                   sumOfClusterDistances );


//---------------------------------------------------------------------------
/// @brief
///   Blur with 5x5 Gaussian filter
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvFilterGaussian5x5s16_v2(). In the 2.0.0 release, 
///   fcvFilterGaussian5x5s16_v2 will be renamed to fcvFilterGaussian5x5s16
///   and the signature of fcvFilterGaussian5x5s16 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///   Convolution with 5x5 Gaussian kernel:
///   \n 1  4  6  4 1
///   \n 4 16 24 16 4
///   \n 6 24 36 24 6
///   \n 4 16 24 16 4
///   \n 1  4  6  4 1
///
/// @param src
///   Input int data (can be sq. of gradient, etc).
///   \n\b NOTE: Size of buffer is srcWidth*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///
/// @param srcHeight
///   Image height.
///
/// @param dst
///   Output int data. Size of buffer is srcWidth*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param blurBorder
///   If set to 0, border is ignored.
///   If set to 1, border is blurred by 0-padding adjacent values.
///
/// @ingroup image_processing
//---------------------------------------------------------------------------

FASTCV_API void
fcvFilterGaussian5x5s16( const int16_t* __restrict src,
                         unsigned int              srcWidth,
                         unsigned int              srcHeight,
                         int16_t* __restrict       dst,
                         int                       blurBorder );


//---------------------------------------------------------------------------
/// @brief
///   Blur with 5x5 Gaussian filter
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvFilterGaussian5x5s16() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvFilterGaussian5x5s16,
///   \a fcvFilterGaussian5x5s16_v2 will be removed, and the current signature
///   for \a fcvFilterGaussian5x5s16 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvFilterGaussian5x5s16 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   Convolution with 5x5 Gaussian kernel:
///   \n 1  4  6  4 1
///   \n 4 16 24 16 4
///   \n 6 24 36 24 6
///   \n 4 16 24 16 4
///   \n 1  4  6  4 1
///
/// @param src
///   Input int data (can be sq. of gradient, etc).
///   Size of buffer is srcStride*srcHeight*2 bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcStride
///   Image stride.
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param dst
///   Output int data. Size of buffer is dstStride*srcHeight*2 bytes.
///   \n\b NOTE: data should be 128-bit aligned.
/// 
/// @param dstStride
///   Output stride.
///   \n\b NOTE: if 0, edstStrid is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param blurBorder
///   If set to 0, border is ignored.
///   If set to 1, border is blurred by 0-padding adjacent values.
///
/// @ingroup image_processing
//---------------------------------------------------------------------------

FASTCV_API void
fcvFilterGaussian5x5s16_v2( const int16_t* __restrict src,
                            unsigned int              srcWidth,
                            unsigned int              srcHeight,
                            unsigned int              srcStride,
                            int16_t* __restrict       dst,
                            unsigned int              dstStride,
                            int                       blurBorder );


//---------------------------------------------------------------------------
/// @brief
///   Blur with 5x5 Gaussian filter
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvFilterGaussian5x5s32_v2(). In the 2.0.0 release, 
///   fcvFilterGaussian5x5s32_v2 will be renamed to fcvFilterGaussian5x5s32
///   and the signature of fcvFilterGaussian5x5s32 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///   Convolution with 5x5 Gaussian kernel:
///   \n 1  4  6  4 1
///   \n 4 16 24 16 4
///   \n 6 24 36 24 6
///   \n 4 16 24 16 4
///   \n 1  4  6  4 1
///
/// @param src
///   Input int data (can be sq. of gradient, etc).
///   Size of buffer is srcWidth*srcHeight*4 bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///
/// @param srcHeight
///   Image height.
///
/// @param dst
///   Output int data. Size of buffer is srcWidth*srcHeight*4 bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param blurBorder
///   If set to 0, border is ignored.
///   If set to 1, border is blurred by 0-padding adjacent values.
///
/// @ingroup image_processing
//---------------------------------------------------------------------------

FASTCV_API void
fcvFilterGaussian5x5s32( const int32_t* __restrict src,
                         unsigned int              srcWidth,
                         unsigned int              srcHeight,
                         int32_t* __restrict       dst,
                         int                       blurBorder );


//---------------------------------------------------------------------------
/// @brief
///   Blur with 5x5 Gaussian filter
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvFilterGaussian5x5s32() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvFilterGaussian5x5s32,
///   \a fcvFilterGaussian5x5s32_v2 will be removed, and the current signature
///   for \a fcvFilterGaussian5x5s32 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvFilterGaussian5x5s32 when transitioning to 2.0.0.
///   \n\n
///
/// @details
///   Convolution with 5x5 Gaussian kernel:
///   \n 1  4  6  4 1
///   \n 4 16 24 16 4
///   \n 6 24 36 24 6
///   \n 4 16 24 16 4
///   \n 1  4  6  4 1
///
/// @param src
///   Input int data (can be sq. of gradient, etc).
///   Size of buffer is srcStride*srcHeight*4 bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcStride
///   Image stride.
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param dst
///   Output int data. Size of buffer is dstStride*srcHeight*4 bytes.
///   \n\b NOTE: data should be 128-bit aligned.
/// 
/// @param dstStride
///   Output stride.
///   \n\b NOTE: if 0, dstStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param blurBorder
///   If set to 0, border is ignored.
///   If set to 1, border is blurred by 0-padding adjacent values.
///
/// @ingroup image_processing
//---------------------------------------------------------------------------

FASTCV_API void
fcvFilterGaussian5x5s32_v2( const int32_t* __restrict src,
                            unsigned int              srcWidth,
                            unsigned int              srcHeight,
                            unsigned int              srcStride,
                            int32_t* __restrict       dst,
                            unsigned int              dstStride,
                            int                       blurBorder );


//---------------------------------------------------------------------------
/// @brief
///   Warps the patch centered at nPos in the input image using the affine
///   transform in nAffine
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvTransformAffineu8_v2(). In the 2.0.0 release, 
///   fcvTransformAffineu8_v2 will be renamed to fcvTransformAffineu8
///   and the signature of fcvTransformAffineu8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param src
///   Input image. Size of buffer is srcWidth*srcHeight bytes.
///
/// @param srcWidth
///   Image width.
///
/// @param srcHeight
///   Image height.
///
/// @param position[ 2 ]
///   Position in the image
///   \n\b WARNING: must be 64-bit aligned.
///
/// @param affine[ 2 ][ 2 ]
///   Transformation matrix. The matrix stored
///    in affine is using row major ordering: \n
///    a11, a12, a21, a22 where the matrix is: \n
///    | a11, a12 |\n
///    | a21, a22 |\n
///    
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param patch
///   Transformed patch.
///
/// @param patchWidth
///   Patch width.
///
/// @param patchHeight
///   Patch height.
///
/// @returns 0 if the transformation is valid
///
/// @ingroup image_transform
//---------------------------------------------------------------------------

FASTCV_API int
fcvTransformAffineu8( const uint8_t* __restrict src,
                      unsigned int              srcWidth,
                      unsigned int              srcHeight,
                      const float* __restrict   position,
                      const float* __restrict   affine,
                      uint8_t* __restrict       patch,
                      unsigned int              patchWidth,
                      unsigned int              patchHeight );


//---------------------------------------------------------------------------
/// @brief
///   Warps the patch centered at nPos in the input image using the affine
///   transform in nAffine
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvTransformAffineu8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvTransformAffineu8,
///   \a fcvTransformAffineu8_v2 will be removed, and the current signature
///   for \a fcvTransformAffineu8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvTransformAffineu8 when transitioning to 2.0.0.
///   \n\n
///
/// @param src
///   Input image. Size of buffer is srcStride*srcHeight bytes.
///
/// @param srcWidth
///   Image width.
///
/// @param srcHeight
///   Image height.
/// 
/// @param srcStride
///   Stride of image (in bytes) - i.e., how many bytes between column 0 of row N 
///   and column 0 of row N+1.
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be at least as much as srcWidth if not 0.
///
/// @param position[ 2 ]
///   Position in the image
///   \n\b WARNING: must be 64-bit aligned.
///
/// @param affine[ 2 ][ 2 ]
///   Transformation matrix. The matrix stored
///    in affine is using row major ordering: \n
///    a11, a12, a21, a22 where the matrix is: \n
///    | a11, a12 |\n
///    | a21, a22 |\n
///    
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param patch
///   Transformed patch.
///
/// @param patchWidth
///   Patch width.
///
/// @param patchHeight
///   Patch height.
/// 
/// @param patchStride
///   Stride of patch (in bytes) - i.e., how many bytes between column 0 of row N 
///   and column 0 of row N+1.
///   \n\b NOTE: if 0, patchStride is set as patchWidth.
///   \n\b WARNING: must be at least as much as patchWidth if not 0.
///
/// @returns 0 if the transformation is valid
///
/// @ingroup image_transform
//---------------------------------------------------------------------------

FASTCV_API int
fcvTransformAffineu8_v2( const uint8_t* __restrict src,
                         unsigned int              srcWidth,
                         unsigned int              srcHeight,
                         unsigned int              srcStride,
                         const float* __restrict   position,
                         const float* __restrict   affine,
                         uint8_t* __restrict       patch,
                         unsigned int              patchWidth,
                         unsigned int              patchHeight,
                         unsigned int              patchStride );


//---------------------------------------------------------------------------
/// @brief
///   Extracts a 17x17 rotation corrected patch from a 25x25 image.
///
/// @param src
///   25x25 input image in continuous memory.
///
/// @param dst
///   17x17 output patch.
///
/// @param orientation
///   Rotation angle of patch relative to src.
///   \n 10-bit fixed-point angle around unit circle.
///   \n NOTE: 0 = 0 degrees and 1024 = 360 degrees.
///
/// @ingroup image_transform
//---------------------------------------------------------------------------

FASTCV_API void
fcvCopyRotated17x17u8( const uint8_t* __restrict src,
                       uint8_t* __restrict       dst,
                       int                       orientation );


//------------------------------------------------------------------------------
/// @brief
///   Counts "1" bits in supplied vector.
///
/// @param src
///   Pointer to vector to count bits that are 1.
///
/// @param srcLength
///   Length of the vector to count bits. Assumed that the remainder of bits modulo 8
///   will be set to 0 a priori.
///
/// @returns total number of "1" bits in supplied vector
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API uint32_t
fcvBitCountu8( const uint8_t* __restrict src,
               unsigned int              srcLength );


//------------------------------------------------------------------------------
/// @brief
///   Counts "1" bits in supplied 32-byte vector.
///
/// @param src
///   Pointer to 32-byte vector(s) to count bits that are 1.
///
/// @returns total number of "1" bits in supplied vector
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API uint32_t
fcvBitCount32x1u8( const uint8_t* __restrict src );


//------------------------------------------------------------------------------
/// @brief
///   Counts bits in supplied 4, 32-byte vectors.
///
/// @param a
///   Pointer to 32-byte vector to count bits.
///
/// @param b
///   Pointer to 32-byte vector to count bits.
///
/// @param c
///   Pointer to 32-byte vector to count bits.
///
/// @param d
///   Pointer to 32-byte vector to count bits.
///
/// @param bitCount
///   Array to store the four resultant bit counts.
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API void
fcvBitCount32x4u8( const uint8_t* __restrict a,
                   const uint8_t* __restrict b,
                   const uint8_t* __restrict c,
                   const uint8_t* __restrict d,
                   uint32_t* __restrict      bitCount );


//------------------------------------------------------------------------------
/// @brief
///   Counts bits in supplied 64-byte vector.
///
/// @param src
///   Pointer to 64-byte vector(s) to count bits.
///
/// @return
///   Bit count.
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API uint32_t
fcvBitCount64x1u8( const uint8_t* __restrict src );


//------------------------------------------------------------------------------
/// @brief
///   Counts bits in supplied 4, 64-byte vectors.
///
/// @param a
///   Pointer to 64-byte vector to count bits.
///
/// @param b
///   Pointer to 64-byte vector to count bits.
///
/// @param c
///   Pointer to 64-byte vector to count bits.
///
/// @param d
///   Pointer to 64-byte vector to count bits.
///
/// @param bitCount
///   Array to store the four resultant bit counts.
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API void
fcvBitCount64x4u8( const uint8_t* __restrict a,
                   const uint8_t* __restrict b,
                   const uint8_t* __restrict c,
                   const uint8_t* __restrict d,
                   uint32_t* __restrict      bitCount );


//------------------------------------------------------------------------------
/// @brief
///   Counts bits in supplied  vector of unsigned intergers.
///
/// @param src
///   Pointer to vector(s) to count bits.
///
/// @param srcLength
///   Number of elements in vector
///
/// @return
///   Bit count.
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API uint32_t
fcvBitCountu32( const uint32_t* __restrict src,
                unsigned int               srcLength );


//------------------------------------------------------------------------------
/// @brief
///   Computes the Hamming distance between the two supplied arbitrary length
///   vectors.
///
/// @param a
///   Pointer to vector to compute distance.
///
/// @param b
///   Pointer to vector to compute distance.
///
/// @param abLength
///   Length in bits of each of the vectors. Assumed that the remainder of
///   bits modulo 8 will be set to 0 a priori.
///
/// @return
///   Hamming distance between the two vectors.
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API uint32_t
fcvHammingDistanceu8( const uint8_t* __restrict a,
                      const uint8_t* __restrict b,
                      unsigned int              abLength );


//------------------------------------------------------------------------------
/// @brief
///   Computes the Hamming distance between the two supplied 32-byte vectors.
///
/// @param a
///   Pointer to 32-byte vector to compute distance.
///   \n\b WARNING: must be 32-bit aligned
///
/// @param b
///   Pointer to 32-byte vector to compute distance.
///   \n\b WARNING: must be 32-bit aligned
///
/// @return
///   Hamming distance between the two vectors.
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API uint32_t
fcvHammingDistance32x1u8a4( const uint8_t* __restrict a,
                            const uint8_t* __restrict b );


//------------------------------------------------------------------------------
/// @brief
///   Computes the Hamming distance between the two supplied 64-byte vectors.
///
/// @param a
///   Pointer to 64-byte vector to compute distance.
///   \n\b WARNING: must be 32-bit aligned
///
/// @param b
///   Pointer to 64-byte vector to compute distance.
///   \n\b WARNING: must be 32-bit aligned
///
/// @return
///   Hamming distance between the two vectors.
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API uint32_t
fcvHammingDistance64x1u8a4( const uint8_t* __restrict a,
                            const uint8_t* __restrict b );


//------------------------------------------------------------------------------
/// @brief
///   Computes the Hamming distance between the two supplied 32-byte vectors.
///
/// @param a
///   Pointer to 32-byte vector to compute distance.
///
/// @param b
///   Pointer to 32-byte vector to compute distance.
///
/// @return
///   Hamming distance between the two vectors.
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API uint32_t
fcvHammingDistance32x1u8( const uint8_t* __restrict a,
                          const uint8_t* __restrict b );


//------------------------------------------------------------------------------
/// @brief
///   Computes the Hamming distance between the two supplied 64-byte vectors.
///
/// @param a
///   Pointer to 64-byte vector to compute distance.
///   \n\b WARNING: must be 32-bit aligned
///
/// @param b
///   Pointer to 64-byte vector to compute distance.
///   \n\b WARNING: must be 32-bit aligned
///
/// @return
///   Hamming distance between the two vectors.
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API uint32_t
fcvHammingDistance64x1u8( const uint8_t* __restrict a,
                          const uint8_t* __restrict b );


//------------------------------------------------------------------------------
/// @brief
///   Computes the Hamming distance between A and each of B,C,D,E 32-byte vectors.
///
/// @param a
///   Pointer to 32-byte vector to compute distance.
///   \n\b WARNING: must be 32-bit aligned
///
/// @param b
///   Pointer to 32-byte vector to compute distance from A.
///   \n\b WARNING: must be 32-bit aligned
///
/// @param c
///   Pointer to 32-byte vector to compute distance from A.
///   \n\b WARNING: must be 32-bit aligned
///
/// @param d
///   Pointer to 32-byte vector to compute distance from A.
///   \n\b WARNING: must be 32-bit aligned
///
/// @param e
///   Pointer to 32-byte vector to compute distance from A.
///   \n\b WARNING: must be 32-bit aligned
///
/// @param hammingDistances
///   Array to store each Hamming distance between the vectors.
///   \n\b WARNING: must be 128-bit aligned
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API void
fcvHammingDistance32x4u8a4( const uint8_t* __restrict a,
                            const uint8_t* __restrict b,
                            const uint8_t* __restrict c,
                            const uint8_t* __restrict d,
                            const uint8_t* __restrict e,
                            uint32_t* __restrict      hammingDistances );


//------------------------------------------------------------------------------
/// @brief
///   Computes the Hamming distance between A and each of B,C,D,E 64-byte
///   vectors.
///
/// @param a
///   Pointer to 32-byte vector to compute distance.
///   \n\b WARNING: must be 32-bit aligned
///
/// @param b
///   Pointer to 32-byte vector to compute distance from A.
///   \n\b WARNING: must be 32-bit aligned
///
/// @param c
///   Pointer to 32-byte vector to compute distance from A.
///   \n\b WARNING: must be 32-bit aligned
///
/// @param d
///   Pointer to 32-byte vector to compute distance from A.
///   \n\b WARNING: must be 32-bit aligned
///
/// @param e
///   Pointer to 32-byte vector to compute distance from A.
///   \n\b WARNING: must be 32-bit aligned
///
/// @param hammingDistances
///   Array to store each Hamming distance between the vectors.
///   \n\b WARNING: must be 128-bit aligned
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API void
fcvHammingDistance64x4u8a4( const uint8_t* __restrict a,
                            const uint8_t* __restrict b,
                            const uint8_t* __restrict c,
                            const uint8_t* __restrict d,
                            const uint8_t* __restrict e,
                            uint32_t* __restrict      hammingDistances );


//------------------------------------------------------------------------------
/// @brief
///   Computes the Hamming distance between A and each of B,C,D,E 64-byte vectors.
///
/// @param a
///   Pointer to 64-byte vector to compute distance.
///   \n\b WARNING: must be 32-bit aligned
///
/// @param b
///   Pointer to 64-byte vector to compute distance from A.
///   \n\b WARNING: must be 32-bit aligned
///
/// @param c
///   Pointer to 64-byte vector to compute distance from A.
///   \n\b WARNING: must be 32-bit aligned
///
/// @param d
///   Pointer to 64-byte vector to compute distance from A.
///   \n\b WARNING: must be 32-bit aligned
///
/// @param e
///   Pointer to 64-byte vector to compute distance from A.
///   \n\b WARNING: must be 32-bit aligned
///
/// @param hammingDistances
///   Array to store each Hamming distance between the vectors.
///
/// @ingroup math_vector
//------------------------------------------------------------------------------

FASTCV_API void
fcvHammingDistance64x4u8( const uint8_t* __restrict a,
                          const uint8_t* __restrict b,
                          const uint8_t* __restrict c,
                          const uint8_t* __restrict d,
                          const uint8_t* __restrict e,
                          uint32_t* __restrict      hammingDistances );


//---------------------------------------------------------------------------
/// @brief
///   Extracts FAST corners and scores from the image
///
/// @param src
///   8-bit image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width
///   \n\b NOTE: must be a multiple of 8.
///   \n\b WARNING: must be <= 2048.
///
/// @param srcHeight
///   image height
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///
/// @param barrier
///  FAST threshold. The threshold is used to compare difference between intensity value of 
///  the central pixel and pixels on a circle surrounding this pixel.
///
/// @param border
///   Number for pixels to ignore from top,bottom,right,left of the image
///
/// @param xy
///   pointer to the output array cointaining the interleaved x,y position of the
///   detected corners.
///   \n\b NOTE: Remember to allocate double the size of @param nCornersMax
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param scores
///   Pointer to the output array containing the scores of the detected corners.
///   The score is computed as the sum of the absolute difference between the pixels in the 
///   contiguous arc and the centre pixel. A higher score value indicates a stronger corner feature. 
///   For example, a corner of score 108 is stronger than a corner of score 50.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param nCornersMax
///   Maximum number of corners. The function exits when the maximum number of
///   corners is exceeded.
///
/// @param nCorners
///   pointer to an integer storing the number of corners detected
///
/// @ingroup feature_detection
//------------------------------------------------------------------------------

FASTCV_API void
fcvCornerFast9Scoreu8( const uint8_t* __restrict src,
                       unsigned int              srcWidth,
                       unsigned int              srcHeight,
                       unsigned int              srcStride,
                       int                       barrier,
                       unsigned int              border,
                       uint32_t* __restrict      xy,
                       uint32_t* __restrict      scores,
                       unsigned int              nCornersMax,
                       uint32_t* __restrict      nCorners );


//---------------------------------------------------------------------------
/// @brief
///   Extracts FAST corners and scores from the image
///
/// @param src
///   Grayscale image with one byte per pixel
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   image width
///   \n\b NOTE: must be a multiple of 8.
///   \n\b WARNING: must be <= 2048.
///
/// @param srcHeight
///   image height
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///
/// @param barrier
///  FAST threshold. The threshold is used to compare difference between intensity value of 
///  the central pixel and pixels on a circle surrounding this pixel.
///
/// @param border
///   Number for pixels to ignore from top,bottom,right,left of the image
///
/// @param xy
///   Pointer to the output array cointaining the interleaved x,y position of the
///   detected corners.
///   \n\b NOTE: Remember to allocate double the size of @param nCornersMax
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param scores
///   Pointer to the output array containing the scores of the detected corners.
///   The score is computed as the sum of the absolute difference between the pixels in the 
///   contiguous arc and the centre pixel. A higher score value indicates a stronger corner feature. 
///   For example, a corner of score 108 is stronger than a corner of score 50.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param nCornersMax
///   Maximum number of corners. The function exits when the maximum number of
///   corners is exceeded.
///
/// @param nCorners
///   pointer to an integer storing the number of corners detected
///
/// @param mask
///   Per-pixel mask for each pixel represented in input image. 
///   If a bit set to 0, pixel will be a candidate for corner detection. 
///   If a bit set to 1, pixel will be ignored.
///
/// @param maskWidth
///   Width of the mask. Both width and height of the mask must be 'k' times image width and height, 
///   where k = 1/2, 1/4 , 1/8 , 1, 2, 4 and 8.
///
/// @param maskHeight
///   Height of the mask. Both width and height of the mask must be 'k' times image width and height, 
///   where k = 1/2, 1/4 , 1/8 , 1, 2, 4 and 8.
///
/// @ingroup feature_detection
//------------------------------------------------------------------------------

FASTCV_API void
fcvCornerFast9InMaskScoreu8( const uint8_t* __restrict src,
                             unsigned int              srcWidth,
                             unsigned int              srcHeight,
                             unsigned int              srcStride,
                             int                       barrier,
                             unsigned int              border,
                             uint32_t* __restrict      xy,
                             uint32_t* __restrict      scores,
                             unsigned int              nCornersMax,
                             uint32_t* __restrict      nCorners,
                             const uint8_t* __restrict mask,
                             unsigned int              maskWidth,
                             unsigned int              maskHeight );

//---------------------------------------------------------------------------
/// @brief
///   Extracts FAST corners and scores from the image
///
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvCornerFast9Scoreu8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvCornerFast9Scoreu8,
///   \a fcvCornerFast9Scoreu8_v2 will be removed, and the current signature
///   for \a fcvCornerFast9Scoreu8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvCornerFast9Scoreu8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
/// non-maximum suppression can be enabled to reduce the number of false corners.
///
/// @param src
///   8-bit image where keypoints are detected. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width, the number of pixels in a row.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param barrier
///  FAST threshold. The threshold is used to compare difference between intensity value of 
///  the central pixel and pixels on a circle surrounding this pixel.
///
/// @param border
///   Number for pixels to ignore from top,bottom,right,left of the image
///
/// @param xy
///   Pointer to the output array cointaining the interleaved x,y position of the
///   detected corners.
///   \n\b NOTE: Remember to allocate double the size of @param nCornersMax
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param scores
///  Pointer to the output array containing the scores of the detected corners.
///  The score is computed as the sum of the absolute difference between the pixels in the 
///  contiguous arc and the centre pixel. A higher score value indicates a stronger corner feature. 
///  For example, a corner of score 108 is stronger than a corner of score 50.
///   \n\b NOTE: must be 128-bit aligned.
///   \n\b NOTE: size of buffer is @param nCornersMax
///
/// @param nCornersMax
///   Maximum number of corners. The function exits when the maximum number of
///   corners is exceeded. This number should account for the number of key points before non-maximum suppression.
///
/// @param nCorners
///   Pointer to an integer storing the number of corners detected
///
/// @param nmsEnabled
///   Enable non-maximum suppresion to prune weak key points (0=disabled, 1=enabled)
///    
/// @param tempBuf
///   Pointer to scratch buffer if nms is enabled, otherwise it can be NULL.
///   Size of buffer: (3*nCornersMax+srcHeight+1)*4 bytes
///   \n\b NOTE: must be 128-bit aligned.
///    
///    
/// @ingroup feature_detection
//------------------------------------------------------------------------------

FASTCV_API void
fcvCornerFast9Scoreu8_v2( const uint8_t* __restrict src,
                           unsigned int             srcWidth,
                           unsigned int             srcHeight,
                           unsigned int             srcStride,
                                    int             barrier,
                           unsigned int             border,
                               uint32_t* __restrict xy,
                               uint32_t* __restrict scores,
                           unsigned int             nCornersMax,
                               uint32_t* __restrict nCorners,
                               uint32_t             nmsEnabled,
                                   void* __restrict tempBuf);


//---------------------------------------------------------------------------
/// @brief
///   Extracts FAST corners and scores from the image based on the mask. The mask specifies pixels 
///   to be ignored by the detector.
///
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvCornerFast9InMaskScoreu8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvCornerFast9InMaskScoreu8,
///   \a fcvCornerFast9InMaskScoreu8_v2 will be removed, and the current signature
///   for \a fcvCornerFast9InMaskScoreu8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvCornerFast9InMaskScoreu8 when transitioning to 2.0.0.
///   \n\n
///
/// @details
/// non-maximum suppression can be enabled to reduce the number of false corners.
///
/// @param src
///   8-bit grayscale image where keypoints are detected. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width, the number of pixels in a row.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param barrier
///  FAST threshold. The threshold is used to compare difference between intensity value of 
///  the central pixel and pixels on a circle surrounding this pixel.
///
/// @param border
///   Number for pixels to ignore from top,bottom,right,left of the image
///
/// @param xy
///   Pointer to the output array cointaining the interleaved x,y position of the
///   detected corners.
///   \n\b NOTE: Remember to allocate double the size of @param nCornersMax
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param scores
///  Pointer to the output array containing the scores of the detected corners.
///  The score is computed as the sum of the absolute difference between the pixels in the 
///  contiguous arc and the centre pixel. A higher score value indicates a stronger corner feature. 
///  For example, a corner of score 108 is stronger than a corner of score 50.
///   \n\b NOTE: must be 128-bit aligned.
///   \n\b NOTE: size of buffer is @param nCornersMax
///
/// @param nCornersMax
///   Maximum number of corners. The function exits when the maximum number of
///   corners is exceeded. This number should account for the number of key points before non-maximum suppression.
///
/// @param nCorners
///   Pointer to an integer storing the number of corners detected
///
/// @param mask
///  Mask used to omit regions of the image. For allowed mask sizes refer to
///  @param maskWidth and @param maskHeight . The mask is so defined to work with multiple 
///  scales if necessary. For any pixel set to '1' in the mask, the corresponding pixel in the image
///  will be ignored.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param maskWidth
/// Width of the mask. Both width and height of the mask must be 'k' times image width and height, 
/// where k = 1/2, 1/4 , 1/8 , 1, 2, 4 and 8.
///
/// @param maskHeight
/// Height of the mask. Both width and height of the mask must be 'k' times image width and height, 
/// where k = 1/2, 1/4 , 1/8 , 1, 2, 4 and 8.
///
/// @param nmsEnabled
///   Enable non-maximum suppresion to prune weak key points (0=disabled, 1=enabled)
///    
/// @param tempBuf
///   Pointer to scratch buffer if nms is enabled, otherwise it can be NULL.
///   Size of buffer: (3*nCornersMax+srcHeight+1)*4 bytes
///   \n\b NOTE: must be 128-bit aligned.
///    
///    
/// @ingroup feature_detection
//------------------------------------------------------------------------------

FASTCV_API void
fcvCornerFast9InMaskScoreu8_v2( const uint8_t* __restrict src,
                                 unsigned int             srcWidth,
                                 unsigned int             srcHeight,
                                 unsigned int             srcStride,
                                          int             barrier,
                                 unsigned int             border,
                                     uint32_t* __restrict xy,
                                     uint32_t* __restrict scores,
                                 unsigned int             nCornersMax,
                                     uint32_t* __restrict nCorners,
                                const uint8_t* __restrict mask,
                                 unsigned int             maskWidth,
                                 unsigned int             maskHeight,
                                     uint32_t             nmsEnabled,
                                         void* __restrict tempBuf);


//---------------------------------------------------------------------------
/// @brief
///   Extracts FAST corners and scores from the image
///
/// @details
/// non-maximum suppression can be enabled to reduce the number of false corners.
///
/// @param src
///   8-bit image where keypoints are detected. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width, the number of pixels in a row.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param barrier
///  FAST threshold. The threshold is used to compare difference between intensity value of 
///  the central pixel and pixels on a circle surrounding this pixel.
///
/// @param border
///   Number for pixels to ignore from top,bottom,right,left of the image
///
/// @param xy
///   Pointer to the output array cointaining the interleaved x,y position of the
///   detected corners.
///   \n\b NOTE: Remember to allocate double the size of @param nCornersMax
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param scores
///  Pointer to the output array containing the scores of the detected corners.
///  The score is computed as the sum of the absolute difference between the pixels in the 
///  contiguous arc and the centre pixel. A higher score value indicates a stronger corner feature. 
///  For example, a corner of score 108 is stronger than a corner of score 50.
///   \n\b NOTE: must be 128-bit aligned.
///   \n\b NOTE: size of buffer is @param nCornersMax
///
/// @param nCornersMax
///  Maximum number of corners. The function exits when the maximum number of
///  corners is exceeded. This number should account for the number of key points before non-maximum suppression.
///
/// @param nCorners
///  pointer to an integer storing the number of corners detected
///
/// @param nmsEnabled
///   Enable non-maximum suppresion to prune weak key points (0=disabled, 1=enabled)
///    
/// @param tempBuf
///   Pointer to scratch buffer if nms is enabled, otherwise it can be NULL.
///   Size of buffer: (3*nCornersMax+srcHeight+1)*4 bytes
///   \n\b NOTE: must be 128-bit aligned.
///    
/// @return
///   void.
///
/// @ingroup feature_detection
//---------------------------------------------------------------------------

FASTCV_API void
fcvCornerFast10Scoreu8( const uint8_t* __restrict src,
                             uint32_t             srcWidth,
                             uint32_t             srcHeight,
                             uint32_t             srcStride,
                              int32_t             barrier,
                             uint32_t             border,
                             uint32_t* __restrict xy,
                             uint32_t* __restrict scores,
                             uint32_t             nCornersMax,
                             uint32_t* __restrict nCorners,
                             uint32_t             nmsEnabled,
                                 void* __restrict tempBuf);

//---------------------------------------------------------------------------
/// @brief
///   Extracts FAST corners and scores from the image based on the mask. The mask specifies pixels 
///   to be ignored by the detector.
///
/// @param src
///   8-bit grayscale image where keypoints are detected
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   image width
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: must be a multiple of 8.
///
/// @param barrier
///  FAST threshold. The threshold is used to compare difference between intensity value of 
///  the central pixel and pixels on a circle surrounding this pixel.
///
/// @param border
///   Number for pixels to ignore from top,bottom,right,left of the image
///
/// @param xy
///   Pointer to the output array cointaining the interleaved x,y position of the
///   detected corners.
///   \n\b NOTE: Remember to allocate double the size of @param nCornersMax
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param scores
///  Pointer to the output array containing the scores of the detected corners.
///  The score is computed as the sum of the absolute difference between the pixels in the 
///  contiguous arc and the centre pixel. A higher score value indicates a stronger corner feature. 
///  For example, a corner of score 108 is stronger than a corner of score 50.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param nCornersMax
///  Maximum number of corners. The function exits when the maximum number of
///  corners is exceeded. This number should account for the number of key points before non-maximum suppression.
///
/// @param nCorners
///  pointer to an integer storing the number of corners detected
///
/// @param mask
///  Mask used to omit regions of the image. For allowed mask sizes refer to
///  @param maskWidth and @param maskHeight . The mask is so defined to work with multiple 
///  scales if necessary. For any pixel set to '1' in the mask, the corresponding pixel in the image
///  will be ignored.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param maskWidth
/// Width of the mask. Both width and height of the mask must be 'k' times image width and height, 
/// where k = 1/2, 1/4 , 1/8 , 1, 2, 4 and 8.
///
/// @param maskHeight
/// Height of the mask. Both width and height of the mask must be 'k' times image width and height, 
/// where k = 1/2, 1/4 , 1/8 , 1, 2, 4 and 8.
///
/// @param nmsEnabled
///   Enable non-maximum suppresion to prune weak key points (0=disabled, 1=enabled)
///    
/// @param tempBuf
///   Pointer to scratch buffer if nms is enabled, otherwise it can be NULL.
///   Size of buffer: (3*nCornersMax+srcHeight+1)*4 bytes
///   \n\b NOTE: data should be 128-bit aligned.
///    
/// @return
///   void.
///
/// @ingroup feature_detection
//---------------------------------------------------------------------------

FASTCV_API void
fcvCornerFast10InMaskScoreu8( const uint8_t* __restrict src,
                                   uint32_t             srcWidth,
                                   uint32_t             srcHeight,
                                   uint32_t             srcStride,
                                    int32_t             barrier,
                                   uint32_t             border,
                                   uint32_t* __restrict xy,
                                   uint32_t* __restrict scores,
                                   uint32_t             nCornersMax,
                                   uint32_t* __restrict nCorners,
                              const uint8_t* __restrict mask,
                                   uint32_t             maskWidth,
                                   uint32_t             maskHeight,
                                   uint32_t             nmsEnabled,
                                       void* __restrict tempBuf);

// -----------------------------------------------------------------------------
/// @brief
///   Optical flow. Bitwidth optimized implementation
///
///   \n\b ATTENTION: The signature of this function will be changed to remove 
///   unused parameters when the 2.0.0 release of this library is made.  
///   Until 2.0.0, the developer should use this implementation with the expectation of
///   moving to a different signature when transitioning to 2.0.0.
///   \n\n
///    
/// @param src1
///   Input image from frame #1.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param src2
///   Input image from frame #2.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Input image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Input image height.
///
/// @param src1Pyr
///   Image Pyradmid of src1
///   \n\b WARNING: obtained by calling fcvPyramidCreateu8
///
/// @param src2Pyr
///   Image Pyradmid of src2
///   \n\b WARNING: obtained by calling fcvPyramidCreateu8
///
/// @param dx1Pyr 
///   Horizontal Sobel gradient pyramid for src1
///  \n\b NOTE: To be left NULL. In this case the function will 
///   build the pyramid internally. 
///
/// @param dy1Pyr
///  Vertical Sobel grading pyraid for src1 
///  \n\b NOTE: To be left NULL. In this case the function will 
///   build the pyramid internally. 
///
/// @param featureXY
///   Pointer to X,Y floating point, sub-pixel coordinates for features to
///   track. Stored as X,Y tuples. featureXY array storage must
///   be >= featureLen*2.
///
/// @param featureXY_out
///   Pointer to X,Y floating point, sub-pixel coordinates for tracked features
///   Stored as X,Y tuples. featureXY array storage must
///   be >= featureLen*2.
///
/// @param featureStatus
///   Pointer to integer array for status of each feature defined in
///   featureXY. featureStatus array storage must
///   be >= featureLen.
///
/// @param featureLen
///   Number of features in featuresXY and featureStatus array.
///
/// @param windowWidth
///   Width of window for optical flow searching.
///    \n\b NOTE: suggested value 5, 7 or 9
///
/// @param windowHeight
///   Height of window for optical flow searching.
///   \n\b NOTE:: suggested value 5, 7 or 9
///
/// @param maxIterations
///   Maximum number of LK iterations to perform per pyramid level.
///   \n\b NOTE: suggested value 5 or 7
///
/// @param nPyramidLevels
///   Number of pyramid levels.
///   \n\b NOTE: suggested value 3 or 4 depending on size of image
///
/// @param maxResidue
///   Maximum feature residue above which feature is declared lost.
///   \n\b NOTE: obsolete parameters, set to 0
///
/// @param minDisplacement
///   Minimum displacement solved below which iterations are stopped.
///   \n\b NOTE: obsolete parameters, set to 0
///
/// @param minEigenvalue 
///  Threshold for feature goodness. If it is  set it to 0, the check is disabled. 
///  \n\b NOTE: If good features are passed  to the function, then it is  suggested 
///  that you set it to 0 to have faster function time
///   \n\b NOTE: obsolete parameters, set to 0
///
/// @param lightingNormalized
///   if 1 Enable  lightning normalization
///   \n if 0 Disable lightning normalization
///   \n\b NOTE: obsolete parameters, set to 0
///
/// @ingroup object_detection
// -----------------------------------------------------------------------------

FASTCV_API void
fcvTrackLKOpticalFlowu8( const uint8_t* __restrict src1,
                         const uint8_t* __restrict src2,
                         int                       srcWidth,
                         int                       srcHeight,
                         const fcvPyramidLevel*    src1Pyr,
                         const fcvPyramidLevel*    src2Pyr,
                         const fcvPyramidLevel*    dx1Pyr,
                         const fcvPyramidLevel*    dy1Pyr,
                         const float*              featureXY,
                         float*                    featureXY_out,
                         int32_t*                  featureStatus,
                         int                       featureLen,
                         int                       windowWidth,                          
                         int                       windowHeight,
                         int                       maxIterations,
                         int                       nPyramidLevels,
                         float                     maxResidue,
                         float                     minDisplacement,
                         float                     minEigenvalue,
                         int                       lightingNormalized );


// -----------------------------------------------------------------------------
/// @brief
///   Optical flow.
///
///   \n\b ATTENTION: This function will be removed when the 2.0.0 release of this library
///   is made. Until 2.0.0, the developer should use this implementation with the expectation of
///   using \a fcvTrackLKOpticalFlowu8 when transitioning to 2.0.0.
///   \n\n
///
/// @param src1
///   Input image from frame #1.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param src2
///   Input image from frame #2.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Input image width.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Input image height.
///
/// @param src1Pyr
///   Image Pyradmid of src1
///   \n\b WARNING: obtained by calling fcvPyramidCreateu8
///
/// @param src2Pyr
///   Image Pyradmid of src2
///   \n\b WARNING: obtained by calling fcvPyramidCreateu8
///
/// @param dx1Pyr 
///   Horizontal Sobel gradient pyramid for src1
///  \n\b NOTE: To be left NULL. In this case the function will 
///   build the pyramid internally. 
///
/// @param dy1Pyr
///  Vertical Sobel grading pyraid for src1 
///  \n\b NOTE: To be left NULL. In this case the function will 
///   build the pyramid internally. 
///
/// @param featureXY
///   Pointer to X,Y floating point, sub-pixel coordinates for features to
///   track. Stored as X,Y tuples. featureXY array storage must
///   be >= featureLen*2.
///
/// @param featureXY_out
///   Pointer to X,Y floating point, sub-pixel coordinates for tracked features
///   Stored as X,Y tuples. featureXY array storage must
///   be >= featureLen*2.
///
/// @param featureStatus
///   Pointer to integer array for status of each feature defined in
///   featureXY. featureStatus array storage must
///   be >= featureLen.
///   \n\b NOTE: Possible status are :
///   \n    TRACKED           1
///   \n    NOT_FOUND        -1
///   \n    SMALL_DET        -2
///   \n    MAX_ITERATIONS   -3
///   \n    OUT_OF_BOUNDS    -4
///   \n    LARGE_RESIDUE    -5
///   \n    SMALL_EIGVAL     -6
///   \n    INVALID          -99
///
/// @param featureLen
///   Number of features in featuresXY and featureStatus array.
///
/// @param windowWidth
///   Width of window for optical flow searching.
///    \n\b NOTE: suggested value 5, 7 or 9
///
/// @param windowHeight
///   Height of window for optical flow searching.
///   \n\b NOTE:: suggested value 5, 7 or 9
///
/// @param maxIterations
///   Maximum number of LK iterations to perform per pyramid level.
///   \n\b NOTE: suggested value 5 or 7
///
/// @param nPyramidLevels
///   Number of pyramid levels.
///   \n\b NOTE: suggested value 3 or 4 depending on size of image
///
/// @param maxResidue
///   Maximum feature residue above which feature is declared lost.
///   \n\b NOTE: obsolete parameters, set to 0
///
/// @param minDisplacement
///   Minimum displacement solved below which iterations are stopped.
///   \n\b NOTE : Suggest that be set  to between 0.1 and 0.2, say 0.15 
///   \n\b NOTE: obsolete parameters, set to 0
///
/// @param minEigenvalue 
///  Threshold for feature goodness. If it is  set it to 0, the check is disabled. 
///  \n\b NOTE: If good features are passed  to the function, then it is  suggested 
///  that you set it to 0 to have faster function time
///   \n\b NOTE: obsolete parameters, set to 0
///
/// @param lightingNormalized
///   if 1 Enable  lightning normalization
///   \n if 0 Disable lightning normalization
///   \n\b NOTE: obsolete parameters, set to 0
///
/// @ingroup object_detection
//------------------------------------------------------------------------------

FASTCV_API void
fcvTrackLKOpticalFlowf32( const uint8_t* __restrict src1,
                          const uint8_t* __restrict src2,
                          unsigned int              srcWidth,
                          unsigned int              srcHeight,
                          const fcvPyramidLevel*    src1Pyr,
                          const fcvPyramidLevel*    src2Pyr,
                          const fcvPyramidLevel*    dx1Pyr,
                          const fcvPyramidLevel*    dy1Pyr,
                          const float*              featureXY,
                          float*                    featureXY_out,
                          int32_t*                  featureStatus,
                          int                       featureLen,
                          int                       windowWidth,
                          int                       windowHeight,
                          int                       maxIterations,
                          int                       nPyramidLevels,
                          float                     maxResidue,
                          float                     minDisplacement,
                          float                     minEigenvalue,
                          int                       lightingNormalized );


// -----------------------------------------------------------------------------
/// @brief
///    Builds an image pyramid of float32 arising from a single 
///    original image - that are successively downscaled w.r.t. the
///    pre-set levels.
///    \n\b NOTE: Memory should be deallocated using fcvPyramidDelete
///
/// @param src
///    Base image. Size of buffer is srcWidth*srcHeight*4 bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///    Width of base image
///    \n\b WARNING: must be a multiple of 2^numLevels
///
/// @param srcHeight
///    Height of base image
///    \n\b WARNING: must be a multiple of 2^numLevels
///
/// @param numLevels
///    Number of levels of  the pyramid
///
/// @param pyramid
///    Output pyramid of numLevels+1 images of the same type as src . 
///    pyramid[0] will be the same as src . pyramid[1] is the next 
///    pyramid layer, a smoothed and down-sized src , and so on.
///
/// @ingroup image_processing
//-------------------------------------------------------------------------------

FASTCV_API int
fcvPyramidCreatef32( const float* __restrict src,
                     unsigned int            srcWidth,
                     unsigned int            srcHeight,
                     unsigned int            numLevels,
                     fcvPyramidLevel*        pyramid );


// -----------------------------------------------------------------------------
/// @brief
///    Builds an image pyramid of uint8_t arising from a single 
///    original image - that are successively downscaled w.r.t. the
///    pre-set levels.
///    \n\b NOTE: Memory should be deallocated using fcvPyramidDelete
///
/// @param src
///    Base image. Size of buffer is srcWidth*srcHeight bytes.
///
/// @param srcWidth
///    Width of base image
///    \n\b WARNING: must be a multiple of 2^(numLevels-1)
///
/// @param srcHeight
///    height of base image
///    \n\b NOTE: must be a multiple of 2^(numLevels-1)
///
/// @param numLevels
///    Number of levels of  the pyramid
///
/// @param pyramid
///    Output pyramid of numLevels+1 images of the same type as src . 
///    pyramid[0] will be the same as src . pyramid[1] is the next 
///    pyramid layer, a smoothed and down-sized src , and so on.
///
/// @ingroup image_processing
//-------------------------------------------------------------------------------

FASTCV_API int
fcvPyramidCreateu8( const uint8_t* __restrict src,
                    unsigned int              srcWidth,
                    unsigned int              srcHeight,
                    unsigned int              numLevels,
                    fcvPyramidLevel*          pyramid );


// -----------------------------------------------------------------------------
/// @brief
///    Creates a gradient pyramid of int16_t from an image pyramid of uint8_t
///
/// @param imgPyr
///    Input Image Pyramid
///
/// @param dxPyr
///    Horizontal Sobel gradient pyramid
///
/// @param dyPyr
///    Verical Sobel gradient pyramid
///
/// @param numLevels
///    Number of levels in the pyramids
///
/// @ingroup image_processing
//-------------------------------------------------------------------------------

FASTCV_API int
fcvPyramidSobelGradientCreatei16( const fcvPyramidLevel* imgPyr,
                                  fcvPyramidLevel*       dxPyr,
                                  fcvPyramidLevel*       dyPyr,
                                  unsigned int           numLevels );


// -----------------------------------------------------------------------------
/// @brief
///    Creates a gradient pyramid of float32 from an image pyramid of uint8_t
///
/// @param imgPyr
///    input Image Pyramid
///
/// @param dxPyr
///    Horizontal Sobel gradient pyramid
///
/// @param dyPyr
///    Verical Sobel gradient pyramid
///
/// @param numLevels
///    Number of levels in the pyramids
///
/// @ingroup image_processing
//-------------------------------------------------------------------------------

FASTCV_API int
fcvPyramidSobelGradientCreatef32( const fcvPyramidLevel* imgPyr,
                                  fcvPyramidLevel*       dxPyr,
                                  fcvPyramidLevel*       dyPyr,
                                  unsigned int           numLevels  );


// -----------------------------------------------------------------------------
/// @brief
///    Creates a gradient pyramid of integer8 from an image pyramid of uint8_t
///
/// @param imgPyr
///    input Image Pyramid
///
/// @param dxPyr
///    Horizontal Sobel gradient pyramid
///
/// @param dyPyr
///    Verical Sobel gradient pyramid
///
/// @param numLevels
///    Number of levels in the pyramids
///
/// @ingroup image_processing
// -----------------------------------------------------------------------------

FASTCV_API int
fcvPyramidSobelGradientCreatei8( const fcvPyramidLevel* imgPyr,
                                 fcvPyramidLevel*       dxPyr,
                                 fcvPyramidLevel*       dyPyr,
                                 unsigned int           numLevels );


//------------------------------------------------------------------------------
/// @brief
///   Creates a 2D gradient image from source luminance data. This function computes
///   central differences on 3x3 neighborhood and then convolves the result with Sobel
///   kernel
///   \n
///   \n      [ -1 0 +1 ]              [ -1 -2 -1 ]
///   \n dx = [ -2 0 +2 ] * src   dy = [  0  0  0 ] * src
///   \n      [ -1 0 +1 ]              [ +1 +2 +1 ]
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvImageGradientSobelPlanars16_v3(). In the 2.0.0 release, 
///   fcvImageGradientSobelPlanars16_v3 will be renamed to fcvImageGradientSobelPlanars16
///   and the signature of fcvImageGradientSobelPlanars16 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param src
///   Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient.
///
/// @param srcHeight
///   Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8
///
///  @param dx
///   Buffer to store horizontal gradient. Must be (width)*(height) in size.
///   \n\b NOTE: a border of 1 pixel in size on top, bottom, left, and right
///   contains undefined values
///   Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param dy
///   Buffer to store vertical gradient. Must be (width)*(height) in size.
///   \n\b NOTE: a border of 1 pixel in size on top, bottom, left, and right
///   contains undefined values
///   Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageGradientSobelPlanars16( const uint8_t* __restrict  src,
                                unsigned int               srcWidth,
                                unsigned int               srcHeight,
                                unsigned int               srcStride,
                                int16_t* __restrict        dx,
                                int16_t* __restrict        dy);

//------------------------------------------------------------------------------
/// @brief
///   Creates a 2D gradient image from source luminance data. This function computes
///   central differences on 3x3 neighborhood and then convolves the result with Sobel
///   kernel
///   \n
///   \n      [ -1 0 +1 ]              [ -1 -2 -1 ]
///   \n dx = [ -2 0 +2 ] * src   dy = [  0  0  0 ] * src
///   \n      [ -1 0 +1 ]              [ +1 +2 +1 ]
/// 
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvImageGradientSobelPlanars16_v3(). In the 2.0.0 release, 
///   fcvImageGradientSobelPlanars16_v3 will be renamed to fcvImageGradientSobelPlanars16
///   and the signature of fcvImageGradientSobelPlanars16_v2 and 
///   fcvImageGradientSobelPlanars16_v3 as it appears now, will be removed.
///   \n\n
///
/// @param src
///   Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
///  @param dx
///   Buffer to store horizontal gradient. Must be (dxyStride)*(height) bytes in size.
///   \n\b NOTE: a border of 1 pixel in size on top, bottom, left, and right
///   contains undefined values
///   Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param dy
///   Buffer to store vertical gradient. Must be (dxyStride)*(height) bytes in size.
///   \n\b NOTE: a border of 1 pixel in size on top, bottom, left, and right
///   contains undefined values
///   Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
/// 
/// @param dxyStride
///   Stride (in bytes) of 'dx' and 'dy' gradient arrays.
///   \n\b NOTE: if 0, dxyStride is set as (srcWidth*sizeof(int16_t)).
///   \n\b WARNING: must be multiple of 16 (8 * 2-bytes per gradient value), and at least as much as srcWidth if not 0.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageGradientSobelPlanars16_v2( const uint8_t* __restrict  src,
                                   unsigned int               srcWidth,
                                   unsigned int               srcHeight,
                                   unsigned int               srcStride,
                                   int16_t* __restrict        dx,
                                   int16_t* __restrict        dy,
                                   unsigned int               dxyStride );


//------------------------------------------------------------------------------
/// @brief
///   Creates a 2D gradient image from source luminance data without normalization. 
///   This function computes central differences on 3x3 neighborhood and then convolves
///   the result with Sobel kernel
///   \n
///   \n      [ -1 0 +1 ]              [ -1 -2 -1 ]
///   \n dx = [ -2 0 +2 ] * src   dy = [  0  0  0 ] * src
///   \n      [ -1 0 +1 ]              [ +1 +2 +1 ]
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvImageGradientSobelPlanars16_v2() with a change in behavior: no normalization
///   at the end of the calculation.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvImageGradientSobelPlanars16,
///   \a fcvImageGradientSobelPlanars16_v2 and fcvImageGradientSobelPlanars16_v3 
///   will be removed, and the current signature for \a fcvImageGradientSobelPlanars16 
///   and fcvImageGradientSobelPlanars16_v3 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvImageGradientSobelPlanars16 when transitioning to 2.0.0.
///   \n\n
///
/// @param src
///   Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient. The number of pixels in a row.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth.
///  \n\b NOTE: must be multiple of 8.
///
///  @param dx
///   Buffer to store horizontal gradient. Must be (dxyStride)*(height) bytes in size.
///   \n\b NOTE: a border of 1 pixel in size on top, bottom, left, and right
///   contains undefined values
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dy
///   Buffer to store vertical gradient. Must be (dxyStride)*(height) bytes in size.
///   \n\b NOTE: a border of 1 pixel in size on top, bottom, left, and right
///   contains undefined values
///   \n\b NOTE: must be 128-bit aligned.
/// 
/// @param dxyStride
///   Stride (in bytes) of 'dx' and 'dy' gradient arrays, is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in the gradient arrays dx or dy. If left at 0 gradStride is default to 2 * srcWidth.
///   \n\b NOTE: must be multiple of 8.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------
FASTCV_API void
fcvImageGradientSobelPlanars16_v3( const uint8_t* __restrict  src,
                                    unsigned int              srcWidth,
                                    unsigned int              srcHeight,
                                    unsigned int              srcStride,
                                         int16_t* __restrict  dx,
                                         int16_t* __restrict  dy,
                                    unsigned int              dxyStride );

//------------------------------------------------------------------------------
/// @brief
///   Creates a 2D gradient image from source luminance data. This function computes
///   central differences on 3x3 neighborhood and then convolves the result with Sobel
///   kernel. The output is in interleaved format (i.e.) [dx][dy][dx][dy]....
///   \n
///   \n      [ -1 0 +1 ]              [ -1 -2 -1 ]
///   \n dx = [ -2 0 +2 ] * src   dy = [  0  0  0 ] * src
///   \n      [ -1 0 +1 ]              [ +1 +2 +1 ]
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvImageGradientSobelInterleaveds16_v3(). In the 2.0.0 release, 
///   fcvImageGradientSobelInterleaveds16_v3 will be renamed to fcvImageGradientSobelInterleaveds16
///   and the signature of fcvImageGradientSobelInterleaveds16 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param src
///   Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient.
///
/// @param srcHeight
///   Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8
///
/// @param gradients
///   Buffer to store horizontal and vertical gradient. Must be
///   (width-2)*(height-2) *2 in size.
///   Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageGradientSobelInterleaveds16( const uint8_t* __restrict  src,
                                     unsigned int               srcWidth,
                                     unsigned int               srcHeight,
                                     unsigned int               srcStride,
                                     int16_t* __restrict        gradients );

//------------------------------------------------------------------------------
/// @brief
///   Creates a 2D gradient image from source luminance data. This function computes
///   central differences on 3x3 neighborhood and then convolves the result with Sobel
///   kernel. The output is in interleaved format (i.e.) [dx][dy][dx][dy]....
///   \n
///   \n      [ -1 0 +1 ]              [ -1 -2 -1 ]
///   \n dx = [ -2 0 +2 ] * src   dy = [  0  0  0 ] * src
///   \n      [ -1 0 +1 ]              [ +1 +2 +1 ]
/// 
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvImageGradientSobelInterleaveds16_v3(). In the 2.0.0 release, 
///   fcvImageGradientSobelInterleaveds16_v3 will be renamed to fcvImageGradientSobelInterleaveds16
///   and the signature of fcvImageGradientSobelInterleaveds16 and 
///   fcvImageGradientSobelInterleaveds16_v2 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param src
///   Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param gradients
///   Buffer to store horizontal and vertical gradient. Must be
///   gradStride*(height-2) *2 bytes in size.
///   Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
/// 
/// @param gradStride
///   Stride (in bytes) of the interleaved gradients array.
///   \n\b NOTE: if 0, gradStride is set as (srcWidth-2)*2*sizeof(int16_t).
///   \n\b WARNING: must be multiple of 8, and at least as much as 4*srcWidth if not 0.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageGradientSobelInterleaveds16_v2( const uint8_t* __restrict  src,
                                        unsigned int               srcWidth,
                                        unsigned int               srcHeight,
                                        unsigned int               srcStride,
                                        int16_t* __restrict        gradients,
                                        unsigned int               gradStride );

// -----------------------------------------------------------------------------
/// @brief
///   Creates a 2D gradient image from source luminance data. This function computes
///   central differences on 3x3 neighborhood and then convolves the result with Sobel
///   kernel. The output is in interleaved format (i.e.) [dx][dy][dx][dy]....
///   \n
///   \n      [ -1 0 +1 ]              [ -1 -2 -1 ]
///   \n dx = [ -2 0 +2 ] * src   dy = [  0  0  0 ] * src
///   \n      [ -1 0 +1 ]              [ +1 +2 +1 ]
///   \n Compared to the original and v2 functions, this v3 functions does not normalize
///   \n the gradients (divide by 8). It just returns the actual dx, dy values.
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvImageGradientSobelInterleaveds16_v2() with a change in behavior: no
///   normalization at the end of the calculation.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvImageGradientSobelInterleaveds16,
///   \a fcvImageGradientSobelInterleaveds16_v2 and fcvImageGradientSobelInterleaveds16_v3
///   will be removed, and the current signature for \a fcvImageGradientSobelInterleaveds16 
///   and fcvImageGradientSobelInterleaveds16_v3 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvImageGradientSobelInterleaveds16 when transitioning to 2.0.0.
///   \n\n
/// @param src
///   Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient. The number of pixels in a row.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param gradients
///   Buffer to store horizontal and vertical gradient. Must be
///   gradStride*(height-2) *2 bytes in size.
///   \n\b NOTE: must be 128-bit aligned.
/// 
/// @param gradStride
///   Stride (in bytes) is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in the interleaved gradients array. If left at 0 gradStride is default to 4 * (srcWidth-2).
///   \n\b WARNING: must be multiple of 8.
///
/// @ingroup image_processing
// -----------------------------------------------------------------------------
FASTCV_API void
fcvImageGradientSobelInterleaveds16_v3( const uint8_t* __restrict  src,
                                        unsigned int               srcWidth,
                                        unsigned int               srcHeight,
                                        unsigned int               srcStride,
                                        int16_t*       __restrict  gradients,
                                        unsigned int               gradStride );

//------------------------------------------------------------------------------
/// @brief
///   Creates a 2D gradient image from source luminance data. This function computes
///   central differences on 3x3 neighborhood and then convolves the result with Sobel
///   kernel. The output is in interleaved format (i.e.) [dx][dy][dx][dy]....
///   \n
///   \n      [ -1 0 +1 ]              [ -1 -2 -1 ]
///   \n dx = [ -2 0 +2 ] * src   dy = [  0  0  0 ] * src
///   \n      [ -1 0 +1 ]              [ +1 +2 +1 ]
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvImageGradientSobelInterleavedf32_v2(). In the 2.0.0 release, 
///   fcvImageGradientSobelInterleavedf32_v2 will be renamed to fcvImageGradientSobelInterleavedf32
///   and the signature of fcvImageGradientSobelInterleavedf32 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param src
///   Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient.
///
/// @param srcHeight
///   Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8
///
/// @param gradients
///   Buffer to store horizontal and vertical gradient. Must be
///   (width-2)*(height-2) *2 floats in size.
///   Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageGradientSobelInterleavedf32( const uint8_t* __restrict src,
                                     unsigned int              srcWidth,
                                     unsigned int              srcHeight,
                                     unsigned int              srcStride,
                                     float* __restrict         gradients);

//------------------------------------------------------------------------------
/// @brief
///   Creates a 2D gradient image from source luminance data. This function computes
///   central differences on 3x3 neighborhood and then convolves the result with Sobel
///   kernel. The output is in interleaved format (i.e.) [dx][dy][dx][dy]....
///   \n
///   \n      [ -1 0 +1 ]              [ -1 -2 -1 ]
///   \n dx = [ -2 0 +2 ] * src   dy = [  0  0  0 ] * src
///   \n      [ -1 0 +1 ]              [ +1 +2 +1 ]
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvImageGradientSobelInterleavedf32() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvImageGradientSobelInterleavedf32,
///   \a fcvImageGradientSobelInterleavedf32_v2 will be removed, and the current signature
///   for \a fcvImageGradientSobelInterleavedf32 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvImageGradientSobelInterleavedf32 when transitioning to 2.0.0.
///   \n\n
///
/// @param src
///   Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @param gradients
///   Buffer to store horizontal and vertical gradient. Must be
///   gradStride*(height-2) *2 bytes in size.
///   Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
/// 
/// @param gradStride
///   Stride (in bytes) of the interleaved gradients array.
///   \n\b NOTE: if 0, gradStride is set as (srcWidth-2)*2*sizeof(float).
///   \n\b WARNING: must be multiple of 8, and at least as much as (srcWidth-2)*2*sizeof(float) if not 0.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageGradientSobelInterleavedf32_v2( const uint8_t* __restrict src,
                                        unsigned int              srcWidth,
                                        unsigned int              srcHeight,
                                        unsigned int              srcStride,
                                        float* __restrict         gradients,
                                        unsigned int              gradStride);


//------------------------------------------------------------------------------
/// @brief
///   Creates a 2D gradient image from source luminance data. This function
///   computes central differences on 3x3 neighborhood and then convolves the
///   result with Sobel kernel
///   \n
///   \n      [ -1 0 +1 ]              [ -1 -2 -1 ]
///   \n dx = [ -2 0 +2 ] * src   dy = [  0  0  0 ] * src
///   \n      [ -1 0 +1 ]              [ +1 +2 +1 ]
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvFilterGaussian3x3u8_v2(). In the 2.0.0 release, 
///   fcvImageGradientSobelPlanars8_v2 will be renamed to fcvImageGradientSobelPlanars8
///   and the signature of fcvImageGradientSobelPlanars8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param src
///   Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient.
///
/// @param srcHeight
///   Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8
///
///  @param dx
///   Buffer to store horizontal gradient. Must be (width)*(height) in size.
///   \n\b NOTE: a border of 1 pixel in size on top, bottom, left, and right
///   contains undefined values. Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param dy
///   Buffer to store vertical gradient. Must be (width)*(height) in size.
///   \n\b NOTE: a border of 1 pixel in size on top, bottom, left, and right
///   contains undefined values Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageGradientSobelPlanars8( const uint8_t* __restrict src,
                               unsigned int              srcWidth,
                               unsigned int              srcHeight,
                               unsigned int              srcStride,
                               int8_t* __restrict        dx,
                               int8_t* __restrict        dy);

//------------------------------------------------------------------------------
/// @brief
///   Creates a 2D gradient image from source luminance data. This function
///   computes central differences on 3x3 neighborhood and then convolves the
///   result with Sobel kernel
///   \n
///   \n      [ -1 0 +1 ]              [ -1 -2 -1 ]
///   \n dx = [ -2 0 +2 ] * src   dy = [  0  0  0 ] * src
///   \n      [ -1 0 +1 ]              [ +1 +2 +1 ]
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvImageGradientSobelPlanars8() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvImageGradientSobelPlanars8,
///   \a fcvImageGradientSobelPlanars8_v2 will be removed, and the current signature
///   for \a fcvImageGradientSobelPlanars8 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvImageGradientSobelPlanars8 when transitioning to 2.0.0.
///   \n\n
///
/// @param src
///   Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
///  @param dx
///   Buffer to store horizontal gradient. Must be (dxyStride)*(height) in size.
///   \n\b NOTE: a border of 1 pixel in size on top, bottom, left, and right
///   contains undefined values. Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param dy
///   Buffer to store vertical gradient. Must be (dxyStride)*(height) bytes in size.
///   \n\b NOTE: a border of 1 pixel in size on top, bottom, left, and right
///   contains undefined values. Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
/// 
/// @param dxyStride
///   Stride (in bytes) of 'dx' and 'dy' gradient arrays.
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvImageGradientSobelPlanars8_v2( const uint8_t* __restrict src,
                                  unsigned int              srcWidth,
                                  unsigned int              srcHeight,
                                  unsigned int              srcStride,
                                  int8_t* __restrict        dx,
                                  int8_t* __restrict        dy,
                                  unsigned int              dxyStride );


//------------------------------------------------------------------------------
/// @brief
///   Creates a 2D gradient image from source luminance data. This function computes
///   central differences on 3x3 neighborhood and then convolves the result with Sobel
///   kernel
///   \n
///   \n      [ -1 0 +1 ]              [ -1 -2 -1 ]
///   \n dx = [ -2 0 +2 ] * src   dy = [  0  0  0 ] * src
///   \n      [ -1 0 +1 ]              [ +1 +2 +1 ]
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvImageGradientSobelPlanarf32_v2(). In the 2.0.0 release, 
///   fcvImageGradientSobelPlanarf32_v2 will be renamed to fcvImageGradientSobelPlanarf32
///   and the signature of fcvImageGradientSobelPlanarf32 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param src
///   Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient.
///
/// @param srcHeight
///   Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8
///
///  @param dx
///   Buffer to store horizontal gradient. Must be (width)*(height) in size.
///   \n\b NOTE: a border of 1 pixel in size on top, bottom, left, and right
///   contains undefined values. Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param dy
///   Buffer to store vertical gradient. Must be (width)*(height) in size.
///   \n\b NOTE: a border of 1 pixel in size on top, bottom, left, and right
///   contains undefined values. Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------
FASTCV_API void
fcvImageGradientSobelPlanarf32( const uint8_t* __restrict  src,
                                unsigned int               srcWidth,
                                unsigned int               srcHeight,
                                unsigned int               srcStride,
                                float*                     dx,
                                float*                     dy);

//------------------------------------------------------------------------------
/// @brief
///   Creates a 2D gradient image from source luminance data. This function computes
///   central differences on 3x3 neighborhood and then convolves the result with Sobel
///   kernel
///   \n
///   \n      [ -1 0 +1 ]              [ -1 -2 -1 ]
///   \n dx = [ -2 0 +2 ] * src   dy = [  0  0  0 ] * src
///   \n      [ -1 0 +1 ]              [ +1 +2 +1 ]
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvImageGradientSobelPlanarf32() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvImageGradientSobelPlanarf32,
///   \a fcvImageGradientSobelPlanarf32_v2 will be removed, and the current signature
///   for \a fcvImageGradientSobelPlanarf32 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvImageGradientSobelPlanarf32 when transitioning to 2.0.0.
///   \n\n
///
/// @param src
///   Input image/patch. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///   \n\b WARNING: must be multiple of 8, and at least as much as srcWidth if not 0.
///
///  @param dx
///   Buffer to store horizontal gradient. Must be (dxyStride)*(height) bytes in size.
///   \n\b NOTE: a border of 1 pixel in size on top, bottom, left, and right
///   contains undefined values. Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param dy
///   Buffer to store vertical gradient. Must be (dxyStride)*(height) bytes in size.
///   \n\b NOTE: a border of 1 pixel in size on top, bottom, left, and right
///   contains undefined values. Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
/// 
/// @param dxyStride
///   Stride (in bytes) of 'dx' and 'dy' gradient arrays.
///   \n\b NOTE: if 0, dxyStride is set as 4*srcWidth.
///   \n\b WARNING: must be multiple of 32 (8 * 4-bytes per gradient value),and at least as much as srcWidth*sizeof(float) if not 0.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------
FASTCV_API void
fcvImageGradientSobelPlanarf32_v2( const uint8_t* __restrict  src,
                                   unsigned int               srcWidth,
                                   unsigned int               srcHeight,
                                   unsigned int               srcStride,
                                   float*                     dx,
                                   float*                     dy,
                                   unsigned int               dxyStride );


//------------------------------------------------------------------------------
/// @brief
///   Creates a 2D gradient image from source luminance data. This function computes
///   central differences on 3x3 neighborhood and then convolves the result with Sobel
///   kernel
///   \n
///   \n      [ -1 0 +1 ]              [ -1 -2 -1 ]
///   \n dx = [ -2 0 +2 ] * src   dy = [  0  0  0 ] * src
///   \n      [ -1 0 +1 ]              [ +1 +2 +1 ]
///   
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvImageGradientSobelPlanarf32f32_v2(). In the 2.0.0 release, 
///   fcvImageGradientSobelPlanarf32f32_v2 will be renamed to fcvImageGradientSobelPlanarf32f32
///   and the signature of fcvImageGradientSobelPlanarf32f32 as it appears now, 
///   will be removed.
///   \n\n
///
/// @param src
///   Input image/patch. Size of buffer is srcStride*srcHeight floats.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Height of src data to create gradient.
///
/// @param srcStride
///   Stride of image (i.e., how many pixels (not bytes) between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: if 0, srcStride is set as srcWidth.
///
///  @param dx
///   Buffer to store horizontal gradient. Must be (width)*(height) floats in size.
///   \n\b NOTE: a border of 1 pixel in size on top, bottom, left, and right
///   contains undefined values. Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param dy
///   Buffer to store vertical gradient. Must be (width)*(height) floats in size.
///   \n\b NOTE: a border of 1 pixel in size on top, bottom, left, and right
///   contains undefined values. Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------
FASTCV_API void
fcvImageGradientSobelPlanarf32f32( const float * __restrict  src,
                                   unsigned int              srcWidth,
                                   unsigned int              srcHeight,
                                   unsigned int              srcStride,
                                   float*                    dx,
                                   float*                    dy);


//------------------------------------------------------------------------------
/// @brief
///   Creates a 2D gradient image from source luminance data. This function computes
///   central differences on 3x3 neighborhood and then convolves the result with Sobel
///   kernel
///   \n
///   \n      [ -1 0 +1 ]              [ -1 -2 -1 ]
///   \n dx = [ -2 0 +2 ] * src   dy = [  0  0  0 ] * src
///   \n      [ -1 0 +1 ]              [ +1 +2 +1 ]
/// 
///   \n\b ATTENTION: This function is a duplication of of 
///   fcvImageGradientSobelPlanarf32f32()() with the addition of extra parameters.
///   This function has been added to allow for backward compatibility
///   with the original function.  When the 2.0.0 release of this library
///   is made, this function will be renamed to: \a fcvImageGradientSobelPlanarf32f32(),
///   \a fcvImageGradientSobelPlanarf32f32_v2 will be removed, and the current signature
///   for \a fcvImageGradientSobelPlanarf32f32 will be removed.  Until 2.0.0, the 
///   developer should use this implementation with the expectation of
///   renaming it to \a fcvImageGradientSobelPlanarf32f32 when transitioning to 2.0.0.
///   \n\n
///
/// @param src
///   Input image/patch. Size of buffer is srcStride*srcHeight floats.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param srcWidth
///   Width of src data to create gradient.
///   \n\b WARNING: must be multiple of 8.
///
/// @param srcHeight
///   Height of src data to create gradient.
///
/// @param srcStride
///   Stride (in bytes) of image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b WARNING: must be multiple of 32 (8 * 4-bytes), and at least as much as srcWidth*sizeof(float) if not 0.
///   \n\b NOTE: if 0, srcStride is set as srcWidth*4.
///
///  @param dx
///   Buffer to store horizontal gradient. Must be (dxyStride)*(height) bytes in size.
///   \n\b NOTE: a border of 1 pixel in size on top, bottom, left, and right
///   contains undefined values. Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param dy
///   Buffer to store vertical gradient. Must be (dxyStride)*(height) bytes in size.
///   \n\b NOTE: a border of 1 pixel in size on top, bottom, left, and right
///   contains undefined values. Gradient output is scaled by 1/8.
///   \n\b NOTE: data should be 128-bit aligned.
/// 
/// @param dxyStride
///   Stride (in bytes) of 'dx' and 'dy' gradient arrays.
///   \n\b WARNING: must be multiple of 32 (8 * 4-bytes per gradient value).
///   \n\b WARNING: must be multiple of 32 (8 * 4-bytes), and at least as much as srcWidth*sizeof(float) if not 0.
///   \n\b NOTE: if 0, dxyStride is set as srcWidth*4.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------
FASTCV_API void
fcvImageGradientSobelPlanarf32f32_v2( const float * __restrict  src,
                                      unsigned int              srcWidth,
                                      unsigned int              srcHeight,
                                      unsigned int              srcStride,
                                      float*                    dx,
                                      float*                    dy,
                                      unsigned int              dxyStride );


//------------------------------------------------------------------------------
/// @brief
///   Block Optical Flow 16x16 - Tracks all 16x16 blocks in the Region of Interest
///   (ROI) from Source-1 to Source-2. Generates Motion Vectors for blocks where
///   motion is detected.
/// 
/// @details
///   
/// @param[in] src1
///   Pointer to source image where the original blocks are present. 
///   \n Dimensions should be same as \a src2, and equal to \a srcWidth, 
///   \a srcHeight, \a srcStride.
///   \n\b WARNING: must be 128-bit aligned. Buffer size is srcStride*srcHeight bytes.
/// 
/// @param[in] src2
///   Pointer to second source image where motion vectors for blocks in \a img1
///   are to be located. 
///   \n Dimensions should be same as \a src1, and equal to \a srcWidth, 
///   \a srcHeight, \a srcStride.
///   \n\b WARNING: must be 128-bit aligned.
/// 
/// @param[in] srcWidth
///   Width of source images pointed by \a src1 and \a src2.
/// 
/// @param[in] srcHeight
///   Height of source images pointed by \a src1 and \a src2.
/// 
/// @param[in] srcStride
///   Stride of source images pointed by \a src1 and \a src2.
/// 
/// @param[in] roiLeft
///   Left co-ordinate (x0) of Region-of-Interest (ROI).
/// 
/// @param[in] roiTop
///   Top co-orgdinate (y0) of Region-of-Interest (ROI).
/// 
/// @param[in] roiRight
///   Right co-ordinate (x1) of Region-of-Interest (ROI).
/// 
/// @param[in] roiBottom
///   Bottom co-ordinate (y1) of Region-of-Interest (ROI).
/// 
/// @param[in] shiftSize
///   Distance in number of pixels (both horizontally and vertically) between
///   consecutive blocks for which motion vector is searched.
///   \n\b NOTE: Larger the value, less number of blocks will be tracked, and 
///   hence the function will run faster.
/// 
/// @param[in] searchWidth
///   Numbers of pixels horizontally on left and right of the source block (src2) where a
///   match is searched for. For example, if searchWidth is 8 and searchHeight 
///   is 8, then the search area for any given block will be 32x32 around
///   the location of that block.
/// 
/// @param[in] searchHeight
///   Numbers of pixels vertically on top and bottom of the source block (src2) where a
///   match is searched for. For example, if searchWidth is 8 and searchHeight 
///   is 8, then the search area for any given block will be 32x32 around
///   the location of that block.
/// 
/// @param[in] searchStep
///   Distance in number of pixels between consecutive search targets within
///   the above search window.
///   \n\b NOTE: Larger the value, more coarse the search will be and thus 
///   will make the fucntion run faster. Smaller the value, more dense the 
///   search will be, making the funciton run slower. 
/// 
/// @param[in] usePrevious
///   Indicates if the function should use the existing motion vectors in 
///   locX and locY as the starting point for motion vector search.
///   \n\b NOTE: This parameter is redundant at the moment. 
/// 
/// @param[out] numMv
///   Pointer to variable that will store the count of Motion Vectors 
///   generated by the function.
///   \n\b WARNING: This pointer should be Non-NULL.
/// 
/// @param[out] locX
///   Pointer to an array which will store the X co-ordinates of the
///   original Block for which a Motion Vector is generated.
///   \n\b NOTE: The array will contain \a numMv valid entries.
///   \n\b WARNING: This pointer should be Non-NULL, and the array size should 
///   be >= number of 16x16 blocks in ROI.
///
/// @param[out] locY
///   Pointer to an array which will store the Y co-ordinates of the
///   original Block for which a Motion Vector is generated.
///   \n\b NOTE: The array will contain \a numMv valid entries.
///   \n\b WARNING: This pointer should be Non-NULL, and the array size should 
///   be >= number of 16x16 blocks in ROI.
///
/// @param[out] mvX
///   Pointer to an array which will store the X co-ordinates of the block in \a src2 
///   corresponding block in \a src1. (\a mvX[i]-\a locX[i]) will give the motion
///   vector for the block in \a src1.
///   \n\b NOTE: The array will contain \a numMv valid entries.
///   \n\b WARNING: This pointer should be Non-NULL, and the array size should 
///   be >= number of 16x16 blocks in ROI.
///
/// @param[out] mvY
///   Pointer to an array which will store the Y co-ordinates of the block in \a src2 
///   corresponding block in \a src1. (\a mvY[i]-\a locY[i]) will give the motion
///   vector for the block in \a src1.
///   \n\b NOTE: The array will contain \a numMv valid entries.
///   \n\b WARNING: This pointer should be Non-NULL, and the array size should 
///   be >= number of 16x16 blocks in ROI.
/// 
/// @return
///    0 - Success, Failure otherwise.
/// 
/// @ingroup object_detection
//------------------------------------------------------------------------------
FASTCV_API int
fcvTrackBMOpticalFlow16x16u8( const uint8_t* __restrict   src1,
                              const uint8_t* __restrict   src2,
                              uint32_t                    srcWidth,
                              uint32_t                    srcHeight,
                              uint32_t                    srcStride,
                              uint32_t                    roiLeft,
                              uint32_t                    roiTop,
                              uint32_t                    roiRight,
                              uint32_t                    roiBottom,
                              uint32_t                    shiftSize,
                              uint32_t                    searchWidth,
                              uint32_t                    searchHeight,
                              uint32_t                    searchStep,
                              uint32_t                    usePrevious,
                              uint32_t *                  numMv,
                              uint32_t *                  locX,
                              uint32_t *                  locY,
                              uint32_t *                  mvX,
                              uint32_t *                  mvY);

//------------------------------------------------------------------------------
/// @brief
///   Performs per-element bitwise-OR operation on two 8-bit single channel images. 
///   Two images should have the same size. dst(I)=src1(I) V src2(I) if mask(I) is not zero.
///
/// @param src1
///   Pointer to the 8-bit source image 1.
///
/// @param src2
///   Pointer to the 8-bit source image 2.
///
/// @param srcWidth
///   Width of source images pointed by src1 and src2.
///
/// @param srcHeight
///   Height of source images pointed by src1 and src2.
///
/// @param srcStride
///   Stride of source images (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///
/// @param dst
///    Pointer to the 8-bit destination image.
///
/// @param dstStride
///   Stride of destination image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///
/// @param mask
///   Pointer to the 8-bit single channel mask. It specifies elements of the destination array to be changed.
///   The mask is optional. If there is no mask, the value is NULL.
///
/// @param maskStride
///   Stride of the mask (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   If there is no mask, the value is 0.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void 
fcvBitwiseOru8(const uint8_t* __restrict src1, 
								const uint8_t* __restrict src2,  
								uint32_t                  srcWidth,
								uint32_t                  srcHeight,
								uint32_t                  srcStride,
								uint8_t * __restrict      dst,
								uint32_t                  dstStride,
								uint8_t * __restrict      mask,
								uint32_t                  maskStride );

//------------------------------------------------------------------------------
/// @brief
///   Performs per-element bitwise-OR operation on two 32-bit single channel images. 
///   Two images should have the same size. dst(I)=src1(I) V src2(I) if mask(I) is not zero.
///
/// @param src1
///   Pointer to the 32-bit source image 1.
///
/// @param src2
///   Pointer to the 32-bit source image 2.
///
/// @param srcWidth
///   Width of source images pointed by src1 and src2.
///
/// @param srcHeight
///   Height of source images pointed by src1 and src2.
///
/// @param srcStride
///   Stride of source images (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///
/// @param dst
///    Pointer to the 8-bit destination image.
///
/// @param dstStride
///   Stride of destination image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///
/// @param mask
///   Pointer to the 8-bit single channel mask. It specifies elements of the destination array to be changed.
///   The mask is optional. If there is no mask, the value is NULL.
///
/// @param maskStride
///   Stride of the mask (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   If there is no mask, the value is 0.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void 
fcvBitwiseOrs32(const int32_t* __restrict src1, 
								 const int32_t* __restrict src2,  
								 uint32_t                  srcWidth,
								 uint32_t                  srcHeight,
								 uint32_t                  srcStride,
								 int32_t * __restrict      dst,
								 uint32_t                  dstStride,
								 uint8_t * __restrict      mask,
								 uint32_t                  maskStride);


//------------------------------------------------------------------------------
/// @brief
///   Converts an image from RGB space to grayscale
///
/// @details
///   
/// @param src
///   Source 8-bit image, BGR888 format (R is lowest byte for the pixel)
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Source image width.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Source image height.
///
/// @param srcStride
///   Stride of source image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   If set to 0, srcStride=srcWidth as default
///
/// @param dst
///   Destination 8-bit gray-scale image.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of destination image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   If set to 0, dstStride=srcStride as default
///
/// 
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------
FASTCV_API void
fcvColorRGB888ToGrayu8( const uint8_t* __restrict src,
                     uint32_t srcWidth,
                     uint32_t srcHeight,
                     uint32_t srcStride,
                     uint8_t* __restrict dst,
                     uint32_t  dstStride);


//------------------------------------------------------------------------------
/// @brief
///   Integral of the image tilted by 45 degrees
///
/// @details
///   Calculates the tilted integral image of an input image
///   and adds an zero-filled border on top. Left border not zero.
///   dst[i,j]=sum (src[m,n]), where n<j, abs(m-i+1) <= j-n-1
///
/// @param src
///   Source image, single channel, unsigned char type
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width.
///
/// @param srcHeight
///   Image height.
///
/// @param srcStride
///   Stride of source image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   If set to 0, srcStride is srcWidth in bytes as default
///
/// @param dst
///   Destination image of size (srcWidth+1)*(srcHeight+1)
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of destination image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void 
fcvTiltedIntegralsu8s32( const uint8_t* __restrict src,
          						        uint32_t 			 srcWidth,
                             			uint32_t 			srcHeight,
                             			uint32_t 			srcStride,
                             			int32_t* __restrict 	  dst,
                             			uint32_t 			dstStride);

//------------------------------------------------------------------------------
/// @brief
///   Performs a valid convolution of two images
///
/// @details
///   This function does convolution of two images. 
///   Values are computed for the region where one image is completely within the other image.
///   
/// @param src1
///   First source image of int16 type
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param src1Width
///   Image width.
///
/// @param src1Height
///   Image height.
///
/// @param src1Stride
///   Stride of source image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   If set to 0, srcStride is srcWidth in bytes as default
///
/// @param src2
///   Second source image of int16 type
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param src2Width
///   Image width.
///   Must meet this condition: src2Width <= src1Width
///
/// @param src2Height
///   Image height.
///   Must meet this condition: src2Height <= src1Height
///
/// @param src2Stride
///   Stride of source images (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   If set to 0, srcStride is src2Width in bytes as default
///
/// @param dst
///   Destination image of int32 type.
///   Size of destination is (src1Width-src2Width+1) x (src1Height-src2Height+1)
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of destination image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///
/// 
///
/// @ingroup image_processing
//------------------------------------------------------------------------------
FASTCV_API void
fcvConvValids16( const int16_t* __restrict src1,
                 uint32_t src1Width,
                 uint32_t src1Height,
                 uint32_t src1Stride,
                 const int16_t* __restrict src2,
                 uint32_t src2Width,
                 uint32_t src2Height,
                 uint32_t src2Stride,
                 int32_t* __restrict dst,
                 uint32_t dstStride);




//---------------------------------------------------------------------------
/// @brief
///  Function to find the bounding rectangle of a set of points.
///
/// 
/// @param [in] xy               Set of points (x,y) for which the bounding rectangle has to be found. 
///                              The points are expressed in interleaved format:  [x1,y1,x2,y2....] 
/// @param [in] numPoints        Number of points in the array.
/// @param [out] rectTopLeftX    Lower left's X value for the rectangle.
/// @param [out] rectTopLeftY    Lower Left's Y value for the rectangle;
/// @param [out] rectWidth       Width of the rectangle.
/// @param [out] rectHeight      Height of the rectangle.
/// 
/// @return FASTCV_API void      
///
/// @ingroup feature_detection
//------------------------------------------------------------------------------
FASTCV_API void 
fcvBoundingRectangle(const uint32_t * __restrict xy, uint32_t numPoints, 
                                      uint32_t * rectTopLeftX, uint32_t * rectTopLeftY,
                                      uint32_t * rectWidth, uint32_t *rectHeight);
                                                                                    

//------------------------------------------------------------------------------
/// @brief
///   Performs vertical upsampling on input Chroma data 
///
/// @details
///   This function performs vertical 1:2 upsampling on the input Chroma data.
///   The input shall be non-inteleaved planar Chroma data. The Chroma data
///   can be either Cb component or Cr component.
///   The output height is doubled after upsampling. Caller needs to pass in 
///   the output buffer large enough to hold the upsampled data.
///
/// @param src
///   Input Chroma component, either Cb or Cr
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Input width in number of Chroma pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Input height in number of Chroma lines
///
/// @param srcStride
///   Stride of input data (i.e., number of bytes between column 0 of row 0 and
///   column 0 of row 1). If left at 0, srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output Chroma data that has been upsampled vertically
///   \n\b WARNING: output height is doubled
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output data(i.e., number of bytes between column 0 of row 0 and 
///   column 0 of row 1). If left at 0, dstStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvUpsampleVerticalu8( const uint8_t* __restrict src,
                       uint32_t                  srcWidth,
                       uint32_t                  srcHeight,
                       uint32_t                  srcStride,
                       uint8_t* __restrict       dst,
                       uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Performs horizontal upsampling on input Chroma data
///
/// @details
///   This function performs horizontal 1:2 upsampling on the input Chroma data.
///   The input shall be non-interleaved planar Chroma data. The Chroma data
///   can be either Cb component or Cr component.
///   The output width is doubled after upsampling. Caller needs to pass in 
///   the output buffer large enough to hold the upsampled data.
///
/// @param src
///   Input Chroma component, either Cb or Cr
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Input width in number of Chroma pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Input height in number of Chroma lines
///
/// @param srcStride
///   Stride of input data (i.e., number of bytes between column 0 of row 0 and
///   column 0 of row 1). If left at 0, srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output Chroma data that has been upsampled horizontally
///   \n\b WARNING: output width is doubled
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output data(i.e., number of bytes between column 0 of row 0 and 
///   column 0 of row 1). If left at 0, dstStride is default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvUpsampleHorizontalu8( const uint8_t* __restrict src,
                         uint32_t                  srcWidth,
                         uint32_t                  srcHeight,
                         uint32_t                  srcStride,
                         uint8_t* __restrict       dst,
                         uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Performs both horizontal and vertical upsampling on input Chroma data
///
/// @details
///   This function performs horizontal 1:2 upsampling and vertical 1:2 
///   upsampling on the input Chroma data.
///   The input shall be non-interleaved planar Chroma data. The Chroma data
///   can be either Cb component or Cr component.
///   The output width and height are doubled after upsampling. Caller needs 
///   to pass in the output buffer large enough to hold the upsampled data.
///
/// @param src
///   Input Chroma component, either Cb or Cr
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Input width in number of Chroma pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Input height in number of Chroma lines
///
/// @param srcStride
///   Stride of input data (i.e., number of bytes between column 0 of row 0 and
///   column 0 of row 1). If left at 0, srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output Chroma data that has been upsampled both horizontally and vertically
///   \n\b WARNING: both output width and output height are doubled
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output data(i.e., number of bytes between column 0 of row 0 and 
///   column 0 of row 1). If left at 0, dstStride is default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvUpsample2Du8( const uint8_t* __restrict src,
                 uint32_t                  srcWidth,
                 uint32_t                  srcHeight,
                 uint32_t                  srcStride,
                 uint8_t* __restrict       dst,
                 uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Performs vertical upsampling on input interleaved Chroma data 
///
/// @details
///   This function performs vertical 1:2 upsampling on the input 
///   interleaved Chroma data.
///   The input shall be interleaved Chroma data in pairs of CbCr or CrCb. 
///   The output height is doubled after upsampling. Caller needs to pass in 
///   the output buffer large enough to hold the upsampled data.
///
/// @param src
///   Input interleaved Chroma data in pairs of CbCr or CrCb
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Input width in number of Chroma pairs (CbCr pair or CrCb pair)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Input height in number of Chroma lines
///
/// @param srcStride
///   Stride of input data (i.e., number of bytes between column 0 of row 0 and
///   column 0 of row 1). If left at 0, srcStride is default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output Chroma data that has been upsampled vertically
///   \n\b WARNING: output height is doubled
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output data(i.e., number of bytes between column 0 of row 0 and 
///   column 0 of row 1). If left at 0, dstStride is default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvUpsampleVerticalInterleavedu8( const uint8_t* __restrict src,
                                  uint32_t                  srcWidth,
                                  uint32_t                  srcHeight,
                                  uint32_t                  srcStride,
                                  uint8_t* __restrict       dst,
                                  uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Performs horizontal upsampling on input interleaved Chroma data
///
/// @details
///   This function performs horizontal 1:2 upsampling on the input 
///   interleaved Chroma data.
///   The input shall be interleaved Chroma data in pairs of CbCr or CrCb. 
///   The output width is doubled after upsampling. Caller needs to pass in 
///   the output buffer large enough to hold the upsampled data.
///
/// @param src
///   Input interleaved Chroma data in pairs of CbCr or CrCb
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Input width in number of Chroma pairs (CbCr pair or CrCb pair)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Input height in number of Chroma lines
///
/// @param srcStride
///   Stride of input data (i.e., number of bytes between column 0 of row 0 and
///   column 0 of row 1). If left at 0, srcStride is default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output Chroma data that has been upsampled horizontally
///   \n\b WARNING: output width is doubled
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output data(i.e., number of bytes between column 0 of row 0 and 
///   column 0 of row 1). If left at 0, dstStride is default to srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvUpsampleHorizontalInterleavedu8( const uint8_t* __restrict src,
                                    uint32_t                  srcWidth,
                                    uint32_t                  srcHeight,
                                    uint32_t                  srcStride,
                                    uint8_t* __restrict       dst,
                                    uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Performs both horizontal and vertical upsampling on input interleaved 
///   Chroma data
///
/// @details
///   This function performs horizontal 1:2 upsampling and vertical 1:2 
///   upsampling on the input interleaved Chroma data.
///   The input shall be interleaved Chroma data in pairs of CbCr or CrCb. 
///   The output width and height are doubled after upsampling. Caller needs 
///   to pass in the output buffer large enough to hold the upsampled data.
///
/// @param src
///   Input interleaved Chroma data in pairs of CbCr or CrCb
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Input width in number of Chroma pairs (CbCr pair or CrCb pair)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Input height in number of Chroma lines
///
/// @param srcStride
///   Stride of input data (i.e., number of bytes between column 0 of row 0 and
///   column 0 of row 1). If left at 0, srcStride is default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output Chroma data that has been upsampled both horizontally and vertically
///   \n\b WARNING: both output width and output height are doubled
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output data(i.e., number of bytes between column 0 of row 0 and 
///   column 0 of row 1). If left at 0, dstStride is default to srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvUpsample2DInterleavedu8( const uint8_t* __restrict src,
                            uint32_t                  srcWidth,
                            uint32_t                  srcHeight,
                            uint32_t                  srcStride,
                            uint8_t* __restrict       dst,
                            uint32_t                  dstStride );

//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB565 to YCbCr444  
///
/// @details
///   This function performs color space conversion from interleaved RGB565 to 
///   planar YCbCr444.
///
///   The input is one interleaved RGB565 plane with blue stored at the lowest 
///   address, green next then red:
///   RGB565 plane: B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3...
///
///   RGB565 pixel is arranged with 5-bit Red component, 6-bit Green component,
///   and 5-bit Blue component. One RGB565 pixel is made up of 16-bit data.
///
///   The output are three separated Y, Cb and Cr planes:
///   Y plane:  Y0  Y1  Y2  Y3 ...
///   Cb plane: Cb0 Cb1 Cb2 Cb3...
///   Cr plane: Cr0 Cr1 Cr2 Cr3...
///
/// @param src
///   The intput of interleaved RGB565 image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of RGB565 pixels (2 bytes per pixel)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of RGB565 lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Output image Y component
///   \n\b WARNING: size must match input RGB565
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstCb
///   Output image Cb component
///   \n\b WARNING: size must match input RGB565
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstCr
///   Output image Cr component
///   \n\b WARNING: size must match input RGB565
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstYStride
///   Stride of output image Y component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstYStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output image Cb component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCbStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output image Cr component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCrStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB565ToYCbCr444Planaru8( const uint8_t* __restrict src,                                  
                                  uint32_t                  srcWidth,
                                  uint32_t                  srcHeight,
                                  uint32_t                  srcStride,
                                  uint8_t* __restrict       dstY,
                                  uint8_t* __restrict       dstCb,
                                  uint8_t* __restrict       dstCr,
                                  uint32_t                  dstYStride,
                                  uint32_t                  dstCbStride,
                                  uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB565 to YCbCr422  
///
/// @details
///   This function performs color space conversion from interleaved RGB565 to 
///   planar YCbCr422.
///
///   The input is one interleaved RGB565 plane with blue stored at the lowest 
///   address, green next then red:
///   RGB565 plane: B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3...
///
///   RGB565 pixel is arranged with 5-bit Red component, 6-bit Green component,
///   and 5-bit Blue component. One RGB565 pixel is made up of 16-bit data.
///
///   The output are three separated Y, Cb and Cr planes, with Cb and Cb planes
///   horizontally sub-sampled:
///   Y plane                          : Y0  Y1  Y2  Y3 ...
///   Horizontally sub-sampled Cb plane:   Cb0     Cb1  ...
///   Horizontally sub-sampled Cr plane:   Cr0     Cr1  ...
///
/// @param src
///   The intput of interleaved RGB565 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of RGB565 pixels (2 bytes per pixel)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of RGB565 lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Output image Y component
///   \n\b WARNING: size must match input RGB565
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstCb
///   Output image Cb component that has been sub-sampled horizontally
///   \n\b WARNING: width is half of the input RGB565 image width, height is  
///   the same to the input RGB565
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstCr
///   Output image Cr component that has been sub-sampled horizontally
///   \n\b WARNING: width is half of the input RGB565 image width, height is  
///   the same to the input RGB565
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstYStride
///   Stride of output image Y component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstYStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output image Cb component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCbStride is 
///   default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output image Cr component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCrStride is 
///   default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB565ToYCbCr422Planaru8( const uint8_t* __restrict src,                                  
                                  uint32_t                  srcWidth,
                                  uint32_t                  srcHeight,
                                  uint32_t                  srcStride,
                                  uint8_t* __restrict       dstY,
                                  uint8_t* __restrict       dstCb,
                                  uint8_t* __restrict       dstCr,
                                  uint32_t                  dstYStride,
                                  uint32_t                  dstCbStride,
                                  uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB565 to YCbCr420  
///
/// @details
///   This function performs color space conversion from interleaved RGB565 to 
///   planar YCbCr420.
///
///   The input is one interleaved RGB565 plane with blue stored at the lowest 
///   address, green next then red:
///   RGB565 plane: B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3...
///
///   RGB565 pixel is arranged with 5-bit Red component, 6-bit Green component,
///   and 5-bit Blue component. One RGB565 pixel is made up of 16-bit data.
///
///   The output are three separated Y, Cb and Cr planes, with Cb and Cb planes
///   horizontally and vertically (2D) sub-sampled:
///   Y plane                : Y00  Y01  Y02  Y03 ...
///                            Y10  Y11  Y12  Y13 ... 
///   2D sub-sampled Cb plane:    Cb0     Cb1     ...
///   2D sub-sampled Cr plane:    Cr0     Cr1     ...
///
/// @param src
///   The intput of interleaved RGB565 image
///   \n\b NOTE: must be 128-bit aligned. 
///
/// @param srcWidth
///   Image width in number of RGB565 pixels (2 bytes per pixel)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of RGB565 lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Output image Y component
///   \n\b WARNING: size must match input RGB565
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstCb
///   Output image Cb component that has been sub-sampled both horizontally 
///   and vertically
///   \n\b WARNING: width and height are both half of the input RGB565 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstCr
///   Output image Cr component that has been sub-sampled both horizontally 
///   and vertically
///   \n\b WARNING: width and height are both half of the input RGB565 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstYStride
///   Stride of output image Y component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstYStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output image Cb component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCbStride is 
///   default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output image Cr component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCrStride is 
///   default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB565ToYCbCr420Planaru8( const uint8_t* __restrict src,                                  
                                  uint32_t                  srcWidth,
                                  uint32_t                  srcHeight,
                                  uint32_t                  srcStride,
                                  uint8_t* __restrict       dstY,
                                  uint8_t* __restrict       dstCb,
                                  uint8_t* __restrict       dstCr,
                                  uint32_t                  dstYStride,
                                  uint32_t                  dstCbStride,
                                  uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB888 to YCbCr444  
///
/// @details
///   This function performs color space conversion from interleaved RGB888 to 
///   planar YCbCr444.
///
///   The input is one interleaved RGB888 plane with blue stored at the lowest 
///   address, green next then red:
///   RGB888 plane: B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3...
///
///   RGB888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   and 8-bit Blue component. One RGB888 pixel is made up of 24-bit data.
///
///   The output are three separated Y, Cb and Cr planes:
///   Y plane:  Y0  Y1  Y2  Y3 ...
///   Cb plane: Cb0 Cb1 Cb2 Cb3...
///   Cr plane: Cr0 Cr1 Cr2 Cr3...
///
/// @param src
///   The intput of interleaved RGB888 image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of RGB888 pixels (3 bytes per pixel)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of RGB888 lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Output image Y component
///   \n\b WARNING: size must match input RGB888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstCb
///   Output image Cb component
///   \n\b WARNING: size must match input RGB888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstCr
///   Output image Cr component
///   \n\b WARNING: size must match input RGB888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstYStride
///   Stride of output image Y component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstYStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output image Cb component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCbStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output image Cr component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCrStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB888ToYCbCr444Planaru8( const uint8_t* __restrict src,                                  
                                  uint32_t                  srcWidth,
                                  uint32_t                  srcHeight,
                                  uint32_t                  srcStride,
                                  uint8_t* __restrict       dstY,
                                  uint8_t* __restrict       dstCb,
                                  uint8_t* __restrict       dstCr,
                                  uint32_t                  dstYStride,
                                  uint32_t                  dstCbStride,
                                  uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB888 to YCbCr422  
///
/// @details
///   This function performs color space conversion from interleaved RGB888 to 
///   planar YCbCr422.
///
///   The input is one interleaved RGB888 plane with blue stored at the lowest 
///   address, green next then red:
///   RGB888 plane: B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3...
///
///   RGB888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   and 8-bit Blue component. One RGB888 pixel is made up of 24-bit data.
///
///   The output are three separated Y, Cb and Cr planes, with Cb and Cb planes
///   horizontally sub-sampled:
///   Y plane                          : Y0  Y1  Y2  Y3 ...
///   Horizontally sub-sampled Cb plane:   Cb0     Cb1  ...
///   Horizontally sub-sampled Cr plane:   Cr0     Cr1  ...
///
/// @param src
///   The intput of interleaved RGB888 image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of RGB888 pixels (3 bytes per pixel)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of RGB888 lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Output image Y component
///   \n\b WARNING: size must match input RGB888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstCb
///   Output image Cb component that has been sub-sampled horizontally
///   \n\b WARNING: width is half of the input RGB888 image width, height is  
///   the same to the input RGB888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstCr
///   Output image Cr component that has been sub-sampled horizontally
///   \n\b WARNING: width is half of the input RGB888 image width, height is  
///   the same to the input RGB888
///   \n\b NOTE: must be 128-bit aligned.

///
/// @param dstYStride
///   Stride of output image Y component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstYStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output image Cb component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCbStride is 
///   default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output image Cr component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCrStride is 
///   default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB888ToYCbCr422Planaru8( const uint8_t* __restrict src,                                  
                                  uint32_t                  srcWidth,
                                  uint32_t                  srcHeight,
                                  uint32_t                  srcStride,
                                  uint8_t* __restrict       dstY,
                                  uint8_t* __restrict       dstCb,
                                  uint8_t* __restrict       dstCr,
                                  uint32_t                  dstYStride,
                                  uint32_t                  dstCbStride,
                                  uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB888 to YCbCr420  
///
/// @details
///   This function performs color space conversion from interleaved RGB888 to 
///   planar YCbCr420.
///
///   The input is one interleaved RGB888 plane with blue stored at the lowest 
///   address, green next then red:
///   RGB888 plane: B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3...
///
///   RGB888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   and 8-bit Blue component. One RGB888 pixel is made up of 24-bit data.
///
///   The output are three separated Y, Cb and Cr planes, with Cb and Cb planes
///   horizontally and vertically (2D) sub-sampled:
///   Y plane                : Y00  Y01  Y02  Y03 ...
///                            Y10  Y11  Y12  Y13 ... 
///   2D sub-sampled Cb plane:    Cb0     Cb1     ...
///   2D sub-sampled Cr plane:    Cr0     Cr1     ...
///
/// @param src
///   The intput of interleaved RGB888 image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of RGB888 pixels (3 bytes per pixel)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of RGB888 lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Output image Y component
///   \n\b WARNING: size must match input RGB888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstCb
///   Output image Cb component that has been sub-sampled both horizontally 
///   and vertically
///   \n\b WARNING: width and height are both half of the input RGB888 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstCr
///   Output image Cr component that has been sub-sampled both horizontally 
///   and vertically
///   \n\b WARNING: width and height are both half of the input RGB888 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstYStride
///   Stride of output image Y component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstYStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output image Cb component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCbStride is 
///   default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output image Cr component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCrStride is 
///   default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB888ToYCbCr420Planaru8( const uint8_t* __restrict src,                                  
                                  uint32_t                  srcWidth,
                                  uint32_t                  srcHeight,
                                  uint32_t                  srcStride,
                                  uint8_t* __restrict       dstY,
                                  uint8_t* __restrict       dstCb,
                                  uint8_t* __restrict       dstCr,
                                  uint32_t                  dstYStride,
                                  uint32_t                  dstCbStride,
                                  uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGBA8888 to YCbCr444  
///
/// @details
///   This function performs color space conversion from interleaved RGBA8888 to 
///   planar YCbCr444.
///
///   The input is one interleaved RGBA8888 plane with blue stored at the lowest 
///   address, green next then red:
///   RGBA8888 plane: B0 G0 R0 A0 B1 G1 R1 A1 B2 G2 R2 A2 B3 G3 R3 A3...
///
///   RGBA8888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   8-bit Blue component, and 8-bit A component. One RGBA8888 pixel is made 
///   up of 32-bit data.
///
///   The output are three separated Y, Cb and Cr planes:
///   Y plane:  Y0  Y1  Y2  Y3 ...
///   Cb plane: Cb0 Cb1 Cb2 Cb3...
///   Cr plane: Cr0 Cr1 Cr2 Cr3...
///
/// @param src
///   The intput of interleaved RGBA8888 image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of RGBA8888 pixels (4 bytes per pixel)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of RGBA8888 lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Output image Y component
///   \n\b WARNING: size must match input RGBA8888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstCb
///   Output image Cb component
///   \n\b WARNING: size must match input RGBA8888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstCr
///   Output image Cr component
///   \n\b WARNING: size must match input RGBA8888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstYStride
///   Stride of output image Y component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstYStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output image Cb component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCbStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output image Cr component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCrStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGBA8888ToYCbCr444Planaru8( const uint8_t* __restrict src,                                  
                                    uint32_t                  srcWidth,
                                    uint32_t                  srcHeight,
                                    uint32_t                  srcStride,
                                    uint8_t* __restrict       dstY,
                                    uint8_t* __restrict       dstCb,
                                    uint8_t* __restrict       dstCr,
                                    uint32_t                  dstYStride,
                                    uint32_t                  dstCbStride,
                                    uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGBA8888 to YCbCr422  
///
/// @details
///   This function performs color space conversion from interleaved RGBA8888 to 
///   planar YCbCr422.
///
///   The input is one interleaved RGBA8888 plane with blue stored at the lowest 
///   address, green next then red:
///   RGBA8888 plane: B0 G0 R0 A0 B1 G1 R1 A1 B2 G2 R2 A2 B3 G3 R3 A3...
///
///   RGBA8888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   8-bit Blue component, and 8-bit A component. One RGBA8888 pixel is made 
///   up of 32-bit data.
///
///   The output are three separated Y, Cb and Cr planes, with Cb and Cb planes
///   horizontally sub-sampled:
///   Y plane                          : Y0  Y1  Y2  Y3 ...
///   Horizontally sub-sampled Cb plane:   Cb0     Cb1  ...
///   Horizontally sub-sampled Cr plane:   Cr0     Cr1  ...
///
/// @param src
///   The intput of interleaved RGBA8888 image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of RGBA8888 pixels (4 bytes per pixel)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of RGBA8888 lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Output image Y component
///   \n\b WARNING: size must match input RGBA8888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstCb
///   Output image Cb component that has been sub-sampled horizontally
///   \n\b WARNING: width is half of the input RGBA8888 image width, height is  
///   the same to the input RGBA8888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstCr
///   Output image Cr component that has been sub-sampled horizontally
///   \n\b WARNING: width is half of the input RGBA8888 image width, height is  
///   the same to the input RGBA8888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstYStride
///   Stride of output image Y component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstYStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output image Cb component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCbStride is 
///   default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output image Cr component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCrStride is 
///   default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGBA8888ToYCbCr422Planaru8( const uint8_t* __restrict src,                                  
                                    uint32_t                  srcWidth,
                                    uint32_t                  srcHeight,
                                    uint32_t                  srcStride,
                                    uint8_t* __restrict       dstY,
                                    uint8_t* __restrict       dstCb,
                                    uint8_t* __restrict       dstCr,
                                    uint32_t                  dstYStride,
                                    uint32_t                  dstCbStride,
                                    uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGBA8888 to YCbCr420  
///
/// @details
///   This function performs color space conversion from interleaved RGBA8888 to 
///   planar YCbCr420.
///
///   The input is one interleaved RGBA8888 plane with blue stored at the lowest 
///   address, green next then red:
///   RGBA8888 plane: B0 G0 R0 A0 B1 G1 R1 A1 B2 G2 R2 A2 B3 G3 R3 A3...
///
///   RGBA8888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   8-bit Blue component, and 8-bit A component. One RGBA8888 pixel is made 
///   up of 32-bit data.
///
///   The output are three separated Y, Cb and Cr planes, with Cb and Cb planes
///   horizontally and vertically (2D) sub-sampled:
///   Y plane                : Y00  Y01  Y02  Y03 ...
///                            Y10  Y11  Y12  Y13 ... 
///   2D sub-sampled Cb plane:    Cb0     Cb1     ...
///   2D sub-sampled Cr plane:    Cr0     Cr1     ...
///
/// @param src
///   The intput of interleaved RGBA8888 image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of RGBA8888 pixels (4 bytes per pixel)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of RGBA8888 lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Output image Y component
///   \n\b WARNING: size must match input RGBA8888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstCb
///   Output image Cb component that has been sub-sampled both horizontally 
///   and vertically
///   \n\b WARNING: width and height are both half of the input RGBA8888 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstCr
///   Output image Cr component that has been sub-sampled both horizontally 
///   and vertically
///   \n\b WARNING: width and height are both half of the input RGBA8888 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstYStride
///   Stride of output image Y component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstYStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output image Cb component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCbStride is 
///   default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output image Cr component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCrStride is 
///   default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGBA8888ToYCbCr420Planaru8( const uint8_t* __restrict src,                                  
                                    uint32_t                  srcWidth,
                                    uint32_t                  srcHeight,
                                    uint32_t                  srcStride,
                                    uint8_t* __restrict       dstY,
                                    uint8_t* __restrict       dstCb,
                                    uint8_t* __restrict       dstCr,
                                    uint32_t                  dstYStride,
                                    uint32_t                  dstCbStride,
                                    uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB565 to pseudo-planar YCbCr444  
///
/// @details
///   This function performs color space conversion from interleaved RGB565 to 
///   pseudo-planar YCbCr444.
///
///   The input is one interleaved RGB565 plane with blue stored at the lowest 
///   address, green next then red:
///   RGB565 plane: B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3...
///
///   RGB565 pixel is arranged with 5-bit Red component, 6-bit Green component,
///   and 5-bit Blue component. One RGB565 pixel is made up of 16-bit data.
///
///   The output are one Y plane followed by one interleaved CbCr (or CrCb) plane:
///   Y plane          :    Y0      Y1      Y2      Y3   ...
///   Interleaved plane: Cb0 Cr0 Cb1 Cr1 Cb2 Cr2 Cb3 Cr3 ...
///
/// @param src
///   The intput of interleaved RGB565 image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of RGB565 pixels (2 bytes per pixel)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of RGB565 lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Output image Y component
///   \n\b WARNING: size must match input RGB565
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstC
///   Output image Chroma component
///   \n\b WARNING: size must match input RGB565 (Chroma width is in number 
///   of CbCr pairs)
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstYStride
///   Stride of output image Y component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstYStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output image Chroma component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCStride is 
///   default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB565ToYCbCr444PseudoPlanaru8( const uint8_t* __restrict src,                                  
                                        uint32_t                  srcWidth,
                                        uint32_t                  srcHeight,
                                        uint32_t                  srcStride,
                                        uint8_t* __restrict       dstY,
                                        uint8_t* __restrict       dstC,
                                        uint32_t                  dstYStride,
                                        uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB565 to pseudo-planar YCbCr422  
///
/// @details
///   This function performs color space conversion from interleaved RGB565 to 
///   pseudo-planar YCbCr422.
///
///   The input is one interleaved RGB565 plane with blue stored at the lowest 
///   address, green next then red:
///   RGB565 plane: B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3...
///
///   RGB565 pixel is arranged with 5-bit Red component, 6-bit Green component,
///   and 5-bit Blue component. One RGB565 pixel is made up of 16-bit data.
///
///   The output are Y plane followed by one interleaved and horizontally 
///   sub-sampled CbCr (or CrCb) plane:
///   Y plane                          : Y0  Y1  Y2  Y3  ...
///   Interleaved and sub-sampled plane: Cb0 Cr0 Cb1 Cr1 ...
///
/// @param src
///   The intput of interleaved RGB565 image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of RGB565 pixels (2 bytes per pixel)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of RGB565 lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Output image Y component
///   \n\b WARNING: size must match input RGB565
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstC
///   Output image Chroma component
///   \n\b WARNING: Chroma width in number of pairs is half of the input 
///   RGB565 image width, Chroma height in number of Chroma lines is the same 
///   to input RGB565 image.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstYStride
///   Stride of output image Y component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstYStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output image Chroma component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB565ToYCbCr422PseudoPlanaru8( const uint8_t* __restrict src,                                  
                                        uint32_t                  srcWidth,
                                        uint32_t                  srcHeight,
                                        uint32_t                  srcStride,
                                        uint8_t* __restrict       dstY,
                                        uint8_t* __restrict       dstC,
                                        uint32_t                  dstYStride,
                                        uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB565 to pseudo-planar YCbCr420  
///
/// @details
///   This function performs color space conversion from interleaved RGB565 to 
///   pseudo-planar YCbCr420.
///
///   The input is one interleaved RGB565 plane with blue stored at the lowest 
///   address, green next then red:
///   RGB565 plane: B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3...
///
///   RGB565 pixel is arranged with 5-bit Red component, 6-bit Green component,
///   and 5-bit Blue component. One RGB565 pixel is made up of 16-bit data.
///
///   The output are one Y plane followed by one interleaved and 2D (both
///   horizontally and vertically) sub-sampled CbCr (or CrCb) plane:
///   Y plane                             : Y00  Y01  Y02  Y03 ...
///                                         Y10  Y11  Y12  Y13 ...
///   Interleaved and 2D sub-sampled plane: Cb0  Cr0  Cb1  Cr1 ...
///
/// @param src
///   The intput of interleaved RGB565 image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of RGB565 pixels (2 bytes per pixel)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of RGB565 lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Output image Y component
///   \n\b WARNING: size must match input RGB565
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstC
///   Output image Chroma component
///   \n\b WARNING: Chroma width in number of pairs is half of the input 
///   RGB565 image width, Chroma height in number of Chroma lines is also half   
///   of the input RGB565 image.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstYStride
///   Stride of output image Y component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstYStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output image Chroma component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB565ToYCbCr420PseudoPlanaru8( const uint8_t* __restrict src,                                  
                                        uint32_t                  srcWidth,
                                        uint32_t                  srcHeight,
                                        uint32_t                  srcStride,
                                        uint8_t* __restrict       dstY,
                                        uint8_t* __restrict       dstC,
                                        uint32_t                  dstYStride,
                                        uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB888 to pseudo-planar YCbCr444  
///
/// @details
///   This function performs color space conversion from interleaved RGB888 to 
///   pseudo-planar YCbCr444.
///
///   The input is one interleaved RGB888 plane with blue stored at the lowest 
///   address, green next then red:
///   RGB888 plane: B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3...
///
///   RGB888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   and 8-bit Blue component. One RGB888 pixel is made up of 24-bit data.
///
///   The output are one Y plane followed by one interleaved CbCr (or CrCb) plane:
///   Y plane          :    Y0      Y1      Y2      Y3   ...
///   Interleaved plane: Cb0 Cr0 Cb1 Cr1 Cb2 Cr2 Cb3 Cr3 ...
///
/// @param src
///   The intput of interleaved RGB888 image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of RGB888 pixels (3 bytes per pixel)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of RGB888 lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Output image Y component
///   \n\b WARNING: size must match input RGB888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstC
///   Output image Chroma component
///   \n\b WARNING: size must match input RGB888 (Chroma width is in number 
///   of CbCr pairs)
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstYStride
///   Stride of output image Y component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstYStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output image Chroma component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCStride is 
///   default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB888ToYCbCr444PseudoPlanaru8( const uint8_t* __restrict src,                                  
                                        uint32_t                  srcWidth,
                                        uint32_t                  srcHeight,
                                        uint32_t                  srcStride,
                                        uint8_t* __restrict       dstY,
                                        uint8_t* __restrict       dstC,
                                        uint32_t                  dstYStride,
                                        uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB888 to pseudo-planar YCbCr422  
///
/// @details
///   This function performs color space conversion from interleaved RGB888 to 
///   pseudo-planar YCbCr422.
///
///   The input is one interleaved RGB888 plane with blue stored at the lowest 
///   address, green next then red:
///   RGB888 plane: B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3...
///
///   RGB888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   and 8-bit Blue component. One RGB888 pixel is made up of 24-bit data.
///
///   The output are Y plane followed by one interleaved and horizontally 
///   sub-sampled CbCr (or CrCb) plane:
///   Y plane                          : Y0  Y1  Y2  Y3  ...
///   Interleaved and sub-sampled plane: Cb0 Cr0 Cb1 Cr1 ...
///
/// @param src
///   The intput of interleaved RGB888 image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of RGB888 pixels (3 bytes per pixel)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of RGB888 lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Output image Y component
///   \n\b WARNING: size must match input RGB888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstC
///   Output image Chroma component
///   \n\b WARNING: Chroma width in number of pairs is half of the input 
///   RGB888 image width, Chroma height in number of Chroma lines is the same 
///   to input RGB888 image.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstYStride
///   Stride of output image Y component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstYStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output image Chroma component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB888ToYCbCr422PseudoPlanaru8( const uint8_t* __restrict src,                                  
                                        uint32_t                  srcWidth,
                                        uint32_t                  srcHeight,
                                        uint32_t                  srcStride,
                                        uint8_t* __restrict       dstY,
                                        uint8_t* __restrict       dstC,
                                        uint32_t                  dstYStride,
                                        uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB888 to pseudo-planar YCbCr420  
///
/// @details
///   This function performs color space conversion from interleaved RGB888 to 
///   pseudo-planar YCbCr420.
///
///   The input is one interleaved RGB888 plane with blue stored at the lowest 
///   address, green next then red:
///   RGB888 plane: B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3...
///
///   RGB888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   and 8-bit Blue component. One RGB888 pixel is made up of 24-bit data.
///
///   The output are one Y plane followed by one interleaved and 2D (both
///   horizontally and vertically) sub-sampled CbCr (or CrCb) plane:
///   Y plane                             : Y00  Y01  Y02  Y03 ...
///                                         Y10  Y11  Y12  Y13 ...
///   Interleaved and 2D sub-sampled plane: Cb0  Cr0  Cb1  Cr1 ...
///
/// @param src
///   The intput of interleaved RGB888 image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of RGB888 pixels (3 bytes per pixel)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of RGB888 lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Output image Y component
///   \n\b WARNING: size must match input RGB888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstC
///   Output image Chroma component
///   \n\b WARNING: Chroma width in number of pairs is half of the input 
///   RGB888 image width, Chroma height in number of Chroma lines is also half   
///   of the input RGB888 image.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstYStride
///   Stride of output image Y component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstYStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output image Chroma component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB888ToYCbCr420PseudoPlanaru8( const uint8_t* __restrict src,                                  
                                        uint32_t                  srcWidth,
                                        uint32_t                  srcHeight,
                                        uint32_t                  srcStride,
                                        uint8_t* __restrict       dstY,
                                        uint8_t* __restrict       dstC,
                                        uint32_t                  dstYStride,
                                        uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGBA8888 to pseudo-planar YCbCr444  
///
/// @details
///   This function performs color space conversion from interleaved RGBA8888 to 
///   pseudo-planar YCbCr444.
///
///   The input is one interleaved RGBA8888 plane with blue stored at the lowest 
///   address, green next then red:
///   RGBA8888 plane: B0 G0 R0 A0 B1 G1 R1 A1 B2 G2 R2 A2 B3 G3 R3 A3...
///
///   RGBA8888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   8-bit Blue component, and 8-bit A component. One RGBA8888 pixel is made 
///   up of 32-bit data.
///
///   The output are one Y plane followed by one interleaved CbCr (or CrCb) plane:
///   Y plane          :    Y0      Y1      Y2      Y3   ...
///   Interleaved plane: Cb0 Cr0 Cb1 Cr1 Cb2 Cr2 Cb3 Cr3 ...
///
/// @param src
///   The intput of interleaved RGBA8888 image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of RGBA8888 pixels (4 bytes per pixel)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of RGBA8888 lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Output image Y component
///   \n\b WARNING: size must match input RGBA8888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstC
///   Output image Chroma component
///   \n\b WARNING: size must match input RGBA8888 (Chroma width is in number 
///   of CbCr pairs)
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstYStride
///   Stride of output image Y component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstYStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output image Chroma component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCStride is 
///   default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGBA8888ToYCbCr444PseudoPlanaru8( const uint8_t* __restrict src,                                  
                                          uint32_t                  srcWidth,
                                          uint32_t                  srcHeight,
                                          uint32_t                  srcStride,
                                          uint8_t* __restrict       dstY,
                                          uint8_t* __restrict       dstC,
                                          uint32_t                  dstYStride,
                                          uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGBA8888 to pseudo-planar YCbCr422  
///
/// @details
///   This function performs color space conversion from interleaved RGBA8888 to 
///   pseudo-planar YCbCr422.
///
///   The input is one interleaved RGBA8888 plane with blue stored at the lowest 
///   address, green next then red:
///   RGBA8888 plane: B0 G0 R0 A0 B1 G1 R1 A1 B2 G2 R2 A2 B3 G3 R3 A3...
///
///   RGBA8888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   8-bit Blue component, and 8-bit A component. One RGBA8888 pixel is made 
///   up of 32-bit data.
///
///   The output are Y plane followed by one interleaved and horizontally 
///   sub-sampled CbCr (or CrCb) plane:
///   Y plane                          : Y0  Y1  Y2  Y3  ...
///   Interleaved and sub-sampled plane: Cb0 Cr0 Cb1 Cr1 ...
///
/// @param src
///   The intput of interleaved RGBA8888 image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of RGBA8888 pixels (4 bytes per pixel)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of RGBA8888 lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Output image Y component
///   \n\b WARNING: size must match input RGBA8888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstC
///   Output image Chroma component
///   \n\b WARNING: Chroma width in number of pairs is half of the input 
///   RGBA8888 image width, Chroma height in number of Chroma lines is the same 
///   to input RGBA8888 image.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstYStride
///   Stride of output image Y component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstYStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output image Chroma component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGBA8888ToYCbCr422PseudoPlanaru8( const uint8_t* __restrict src,                                  
                                          uint32_t                  srcWidth,
                                          uint32_t                  srcHeight,
                                          uint32_t                  srcStride,
                                          uint8_t* __restrict       dstY,
                                          uint8_t* __restrict       dstC,
                                          uint32_t                  dstYStride,
                                          uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGBA8888 to pseudo-planar YCbCr420  
///
/// @details
///   This function performs color space conversion from interleaved RGBA8888 to 
///   pseudo-planar YCbCr420.
///
///   The input is one interleaved RGBA8888 plane with blue stored at the lowest 
///   address, green next then red:
///   RGBA8888 plane: B0 G0 R0 A0 B1 G1 R1 A1 B2 G2 R2 A2 B3 G3 R3 A3...
///
///   RGBA8888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   8-bit Blue component, and 8-bit A component. One RGBA8888 pixel is made 
///   up of 32-bit data.
///
///   The output are one Y plane followed by one interleaved and 2D (both
///   horizontally and vertically) sub-sampled CbCr (or CrCb) plane:
///   Y plane                             : Y00  Y01  Y02  Y03 ...
///                                         Y10  Y11  Y12  Y13 ...
///   Interleaved and 2D sub-sampled plane: Cb0  Cr0  Cb1  Cr1 ...
///
/// @param src
///   The intput of interleaved RGBA8888 image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of RGBA8888 pixels (4 bytes per pixel)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of RGBA8888 lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Output image Y component
///   \n\b WARNING: size must match input RGBA8888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstC
///   Output image Chroma component
///   \n\b WARNING: Chroma width in number of pairs is half of the input 
///   RGBA8888 image width, Chroma height in number of Chroma lines is also half   
///   of the input RGBA8888 image.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstYStride
///   Stride of output image Y component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstYStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output image Chroma component (i.e., number of bytes between  
///   column 0 of row 0 and column 0 of row 1). If left at 0, dstCStride is 
///   default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGBA8888ToYCbCr420PseudoPlanaru8( const uint8_t* __restrict src,                                  
                                          uint32_t                  srcWidth,
                                          uint32_t                  srcHeight,
                                          uint32_t                  srcStride,
                                          uint8_t* __restrict       dstY,
                                          uint8_t* __restrict       dstC,
                                          uint32_t                  dstYStride,
                                          uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB565 to RGB888 or from BGR565 to BGR888  
///
/// @details
///   This function performs RGB conversion from 16-bit interleaved RGB565 to 
///   24-bit interleaved RGB888, it can be used to convert 16-bit interleaved 
///   BGR565 to 24-bit interleaved BGR888 as well.
///
/// @param src
///   Pointer to the input RGB565 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input RGB565 image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output RGB888 
///   \n\b WARNING: size must match input RGB565
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB888 image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstStride is default to 
///   srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB565ToRGB888u8( const uint8_t* __restrict src,
                          uint32_t                  srcWidth,
                          uint32_t                  srcHeight,
                          uint32_t                  srcStride,
                          uint8_t* __restrict       dst,
                          uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB565 to RGBA8888 or from BGR565 to BGRA8888 
///
/// @details
///   This function performs RGB conversion from 16-bit interleaved RGB565 to  
///   32-bit interleaved RGBA8888, it can be used to convert 16-bit interleaved  
///   BGR565 to 32-bit interleaved BGRA8888 as well. 
///
/// @param src
///   Pointer to the input RGB565 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input RGB565 image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output RGBA8888 
///   \n\b WARNING: size must match input RGB565
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGBA8888 image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstStride is default to 
///   srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB565ToRGBA8888u8( const uint8_t* __restrict src,
                            uint32_t                  srcWidth,
                            uint32_t                  srcHeight,
                            uint32_t                  srcStride,
                            uint8_t* __restrict       dst,
                            uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB565 to BGR565 or from BGR565 to RGB565  
///
/// @details
///   This function performs RGB conversion from 16-bit interleaved RGB565 to 
///   16-bit interleaved BGR565, it can be used to convert 16-bit interleaved 
///   BGR565 to 16-bit interleaved RGB565 as well.
///
/// @param src
///   Pointer to the input RGB565 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input RGB565 image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output BGR565 
///   \n\b WARNING: size must match input RGB565
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output BGR565 image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB565ToBGR565u8( const uint8_t* __restrict src,
                          uint32_t                  srcWidth,
                          uint32_t                  srcHeight,
                          uint32_t                  srcStride,
                          uint8_t* __restrict       dst,
                          uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB565 to BGR888 or from BGR565 to RGB888  
///
/// @details
///   This function performs RGB conversion from 16-bit interleaved RGB565 to 
///   24-bit interleaved BGR888 it can be used to convert 16-bit interleaved 
///   BGR565 to 24-bit interleaved RGB888 as well.
///
/// @param src
///   Pointer to the input RGB565 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input RGB565 image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output BGR888 
///   \n\b WARNING: size must match input RGB565
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output BGR888 image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstStride is default to 
///   srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB565ToBGR888u8( const uint8_t* __restrict src,
                          uint32_t                  srcWidth,
                          uint32_t                  srcHeight,
                          uint32_t                  srcStride,
                          uint8_t* __restrict       dst,
                          uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB565 to BGRA8888 or from BGR565 to RGBA8888 
///
/// @details
///   This function performs RGB conversion from 16-bit interleaved RGB565 to  
///   32-bit interleaved BGRA8888, it can be used to convert 16-bit interleaved  
///   BGR565 to 32-bit interleaved RGBA8888 as well. 
///
/// @param src
///   Pointer to the input RGB565 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input RGB565 image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output BGRA8888 
///   \n\b WARNING: size must match input RGB565
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output BGRA8888 image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstStride is default to 
///   srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB565ToBGRA8888u8( const uint8_t* __restrict src,
                            uint32_t                  srcWidth,
                            uint32_t                  srcHeight,
                            uint32_t                  srcStride,
                            uint8_t* __restrict       dst,
                            uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB888 to RGB565 or from BGR888 to BGR565  
///
/// @details
///   This function performs RGB conversion from 24-bit interleaved RGB888 to 
///   16-bit interleaved RGB565, it can be used to convert 24-bit interleaved 
///   BGR888 to 16-bit interleaved BGR565 as well. 
///
/// @param src
///   Pointer to the input RGB888 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input RGB888 image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output RGB565 
///   \n\b WARNING: size must match input RGB888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB565 image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB888ToRGB565u8( const uint8_t* __restrict src,
                          uint32_t                  srcWidth,
                          uint32_t                  srcHeight,
                          uint32_t                  srcStride,
                          uint8_t* __restrict       dst,
                          uint32_t                  dstStride );

//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB888 to RGBA8888or from BGR888 to BGRA8888  
///
/// @details
///   This function performs RGB conversion from 24-bit interleaved RGB888 to 
///   32-bit interleaved RGBA8888, it can be used to convert 24-bit interleaved  
///   BGR888 to 32-bit interleaved BGRA8888 as well. 
///
/// @param src
///   Pointer to the input RGB888 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input RGB888 image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output RGBA8888 
///   \n\b WARNING: size must match input RGB888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGBA8888 image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstStride is default to 
///   srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB888ToRGBA8888u8( const uint8_t* __restrict src,
                            uint32_t                  srcWidth,
                            uint32_t                  srcHeight,
                            uint32_t                  srcStride,
                            uint8_t* __restrict       dst,
                            uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB888 to BGR565 or from BGR888 to RGB565 
///
/// @details
///   This function performs RGB conversion from 24-bit interleaved RGB888 to 
///   16-bit interleaved BGR565, it can be used to convert 24-bit interleaved 
///   BGR888 to 16-bit interleaved RGB565 as well.
///
/// @param src
///   Pointer to the input RGB888 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input RGB888 image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output BGR565 
///   \n\b WARNING: size must match input RGB888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output BGR565 image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB888ToBGR565u8( const uint8_t* __restrict src,
                          uint32_t                  srcWidth,
                          uint32_t                  srcHeight,
                          uint32_t                  srcStride,
                          uint8_t* __restrict       dst,
                          uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB888 to BGR888 or from BGR888 to RGB888  
///
/// @details
///   This function performs RGB conversion from 24-bit interleaved RGB888 to 
///   24-bit interleaved BGR888, it can be used to convert 24-bit interleaved 
///   BGR888 to 24-bit interleaved RGB888 as well.
///
/// @param src
///   Pointer to the input RGB888 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input RGB888 image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output BGR888 
///   \n\b WARNING: size must match input RGB888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output BGR888 image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstStride is default to 
///   srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB888ToBGR888u8( const uint8_t* __restrict src,
                          uint32_t                  srcWidth,
                          uint32_t                  srcHeight,
                          uint32_t                  srcStride,
                          uint8_t* __restrict       dst,
                          uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGB888 to BGRA8888 or from BGR888 to RGBA8888 
///
/// @details
///   This function performs RGB conversion from 24-bit interleaved RGB888 to  
///   32-bit interleaved BGRA8888, it can be used to convert 24-bit interleaved  
///   BGR888 to 32-bit interleaved RGBA8888 as well. 
///
/// @param src
///   Pointer to the input RGB888 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input RGB888 image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output BGRA8888 
///   \n\b WARNING: size must match input RGB888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output BGRA8888 image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstStride is default to 
///   srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGB888ToBGRA8888u8( const uint8_t* __restrict src,
                            uint32_t                  srcWidth,
                            uint32_t                  srcHeight,
                            uint32_t                  srcStride,
                            uint8_t* __restrict       dst,
                            uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGBA8888 to RGB565 or BGRA8888 to BGR565 
///
/// @details
///   This function performs RGB conversion from 32-bit interleaved RGBA8888 to 
///   16-bit interleaved RGB565, it can be used to convert 32-bit interleaved 
///   BGRA8888 to 16-bit interleaved BGR565 as well.
///
/// @param src
///   Pointer to the input RGBA8888 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input RGBA8888 image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output RGB565 
///   \n\b WARNING: size must match input RGBA8888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB565 image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGBA8888ToRGB565u8( const uint8_t* __restrict src,
                            uint32_t                  srcWidth,
                            uint32_t                  srcHeight,
                            uint32_t                  srcStride,
                            uint8_t* __restrict       dst,
                            uint32_t                  dstStride );

//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGBA8888 to RGB888 or from BGRA8888 to BGR888 
///
/// @details
///   This function performs RGB conversion from 32-bit interleaved RGBA8888 to  
///   24-bit interleaved RGB888, it can be used to convert 32-bit interleaved 
///   BGRA8888 to 24-bit interleaved BGR888 as well. 
///
/// @param src
///   Pointer to the input RGBA8888 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input RGBA8888 image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output RGB888 
///   \n\b WARNING: size must match input RGBA8888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB888 image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstStride is default to 
///   srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGBA8888ToRGB888u8( const uint8_t* __restrict src,
                            uint32_t                  srcWidth,
                            uint32_t                  srcHeight,
                            uint32_t                  srcStride,
                            uint8_t* __restrict       dst,
                            uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGBA8888 to BGR565 or from BGRA8888 to RGB565 
///
/// @details
///   This function performs RGB conversion from 32-bit interleaved RGBA8888 to 
///   16-bit interleaved BGR565, it can be used to convert 32-bit interleaved 
///   BGRA8888 to 16-bit interleaved RGB565 as well.
///
/// @param src
///   Pointer to the input RGBA8888 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input RGBA8888 image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output BGR565 
///   \n\b WARNING: size must match input RGBA8888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output BGR565 image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGBA8888ToBGR565u8( const uint8_t* __restrict src,
                            uint32_t                  srcWidth,
                            uint32_t                  srcHeight,
                            uint32_t                  srcStride,
                            uint8_t* __restrict       dst,
                            uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGBA8888 to BGR888 or from BGRA8888 to RGB888 
///
/// @details
///   This function performs RGB conversion from 32-bit interleaved RGBA8888 to 
///   24-bit interleaved BGR888, it can be used to convert 32-bit interleaved 
///   BGRA8888 to 24-bit interleaved RGB888 as well.
///
/// @param src
///   Pointer to the input RGBA8888 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input RGBA8888 image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output BGR888 
///   \n\b WARNING: size must match input RGBA8888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output BGR888 image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstStride is default to 
///   srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGBA8888ToBGR888u8( const uint8_t* __restrict src,
                            uint32_t                  srcWidth,
                            uint32_t                  srcHeight,
                            uint32_t                  srcStride,
                            uint8_t* __restrict       dst,
                            uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGBA8888 to BGRA8888 or from BGRA8888 to RGBA8888 
///
/// @details
///   This function performs RGB conversion from 32-bit interleaved RGBA8888 to 
///   32-bit interleaved  BGRA8888, it can be used to convert 32-bit interleaved 
///   BGRA8888 to 32-bit interleaved RGBA8888 as well. 
///
/// @param src
///   Pointer to the input RGBA8888 image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input RGBA8888 image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output BGRA8888 
///   \n\b WARNING: size must match input RGBA8888
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output BGRA8888 image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstStride is default to 
///   srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGBA8888ToBGRA8888u8( const uint8_t* __restrict src,
                              uint32_t                  srcWidth,
                              uint32_t                  srcHeight,
                              uint32_t                  srcStride,
                              uint8_t* __restrict       dst,
                              uint32_t                  dstStride );

//------------------------------------------------------------------------------
/// @brief
///   Color conversion from RGBA8888 to LAB color space 
///
/// @details
///   This function performs color space conversion from interleaved RGBA8888  
///   to interleaved LAB.
///
///   The input is arranged as:
///   [B G R A B G R A ...]
///   However, A is ignored in the conversion.
///
///   The output is arragned as:
///   [L A B 0 L A B 0 L A B 0...]
///
///   Each component (B/G/R/A/L/A/B/0) are 8 bits.    
///
/// @param src
///   Input RGBA image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in the number of pixels. 
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of lines
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of LAB image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output LAB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorRGBA8888ToLABu8( const uint8_t* __restrict src,
                         uint32_t            srcWidth,
                         uint32_t            srcHeight,
                         uint32_t            srcStride,
                         uint8_t* __restrict dst,
                         uint32_t            dstStride );

//------------------------------------------------------------------------------
/// @brief
///   Color conversion from planar YCbCr444 to planar YCbCr422  
///
/// @details
///   This function performs YCbCr color space conversion.

///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCb
///   Pointer to the input Cb component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCr
///   Pointer to the input Cr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCbStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCrStride is default
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component.
///   User can set the output pointer to be the same as the input pointer, 
///   in such case the Y component won't be touched. 
///   \n\b WARNING: size must match input YCbCr444
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCb
///   Pointer to the output Cb component 
///   \n\b WARNING: The width of the output Cb component is half of the input 
///   width, height remains the same.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCr
///   Pointer to the output Cr component 
///   \n\b WARNING: The width of the output Cr component is half of the input 
///   width, height remains the same.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output Cb component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCbStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output Cr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCrStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr444PlanarToYCbCr422Planaru8( const uint8_t*            srcY,
                                          const uint8_t* __restrict srcCb,
                                          const uint8_t* __restrict srcCr,
                                          uint32_t                  srcWidth,
                                          uint32_t                  srcHeight,
                                          uint32_t                  srcYStride,
                                          uint32_t                  srcCbStride,
                                          uint32_t                  srcCrStride,
                                          uint8_t*                  dstY,
                                          uint8_t* __restrict       dstCb,
                                          uint8_t* __restrict       dstCr,
                                          uint32_t                  dstYStride,
                                          uint32_t                  dstCbStride,
                                          uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from planar YCbCr444 to planar YCbCr420  
///
/// @details
///   This function performs YCbCr color space conversion.
///   
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCb
///   Pointer to the input Cb component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCr
///   Pointer to the input Cr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCbStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCrStride is default
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component 
///   \n\b WARNING: size must match input YCbCr444
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCb
///   Pointer to the output Cb component 
///   \n\b WARNING: The width and height of the output Cb component is half of
///   the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCr
///   Pointer to the output Cr component 
///   \n\b WARNING: The width and height of the output Cr component is half of
///   the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output Cb component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCbStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output Cr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCrStride is default to
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr444PlanarToYCbCr420Planaru8( const uint8_t*            srcY,
                                          const uint8_t* __restrict srcCb,
                                          const uint8_t* __restrict srcCr,
                                          uint32_t                  srcWidth,
                                          uint32_t                  srcHeight,
                                          uint32_t                  srcYStride,
                                          uint32_t                  srcCbStride,
                                          uint32_t                  srcCrStride,
                                          uint8_t*                  dstY,
                                          uint8_t* __restrict       dstCb,
                                          uint8_t* __restrict       dstCr,
                                          uint32_t                  dstYStride,
                                          uint32_t                  dstCbStride,
                                          uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from planar YCbCr444 to pseudo planar YCbCr444  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCb
///   Pointer to the input Cb component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCr
///   Pointer to the input Cr component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCbStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCrStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component 
///   \n\b WARNING: size must match input YCbCr444
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstC
///   Pointer to the output CbCr component 
///   \n\b WARNING: The width(number of CbCr pairs) and height of the output   
///   CbCr component are the same to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, srcStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output CbCr component (i.e., number of bytes between column 0  
///   ofrow 0 and column 0 of row 1). If left at 0, dstCStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr444PlanarToYCbCr444PseudoPlanaru8( const uint8_t*            srcY,
                                                const uint8_t* __restrict srcCb,
                                                const uint8_t* __restrict srcCr,
                                                uint32_t                  srcWidth,
                                                uint32_t                  srcHeight,
                                                uint32_t                  srcYStride,
                                                uint32_t                  srcCbStride,
                                                uint32_t                  srcCrStride,
                                                uint8_t*                  dstY,
                                                uint8_t* __restrict       dstC,
                                                uint32_t                  dstYStride,
                                                uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from planar YCbCr444 to pseudo planar YCbCr422  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCb
///   Pointer to the input Cb component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCr
///   Pointer to the input Cr component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCbStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCrStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component 
///   \n\b WARNING: size must match input YCbCr444
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstC
///   Pointer to the output CbCr component 
///   \n\b WARNING: The width(number of CbCr pairs) of the output CbCr  
///   component is half of the input width, height remains the same.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output CbCr component (i.e., number of bytes between column 0  
///   ofrow 0 and column 0 of row 1). If left at 0, dstCStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr444PlanarToYCbCr422PseudoPlanaru8( const uint8_t*            srcY,
                                                const uint8_t* __restrict srcCb,
                                                const uint8_t* __restrict srcCr,
                                                uint32_t                  srcWidth,
                                                uint32_t                  srcHeight,
                                                uint32_t                  srcYStride,
                                                uint32_t                  srcCbStride,
                                                uint32_t                  srcCrStride,
                                                uint8_t*                  dstY,
                                                uint8_t* __restrict       dstC,
                                                uint32_t                  dstYStride,
                                                uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from planar YCbCr444 to pseudo planar YCbCr420  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCb
///   Pointer to the input Cb component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCr
///   Pointer to the input Cr component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCbStride is default
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCrStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component 
///   \n\b WARNING: size must match input YCbCr444
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstC
///   Pointer to the output CbCr component 
///   \n\b WARNING: The width(number of CbCr pairs) and height of the output   
///   CbCr component is half of the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output CbCr component (i.e., number of bytes between column 0  
///   ofrow 0 and column 0 of row 1). If left at 0, dstCStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr444PlanarToYCbCr420PseudoPlanaru8( const uint8_t*            srcY,
                                                const uint8_t* __restrict srcCb,
                                                const uint8_t* __restrict srcCr,
                                                uint32_t                  srcWidth,
                                                uint32_t                  srcHeight,
                                                uint32_t                  srcYStride,
                                                uint32_t                  srcCbStride,
                                                uint32_t                  srcCrStride,
                                                uint8_t*                  dstY,
                                                uint8_t* __restrict       dstC,
                                                uint32_t                  dstYStride,
                                                uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from planar YCbCr422 to planar YCbCr444  
///
/// @details
///   This function performs YCbCr color space conversion.

///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCb
///   Pointer to the input Cb component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCr
///   Pointer to the input Cr component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCbStride is default 
///   to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCrStride is default 
///   to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component 
///   \n\b WARNING: size must match input YCbCr422.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCb
///   Pointer to the output Cb component 
///   \n\b WARNING: The width and height of the output Cb component are the  
///   same to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCr
///   Pointer to the output Cr component 
///   \n\b WARNING: The width and height of the output Cr component are the  
///   same to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output Cb component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCbStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output Cr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCrStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr422PlanarToYCbCr444Planaru8( const uint8_t*            srcY,
                                          const uint8_t* __restrict srcCb,
                                          const uint8_t* __restrict srcCr,
                                          uint32_t                  srcWidth,
                                          uint32_t                  srcHeight,
                                          uint32_t                  srcYStride,
                                          uint32_t                  srcCbStride,
                                          uint32_t                  srcCrStride,
                                          uint8_t*                  dstY,
                                          uint8_t* __restrict       dstCb,
                                          uint8_t* __restrict       dstCr,
                                          uint32_t                  dstYStride,
                                          uint32_t                  dstCbStride,
                                          uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from planar YCbCr422 to planar YCbCr420  
///
/// @details
///   This function performs YCbCr color space conversion.

///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCb
///   Pointer to the input Cb component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCr
///   Pointer to the input Cr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCbStride is default 
///   to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCrStride is default 
///   to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component 
///   \n\b WARNING: size must match input YCbCr422
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCb
///   Pointer to the output Cb component 
///   \n\b WARNING: The width and height of the output Cb component is half of
///   the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCr
///   Pointer to the output Cr component 
///   \n\b WARNING: The width and height of the output Cr component is half of
///   the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output Cb component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCbStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output Cr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCrStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr422PlanarToYCbCr420Planaru8( const uint8_t*            srcY,
                                          const uint8_t* __restrict srcCb,
                                          const uint8_t* __restrict srcCr,
                                          uint32_t                  srcWidth,
                                          uint32_t                  srcHeight,
                                          uint32_t                  srcYStride,
                                          uint32_t                  srcCbStride,
                                          uint32_t                  srcCrStride,
                                          uint8_t*                  dstY,
                                          uint8_t* __restrict       dstCb,
                                          uint8_t* __restrict       dstCr,
                                          uint32_t                  dstYStride,
                                          uint32_t                  dstCbStride,
                                          uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from planar YCbCr422 to pseudo planar YCbCr444  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCb
///   Pointer to the input Cb component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCr
///   Pointer to the input Cr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCbStride is default 
///   to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCrStride is default
///   to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component 
///   \n\b WARNING: size must match input YCbCr422
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstC
///   Pointer to the output CbCr component 
///   \n\b WARNING: The width(number of CbCr pairs) and height of the output   
///   CbCr component are the same to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output CbCr component (i.e., number of bytes between column 0  
///   ofrow 0 and column 0 of row 1). If left at 0, dstCStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr422PlanarToYCbCr444PseudoPlanaru8( const uint8_t*            srcY,
                                                const uint8_t* __restrict srcCb,
                                                const uint8_t* __restrict srcCr,
                                                uint32_t                  srcWidth,
                                                uint32_t                  srcHeight,
                                                uint32_t                  srcYStride,
                                                uint32_t                  srcCbStride,
                                                uint32_t                  srcCrStride,
                                                uint8_t*                  dstY,
                                                uint8_t* __restrict       dstC,
                                                uint32_t                  dstYStride,
                                                uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from planar YCbCr422 to pseudo planar YCbCr422  
///
/// @details
///   This function performs YCbCr color space conversion.
///   
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCb
///   Pointer to the input Cb component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCr
///   Pointer to the input Cr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCbStride is default 
///   to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCrStride is default 
///   to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component 
///   \n\b WARNING: size must match input YCbCr422
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstC
///   Pointer to the output CbCr component 
///   \n\b WARNING: The width(number of CbCr pairs) of the output CbCr  
///   component is half of the input width, height remains the same.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output CbCr component (i.e., number of bytes between column 0  
///   ofrow 0 and column 0 of row 1). If left at 0, dstCStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr422PlanarToYCbCr422PseudoPlanaru8( const uint8_t*            srcY,
                                                const uint8_t* __restrict srcCb,
                                                const uint8_t* __restrict srcCr,
                                                uint32_t                  srcWidth,
                                                uint32_t                  srcHeight,
                                                uint32_t                  srcYStride,
                                                uint32_t                  srcCbStride,
                                                uint32_t                  srcCrStride,
                                                uint8_t*                  dstY,
                                                uint8_t* __restrict       dstC,
                                                uint32_t                  dstYStride,
                                                uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from planar YCbCr422 to pseudo planar YCbCr420  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCb
///   Pointer to the input Cb component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCr
///   Pointer to the input Cr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCbStride is default 
///   to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCrStride is default 
///   to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component 
///   \n\b WARNING: size must match input YCbCr422
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstC
///   Pointer to the output CbCr component 
///   \n\b WARNING: The width(number of CbCr pairs) and height of the output   
///   CbCr component is half of the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output CbCr component (i.e., number of bytes between column 0  
///   ofrow 0 and column 0 of row 1). If left at 0, dstCStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr422PlanarToYCbCr420PseudoPlanaru8( const uint8_t*            srcY,
                                                const uint8_t* __restrict srcCb,
                                                const uint8_t* __restrict srcCr,
                                                uint32_t                  srcWidth,
                                                uint32_t                  srcHeight,
                                                uint32_t                  srcYStride,
                                                uint32_t                  srcCbStride,
                                                uint32_t                  srcCrStride,
                                                uint8_t*                  dstY,
                                                uint8_t* __restrict       dstC,
                                                uint32_t                  dstYStride,
                                                uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from planar YCbCr420 to planar YCbCr444  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCb
///   Pointer to the input Cb component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCr
///   Pointer to the input Cr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCbStride is default to
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCrStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component 
///   \n\b WARNING: size must match input YCbCr420.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCb
///   Pointer to the output Cb component 
///   \n\b WARNING: The width and height of the output Cb component are the  
///   same to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCr
///   Pointer to the output Cr component 
///   \n\b WARNING: The width and height of the output Cr component are the  
///   same to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output Cb component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCbStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output Cr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCrStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr420PlanarToYCbCr444Planaru8( const uint8_t*            srcY,
                                          const uint8_t* __restrict srcCb,
                                          const uint8_t* __restrict srcCr,
                                          uint32_t                  srcWidth,
                                          uint32_t                  srcHeight,
                                          uint32_t                  srcYStride,
                                          uint32_t                  srcCbStride,
                                          uint32_t                  srcCrStride,
                                          uint8_t*                  dstY,
                                          uint8_t* __restrict       dstCb,
                                          uint8_t* __restrict       dstCr,
                                          uint32_t                  dstYStride,
                                          uint32_t                  dstCbStride,
                                          uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from planar YCbCr420 to planar YCbCr422  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCb
///   Pointer to the input Cb component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCr
///   Pointer to the input Cr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCbStride is default 
///   to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCrStride is default 
///   to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component 
///   \n\b WARNING: size must match input YCbCr420
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCb
///   Pointer to the output Cb component 
///   \n\b WARNING: The width of the output Cb component is half of the input 
///   width, height remains the same.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCr
///   Pointer to the output Cr component 
///   \n\b WARNING:  The width of the output Cr component is half of the input 
///   width, height remains the same.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output Cb component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCbStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output Cr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCrStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr420PlanarToYCbCr422Planaru8( const uint8_t*            srcY,
                                          const uint8_t* __restrict srcCb,
                                          const uint8_t* __restrict srcCr,
                                          uint32_t                  srcWidth,
                                          uint32_t                  srcHeight,
                                          uint32_t                  srcYStride,
                                          uint32_t                  srcCbStride,
                                          uint32_t                  srcCrStride,
                                          uint8_t*                  dstY,
                                          uint8_t* __restrict       dstCb,
                                          uint8_t* __restrict       dstCr,
                                          uint32_t                  dstYStride,
                                          uint32_t                  dstCbStride,
                                          uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from planar YCbCr420 to pseudo planar YCbCr444  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCb
///   Pointer to the input Cb component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCr
///   Pointer to the input Cr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCbStride is default 
///   to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCrStride is default 
///   to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component 
///   \n\b WARNING: size must match input YCbCr420
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstC
///   Pointer to the output CbCr component 
///   \n\b WARNING: The width(number of CbCr pairs) and height of the output   
///   CbCr component are the same to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output CbCr component (i.e., number of bytes between column 0  
///   ofrow 0 and column 0 of row 1). If left at 0, dstCStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr420PlanarToYCbCr444PseudoPlanaru8( const uint8_t*            srcY,
                                                const uint8_t* __restrict srcCb,
                                                const uint8_t* __restrict srcCr,
                                                uint32_t                  srcWidth,
                                                uint32_t                  srcHeight,
                                                uint32_t                  srcYStride,
                                                uint32_t                  srcCbStride,
                                                uint32_t                  srcCrStride,
                                                uint8_t*                  dstY,
                                                uint8_t* __restrict       dstC,
                                                uint32_t                  dstYStride,
                                                uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from planar YCbCr420 to pseudo planar YCbCr422  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCb
///   Pointer to the input Cb component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCr
///   Pointer to the input Cr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCbStride is default
///   to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCrStride is default
///   to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component 
///   \n\b WARNING: size must match input YCbCr420
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstC
///   Pointer to the output CbCr component 
///   \n\b WARNING: The width(number of CbCr pairs) of the output CbCr  
///   component is half of the input width, height remains the same.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output CbCr component (i.e., number of bytes between column 0  
///   ofrow 0 and column 0 of row 1). If left at 0, dstCStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr420PlanarToYCbCr422PseudoPlanaru8( const uint8_t*            srcY,
                                                const uint8_t* __restrict srcCb,
                                                const uint8_t* __restrict srcCr,
                                                uint32_t                  srcWidth,
                                                uint32_t                  srcHeight,
                                                uint32_t                  srcYStride,
                                                uint32_t                  srcCbStride,
                                                uint32_t                  srcCrStride,
                                                uint8_t*                  dstY,
                                                uint8_t* __restrict       dstC,
                                                uint32_t                  dstYStride,
                                                uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from planar YCbCr420 to pseudo planar YCbCr420  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCb
///   Pointer to the input Cb component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcCr
///   Pointer to the input Cr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCbStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCrStride is default to
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component 
///   \n\b WARNING: size must match input YCbCr420
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstC
///   Pointer to the output CbCr component 
///   \n\b WARNING: The width(number of CbCr pairs) and height of the output   
///   CbCr component is half of the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output CbCr component (i.e., number of bytes between column 0  of
///   row 0 and column 0 of row 1). If left at 0, dstCStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr420PlanarToYCbCr420PseudoPlanaru8( const uint8_t*            srcY,
                                                const uint8_t* __restrict srcCb,
                                                const uint8_t* __restrict srcCr,
                                                uint32_t                  srcWidth,
                                                uint32_t                  srcHeight,
                                                uint32_t                  srcYStride,
                                                uint32_t                  srcCbStride,
                                                uint32_t                  srcCrStride,
                                                uint8_t*                  dstY,
                                                uint8_t* __restrict       dstC,
                                                uint32_t                  dstYStride,
                                                uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo planar YCbCr444 to pseudo planar YCbCr422  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcC
///   Pointer to the input CbCr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input CbCr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCStride is default 
///   to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component.
///   User can set the output pointer to be the same as the input pointer, 
///   in such case the Y component won't be touched. 
///   \n\b WARNING: size must match input YCbCr444
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstC
///   Pointer to the output CbCr component 
///   \n\b WARNING: The width of the output CbCr component, which is the number  
///   of the CbCr pairs, is half of the input width, height remains the same.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output CbCr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCStride is default to
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr444PseudoPlanarToYCbCr422PseudoPlanaru8( const uint8_t*            srcY,
                                                      const uint8_t* __restrict srcC,
                                                      uint32_t                  srcWidth,
                                                      uint32_t                  srcHeight,
                                                      uint32_t                  srcYStride,
                                                      uint32_t                  srcCStride,
                                                      uint8_t*                  dstY,
                                                      uint8_t* __restrict       dstC,
                                                      uint32_t                  dstYStride,
                                                      uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo planar YCbCr444 to pseudo planar YCbCr420  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcC
///   Pointer to the input CbCr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input CbCr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCStride is default 
///   to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component.
///   User can set the output pointer to be the same as the input pointer, 
///   in such case the Y component won't be touched. 
///   \n\b WARNING: size must match input YCbCr444
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstC
///   Pointer to the output CbCr component 
///   \n\b WARNING: The width(the number of the CbCr pairs) and height of the 
///   output CbCr component are half of the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output CbCr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCStride is default to
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr444PseudoPlanarToYCbCr420PseudoPlanaru8( const uint8_t*            srcY,
                                                      const uint8_t* __restrict srcC,
                                                      uint32_t                  srcWidth,
                                                      uint32_t                  srcHeight,
                                                      uint32_t                  srcYStride,
                                                      uint32_t                  srcCStride,
                                                      uint8_t*                  dstY,
                                                      uint8_t* __restrict       dstC,
                                                      uint32_t                  dstYStride,
                                                      uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo planar YCbCr444 to planar YCbCr444  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcC
///   Pointer to the input CbCr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input CbCr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCStride is default 
///   to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component.
///   User can set the output pointer to be the same as the input pointer, 
///   in such case the Y component won't be touched. 
///   \n\b WARNING: size must match input YCbCr444
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCb
///   Pointer to the output Cb component 
///   \n\b WARNING: The width and height of the output Cb component are the 
///   same to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCr
///   Pointer to the output Cr component 
///   \n\b WARNING: The width and height of the output Cr component are the 
///   same to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output Cb component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCbStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output Cr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCrStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr444PseudoPlanarToYCbCr444Planaru8( const uint8_t*            srcY,
                                                const uint8_t* __restrict srcC,
                                                uint32_t                  srcWidth,
                                                uint32_t                  srcHeight,
                                                uint32_t                  srcYStride,
                                                uint32_t                  srcCStride,
                                                uint8_t*                  dstY,
                                                uint8_t* __restrict       dstCb,
                                                uint8_t* __restrict       dstCr,
                                                uint32_t                  dstYStride,
                                                uint32_t                  dstCbStride,
                                                uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo planar YCbCr444 to planar YCbCr422  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcC
///   Pointer to the input CbCr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input CbCr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCStride is default 
///   to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component.
///   User can set the output pointer to be the same as the input pointer, 
///   in such case the Y component won't be touched. 
///   \n\b WARNING: size must match input YCbCr444
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCb
///   Pointer to the output Cb component 
///   \n\b WARNING: The width of the output Cb component is half to the input 
///   width, the output height remains the same.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCr
///   Pointer to the output Cr component 
///   \n\b WARNING: The width of the output Cr component is half to the input 
///   width, the output height remains the same.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output Cb component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCbStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output Cr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCrStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr444PseudoPlanarToYCbCr422Planaru8( const uint8_t*            srcY,
                                                const uint8_t* __restrict srcC,
                                                uint32_t                  srcWidth,
                                                uint32_t                  srcHeight,
                                                uint32_t                  srcYStride,
                                                uint32_t                  srcCStride,
                                                uint8_t*                  dstY,
                                                uint8_t* __restrict       dstCb,
                                                uint8_t* __restrict       dstCr,
                                                uint32_t                  dstYStride,
                                                uint32_t                  dstCbStride,
                                                uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo planar YCbCr444 to planar YCbCr420  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcC
///   Pointer to the input CbCr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input CbCr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCStride is default 
///   to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component.
///   User can set the output pointer to be the same as the input pointer, 
///   in such case the Y component won't be touched. 
///   \n\b WARNING: size must match input YCbCr444
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCb
///   Pointer to the output Cb component 
///   \n\b WARNING: The width and height of the output Cb component is half  
///   to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCr
///   Pointer to the output Cr component 
///   \n\b WARNING: The width and height of the output Cr component is half  
///   to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output Cb component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCbStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output Cr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCrStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr444PseudoPlanarToYCbCr420Planaru8( const uint8_t*            srcY,
                                                const uint8_t* __restrict srcC,
                                                uint32_t                  srcWidth,
                                                uint32_t                  srcHeight,
                                                uint32_t                  srcYStride,
                                                uint32_t                  srcCStride,
                                                uint8_t*                  dstY,
                                                uint8_t* __restrict       dstCb,
                                                uint8_t* __restrict       dstCr,
                                                uint32_t                  dstYStride,
                                                uint32_t                  dstCbStride,
                                                uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo planar YCbCr422 to pseudo planar YCbCr444  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcC
///   Pointer to the input CbCr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input CbCr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component.
///   User can set the output pointer to be the same as the input pointer, 
///   in such case the Y component won't be touched. 
///   \n\b WARNING: size must match input YCbCr422
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstC
///   Pointer to the output CbCr component 
///   \n\b WARNING: The width and height of the output CbCr component are the  
///   same to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output CbCr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCStride is default to
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr422PseudoPlanarToYCbCr444PseudoPlanaru8( const uint8_t*            srcY,
                                                      const uint8_t* __restrict srcC,
                                                      uint32_t                  srcWidth,
                                                      uint32_t                  srcHeight,
                                                      uint32_t                  srcYStride,
                                                      uint32_t                  srcCStride,
                                                      uint8_t*                  dstY,
                                                      uint8_t* __restrict       dstC,
                                                      uint32_t                  dstYStride,
                                                      uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo planar YCbCr422 to pseudo planar YCbCr420  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcC
///   Pointer to the input CbCr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input CbCr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component.
///   User can set the output pointer to be the same as the input pointer, 
///   in such case the Y component won't be touched. 
///   \n\b WARNING: size must match input YCbCr422
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstC
///   Pointer to the output CbCr component 
///   \n\b WARNING: The width(the number of CbCr pairs) and height of the 
///   output CbCr component are half of the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output CbCr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr422PseudoPlanarToYCbCr420PseudoPlanaru8( const uint8_t*            srcY,
                                                      const uint8_t* __restrict srcC,
                                                      uint32_t                  srcWidth,
                                                      uint32_t                  srcHeight,
                                                      uint32_t                  srcYStride,
                                                      uint32_t                  srcCStride,
                                                      uint8_t*                  dstY,
                                                      uint8_t* __restrict       dstC,
                                                      uint32_t                  dstYStride,
                                                      uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo planar YCbCr422 to planar YCbCr444  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcC
///   Pointer to the input CbCr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input CbCr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component.
///   User can set the output pointer to be the same as the input pointer, 
///   in such case the Y component won't be touched. 
///   \n\b WARNING: size must match input YCbCr422
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCb
///   Pointer to the output Cb component 
///   \n\b WARNING: The width and height of the output Cb component are the 
///   same to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCr
///   Pointer to the output Cr component 
///   \n\b WARNING: The width and height of the output Cr component are the 
///   same to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output Cb component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCbStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output Cr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCrStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr422PseudoPlanarToYCbCr444Planaru8( const uint8_t*            srcY,
                                                const uint8_t* __restrict srcC,
                                                uint32_t                  srcWidth,
                                                uint32_t                  srcHeight,
                                                uint32_t                  srcYStride,
                                                uint32_t                  srcCStride,
                                                uint8_t*                  dstY,
                                                uint8_t* __restrict       dstCb,
                                                uint8_t* __restrict       dstCr,
                                                uint32_t                  dstYStride,
                                                uint32_t                  dstCbStride,
                                                uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo planar YCbCr422 to planar YCbCr422  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcC
///   Pointer to the input CbCr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input CbCr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component.
///   User can set the output pointer to be the same as the input pointer, 
///   in such case the Y component won't be touched. 
///   \n\b WARNING: size must match input YCbCr422
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCb
///   Pointer to the output Cb component 
///   \n\b WARNING: The width of the output Cb component is half to the input 
///   width, the output height remains the same.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCr
///   Pointer to the output Cr component 
///   \n\b WARNING: The width of the output Cr component is half to the input 
///   width, the output height remains the same.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output Cb component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCbStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output Cr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCrStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr422PseudoPlanarToYCbCr422Planaru8( const uint8_t*            srcY,
                                                const uint8_t* __restrict srcC,
                                                uint32_t                  srcWidth,
                                                uint32_t                  srcHeight,
                                                uint32_t                  srcYStride,
                                                uint32_t                  srcCStride,
                                                uint8_t*                  dstY,
                                                uint8_t* __restrict       dstCb,
                                                uint8_t* __restrict       dstCr,
                                                uint32_t                  dstYStride,
                                                uint32_t                  dstCbStride,
                                                uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo planar YCbCr422 to planar YCbCr420  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcC
///   Pointer to the input CbCr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input CbCr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, dstCStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component.
///   User can set the output pointer to be the same as the input pointer, 
///   in such case the Y component won't be touched. 
///   \n\b WARNING: size must match input YCbCr422
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCb
///   Pointer to the output Cb component 
///   \n\b WARNING: The width and height of the output Cb component are half  
///   to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCr
///   Pointer to the output Cr component 
///   \n\b WARNING: The width and height of the output Cr component are half  
///   to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output Cb component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCbStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output Cr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCrStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr422PseudoPlanarToYCbCr420Planaru8( const uint8_t*            srcY,
                                                const uint8_t* __restrict srcC,
                                                uint32_t                  srcWidth,
                                                uint32_t                  srcHeight,
                                                uint32_t                  srcYStride,
                                                uint32_t                  srcCStride,
                                                uint8_t*                  dstY,
                                                uint8_t* __restrict       dstCb,
                                                uint8_t* __restrict       dstCr,
                                                uint32_t                  dstYStride,
                                                uint32_t                  dstCbStride,
                                                uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo planar YCbCr420 to pseudo planar YCbCr444  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcC
///   Pointer to the input CbCr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input CbCr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component.
///   User can set the output pointer to be the same as the input pointer, 
///   in such case the Y component won't be touched. 
///   \n\b WARNING: size must match input YCbCr420
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstC
///   Pointer to the output CbCr component 
///   \n\b WARNING: The width and height of the output CbCr component are the  
///   same to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output CbCr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCStride is default to 
///   srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr420PseudoPlanarToYCbCr444PseudoPlanaru8( const uint8_t*            srcY,
                                                      const uint8_t* __restrict srcC,
                                                      uint32_t                  srcWidth,
                                                      uint32_t                  srcHeight,
                                                      uint32_t                  srcYStride,
                                                      uint32_t                  srcCStride,
                                                      uint8_t*                  dstY,
                                                      uint8_t* __restrict       dstC,
                                                      uint32_t                  dstYStride,
                                                      uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo planar YCbCr420 to pseudo planar YCbCr422  
///
/// @details
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcC
///   Pointer to the input CbCr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input CbCr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component.
///   User can set the output pointer to be the same as the input pointer, 
///   in such case the Y component won't be touched. 
///   \n\b WARNING: size must match input YCbCr420
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstC
///   Pointer to the output CbCr component 
///   \n\b WARNING: The width and height of the output CbCr component are the  
///   same to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCStride
///   Stride of output CbCr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr420PseudoPlanarToYCbCr422PseudoPlanaru8( const uint8_t*            srcY,
                                                      const uint8_t* __restrict srcC,
                                                      uint32_t                  srcWidth,
                                                      uint32_t                  srcHeight,
                                                      uint32_t                  srcYStride,
                                                      uint32_t                  srcCStride,
                                                      uint8_t*                  dstY,
                                                      uint8_t* __restrict       dstC,
                                                      uint32_t                  dstYStride,
                                                      uint32_t                  dstCStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo planar YCbCr420 to planar YCbCr444  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcC
///   Pointer to the input CbCr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input CbCr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component.
///   User can set the output pointer to be the same as the input pointer, 
///   in such case the Y component won't be touched. 
///   \n\b WARNING: size must match input YCbCr420
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCb
///   Pointer to the output Cb component 
///   \n\b WARNING: The width and height of the output Cb component are the 
///   same to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCr
///   Pointer to the output Cr component 
///   \n\b WARNING: The width and height of the output Cr component are the 
///   same to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output Cb component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCbStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output Cr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCrStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr420PseudoPlanarToYCbCr444Planaru8( const uint8_t*            srcY,
                                                const uint8_t* __restrict srcC,
                                                uint32_t                  srcWidth,
                                                uint32_t                  srcHeight,
                                                uint32_t                  srcYStride,
                                                uint32_t                  srcCStride,
                                                uint8_t*                  dstY,
                                                uint8_t* __restrict       dstCb,
                                                uint8_t* __restrict       dstCr,
                                                uint32_t                  dstYStride,
                                                uint32_t                  dstCbStride,
                                                uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo planar YCbCr420 to planar YCbCr422  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcC
///   Pointer to the input CbCr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input CbCr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component.
///   User can set the output pointer to be the same as the input pointer, 
///   in such case the Y component won't be touched. 
///   \n\b WARNING: size must match input YCbCr420
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCb
///   Pointer to the output Cb component 
///   \n\b WARNING: The width of the output Cb component is half to the input 
///   width, the output height remains the same.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCr
///   Pointer to the output Cr component 
///   \n\b WARNING: The width of the output Cr component is half to the input 
///   width, the output height remains the same.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output Cb component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCbStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output Cr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCrStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr420PseudoPlanarToYCbCr422Planaru8( const uint8_t*            srcY,
                                                const uint8_t* __restrict srcC,
                                                uint32_t                  srcWidth,
                                                uint32_t                  srcHeight,
                                                uint32_t                  srcYStride,
                                                uint32_t                  srcCStride,
                                                uint8_t*                  dstY,
                                                uint8_t* __restrict       dstCb,
                                                uint8_t* __restrict       dstCr,
                                                uint32_t                  dstYStride,
                                                uint32_t                  dstCbStride,
                                                uint32_t                  dstCrStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo planar YCbCr420 to planar YCbCr420  
///
/// @details
///   This function performs YCbCr color space conversion.
///   User can specify the destination Y pointer to be the same as the source
///   Y pointer, in such case, 
///   the source and destination Y components share the same allocated memory.
///
/// @param srcY
///   Pointer to the input Y component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcC
///   Pointer to the input CbCr component
///   \n\b NOTE: must be 128-bit aligned
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcYStride
///   Stride of input Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcYStride is default 
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input CbCr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). If left at 0, srcCStride is default
///   to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstY
///   Pointer to the output Y component.
///   User can set the output pointer to be the same as the input pointer, 
///   in such case the Y component won't be touched. 
///   \n\b WARNING: size must match input YCbCr420
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCb
///   Pointer to the output Cb component 
///   \n\b WARNING: The width and height of the output Cb component is half  
///   to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstCr
///   Pointer to the output Cr component 
///   \n\b WARNING: The width and height of the output Cr component is half  
///   to the input width and height.
///   \n\b NOTE: must be 128-bit aligned
///
/// @param dstYStride
///   Stride of output Y component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstYStride is default to 
///   srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCbStride
///   Stride of output Cb component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCbStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstCrStride
///   Stride of output Cr component (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). If left at 0, dstCrStride is default to 
///   srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr420PseudoPlanarToYCbCr420Planaru8( const uint8_t*            srcY,
                                                const uint8_t* __restrict srcC,
                                                uint32_t                  srcWidth,
                                                uint32_t                  srcHeight,
                                                uint32_t                  srcYStride,
                                                uint32_t                  srcCStride,
                                                uint8_t*                  dstY,
                                                uint8_t* __restrict       dstCb,
                                                uint8_t* __restrict       dstCr,
                                                uint32_t                  dstYStride,
                                                uint32_t                  dstCbStride,
                                                uint32_t                  dstCrStride );

//------------------------------------------------------------------------------
/// @brief
///   Color conversion from YCbCr444 to RGB565  
///
/// @details
///   This function performs color space conversion from YCbCr444 to RGB565.
///
///   The input are three separated Y, Cb and Cr planes:
///   Y plane:  Y0  Y1  Y2  Y3 ...
///   Cb plane: Cb0 Cb1 Cb2 Cb3...
///   Cr plane: Cr0 Cr1 Cr2 Cr3...
///
///   The output is one interleaved RGB565 plane:
///   RGB565 plane: R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3...
///
///   RGB565 pixel is arranged with 5-bit Red component, 6-bit Green component,
///   and 5-bit Blue component. One RGB565 pixel is made up of 16-bit data.
///
/// @param srcY
///   Input image Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCb
///   Input image Cb component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCr
///   Input image Cr component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of Y pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of Y lines
///
/// @param srcYStride
///   Stride of input image Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input image Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcCbStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input image Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcCrStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of interleaved RGB565 image 
///   \n\b WARNING: size must match input YCbCr444
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr444PlanarToRGB565u8( const uint8_t* __restrict srcY,
                                  const uint8_t* __restrict srcCb,
                                  const uint8_t* __restrict srcCr,
                                  uint32_t                  srcWidth,
                                  uint32_t                  srcHeight,
                                  uint32_t                  srcYStride,
                                  uint32_t                  srcCbStride,
                                  uint32_t                  srcCrStride,
                                  uint8_t* __restrict       dst,
                                  uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from YCbCr444 to RGB888  
///
/// @details
///   This function performs color space conversion from YCbCr444 to RGB888.
///
///   The input are three separated Y, Cb and Cr planes:
///   Y plane:  Y0  Y1  Y2  Y3 ...
///   Cb plane: Cb0 Cb1 Cb2 Cb3...
///   Cr plane: Cr0 Cr1 Cr2 Cr3...
///
///   The output is one interleaved RGB888 plane:
///   RGB888 plane: R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3...
///
///   RGB888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   and 8-bit Blue component. One RGB888 pixel is made up of 24-bit data.  
///
/// @param srcY
///   Input image Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCb
///   Input image Cb component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCr
///   Input image Cr component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of Y pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of Y lines
///
/// @param srcYStride
///   Stride of input image Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input image Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcCbStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input image Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcCrStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of interleaved RGB888 image  
///   \n\b WARNING: size must match input YCbCr444
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr444PlanarToRGB888u8( const uint8_t* __restrict srcY,
                                  const uint8_t* __restrict srcCb,
                                  const uint8_t* __restrict srcCr,
                                  uint32_t                  srcWidth,
                                  uint32_t                  srcHeight,
                                  uint32_t                  srcYStride,
                                  uint32_t                  srcCbStride,
                                  uint32_t                  srcCrStride,
                                  uint8_t* __restrict       dst,
                                  uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from YCbCr444 to RGBA8888  
///
/// @details
///   This function performs color space conversion from YCbCr444 to RGBA8888.
///
///   The input are three separated Y, Cb and Cr planes:
///   Y plane:  Y0  Y1  Y2  Y3 ...
///   Cb plane: Cb0 Cb1 Cb2 Cb3...
///   Cr plane: Cr0 Cr1 Cr2 Cr3...
///
///   The output is one interleaved RGBA8888 plane:
///   RGBA8888 plane: R0 G0 B0 A0 R1 G1 B1 A1 R2 G2 B2 A2 R3 G3 B3 A3...
///
///   RGBA8888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   8-bit Blue component, and 8-bit A component. One RGBA8888 pixel is made 
///   up of 32-bit data.
///   
///
/// @param srcY
///   Input image Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCb
///   Input image Cb component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCr
///   Input image Cr component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of Y pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of Y lines
///
/// @param srcYStride
///   Stride of input image Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input image Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcCbStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input image Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcCrStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of interleaved RGBA8888 image 
///   \n\b WARNING: size must match input YCbCr 444
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * 4
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr444PlanarToRGBA8888u8( const uint8_t* __restrict srcY,
                                    const uint8_t* __restrict srcCb,
                                    const uint8_t* __restrict srcCr,
                                    uint32_t                  srcWidth,
                                    uint32_t                  srcHeight,
                                    uint32_t                  srcYStride,
                                    uint32_t                  srcCbStride,
                                    uint32_t                  srcCrStride,
                                    uint8_t* __restrict       dst,
                                    uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from YCbCr422 to RGB565  
///
/// @details
///   This function performs color space conversion from YCbCr422 to RGB565.
///
///   The input are three separated Y, Cb and Cr planes, with horizontally
///   sub-sampled Cb and Cr planes:
///   Y plane                          : Y0  Y1  Y2  Y3 ...
///   Horizontally sub-sampled Cb plane:   Cb0     Cb1  ...
///   Horizontally sub-sampled Cr plane:   Cr0     Cr1  ...
///
///   The output is one interleaved RGB565 plane:
///   RGB565 plane: R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3...
///
///   RGB565 pixel is arranged with 5-bit Red component, 6-bit Green component,
///   and 5-bit Blue component. One RGB565 pixel is made up of 16-bit data.
///   
///
/// @param srcY
///   Input image Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCb
///   Input image Cb component that has been sub-sampled horizontally
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCr
///   Input image Cr component that has been sub-sampled horizontally
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of Y pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of Y lines
///
/// @param srcYStride
///   Stride of input image Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input image Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcCbStride is default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input image Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcCrStride is default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of interleaved RGB565 image 
///   \n\b WARNING: size must match input YCbCr422
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr422PlanarToRGB565u8( const uint8_t* __restrict srcY,
                                  const uint8_t* __restrict srcCb,
                                  const uint8_t* __restrict srcCr,
                                  uint32_t                  srcWidth,
                                  uint32_t                  srcHeight,
                                  uint32_t                  srcYStride,
                                  uint32_t                  srcCbStride,
                                  uint32_t                  srcCrStride,
                                  uint8_t* __restrict       dst,
                                  uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from YCbCr422 to RGB888  
///
/// @details
///   This function performs color space conversion from YCbCr422 to RGB888.
///
///   The input are three separated Y, Cb and Cr planes, with horizontally
///   sub-sampled Cb and Cr planes:
///   Y plane                          : Y0  Y1  Y2  Y3 ...
///   Horizontally sub-sampled Cb plane:   Cb0     Cb1  ...
///   Horizontally sub-sampled Cr plane:   Cr0     Cr1  ...
///
///   The output is one interleaved RGB888 plane:
///   RGB888 plane: R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3...
///
///   RGB888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   and 8-bit Blue component. One RGB888 pixel is made up of 24-bit data.
///   
///
/// @param srcY
///   Input image Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCb
///   Input image Cb component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCr
///   Input image Cr component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of Y pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of Y lines
///
/// @param srcYStride
///   Stride of input image Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input image Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcCbStride is default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input image Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcCrStride is default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of interleaved RGB888 image  
///   \n\b WARNING: size must match input YCbCr422
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr422PlanarToRGB888u8( const uint8_t* __restrict srcY,
                                  const uint8_t* __restrict srcCb,
                                  const uint8_t* __restrict srcCr,
                                  uint32_t                  srcWidth,
                                  uint32_t                  srcHeight,
                                  uint32_t                  srcYStride,
                                  uint32_t                  srcCbStride,
                                  uint32_t                  srcCrStride,
                                  uint8_t* __restrict       dst,
                                  uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from YCbCr422 to RGBA8888  
///
/// @details
///   This function performs color space conversion from YCbCr422 to RGBA8888.
///
///   The input are three separated Y, Cb and Cr planes, with horizontally
///   sub-sampled Cb and Cr planes:
///   Y plane                          : Y0  Y1  Y2  Y3 ...
///   Horizontally sub-sampled Cb plane:   Cb0     Cb1  ...
///   Horizontally sub-sampled Cr plane:   Cr0     Cr1  ...
///
///   The output is one interleaved RGBA8888 plane:
///   RGBA8888 plane: R0 G0 B0 A0 R1 G1 B1 A1 R2 G2 B2 A2 R3 G3 B3 A3...
///
///   RGBA8888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   8-bit Blue component, and 8-bit A component. One RGBA8888 pixel is made 
///   up of 32-bit data.
///   
///
/// @param srcY
///   Input image Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCb
///   Input image Cb component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCr
///   Input image Cr component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of Y pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of Y lines
///
/// @param srcYStride
///   Stride of input image Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input image Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcCbStride is default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input image Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcCrStride is default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of interleaved RGBA8888 image  
///   \n\b WARNING: size must match input YCbCr422
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr422PlanarToRGBA8888u8( const uint8_t* __restrict srcY,
                                    const uint8_t* __restrict srcCb,
                                    const uint8_t* __restrict srcCr,
                                    uint32_t                  srcWidth,
                                    uint32_t                  srcHeight,
                                    uint32_t                  srcYStride,
                                    uint32_t                  srcCbStride,
                                    uint32_t                  srcCrStride,
                                    uint8_t* __restrict       dst,
                                    uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from YCbCr420 to RGB565  
///
/// @details
///   This function performs color space conversion from YCbCr420 to RGB565.
///
///   The input are three separated Y, Cb and Cr planes, with horizontally
///   and vertically (2D) sub-sampled Cb and Cr planes:
///   Y plane                : Y00  Y01  Y02  Y03 ...
///                            Y10  Y11  Y12  Y13 ... 
///   2D sub-sampled Cb plane:    Cb0     Cb1     ...
///   2D sub-sampled Cr plane:    Cr0     Cr1     ...
///
///   The output is one interleaved RGB565 plane:
///   RGB565 plane: R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3...
///
///   RGB565 pixel is arranged with 5-bit Red component, 6-bit Green component,
///   and 5-bit Blue component. One RGB565 pixel is made up of 16-bit data.
///   
///
/// @param srcY
///   Input image Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCb
///   Input image Cb component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCr
///   Input image Cr component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of Y pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of Y lines
///
/// @param srcYStride
///   Stride of input image Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input image Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcCbStride is default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input image Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcCrStride is default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of interleaved RGB565 image 
///   \n\b WARNING: size must match input YCbCr420
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr420PlanarToRGB565u8( const uint8_t* __restrict srcY,
                                  const uint8_t* __restrict srcCb,
                                  const uint8_t* __restrict srcCr,
                                  uint32_t                  srcWidth,
                                  uint32_t                  srcHeight,
                                  uint32_t                  srcYStride,
                                  uint32_t                  srcCbStride,
                                  uint32_t                  srcCrStride,
                                  uint8_t* __restrict       dst,
                                  uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from YCbCr420 to RGB888  
///
/// @details
///   This function performs color space conversion from YCbCr420 to RGB888.
///
///   The input are three separated Y, Cb and Cr planes, with horizontally
///   and vertically (2D) sub-sampled Cb and Cr planes:
///   Y plane                : Y00  Y01  Y02  Y03 ...
///                            Y10  Y11  Y12  Y13 ... 
///   2D sub-sampled Cb plane:    Cb0     Cb1     ...
///   2D sub-sampled Cr plane:    Cr0     Cr1     ...
///
///   The output is one interleaved RGB888 plane:
///   RGB888 plane: R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3...
///
///   RGB888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   and 8-bit Blue component. One RGB888 pixel is made up of 24-bit data.
///   
///
/// @param srcY
///   Input image Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCb
///   Input image Cb component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCr
///   Input image Cr component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of Y pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of Y lines
///
/// @param srcYStride
///   Stride of input image Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input image Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcCbStride is default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input image Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcCrStride is default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of interleaved RGB888 image 
///   \n\b WARNING: size must match input YCbCr420
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr420PlanarToRGB888u8( const uint8_t* __restrict srcY,
                                  const uint8_t* __restrict srcCb,
                                  const uint8_t* __restrict srcCr,
                                  uint32_t                  srcWidth,
                                  uint32_t                  srcHeight,
                                  uint32_t                  srcYStride,
                                  uint32_t                  srcCbStride,
                                  uint32_t                  srcCrStride,
                                  uint8_t* __restrict       dst,
                                  uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from YCbCr420 to RGBA8888  
///
/// @details
///   This function performs color space conversion from YCbCr420 to RGBA8888.
///
///   The input are three separated Y, Cb and Cr planes, with horizontally
///   and vertically (2D) sub-sampled Cb and Cr planes:
///   Y plane                : Y00  Y01  Y02  Y03 ...
///                            Y10  Y11  Y12  Y13 ... 
///   2D sub-sampled Cb plane:    Cb0     Cb1     ...
///   2D sub-sampled Cr plane:    Cr0     Cr1     ...
///
///   The output is one interleaved RGBA8888 plane:
///   RGBA8888 plane: R0 G0 B0 A0 R1 G1 B1 A1 R2 G2 B2 A2 R3 G3 B3 A3...
///
///   RGBA8888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   8-bit Blue component, and 8-bit A component. One RGBA8888 pixel is made 
///   up of 32-bit data.
///   
///
/// @param srcY
///   Input image Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCb
///   Input image Cb component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcCr
///   Input image Cr component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of Y pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of Y lines
///
/// @param srcYStride
///   Stride of input image Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   Stride of input image Cb component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcCbStride is default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   Stride of input image Cr component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcCrStride is default to srcWidth / 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of interleaved RGBA8888 image  
///   \n\b WARNING: size must match input YCbCr 420
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr420PlanarToRGBA8888u8( const uint8_t* __restrict srcY,
                                    const uint8_t* __restrict srcCb,
                                    const uint8_t* __restrict srcCr,
                                    uint32_t                  srcWidth,
                                    uint32_t                  srcHeight,
                                    uint32_t                  srcYStride,
                                    uint32_t                  srcCbStride,
                                    uint32_t                  srcCrStride,
                                    uint8_t* __restrict       dst,
                                    uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo-planar YCbCr444 to RGB565  
///
/// @details
///   This function performs color space conversion from YCbCr444 to RGB565.
///
///   The input are one Y plane followed by one interleaved CbCr (or CrCb) plane:
///   Y plane          :    Y0      Y1      Y2      Y3   ...
///   Interleaved plane: Cb0 Cr0 Cb1 Cr1 Cb2 Cr2 Cb3 Cr3 ...
///
///   The output is one interleaved RGB565 plane:
///   RGB565 plane: R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3...
///
///   RGB565 pixel is arranged with 5-bit Red component, 6-bit Green component,
///   and 5-bit Blue component. One RGB565 pixel is made up of 16-bit data.
///   
///
/// @param srcY
///   Input image Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcC
///   Input image Chroma component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of Y pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of Y lines
///
/// @param srcYStride
///   Stride of input image Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input image Chroma component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). 
///   If left at 0, srcCStride is default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of interleaved RGB565 image  
///   \n\b WARNING: size must match input YCbCr444
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). 
///   If left at 0, srcStride is default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr444PseudoPlanarToRGB565u8( const uint8_t* __restrict srcY,
                                        const uint8_t* __restrict srcC,
                                        uint32_t                  srcWidth,
                                        uint32_t                  srcHeight,
                                        uint32_t                  srcYStride,
                                        uint32_t                  srcCStride,
                                        uint8_t* __restrict       dst,
                                        uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo-planar YCbCr444 to RGB888  
///
/// @details
///   This function performs color space conversion from YCbCr444 to RGB888.
///
///   The input are one Y plane followed by one interleaved CbCr (or CrCb) plane:
///   Y plane          :    Y0      Y1      Y2      Y3   ...
///   Interleaved plane: Cb0 Cr0 Cb1 Cr1 Cb2 Cr2 Cb3 Cr3 ...
///
///   The output is one interleaved RGB888 plane:
///   RGB888 plane: R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3...
///
///   RGB888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   and 8-bit Blue component. One RGB888 pixel is made up of 24-bit data.
///   
///
/// @param srcY
///   Input image Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcC
///   Input image Chroma component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of Y pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of Y lines
///
/// @param srcYStride
///   Stride of input image Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input image Chroma component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). 
///   If left at 0, srcCStride is default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of interleaved RGB888 image 
///   \n\b WARNING: size must match input YCbCr444
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr444PseudoPlanarToRGB888u8( const uint8_t* __restrict srcY,
                                        const uint8_t* __restrict srcC,
                                        uint32_t                  srcWidth,
                                        uint32_t                  srcHeight,
                                        uint32_t                  srcYStride,
                                        uint32_t                  srcCStride,
                                        uint8_t* __restrict       dst,
                                        uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo-planar YCbCr444 to RGBA8888  
///
/// @details
///   This function performs color space conversion from YCbCr444 to RGBA8888.
///
///   The input are one Y plane followed by one interleaved CbCr (or CrCb) plane:
///   Y plane          :    Y0      Y1      Y2      Y3   ...
///   Interleaved plane: Cb0 Cr0 Cb1 Cr1 Cb2 Cr2 Cb3 Cr3 ...
///
///   The output is one interleaved RGBA8888 plane:
///   RGBA8888 plane: R0 G0 B0 A0 R1 G1 B1 A1 R2 G2 B2 A2 R3 G3 B3 A3 ...
///
///   RGBA8888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   8-bit Blue component, and 8-bit A component. One RGBA8888 pixel is made 
///   up of 32-bit data.
///   
///
/// @param srcY
///   Input image Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcC
///   Input image Chroma component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of Y pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of Y lines
///
/// @param srcYStride
///   Stride of input image Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input image Chroma component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). 
///   If left at 0, srcCStride is default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of interleaved RGBA8888 image  
///   \n\b WARNING: size must match input YCbCr444
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr444PseudoPlanarToRGBA8888u8( const uint8_t* __restrict srcY,
                                          const uint8_t* __restrict srcC,
                                          uint32_t                  srcWidth,
                                          uint32_t                  srcHeight,
                                          uint32_t                  srcYStride,
                                          uint32_t                  srcCStride,
                                          uint8_t* __restrict       dst,
                                          uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo-planar YCbCr422 to RGB565  
///
/// @details
///   This function performs color space conversion from YCbCr422 to RGB565.
///
///   The input are one Y plane followed by one interleaved and horizontally 
///   sub-sampled CbCr (or CrCb) plane:
///   Y plane                          : Y0  Y1  Y2  Y3  ...
///   Interleaved and sub-sampled plane: Cb0 Cr0 Cb1 Cr1 ...
///
///   The output is one interleaved RGB565 plane:
///   RGB565 plane: R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3...
///
///   RGB565 pixel is arranged with 5-bit Red component, 6-bit Green component,
///   and 5-bit Blue component. One RGB565 pixel is made up of 16-bit data.
///   
///
/// @param srcY
///   Input image Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcC
///   Input image Chroma component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of Y pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of Y lines
///
/// @param srcYStride
///   Stride of input image Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input image Chroma component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). 
///   If left at 0, srcCStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of interleaved RGB565 image  
///   \n\b WARNING: size must match input YCbCr 422
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr422PseudoPlanarToRGB565u8( const uint8_t* __restrict srcY,
                                        const uint8_t* __restrict srcC,
                                        uint32_t                  srcWidth,
                                        uint32_t                  srcHeight,
                                        uint32_t                  srcYStride,
                                        uint32_t                  srcCStride,
                                        uint8_t* __restrict       dst,
                                        uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo-planar YCbCr422 to RGB888  
///
/// @details
///   This function performs color space conversion from YCbCr422 to RGB888.
///
///   The input are one Y plane followed by one interleaved and horizontally 
///   sub-sampled CbCr (or CrCb) plane:
///   Y plane                          : Y0  Y1  Y2  Y3  ...
///   Interleaved and sub-sampled plane: Cb0 Cr0 Cb1 Cr1 ...
///
///   The output is one interleaved RGB888 plane:
///   RGB888 plane: R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3...
///
///   RGB888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   and 8-bit Blue component. One RGB888 pixel is made up of 24-bit data.
///   
///
/// @param srcY
///   Input image Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcC
///   Input image Chroma component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of Y pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of Y lines
///
/// @param srcYStride
///   Stride of input image Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input image Chroma component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). 
///   If left at 0, srcCStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of interleaved RGB888 image  
///   \n\b WARNING: size must match input YCbCr422
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr422PseudoPlanarToRGB888u8( const uint8_t* __restrict srcY,
                                        const uint8_t* __restrict srcC,
                                        uint32_t                  srcWidth,
                                        uint32_t                  srcHeight,
                                        uint32_t                  srcYStride,
                                        uint32_t                  srcCStride,
                                        uint8_t* __restrict       dst,
                                        uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo-planar YCbCr422 to RGBA8888  
///
/// @details
///   This function performs color space conversion from YCbCr422 to RGBA8888.
///
///   The input are one Y plane followed by one interleaved and horizontally 
///   sub-sampled CbCr (or CrCb) plane:
///   Y plane                          : Y0  Y1  Y2  Y3  ...
///   Interleaved and sub-sampled plane: Cb0 Cr0 Cb1 Cr1 ...
///
///   The output is one interleaved RGBA8888 plane:
///   RGBA8888 plane: R0 G0 B0 A0 R1 G1 B1 A1 R2 G2 B2 A2 R3 G3 B3 A3...
///
///   RGBA8888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   8-bit Blue component, and 8-bit A component. One RGBA8888 pixel is made 
///   up of 32-bit data.
///   
///
/// @param srcY
///   Input image Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcC
///   Input image Chroma component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of Y pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of Y lines
///
/// @param srcYStride
///   Stride of input image Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input image Chroma component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). 
///   If left at 0, srcCStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of interleaved RGBA8888 image  
///   \n\b WARNING: size must match input YCbCr422
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr422PseudoPlanarToRGBA8888u8( const uint8_t* __restrict srcY,
                                          const uint8_t* __restrict srcC,
                                          uint32_t                  srcWidth,
                                          uint32_t                  srcHeight,
                                          uint32_t                  srcYStride,
                                          uint32_t                  srcCStride,
                                          uint8_t* __restrict       dst,
                                          uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo-planar YCbCr420 to RGB565  
///
/// @details
///   This function performs color space conversion from YCbCr420 to RGB565.
///
///   The input are one Y plane followed by one interleaved and 2D (both
///   horizontally and vertically) sub-sampled CbCr (or CrCb) plane:
///   Y plane                             : Y00  Y01  Y02  Y03 ...
///                                         Y10  Y11  Y12  Y13 ...
///   Interleaved and 2D sub-sampled plane: Cb0  Cr0  Cb1  Cr1 ...
///
///   The output is one interleaved RGB565 plane:
///   RGB565 plane: R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3...
///
///   RGB565 pixel is arranged with 5-bit Red component, 6-bit Green component,
///   and 5-bit Blue component. One RGB565 pixel is made up of 16-bit data.
///   
///
/// @param srcY
///   Input image Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcC
///   Input image Chroma component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of Y pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of Y lines
///
/// @param srcYStride
///   Stride of input image Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input image Chroma component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). 
///   If left at 0, srcCStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of interleaved RGB565 image  
///   \n\b WARNING: size must match input YCbCr420
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr420PseudoPlanarToRGB565u8( const uint8_t* __restrict srcY,
                                        const uint8_t* __restrict srcC,
                                        uint32_t                  srcWidth,
                                        uint32_t                  srcHeight,
                                        uint32_t                  srcYStride,
                                        uint32_t                  srcCStride,
                                        uint8_t* __restrict       dst,
                                        uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo-planar YCbCr420 to RGB888  
///
/// @details
///   This function performs color space conversion from YCbCr420 to RGB888.
///
///   The input are one Y plane followed by one interleaved and 2D (both
///   horizontally and vertically) sub-sampled CbCr (or CrCb) plane:
///   Y plane                             : Y00  Y01  Y02  Y03 ...
///                                         Y10  Y11  Y12  Y13 ...
///   Interleaved and 2D sub-sampled plane: Cb0  Cr0  Cb1  Cr1 ...
///
///   The output is one interleaved RGB888 plane:
///   RGB888 plane: R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3...
///
///   RGB888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   and 8-bit Blue component. One RGB888 pixel is made up of 24-bit data.
///   
///
/// @param srcY
///   Input image Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcC
///   Input image Chroma component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of Y pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of Y lines
///
/// @param srcYStride
///   Stride of input image Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input image Chroma component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). 
///   If left at 0, srcCStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of interleaved RGB888 image  
///   \n\b WARNING: size must match input YCbCr420
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * 3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr420PseudoPlanarToRGB888u8( const uint8_t* __restrict srcY,
                                        const uint8_t* __restrict srcC,
                                        uint32_t                  srcWidth,
                                        uint32_t                  srcHeight,
                                        uint32_t                  srcYStride,
                                        uint32_t                  srcCStride,
                                        uint8_t* __restrict       dst,
                                        uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Color conversion from pseudo-planar YCbCr420 to RGBA8888  
///
/// @details
///   This function performs color space conversion from YCbCr420 to RGBA8888.
///
///   The input are one Y plane followed by one interleaved and 2D (both
///   horizontally and vertically) sub-sampled CbCr (or CrCb) plane:
///   Y plane                             : Y00  Y01  Y02  Y03 ...
///                                         Y10  Y11  Y12  Y13 ...
///   Interleaved and 2D sub-sampled plane: Cb0  Cr0  Cb1  Cr1 ...
///
///   The output is one interleaved RGBA8888 plane:
///   RGBA8888 plane: R0 G0 B0 A0 R1 G1 B1 A1 R2 G2 B2 A2 R3 G3 B3 A3...
///
///   RGBA8888 pixel is arranged with 8-bit Red component, 8-bit Green component,
///   8-bit Blue component, and 8-bit A component. One RGBA8888 pixel is made 
///   up of 32-bit data.
///   
///
/// @param srcY
///   Input image Y component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcC
///   Input image Chroma component
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width in number of Y pixels
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height in number of Y lines
///
/// @param srcYStride
///   Stride of input image Y component (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1). 
///   If left at 0, srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCStride
///   Stride of input image Chroma component (i.e., number of bytes between 
///   column 0 of row 0 and column 0 of row 1). 
///   If left at 0, srcCStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The output of interleaved RGBA8888 image  
///   \n\b WARNING: size must match input YCbCr420
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output RGB image (i.e., number of bytes between column 0 of 
///   row 0 and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * 4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvColorYCbCr420PseudoPlanarToRGBA8888u8( const uint8_t* __restrict srcY,
                                          const uint8_t* __restrict srcC,
                                          uint32_t                  srcWidth,
                                          uint32_t                  srcHeight,
                                          uint32_t                  srcYStride,
                                          uint32_t                  srcCStride,
                                          uint8_t* __restrict       dst,
                                          uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Performs edge weighting on input image.  
///
/// @details
///   The following filtes are used for edge weighting.
///
///                           [  0  1 -1  ]
///   Vertical edge filter:   [  0  2 -2  ]
///                           [  0  1 -1  ]
///
///                           [  0  0  0  ]
///   Horizontal edge filter: [  1  2  1  ]
///                           [ -1 -2 -1  ]
///  
/// @param edgeMap
///   Input edge map data
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param edgeMapWidth
///   Input edge map width
///   \n\b NOTE: must be a multiple of 8.
///
/// @param edgeMapHeight
///   Input edge map height
///
/// @param edgeMapStride
///   Stride of input edge map (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///   If left at 0, edgeMapStride is default to edgeMapWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param weight
///   The given edge weighting weight. 
///   It is set to be 6554 (0.2 in Q15 format). 
///
/// @param edge_limit
///   The threshold to distinguish edges from noises. A pixel is from
///   an edge if the filtered value is greater than the edge_limit.
///    
///
/// @param hl_threshold
///   The limit of a pixel value reduction in HL band.  
///    
///
/// @param hh_threshold
///   The limit of a pixel value reduction in HH band.   
///    
///
/// @param edge_denoise_factor
///   Edge denoising factor to make sure a pixel value is reduced only when 
///   the pixel is a noise pixel.
///
/// @return
///   No return value
///
/// @ingroup image_processing
//------------------------------------------------------------------------------

FASTCV_API void
fcvEdgeWeightings16( int16_t* __restrict edgeMap,        
                     const uint32_t      edgeMapWidth,          
                     const uint32_t      edgeMapHeight,         
                     const uint32_t      edgeMapStride,        
                     const uint32_t      weight,         
                     const uint32_t      edge_limit,     
                     const uint32_t      hl_threshold,   
                     const uint32_t      hh_threshold,   
                     const uint32_t      edge_denoise_factor ); 


//------------------------------------------------------------------------------
/// @brief
///   Performe image deinterleave for unsigned byte data.
///
/// @details
///   Deinterleave color compoentonts from src to dst0 and dst1. 
///   Data in src [d0 t0 d1 t1 d2 t2...]
///   Results in dst0 [d0 d1 d2...]
///   Results in dst1 [t0 t1 t2...]
///
/// @param src
///   Input image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Input image width, number of data pairs. For example, CrCb or CbCr pairs. 
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Input image height
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///   If left at 0, srcStride is default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst0
///   Pointer to one of the output image. For example,  Cb or Cr components.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dst0Stride
///   Stride of one of the output image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///   If left at 0, dst0Stride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst1
///   Pointer to one of the output image. For example,  Cb or Cr components.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dst1Stride
///   Stride of one of the output image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///   If left at 0, dst1Stride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @return
///   No return value
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvDeinterleaveu8( const uint8_t* __restrict src,
                   uint32_t                  srcWidth,
                   uint32_t                  srcHeight,
                   uint32_t                  srcStride,
                   uint8_t* __restrict       dst0,
                   uint32_t                  dst0Stride,
                   uint8_t* __restrict       dst1,
                   uint32_t                  dst1Stride );  


//------------------------------------------------------------------------------
/// @brief
///   Performe image interleave
///
/// @details
///   Interleav data from src0 and src1 to dst.
///   Data in src0 [d0 d1 d2 d3...]
///   Data in src1 [t0 t1 t2 t3...]
///   Results in dst [d0 t0 d1 t1 d2 t2 d3 t3...]
///
/// @param src0
///   One of the input images ( For example, Cb or Cr component)
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param src1
///   One of the input images ( For example, Cb or Cr component)
///   \n\b NOTE: must be 128-bit aligned.

/// @param imageWidth
///   Input image width
///   \n\b NOTE: must be a multiple of 8.
///
/// @param imageHeight
///   Input image height
///
/// @param src0Stride
///   Stride of input image 0 (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///   If left at 0, src0Stride is default to imageWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param src1Stride
///   Stride of input image 1 (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///   If left at 0, src1Stride is default to imageWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Pointer to the output image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of the output image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///   If left at 0, dstStride is default to imageWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @return
///   No return value
///
/// @ingroup color_conversion
//------------------------------------------------------------------------------

FASTCV_API void
fcvInterleaveu8( const uint8_t* __restrict src0,
                 const uint8_t* __restrict src1,
                 uint32_t                  imageWidth,
                 uint32_t                  imageHeight,
                 uint32_t                  src0Stride,
                 uint32_t                  src1Stride,
                 uint8_t* __restrict       dst,
                 uint32_t                  dstStride );

//------------------------------------------------------------------------------
/// @brief
///   Performs forward Haar discrete wavelet transform on input image and  
///   transpose the result.
///
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvDWTHaarTransposeu8(). In the 2.0.0 release, 
///   the signature of fcvDWTHarrTransposeu8 as it appears now, 
///   will be removed.
///   \n\n
///
/// @details
///   
///
/// @param src
///   Input image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 of row 0
///   and column 0 of row 1)
///   If left at 0, srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output image (i.e., number of bytes between column 0 of row 0 
///   and column 0 of row 1)
///   If left at 0, dstStride is default to srcWidth * sizeof(int16_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup image_transform    
//------------------------------------------------------------------------------

FASTCV_API void
fcvDWTHarrTransposeu8( const uint8_t* __restrict src,                       
                       uint32_t                  srcWidth,
                       uint32_t                  srcHeight,
                       uint32_t                  srcStride,
                       int16_t* __restrict       dst,
                       uint32_t                  dstStride );

//------------------------------------------------------------------------------
/// @brief
///   Performs forward Haar discrete wavelet transform on input image and  
///   transposes the result.
///
/// @details
///   This function performs forward discrete wavelet transform on input image
///   using the Haar kernel:
///   Low pass:  [ 1   1 ] * 2^(-1/2)
///   High pass: [ 1  -1 ] * 2^(-1/2)
///   This function also transposes the result.
///
/// @param src
///   Input image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 of row 0
///   and column 0 of row 1). 
///   If left at 0, srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output image that has been transformed and transposed 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output image (i.e., number of bytes between column 0 of row 0 
///   and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * sizeof(int16_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup image_transform    
//------------------------------------------------------------------------------

FASTCV_API void
fcvDWTHaarTransposeu8( const uint8_t* __restrict src,                       
                       uint32_t                  srcWidth,
                       uint32_t                  srcHeight,
                       uint32_t                  srcStride,
                       int16_t* __restrict       dst,
                       uint32_t                  dstStride );

//------------------------------------------------------------------------------
/// @brief
///   Performs forward 5-3 Tab discrete wavelet transform on input image and  
///   transposes the result.
///
/// @details
///   This function performs forward discrete wavelet transform on input image
///   using the 5-tab low pass filter and the 3-tab high pass filter:
///   5-tab low pass:  [ -1/8  1/4  3/4  1/4  -1/8 ] * 2^(1/2)
///   3-tab high pass: [      -1/2   1  -1/2       ] * 2^(-1/2)
///   This function also transposes the result.  
///
/// @param src
///   Input image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 of row 0
///   and column 0 of row 1). 
///   If left at 0, srcStride is default to srcWidth * sizeof(int16_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output image that has been transformed and transposed  
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output image (i.e., number of bytes between column 0 of row 0 
///   and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * sizeof(int16_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup image_transform    
//------------------------------------------------------------------------------

FASTCV_API void
fcvDWT53TabTransposes16( const int16_t* __restrict src,                       
                         uint32_t                  srcWidth,
                         uint32_t                  srcHeight,
                         uint32_t                  srcStride,
                         int16_t* __restrict       dst,
                         uint32_t                  dstStride );

//------------------------------------------------------------------------------
/// @brief
///   Performs inverse 5-3 Tab discrete wavelet transform on input image and  
///   transposes the result.
///
/// @details
///   This function performs inverse discrete wavelet transform on input image
///   using the 3-tab low pass filter and the 5-tab high pass filter:
///   3-tab low  pass: [      -1/2   1  -1/2       ] * 2^(-1/2)
///   5-tab high pass: [ -1/8  1/4  3/4  1/4  -1/8 ] * 2^(1/2)
///   This function also transposes the result.  
///
/// @param src
///   Input image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 of row 0
///   and column 0 of row 1). 
///   If left at 0, srcStride is default to srcWidth * sizeof(int16_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output image that has been transformed and transposed 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output image (i.e., number of bytes between column 0 of row 0 
///   and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth * sizeof(int16_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup image_transform    
//------------------------------------------------------------------------------

FASTCV_API void
fcvIDWT53TabTransposes16( const int16_t*   __restrict src,                       
                          uint32_t                    srcWidth,
                          uint32_t                    srcHeight,
                          uint32_t                    srcStride,
                          int16_t* __restrict         dst,
                          uint32_t                    dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Performs inverse Haar discrete wavelet transform on input image and  
///   transpose the result.
///
///   \n\b ATTENTION: This function's signature will become \b OBSOLETE in a future
///   release of this library (2.0.0).  The new interface is specified in the 
///   function: fcvIDWTHaarTransposes16(). In the 2.0.0 release, 
///   the signature of fcvIDWTHarrTransposes16 as it appears now, 
///   will be removed.
///   \n\n
/// 
/// @details
///   
///
/// @param src
///   Input image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 of row 0
///   and column 0 of row 1). 
///   If left at 0, srcStride is default to srcWidth * sizeof(int16_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output image (i.e., number of bytes between column 0 of row 0 
///   and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup image_transform    
//------------------------------------------------------------------------------

FASTCV_API void
fcvIDWTHarrTransposes16( const int16_t* __restrict src,                       
                         uint32_t                  srcWidth,
                         uint32_t                  srcHeight,
                         uint32_t                  srcStride,
                         uint8_t* __restrict       dst,
                         uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Performs inverse Haar discrete wavelet transform on input image and  
///   transposes the result.
/// 
/// @details
///   This function performs inverse discrete wavelet transform on input image
///   using the Haar kernel:
///   Low pass:  [ 1   1 ] * 2^(-1/2)
///   High pass: [ 1  -1 ] * 2^(-1/2)
///   This function also transposes the result.
///
/// @param src
///   Input image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 of row 0
///   and column 0 of row 1). 
///   If left at 0, srcStride is default to srcWidth * sizeof(int16_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output image that has been transformed and transposed 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output image (i.e., number of bytes between column 0 of row 0 
///   and column 0 of row 1). 
///   If left at 0, dstStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup image_transform    
//------------------------------------------------------------------------------

FASTCV_API void
fcvIDWTHaarTransposes16( const int16_t* __restrict src,                       
                         uint32_t                  srcWidth,
                         uint32_t                  srcHeight,
                         uint32_t                  srcStride,
                         uint8_t* __restrict       dst,
                         uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Performs forward Haar discrete wavelet transform on input image   
///   
///
/// @details
///   This function performs forward discrete wavelet transform on the input 
///   image using Haar kernel. 
///   
///
/// @param src
///   Pointer to input single plane image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 of row 1
///   and column 0 of row 2). If left at 0, srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Pointer to output image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output image (i.e., number of bytes between column 0 of row 1 
///   and column 0 of row 2). If left at 0, dstStride is default to 
///   srcWidth * sizeof(int16_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup image_transform    
//------------------------------------------------------------------------------

FASTCV_API void
fcvDWTHaaru8( const uint8_t* __restrict src,                       
              uint32_t                  srcWidth,
              uint32_t                  srcHeight,
              uint32_t                  srcStride,
              int16_t* __restrict       dst,
              uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Performs forward 5-3 Tab discrete wavelet transform on input image   
///   
///
/// @details
///   This function performs forward discrete wavelet transform on the input 
///   image using 5-3 Tab kernel. 
///
/// @param src
///   Pointer to input single plane image  
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 of row 1
///   and column 0 of row 2). If left at 0, srcStride is default to 
///   srcWidth * sizeof(int16_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Pointer to output image  
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output image (i.e., number of bytes between column 0 of row 1 
///   and column 0 of row 2). If left at 0, dstStride is default to 
///   srcWidth * sizeof(int16_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup image_transform    
//------------------------------------------------------------------------------

FASTCV_API void
fcvDWT53Tabs16( const int16_t* __restrict src,                       
                uint32_t                  srcWidth,
                uint32_t                  srcHeight,
                uint32_t                  srcStride,
                int16_t* __restrict       dst,
                uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Performs inverse 5-3 Tab discrete wavelet transform on input image   
///   
///
/// @details
///   This function performs inverse discrete wavelet transform on the input 
///   image using 5-3 Tab kernel. 
///
/// @param src
///   Pointer to input single plane image  
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 of row 1
///   and column 0 of row 2). If left at 0, srcStride is default to 
///   srcWidth * sizeof(int16_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Pointer to output image  
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output image (i.e., number of bytes between column 0 of row 1 
///   and column 0 of row 2). If left at 0, dstStride is default to 
///   srcWidth * sizeof(int16_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup image_transform    
//------------------------------------------------------------------------------

FASTCV_API void
fcvIDWT53Tabs16( const int16_t*   __restrict src,                       
                 uint32_t                    srcWidth,
                 uint32_t                    srcHeight,
                 uint32_t                    srcStride,
                 int16_t* __restrict         dst,
                 uint32_t                    dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Performs inverse Haar discrete wavelet transform on input image   
///   
///
/// @details
///   This function performs inverse discrete wavelet transform on the input 
///   image using Haar kernel. 
///
///
/// @param src
///   Pointer to input single plane image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 of row 1
///   and column 0 of row 2). If left at 0, srcStride is default to 
///   srcWidth * sizeof(int16_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Pointer to output image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output image (i.e., number of bytes between column 0 of row 1 
///   and column 0 of row 2). If left at 0, dstStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup image_transform    
//------------------------------------------------------------------------------

FASTCV_API void
fcvIDWTHaars16( const int16_t* __restrict src,                       
                uint32_t                  srcWidth,
                uint32_t                  srcHeight,
                uint32_t                  srcStride,
                uint8_t* __restrict       dst,
                uint32_t                  dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Performs forward discrete Cosine transform on uint8_t pixels   
///   
///
/// @details
///   This function performs 8x8 forward discrete Cosine transform on input
///   image
///
/// @param src
///   Pointer to input single plane image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 of row 1
///   and column 0 of row 2). If left at 0, srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Pointer to output image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output image (i.e., number of bytes between column 0 of row 1 
///   and column 0 of row 2). If left at 0, dstStride is default to 
///   srcWidth * sizeof(int16_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup image_transform    
//------------------------------------------------------------------------------

FASTCV_API void
fcvDCTu8( const uint8_t* __restrict src,                       
          uint32_t                  srcWidth,
          uint32_t                  srcHeight,
          uint32_t                  srcStride,
          int16_t* __restrict       dst,
          uint32_t                  dstStride );              


//------------------------------------------------------------------------------
/// @brief
///   Performs inverse discrete cosine transform on int16_t coefficients   
///   
///
/// @details
///  This function performs 8x8 inverse discrete Cosine transform on input
///   image
///
/// @param src
///   Pointer to input single plane image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Width of the input image
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Height of the input image
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 of row 1
///   and column 0 of row 2). If left at 0, srcStride is default to 
///   srcWidth * sizeof(int16_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Pointer to output image 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride of output image (i.e., number of bytes between column 0 of row 1 
///   and column 0 of row 2). If left at 0, dstStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup image_transform    
//------------------------------------------------------------------------------

FASTCV_API void
fcvIDCTs16( const int16_t* __restrict src,                       
            uint32_t                  srcWidth,
            uint32_t                  srcHeight,
            uint32_t                  srcStride,
            uint8_t* __restrict       dst,
            uint32_t                  dstStride );

//------------------------------------------------------------------------------
/// @brief
///   Perform image upscaling using polyphase filters
///
/// @details
///   Perform image upscaling using polyphase filters. The image data type is 
///   unsigned byte.
///
/// @param src
///   Input image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Input image width
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Input image height
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///   If left at 0, srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstWidth
///   Output image width
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstHeight
///   Output image height
///
/// @param dstStride
///   Stride of output image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///   If left at 0, dstStride is default to dstWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @return
///   No return value.
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API void
fcvScaleUpPolyu8( const uint8_t* __restrict src,
                  uint32_t                  srcWidth,
                  uint32_t                  srcHeight,
                  uint32_t                  srcStride,
                  uint8_t* __restrict       dst,
                  uint32_t                  dstWidth,
                  uint32_t                  dstHeight,
                  uint32_t                  dstStride );   


//------------------------------------------------------------------------------
/// @brief
///   Interleaved image (CbCr or CrCb) upscaling using polyphase filters
///
/// @details
///   Perform interleaved image (CbCr or CrCb) upscaling using polyphase 
///   filters. Data type is unsigned byte.
///
/// @param src
///   Input image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Input image width, number of (CrCb/CbCr) pairs
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Input image height
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///   If left at 0, srcStride is default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstWidth
///   Output image width, number of (CrCb/CbCr) pairs
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstHeight
///   Output image height
///
/// @param dstStride
///   Stride of output image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///   If left at 0, dstStride is default to dstWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @return
///   No return value.
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API void
fcvScaleUpPolyInterleaveu8( const uint8_t* __restrict src,
                            uint32_t                  srcWidth,
                            uint32_t                  srcHeight,
                            uint32_t                  srcStride,
                            uint8_t* __restrict       dst,
                            uint32_t                  dstWidth,
                            uint32_t                  dstHeight,
                            uint32_t                  dstStride );  


//------------------------------------------------------------------------------
/// @brief
///   Image downscaling using MN method
///
/// @details
///   The M over N downscale algorithm works on an arbitrary length (N) of 
///   input data, and generates another arbitrary length (M) of output data,
///   with the output length M less or equal to the input length N.
 
///
/// @param src
///   Input image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Input image width
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Input image height
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///   If left at 0, srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstWidth
///   Output image width
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstHeight
///   Output image height
///
/// @param dstStride
///   Stride of output image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1)
///   If left at 0, dstStride is default to dstWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @return
///   No return value.
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API void
fcvScaleDownMNu8( const uint8_t* __restrict src,
                  uint32_t                  srcWidth,
                  uint32_t                  srcHeight,
                  uint32_t                  srcStride,
                  uint8_t* __restrict       dst,
                  uint32_t                  dstWidth,
                  uint32_t                  dstHeight,
                  uint32_t                  dstStride );   


//------------------------------------------------------------------------------
/// @brief
///   Interleaved image downscaling using MN method
///
/// @details
///   The M over N downscale algorithm works on an arbitrary length (N) of 
///   input data, and generates another arbitrary length (M) of output data,
///   with the output length M less or equal to the input length N.
///
/// @param src
///   Input image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Input image width, number of (CrCb/CbCr) pair
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Input image height
///
/// @param srcStride
///   Stride of input image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///   If left at 0, srcStride is default to srcWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output image
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstWidth
///   Output image width , number of (CrCb/CbCr) pair
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstHeight
///   Output image height
///
/// @param dstStride
///   Stride of output image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///   If left at 0, dstStride is default to dstWidth * 2.
///   \n\b NOTE: must be a multiple of 8.
///
/// @return
///   No return value.
///
/// @ingroup image_transform
//------------------------------------------------------------------------------

FASTCV_API void
fcvScaleDownMNInterleaveu8( const uint8_t* __restrict src,
                            uint32_t                  srcWidth,
                            uint32_t                  srcHeight,
                            uint32_t                  srcStride,
                            uint8_t* __restrict       dst,
                            uint32_t                  dstWidth,
                            uint32_t                  dstHeight,
                            uint32_t                  dstStride ); 

//---------------------------------------------------------------------------
/// @brief
///   Search K-Means tree, where each node connects to up to 10 children,
///   and the center (mean) is a 36-tuple vector of 8-bit signed value.
///
/// @param nodeChildrenCenter
///   A pointer to uint8_t [numNodes][10][36],
///   which stores the center vectors of node children.
///   The outer-most dimension represents the nodes in the tree.
///   The middle dimension represents the children of each node.
///   The inner-most dimension represents the tuples of the center vector.
///   \n\b WARNING: must be 64-bit aligned.
///
/// @param nodeChildrenInvLenQ32
///   A pointer to uint32_t [numNodes][10],
///   which stores the inverse lengths of the center vectors.
///   The inverse lengths are in Q32 format.
///   The outer-most dimension represents the nodes in the tree.
///   The inner-most dimension represents the children of each node.
///   \n\b WARNING: must be 64-bit aligned.
///
/// @param nodeChildrenIndex
///   A pointer to uint32_t [numNodes][10],
///   which stores the indices of the children nodes.
///   If the MSB is 0, the index points to a node within the tree.
///   If the MSB is 1, the index (with MSB removed) points to a leaf node,
///   which is returned by this function as the search result.
///   See nodeChildrenInvLenQ32 for the definition of each dimension.
///   \n\b WARNING: must be 64-bit aligned.
///
/// @param nodeNumChildren
///   A pointer to uint8_t [numNodes],
///   which stores the number of children in each node.
///
/// @param numNodes
///   Number of nodes in the K-Means tree.
///
/// @param key
///   A pointer to int8_t [36], which stores the key to be searched.
///
/// @return
///   Index of the leaf node.
///
/// @ingroup feature_detection
//---------------------------------------------------------------------------

FASTCV_API uint32_t
fcvKMeansTreeSearch36x10s8( const   int8_t* __restrict  nodeChildrenCenter,
                            const uint32_t* __restrict  nodeChildrenInvLenQ32,
                            const uint32_t* __restrict  nodeChildrenIndex,
                            const  uint8_t* __restrict  nodeNumChildren,
                                  uint32_t              numNodes,
                            const  int8_t * __restrict  key );

//---------------------------------------------------------------------------
/// @brief
///   Sorts in-place the pairs of <descDB, descDBInvLenQ38 > according to
///   descDBTargetId.
///
/// @param dbLUT
///   A pointer to uint32_t [numDBLUT][2],
///   which stores the starting index of descDB and
///   the number of descriptors
///   \n\b WARNING: must be 64-bit aligned.
///
/// @param numDBLUT
///   The size of dbLUT
///
/// @param descDB
///   A pointer to int8_t [numDescDB][36],
///   which stores descriptors
///   \n\b WARNING: must be 64-bit aligned.
///
/// @param descDBInvLenQ38
///   A pointer to uint32_t [numDescDB],
///   which stores the inverse length of descDB.
///   The value is in Q38 format.
///
/// @param descDBTargetId
///   A pointer to uint16_t [numDescDB],
///   which stores the target id.
///
/// @param descDBOldIdx
///   A pointer to uint32_t [numDescDB],
///   which stores the old index of the desc before sorting
///
/// @param numDescDB
///   Number of descriptor in the database.
///
/// @ingroup feature_detection
//---------------------------------------------------------------------------

FASTCV_API int
fcvLinearSearchPrepare8x36s8(  uint32_t * __restrict   dbLUT,
                               uint32_t                numDBLUT,
                               int8_t   * __restrict   descDB,
                               uint32_t * __restrict   descDBInvLenQ38,
                               uint16_t * __restrict   descDBTargetId,
                               uint32_t * __restrict   descDBOldIdx,
                               uint32_t                numDescDB );

//---------------------------------------------------------------------------
/// @brief
///   Perform linear search of descriptor in a database
///
/// @param dbLUT
///   A pointer to uint32_t [numDBLUT][2],
///   which stores the starting index of descDB and
///   the number of descriptors
///   \n\b WARNING: must be 64-bit aligned.
///
/// @param numDBLUT
///   The size of dbLUT
///
/// @param descDB
///   A pointer to int8_t [numDescDB][36],
///   which stores descriptors
///   \n\b WARNING: must be 64-bit aligned.
///
/// @param descDBInvLenQ38
///   A pointer to uint32_t [numDescDB],
///   which stores the inverse length of descDB.
///   The value is in Q38 format.
///
/// @param descDBTargetId
///   A pointer to uint16_t [numDescDB],
///   which stores the target id.
///
/// @param numDescDB
///   Number of descriptor in the database.
///
/// @param srcDesc
///   A pointer to int8_t [numSrcDesc][36],
///   which stores descriptors.
///   \n\b WARNING: must be 64-bit aligned.
///
/// @param srcDescInvLenQ38
///   A pointer to uint32_t [numSrcDec],
///   which stores the inverse length of srcDesc.
///   The value is in Q38 format.
///
/// @param srcDescIdx
///   A pointer to the dbLUT data
///
/// @param numSrcDesc
///   Number of source descriptor
///
/// @param targetsToIgnore
///   A list of target IDs to be ignored
///
/// @param numTargetsToIgnore
///   Number of targets to be ignored
///
/// @param maxDistanceQ31
///   Maximum distance for correspondences.
///   In Q31 format.
///
/// @param correspondenceDBIdx
///   A pointer to uint32_t [maxNumCorrespondences],
///   which will be used by this function to output indices of featuresDB
///   as a part of correspondences.
///
/// @param correspondenceSrcDescIdx
///   A pointer to uint32_t [maxNumCorrespondences],
///   which will be used by this function to output indices of descriptors
///   as a part of correspondences.
///
/// @param correspondenceDistanceQ31
///   A pointer to uint32_t [maxNumCorrespondences],
///   which will be used by this function to output the distances
///   as a part of correspondences.
///   In Q31 format.
///
/// @param maxNumCorrespondences
///   Maximum number of correspondences allowed
///
/// @param numCorrespondences
///   Number of correspondences returned by this function
///
/// @ingroup feature_detection
//---------------------------------------------------------------------------

FASTCV_API void
fcvLinearSearch8x36s8(
   const uint32_t * __restrict dbLUT,
   uint32_t                    numDBLUT,
   const int8_t   * __restrict descDB,
   const uint32_t * __restrict descDBInvLenQ38,
   const uint16_t * __restrict descDBTargetId,
   uint32_t                    numDescDB,
   const int8_t   * __restrict srcDesc,
   const uint32_t * __restrict srcDescInvLenQ38,
   const uint32_t * __restrict srcDescIdx,
   uint32_t                    numSrcDesc,
   const uint16_t * __restrict targetsToIgnore,
   uint32_t                    numTargetsToIgnore,
   uint32_t                    maxDistanceQ31,
   uint32_t       * __restrict correspondenceDBIdx,
   uint32_t       * __restrict correspondenceSrcDescIdx,
   uint32_t       * __restrict correspondenceDistanceQ31,
   uint32_t                    maxNumCorrespondences,
   uint32_t       * __restrict numCorrespondences );


//------------------------------------------------------------------------------
/// @brief
///   Finds only extreme outer contours in a binary image.  There is no nesting
///   relationship between contours.  It sets hierarchy[i][2]=hierarchy[i][3]=-1
///   for all the contours.
///
/// @param src
///   Grayscale image with one byte per pixel.  Non-zero pixels are treated as
///   1's. Zero pixels remain 0's, so the image is treated as binary.
///
/// @param srcWidth
///   Image width
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///
/// @param maxNumContours
///   Maximum number of contours can be found
///
/// @param numContours
///   Number of actually found contours
///
/// @param numContourPoints
///   Number of points in each found contour
///
/// @param contourStartPoints
///   Pointers to the start point of each found contour
///
/// @param pointBuffer
///   Pointer to point buffer for contour points' coordinates.  It should
///   be allocated before calling this function.
///
/// @param pointBufferSize
///   Size of point buffer in terms of uint32_t
///
/// @param hierarchy
///   Information about the image topology.  It has numContours elements.
///   For each contour i, the elements hierarchy[i][0], hiearchy[i][1], 
///   hiearchy[i][2], and hiearchy[i][3] are set to 0-based indices of the
///   next and previous contours at the same hierarchical level, the first
///   child contour and the parent contour, respectively.  If for a contour i
///   there are no next, previous, parent, or nested contours, the corresponding
///   elements of hierarchy[i] will be negative.
///
/// @param contourHandle
///   Pointer to assistant and intermediate data.  It should be allocated by 
///   fcvFindContoursAllocate() and deallocated by fcvFindContoursDelete().
///
/// @ingroup feature_detection
//------------------------------------------------------------------------------
FASTCV_API void
fcvFindContoursExternalu8( uint8_t* __restrict   src,
                           uint32_t              srcWidth,
                           uint32_t              srcHeight,
                           uint32_t              srcStride,
                           uint32_t              maxNumContours,
                           uint32_t* __restrict  numContours,
                           uint32_t* __restrict  numContourPoints,
                           uint32_t** __restrict contourStartPoints,
                           uint32_t* __restrict  pointBuffer,
                           uint32_t              pointBufferSize,
                           int32_t               hierarchy[][4],
                           void*                 contourHandle );


//------------------------------------------------------------------------------
/// @brief
///   Finds contours in a binary image without any hierarchical relationships.
///
/// @param src
///   Grayscale image with one byte per pixel.  Non-zero pixels are treated as
///   1's. Zero pixels remain 0's, so the image is treated as binary.
///
/// @param srcWidth
///   Image width
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///
/// @param maxNumContours
///   Maximum number of contours can be found
///
/// @param numContours
///   Number of actually found contours
///
/// @param numContourPoints
///   Number of points in each found contour
///
/// @param contourStartPoints
///   Pointers to the start point of each found contour
///
/// @param pointBuffer
///   Pointer to point buffer for contour points' coordinates.  It should
///   be allocated before calling this function.
///
/// @param pointBufferSize
///   Size of point buffer in terms of uint32_t
///
/// @param contourHandle
///   Pointer to assistant and intermediate data.  It should be allocated by 
///   fcvFindContoursAllocate() and deallocated by fcvFindContoursDelete().
///
/// @ingroup feature_detection
//------------------------------------------------------------------------------
FASTCV_API void
fcvFindContoursListu8( uint8_t* __restrict   src,
                       uint32_t              srcWidth,
                       uint32_t              srcHeight,
                       uint32_t              srcStride,
                       uint32_t              maxNumContours,
                       uint32_t* __restrict  numContours,
                       uint32_t* __restrict  numContourPoints,
                       uint32_t** __restrict contourStartPoints,
                       uint32_t* __restrict  pointBuffer,
                       uint32_t              pointBufferSize,
                       void*                 contourHandle );


//------------------------------------------------------------------------------
/// @brief
///   Finds contours in a binary image and organizes them into a two-level 
///   hierarchy.  At the top level, there are external boundaries of the 
///   components. At the second level, there are boundaries of the holes. 
///   If there is another contour inside a hole of a connected component, 
///   it is still put at the top level. 
///
/// @param src
///   Grayscale image with one byte per pixel.  Non-zero pixels are treated as
///   1's. Zero pixels remain 0's, so the image is treated as binary.
///
/// @param srcWidth
///   Image width
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///
/// @param maxNumContours
///   Maximum number of contours can be found (<= 126)
///
/// @param numContours
///   Number of actually found contours
///
/// @param holeFlag
///   Hole flag for each found contour to indicate whether it is a hole or not
///
/// @param numContourPoints
///   Number of points in each found contour
///
/// @param contourStartPoints
///   Pointers to the start point of each found contour
///
/// @param pointBuffer
///   Pointer to point buffer for contour points' coordinates.  It should
///   be allocated before calling this function.
///
/// @param pointBufferSize
///   Size of point buffer in terms of uint32_t
///
/// @param hierarchy
///   Information about the image topology.  It has numContours elements.
///   For each contour i, the elements hierarchy[i][0], hiearchy[i][1], 
///   hiearchy[i][2], and hiearchy[i][3] are set to 0-based indices of the
///   next and previous contours at the same hierarchical level, the first
///   child contour and the parent contour, respectively.  If for a contour i
///   there are no next, previous, parent, or nested contours, the corresponding
///   elements of hierarchy[i] will be negative.
///
/// @param contourHandle
///   Pointer to assistant and intermediate data.  It should be allocated by 
///   fcvFindContoursAllocate() and deallocated by fcvFindContoursDelete().
///
/// @ingroup feature_detection
//------------------------------------------------------------------------------
FASTCV_API void
fcvFindContoursCcompu8( uint8_t* __restrict   src,
                        uint32_t              srcWidth,
                        uint32_t              srcHeight,
                        uint32_t              srcStride,
                        uint32_t              maxNumContours,
                        uint32_t* __restrict   numContours,
                        uint32_t* __restrict  holeFlag,
                        uint32_t* __restrict  numContourPoints,
                        uint32_t** __restrict contourStartPoints,
                        uint32_t* __restrict  pointBuffer,
                        uint32_t              pointBufferSize,
                        int32_t               hierarchy[][4],
                        void*                 contourHandle );


//------------------------------------------------------------------------------
/// @brief
///   Finds contours in a binary image and reconstructs a full hierarchy of 
///   nested contours
///
/// @param src
///   Grayscale image with one byte per pixel.  Non-zero pixels are treated as
///   1's. Zero pixels remain 0's, so the image is treated as binary.
///
/// @param srcWidth
///   Image width
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///
/// @param numContours
///   Number of actually found contours
///
/// @param maxNumContours
///   Maximum number of contours can be found (<= 126)
///
/// @param holeFlag
///   Hole flag for each found contour to indicate whether it is a hole or not
///
/// @param numContourPoints
///   Number of points in each found contour
///
/// @param contourStartPoints
///   Pointers to the start point of each found contour
///
/// @param pointBuffer
///   Pointer to point buffer for contour points' coordinates.  It should
///   be allocated before calling this function.
///
/// @param pointBufferSize
///   Size of point buffer in terms of uint32_t
///
/// @param hierarchy
///   Information about the image topology.  It has numContours elements.
///   For each contour i, the elements hierarchy[i][0], hiearchy[i][1], 
///   hiearchy[i][2], and hiearchy[i][3] are set to 0-based indices of the
///   next and previous contours at the same hierarchical level, the first
///   child contour and the parent contour, respectively.  If for a contour i
///   there are no next, previous, parent, or nested contours, the corresponding
///   elements of hierarchy[i] will be negative.
///
/// @param contourHandle
///   Pointer to assistant and intermediate data.  It should be allocated by 
///   fcvFindContoursAllocate() and deallocated by fcvFindContoursDelete().
///
/// @ingroup feature_detection
//------------------------------------------------------------------------------
FASTCV_API void
fcvFindContoursTreeu8( uint8_t* __restrict   src,
                       uint32_t              srcWidth,
                       uint32_t              srcHeight,
                       uint32_t              srcStride,
                       uint32_t              maxNumContours,
                       uint32_t* __restrict  numContours,
                       uint32_t* __restrict  holeFlag,
                       uint32_t* __restrict  numContourPoints,
                       uint32_t** __restrict contourStartPoints,
                       uint32_t* __restrict  pointBuffer,
                       uint32_t              pointBufferSize,
                       int32_t               hierarchy[][4],
                       void*                 contourHandle );


//------------------------------------------------------------------------------
/// @brief
///   Allocates assistant and intermediate data for contour
///
/// @param srcStride
///   Stride of image (i.e., how many pixels between column 0 of row 1 and
///   column 0 of row 2).
///
/// @return
///   Pointer to allocated data
///
/// @ingroup feature_detection
//------------------------------------------------------------------------------
FASTCV_API void*
fcvFindContoursAllocate( uint32_t srcStride );


//------------------------------------------------------------------------------
/// @brief
///   Deallocates assistant and intermediate data for contour
///
/// @param contourHandle
///   Pointer to assistant and intermediate data
///
/// @ingroup feature_detection
//------------------------------------------------------------------------------
FASTCV_API void
fcvFindContoursDelete( void* contourHandle );

//------------------------------------------------------------------------------
/// @brief
///   Solve linear equation system
///          Ax = b
///
/// @details
///   
///
/// @param A
///    The matrix contains coefficients of the linear equation system
///
/// @param numRows
///    The number of rows for the matrix A
///
/// @param numCols
///    The number of columns for the matrix A
///
/// @param b
///    The right side value
///
/// @param x
///    The solution vector
///
///
/// @return
///    
/// @ingroup math_vector 
//------------------------------------------------------------------------------
FASTCV_API void 
fcvSolvef32(const float32_t * __restrict A, 
            int32_t numCols, 
            int32_t numRows, 
            const float32_t * __restrict b, 
            float32_t * __restrict x);

//------------------------------------------------------------------------------
/// @brief
///   Calculates a perspective transform from four pairs of the corresponding
///   points.
///   NOTE: in order to guarantee a valid output transform, any three points
///         in src1 or src2 cannot be collinear. 
///
/// @param src1
///   Coordinates of quadrangle vertices in the source image
///
/// @param src2
///   Coordinates of the corresponding quadrangle vertices in the destination
///   image
///
/// @param transformCoefficient
///   3x3 matrix of a perspective transform 
///
/// @ingroup image_transform
//------------------------------------------------------------------------------
FASTCV_API void
fcvGetPerspectiveTransformf32( const float32_t src1[8],
                               const float32_t src2[8],
                               float32_t  transformCoefficient[9] );

//------------------------------------------------------------------------------
/// @brief
///   Sets every element of a uint8_t single channel array to a given value.
///
/// @details  
///   A non-zero element of the mask array indicates the corresponding element 
///   of the destination array to be changed. The mask itself equals to zero means that
///   all elements of the dst array need to be changed. The mask is assumed to
///    have the same width and height( in terms of pixels) as the destination array.
///
/// @param dst
///    The destination matrix
///
/// @param dstWidth
///    Destination matrix width
///
/// @param dstHeight
///    Destination matrix height
///
/// @param dstStride
///    Stride for the destination matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param value
///    the input uint8_t value 
///
/// @param mask
///    Operation mask, 8-bit single channel array; specifies elements of the src
///       array to be changed. 
///
/// @param maskStride
///    Stride for the mask, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @return
///   No return value
///
/// @ingroup math_vector 
//------------------------------------------------------------------------------
  
FASTCV_API void
fcvSetElementsu8(        uint8_t * __restrict dst, 
                         uint32_t             dstWidth, 
                         uint32_t             dstHeight,
                         uint32_t             dstStride, 
                         uint8_t              value,
                   const uint8_t * __restrict mask,
                         uint32_t             maskStride
                 );

//------------------------------------------------------------------------------
/// @brief
///   Sets every element of an int32_t  single channel array to a given value.
///
/// @details
///   A non-zero element of the mask array indicates the corresponding element 
///   of the destination array to be changed. The mask itself equals to zero means that
///   all elements of the dst array need to be changed. The mask is assumed to
///    have the same width and height( in terms of pixels) as the destination array.
///
/// @param dst
///    The destination matrix
///
/// @param dstWidth
///    Destination matrix width
///
/// @param dstHeight
///    Destination matrix height
///
/// @param dstStride
///    Stride for the destination matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param value
///    the input int32_t value
///
/// @param mask
///    Operation mask, 8-bit single channel array; specifies elements of the src
///    array to be changed
///
/// @param maskStride
///    Stride for input mask, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row 
///
/// @return
///   No return value
///
/// @ingroup math_vector 
//------------------------------------------------------------------------------
FASTCV_API void
fcvSetElementss32(          int32_t * __restrict dst, 
                             uint32_t             dstWidth, 
                             uint32_t             dstHeight,
                             uint32_t             dstStride, 
                             int32_t              value,
                       const uint8_t * __restrict mask ,
                             uint32_t             maskStride
                     );

//------------------------------------------------------------------------------
/// @brief
///   Sets every element of a float32_t single channel array to a given value.
///
/// @details
///   A non-zero element of the mask array indicates the corresponding element 
///   of the destination array to be changed. The mask itself equals to zero means that
///   all elements of the dst array need to be changed. The mask is assumed to
///    have the same width and height( in terms of pixels) as the destination array.
///
/// @param dst
///    The destination matrix
///
/// @param dstWidth
///    Destination matrix width
///
/// @param dstHeight
///    Destination matrix height
///
/// @param dstStride
///    Stride for the destination matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param value
///    the input float32_t value
///
/// @param mask
///    Operation mask, 8-bit single channel array; specifies elements of the src
///    array to be changed
///
/// @param maskStride
///    Stride for input mask, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @return
///   No return value
///
/// @ingroup math_vector 
//------------------------------------------------------------------------------
FASTCV_API void
fcvSetElementsf32(        float32_t * __restrict dst, 
                          uint32_t               dstWidth, 
                          uint32_t               dstHeight,
                          uint32_t               dstStride, 
                          float32_t              value,
                    const uint8_t   * __restrict mask,
                          uint32_t               maskStride
                   );

//------------------------------------------------------------------------------
/// @brief
///   Sets every element of a uint8_t 4-channel  array to a given 4-element scalar.
///
/// @details
///   A non-zero element of the mask array indicates the corresponding element 
///   of the destination array to be changed. The mask itself equals to zero means that
///   all elements of the dst array need to be changed. The mask is assumed to
///    have the same width and height( in terms of pixels) as the destination array.
///
/// @param dst
///    The destination matrix
///
/// @param dstWidth
///    Destination matrix width
///
/// @param dstHeight
///    Destination matrix height
///
/// @param dstStride
///    Stride for the destination matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param value1
///    First uint8_t value of the Scalar
///
/// @param value2
///    Second uint8_t value of the Scalar
///
/// @param value3
///    Third uint8_t value of the Scalar
///
/// @param value4
///    Fourth uint8_t value of the Scalar
///
/// @param mask
///    Operation mask, 8-bit single channel array; specifies elements of the src
///    array to be changed
///
/// @param maskStride
///    Stride for input mask, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @return
///   No return value
///
/// @ingroup math_vector 
//------------------------------------------------------------------------------
FASTCV_API void
fcvSetElementsc4u8(         uint8_t * __restrict dst, 
                            uint32_t             dstWidth, 
                            uint32_t             dstHeight,
                            uint32_t             dstStride, 
                            uint8_t              value1,
                            uint8_t              value2,
                            uint8_t              value3,
                            uint8_t              value4,
                      const uint8_t * __restrict mask,
                            uint32_t             maskStride
                    );

//------------------------------------------------------------------------------
/// @brief
///   Sets every element of an int32_t 4-channel  array to a given 4-element scalar.
///
/// @details
///   A non-zero element of the mask array indicates the corresponding element 
///   of the destination array to be changed. The mask itself equals to zero means that
///   all elements of the dst array need to be changed. The mask is assumed to
///    have the same width and height( in terms of pixels) as the destination array.
///
/// @param dst
///    The destination matrix
///
/// @param dstWidth
///    Destination matrix width
///
/// @param dstHeight
///    Destination matrix height
///
/// @param dstStride
///    Stride for the destination matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param value1
///    First int32_t value of the Scalar
///
/// @param value2
///    Second int32_t value of the Scalar
///
/// @param value3
///    Third int32_t value of the Scalar
///
/// @param value4
///    Fourth int32_t value of the Scalar
///
/// @param mask
///    Operation mask, 8-bit single channel array; specifies elements of the src
///    array to be changed. 
///
/// @param maskStride
///    Stride for input mask, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @return
///   No return value
///
/// @ingroup math_vector 
//------------------------------------------------------------------------------
FASTCV_API void
fcvSetElementsc4s32(         int32_t * __restrict dst, 
                             uint32_t             dstWidth, 
                             uint32_t             dstHeight,
                             uint32_t             dstStride, 
                             int32_t              value1,
                             int32_t              value2,
                             int32_t              value3,
                             int32_t              value4,
                       const uint8_t * __restrict mask,
                             uint32_t             maskStride 
                     );

//------------------------------------------------------------------------------
/// @brief
///   Sets every element of a float32_t 4-channel  array to a given 4-element scalar.
///
/// @details
///   A non-zero element of the mask array indicates the corresponding element 
///   of the destination array to be changed. The mask itself equals to zero means that
///   all elements of the dst array need to be changed. The mask is assumed to
///    have the same width and height( in terms of pixels) as the destination array.
///
/// @param dst
///    The destination matrix
///
/// @param dstWidth
///    Destination matrix width
///
/// @param dstHeight
///    Destination matrix height
///
/// @param dstStride
///    Stride for the destination matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param value1
///    First float32_t value of the Scalar
///
/// @param value2
///    Second float32_t value of the Scalar
///
/// @param value3
///    Third float32_t value of the Scalar
///
/// @param value4
///    Fourth float32_t value of the Scalar
///
/// @param mask
///    Operation mask, 8-bit single channel array; specifies elements of the src
///    array to be changed
///
/// @param maskStride
///    Stride for input mask, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @return
///   No return value
///
/// @ingroup math_vector 
//------------------------------------------------------------------------------
FASTCV_API void
fcvSetElementsc4f32(         float32_t * __restrict dst, 
                             uint32_t               dstWidth, 
                             uint32_t               dstHeight,
                             uint32_t               dstStride, 
                             float32_t              value1,
                             float32_t              value2,
                             float32_t              value3,
                             float32_t              value4,
                       const uint8_t   * __restrict mask,
                             uint32_t               maskStride
                     );
		
//------------------------------------------------------------------------------
/// @brief
///   Sets every element of a uint8_t 3-channel  array to a given 3-element scalar.
///
/// @details
///   A non-zero element of the mask array indicates the corresponding element 
///   of the destination array to be changed. The mask itself equals to zero means that
///   all elements of the dst array need to be changed. The mask is assumed to
///    have the same width and height( in terms of pixels) as the destination array.
///
/// @param dst
///    The destination matrix
///
/// @param dstWidth
///    Destination matrix width
///
/// @param dstHeight
///    Destination matrix height
///
/// @param dstStride
///    Stride for the destination matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param value1
///    First uint8_t value of the Scalar
///
/// @param value2
///    Second uint8_t value of the Scalar
///
/// @param value3
///    Third uint8_t value of the Scalar
///
/// @param mask
///    Operation mask, 8-bit single channel array; specifies elements of the src
///    array to be changed
///
/// @param maskStride
///    Stride for input mask, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @return
///   No return value
///
/// @ingroup math_vector 
//------------------------------------------------------------------------------
FASTCV_API void
fcvSetElementsc3u8(         uint8_t * __restrict dst, 
                            uint32_t             dstWidth, 
                            uint32_t             dstHeight,
                            uint32_t             dstStride, 
                            uint8_t              value1,
                            uint8_t              value2,
                            uint8_t              value3,
                      const uint8_t * __restrict mask,
                            uint32_t             maskStride
                    );

//------------------------------------------------------------------------------
/// @brief
///   Sets every element of an int32_t 3-channel  array to a given 3-element scalar.
///
/// @details
///   A non-zero element of the mask array indicates the corresponding element 
///   of the destination array to be changed. The mask itself equals to zero means that
///   all elements of the dst array need to be changed. The mask is assumed to
///    have the same width and height( in terms of pixels) as the destination array.
///
/// @param dst
///    The destination matrix
///
/// @param dstWidth
///    Destination matrix width
///
/// @param dstHeight
///    Destination matrix height
///
/// @param dstStride
///    Stride for the destination matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param value1
///    First int32_t value of the Scalar
///
/// @param value2
///    Second int32_t value of the Scalar
///
/// @param value3
///    Third int32_t value of the Scalar
///
/// @param mask
///    Operation mask, 8-bit single channel array; specifies elements of the src
///    array to be changed. 
///
/// @param maskStride
///    Stride for input mask, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @return
///   No return value
///
/// @ingroup math_vector 
//------------------------------------------------------------------------------
FASTCV_API void
fcvSetElementsc3s32(         int32_t * __restrict dst, 
                             uint32_t             dstWidth, 
                             uint32_t             dstHeight,
                             uint32_t             dstStride, 
                             int32_t              value1,
                             int32_t              value2,
                             int32_t              value3,
                       const uint8_t * __restrict mask,
                             uint32_t             maskStride 
                     );

//------------------------------------------------------------------------------
/// @brief
///   Sets every element of a float32_t 3-channel  array to a given 3-element scalar.
///
/// @details
///   A non-zero element of the mask array indicates the corresponding element 
///   of the destination array to be changed. The mask itself equals to zero means that
///   all elements of the dst array need to be changed. The mask is assumed to
///    have the same width and height( in terms of pixels) as the destination array.
///
/// @param dst
///    The destination matrix
///
/// @param dstWidth
///    Destination matrix width
///
/// @param dstHeight
///    Destination matrix height
///
/// @param dstStride
///    Stride for the destination matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param value1
///    First float32_t value of the Scalar
///
/// @param value2
///    Second float32_t value of the Scalar
///
/// @param value3
///    Third float32_t value of the Scalar
///
/// @param mask
///    Operation mask, 8-bit single channel array; specifies elements of the src
///    array to be changed
///
/// @param maskStride
///    Stride for input mask, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @return
///   No return value
///
/// @ingroup math_vector 
//------------------------------------------------------------------------------
FASTCV_API void
fcvSetElementsc3f32(         float32_t * __restrict dst, 
                             uint32_t               dstWidth, 
                             uint32_t               dstHeight,
                             uint32_t               dstStride, 
                             float32_t              value1,
                             float32_t              value2,
                             float32_t              value3,
                       const uint8_t   * __restrict mask,
                             uint32_t               maskStride
                     );


//------------------------------------------------------------------------------
/// @brief
///   Defines an enumeration to list threshold types used in fcvAdaptiveThreshold
//------------------------------------------------------------------------------

typedef enum {
    FCV_THRESH_BINARY      = 0,   // value = value > threshold ? max_value : 0      
    FCV_THRESH_BINARY_INV       // value = value > threshold ? 0 : max_value
} fcvThreshType;


//---------------------------------------------------------------------------
/// @brief
///   Binarizes a grayscale image based on an adaptive threshold value calculated from 3x3 Gaussian kernel.  
///
/// @details
///   For each pixel, the threshold is computed adaptively based on cross-correlation with a
///   3x3 Gaussian kernel minus value (parameter). The standard deviation is used for Gaussian kernel.
///   For FCV_THRESH_BINARY threshold type, the pixel is set as maxValue if it's value is greater than the threshold;
///   else, it is set as zero. For FCV_THRESH_BINARY_INV threshold type, the pixel is set as zero if it's value is greater than the threshold;
///   else, it is set as maxValue.   
///
/// @param src
///   Pointer to the 8-bit input image.
///
/// @param srcWidth
///   Width of source images pointed by src.
///
/// @param srcHeight
///   Height of source images pointed by src.
///
/// @param srcStride
///   Stride of source image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///
/// @param maxValue
///   The maximum integer value to be used. 0<maxValue<256.
///
/// @param thresholdType
///   Threshold type. It could be either FCV_THRESH_BINARY or FCV_THRESH_BINARY_INV.
///
/// @param value 
///   The constant value subtracted after the cross-correlation with Gaussian kernel. 
///   It is usually positive but could be 0 or negative too.
///
/// @param dst
///   Pointer to the 8-bit destination image. Destination iamge has the same size as input image. 
///
/// @param dstStride
///   Stride of destination image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///
/// @ingroup image_processing
//---------------------------------------------------------------------------

FASTCV_API void
fcvAdaptiveThresholdGaussian3x3u8( const uint8_t* __restrict src,
                        uint32_t             srcWidth,
                        uint32_t             srcHeight,
                        uint32_t             srcStride,
                        uint8_t              maxValue,
                        fcvThreshType        thresholdType,
                        int32_t              value,
                        uint8_t* __restrict  dst,
                        uint32_t             dstStride );

//---------------------------------------------------------------------------
/// @brief
///   Binarizes a grayscale image based on an adaptive threshold value calculated from 5x5 Gaussian kernel.  
///
/// @details
///   For each pixel, the threshold is computed adaptively based on cross-correlation with a
///   5x5 Gaussian kernel minus value (parameter). The standard deviation is used for Gaussian kernel.
///   For FCV_THRESH_BINARY threshold type, the pixel is set as maxValue if it's value is greater than the threshold;
///   else, it is set as zero. For FCV_THRESH_BINARY_INV threshold type, the pixel is set as zero if it's value is greater than the threshold;
///   else, it is set as maxValue.    
///
/// @param src
///   Pointer to the 8-bit input image.
///
/// @param srcWidth
///   Width of source images pointed by src.
///
/// @param srcHeight
///   Height of source images pointed by src.
///
/// @param srcStride
///   Stride of source image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///
/// @param maxValue
///   The maximum integer value to be used. 0<maxValue<256.
///
/// @param thresholdType
///   Threshold type. It could be either FCV_THRESH_BINARY or FCV_THRESH_BINARY_INV.
///
/// @param value 
///   The constant value subtracted after the cross-correlation with Gaussian kernel. 
///   It is usually positive but could be 0 or negative too.
///
/// @param dst
///   Pointer to the 8-bit destination image. Destination iamge has the same size as input image. 
///
/// @param dstStride
///   Stride of destination image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///
/// @ingroup image_processing
//---------------------------------------------------------------------------
FASTCV_API void
fcvAdaptiveThresholdGaussian5x5u8( const uint8_t* __restrict src,
                        uint32_t             srcWidth,
                        uint32_t             srcHeight,
                        uint32_t             srcStride,
                        uint8_t              maxValue,
                        fcvThreshType        thresholdType,
                        int32_t              value,
                        uint8_t* __restrict  dst,
                        uint32_t             dstStride );

//---------------------------------------------------------------------------
/// @brief
///   Binarizes a grayscale image based on an adaptive threshold value calculated from 11x11 Gaussian kernel.  
///
/// @details
///   For each pixel, the threshold is computed adaptively based on cross-correlation with a
///   11x11 Gaussian kernel minus value (parameter). The standard deviation is used for Gaussian kernel.
///   For FCV_THRESH_BINARY threshold type, the pixel is set as maxValue if it's value is greater than the threshold;
///   else, it is set as zero. For FCV_THRESH_BINARY_INV threshold type, the pixel is set as zero if it's value is greater than the threshold;
///   else, it is set as maxValue.  
///
/// @param src
///   Pointer to the 8-bit input image.
///
/// @param srcWidth
///   Width of source images pointed by src.
///
/// @param srcHeight
///   Height of source images pointed by src.
///
/// @param srcStride
///   Stride of source image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///
/// @param maxValue
///   The maximum integer value to be used. 0<maxValue<256.
///
/// @param thresholdType
///   Threshold type. It could be either FCV_THRESH_BINARY or FCV_THRESH_BINARY_INV.
///
/// @param value 
///   The constant value subtracted after the cross-correlation with Gaussian kernel. 
///   It is usually positive but could be 0 or negative too.
///
/// @param dst
///   Pointer to the 8-bit destination image. Destination iamge has the same size as input image. 
///
/// @param dstStride
///   Stride of destination image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///
/// @ingroup image_processing
//---------------------------------------------------------------------------
FASTCV_API void
fcvAdaptiveThresholdGaussian11x11u8( const uint8_t* __restrict src,
                        uint32_t             srcWidth,
                        uint32_t             srcHeight,
                        uint32_t             srcStride,
                        uint8_t              maxValue,
                        fcvThreshType        thresholdType,
                        int32_t              value,
                        uint8_t* __restrict  dst,
                        uint32_t             dstStride );

//---------------------------------------------------------------------------
/// @brief
///   Binarizes a grayscale image based on an adaptive threshold value calculated from 3x3 mean.  
///
/// @details
///   For each pixel, the threshold is computed adaptively based on the mean of 3x3 block centered on the pixel 
///   minus value (parameter). For FCV_THRESH_BINARY threshold type, the pixel is set as maxValue if it's value is greater than the threshold;
///   else, it is set as zero. For FCV_THRESH_BINARY_INV threshold type, the pixel is set as zero if it's value is greater than the threshold;
///   else, it is set as maxValue.  
///
/// @param src
///   Pointer to the 8-bit input image.
///
/// @param srcWidth
///   Width of source images pointed by src.
///
/// @param srcHeight
///   Height of source images pointed by src.
///
/// @param srcStride
///   Stride of source image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///
/// @param maxValue
///   The maximum integer value to be used. 0<maxValue<256.
///
/// @param thresholdType
///   Threshold type. It could be either FCV_THRESH_BINARY or FCV_THRESH_BINARY_INV.
///
/// @param value 
///   The constant value subtracted from the mean. 
///   It is usually positive but could be 0 or negative too.
///
/// @param dst
///   Pointer to the 8-bit destination image. Destination iamge has the same size as input image. 
///
/// @param dstStride
///   Stride of destination image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///
/// @ingroup image_processing
//---------------------------------------------------------------------------
FASTCV_API void
fcvAdaptiveThresholdMean3x3u8( const uint8_t* __restrict src,
                        uint32_t             srcWidth,
                        uint32_t             srcHeight,
                        uint32_t             srcStride,
                        uint8_t              maxValue,
                        fcvThreshType        thresholdType,
                        int32_t              value,
                        uint8_t* __restrict  dst,
                        uint32_t             dstStride );


//---------------------------------------------------------------------------
/// @brief
///   Binarizes a grayscale image based on an adaptive threshold value calculated from 5x5 mean.  
///
/// @details
///   For each pixel, the threshold is computed adaptively based on the mean of 5x5 block centered on the pixel 
///   minus value (parameter). For FCV_THRESH_BINARY threshold type, the pixel is set as maxValue if it's value is greater than the threshold;
///   else, it is set as zero. For FCV_THRESH_BINARY_INV threshold type, the pixel is set as zero if it's value is greater than the threshold;
///   else, it is set as maxValue.  
///
/// @param src
///   Pointer to the 8-bit input image.
///
/// @param srcWidth
///   Width of source images pointed by src.
///
/// @param srcHeight
///   Height of source images pointed by src.
///
/// @param srcStride
///   Stride of source image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///
/// @param maxValue
///   The maximum integer value to be used. 0<maxValue<256.
///
/// @param thresholdType
///   Threshold type. It could be either FCV_THRESH_BINARY or FCV_THRESH_BINARY_INV.
///
/// @param value 
///   The constant value subtracted from the mean. 
///   It is usually positive but could be 0 or negative too.
///
/// @param dst
///   Pointer to the 8-bit destination image. Destination iamge has the same size as input image. 
///
/// @param dstStride
///   Stride of destination image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///
/// @ingroup image_processing
//---------------------------------------------------------------------------
FASTCV_API void
fcvAdaptiveThresholdMean5x5u8( const uint8_t* __restrict src,
                        uint32_t             srcWidth,
                        uint32_t             srcHeight,
                        uint32_t             srcStride,
                        uint8_t              maxValue,
                        fcvThreshType        thresholdType,
                        int32_t              value,
                        uint8_t* __restrict  dst,
                        uint32_t             dstStride );
//---------------------------------------------------------------------------
/// @brief
///   Binarizes a grayscale image based on an adaptive threshold value calculated from 11x11 mean.  
///
/// @details
///   For each pixel, the threshold is computed adaptively based on the mean of 11x11 block centered on the pixel 
///   minus value (parameter). For FCV_THRESH_BINARY threshold type, the pixel is set as maxValue if it's value is greater than the threshold;
///   else, it is set as zero. For FCV_THRESH_BINARY_INV threshold type, the pixel is set as zero if it's value is greater than the threshold;
///   else, it is set as maxValue.  
///
/// @param src
///   Pointer to the 8-bit input image.
///
/// @param srcWidth
///   Width of source images pointed by src.
///
/// @param srcHeight
///   Height of source images pointed by src.
///
/// @param srcStride
///   Stride of source image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///
/// @param maxValue
///   The maximum integer value to be used. 0<maxValue<256.
///
/// @param thresholdType
///   Threshold type. It could be either FCV_THRESH_BINARY or FCV_THRESH_BINARY_INV.
///
/// @param value 
///   The constant value subtracted from the mean. 
///   It is usually positive but could be 0 or negative too.
///
/// @param dst
///   Pointer to the 8-bit destination image. Destination iamge has the same size as input image. 
///
/// @param dstStride
///   Stride of destination image (i.e., number of bytes between column 0 
///   of row 0 and column 0 of row 1).
///
/// @ingroup image_processing
//---------------------------------------------------------------------------
FASTCV_API void
fcvAdaptiveThresholdMean11x11u8( const uint8_t* __restrict src,
                        uint32_t             srcWidth,
                        uint32_t             srcHeight,
                        uint32_t             srcStride,
                        uint8_t              maxValue,
                        fcvThreshType        thresholdType,
                        int32_t              value,
                        uint8_t* __restrict  dst,
                        uint32_t             dstStride );

//---------------------------------------------------------------------------
/// @brief
///   Smooth a uint8_t image with a 3x3 box filter
///
/// @details
///   smooth with 3x3 box kernel and normalize:
///   \n[ 1 1 1
///   \n  1 1 1
///   \n  1 1 1 ]/9
///
/// @param src
///   Input uint8_t image.
///   
/// @param srcWidth
///   Input image width.
///
/// @param srcHeight
///   Input image height.
///
/// @param srcStride
///   Input image stride, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param dst
///   Output image which has the same type, and size as the input image.
///   
/// @param dstStride
///   Output image stride, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @return
///   No return value
///
/// @ingroup image_processing
//---------------------------------------------------------------------------

FASTCV_API void 
fcvBoxFilter3x3u8( const uint8_t* __restrict src, 
                         uint32_t            srcWidth,
                         uint32_t            srcHeight,
                         uint32_t            srcStride, 
                         uint8_t* __restrict dst, 
                         uint32_t            dstStride 
                   );

//---------------------------------------------------------------------------
/// @brief
///   Smooth a uint8_t image with a 5x5 box filter
///
/// @details
///   smooth with 5x5 box kernel and normalize:
///   \n[ 1 1 1 1 1
///   \n  1 1 1 1 1
///   \n  1 1 1 1 1
///   \n  1 1 1 1 1
///   \n  1 1 1 1 1 ]/25
///   
/// @param src
///   Input uint8_t image.
///   
/// @param srcWidth
///   Input image width.
///
/// @param srcHeight
///   Input image height.
///
/// @param srcStride
///   Input image stride, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param dst
///   Output image which has the same type, and size as the input image.
///   
/// @param dstStride
///   Output image stride, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @return
///   No return value
///
/// @ingroup image_processing
//---------------------------------------------------------------------------

FASTCV_API void 
fcvBoxFilter5x5u8( const uint8_t* __restrict src, 
                         uint32_t            srcWidth,
                         uint32_t            srcHeight,
                         uint32_t            srcStride, 
                         uint8_t* __restrict dst, 
                         uint32_t            dstStride 
                   );

//---------------------------------------------------------------------------
/// @brief
///   Smooth a uint8_t image with a 11x11 box filter
///
/// @details
///   smooth with 11x11 box kernel and normalize:
///
/// @param src
///   Input uint8_t image.
///   
/// @param srcWidth
///   Input image width.
///
/// @param srcHeight
///   Input image height.
///
/// @param srcStride
///   Input image stride, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param dst
///   Output image which has the same type, and size as the input image.
///   
/// @param dstStride
///   Output image stride, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @return
///   No return value
///
/// @ingroup image_processing
//---------------------------------------------------------------------------

FASTCV_API void 
fcvBoxFilter11x11u8(const uint8_t* __restrict src, 
                          uint32_t            srcWidth,
                          uint32_t            srcHeight,
                          uint32_t            srcStride, 
                          uint8_t* __restrict dst, 
                          uint32_t            dstStride 
                   );


//---------------------------------------------------------------------------
/// @brief
///   bilateral smoothing with a 5x5 bilateral kernel
///
/// @details
///   The bilateral filter applied here considered 5-pixel diameter of each pixel's neighborhood 
/// and both the filter sigma in color space and the sigma in coordinate space are set to 50 
///
/// @param src
///   Input uint8_t image.
///   
/// @param srcWidth
///   Input image width.
///
/// @param srcHeight
///   Input image height.
///
/// @param srcStride
///   Input image stride, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param dst
///   Output image which has the same type, and size as the input image.
///   
/// @param dstStride
///   Output image stride, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @return
///   No return value
///
/// @ingroup image_processing
//---------------------------------------------------------------------------
FASTCV_API void 
fcvBilateralFilter5x5u8(const uint8_t* __restrict src, 
                               uint32_t            srcWidth,
                               uint32_t            srcHeight,
                               uint32_t            srcStride, 
                               uint8_t* __restrict dst, 
                               uint32_t            dstStride 
                        );


//---------------------------------------------------------------------------
/// @brief
///   Bilateral smoothing with 7x7 bilateral kernel
///
/// @details
///   The bilateral filter applied here considered 7-pixel diameter of each pixel's neighborhood 
/// and both the filter sigma in color space and the sigma in coordinate space are set to 50 
///
/// @param src
///   Input uint8_t image.
///   
/// @param srcWidth
///   Input image width.
///
/// @param srcHeight
///   Input image height.
///
/// @param srcStride
///   Input image stride, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param dst
///   Output image which has the same type, and size as the input image.
///   
/// @param dstStride
///   Output image stride, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @return
///   No return value
///
/// @ingroup image_processing
//---------------------------------------------------------------------------
FASTCV_API void 
fcvBilateralFilter7x7u8(const uint8_t* __restrict src, 
                               uint32_t            srcWidth,
                               uint32_t            srcHeight,
                               uint32_t            srcStride, 
                               uint8_t* __restrict dst, 
                               uint32_t            dstStride 
                    );

//---------------------------------------------------------------------------
/// @brief
///   Bilateral smoothing with 9x9 bilateral kernel
///
/// @details
///   The bilateral filter applied here considered 9-pixel diameter of each pixel's neighborhood 
/// and both the filter sigma in color space and the sigma in coordinate space are set to 50 
///
/// @param src
///   Input uint8_t image.
///   
/// @param srcWidth
///   Input image width.
///
/// @param srcHeight
///   Input image height.
///
/// @param srcStride
///   Input image stride, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param dst
///   Output image which has the same type, and size as the input image.
///   
/// @param dstStride
///   Output image stride, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @return
///   No return value
///
/// @ingroup image_processing
//---------------------------------------------------------------------------
FASTCV_API void 
fcvBilateralFilter9x9u8(const uint8_t* __restrict src, 
                               uint32_t            srcWidth,
                               uint32_t            srcHeight,
                               uint32_t            srcStride, 
                               uint8_t* __restrict dst, 
                               uint32_t            dstStride );

//---------------------------------------------------------------------------
/// @brief
///   This function will remove small patches in the source image based on the input threshold.
///
/// @details
///   The function will remove the small contoured area of the source image. The input is a 8 bit 
/// grayscale image, where zero value denotes the background. 
///
/// @param src
///   The input image/patch. Must be 8 bit grayscale and zero value indicates the background.
///
/// @param srcWidth
///   The width of the input source image.
///
/// @param srcHeight
///   The height of the input source image.
///
/// @param srcStride
///   The stride of the input source image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///
/// @param Polygonal
///   If it is 0 then we use convex hull to do approximation on the original contour, otherwise we do
/// polygonal approximation. Currently it simple use the original contour, the parameter will be
/// valid after the convex hull or polygonal approximation function is ready.
///
/// @param perimScale
///   The minimum perimscale of the contours; If a contour's perimeter is smaller than this value,
/// It will be removed from the original source image.
///
/// @return
///   No return value.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------
FASTCV_API void
fcvSegmentFGMasku8(uint8_t* __restrict    src,
                   uint32_t               srcWidth,
                   uint32_t               srcHeight,
                   uint32_t               srcStride,
                   uint8_t                Polygonal,
                   uint32_t               perimScale);

//------------------------------------------------------------------------------
/// @brief
///   Computes the per-element absolute difference between two 
///   uint8_t matrices
///
/// @details
///   
///
/// @param src1
///    The first input matrix
///
/// @param src2
///    Second input matrix which has the same width and length as src1
///
/// @param srcWidth
///    Input matrix width
///
/// @param srcHeight
///    Input matrix height
///
/// @param srcStride
///    Stride for the input matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row 
///
/// @param dst
///    Output matrix which has the same width and length as src1
///
/// @param dstStride
///   Stride for output image, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row 
///
/// @return
///   No return value
///    
/// @ingroup math_vector 
//------------------------------------------------------------------------------
FASTCV_API void
fcvAbsDiffu8(const uint8_t * __restrict src1, 
             const uint8_t * __restrict src2,   
                   uint32_t             srcWidth,
                   uint32_t             srcHeight,
                   uint32_t             srcStride,
                   uint8_t * __restrict dst,
                   uint32_t             dstStride );

//------------------------------------------------------------------------------
/// @brief
///   Computes the per-element absolute difference between two 
///   int32_t matrices
///
/// @details
///   
///
/// @param src1
///    The first input matrix
///
/// @param src2
///    Second input matrix which has the same width and length as src1
///
/// @param srcWidth
///    Input matrix width
///
/// @param srcHeight
///    Input matrix height
///
/// @param srcStride
///    Stride for the input matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param dst
///    Output matrix which has the same width and length as src1
///
/// @param dstStride
///   Stride for output image, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row 
///
/// @return
///   No return value
///    
/// @ingroup math_vector 
//------------------------------------------------------------------------------

FASTCV_API void
fcvAbsDiffs32(const int32_t * __restrict  src1, 
              const int32_t * __restrict  src2,   
                    uint32_t              srcWidth,
                    uint32_t              srcHeight,
                    uint32_t              srcStride,
                    int32_t * __restrict  dst,
                    uint32_t              dstStride );

//------------------------------------------------------------------------------
/// @brief
///   Computes the per-element absolute difference between two 
///   float32_t matrices
///
/// @details
///   
///
/// @param src1
///    First input matrix
///
/// @param src2
///    Second input matrix which has the same width and length as src1
///
/// @param srcWidth
///    Input matrix width
///
/// @param srcHeight
///    Input matrix height
///
/// @param srcStride
///    Stride for the input matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param dst
///    Output matrix which has the same width and length as src1
///
/// @param dstStride
///   Stride for output image, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @return
///   No return value
///    
/// @ingroup math_vector 
//------------------------------------------------------------------------------

FASTCV_API void
fcvAbsDifff32(const float32_t * __restrict  src1, 
              const float32_t * __restrict  src2,    
                    uint32_t                srcWidth,
                    uint32_t                srcHeight,
                    uint32_t                srcStride,
                    float32_t * __restrict  dst,
                    uint32_t                dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Computes the per-element absolute difference between one matrix and one value 
///
/// @details
///   
///
/// @param src
///    Input matrix
///
/// @param value
///    Input value
///
/// @param srcWidth
///    Input matrix width
///
/// @param srcHeight
///    Input matrix height
///
/// @param srcStride
///    Stride for the input matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param dst 
///    Output matrix which has the same width and length as src
///
/// @param dstStride
///   Stride for output image, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row 
///
/// @return
///   No return value
///    
/// @ingroup math_vector 
//------------------------------------------------------------------------------

FASTCV_API void
fcvAbsDiffVu8(const uint8_t * __restrict src, 
                    uint8_t              value,   
                    uint32_t             srcWidth,
                    uint32_t             srcHeight,
                    uint32_t             srcStride,
                    uint8_t * __restrict dst,
                    uint32_t             dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Computes the per-element absolute difference between one matrix and one value 
///
/// @details
///   
///
/// @param src
///    Input matrix
///
/// @param value
///    Input value
///
/// @param srcWidth
///    Input matrix width
///
/// @param srcHeight
///    Input matrix height
///
/// @param srcStride
///    Stride for the input matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param dst
///    Output matrix which has the same width and length as src
///
/// @param dstStride
///   Stride for output image , i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @return
///   No return value
///    
/// @ingroup math_vector 
//------------------------------------------------------------------------------

FASTCV_API void
fcvAbsDiffVs32(const int32_t * __restrict src, 
                     int32_t              value, 
                     uint32_t             srcWidth,
                     uint32_t             srcHeight,
                     uint32_t             srcStride,
                     int32_t * __restrict dst,
                     uint32_t             dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Computes the per-element absolute difference between one matrix and one value 
///
/// @details
///   
///
/// @param src
///    Input matrix
///
/// @param value
///    Input value
///
/// @param srcWidth
///    Input matrix width
///
/// @param srcHeight
///    Input matrix height
///
/// @param srcStride
///    Stride for the input matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param dst
///    Output matrix which has the same width and length as src
///
/// @param dstStride
///   Stride for output image, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row 
///
/// @return
///   No return value
///    
/// @ingroup math_vector 
//------------------------------------------------------------------------------

FASTCV_API void
fcvAbsDiffVf32(const float32_t * __restrict src, 
                     float32_t              value,   
                     uint32_t               srcWidth,
                     uint32_t               srcHeight,
                     uint32_t               srcStride,
                     float32_t * __restrict dst,
                     uint32_t               dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Computes the per-element absolute difference between one 4-channel matrix and a 4-element Scalar  
///
/// @details
///   
///
/// @param src
///    Input matrix
///
/// @param value1
///    First value of the Scalar
///
/// @param value2
///    Second value of the Scalar
///
/// @param value3
///    Third value of the Scalar
///
/// @param value4
///    Fourth value of the Scalar
///
/// @param srcWidth
///    Input matrix width
///
/// @param srcHeight
///    Input matrix height
///
/// @param srcStride
///    Stride for the input matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param dst
///    Output matrix which has the same width, length and channel number as src
///
/// @param dstStride
///   Stride for output image, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row 
///
/// @return
///   No return value
///    
/// @ingroup math_vector 
//------------------------------------------------------------------------------

FASTCV_API void
fcvAbsDiffVc4u8(const uint8_t * __restrict src, 
                    uint8_t              value1,
                    uint8_t              value2,
                    uint8_t              value3,
                    uint8_t              value4,
                    uint32_t             srcWidth,
                    uint32_t             srcHeight,
                    uint32_t             srcStride,
                    uint8_t * __restrict dst,
                    uint32_t             dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Computes the per-element absolute difference between one 4-channel matrix and a 4-element Scalar 
/// 
/// @details
///   
///
/// @param src
///    Input matrix
///
/// @param value1
///    First value of the Scalar
///
/// @param value2
///    Second value of the Scalar
///
/// @param value3
///    Third value of the Scalar
///
/// @param value4
///    Fourth value of the Scalar
///
/// @param srcWidth
///    Input matrix width
///
/// @param srcHeight
///    Input matrix height
///
/// @param srcStride
///    Stride for the input matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param dst
///    Output matrix which has the same width, length and channel number as src
///
/// @param dstStride
///   Stride for output image, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row 
///
/// @return
///   No return value
///    
/// @ingroup math_vector 
//------------------------------------------------------------------------------

FASTCV_API void
fcvAbsDiffVs32c4(const int32_t * __restrict src, 
                     int32_t              value1,
                     int32_t              value2,
                     int32_t              value3,
                     int32_t              value4,
                     uint32_t             srcWidth,
                     uint32_t             srcHeight,
                     uint32_t             srcStride,
                     int32_t * __restrict dst,
                     uint32_t             dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Computes the per-element absolute difference between one 4-channel matrix and a 4-element Scalar 
///
/// @details
///   
///
/// @param src
///    Input matrix
///
/// @param value1
///    First value of the Scalar
///
/// @param value2
///    Second value of the Scalar
///
/// @param value3
///    Third value of the Scalar
///
/// @param value4
///    Fourth value of the Scalar
///
/// @param srcWidth
///    Input matrix width
///
/// @param srcHeight
///    Input matrix height
///
/// @param srcStride
///    Stride for the input matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param dst
///    Output matrix which has the same width, length and channel number as src
///
/// @param dstStride
///   Stride for output image , i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @return
///   No return value
///    
/// @ingroup math_vector 
//------------------------------------------------------------------------------

FASTCV_API void
fcvAbsDiffVc4f32(const float32_t * __restrict src, 
                     float32_t              value1,   
                     float32_t              value2,
                     float32_t              value3,
                     float32_t              value4,
                     uint32_t               srcWidth,
                     uint32_t               srcHeight,
                     uint32_t               srcStride,
                     float32_t * __restrict dst,
                     uint32_t               dstStride);

//------------------------------------------------------------------------------
/// @brief
///   Computes the per-element absolute difference between one 3-channel matrix and a 3-element Scalar  
///
/// @details
///   
///
/// @param src
///    Input matrix
///
/// @param value1
///    First value of the Scalar
///
/// @param value2
///    Second value of the Scalar
///
/// @param value3
///    Third value of the Scalar
///
/// @param srcWidth
///    Input matrix width
///
/// @param srcHeight
///    Input matrix height
///
/// @param srcStride
///    Stride for the input matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param dst
///    Output matrix which has the same width, length and channel number as src
///
/// @param dstStride
///   Stride for output image, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row 
///
/// @return
///   No return value
///    
/// @ingroup math_vector 
//------------------------------------------------------------------------------

FASTCV_API void
fcvAbsDiffVc3u8(const uint8_t * __restrict src, 
                    uint8_t              value1,
                    uint8_t              value2,
                    uint8_t              value3,
                    uint32_t             srcWidth,
                    uint32_t             srcHeight,
                    uint32_t             srcStride,
                    uint8_t * __restrict dst,
                    uint32_t             dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Computes the per-element absolute difference between one 3-channel matrix and a 3-element Scalar 
/// 
/// @details
///   
///
/// @param src
///    Input matrix
///
/// @param value1
///    First value of the Scalar
///
/// @param value2
///    Second value of the Scalar
///
/// @param value3
///    Third value of the Scalar
///
/// @param srcWidth
///    Input matrix width
///
/// @param srcHeight
///    Input matrix height
///
/// @param srcStride
///    Stride for the input matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param dst
///    Output matrix which has the same width, length and channel number as src
///
/// @param dstStride
///   Stride for output image, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row 
///
/// @return
///   No return value
///
/// @ingroup math_vector 
//------------------------------------------------------------------------------

FASTCV_API void
fcvAbsDiffVc3s32(const int32_t * __restrict src, 
                     int32_t              value1,
                     int32_t              value2,
                     int32_t              value3,
                     uint32_t             srcWidth,
                     uint32_t             srcHeight,
                     uint32_t             srcStride,
                     int32_t * __restrict dst,
                     uint32_t             dstStride );


//------------------------------------------------------------------------------
/// @brief
///   Computes the per-element absolute difference between one 3-channel matrix and a 3-element Scalar 
///
/// @details
///   
///
/// @param src
///    Input matrix
///
/// @param value1
///    First value of the Scalar
///
/// @param value2
///    Second value of the Scalar
///
/// @param value3
///    Third value of the Scalar
///
/// @param srcWidth
///    Input matrix width
///
/// @param srcHeight
///    Input matrix height
///
/// @param srcStride
///    Stride for the input matrix, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///
/// @param dst
///    Output matrix which has the same width, length and channel number as src
///
/// @param dstStride
///   Stride for output image, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row 
///
/// @return
///   No return value
///    
/// @ingroup math_vector 
//------------------------------------------------------------------------------

FASTCV_API void
fcvAbsDiffVc3f32(const float32_t * __restrict src, 
                     float32_t              value1,   
                     float32_t              value2,
                     float32_t              value3,
                     uint32_t               srcWidth,
                     uint32_t               srcHeight,
                     uint32_t               srcStride,
                     float32_t * __restrict dst,
                     uint32_t               dstStride);

// -----------------------------------------------------------------------------
/// @brief
///   create KDTrees for dataset of 36D vectors
///
/// @details
///   KDTree is very efficient search structure for multidimensional data.
///   Usage:
///   assume we have 36D of type int8_t data e.g. target image feature 
///   descriptors in array named vectors
///
///      int8_t* descriptors;
///
///   the number of descriptors is numVectors
///
///      int numDescriptors;
///
///   the inverse length of each descriptor in array invLengths
///
///      float32_t* invLenghts;
///     
/// pointer to KDTree structure
///         
///      fcvKDTreeDatas8f32* kdTree = 0;
///
///   kdTree is created as
///
///      int err = fcvKDTreeCreate36s8f32( descriptors, invLengths, 
///                                        numDescriptors, kdTree );
/// @param vectors
///   pointer to dataset being array of 36D vectors
///
/// @param invLengths
///   array of inverse lengths for each vector in the dataset
///
/// @param numVectors
///   number of 36D vectors in the dataset
///
/// @param kdtrees
///   address for pointer to the newly created KDTrees
///
/// @return
///   0      - success
///   EINVAL - invalid parameter
///   ENOMEM - not enough memory
///   -1     - other error
///
/// @ingroup feature_detection
// -----------------------------------------------------------------------------
FASTCV_API int
fcvKDTreeCreate36s8f32( const        int8_t*  __restrict vectors,
                            const     float32_t*  __restrict invLengths,
                                            int              numVectors,
                             fcvKDTreeDatas8f32**            kdtrees );

// -----------------------------------------------------------------------------
/// @brief
///   release KDTrees data structures
///
/// @details
///   Once we are done with all searches we should release kdTree resources
///
/// @param kdtrees
///   KDTrees to be destroyed
///
/// @return
///   0      - success
///   EINVAL - invalid parameter
///   -1     - other error
///
/// @ingroup feature_detection
// -----------------------------------------------------------------------------
FASTCV_API int 
fcvKDTreeDestroy36s8f32( fcvKDTreeDatas8f32* kdtrees );

// -----------------------------------------------------------------------------
/// @brief
///   find nearest neighbors (NN) for query
///
/// @details
///   Assuming KD tree creation is successful we may start using our kdTree
///   for nearest neighbors (NN) for descriptors of camera features. Let our
///   camera descriptors be in array camDescriptors and their number
///   in numCamDescriptors
///
///      int8_t* camDescriptors;
///      int numCamDescriptors;
///
///   The inverse lengths of descriptors is in
///
///      float* camDescriptorsInvLengths;
///         
/// Assume we want to find 8 NNs for each camera
///   descriptor. We declare variables for results of NN searches
///
///      \#define NUM_NN 8                 // number of NN required
///      \#define MAX_CHECKS 32            // max number of checks in kdtree
///
///      int32_t numFound = 0;            // for numer of NNs found
///      int32_t foundInds[ NUM_NN ];     // for indices to target descriptors
///      float32_t foundDists[ NUM_NN ];  // for distances to target descriptors
///      float32_t maxDist = 0.1f;        // max distance to query allowed
///      
///   the search for NNs for i-th query would be like this
///
///      err = fcvKDTreeQuery36s8f32( kdTree, camDescriptors + i * 36, 
///            camDescriptorsInvLengths[ i ], maxDist, MAX_CHECKS, 0,
///            &numFound, foundInds, foundDists );
///
///   where maxDists is an upper bound on distance of NN from the query
///   and MAX_CHECKS is max number of comparisons of query to target
///   descriptors. The higher MAX_CHECKS the better NNs we get at the cost
///   of longer search. Assuming everything went fine will return us
///   search results. numFound will contain the number of target descriptors
///   found whose distance to query is less than maxDist. foundInds will
///   contain indices to target descriptors being NNs and foundDists their
///   distances to query.
///
/// @param kdtrees
///   KDTrees
///
/// @param query
///   query vector
///
/// @param queryInvLen
///   inverse length of query vector
///
/// @param maxNNs
///   max number of NNs to be found
///
/// @param maxDist
///   max distance between NN and query
///
/// @param maxChecks
///   max number of leafs to check
///
/// @param mask
///   array of flags for all vectors in the dataset; may be NULL;
///   if not NULL then its legth must be equal to number of dataset
///   vectors and i-th mask corresponds to i-th vector; values:
///      0x00 - corresponding vector must not be considered NN regardless
///             of its distance to query
///      0xFF - corresponding vector may be candidate for NN
///      other - not supported
/// @param numNNsFound
///   for number of NNs found
///
/// @param NNInds
///   array for indices of found NNs; must have maxNNs length
///
/// @param NNDists
///   array for NN distances to query; must have maxNNs length
///
/// @return
///   0      - success
///   EINVAL - invalid parameter
///   -1     - other error
///
/// @ingroup feature_detection
// -----------------------------------------------------------------------------
FASTCV_API int
fcvKDTreeQuery36s8f32( fcvKDTreeDatas8f32*       kdtrees,
                           const  int8_t* __restrict query,
                               float32_t             queryInvLen,
                                     int             maxNNs,
                               float32_t             maxDist,
                                     int             maxChecks,
                           const uint8_t* __restrict mask,
                                 int32_t*            numNNsFound,
                                 int32_t* __restrict NNInds,
                               float32_t* __restrict NNDists );

typedef struct fcvConnectedComponent
{
    uint32_t area;    //area of the cc
    uint32_t avgValue; //average value of the cc
    uint32_t rectTopLeftX; // the x of the topleft corner of the bounding box of the cc.
    uint32_t rectTopLeftY; // the y of the topleft corner of the bounding box of the cc.
    uint32_t rectWidth;    // the width of the bounding box of the cc.
    uint32_t rectHeight;   // the height of the bounding box of the cc.
}fcvConnectedComponent;

//---------------------------------------------------------------------------
/// @brief
///   This function fills the image with a starting seed  and
///   neighborhood (4 or 8). It then returns the connected component (cc) that's filled. 
///
/// @details
///   This function first obtains the grayscale value at the (xBegin,yBegin) position of the src
/// image. Then it finds all the neighbor pixels that has the same value based on 4 or 8 connectivity.
/// The corresponding positions of all of these pixels in the image dst then will be set
/// to the new value. Note that the new value cannot be zero since zero is indicating a mask background
/// here. The dst image will have the new value at the corresponding positions and 0 at all
/// other positions. 
///
/// @param src
///   The input image/patch. Must be 8 bit grayscale image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE:must be 128-bit aligned.
///
/// @param srcWidth
///   Image width, the number of pixels in a row
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height.
///
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. 
///   \n\b NOTE: must be a multiple of 8. If left at 0 srcStride is default to srcWidth.
///
/// @param dst
///   The output image/patch. Must be 8 bit grayscale image.
///   \n\b NOTE:must be 128-bit aligned.
///
/// @param dstStride
///   The stride of the output image (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2).
///   \n\b NOTE: must be a multiple of 8. If left at 0 dstStride is default to srcWidth.
///
/// @param xBegin
///   The x coordinate of the pixel where we start the floodfill.
///
/// @param yBegin
///   The y coordinate of the pixel where we start the floodfill.
///
/// @param newVal
///   The new value that will be set on the dst image, correspoinding to the area
///  that's floodfilled starting from the (xBegin,yBegin) position.
///
/// @param cc
///   The pointer that's pointing to the connected component that's representing the 
/// floodfilled area.
///
/// @param connectivity
///   It can be either 4 or 8, indicating whether we use a 4-neighborhood or 
/// 8-neighborhood to do the floodfill.
///
/// @param lineBuffer
///   The input scratch buffer that needs to be allocated by the user and passed in.
/// The size of the buffer must be: Max(srcWidth,srcHeight)*48 bytes.
///   \n\b NOTE:must be 128-bit aligned.
///
/// @return
///   No return value.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------
FASTCV_API void
fcvFloodfillSimpleu8( const uint8_t* __restrict src, 
                           uint32_t             srcWidth, 
                           uint32_t             srcHeight,
                           uint32_t             srcStride,
                            uint8_t* __restrict dst,
                           uint32_t             dstStride, 
                           uint32_t             xBegin, 
                           uint32_t             yBegin, 
                            uint8_t             newVal, //new Val can't be zero. zero is background.
              fcvConnectedComponent*            cc,
                            uint8_t             connectivity,
                               void*            lineBuffer);

//---------------------------------------------------------------------------
/// @brief
///   This function calculates the motion history image.
///
/// @details
///   This function updates the motion history image based on the input motion image. 
/// src is a motion image where pixelvalue!=0 indicates a moving pixel. The function go through
/// all the pixels in the src image. If the value is non zero, it sets the corresponding value
/// of the dst image as the timestamp value. If the value is zero, it compares the corresponding
/// value at the dst image with the timestamp value, if the difference is larger than the
//  maxhistory, it resets the value to zero. 
///
/// @param src
///   The input image/patch. Must be 8 bit grayscale image. Size of buffer is srcStride*srcHeight bytes.
/// \n\b NOTE:must be 128-bit aligned.
///
/// @param srcWidth
///   Image width, the number of pixels in a row
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height.
///
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   The input image/patch. Must be 8 bit grayscale image. Size of buffer is dstStride*srcHeight bytes.
/// \n\b NOTE:must be 128-bit aligned.
///
/// @param dstStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param timeStamp
///   The timestamp value of the current frame that's being updated.
///
/// @param maxHistory
///   The maximum window size that the motion history image will keep.
///
/// @return
///   No return value.
///
/// @ingroup Motion_and_Object_Tracking
//------------------------------------------------------------------------------
FASTCV_API void
fcvUpdateMotionHistoryu8s32( const uint8_t* __restrict src,
                                  uint32_t             srcWidth, 
                                  uint32_t             srcHeight,
                                  uint32_t             srcStride,
                                   int32_t* __restrict dst,
                                  uint32_t             dstStride,
                                   int32_t             timeStamp,
                                   int32_t             maxHistory);


//---------------------------------------------------------------------------
/// @brief
///   This function calculates the integral image of a YCbCr image.
///
/// @details
///   This function calculates the integral images of a YCbCr420 image, where the input YCbCr420 has 
/// UV interleaved. The output is 3 seperate channels. The output integralY will be (srcWidth+1)x(srcHeight+1).
/// IntegralU and IntegralV are (srcWidth/2+1)x(srcHeight/2+1). 
///
/// @param srcY
///   The input image/patch – Y in planar format. 
///   Size of buffer is srcYStride*srcHeight bytes
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcC
///   The input image/patch. Pointer to CbCr are interleaved. Size of buffer is srcCStride*srcHeight/2 bytes
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width, the number of pixels in a row
///   \n\b NOTE: must be a multiple of 16.
///
/// @param srcHeight
///   Image height.
///   \n\b NOTE: must be a multiple of 2.
///
/// @param srcYStride
///   The stride of the input source image's Y channel. (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2). If left at 0 srcYStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 16.
///
/// @param srcCStride
///   The stride of the input source image's CbCr channel. (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2). If left at 0 srcCStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 16.
///
/// @param integralY
///   The output integral image/patch for Y channel. Must be 32 bit image. The size will be
///  (srcWidth+1)x(srcHeight+1). Size of buffer is integralYStride*(srcHeight+1) bytes
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param integralCb
///   The output integral image/patch for Cb channel. Must be 32 bit image. The size will be
///  (srcWidth/2+1)x(srcHeight/2+1). Size of buffer is integralCbStride*(srcHeight/2+1) bytes
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param integralCr
///   The output integral image/patch for Cr channel. Must be 32 bit image. The size will be
///  (srcWidth/2+1)x(srcHeight/2+1). Size of buffer is integralCrStride*(srcHeight/2+1) bytes
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param integralYStride
///   The stride of integralY. (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2). If left at 0 integralYStride is default to (srcWidth+8)*sizeof(uint32_t)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param integralCbStride
///   The stride of integralCb. (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2). If left at 0 integralCbStride is default to (srcWidth>>1+8) *sizeof(uint32_t)
///   \n\b NOTE: must be a multiple of 8.
///
/// @param integralCrStride
///   The stride of integralCr. (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2). If left at 0 integralCrStride is default to (srcWidth>>1+8) *sizeof(uint32_t)
///   \n\b NOTE: must be a multiple of 8.
///
/// @return
///   No return value.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------
FASTCV_API void
fcvIntegrateImageYCbCr420PseudoPlanaru8( 
                        const uint8_t* __restrict srcY,
                        const uint8_t* __restrict srcC,
                             uint32_t             srcWidth,
                             uint32_t             srcHeight,
                             uint32_t             srcYStride,
                             uint32_t             srcCStride,
                             uint32_t* __restrict integralY,
                             uint32_t* __restrict integralCb,
                             uint32_t* __restrict integralCr,
                             uint32_t             integralYStride,
                             uint32_t             integralCbStride,
                             uint32_t             integralCrStride);


//---------------------------------------------------------------------------
/// @brief
///   This function finds the foreground.
///
/// @details
///   This function tries to find a forgound in the current image (represented by: fgIntegralY,
/// fgIntegralCb, fgIntegralCr) based on the current background model (represented by: bgIntegralY,
/// bgIntegralCb, bgIntegralCr). For example, the tuple (bgIntegralY, bgIntegralCb, bgIntegralCr) may be
/// from a picture shooting a wall. Then the tuple (fgIntegralY, fgIntegralCb, fgIntegralCr) may be
/// the wall with a paint on it. Note that all the first six parameters are indicating integral images
/// that's computed from a YUV420 image, which maybe computed from the function: 
/// fcvIntegrateImageYCbCr420PseudoPlanaru8. Generally the size of fgIntegralY and bgIntegralY are
/// (srcWidth+1)*(srcHeight+1). And the size of fgIntegralU, fgIntegralV, bgIntegralU and bgIntegralV
/// are (srcWidth/2+1)*(srcHeight/2+1). The value of the outputWidth and outputHeight are usually indicating
/// the desired block size. For example, if the user wants a 20x15 blocks on a 800x480 image. Then
/// outputWidth=800/20 and outputHeight=480/15. After return, if the value in the outputMask image is
/// 255, then a moving block is indicated, otherwise a non-moving block is indicated. 
///
/// @param bgIntegralY
///   The input image/patch that's indicating the Y channel of the integral image of the background image.
///   Size of buffer is srcYStride*srcHeight bytes
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param bgIntegralCb
///   The input image/patch that's indicating the Cb channel of the integral image of the background image.
///   Size of buffer is srcCbStride*srcHeight/2 bytes
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param bgIntegralCr
///   The input image/patch that's indicating the Cr channel of the integral image of the background image.
///   Size of buffer is srcCrStride*srcHeight/2 bytes
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param fgIntegralY
///   The input image/patch that's indicating the Y channel of the integral image of the image
///  on which we want to find the foreground. 
///   Size of buffer is srcYStride*srcHeight bytes
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param fgIntegralCb
///   The input image/patch that's indicating the Cb channel of the integral image of the image
///   on which we want to find the foreground. 
///   Size of buffer is srcCbStride*srcHeight/2 bytes
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param fgIntegralCr
///   The input image/patch that's indicating the Cr channel of the integral image of the image
///  on which we want to find the foreground. 
///   Size of buffer is srcCrStride*srcHeight/2 bytes
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width, the number of pixels in a row. See the details.
///   \n\b NOTE: must be a multiple of 16.
///
/// @param srcHeight
///   The height of the source image. See the details.
///   \n\b NOTE: must be a multiple of 2.
///
/// @param srcYStride
///   The stride of the input source image's Y channel. (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2). If left at 0 srcStride is default to (srcWidth+8)*sizeof(uint32_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCbStride
///   The stride of the input source image's Cb channel. (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2). If left at 0 srcStride is default to (srcWidth>>1+8)*sizeof(uint32_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcCrStride
///   The stride of the input source image's Cr channel. (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2). If left at 0 srcStride is default to (srcWidth>>1+8)*sizeof(uint32_t).
///   \n\b NOTE: must be a multiple of 8.
///
/// @param outputMask
///   The output mask image. Each pixel represent the motion condition for a block in the original image.
///   Size of buffer is outputMaskStride*outputHeight bytes
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param outputWidth
///   The width of the output mask image.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param outputHeight
///   The height of the output mask image.
///
/// @param outputMaskStride
///   The stride of the output mask image. (i.e., how many bytes between column 0 of row 1 and
///   column 0 of row 2). If left at 0 outputMaskStride is default to outputWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param threshold
///   The threshold that's used to decide if a block is moving or not. (recommend the value of 20).
///
/// @return
///   No return value.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------
FASTCV_API void
fcvFindForegroundIntegrateImageYCbCr420u32(
    const uint32_t * __restrict bgIntegralY,
    const uint32_t * __restrict bgIntegralCb,
    const uint32_t * __restrict bgIntegralCr,
    const uint32_t * __restrict fgIntegralY,
    const uint32_t * __restrict fgIntegralCb,
    const uint32_t * __restrict fgIntegralCr,
          uint32_t              srcWidth,
          uint32_t              srcHeight,
          uint32_t              srcYStride,
          uint32_t              srcCbStride,
          uint32_t              srcCrStride,
           uint8_t * __restrict outputMask,
          uint32_t              outputWidth,
          uint32_t              outputHeight,
          uint32_t              outputMaskStride,
         float32_t              threshold );
      

//---------------------------------------------------------------------------
/// @brief
///   This function calculates the average value of an image.
///
/// @details
///   This function sums all the pixel value in an image and divide the result by the number of pixels in the image.
///
/// @param src
///   The input image/patch. Must be 32 bit image. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width, the number of pixels in a row
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param avgValue
///   The output average value.
///
/// @return
///   No return value.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------
FASTCV_API void
fcvAverages32(
              const int32_t* __restrict src,
                   uint32_t             srcWidth, 
                   uint32_t             srcHeight,             
                   uint32_t             srcStride, 
                  float32_t* __restrict avgValue);

//---------------------------------------------------------------------------
/// @brief
///   This function calculates the average value of an image.
///
/// @details
///   This function sums all the pixel value in an image and divide the result by the number of pixels in the image.
///
/// @param src
///   8-bit image where keypoints are detected. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width, the number of pixels in a row
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param avgValue
///   The output average value.
///
/// @return
///   No return value.
///
/// @ingroup image_processing
//------------------------------------------------------------------------------
FASTCV_API void
fcvAverageu8(
             const uint8_t* __restrict src,
                  uint32_t             srcWidth, 
                  uint32_t             srcHeight, 
                  uint32_t             srcStride,
                 float32_t* __restrict avgValue);


//------------------------------------------------------------------------------
/// @brief
///   Applies the meanshift procedure and obtains the final converged position
///
/// @details
///   This function applies the meanshift procedure to an original image (usually a probability image) and obtains the final converged position.
///   The converged position search will stop either it has reached the required accuracy or the maximum number of iterations. 
///
/// @param src
///   Pointer to the original image which is usually a probability image computed based on object histogram. Must be 8 bit grayscale image.
///   Size of buffer is srcStride*srcHeight bytes.
///   NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   The width of the input source image.
///   NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   The height of the input source image.
///
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth.
///   NOTE: must be a multiple of 8.
///
/// @param window
///   Pointer to the initial search window position which also returns the final converged window position.
///
/// @param criteria
///   The criteria used to finish the MeanShift which consists of two termination criteria: 
///   1) epsilon: required accuracy; 2) max_iter: maximum number of iterations
///
/// @return
///   The actually number of iterations
///
/// @ingroup Motion_and_Object_Tracking
//------------------------------------------------------------------------------
FASTCV_API uint32_t 
fcvMeanShiftu8(const uint8_t* __restrict src,
                    uint32_t             srcWidth,
                    uint32_t             srcHeight,
                    uint32_t             srcStride, 
             fcvRectangleInt*            window, 
             fcvTermCriteria             criteria);

//------------------------------------------------------------------------------
/// @brief
///   Applies the meanshift procedure and obtains the final converged position
///
/// @details
///   This function applies the meanshift procedure to an original image (usually a probability image) and obtains the final converged position.
///   The converged position search will stop either it has reached the required accuracy or the maximum number of iterations. 
///
/// @param src
///   Pointer to the original image which is usually a probability image computed based on object histogram. Must be int 32bit grayscale image.
///   Size of buffer is srcStride*srcHeight bytes.
///   NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   The width of the input source image.
///   NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   The height of the input source image.
///
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth*4.
///   NOTE: must be a multiple of 8.
///
/// @param window
///   Pointer to the initial search window position which also returns the final converged window position.
///
/// @param criteria
///   The criteria used to finish the MeanShift which consists of two termination criteria: 
///   1) epsilon: required accuracy; 2) max_iter: maximum number of iterations
///
/// @return
///   Number of iterations
///
/// @ingroup Motion_and_Object_Tracking
//------------------------------------------------------------------------------
FASTCV_API uint32_t 
fcvMeanShifts32(const int32_t* __restrict src,
                     uint32_t             srcWidth,
                     uint32_t             srcHeight,
                     uint32_t             srcStride, 
              fcvRectangleInt*            window, 
              fcvTermCriteria             criteria);

//------------------------------------------------------------------------------
/// @brief
///   Applies the meanshift procedure and obtains the final converged position
///
/// @details
///   This function applies the meanshift procedure to an original image (usually a probability image) and obtains the final converged position.
///   The converged position search will stop either it has reached the required accuracy or the maximum number of iterations. 
///
/// @param src
///   Pointer to the original image which is usually a probability image computed based on object histogram. Must be float 32bit grayscale image.
///   Size of buffer is srcStride*srcHeight bytes.
///   NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   The width of the input source image.
///   NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   The height of the input source image.
///
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth*4.
///   NOTE: must be a multiple of 8.
///
/// @param window
///   Pointer to the initial search window position which also returns the final converged window position.
///
/// @param criteria
///   The criteria used to finish the MeanShift which consists of two termination criteria: 
///   1) epsilon: required accuracy; 2) max_iter: maximum number of iterations
///
/// @return
///   Number of iterations
///
/// @ingroup Motion_and_Object_Tracking
//------------------------------------------------------------------------------
FASTCV_API uint32_t 
fcvMeanShiftf32(const float32_t* __restrict src,
                       uint32_t             srcWidth,
                       uint32_t             srcHeight,
                       uint32_t             srcStride, 
                fcvRectangleInt*            window, 
                fcvTermCriteria             criteria);


//------------------------------------------------------------------------------
/// @brief
///   Applies the ConAdaTrack procedure and find the object center, size and orientation
///
/// @details
///   This function applies the ConAdaTrack procedure to an original image (usually a probability image) and obtains the final converged object.
///   The optimal object search will stop either it has reached the required accuracy or the maximum number of iterations. 
///
/// @param src
///   Pointer to the original image which is usually a probability image computed based on object histogram. Must be 8bit grayscale image.
///   Size of buffer is srcStride*srcHeight bytes.
///   NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   The width of the input source image.
///   NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   The height of the input source image.
///
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth.
///   NOTE: must be a multiple of 8.
///
/// @param window
///   Pointer to the initial search window position which also returns the final converged window position.
///
/// @param criteria
///   The criteria used to finish the object search which consists of two termination criteria: 
///   1) epsilon: required accuracy; 2) max_iter: maximum number of iterations
///
/// @param circuBox
///   The circumscribed box around the object 
///
/// @return
///   Number of iterations
///
/// @ingroup Motion_and_Object_Tracking
//------------------------------------------------------------------------------
FASTCV_API uint32_t 
fcvConAdaTracku8(const uint8_t* __restrict src,
                   uint32_t             srcWidth,
                   uint32_t             srcHeight,
                   uint32_t             srcStride, 
            fcvRectangleInt*            window,  
            fcvTermCriteria             criteria, 
                   fcvBox2D*            circuBox);


//------------------------------------------------------------------------------
/// @brief
///   Applies the ConAdaTrack procedure and find the object center, size and orientation
///
/// @details
///   This function applies the ConAdaTrack procedure to an original image (usually a probability image) and obtains the final converged object.
///   The optimal object search will stop either it has reached the required accuracy or the maximum number of iterations. 
///
/// @param src
///   Pointer to the original image which is usually a probability image computed based on object histogram. Must be int 32bit grayscale image.
///   Size of buffer is srcStride*srcHeight bytes.
///   NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   The width of the input source image.
///   NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   The height of the input source image.
///
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth*4.
///   NOTE: must be a multiple of 8.
///
/// @param window
///   Pointer to the initial search window position which also returns the final converged window position.
///
/// @param criteria
///   The criteria used to finish the object search which consists of two termination criteria: 
///   1) epsilon: required accuracy; 2) max_iter: maximum number of iterations
///
/// @param circuBox
///   The circumscribed box around the object 
///
/// @return
///   Number of iterations
///
/// @ingroup Motion_and_Object_Tracking
//------------------------------------------------------------------------------
FASTCV_API uint32_t 
fcvConAdaTracks32(const int32_t* __restrict src,
                    uint32_t             srcWidth,
                    uint32_t             srcHeight,
                    uint32_t             srcStride, 
             fcvRectangleInt*            window,  
             fcvTermCriteria             criteria, 
                    fcvBox2D*            circuBox);


//------------------------------------------------------------------------------
/// @brief
///   Applies the ConAdaTrack procedure and find the object center, size and orientation
///
/// @details
///   This function applies the ConAdaTrack procedure to an original image (usually a probability image) and obtains the final converged object.
///   The optimal object search will stop either it has reached the required accuracy or the maximum number of iterations. 
///
/// @param src
///   Pointer to the original image which is usually a probability image computed based on object histogram. Must be float 32bit grayscale image.
///   Size of buffer is srcStride*srcHeight bytes.
///   NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   The width of the input source image.
///   NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   The height of the input source image.
///
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth*4.
///   NOTE: must be a multiple of 8.
///
/// @param window
///   Pointer to the initial search window position which also returns the final converged window position.
///
/// @param criteria
///   The criteria used to finish the object search which consists of two termination criteria: 
///   1) epsilon: required accuracy; 2) max_iter: maximum number of iterations
///
/// @param circuBox
///   The circumscribed box around the object 
///
/// @return
///   Number of iterations
///
/// @ingroup Motion_and_Object_Tracking
//------------------------------------------------------------------------------
FASTCV_API uint32_t 
fcvConAdaTrackf32(const float32_t* __restrict src,
                      uint32_t             srcWidth,
                      uint32_t             srcHeight,
                      uint32_t             srcStride, 
               fcvRectangleInt*            window,  
               fcvTermCriteria             criteria, 
                      fcvBox2D*            circuBox);

//------------------------------------------------------------------------------
/// @brief
///   Compute a singular value decomposition of a matrix of a float type
///         A = U*diag[w]*Vt;
///   It is used for solving problems like least-squares, under-determined linear systems, matrix
///   inversion and so forth. The algorithm used here does not compute the full U and V matrices
///   however it computes a condensed version of U and V described below which is sufficient to solve
///   most problems which use SVD.
///
/// @details
///   
///
/// @param A
///    The input matrix of dimensions m x n
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param m
///    The number of rows of matrix A
///
/// @param n
///    The number of columns of matrix A
///
/// @param w
///    The pointer to the buffer that holds n singular values. When m>n it
///    contains n singular values while when m<n, only the first m singular values
///    are of any significance. However, during allocation, it should be allocated as
///    a buffer to hold n floats.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param U
///    The U matrix whose dimension is m x min(m,n). This is not the full size U 
///    matrix obtained from the conventional SVD algorithm but is sufficient for 
///    solving problems like least-squares, under-determined linear systems, matrix
///    inversion and so forth. While allocating, allocate as a matrix of m x n floats. 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param Vt
///    The V matrix whose dimension is n x min(m,n). This is not the full size V 
///    matrix obtained from the conventional SVD algorithm but is sufficient for 
///    solving problems like least-squares, under-determined linear systems, matrix
///    inversion and so forth. While allocating, allocate as a matrix of n x n floats.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param tmpU
///    Temporary buffer used in processing. It must be allocated as an array of size m x n 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param tmpV
///    Temporary buffer used in processing. It must be allocated as an array of size n x n 
///   \n\b NOTE: must be 128-bit aligned.
///
/// @return
///    
/// @ingroup math_vector 
//------------------------------------------------------------------------------
FASTCV_API void
fcvSVDf32(const float32_t * __restrict A, 
                 uint32_t              m, 
                 uint32_t              n, 
                float32_t * __restrict w, 
                float32_t * __restrict U, 
                float32_t * __restrict Vt,
                float32_t *            tmpU,
                float32_t *            tmpV);

//------------------------------------------------------------------------------
/// @brief
///   Draw convex polygon
///
/// @details
///   This function fills the interior of a convex polygon with the specified color.
///   
/// @param polygon
///   Coordinates of polygon vertices (x0,y0,x1,y1,...), size of buffer is 2*nPts
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param nPts
///   Number of polygon vertices
///
/// @param color
///   Color of drawn polygon stored as B,G,R and A(if supported)
///
/// @param nChannel
///   Number of color channels (typical value is 1 or 3)
///
/// @param dst
///   Destination image, size of image buffer is (dstStride * dstHeight) bytes
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param dstWidth
///   Image width, the number of pixels in a row.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstHeight
///   Image height.
///
/// @param dstStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 dstStride is default to (dstWidth * nChannel).
///   \n\b NOTE: must be a multiple of 8.
///
///
/// @ingroup Structural_Analysis_and_Drawing
//------------------------------------------------------------------------------
FASTCV_API void
fcvFillConvexPolyu8( uint32_t             nPts,
               const uint32_t* __restrict polygon,
	             uint32_t             nChannel,
                const uint8_t* __restrict color,
	              uint8_t* __restrict dst,
	             uint32_t             dstWidth,
	             uint32_t             dstHeight,
	             uint32_t             dstStride);


//------------------------------------------------------------------------------
/// @brief
///   Determines whether a given point is inside a contour, outside, or lies on an edge
///   (or coincides with a vertex). It returns positive, negative or zero value, correspondingly.
///   Also measures distance between the point and the nearest contour edge if distance 
///   is requested.
///
/// @details
///   
/// @param nPts
///   Total number of points in the contour. For example if there are 10 point sets, i.e. (x,y) in the contour,
///   then nPts equals 20.
///
/// @param polygonContour
///   Input contour containing the points of the polygon. Coordinates are stored in the interleaved form as x y x y.
///   Size of buffer is @param nPts    
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param px
///   The x-coordinate of the input point to be tested.
///
/// @param py
///   The y-coordinate of the input point to be tested.
///
/// @param distance
///   It contains the signed distance of the point to the closest edge of the contour
///   If passed as a NULL pointer, then no distance is computed.
///
/// @param resultFlag
///   Assumes the value of -1, 0 or 1 based on whether the point is outside, coincides with 
///   a vertex, or lies inside the contour respectively.
///
///
///
/// @ingroup Structural_Analysis_and_Drawing
//------------------------------------------------------------------------------
FASTCV_API void 
fcvPointPolygonTest(uint32_t             nPts,
              const uint32_t* __restrict polygonContour,
                    uint32_t             px,
                    uint32_t             py,
                   float32_t*            distance,
                     int16_t*            resultFlag);


//------------------------------------------------------------------------------
/// @brief
///   Find the convex hull of the input polygon
///
/// @details
///   Determines the convex hull of a simple polygon using the Melkman algorithm.
///   The input to the function is the interleaved coordinates of the polygon and
///   the output is the set of interleaved coordinates of the convex hull of the polygon.
///   The algorithm assumes that the coordinates of the polygon are provided in the manner
///   of an ordered traversal.  
/// 
/// @param polygonContour
///   Input contour containing the points of the polygon for which the convex hull is to be found.
///   Coordinates are stored in the interleaved form as x y x y. NOTE: The polygon must be a simple
///   polygon, i.e., it has no self intersections. Also coordinates are assumed to be stored in the 
///   manner of an ordered traversal.
///   \n\b WARNING: must be 128-bit aligned. Size of buffer is @param nPtsContour
///
/// @param nPtsContour
///   Total number of points in the contour. For example if there are 10 point sets, i.e. (x,y) in the contour,
///   then nPtsContour equals 20.
///
///
/// @param convexHull
///   The output buffer containing the interleaved coordinates of the convex hull.
///   \n\b WARNING: must be 128-bit aligned. Size of buffer is @param nPtsHull
///
/// @param nPtsHull
///   Total number of points in the convex hull. For example if there are 10 point sets, i.e. (x,y) in the contour,
///   then nPtsHull equals 20.
///
/// @param tmpBuff
///   Scratch buffer used in the computation of the convex hull. 
///   NOTE: MUST be allocated twice as large in size as the input polygonContour. 
///   \n\b WARNING: must be 128-bit aligned.
///
///
/// @ingroup Structural_Analysis_and_Drawing
//------------------------------------------------------------------------------

FASTCV_API void 
fcvFindConvexHull( uint32_t* __restrict polygonContour, 
                                   uint32_t             nPtsContour, 
                                   uint32_t* __restrict convexHull,
                                   uint32_t*            nPtsHull, 
                                   uint32_t* __restrict tmpBuff);


//---------------------------------------------------------------------------
/// @brief
///   Executes Cholesky decomposition algorithm on a symmetric and positive
///   definite matrix to solve the linear system A*x = b, where A is an NxN
///   matrix and x & b are vectors of size N.
///
/// @param A
///   Pointer to the matrix A or size NxN.
///   NOTE: This matrix WILL BE MODIFIED during computation.
///         Please SAVE THE ORIGINAL MATRIX properly if necessary.
///   NOTE: must be 128-bit aligned.
///
/// @param b
///   Pointer to the vector b of size N.
///   NOTE: must be 128-bit aligned.
///
/// @param diag
///   Pointer to the buffer for the diagonal of matrix A.
///   This buffer is used for computation.
///   NOTE: must be 128-bit aligned.
///
/// @param N
///   Size of matrix and vectors.
///
/// @param x
///   Pointer to the output vector x of size N.
///   NOTE: must be 128-bit aligned.
///
/// @return Returns 1 if the linear system could be solved or 0 otherwise.
///
/// @ingroup math_vector
//---------------------------------------------------------------------------
FASTCV_API int32_t
fcvSolveCholeskyf32( float32_t* __restrict A,
               const float32_t* __restrict b,
                     float32_t* __restrict diag,
                      uint32_t             N,
                     float32_t* __restrict x);


//---------------------------------------------------------------------------
/// @brief
///   Applies radial distortion to a 2D coordinate in camera coordinates
///   and returns the distorted coordinate in device coordinates.
///
/// @details
///   input: (x,y)
///   focal length: f1,f2
///   principle point: p1,p2
///   radical distortion: k1,k2
///   tangential distortion: t1,t2
///   Output (xd,yd)
///   r^2 = x^2+y^2
///   cdist = 1+k1*r^2 + k2*r^4
///   a0 = 2*x*y, a1 = 3*x^2 + y^2, a2 = x^2+3*y^2
///   xd = (x*cdist + t1*a1 + t2*a2)*f1 + p1
///   yd = (y*cdist + t1*a3 + t2*a1)*f2 + p2
///
/// @param cameraCalibration
///   Camera calibration with 8 parameter: focal-length (x and y),
///   principal point (x & y), radial distortion (2 parameters),
///   tangential distortion (2 parameters).
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param xyCamera
///   Input of the undistorted 2D camera coordinate (2 float values).
///
/// @param xyDevice
///   Output of the distorted 2D device coordinate (2 float values).
///
/// @ingroup 3D_reconstruction
//---------------------------------------------------------------------------
FASTCV_API void
fcvGeomDistortPoint2x1f32(const float32_t* __restrict cameraCalibration,
                          const float32_t* __restrict xyCamera,
                                float32_t* __restrict xyDevice);

//---------------------------------------------------------------------------
/// @brief
///   Applies radial distortion to a set of 2D coordinates in camera coordinates
///   and returns the distorted coordinates in device coordinates.
///   brief algorithm desribed in fcvGeomDistortPoint2x1f32
///
/// @param cameraCalibration
///   Camera calibration with 8 parameter: focal-length (x and y),
///   principal point (x & y), radial distortion (2 parameters),
///   tangential distortion (2 parameters).
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param xyCamera
///   Input of the undistorted 2D camera coordinates
///   (Nx2 float values). While allocating, allocate enough memory to accomodate
///   the points (Nx2) keeping in mind the stride of the input points. 
///   Total memory allocated must be large enough to accomodate the input+padding,
///   which has size of N*srcStride (in bytes)
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcStride
///   Stride between consecutive input camera coordinates. Stride here is defined as
///   the number of units between consecutive x coordinates. For example, if the input
///   array has points as follows x0 y0 0 0 0 x1 y1 0 0 0 x2 y2 ..., then the stride is
///   5 * size(float32_t) = 20
///   \n\b NOTE: must be a multiple of 8.
///
/// @param xySize
///   Number of points N
///   
/// @param xyDevice
///   Output of the distorted 2D device coordinates
///   (Nx2 float values). While allocating, allocate enough memory to accomodate
///   the points (Nx2) keeping in mind the stride of the output points. 
///   Total memory allocated must be large enough to accomodate the output+padding,
///   which has size of N*dstStride (in bytes)
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride between consecutive input camera coordinates. Stride here is defined as
///   the number of units between consecutive x coordinates. For example, if the output
///   array has points as follows x0 y0 0 0 0 x1 y1 0 0 0 x2 y2 ..., then the stride is
///   5 * size(float32_t) = 20
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup 3D_reconstruction
//---------------------------------------------------------------------------

FASTCV_API void
fcvGeomDistortPoint2xNf32(const float32_t* __restrict cameraCalibration,
                          const float32_t* __restrict xyCamera,
                                 uint32_t             srcStride,
                                 uint32_t             xySize,
                                float32_t* __restrict xyDevice,
                                 uint32_t             dstStride);

//---------------------------------------------------------------------------
/// @brief
///   Applies radial undistortion to a 2D coordinate in device coordinates
///   and returns the undistorted coordinate in camera coordinates.
///
/// @param cameraCalibration
///   Camera calibration with 8 parameter: Inverse focal-length (x and y),
///   principal point (x & y), radial distortion (2 parameters),
///   tangential distortion (2 parameters).
///   NOTE: The first two entries of this parameter for this function are the 
///   Inverse of the focal length and not the focal length itself.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param xyDevice
///   Input of the distorted 2D device coordinate (2 float values).
///
/// @param xyCamera
///   Output of the undistorted 2D camera coordinate (2 float values).
///
/// @ingroup 3D_reconstruction
//---------------------------------------------------------------------------
FASTCV_API void
fcvGeomUndistortPoint2x1f32(const float32_t* __restrict cameraCalibration,
                            const float32_t* __restrict xyDevice,
                                  float32_t* __restrict xyCamera);

//---------------------------------------------------------------------------
/// @brief
///   Applies radial undistortion to a 2D coordinate in device coordinates
///   and returns the undistorted coordinate in camera coordinates.
///   brief algorithm desribed in fcvGeomUndistortPoint2x1f32
///
/// @param cameraCalibration
///   Camera calibration with 8 parameter: inverse focal-length (x and y),
///   principal point (x & y), radial distortion (2 parameters),
///   tangential distortion (2 parameters).
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param xyDevice
///   Input of the distorted 2D device coordinates
///   (Nx2 float values). While allocating, allocate enough memory to accomodate
///   the points (Nx2) keeping in mind the stride of the input points. 
///   Total memory allocated must be large enough to accomodate the input+padding,
///   which has size of N*srcStride (in bytes)
///   \n\b NOTE: must be 128-bit aligned.
///  
/// @param srcStride
///   Stride between consecutive input camera coordinates. Stride here is defined as
///   the number of units between consecutive x coordinates. For example, if the input
///   array has points as follows x0 y0 0 0 0 x1 y1 0 0 0 x2 y2 ..., then the stride is
///   5 * size(float32_t) = 20 
///   \n\b NOTE: must be a multiple of 8.
///
/// @param xySize
///   Number of points N
///   
/// @param xyCamera
///   Output of the undistorted 2D camera coordinates 
///   (Nx2 float values). While allocating, allocate enough memory to accomodate
///   the points (Nx2) keeping in mind the stride of the output points. 
///   Total memory allocated must be large enough to accomodate the output+padding,
///   which has size of N*dstStride (in bytes)
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride between consecutive input camera coordinates. Stride here is defined as
///   the number of units between consecutive x coordinates. For example, if the output
///   array has points as follows x0 y0 0 0 0 x1 y1 0 0 0 x2 y2 ..., then the stride is
///   5 * size(float32_t) = 20 
///   \n\b NOTE: must be a multiple of 8.
///
/// @ingroup math_vector
//---------------------------------------------------------------------------

FASTCV_API void
fcvGeomUndistortPoint2xNf32(const float32_t* __restrict cameraCalibration,
                            const float32_t* __restrict xyDevice,
                                   uint32_t             srcStride,
                                   uint32_t             xySize,
                                  float32_t* __restrict xyCamera,
                                   uint32_t             dstStride);

//---------------------------------------------------------------------------
/// @brief
///   Transforms a 3D point using a pose-matrix, projects the transformed point,
///   distorts the projected 2D point and converts to device coordinates.
///
/// @details
///   (x_camera, y_camera, z_camera) = Pose * (x,y,z,1)'
///   xCamera = x_camera/z_camera, yCamera = y_camera/z_camera
///   xyDevice = distortion(xyCamera) - described in fcvGeomDistortPoint2x1f32
///
/// @param pose
///   Pose matrix of size 3x4 (12 float values) in row-major format.
///
/// @param cameraCalibration
///   Camera calibration with 8 parameter: focal-length (x and y),
///   principal point (x & y), radial distortion (2 parameters),
///   tangential distortion (2 parameters).
///   NOTE: must be 128-bit aligned.
///
/// @param xyz
///   3D point (x,y,z) as three float values
///   
/// @param xyCamera
///   Output of the projected 2D camera coordinate (2 float values)
///
/// @param xyDevice
///   Output of the projected and distorted 2D device coordinate
///   (2 float values)
///
/// @return  Returns 1 if transformed point lies in front of the camera plane.
///          Returns 0 otherwise
///
/// @ingroup 3D_reconstruction
//---------------------------------------------------------------------------
FASTCV_API int32_t
fcvGeomProjectPoint3x1f32(const float32_t* __restrict pose,
                          const float32_t* __restrict cameraCalibration,
                          const float32_t* __restrict xyz,
                                float32_t* __restrict xyCamera,
                                float32_t* __restrict xyDevice);

//---------------------------------------------------------------------------
/// @brief
///   Transforms a 3D point using a pose-matrix, projects the transformed point,
///   distorts the projected 2D point and converts to device coordinates.
///   brief algorithm desribed in fcvGeomProjectPoint3x1f32
///
/// @param pose
///   Pose matrix of size 3x4 (12 float values) in row-major format.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param cameraCalibration
///   Camera calibration with 8 parameter: focal-length (x and y),
///   principal point (x & y), radial distortion (2 parameters),
///   tangential distortion (2 parameters).
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param xyz
///   3D points (x,y,z) as sets of three float values. While allocating, allocate
///   enough memory to accomodate the points (Nx3) keeping in mind the stride of the
///   input points. Total memory allocated must be large enough to accomodate the input
///   +padding.
///   \n\b NOTE: must be 128-bit aligned.
///   
/// @param srcStride
///   Stride between consecutive input camera coordinates. Stride here is defined as
///   the number of units between consecutive x coordinates. For example, if the input
///   array has points as follows x0 y0 z0 0 0 x1 y1 z1 0 0 x2 y2 z2.., then the stride is
///   5 * size(float32_t) = 20 
///
/// @param xyzSize
///   Number of points N
///   
/// @param xyCamera
///   Output of the projected 2D camera coordinates
///   (Nx2 float values). While allocating, allocate enough memory to accomodate
///   the points (Nx2) keeping in mind the stride of the output points. 
///   Total memory allocated must be large enough to accomodate the output+padding.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param xyDevice
///   Output of the projected and distorted 2D device coordinates.
///   (Nx2 float values). While allocating, allocate enough memory to accomodate
///   the points (Nx2) keeping in mind the stride of the output points. 
///   Total memory allocated must be large enough to accomodate the output+padding.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param dstStride
///   Stride between consecutive input camera coordinates. Stride here is defined as
///   the number of units between consecutive x coordinates. For example, if the output
///   array has points as follows x0 y0 0 0 0 x1 y1 0 0 0 x2 y2 ..., then the stride is
///   5 * size(float32_t) = 20 
///   \n\b NOTE: must be a multiple of 8.
///
/// @param inFront
///   Is 1 if transformed point lies in front of the camera plane and 0 otherwise
///   It must be allocated as a Nx1 vector.
///
/// @ingroup math_vector
//---------------------------------------------------------------------------

FASTCV_API void
fcvGeomProjectPoint3xNf32(const float32_t* __restrict pose,
                          const float32_t* __restrict cameraCalibration,
                          const float32_t* __restrict xyz,
                                 uint32_t             srcStride,
                                 uint32_t             xyzSize,
                                float32_t* __restrict xyCamera,
                                float32_t* __restrict xyDevice,
                                 uint32_t             dstStride,
                                 uint32_t*            inFront);

//---------------------------------------------------------------------------
/// @brief
///   Applies a generic geometrical transformation to a 4-channel uint8 image. 
///   The interpolation method is nearest neighbor.
///
/// @details
///   The brightness of each pixel in the destination image is obtained from a location of the source image 
/// through a per-element mapping as defined in the mapping matrices. The mapping has subpixel precision, thus interpolations
/// are involved. 
///
/// @param src
///   Input uint8_t image. The size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: must be 128-bit aligned.
///   
/// @param srcWidth
///   Input image width.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Input image height.
///
/// @param srcStride
///   Input image stride, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///   if srcStride is equal to 0, it will be set to srcWidth*4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output image which has the same type, and size as the input image. The size of buffer is dstStride*dstHeight bytes.
///   \n\b NOTE: must be 128-bit aligned.
///   
/// @param dstWidth
///   Output image width.
///
/// @param dstHeight
///   Output image height.
///
/// @param dstStride
///   Output image stride, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///   if dstStride is equal to 0, it will be set to dstWidth*4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param mapX
///   a floating point matrix, each element is the column coordinate of the mapped location in the src image. E.g. if dst(i,j) is
///   mapped to src(ii,jj), then mapX(i,j) =jj. 
///   the matrix has the same width, height as the dst image.  The size of buffer is mapStride*dstHeight bytes.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param mapY
///   a floating point matrix, each element is the row coordinate of the mapped location in the src image.E.g. if dst(i,j) is
///   mapped to src(ii,jj), then mapY(i,j) =ii.
///   the matrix has the same width, height as the dst image.  The size of buffer is mapStride*dstHeight bytes.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param mapStride
///   the stride of the mapX and mapY
///   if mapStride is equal to 0, it will be set to dstWidth*4.
///   \n\b NOTE: must be a multiple of 8.
///
///
/// @return
///   No return value
///
/// @ingroup image_transform
//---------------------------------------------------------------------------

FASTCV_API void 
fcvRemapRGBA8888NNu8( const uint8_t*  __restrict src, 
                            uint32_t             srcWidth, 
                            uint32_t             srcHeight,
                            uint32_t             srcStride,
                             uint8_t* __restrict dst,
                            uint32_t             dstWidth, 
                            uint32_t             dstHeight,  
                            uint32_t             dstStride, 
                     const float32_t* __restrict mapX, 
                     const float32_t* __restrict mapY, 
                            uint32_t             mapStride
                      );

//---------------------------------------------------------------------------
/// @brief
///   Applies a generic geometrical transformation to a 4-channel uint8 image with bilinear interpolation.
///
/// @details
///   The brightness of each pixel in the destination image is obtained from a location of the source image 
/// through a per-element mapping as defined in the mapping matrices. The mapping has subpixel precision, thus interpolations
/// are involved. 
///
/// @param src
///   Input uint8_t image. The size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: must be 128-bit aligned.
///   
/// @param srcWidth
///   Input image width.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Input image height.
///
/// @param srcStride
///   Input image stride, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///   if srcStride is equal to 0, it will be set to srcWidth*4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dst
///   Output image which has the same type, and size as the input image.  The size of buffer is dstStride*dstHeight bytes.
///   \n\b NOTE: must be 128-bit aligned.
///   
/// @param dstWidth
///   Output image width.
///
/// @param dstHeight
///   Output image height.
///
/// @param dstStride
///   Output image stride, i.e. the gap (in terms of bytes) between the first element of a row and that of the successive row
///   if dstStride equals to 0, it will be set to dstWidth*4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param mapX
///   a floating point matrix, each element is the column coordinate of the mapped location in the src image. E.g. if dst(i,j) is
///   mapped to src(ii,jj), then mapX(i,j) =jj. 
///   the matrix has the same width, height as the dst image.  The size of buffer is mapStride*dstHeight bytes.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param mapY
///   a floating point matrix, each element is the row coordinate of the mapped location in the src image.E.g. if dst(i,j) is
///   mapped to src(ii,jj), then mapY(i,j) =ii.
///   the matrix has the same width, height as the dst image.  The size of buffer is mapStride*dstHeight bytes.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param mapStride
///   the stride of the mapX and mapY
///   if mapStride equals to 0, it will be set to dstWidth*4.
///   \n\b NOTE: must be a multiple of 8.
///
/// @return
///   No return value
///
/// @ingroup image_transform
//---------------------------------------------------------------------------

FASTCV_API void 
fcvRemapRGBA8888BLu8(  const uint8_t* __restrict src, 
                            uint32_t             srcWidth, 
                            uint32_t             srcHeight,
                            uint32_t             srcStride,
                             uint8_t* __restrict dst,
                            uint32_t             dstWidth, 
                            uint32_t             dstHeight,  
                            uint32_t             dstStride, 
                     const float32_t* __restrict mapX, 
                     const float32_t* __restrict mapY, 
                            uint32_t             mapStride );

//---------------------------------------------------------------------------
/// @brief
///   Calculates JTJ, JTE and the sum absolute, normalized pixel differences
///   for a target image and a reference image of same size for an SE2 image
///   motion model.
///   Since gradients are required for this algorithm all border pixels in
///   referenceImage and targetImage are ignored.
///   NOTE: Only works for images with even width and height.
///   
/// @param warpedImage
///   Grayscale 8-bit image.
///   NOTE: must be 128-bit aligned.
///   
/// @param warpedBorder
///   Array with the x-coordinates of left-most and right-most
///   pixels for each scanline to consider in warpedImage.
///   Format is l0,r0,l1,r1,l2,... where l_ and r_ are the left-most
///   and right-most pixel coordinates for a scanline.
///   NOTE: must be 128-bit aligned.
///
/// @param targetImage
///   Grayscale 8-bit image.
///   NOTE: must be 128-bit aligned.
///
/// @param targetDX
///   X-gradients of the target image as 16-bit signed integers.
///   NOTE: must be 128-bit aligned.
///   
/// @param targetDY
///   Y-gradients of the target image as 16-bit signed integers.
///   NOTE: must be 128-bit aligned.
///   
/// @param width
///   Width of the reference image and target image. Must be even.
///   
/// @param height
///   Height of the reference image and target image. Must be even.
///   
/// @param stride
///   Stride (in bytes) of reference image and target image, is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 stride is default to width.
///   NOTE: must be a multiple of 8.
///   
/// @param sumJTJ
///   3x3 matrix (9 floats) receiving the sum of JTJ for all pixels.
///   Only the upper half triangle matrix is filled.
///   NOTE: must be 128-bit aligned.
///   
/// @param sumJTE
///   3 vector (3 floats) receiving the sum of JTE for all pixels.
///
/// @param sumError
///   Sum of absolute, normalized pixel differences for all
///   processed pixels (1 float).
///
/// @param numPixels
///   Number of pixels that have been processed (1 integer).
///
/// @ingroup math_vector
//---------------------------------------------------------------------------

FASTCV_API void
fcvJacobianSE2f32(const uint8_t* __restrict warpedImage,
                 const uint16_t* __restrict warpedBorder,
                  const uint8_t* __restrict targetImage,
                  const int16_t* __restrict targetDX,
                  const int16_t* __restrict targetDY,
                       uint32_t             width,
                       uint32_t             height,
                       uint32_t             stride,
                      float32_t* __restrict sumJTJ,
                      float32_t* __restrict sumJTE,
                      float32_t* __restrict sumError,
                       uint32_t* __restrict numPixels);

//---------------------------------------------------------------------------
/// @brief
///   Applies an affine transformation on a grayscale image using a 2x3
///   matrix. Pixels are sampled using bi-linear interpolation.
///   Pixels that would be sampled from outside the source image
///   are not modified in the target image. The left-most and right-most
///   pixel coordinates of each scanline are written to dstBorder.
///
/// @param src
///   Input 8-bit image.
///   \n\b NOTE: data should be 128-bit aligned. Size of buffer is srcStride*srcHeight bytes.
///
/// @param srcWidth
///   Input image width. The number of pixels in a row.
///
/// @param srcHeight
///   Input image height.
/// 
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param affineMatrix
///   2x3 perspective transformation matrix. The matrix stored
///    in affineMatrix is using row major ordering: \n
///    a11, a12, a13, a21, a22, a23 where the matrix is: \n
///    | a11, a12, a13 |\n
///    | a21, a22, a23 |\n
///    Warning: the convention for rotation angle is: positive for clockwise rotation 
///             and negative for counter-clockwise rotation. If there's unexpected
///             result, it could be due to different rotation convention. 
///     If that's the case, nagate the angle before calculating transform matrix.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param dst
///   Transformed output 8-bit image.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @param dstWidth
///   Dst image width.
///
/// @param dstHeight
///   Dst image height.
/// 
/// @param dstStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 dstStride is default to dstWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param dstBorder
///   Output array receiving the x-coordinates of left-most and right-most
///   pixels for each scanline. The format of the array is:
///   l0,r0,l1,r1,l2,r2,... where l0 is the left-most pixel coordinate
///   in scanline 0 and r0 is the right-most pixel coordinate
///   in scanline 0.
///   The buffer must therefore be 2*dstHeight integers in size.
///   \n\b NOTE: data should be 128-bit aligned.
///
/// @ingroup image_transform
//------------------------------------------------------------------------------/

FASTCV_API void
fcvTransformAffineClippedu8(const uint8_t* __restrict src,
                                 uint32_t             srcWidth,
                                 uint32_t             srcHeight,
                                 uint32_t             srcStride,
                          const float32_t* __restrict affineMatrix,
                                  uint8_t* __restrict dst,
                                 uint32_t             dstWidth,
                                 uint32_t             dstHeight,
                                 uint32_t             dstStride,
                                 uint32_t* __restrict dstBorder);

//------------------------------------------------------------------------------
/// @brief
///     Creates codebook model according to the image size
///
/// @details
///     This function creates codebook model and returns codebook map.
///     These 2 parameters will be used in fcvBGCodeBookUpdateu8(), fcvBGCodeBookDiffu8()
///     and fcvBGCodeBookClearStaleu8(). Codebook functions are useful in background subtraction
///     in many use cases, such as video surveillance. 
/// @param srcWidth 
///     Width of the input image.
///     \n\b NOTE: must be multiple of 8. The number of pixels in a row.
/// @param srcHeight 
///     Height of the input image.
///
/// @param cbmodel 
///     Double pointer to codebook model.
///     Codebook model contains parameters for generating and maintaining codebook model. 
/// @return
///     Double pointer to codebook map.
///     Codebook map is a pointer map consisting of code word for each pixel of input image.
/// @ingroup Motion_and_Object_Tracking
//------------------------------------------------------------------------------
FASTCV_API fcvBGCodeWord**
fcvCreateBGCodeBookModel( uint32_t            srcWidth,
                          uint32_t            srcHeight,
                          void**   __restrict cbmodel );

//------------------------------------------------------------------------------
/// @brief
///     Releases codebook model and codebook map
///
/// @details
///     This function release codebook model and codebook map. Codebook map is
///     referred in codebook model.
/// @param cbmodel 
///     Double pointer to codebook model
///
/// @ingroup Motion_and_Object_Tracking
//------------------------------------------------------------------------------
FASTCV_API void
fcvReleaseBGCodeBookModel( void** cbmodel );

//------------------------------------------------------------------------------
/// @brief
///     Updates codebook map according to input image. fgMask can be a reference.
///
/// @details
///     This function updates codebook map according to input image. fgMask is generated by
///     fcvBGCodeBookDiffu8() and can be a reference in this function. Therefore, fgMask is
///     NULL at the first time.
///     Codebook functions are useful in background subtraction in many use cases, such as
///     video surveillance.
/// @param cbmodel 
///     Pointer to codebook model
///     Codebook model contains parameters for generating and maintaining codebook model
/// @param src 
///     Pointer to the input image
///     \n\b WARNING: must be 128-bit aligned. must be a 3-channel image.
/// @param srcWidth 
///     Width of the image in pixel. The number of pixels in a row.
///     \n\b NOTE: must be multiple of 8.
/// @param srcHeight 
///     Height of the image in pixel
/// @param srcStride 
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth*3.
///     \n\b NOTE: must be multiple of 8.
/// @param fgMask 
///     Pointer to the returned foreground mask image. Use NULL as default
///     \n\b WARNING: must be 128-bit aligned. must be a 1-channel image, same width & height as src.
/// @param fgMaskStride 
///     Stride of the foreground mask image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 fgMaskStride is default to srcWidth.
///     \n\b NOTE: must be multiple of 8.
/// @param cbMap 
///     Pointer to codebook map
///     \n Codebook map is a pointer map consisting of code word for each pixel of input image.
/// @param updateTime 
///     Update time.
///     \n updateTime is a return value.
///
/// @ingroup Motion_and_Object_Tracking
//------------------------------------------------------------------------------
FASTCV_API void
fcvBGCodeBookUpdateu8( void*  __restrict cbmodel,
              const uint8_t*  __restrict src,
                   uint32_t              srcWidth,
                   uint32_t              srcHeight,
                   uint32_t              srcStride,
              const uint8_t*  __restrict fgMask,
                   uint32_t              fgMaskStride,
              fcvBGCodeWord** __restrict cbMap,
                    int32_t*  __restrict updateTime );

//------------------------------------------------------------------------------
/// @brief
///     Generates differential mask of input frame according to background codebook map.
///
/// @details
///     This function generates differential mask  of input frame according to background codebook map.
///     Codebook functions are useful in background subtraction in many use cases, such as
///     video surveillance.
///
/// @param cbmodel 
///     Pointer to codebook model
///     Codebook model contains parameters for generating and maintaining codebook model
///
/// @param src
///     Pointer to the input image
///     \n\b WARNING: must be 128-bit aligned. must be a 3-channel image.
/// @param srcWidth 
///     Width of the image in pixel. The number of pixels in a row.
///     \n\b NOTE: must be multiple of 8.
/// @param srcHeight 
///     Height of the image in pixel
///
/// @param srcStride 
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth*3.
///     \n\b NOTE: must be multiple of 8.
///
/// @param fgMask 
///     Pointer to the returned foreground mask image.
///     \n\b WARNING: must be 128-bit aligned. must be a 1-channel image, same width & height as src.
/// @param fgMaskStride 
///     Stride of the foreground mask image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 fgMaskStride is default to srcWidth.
///     \n\b NOTE: must be multiple of 8.
///
/// @param cbMap 
///     Pointer to code book map.
///     \n Codebook map is a pointer map consisting of code word for each pixel of input image.
/// @param numFgMask
///     Number of foreground pixels in the mask
///     \n numFgMask is a return value.
///
/// @ingroup Motion_and_Object_Tracking
//------------------------------------------------------------------------------
FASTCV_API void
fcvBGCodeBookDiffu8( void*  __restrict cbmodel,
            const uint8_t*  __restrict src,
                 uint32_t              srcWidth,
                 uint32_t              srcHeight,
                 uint32_t              srcStride,
                  uint8_t*  __restrict fgMask,
                 uint32_t              fgMaskStride,
            fcvBGCodeWord** __restrict cbMap,
                  int32_t*  __restrict numFgMask );

//------------------------------------------------------------------------------
/// @brief
///     Removes stale element in codebook according to foreground mask
///
/// @details
///     This function removes stale element in codebook according to foreground mask.
///     Threshold is defined in staleThresh.
///     Codebook functions are useful in background subtraction in many use cases, such as
///     video surveillance.
///
/// @param cbmodel 
///     Pointer to codebook model
///     Codebook model contains parameters for generating and maintaining codebook model
///
/// @param staleThresh 
///     Threshold of stale element
///
/// @param fgMask 
///     Pointer to the foreground mask image in ROI. Use NULL as default
///     \n\b NOTE: must be 128-bit aligned. must be a 1-channel image.
/// @param fgMaskWidth 
///     Width of the mask in pixel, which is the same as input image
///     \n\b NOTE: must be multiple of 8.
/// @param fgMaskHeight 
///     Height of the mask in pixel, which is the same as input image
/// 
/// @param fgMaskStride 
///     Stride of the foreground mask image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 fgMaskStride is default to fgMaskWidth.
///     \n\b NOTE: must be multiple of 8.
///
/// @param cbMap 
///     Pointer to code book map
///     \n Codebook map is a pointer map consisting of code word for each pixel of input image.
///
/// @ingroup Motion_and_Object_Tracking
//------------------------------------------------------------------------------
FASTCV_API void
fcvBGCodeBookClearStaleu8( void*  __restrict cbmodel,
                        int32_t              staleThresh,
                  const uint8_t*  __restrict fgMask,
                       uint32_t              fgMaskWidth,
                       uint32_t              fgMaskHeight,
                       uint32_t              fgMaskStride,
                  fcvBGCodeWord** __restrict cbMap );

//------------------------------------------------------------------------------
/// @brief
///     Finds circles in a grayscale image using Hough transform.
/// 
/// @details
///     This function detect circles in a grayscale image. The number is up to maxCircle.
///     The radius of circle varies from 0 to max(srcWidth, srcHeight).
///     
/// @param src
///     8-bit, single-channel, binary source image.  Size of buffer is srcStride*srcHeight bytes.
///     \n\b WARNING: must be 128-bit aligned. must be a 1-channel image.
/// 
/// @param srcWidth
///     Width of input image, the number of pixels in a row.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///     Height of input image.
///
/// @param srcStride
///     Stride of input image(in bytes) is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param circles
///     Pointer to fcvCircle.
///     fcvCircle is a struct including x, y position and radius
/// 
/// @param numCircle
///     Pointer to numCircle.
///     numCircle is an algorithm result indicating
///     the number of circle detected by this algorithm.
/// 
/// @param maxCircle
///     Maximum number of circles.
/// 
/// @param minDist
///     Minimum distance between the centers of the detected circles
/// 
/// @param cannyThreshold
///     The higher threshold of the two passed to the Canny() edge detector 
///     (the lower one is twice smaller). default is 100.
/// 
/// @param accThreshold 
///     The accumulator threshold for the circle centers at the detection 
////    stage. The smaller it is, the more false circles may be detected. 
///     Circles, corresponding to the larger accumulator values, will be 
///     returned first. default is 100.
/// 
/// @param minRadius
///     Minimum circle radius. default is 0.
/// 
/// @param maxRadius
///     Maximum circle radius. default is 0.
/// 
/// @param data
///     Pointer to a buffer required by this algorithm. The recommended size
///     is 16 times of input image (16*srcStride*srcHeigth in bytes).
///     \n\b WARNING: must be 128-bit aligned.
/// 
/// @return
///     void
/// 
/// @ingroup feature_detection
//------------------------------------------------------------------------------

FASTCV_API void
fcvHoughCircleu8( const uint8_t* __restrict src,
                       uint32_t             srcWidth, 
                       uint32_t             srcHeight, 
                       uint32_t             srcStride,
                      fcvCircle*            circles,
                       uint32_t*            numCircle,
                       uint32_t             maxCircle,
                       uint32_t             minDist, 
                       uint32_t             cannyThreshold,
                       uint32_t             accThreshold,
                       uint32_t             minRadius,
                       uint32_t             maxRadius,
                           void*            data);


//------------------------------------------------------------------------------
/// @brief
///   Draw the contour or fill the area enclosed by the contour. The algorithm using even-odd rule to fill the contour.
///   Currently Antialiazing is not supported. 
///
/// @param src
///   8-bit image where keypoints are detected. Size of buffer is srcStride*srcHeight bytes.
///   \n\b NOTE: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width, the number of pixels in a row
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param nContours
///   The total number of contours to be drawn.
///
/// @param holeFlag
///   The flag arrays indicate if the corresponding contour is a hole.
/// 1 indicates a hole and 0 indicates it's not a hole.
///
/// @param numContourPoints
///   The array that stores the length of each contour;
///
/// @param contourStartPoints
///   The array that stores the pointer of the starting point of each contour
///
/// @param pointBufferSize
///   The size of the point buffer, in the number of bytes.
///
/// @param pointBuffer
///   The array that stores all the x,y coordinates of all the contours.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param hierarchy
///   The array that stores the left,right,ancestor and decendant of each contour.
///
/// @param max_level
///   The max level we at which we draw the contour, it stops drawing after we reach this level.
///
/// @param thickness
///   Indicate the thickness of the contour to be drawn, if it's 0, do a fill.
///
///  @param color
///   The color value used to draw/fill the contour, currently support grayscale value from 0-255;
///
///  @param hole_color
///   The color value used to fill the hole;
///
/// @ingroup Structural_Analysis_and_Drawing
//------------------------------------------------------------------------------
FASTCV_API void
fcvDrawContouru8(uint8_t*  __restrict src,
                uint32_t              srcWidth,
                uint32_t              srcHeight,
                uint32_t              srcStride,
                uint32_t              nContours,
          const uint32_t*  __restrict holeFlag,
          const uint32_t*  __restrict numContourPoints,
          const uint32_t** __restrict contourStartPoints,
                uint32_t              pointBufferSize,
          const uint32_t*  __restrict pointBuffer,                 
                 int32_t              hierarchy[][4],
                uint32_t              max_level,
                 int32_t              thickness,
                 uint8_t              color,
                 uint8_t              hole_color);


//------------------------------------------------------------------------------
/// @brief
///   Draw the contour or fill the area enclosed by the contour. 
///   Currently Antialiazing is not supported. 
///
/// @param src
///   Input image/patch. It's 3 channel RGB color image in interleaved format. Size of buffer is srcStride*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width, the number of pixels in a row
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth*3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param nContours
///   The total number of contours to be drawn.
///
/// @param holeFlag
///   The flag arrays indicate if the corresponding contour is a hole.
/// 1 indicates a hole and 0 indicates it's not a hole.
///
/// @param numContourPoints
///   The array that stores the length of each contour;
///
/// @param contourStartPoints
///   The array that stores the pointer of the starting point of each contour
///
/// @param pointBufferSize
///   The size of the point buffer, in the number of bytes.
///
/// @param pointBuffer
///   The array that stores all the x,y coordinates of all the contours.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param hierarchy
///   The array that stores the left,right,ancestor and decendant of each contour.
///
/// @param max_level
///   The max level we at which we draw the contour, it stops drawing after we reach this level.
///
/// @param thickness
///   Indicate the thickness of the contour to be drawn, if it's 0, do a fill.
///
/// @param colorR, colorG, colorB
///   The color value used to draw/fill the contour, currently support value from 0-255;
///
/// @param hole_colorR, hole_colorG, hole_colorB
///   The color value used to fill the hole, currently support value from 0-255;
///
/// @ingroup Structural_Analysis_and_Drawing
//------------------------------------------------------------------------------
FASTCV_API void
fcvDrawContourInterleavedu8( uint8_t*  __restrict src,
                            uint32_t              srcWidth,
                            uint32_t              srcHeight,
                            uint32_t              srcStride,
                            uint32_t              nContours,
                      const uint32_t*  __restrict holeFlag,
                      const uint32_t*  __restrict numContourPoints,
                      const uint32_t** __restrict contourStartPoints,
                            uint32_t              pointBufferSize,
                      const uint32_t*  __restrict pointBuffer,                            
                             int32_t              hierarchy[][4],
                            uint32_t              max_level,
                             int32_t              thickness,
                             uint8_t              colorR,
                             uint8_t              colorG,
                             uint8_t              colorB,
                             uint8_t              hole_colorR,
                             uint8_t              hole_colorG,
                             uint8_t              hole_colorB);

//------------------------------------------------------------------------------
/// @brief
///   Draw the contour or fill the area enclosed by the contour. 
///   Currently Antialiazing is not supported. 
///
/// @param src
///   Input image/patch. It's 3 channel RGB color image in planar format. Size of buffer is srcStride*srcHeight bytes.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param srcWidth
///   Image width, the number of pixels in a row
///   \n\b NOTE: must be a multiple of 8.
///
/// @param srcHeight
///   Image height
///
/// @param srcStride
///   Stride of image is the number of bytes between column 0 of row 1 and
///   column 0 of row 2 in data memory. If left at 0 srcStride is default to srcWidth*3.
///   \n\b NOTE: must be a multiple of 8.
///
/// @param nContours
///   The total number of contours to be drawn.
///
/// @param holeFlag
///   The flag arrays indicate if the corresponding contour is a hole.
/// 1 indicates a hole and 0 indicates it's not a hole.
///
/// @param numContourPoints
///   The array that stores the length of each contour;
///
/// @param contourStartPoints
///   The array that stores the pointer of the starting point of each contour
///
/// @param pointBufferSize
///   The size of the point buffer, in the number of bytes.
///
/// @param pointBuffer
///   The array that stores all the x,y coordinates of all the contours.
///   \n\b WARNING: must be 128-bit aligned.
///
/// @param hierarchy
///   The array that stores the left,right,ancestor and decendant of each contour.
///
/// @param max_level
///   The max level we at which we draw the contour, it stops drawing after we reach this level.
///
/// @param thickness
///   Indicate the thickness of the contour to be drawn, if it's 0, do a fill.
///
/// @param colorR, colorG, colorB
///   The color value used to draw/fill the contour, currently support value from 0-255;
///
/// @param hole_colorR, hole_colorG, hole_colorB
///   The color value used to fill the hole, currently support value from 0-255;
///
/// @ingroup Structural_Analysis_and_Drawing
//------------------------------------------------------------------------------
FASTCV_API void
fcvDrawContourPlanaru8( uint8_t*  __restrict src,
                       uint32_t              srcWidth,
                       uint32_t              srcHeight,
                       uint32_t              srcStride,
                       uint32_t              nContours,
                 const uint32_t*  __restrict holeFlag,
                 const uint32_t*  __restrict numContourPoints,
                 const uint32_t** __restrict contourStartPoints,
                       uint32_t              pointBufferSize,
                 const uint32_t*  __restrict pointBuffer,                       
                        int32_t              hierarchy[][4],
                       uint32_t              max_level,
                        int32_t              thickness,
                        uint8_t              colorR,
                        uint8_t              colorG,
                        uint8_t              colorB,
                        uint8_t              hole_colorR,
                        uint8_t              hole_colorG,
                        uint8_t              hole_colorB);

#ifndef FASTCV_NO_INLINE_FUNCTIONS
#include "fastcv.inl"
#endif


#endif
