/* ----------------------------------------------------------------------
* Copyright (C) 2010-2015 ARM Limited. All rights reserved.
*
* $Date:        20. October 2015
* $Revision:    V1.4.5 b
*
* Project:      CMSIS DSP Library
* Title:        arm_math.h
*
* Description:  Public header file for CMSIS DSP Library
*
* Target Processor: Cortex-M7/Cortex-M4/Cortex-M3/Cortex-M0
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*   - Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   - Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in
*     the documentation and/or other materials provided with the
*     distribution.
*   - Neither the name of ARM LIMITED nor the names of its contributors
*     may be used to endorse or promote products derived from this
*     software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------- */

/**
   \mainpage CMSIS DSP Software Library
   *
   * Introduction
   * ------------
   *
   * This user manual describes the CMSIS DSP software library,
   * a suite of common signal processing functions for use on Cortex-M processor based devices.
   *
   * The library is divided into a number of functions each covering a specific category:
   * - Basic math functions
   * - Fast math functions
   * - Complex math functions
   * - Filters
   * - Matrix functions
   * - Transforms
   * - Motor control functions
   * - Statistical functions
   * - Support functions
   * - Interpolation functions
   *
   * The library has separate functions for operating on 8-bit integers, 16-bit integers,
   * 32-bit integer and 32-bit floating-point values.
   *
   * Using the Library
   * ------------
   *
   * The library installer contains prebuilt versions of the libraries in the <code>Lib</code> folder.
   * - arm_cortexM7lfdp_math.lib (Little endian and Double Precision Floating Point Unit on Cortex-M7)
   * - arm_cortexM7bfdp_math.lib (Big endian and Double Precision Floating Point Unit on Cortex-M7)
   * - arm_cortexM7lfsp_math.lib (Little endian and Single Precision Floating Point Unit on Cortex-M7)
   * - arm_cortexM7bfsp_math.lib (Big endian and Single Precision Floating Point Unit on Cortex-M7)
   * - arm_cortexM7l_math.lib (Little endian on Cortex-M7)
   * - arm_cortexM7b_math.lib (Big endian on Cortex-M7)
   * - arm_cortexM4lf_math.lib (Little endian and Floating Point Unit on Cortex-M4)
   * - arm_cortexM4bf_math.lib (Big endian and Floating Point Unit on Cortex-M4)
   * - arm_cortexM4l_math.lib (Little endian on Cortex-M4)
   * - arm_cortexM4b_math.lib (Big endian on Cortex-M4)
   * - arm_cortexM3l_math.lib (Little endian on Cortex-M3)
   * - arm_cortexM3b_math.lib (Big endian on Cortex-M3)
   * - arm_cortexM0l_math.lib (Little endian on Cortex-M0 / CortexM0+)
   * - arm_cortexM0b_math.lib (Big endian on Cortex-M0 / CortexM0+)
   *
   * The library functions are declared in the public file <code>arm_math.h</code> which is placed in the <code>Include</code> folder.
   * Simply include this file and link the appropriate library in the application and begin calling the library functions. The Library supports single
   * public header file <code> arm_math.h</code> for Cortex-M7/M4/M3/M0/M0+ with little endian and big endian. Same header file will be used for floating point unit(FPU) variants.
   * Define the appropriate pre processor MACRO ARM_MATH_CM7 or ARM_MATH_CM4 or  ARM_MATH_CM3 or
   * ARM_MATH_CM0 or ARM_MATH_CM0PLUS depending on the target processor in the application.
   *
   * Examples
   * --------
   *
   * The library ships with a number of examples which demonstrate how to use the library functions.
   *
   * Toolchain Support
   * ------------
   *
   * The library has been developed and tested with MDK-ARM version 5.14.0.0
   * The library is being tested in GCC and IAR toolchains and updates on this activity will be made available shortly.
   *
   * Building the Library
   * ------------
   *
   * The library installer contains a project file to re build libraries on MDK-ARM Tool chain in the <code>CMSIS\\DSP_Lib\\Source\\ARM</code> folder.
   * - arm_cortexM_math.uvprojx
   *
   *
   * The libraries can be built by opening the arm_cortexM_math.uvprojx project in MDK-ARM, selecting a specific target, and defining the optional pre processor MACROs detailed above.
   *
   * Pre-processor Macros
   * ------------
   *
   * Each library project have differant pre-processor macros.
   *
   * - UNALIGNED_SUPPORT_DISABLE:
   *
   * Define macro UNALIGNED_SUPPORT_DISABLE, If the silicon does not support unaligned memory access
   *
   * - ARM_MATH_BIG_ENDIAN:
   *
   * Define macro ARM_MATH_BIG_ENDIAN to build the library for big endian targets. By default library builds for little endian targets.
   *
   * - ARM_MATH_MATRIX_CHECK:
   *
   * Define macro ARM_MATH_MATRIX_CHECK for checking on the input and output sizes of matrices
   *
   * - ARM_MATH_ROUNDING:
   *
   * Define macro ARM_MATH_ROUNDING for rounding on support functions
   *
   * - ARM_MATH_CMx:
   *
   * Define macro ARM_MATH_CM4 for building the library on Cortex-M4 target, ARM_MATH_CM3 for building library on Cortex-M3 target
   * and ARM_MATH_CM0 for building library on Cortex-M0 target, ARM_MATH_CM0PLUS for building library on Cortex-M0+ target, and
   * ARM_MATH_CM7 for building the library on cortex-M7.
   *
   * - __FPU_PRESENT:
   *
   * Initialize macro __FPU_PRESENT = 1 when building on FPU supported Targets. Enable this macro for M4bf and M4lf libraries
   *
   * <hr>
   * CMSIS-DSP in ARM::CMSIS Pack
   * -----------------------------
   *
   * The following files relevant to CMSIS-DSP are present in the <b>ARM::CMSIS</b> Pack directories:
   * |File/Folder                   |Content                                                                 |
   * |------------------------------|------------------------------------------------------------------------|
   * |\b CMSIS\\Documentation\\DSP  | This documentation                                                     |
   * |\b CMSIS\\DSP_Lib             | Software license agreement (license.txt)                               |
   * |\b CMSIS\\DSP_Lib\\Examples   | Example projects demonstrating the usage of the library functions      |
   * |\b CMSIS\\DSP_Lib\\Source     | Source files for rebuilding the library                                |
   *
   * <hr>
   * Revision History of CMSIS-DSP
   * ------------
   * Please refer to \ref ChangeLog_pg.
   *
   * Copyright Notice
   * ------------
   *
   * Copyright (C) 2010-2015 ARM Limited. All rights reserved.
   */


/**
 * @defgroup groupMath Basic Math Functions
 */

/**
 * @defgroup groupFastMath Fast Math Functions
 * This set of functions provides a fast approximation to sine, cosine, and square root.
 * As compared to most of the other functions in the CMSIS math library, the fast math functions
 * operate on individual values and not arrays.
 * There are separate functions for Q15, Q31, and floating-point data.
 *
 */

/**
 * @defgroup groupCmplxMath Complex Math Functions
 * This set of functions operates on complex data vectors.
 * The data in the complex arrays is stored in an interleaved fashion
 * (real, imag, real, imag, ...).
 * In the API functions, the number of samples in a complex array refers
 * to the number of complex values; the array contains twice this number of
 * real values.
 */

/**
 * @defgroup groupFilters Filtering Functions
 */

/**
 * @defgroup groupMatrix Matrix Functions
 *
 * This set of functions provides basic matrix math operations.
 * The functions operate on matrix data structures.  For example,
 * the type
 * definition for the floating-point matrix structure is shown
 * below:
 * <pre>
 *     typedef struct
 *     {
 *       uint16_t numRows;     // number of rows of the matrix.
 *       uint16_t numCols;     // number of columns of the matrix.
 *       float32_t *pData;     // points to the data of the matrix.
 *     } arm_matrix_instance_f32;
 * </pre>
 * There are similar definitions for Q15 and Q31 data types.
 *
 * The structure specifies the size of the matrix and then points to
 * an array of data.  The array is of size <code>numRows X numCols</code>
 * and the values are arranged in row order.  That is, the
 * matrix element (i, j) is stored at:
 * <pre>
 *     pData[i*numCols + j]
 * </pre>
 *
 * \par Init Functions
 * There is an associated initialization function for each type of matrix
 * data structure.
 * The initialization function sets the values of the internal structure fields.
 * Refer to the function <code>arm_mat_init_f32()</code>, <code>arm_mat_init_q31()</code>
 * and <code>arm_mat_init_q15()</code> for floating-point, Q31 and Q15 types,  respectively.
 *
 * \par
 * Use of the initialization function is optional. However, if initialization function is used
 * then the instance structure cannot be placed into a const data section.
 * To place the instance structure in a const data
 * section, manually initialize the data structure.  For example:
 * <pre>
 * <code>arm_matrix_instance_f32 S = {nRows, nColumns, pData};</code>
 * <code>arm_matrix_instance_q31 S = {nRows, nColumns, pData};</code>
 * <code>arm_matrix_instance_q15 S = {nRows, nColumns, pData};</code>
 * </pre>
 * where <code>nRows</code> specifies the number of rows, <code>nColumns</code>
 * specifies the number of columns, and <code>pData</code> points to the
 * data array.
 *
 * \par Size Checking
 * By default all of the matrix functions perform size checking on the input and
 * output matrices.  For example, the matrix addition function verifies that the
 * two input matrices and the output matrix all have the same number of rows and
 * columns.  If the size check fails the functions return:
 * <pre>
 *     ARM_MATH_SIZE_MISMATCH
 * </pre>
 * Otherwise the functions return
 * <pre>
 *     ARM_MATH_SUCCESS
 * </pre>
 * There is some overhead associated with this matrix size checking.
 * The matrix size checking is enabled via the \#define
 * <pre>
 *     ARM_MATH_MATRIX_CHECK
 * </pre>
 * within the library project settings.  By default this macro is defined
 * and size checking is enabled.  By changing the project settings and
 * undefining this macro size checking is eliminated and the functions
 * run a bit faster.  With size checking disabled the functions always
 * return <code>ARM_MATH_SUCCESS</code>.
 */

/**
 * @defgroup groupTransforms Transform Functions
 */

/**
 * @defgroup groupController Controller Functions
 */

/**
 * @defgroup groupStats Statistics Functions
 */
/**
 * @defgroup groupSupport Support Functions
 */

/**
 * @defgroup groupInterpolation Interpolation Functions
 * These functions perform 1- and 2-dimensional interpolation of data.
 * Linear interpolation is used for 1-dimensional data and
 * bilinear interpolation is used for 2-dimensional data.
 */

/**
 * @defgroup groupExamples Examples
 */
#ifndef _ARM_MATH_H
#define _ARM_MATH_H

/* ignore some GCC warnings */
#if defined ( __GNUC__ )
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#define __CMSIS_GENERIC         /* disable NVIC and Systick functions */

#if defined(ARM_MATH_CM7)
  #include "core_cm7.h"
#elif defined (ARM_MATH_CM4)
  #include "core_cm4.h"
#elif defined (ARM_MATH_CM3)
  #include "core_cm3.h"
#elif defined (ARM_MATH_CM0)
  #include "core_cm0.h"
  #define ARM_MATH_CM0_FAMILY
#elif defined (ARM_MATH_CM0PLUS)
  #include "core_cm0plus.h"
  #define ARM_MATH_CM0_FAMILY
#else
  #error "Define according the used Cortex core ARM_MATH_CM7, ARM_MATH_CM4, ARM_MATH_CM3, ARM_MATH_CM0PLUS or ARM_MATH_CM0"
#endif

#undef  __CMSIS_GENERIC         /* enable NVIC and Systick functions */
#include "string.h"
#include "math.h"
#ifdef   __cplusplus
extern "C"
{
#endif


  /**
   * @brief Macros required for reciprocal calculation in Normalized LMS
   */

#define DELTA_Q31          (0x100)
#define DELTA_Q15          0x5
#define INDEX_MASK         0x0000003F
#ifndef PI
#define PI                 3.14159265358979f
#endif

  /**
   * @brief Macros required for SINE and COSINE Fast math approximations
   */

#define FAST_MATH_TABLE_SIZE  512
#define FAST_MATH_Q31_SHIFT   (32 - 10)
#define FAST_MATH_Q15_SHIFT   (16 - 10)
#define CONTROLLER_Q31_SHIFT  (32 - 9)
#define TABLE_SIZE  256
#define TABLE_SPACING_Q31     0x400000
#define TABLE_SPACING_Q15     0x80

  /**
   * @brief Macros required for SINE and COSINE Controller functions
   */
  /* 1.31(q31) Fixed value of 2/360 */
  /* -1 to +1 is divided into 360 values so total spacing is (2/360) */
#define INPUT_SPACING         0xB60B61

  /**
   * @brief Macro for Unaligned Support
   */
#ifndef UNALIGNED_SUPPORT_DISABLE
    #define ALIGN4
#else
  #if defined  (__GNUC__)
    #define ALIGN4 __attribute__((aligned(4)))
  #else
    #define ALIGN4 __align(4)
  #endif
#endif   /* #ifndef UNALIGNED_SUPPORT_DISABLE */

  /**
   * @brief Error status returned by some functions in the library.
   */

  typedef enum
  {
    ARM_MATH_SUCCESS = 0,                /**< No error */
    ARM_MATH_ARGUMENT_ERROR = -1,        /**< One or more arguments are incorrect */
    ARM_MATH_LENGTH_ERROR = -2,          /**< Length of data buffer is incorrect */
    ARM_MATH_SIZE_MISMATCH = -3,         /**< Size of matrices is not compatible with the operation. */
    ARM_MATH_NANINF = -4,                /**< Not-a-number (NaN) or infinity is generated */
    ARM_MATH_SINGULAR = -5,              /**< Generated by matrix inversion if the input matrix is singular and cannot be inverted. */
    ARM_MATH_TEST_FAILURE = -6           /**< Test Failed  */
  } arm_status;

  /**
   * @brief 8-bit fractional data type in 1.7 format.
   */
  typedef int8_t q7_t;

  /**
   * @brief 16-bit fractional data type in 1.15 format.
   */
  typedef int16_t q15_t;

  /**
   * @brief 32-bit fractional data type in 1.31 format.
   */
  typedef int32_t q31_t;

  /**
   * @brief 64-bit fractional data type in 1.63 format.
   */
  typedef int64_t q63_t;

  /**
   * @brief 32-bit floating-point type definition.
   */
  typedef float float32_t;

  /**
   * @brief 64-bit floating-point type definition.
   */
  typedef double float64_t;

  /**
   * @brief definition to read/write two 16 bit values.
   */
#if defined __CC_ARM
  #define __SIMD32_TYPE int32_t __packed
  #define CMSIS_UNUSED __attribute__((unused))

#elif defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)
  #define __SIMD32_TYPE int32_t
  #define CMSIS_UNUSED __attribute__((unused))

#elif defined __GNUC__
  #define __SIMD32_TYPE int32_t
  #define CMSIS_UNUSED __attribute__((unused))

#elif defined __ICCARM__
  #define __SIMD32_TYPE int32_t __packed
  #define CMSIS_UNUSED

#elif defined __CSMC__
  #define __SIMD32_TYPE int32_t
  #define CMSIS_UNUSED

#elif defined __TASKING__
  #define __SIMD32_TYPE __unaligned int32_t
  #define CMSIS_UNUSED

#else
  #error Unknown compiler
#endif

#define __SIMD32(addr)        (*(__SIMD32_TYPE **) & (addr))
#define __SIMD32_CONST(addr)  ((__SIMD32_TYPE *)(addr))
#define _SIMD32_OFFSET(addr)  (*(__SIMD32_TYPE *)  (addr))
#define __SIMD64(addr)        (*(int64_t **) & (addr))

#if defined (ARM_MATH_CM3) || defined (ARM_MATH_CM0_FAMILY)
  /**
   * @brief definition to pack two 16 bit values.
   */
#define __PKHBT(ARG1, ARG2, ARG3)      ( (((int32_t)(ARG1) <<  0) & (int32_t)0x0000FFFF) | \
                                         (((int32_t)(ARG2) << ARG3) & (int32_t)0xFFFF0000)  )
#define __PKHTB(ARG1, ARG2, ARG3)      ( (((int32_t)(ARG1) <<  0) & (int32_t)0xFFFF0000) | \
                                         (((int32_t)(ARG2) >> ARG3) & (int32_t)0x0000FFFF)  )

#endif


   /**
   * @brief definition to pack four 8 bit values.
   */
#ifndef ARM_MATH_BIG_ENDIAN

#define __PACKq7(v0,v1,v2,v3) ( (((int32_t)(v0) <<  0) & (int32_t)0x000000FF) | \
                                (((int32_t)(v1) <<  8) & (int32_t)0x0000FF00) | \
                                (((int32_t)(v2) << 16) & (int32_t)0x00FF0000) | \
                                (((int32_t)(v3) << 24) & (int32_t)0xFF000000)  )
#else

#define __PACKq7(v0,v1,v2,v3) ( (((int32_t)(v3) <<  0) & (int32_t)0x000000FF) | \
                                (((int32_t)(v2) <<  8) & (int32_t)0x0000FF00) | \
                                (((int32_t)(v1) << 16) & (int32_t)0x00FF0000) | \
                                (((int32_t)(v0) << 24) & (int32_t)0xFF000000)  )

#endif


  /**
   * @brief Clips Q63 to Q31 values.
   */
  static __INLINE q31_t clip_q63_to_q31(
  q63_t x)
  {
    return ((q31_t) (x >> 32) != ((q31_t) x >> 31)) ?
      ((0x7FFFFFFF ^ ((q31_t) (x >> 63)))) : (q31_t) x;
  }

  /**
   * @brief Clips Q63 to Q15 values.
   */
  static __INLINE q15_t clip_q63_to_q15(
  q63_t x)
  {
    return ((q31_t) (x >> 32) != ((q31_t) x >> 31)) ?
      ((0x7FFF ^ ((q15_t) (x >> 63)))) : (q15_t) (x >> 15);
  }

  /**
   * @brief Clips Q31 to Q7 values.
   */
  static __INLINE q7_t clip_q31_to_q7(
  q31_t x)
  {
    return ((q31_t) (x >> 24) != ((q31_t) x >> 23)) ?
      ((0x7F ^ ((q7_t) (x >> 31)))) : (q7_t) x;
  }

  /**
   * @brief Clips Q31 to Q15 values.
   */
  static __INLINE q15_t clip_q31_to_q15(
  q31_t x)
  {
    return ((q31_t) (x >> 16) != ((q31_t) x >> 15)) ?
      ((0x7FFF ^ ((q15_t) (x >> 31)))) : (q15_t) x;
  }

  /**
   * @brief Multiplies 32 X 64 and returns 32 bit result in 2.30 format.
   */

  static __INLINE q63_t mult32x64(
  q63_t x,
  q31_t y)
  {
    return ((((q63_t) (x & 0x00000000FFFFFFFF) * y) >> 32) +
            (((q63_t) (x >> 32) * y)));
  }

/*
  #if defined (ARM_MATH_CM0_FAMILY) && defined ( __CC_ARM   )
  #define __CLZ __clz
  #endif
 */
/* note: function can be removed when all toolchain support __CLZ for Cortex-M0 */
#if defined (ARM_MATH_CM0_FAMILY) && ((defined (__ICCARM__))  )
  static __INLINE uint32_t __CLZ(
  q31_t data);

  static __INLINE uint32_t __CLZ(
  q31_t data)
  {
    uint32_t count = 0;
    uint32_t mask = 0x80000000;

    while((data & mask) == 0)
    {
      count += 1u;
      mask = mask >> 1u;
    }

    return (count);
  }
#endif

  /**
   * @brief Function to Calculates 1/in (reciprocal) value of Q31 Data type.
   */

  static __INLINE uint32_t arm_recip_q31(
  q31_t in,
  q31_t * dst,
  q31_t * pRecipTable)
  {
    q31_t out;
    uint32_t tempVal;
    uint32_t index, i;
    uint32_t signBits;

    if(in > 0)
    {
      signBits = ((uint32_t) (__CLZ( in) - 1));
    }
    else
    {
      signBits = ((uint32_t) (__CLZ(-in) - 1));
    }

    /* Convert input sample to 1.31 format */
    in = (in << signBits);

    /* calculation of index for initial approximated Val */
    index = (uint32_t)(in >> 24);
    index = (index & INDEX_MASK);

    /* 1.31 with exp 1 */
    out = pRecipTable[index];

    /* calculation of reciprocal value */
    /* running approximation for two iterations */
    for (i = 0u; i < 2u; i++)
    {
      tempVal = (uint32_t) (((q63_t) in * out) >> 31);
      tempVal = 0x7FFFFFFFu - tempVal;
      /*      1.31 with exp 1 */
      /* out = (q31_t) (((q63_t) out * tempVal) >> 30); */
      out = clip_q63_to_q31(((q63_t) out * tempVal) >> 30);
    }

    /* write output */
    *dst = out;

    /* return num of signbits of out = 1/in value */
    return (signBits + 1u);
  }


  /**
   * @brief Function to Calculates 1/in (reciprocal) value of Q15 Data type.
   */
  static __INLINE uint32_t arm_recip_q15(
  q15_t in,
  q15_t * dst,
  q15_t * pRecipTable)
  {
    q15_t out = 0;
    uint32_t tempVal = 0;
    uint32_t index = 0, i = 0;
    uint32_t signBits = 0;

    if(in > 0)
    {
      signBits = ((uint32_t)(__CLZ( in) - 17));
    }
    else
    {
      signBits = ((uint32_t)(__CLZ(-in) - 17));
    }

    /* Convert input sample to 1.15 format */
    in = (in << signBits);

    /* calculation of index for initial approximated Val */
    index = (uint32_t)(in >>  8);
    index = (index & INDEX_MASK);

    /*      1.15 with exp 1  */
    out = pRecipTable[index];

    /* calculation of reciprocal value */
    /* running approximation for two iterations */
    for (i = 0u; i < 2u; i++)
    {
      tempVal = (uint32_t) (((q31_t) in * out) >> 15);
      tempVal = 0x7FFFu - tempVal;
      /*      1.15 with exp 1 */
      out = (q15_t) (((q31_t) out * tempVal) >> 14);
      /* out = clip_q31_to_q15(((q31_t) out * tempVal) >> 14); */
    }

    /* write output */
    *dst = out;

    /* return num of signbits of out = 1/in value */
    return (signBits + 1);
  }


  /*
   * @brief C custom defined intrinisic function for only M0 processors
   */
#if defined(ARM_MATH_CM0_FAMILY)
  static __INLINE q31_t __SSAT(
  q31_t x,
  uint32_t y)
  {
    int32_t posMax, negMin;
    uint32_t i;

    posMax = 1;
    for (i = 0; i < (y - 1); i++)
    {
      posMax = posMax * 2;
    }

    if(x > 0)
    {
      posMax = (posMax - 1);

      if(x > posMax)
      {
        x = posMax;
      }
    }
    else
    {
      negMin = -posMax;

      if(x < negMin)
      {
        x = negMin;
      }
    }
    return (x);
  }
#endif /* end of ARM_MATH_CM0_FAMILY */


  /*
   * @brief C custom defined intrinsic function for M3 and M0 processors
   */
#if defined (ARM_MATH_CM3) || defined (ARM_MATH_CM0_FAMILY)

  /*
   * @brief C custom defined QADD8 for M3 and M0 processors
   */
  static __INLINE uint32_t __QADD8(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s, t, u;

    r = __SSAT(((((q31_t)x << 24) >> 24) + (((q31_t)y << 24) >> 24)), 8) & (int32_t)0x000000FF;
    s = __SSAT(((((q31_t)x << 16) >> 24) + (((q31_t)y << 16) >> 24)), 8) & (int32_t)0x000000FF;
    t = __SSAT(((((q31_t)x <<  8) >> 24) + (((q31_t)y <<  8) >> 24)), 8) & (int32_t)0x000000FF;
    u = __SSAT(((((q31_t)x      ) >> 24) + (((q31_t)y      ) >> 24)), 8) & (int32_t)0x000000FF;

    return ((uint32_t)((u << 24) | (t << 16) | (s <<  8) | (r      )));
  }


  /*
   * @brief C custom defined QSUB8 for M3 and M0 processors
   */
  static __INLINE uint32_t __QSUB8(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s, t, u;

    r = __SSAT(((((q31_t)x << 24) >> 24) - (((q31_t)y << 24) >> 24)), 8) & (int32_t)0x000000FF;
    s = __SSAT(((((q31_t)x << 16) >> 24) - (((q31_t)y << 16) >> 24)), 8) & (int32_t)0x000000FF;
    t = __SSAT(((((q31_t)x <<  8) >> 24) - (((q31_t)y <<  8) >> 24)), 8) & (int32_t)0x000000FF;
    u = __SSAT(((((q31_t)x      ) >> 24) - (((q31_t)y      ) >> 24)), 8) & (int32_t)0x000000FF;

    return ((uint32_t)((u << 24) | (t << 16) | (s <<  8) | (r      )));
  }


  /*
   * @brief C custom defined QADD16 for M3 and M0 processors
   */
  static __INLINE uint32_t __QADD16(
  uint32_t x,
  uint32_t y)
  {
/*  q31_t r,     s;  without initialisation 'arm_offset_q15 test' fails  but 'intrinsic' tests pass! for armCC */
    q31_t r = 0, s = 0;

    r = __SSAT(((((q31_t)x << 16) >> 16) + (((q31_t)y << 16) >> 16)), 16) & (int32_t)0x0000FFFF;
    s = __SSAT(((((q31_t)x      ) >> 16) + (((q31_t)y      ) >> 16)), 16) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r      )));
  }


  /*
   * @brief C custom defined SHADD16 for M3 and M0 processors
   */
  static __INLINE uint32_t __SHADD16(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s;

    r = (((((q31_t)x << 16) >> 16) + (((q31_t)y << 16) >> 16)) >> 1) & (int32_t)0x0000FFFF;
    s = (((((q31_t)x      ) >> 16) + (((q31_t)y      ) >> 16)) >> 1) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r      )));
  }


  /*
   * @brief C custom defined QSUB16 for M3 and M0 processors
   */
  static __INLINE uint32_t __QSUB16(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s;

    r = __SSAT(((((q31_t)x << 16) >> 16) - (((q31_t)y << 16) >> 16)), 16) & (int32_t)0x0000FFFF;
    s = __SSAT(((((q31_t)x      ) >> 16) - (((q31_t)y      ) >> 16)), 16) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r      )));
  }


  /*
   * @brief C custom defined SHSUB16 for M3 and M0 processors
   */
  static __INLINE uint32_t __SHSUB16(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s;

    r = (((((q31_t)x << 16) >> 16) - (((q31_t)y << 16) >> 16)) >> 1) & (int32_t)0x0000FFFF;
    s = (((((q31_t)x      ) >> 16) - (((q31_t)y      ) >> 16)) >> 1) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r      )));
  }


  /*
   * @brief C custom defined QASX for M3 and M0 processors
   */
  static __INLINE uint32_t __QASX(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s;

    r = __SSAT(((((q31_t)x << 16) >> 16) - (((q31_t)y      ) >> 16)), 16) & (int32_t)0x0000FFFF;
    s = __SSAT(((((q31_t)x      ) >> 16) + (((q31_t)y << 16) >> 16)), 16) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r      )));
  }


  /*
   * @brief C custom defined SHASX for M3 and M0 processors
   */
  static __INLINE uint32_t __SHASX(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s;

    r = (((((q31_t)x << 16) >> 16) - (((q31_t)y      ) >> 16)) >> 1) & (int32_t)0x0000FFFF;
    s = (((((q31_t)x      ) >> 16) + (((q31_t)y << 16) >> 16)) >> 1) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r      )));
  }


  /*
   * @brief C custom defined QSAX for M3 and M0 processors
   */
  static __INLINE uint32_t __QSAX(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s;

    r = __SSAT(((((q31_t)x << 16) >> 16) + (((q31_t)y      ) >> 16)), 16) & (int32_t)0x0000FFFF;
    s = __SSAT(((((q31_t)x      ) >> 16) - (((q31_t)y << 16) >> 16)), 16) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r      )));
  }


  /*
   * @brief C custom defined SHSAX for M3 and M0 processors
   */
  static __INLINE uint32_t __SHSAX(
  uint32_t x,
  uint32_t y)
  {
    q31_t r, s;

    r = (((((q31_t)x << 16) >> 16) + (((q31_t)y      ) >> 16)) >> 1) & (int32_t)0x0000FFFF;
    s = (((((q31_t)x      ) >> 16) - (((q31_t)y << 16) >> 16)) >> 1) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r      )));
  }


  /*
   * @brief C custom defined SMUSDX for M3 and M0 processors
   */
  static __INLINE uint32_t __SMUSDX(
  uint32_t x,
  uint32_t y)
  {
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y      ) >> 16)) -
                       ((((q31_t)x      ) >> 16) * (((q31_t)y << 16) >> 16))   ));
  }

  /*
   * @brief C custom defined SMUADX for M3 and M0 processors
   */
  static __INLINE uint32_t __SMUADX(
  uint32_t x,
  uint32_t y)
  {
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y      ) >> 16)) +
                       ((((q31_t)x      ) >> 16) * (((q31_t)y << 16) >> 16))   ));
  }


  /*
   * @brief C custom defined QADD for M3 and M0 processors
   */
  static __INLINE int32_t __QADD(
  int32_t x,
  int32_t y)
  {
    return ((int32_t)(clip_q63_to_q31((q63_t)x + (q31_t)y)));
  }


  /*
   * @brief C custom defined QSUB for M3 and M0 processors
   */
  static __INLINE int32_t __QSUB(
  int32_t x,
  int32_t y)
  {
    return ((int32_t)(clip_q63_to_q31((q63_t)x - (q31_t)y)));
  }


  /*
   * @brief C custom defined SMLAD for M3 and M0 processors
   */
  static __INLINE uint32_t __SMLAD(
  uint32_t x,
  uint32_t y,
  uint32_t sum)
  {
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y << 16) >> 16)) +
                       ((((q31_t)x      ) >> 16) * (((q31_t)y      ) >> 16)) +
                       ( ((q31_t)sum    )                                  )   ));
  }


  /*
   * @brief C custom defined SMLADX for M3 and M0 processors
   */
  static __INLINE uint32_t __SMLADX(
  uint32_t x,
  uint32_t y,
  uint32_t sum)
  {
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y      ) >> 16)) +
                       ((((q31_t)x      ) >> 16) * (((q31_t)y << 16) >> 16)) +
                       ( ((q31_t)sum    )                                  )   ));
  }


  /*
   * @brief C custom defined SMLSDX for M3 and M0 processors
   */
  static __INLINE uint32_t __SMLSDX(
  uint32_t x,
  uint32_t y,
  uint32_t sum)
  {
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y      ) >> 16)) -
                       ((((q31_t)x      ) >> 16) * (((q31_t)y << 16) >> 16)) +
                       ( ((q31_t)sum    )                                  )   ));
  }


  /*
   * @brief C custom defined SMLALD for M3 and M0 processors
   */
  static __INLINE uint64_t __SMLALD(
  uint32_t x,
  uint32_t y,
  uint64_t sum)
  {
/*  return (sum + ((q15_t) (x >> 16) * (q15_t) (y >> 16)) + ((q15_t) x * (q15_t) y)); */
    return ((uint64_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y << 16) >> 16)) +
                       ((((q31_t)x      ) >> 16) * (((q31_t)y      ) >> 16)) +
                       ( ((q63_t)sum    )                                  )   ));
  }


  /*
   * @brief C custom defined SMLALDX for M3 and M0 processors
   */
  static __INLINE uint64_t __SMLALDX(
  uint32_t x,
  uint32_t y,
  uint64_t sum)
  {
/*  return (sum + ((q15_t) (x >> 16) * (q15_t) y)) + ((q15_t) x * (q15_t) (y >> 16)); */
    return ((uint64_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y      ) >> 16)) +
                       ((((q31_t)x      ) >> 16) * (((q31_t)y << 16) >> 16)) +
                       ( ((q63_t)sum    )                                  )   ));
  }


  /*
   * @brief C custom defined SMUAD for M3 and M0 processors
   */
  static __INLINE uint32_t __SMUAD(
  uint32_t x,
  uint32_t y)
  {
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y << 16) >> 16)) +
                       ((((q31_t)x      ) >> 16) * (((q31_t)y      ) >> 16))   ));
  }


  /*
   * @brief C custom defined SMUSD for M3 and M0 processors
   */
  static __INLINE uint32_t __SMUSD(
  uint32_t x,
  uint32_t y)
  {
    return ((uint32_t)(((((q31_t)x << 16) >> 16) * (((q31_t)y << 16) >> 16)) -
                       ((((q31_t)x      ) >> 16) * (((q31_t)y      ) >> 16))   ));
  }


  /*
   * @brief C custom defined SXTB16 for M3 and M0 processors
   */
  static __INLINE uint32_t __SXTB16(
  uint32_t x)
  {
    return ((uint32_t)(((((q31_t)x << 24) >> 24) & (q31_t)0x0000FFFF) |
                       ((((q31_t)x <<  8) >>  8) & (q31_t)0xFFFF0000)  ));
  }

#endif /* defined (ARM_MATH_CM3) || defined (ARM_MATH_CM0_FAMILY) */


  /**
   * @brief Instance structure for the Q7 FIR filter.
   */
  typedef struct
  {
    uint16_t numTaps;        /**< number of filter coefficients in the filter. */
    q7_t *pState;            /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    q7_t *pCoeffs;           /**< points to the coefficient array. The array is of length numTaps.*/
  } arm_fir_instance_q7;

  /**
   * @brief Instance structure for the Q15 FIR filter.
   */
  typedef struct
  {
    uint16_t numTaps;         /**< number of filter coefficients in the filter. */
    q15_t *pState;            /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    q15_t *pCoeffs;           /**< points to the coefficient array. The array is of length numTaps.*/
  } arm_fir_instance_q15;

  /**
   * @brief Instance structure for the Q31 FIR filter.
   */
  typedef struct
  {
    uint16_t numTaps;         /**< number of filter coefficients in the filter. */
    q31_t *pState;            /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    q31_t *pCoeffs;           /**< points to the coefficient array. The array is of length numTaps. */
  } arm_fir_instance_q31;

  /**
   * @brief Instance structure for the floating-point FIR filter.
   */
  typedef struct
  {
    uint16_t numTaps;     /**< number of filter coefficients in the filter. */
    float32_t *pState;    /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    float32_t *pCoeffs;   /**< points to the coefficient array. The array is of length numTaps. */
  } arm_fir_instance_f32;


  /**
   * @brief Processing function for the Q7 FIR filter.
   * @param[in]  S          points to an instance of the Q7 FIR filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_fir_q7(
  const arm_fir_instance_q7 * S,
  q7_t * pSrc,
  q7_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q7 FIR filter.
   * @param[in,out] S          points to an instance of the Q7 FIR structure.
   * @param[in]     numTaps    Number of filter coefficients in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of samples that are processed.
   */
  void arm_fir_init_q7(
  arm_fir_instance_q7 * S,
  uint16_t numTaps,
  q7_t * pCoeffs,
  q7_t * pState,
  uint32_t blockSize);


  /**
   * @brief Processing function for the Q15 FIR filter.
   * @param[in]  S          points to an instance of the Q15 FIR structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_fir_q15(
  const arm_fir_instance_q15 * S,
  q15_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Processing function for the fast Q15 FIR filter for Cortex-M3 and Cortex-M4.
   * @param[in]  S          points to an instance of the Q15 FIR filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_fir_fast_q15(
  const arm_fir_instance_q15 * S,
  q15_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q15 FIR filter.
   * @param[in,out] S          points to an instance of the Q15 FIR filter structure.
   * @param[in]     numTaps    Number of filter coefficients in the filter. Must be even and greater than or equal to 4.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of samples that are processed at a time.
   * @return The function returns ARM_MATH_SUCCESS if initialization was successful or ARM_MATH_ARGUMENT_ERROR if
   * <code>numTaps</code> is not a supported value.
   */
  arm_status arm_fir_init_q15(
  arm_fir_instance_q15 * S,
  uint16_t numTaps,
  q15_t * pCoeffs,
  q15_t * pState,
  uint32_t blockSize);


  /**
   * @brief Processing function for the Q31 FIR filter.
   * @param[in]  S          points to an instance of the Q31 FIR filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_fir_q31(
  const arm_fir_instance_q31 * S,
  q31_t * pSrc,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Processing function for the fast Q31 FIR filter for Cortex-M3 and Cortex-M4.
   * @param[in]  S          points to an instance of the Q31 FIR structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_fir_fast_q31(
  const arm_fir_instance_q31 * S,
  q31_t * pSrc,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q31 FIR filter.
   * @param[in,out] S          points to an instance of the Q31 FIR structure.
   * @param[in]     numTaps    Number of filter coefficients in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of samples that are processed at a time.
   */
  void arm_fir_init_q31(
  arm_fir_instance_q31 * S,
  uint16_t numTaps,
  q31_t * pCoeffs,
  q31_t * pState,
  uint32_t blockSize);


  /**
   * @brief Processing function for the floating-point FIR filter.
   * @param[in]  S          points to an instance of the floating-point FIR structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_fir_f32(
  const arm_fir_instance_f32 * S,
  float32_t * pSrc,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Initialization function for the floating-point FIR filter.
   * @param[in,out] S          points to an instance of the floating-point FIR filter structure.
   * @param[in]     numTaps    Number of filter coefficients in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of samples that are processed at a time.
   */
  void arm_fir_init_f32(
  arm_fir_instance_f32 * S,
  uint16_t numTaps,
  float32_t * pCoeffs,
  float32_t * pState,
  uint32_t blockSize);


  /**
   * @brief Instance structure for the Q15 Biquad cascade filter.
   */
  typedef struct
  {
    int8_t numStages;        /**< number of 2nd order stages in the filter.  Overall order is 2*numStages. */
    q15_t *pState;           /**< Points to the array of state coefficients.  The array is of length 4*numStages. */
    q15_t *pCoeffs;          /**< Points to the array of coefficients.  The array is of length 5*numStages. */
    int8_t postShift;        /**< Additional shift, in bits, applied to each output sample. */
  } arm_biquad_casd_df1_inst_q15;

  /**
   * @brief Instance structure for the Q31 Biquad cascade filter.
   */
  typedef struct
  {
    uint32_t numStages;      /**< number of 2nd order stages in the filter.  Overall order is 2*numStages. */
    q31_t *pState;           /**< Points to the array of state coefficients.  The array is of length 4*numStages. */
    q31_t *pCoeffs;          /**< Points to the array of coefficients.  The array is of length 5*numStages. */
    uint8_t postShift;       /**< Additional shift, in bits, applied to each output sample. */
  } arm_biquad_casd_df1_inst_q31;

  /**
   * @brief Instance structure for the floating-point Biquad cascade filter.
   */
  typedef struct
  {
    uint32_t numStages;      /**< number of 2nd order stages in the filter.  Overall order is 2*numStages. */
    float32_t *pState;       /**< Points to the array of state coefficients.  The array is of length 4*numStages. */
    float32_t *pCoeffs;      /**< Points to the array of coefficients.  The array is of length 5*numStages. */
  } arm_biquad_casd_df1_inst_f32;


  /**
   * @brief Processing function for the Q15 Biquad cascade filter.
   * @param[in]  S          points to an instance of the Q15 Biquad cascade structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_biquad_cascade_df1_q15(
  const arm_biquad_casd_df1_inst_q15 * S,
  q15_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q15 Biquad cascade filter.
   * @param[in,out] S          points to an instance of the Q15 Biquad cascade structure.
   * @param[in]     numStages  number of 2nd order stages in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     postShift  Shift to be applied to the output. Varies according to the coefficients format
   */
  void arm_biquad_cascade_df1_init_q15(
  arm_biquad_casd_df1_inst_q15 * S,
  uint8_t numStages,
  q15_t * pCoeffs,
  q15_t * pState,
  int8_t postShift);


  /**
   * @brief Fast but less precise processing function for the Q15 Biquad cascade filter for Cortex-M3 and Cortex-M4.
   * @param[in]  S          points to an instance of the Q15 Biquad cascade structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_biquad_cascade_df1_fast_q15(
  const arm_biquad_casd_df1_inst_q15 * S,
  q15_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Processing function for the Q31 Biquad cascade filter
   * @param[in]  S          points to an instance of the Q31 Biquad cascade structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_biquad_cascade_df1_q31(
  const arm_biquad_casd_df1_inst_q31 * S,
  q31_t * pSrc,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Fast but less precise processing function for the Q31 Biquad cascade filter for Cortex-M3 and Cortex-M4.
   * @param[in]  S          points to an instance of the Q31 Biquad cascade structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_biquad_cascade_df1_fast_q31(
  const arm_biquad_casd_df1_inst_q31 * S,
  q31_t * pSrc,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q31 Biquad cascade filter.
   * @param[in,out] S          points to an instance of the Q31 Biquad cascade structure.
   * @param[in]     numStages  number of 2nd order stages in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     postShift  Shift to be applied to the output. Varies according to the coefficients format
   */
  void arm_biquad_cascade_df1_init_q31(
  arm_biquad_casd_df1_inst_q31 * S,
  uint8_t numStages,
  q31_t * pCoeffs,
  q31_t * pState,
  int8_t postShift);


  /**
   * @brief Processing function for the floating-point Biquad cascade filter.
   * @param[in]  S          points to an instance of the floating-point Biquad cascade structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_biquad_cascade_df1_f32(
  const arm_biquad_casd_df1_inst_f32 * S,
  float32_t * pSrc,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Initialization function for the floating-point Biquad cascade filter.
   * @param[in,out] S          points to an instance of the floating-point Biquad cascade structure.
   * @param[in]     numStages  number of 2nd order stages in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   */
  void arm_biquad_cascade_df1_init_f32(
  arm_biquad_casd_df1_inst_f32 * S,
  uint8_t numStages,
  float32_t * pCoeffs,
  float32_t * pState);


  /**
   * @brief Instance structure for the floating-point matrix structure.
   */
  typedef struct
  {
    uint16_t numRows;     /**< number of rows of the matrix.     */
    uint16_t numCols;     /**< number of columns of the matrix.  */
    float32_t *pData;     /**< points to the data of the matrix. */
  } arm_matrix_instance_f32;


  /**
   * @brief Instance structure for the floating-point matrix structure.
   */
  typedef struct
  {
    uint16_t numRows;     /**< number of rows of the matrix.     */
    uint16_t numCols;     /**< number of columns of the matrix.  */
    float64_t *pData;     /**< points to the data of the matrix. */
  } arm_matrix_instance_f64;

  /**
   * @brief Instance structure for the Q15 matrix structure.
   */
  typedef struct
  {
    uint16_t numRows;     /**< number of rows of the matrix.     */
    uint16_t numCols;     /**< number of columns of the matrix.  */
    q15_t *pData;         /**< points to the data of the matrix. */
  } arm_matrix_instance_q15;

  /**
   * @brief Instance structure for the Q31 matrix structure.
   */
  typedef struct
  {
    uint16_t numRows;     /**< number of rows of the matrix.     */
    uint16_t numCols;     /**< number of columns of the matrix.  */
    q31_t *pData;         /**< points to the data of the matrix. */
  } arm_matrix_instance_q31;


  /**
   * @brief Floating-point matrix addition.
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_add_f32(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
  arm_matrix_instance_f32 * pDst);


  /**
   * @brief Q15 matrix addition.
   * @param[in]   pSrcA  points to the first input matrix structure
   * @param[in]   pSrcB  points to the second input matrix structure
   * @param[out]  pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_add_q15(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
  arm_matrix_instance_q15 * pDst);


  /**
   * @brief Q31 matrix addition.
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_add_q31(
  const arm_matrix_instance_q31 * pSrcA,
  const arm_matrix_instance_q31 * pSrcB,
  arm_matrix_instance_q31 * pDst);


  /**
   * @brief Floating-point, complex, matrix multiplication.
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_cmplx_mult_f32(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
  arm_matrix_instance_f32 * pDst);


  /**
   * @brief Q15, complex,  matrix multiplication.
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_cmplx_mult_q15(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
  arm_matrix_instance_q15 * pDst,
  q15_t * pScratch);


  /**
   * @brief Q31, complex, matrix multiplication.
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_cmplx_mult_q31(
  const arm_matrix_instance_q31 * pSrcA,
  const arm_matrix_instance_q31 * pSrcB,
  arm_matrix_instance_q31 * pDst);


  /**
   * @brief Floating-point matrix transpose.
   * @param[in]  pSrc  points to the input matrix
   * @param[out] pDst  points to the output matrix
   * @return    The function returns either  <code>ARM_MATH_SIZE_MISMATCH</code>
   * or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_trans_f32(
  const arm_matrix_instance_f32 * pSrc,
  arm_matrix_instance_f32 * pDst);


  /**
   * @brief Q15 matrix transpose.
   * @param[in]  pSrc  points to the input matrix
   * @param[out] pDst  points to the output matrix
   * @return    The function returns either  <code>ARM_MATH_SIZE_MISMATCH</code>
   * or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_trans_q15(
  const arm_matrix_instance_q15 * pSrc,
  arm_matrix_instance_q15 * pDst);


  /**
   * @brief Q31 matrix transpose.
   * @param[in]  pSrc  points to the input matrix
   * @param[out] pDst  points to the output matrix
   * @return    The function returns either  <code>ARM_MATH_SIZE_MISMATCH</code>
   * or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_trans_q31(
  const arm_matrix_instance_q31 * pSrc,
  arm_matrix_instance_q31 * pDst);


  /**
   * @brief Floating-point matrix multiplication
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_mult_f32(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
  arm_matrix_instance_f32 * pDst);


  /**
   * @brief Q15 matrix multiplication
   * @param[in]  pSrcA   points to the first input matrix structure
   * @param[in]  pSrcB   points to the second input matrix structure
   * @param[out] pDst    points to output matrix structure
   * @param[in]  pState  points to the array for storing intermediate results
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_mult_q15(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
  arm_matrix_instance_q15 * pDst,
  q15_t * pState);


  /**
   * @brief Q15 matrix multiplication (fast variant) for Cortex-M3 and Cortex-M4
   * @param[in]  pSrcA   points to the first input matrix structure
   * @param[in]  pSrcB   points to the second input matrix structure
   * @param[out] pDst    points to output matrix structure
   * @param[in]  pState  points to the array for storing intermediate results
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_mult_fast_q15(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
  arm_matrix_instance_q15 * pDst,
  q15_t * pState);


  /**
   * @brief Q31 matrix multiplication
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_mult_q31(
  const arm_matrix_instance_q31 * pSrcA,
  const arm_matrix_instance_q31 * pSrcB,
  arm_matrix_instance_q31 * pDst);


  /**
   * @brief Q31 matrix multiplication (fast variant) for Cortex-M3 and Cortex-M4
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_mult_fast_q31(
  const arm_matrix_instance_q31 * pSrcA,
  const arm_matrix_instance_q31 * pSrcB,
  arm_matrix_instance_q31 * pDst);


  /**
   * @brief Floating-point matrix subtraction
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_sub_f32(
  const arm_matrix_instance_f32 * pSrcA,
  const arm_matrix_instance_f32 * pSrcB,
  arm_matrix_instance_f32 * pDst);


  /**
   * @brief Q15 matrix subtraction
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_sub_q15(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
  arm_matrix_instance_q15 * pDst);


  /**
   * @brief Q31 matrix subtraction
   * @param[in]  pSrcA  points to the first input matrix structure
   * @param[in]  pSrcB  points to the second input matrix structure
   * @param[out] pDst   points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_sub_q31(
  const arm_matrix_instance_q31 * pSrcA,
  const arm_matrix_instance_q31 * pSrcB,
  arm_matrix_instance_q31 * pDst);


  /**
   * @brief Floating-point matrix scaling.
   * @param[in]  pSrc   points to the input matrix
   * @param[in]  scale  scale factor
   * @param[out] pDst   points to the output matrix
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_scale_f32(
  const arm_matrix_instance_f32 * pSrc,
  float32_t scale,
  arm_matrix_instance_f32 * pDst);


  /**
   * @brief Q15 matrix scaling.
   * @param[in]  pSrc        points to input matrix
   * @param[in]  scaleFract  fractional portion of the scale factor
   * @param[in]  shift       number of bits to shift the result by
   * @param[out] pDst        points to output matrix
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_scale_q15(
  const arm_matrix_instance_q15 * pSrc,
  q15_t scaleFract,
  int32_t shift,
  arm_matrix_instance_q15 * pDst);


  /**
   * @brief Q31 matrix scaling.
   * @param[in]  pSrc        points to input matrix
   * @param[in]  scaleFract  fractional portion of the scale factor
   * @param[in]  shift       number of bits to shift the result by
   * @param[out] pDst        points to output matrix structure
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
  arm_status arm_mat_scale_q31(
  const arm_matrix_instance_q31 * pSrc,
  q31_t scaleFract,
  int32_t shift,
  arm_matrix_instance_q31 * pDst);


  /**
   * @brief  Q31 matrix initialization.
   * @param[in,out] S         points to an instance of the floating-point matrix structure.
   * @param[in]     nRows     number of rows in the matrix.
   * @param[in]     nColumns  number of columns in the matrix.
   * @param[in]     pData     points to the matrix data array.
   */
  void arm_mat_init_q31(
  arm_matrix_instance_q31 * S,
  uint16_t nRows,
  uint16_t nColumns,
  q31_t * pData);


  /**
   * @brief  Q15 matrix initialization.
   * @param[in,out] S         points to an instance of the floating-point matrix structure.
   * @param[in]     nRows     number of rows in the matrix.
   * @param[in]     nColumns  number of columns in the matrix.
   * @param[in]     pData     points to the matrix data array.
   */
  void arm_mat_init_q15(
  arm_matrix_instance_q15 * S,
  uint16_t nRows,
  uint16_t nColumns,
  q15_t * pData);


  /**
   * @brief  Floating-point matrix initialization.
   * @param[in,out] S         points to an instance of the floating-point matrix structure.
   * @param[in]     nRows     number of rows in the matrix.
   * @param[in]     nColumns  number of columns in the matrix.
   * @param[in]     pData     points to the matrix data array.
   */
  void arm_mat_init_f32(
  arm_matrix_instance_f32 * S,
  uint16_t nRows,
  uint16_t nColumns,
  float32_t * pData);



  /**
   * @brief Instance structure for the Q15 PID Control.
   */
  typedef struct
  {
    q15_t A0;           /**< The derived gain, A0 = Kp + Ki + Kd . */
#ifdef ARM_MATH_CM0_FAMILY
    q15_t A1;
    q15_t A2;
#else
    q31_t A1;           /**< The derived gain A1 = -Kp - 2Kd | Kd.*/
#endif
    q15_t state[3];     /**< The state array of length 3. */
    q15_t Kp;           /**< The proportional gain. */
    q15_t Ki;           /**< The integral gain. */
    q15_t Kd;           /**< The derivative gain. */
  } arm_pid_instance_q15;

  /**
   * @brief Instance structure for the Q31 PID Control.
   */
  typedef struct
  {
    q31_t A0;            /**< The derived gain, A0 = Kp + Ki + Kd . */
    q31_t A1;            /**< The derived gain, A1 = -Kp - 2Kd. */
    q31_t A2;            /**< The derived gain, A2 = Kd . */
    q31_t state[3];      /**< The state array of length 3. */
    q31_t Kp;            /**< The proportional gain. */
    q31_t Ki;            /**< The integral gain. */
    q31_t Kd;            /**< The derivative gain. */
  } arm_pid_instance_q31;

  /**
   * @brief Instance structure for the floating-point PID Control.
   */
  typedef struct
  {
    float32_t A0;          /**< The derived gain, A0 = Kp + Ki + Kd . */
    float32_t A1;          /**< The derived gain, A1 = -Kp - 2Kd. */
    float32_t A2;          /**< The derived gain, A2 = Kd . */
    float32_t state[3];    /**< The state array of length 3. */
    float32_t Kp;          /**< The proportional gain. */
    float32_t Ki;          /**< The integral gain. */
    float32_t Kd;          /**< The derivative gain. */
  } arm_pid_instance_f32;



  /**
   * @brief  Initialization function for the floating-point PID Control.
   * @param[in,out] S               points to an instance of the PID structure.
   * @param[in]     resetStateFlag  flag to reset the state. 0 = no change in state 1 = reset the state.
   */
  void arm_pid_init_f32(
  arm_pid_instance_f32 * S,
  int32_t resetStateFlag);


  /**
   * @brief  Reset function for the floating-point PID Control.
   * @param[in,out] S  is an instance of the floating-point PID Control structure
   */
  void arm_pid_reset_f32(
  arm_pid_instance_f32 * S);


  /**
   * @brief  Initialization function for the Q31 PID Control.
   * @param[in,out] S               points to an instance of the Q15 PID structure.
   * @param[in]     resetStateFlag  flag to reset the state. 0 = no change in state 1 = reset the state.
   */
  void arm_pid_init_q31(
  arm_pid_instance_q31 * S,
  int32_t resetStateFlag);


  /**
   * @brief  Reset function for the Q31 PID Control.
   * @param[in,out] S   points to an instance of the Q31 PID Control structure
   */

  void arm_pid_reset_q31(
  arm_pid_instance_q31 * S);


  /**
   * @brief  Initialization function for the Q15 PID Control.
   * @param[in,out] S               points to an instance of the Q15 PID structure.
   * @param[in]     resetStateFlag  flag to reset the state. 0 = no change in state 1 = reset the state.
   */
  void arm_pid_init_q15(
  arm_pid_instance_q15 * S,
  int32_t resetStateFlag);


  /**
   * @brief  Reset function for the Q15 PID Control.
   * @param[in,out] S  points to an instance of the q15 PID Control structure
   */
  void arm_pid_reset_q15(
  arm_pid_instance_q15 * S);


  /**
   * @brief Instance structure for the floating-point Linear Interpolate function.
   */
  typedef struct
  {
    uint32_t nValues;           /**< nValues */
    float32_t x1;               /**< x1 */
    float32_t xSpacing;         /**< xSpacing */
    float32_t *pYData;          /**< pointer to the table of Y values */
  } arm_linear_interp_instance_f32;

  /**
   * @brief Instance structure for the floating-point bilinear interpolation function.
   */
  typedef struct
  {
    uint16_t numRows;   /**< number of rows in the data table. */
    uint16_t numCols;   /**< number of columns in the data table. */
    float32_t *pData;   /**< points to the data table. */
  } arm_bilinear_interp_instance_f32;

   /**
   * @brief Instance structure for the Q31 bilinear interpolation function.
   */
  typedef struct
  {
    uint16_t numRows;   /**< number of rows in the data table. */
    uint16_t numCols;   /**< number of columns in the data table. */
    q31_t *pData;       /**< points to the data table. */
  } arm_bilinear_interp_instance_q31;

   /**
   * @brief Instance structure for the Q15 bilinear interpolation function.
   */
  typedef struct
  {
    uint16_t numRows;   /**< number of rows in the data table. */
    uint16_t numCols;   /**< number of columns in the data table. */
    q15_t *pData;       /**< points to the data table. */
  } arm_bilinear_interp_instance_q15;

   /**
   * @brief Instance structure for the Q15 bilinear interpolation function.
   */
  typedef struct
  {
    uint16_t numRows;   /**< number of rows in the data table. */
    uint16_t numCols;   /**< number of columns in the data table. */
    q7_t *pData;        /**< points to the data table. */
  } arm_bilinear_interp_instance_q7;


  /**
   * @brief Q7 vector multiplication.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_mult_q7(
  q7_t * pSrcA,
  q7_t * pSrcB,
  q7_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Q15 vector multiplication.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_mult_q15(
  q15_t * pSrcA,
  q15_t * pSrcB,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Q31 vector multiplication.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_mult_q31(
  q31_t * pSrcA,
  q31_t * pSrcB,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Floating-point vector multiplication.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_mult_f32(
  float32_t * pSrcA,
  float32_t * pSrcB,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Instance structure for the Q15 CFFT/CIFFT function.
   */
  typedef struct
  {
    uint16_t fftLen;                 /**< length of the FFT. */
    uint8_t ifftFlag;                /**< flag that selects forward (ifftFlag=0) or inverse (ifftFlag=1) transform. */
    uint8_t bitReverseFlag;          /**< flag that enables (bitReverseFlag=1) or disables (bitReverseFlag=0) bit reversal of output. */
    q15_t *pTwiddle;                 /**< points to the Sin twiddle factor table. */
    uint16_t *pBitRevTable;          /**< points to the bit reversal table. */
    uint16_t twidCoefModifier;       /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
    uint16_t bitRevFactor;           /**< bit reversal modifier that supports different size FFTs with the same bit reversal table. */
  } arm_cfft_radix2_instance_q15;

/* Deprecated */
  arm_status arm_cfft_radix2_init_q15(
  arm_cfft_radix2_instance_q15 * S,
  uint16_t fftLen,
  uint8_t ifftFlag,
  uint8_t bitReverseFlag);

/* Deprecated */
  void arm_cfft_radix2_q15(
  const arm_cfft_radix2_instance_q15 * S,
  q15_t * pSrc);


  /**
   * @brief Instance structure for the Q15 CFFT/CIFFT function.
   */
  typedef struct
  {
    uint16_t fftLen;                 /**< length of the FFT. */
    uint8_t ifftFlag;                /**< flag that selects forward (ifftFlag=0) or inverse (ifftFlag=1) transform. */
    uint8_t bitReverseFlag;          /**< flag that enables (bitReverseFlag=1) or disables (bitReverseFlag=0) bit reversal of output. */
    q15_t *pTwiddle;                 /**< points to the twiddle factor table. */
    uint16_t *pBitRevTable;          /**< points to the bit reversal table. */
    uint16_t twidCoefModifier;       /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
    uint16_t bitRevFactor;           /**< bit reversal modifier that supports different size FFTs with the same bit reversal table. */
  } arm_cfft_radix4_instance_q15;

/* Deprecated */
  arm_status arm_cfft_radix4_init_q15(
  arm_cfft_radix4_instance_q15 * S,
  uint16_t fftLen,
  uint8_t ifftFlag,
  uint8_t bitReverseFlag);

/* Deprecated */
  void arm_cfft_radix4_q15(
  const arm_cfft_radix4_instance_q15 * S,
  q15_t * pSrc);

  /**
   * @brief Instance structure for the Radix-2 Q31 CFFT/CIFFT function.
   */
  typedef struct
  {
    uint16_t fftLen;                 /**< length of the FFT. */
    uint8_t ifftFlag;                /**< flag that selects forward (ifftFlag=0) or inverse (ifftFlag=1) transform. */
    uint8_t bitReverseFlag;          /**< flag that enables (bitReverseFlag=1) or disables (bitReverseFlag=0) bit reversal of output. */
    q31_t *pTwiddle;                 /**< points to the Twiddle factor table. */
    uint16_t *pBitRevTable;          /**< points to the bit reversal table. */
    uint16_t twidCoefModifier;       /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
    uint16_t bitRevFactor;           /**< bit reversal modifier that supports different size FFTs with the same bit reversal table. */
  } arm_cfft_radix2_instance_q31;

/* Deprecated */
  arm_status arm_cfft_radix2_init_q31(
  arm_cfft_radix2_instance_q31 * S,
  uint16_t fftLen,
  uint8_t ifftFlag,
  uint8_t bitReverseFlag);

/* Deprecated */
  void arm_cfft_radix2_q31(
  const arm_cfft_radix2_instance_q31 * S,
  q31_t * pSrc);

  /**
   * @brief Instance structure for the Q31 CFFT/CIFFT function.
   */
  typedef struct
  {
    uint16_t fftLen;                 /**< length of the FFT. */
    uint8_t ifftFlag;                /**< flag that selects forward (ifftFlag=0) or inverse (ifftFlag=1) transform. */
    uint8_t bitReverseFlag;          /**< flag that enables (bitReverseFlag=1) or disables (bitReverseFlag=0) bit reversal of output. */
    q31_t *pTwiddle;                 /**< points to the twiddle factor table. */
    uint16_t *pBitRevTable;          /**< points to the bit reversal table. */
    uint16_t twidCoefModifier;       /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
    uint16_t bitRevFactor;           /**< bit reversal modifier that supports different size FFTs with the same bit reversal table. */
  } arm_cfft_radix4_instance_q31;

/* Deprecated */
  void arm_cfft_radix4_q31(
  const arm_cfft_radix4_instance_q31 * S,
  q31_t * pSrc);

/* Deprecated */
  arm_status arm_cfft_radix4_init_q31(
  arm_cfft_radix4_instance_q31 * S,
  uint16_t fftLen,
  uint8_t ifftFlag,
  uint8_t bitReverseFlag);

  /**
   * @brief Instance structure for the floating-point CFFT/CIFFT function.
   */
  typedef struct
  {
    uint16_t fftLen;                   /**< length of the FFT. */
    uint8_t ifftFlag;                  /**< flag that selects forward (ifftFlag=0) or inverse (ifftFlag=1) transform. */
    uint8_t bitReverseFlag;            /**< flag that enables (bitReverseFlag=1) or disables (bitReverseFlag=0) bit reversal of output. */
    float32_t *pTwiddle;               /**< points to the Twiddle factor table. */
    uint16_t *pBitRevTable;            /**< points to the bit reversal table. */
    uint16_t twidCoefModifier;         /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
    uint16_t bitRevFactor;             /**< bit reversal modifier that supports different size FFTs with the same bit reversal table. */
    float32_t onebyfftLen;             /**< value of 1/fftLen. */
  } arm_cfft_radix2_instance_f32;

/* Deprecated */
  arm_status arm_cfft_radix2_init_f32(
  arm_cfft_radix2_instance_f32 * S,
  uint16_t fftLen,
  uint8_t ifftFlag,
  uint8_t bitReverseFlag);

/* Deprecated */
  void arm_cfft_radix2_f32(
  const arm_cfft_radix2_instance_f32 * S,
  float32_t * pSrc);

  /**
   * @brief Instance structure for the floating-point CFFT/CIFFT function.
   */
  typedef struct
  {
    uint16_t fftLen;                   /**< length of the FFT. */
    uint8_t ifftFlag;                  /**< flag that selects forward (ifftFlag=0) or inverse (ifftFlag=1) transform. */
    uint8_t bitReverseFlag;            /**< flag that enables (bitReverseFlag=1) or disables (bitReverseFlag=0) bit reversal of output. */
    float32_t *pTwiddle;               /**< points to the Twiddle factor table. */
    uint16_t *pBitRevTable;            /**< points to the bit reversal table. */
    uint16_t twidCoefModifier;         /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
    uint16_t bitRevFactor;             /**< bit reversal modifier that supports different size FFTs with the same bit reversal table. */
    float32_t onebyfftLen;             /**< value of 1/fftLen. */
  } arm_cfft_radix4_instance_f32;

/* Deprecated */
  arm_status arm_cfft_radix4_init_f32(
  arm_cfft_radix4_instance_f32 * S,
  uint16_t fftLen,
  uint8_t ifftFlag,
  uint8_t bitReverseFlag);

/* Deprecated */
  void arm_cfft_radix4_f32(
  const arm_cfft_radix4_instance_f32 * S,
  float32_t * pSrc);

  /**
   * @brief Instance structure for the fixed-point CFFT/CIFFT function.
   */
  typedef struct
  {
    uint16_t fftLen;                   /**< length of the FFT. */
    const q15_t *pTwiddle;             /**< points to the Twiddle factor table. */
    const uint16_t *pBitRevTable;      /**< points to the bit reversal table. */
    uint16_t bitRevLength;             /**< bit reversal table length. */
  } arm_cfft_instance_q15;

void arm_cfft_q15(
    const arm_cfft_instance_q15 * S,
    q15_t * p1,
    uint8_t ifftFlag,
    uint8_t bitReverseFlag);

  /**
   * @brief Instance structure for the fixed-point CFFT/CIFFT function.
   */
  typedef struct
  {
    uint16_t fftLen;                   /**< length of the FFT. */
    const q31_t *pTwiddle;             /**< points to the Twiddle factor table. */
    const uint16_t *pBitRevTable;      /**< points to the bit reversal table. */
    uint16_t bitRevLength;             /**< bit reversal table length. */
  } arm_cfft_instance_q31;

void arm_cfft_q31(
    const arm_cfft_instance_q31 * S,
    q31_t * p1,
    uint8_t ifftFlag,
    uint8_t bitReverseFlag);

  /**
   * @brief Instance structure for the floating-point CFFT/CIFFT function.
   */
  typedef struct
  {
    uint16_t fftLen;                   /**< length of the FFT. */
    const float32_t *pTwiddle;         /**< points to the Twiddle factor table. */
    const uint16_t *pBitRevTable;      /**< points to the bit reversal table. */
    uint16_t bitRevLength;             /**< bit reversal table length. */
  } arm_cfft_instance_f32;

  void arm_cfft_f32(
  const arm_cfft_instance_f32 * S,
  float32_t * p1,
  uint8_t ifftFlag,
  uint8_t bitReverseFlag);

  /**
   * @brief Instance structure for the Q15 RFFT/RIFFT function.
   */
  typedef struct
  {
    uint32_t fftLenReal;                      /**< length of the real FFT. */
    uint8_t ifftFlagR;                        /**< flag that selects forward (ifftFlagR=0) or inverse (ifftFlagR=1) transform. */
    uint8_t bitReverseFlagR;                  /**< flag that enables (bitReverseFlagR=1) or disables (bitReverseFlagR=0) bit reversal of output. */
    uint32_t twidCoefRModifier;               /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
    q15_t *pTwiddleAReal;                     /**< points to the real twiddle factor table. */
    q15_t *pTwiddleBReal;                     /**< points to the imag twiddle factor table. */
    const arm_cfft_instance_q15 *pCfft;       /**< points to the complex FFT instance. */
  } arm_rfft_instance_q15;

  arm_status arm_rfft_init_q15(
  arm_rfft_instance_q15 * S,
  uint32_t fftLenReal,
  uint32_t ifftFlagR,
  uint32_t bitReverseFlag);

  void arm_rfft_q15(
  const arm_rfft_instance_q15 * S,
  q15_t * pSrc,
  q15_t * pDst);

  /**
   * @brief Instance structure for the Q31 RFFT/RIFFT function.
   */
  typedef struct
  {
    uint32_t fftLenReal;                        /**< length of the real FFT. */
    uint8_t ifftFlagR;                          /**< flag that selects forward (ifftFlagR=0) or inverse (ifftFlagR=1) transform. */
    uint8_t bitReverseFlagR;                    /**< flag that enables (bitReverseFlagR=1) or disables (bitReverseFlagR=0) bit reversal of output. */
    uint32_t twidCoefRModifier;                 /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
    q31_t *pTwiddleAReal;                       /**< points to the real twiddle factor table. */
    q31_t *pTwiddleBReal;                       /**< points to the imag twiddle factor table. */
    const arm_cfft_instance_q31 *pCfft;         /**< points to the complex FFT instance. */
  } arm_rfft_instance_q31;

  arm_status arm_rfft_init_q31(
  arm_rfft_instance_q31 * S,
  uint32_t fftLenReal,
  uint32_t ifftFlagR,
  uint32_t bitReverseFlag);

  void arm_rfft_q31(
  const arm_rfft_instance_q31 * S,
  q31_t * pSrc,
  q31_t * pDst);

  /**
   * @brief Instance structure for the floating-point RFFT/RIFFT function.
   */
  typedef struct
  {
    uint32_t fftLenReal;                        /**< length of the real FFT. */
    uint16_t fftLenBy2;                         /**< length of the complex FFT. */
    uint8_t ifftFlagR;                          /**< flag that selects forward (ifftFlagR=0) or inverse (ifftFlagR=1) transform. */
    uint8_t bitReverseFlagR;                    /**< flag that enables (bitReverseFlagR=1) or disables (bitReverseFlagR=0) bit reversal of output. */
    uint32_t twidCoefRModifier;                     /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
    float32_t *pTwiddleAReal;                   /**< points to the real twiddle factor table. */
    float32_t *pTwiddleBReal;                   /**< points to the imag twiddle factor table. */
    arm_cfft_radix4_instance_f32 *pCfft;        /**< points to the complex FFT instance. */
  } arm_rfft_instance_f32;

  arm_status arm_rfft_init_f32(
  arm_rfft_instance_f32 * S,
  arm_cfft_radix4_instance_f32 * S_CFFT,
  uint32_t fftLenReal,
  uint32_t ifftFlagR,
  uint32_t bitReverseFlag);

  void arm_rfft_f32(
  const arm_rfft_instance_f32 * S,
  float32_t * pSrc,
  float32_t * pDst);

  /**
   * @brief Instance structure for the floating-point RFFT/RIFFT function.
   */
typedef struct
  {
    arm_cfft_instance_f32 Sint;      /**< Internal CFFT structure. */
    uint16_t fftLenRFFT;             /**< length of the real sequence */
    float32_t * pTwiddleRFFT;        /**< Twiddle factors real stage  */
  } arm_rfft_fast_instance_f32 ;

arm_status arm_rfft_fast_init_f32 (
   arm_rfft_fast_instance_f32 * S,
   uint16_t fftLen);

void arm_rfft_fast_f32(
  arm_rfft_fast_instance_f32 * S,
  float32_t * p, float32_t * pOut,
  uint8_t ifftFlag);

  /**
   * @brief Instance structure for the floating-point DCT4/IDCT4 function.
   */
  typedef struct
  {
    uint16_t N;                          /**< length of the DCT4. */
    uint16_t Nby2;                       /**< half of the length of the DCT4. */
    float32_t normalize;                 /**< normalizing factor. */
    float32_t *pTwiddle;                 /**< points to the twiddle factor table. */
    float32_t *pCosFactor;               /**< points to the cosFactor table. */
    arm_rfft_instance_f32 *pRfft;        /**< points to the real FFT instance. */
    arm_cfft_radix4_instance_f32 *pCfft; /**< points to the complex FFT instance. */
  } arm_dct4_instance_f32;


  /**
   * @brief  Initialization function for the floating-point DCT4/IDCT4.
   * @param[in,out] S          points to an instance of floating-point DCT4/IDCT4 structure.
   * @param[in]     S_RFFT     points to an instance of floating-point RFFT/RIFFT structure.
   * @param[in]     S_CFFT     points to an instance of floating-point CFFT/CIFFT structure.
   * @param[in]     N          length of the DCT4.
   * @param[in]     Nby2       half of the length of the DCT4.
   * @param[in]     normalize  normalizing factor.
   * @return      arm_status function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_ARGUMENT_ERROR if <code>fftLenReal</code> is not a supported transform length.
   */
  arm_status arm_dct4_init_f32(
  arm_dct4_instance_f32 * S,
  arm_rfft_instance_f32 * S_RFFT,
  arm_cfft_radix4_instance_f32 * S_CFFT,
  uint16_t N,
  uint16_t Nby2,
  float32_t normalize);


  /**
   * @brief Processing function for the floating-point DCT4/IDCT4.
   * @param[in]     S              points to an instance of the floating-point DCT4/IDCT4 structure.
   * @param[in]     pState         points to state buffer.
   * @param[in,out] pInlineBuffer  points to the in-place input and output buffer.
   */
  void arm_dct4_f32(
  const arm_dct4_instance_f32 * S,
  float32_t * pState,
  float32_t * pInlineBuffer);


  /**
   * @brief Instance structure for the Q31 DCT4/IDCT4 function.
   */
  typedef struct
  {
    uint16_t N;                          /**< length of the DCT4. */
    uint16_t Nby2;                       /**< half of the length of the DCT4. */
    q31_t normalize;                     /**< normalizing factor. */
    q31_t *pTwiddle;                     /**< points to the twiddle factor table. */
    q31_t *pCosFactor;                   /**< points to the cosFactor table. */
    arm_rfft_instance_q31 *pRfft;        /**< points to the real FFT instance. */
    arm_cfft_radix4_instance_q31 *pCfft; /**< points to the complex FFT instance. */
  } arm_dct4_instance_q31;


  /**
   * @brief  Initialization function for the Q31 DCT4/IDCT4.
   * @param[in,out] S          points to an instance of Q31 DCT4/IDCT4 structure.
   * @param[in]     S_RFFT     points to an instance of Q31 RFFT/RIFFT structure
   * @param[in]     S_CFFT     points to an instance of Q31 CFFT/CIFFT structure
   * @param[in]     N          length of the DCT4.
   * @param[in]     Nby2       half of the length of the DCT4.
   * @param[in]     normalize  normalizing factor.
   * @return      arm_status function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_ARGUMENT_ERROR if <code>N</code> is not a supported transform length.
   */
  arm_status arm_dct4_init_q31(
  arm_dct4_instance_q31 * S,
  arm_rfft_instance_q31 * S_RFFT,
  arm_cfft_radix4_instance_q31 * S_CFFT,
  uint16_t N,
  uint16_t Nby2,
  q31_t normalize);


  /**
   * @brief Processing function for the Q31 DCT4/IDCT4.
   * @param[in]     S              points to an instance of the Q31 DCT4 structure.
   * @param[in]     pState         points to state buffer.
   * @param[in,out] pInlineBuffer  points to the in-place input and output buffer.
   */
  void arm_dct4_q31(
  const arm_dct4_instance_q31 * S,
  q31_t * pState,
  q31_t * pInlineBuffer);


  /**
   * @brief Instance structure for the Q15 DCT4/IDCT4 function.
   */
  typedef struct
  {
    uint16_t N;                          /**< length of the DCT4. */
    uint16_t Nby2;                       /**< half of the length of the DCT4. */
    q15_t normalize;                     /**< normalizing factor. */
    q15_t *pTwiddle;                     /**< points to the twiddle factor table. */
    q15_t *pCosFactor;                   /**< points to the cosFactor table. */
    arm_rfft_instance_q15 *pRfft;        /**< points to the real FFT instance. */
    arm_cfft_radix4_instance_q15 *pCfft; /**< points to the complex FFT instance. */
  } arm_dct4_instance_q15;


  /**
   * @brief  Initialization function for the Q15 DCT4/IDCT4.
   * @param[in,out] S          points to an instance of Q15 DCT4/IDCT4 structure.
   * @param[in]     S_RFFT     points to an instance of Q15 RFFT/RIFFT structure.
   * @param[in]     S_CFFT     points to an instance of Q15 CFFT/CIFFT structure.
   * @param[in]     N          length of the DCT4.
   * @param[in]     Nby2       half of the length of the DCT4.
   * @param[in]     normalize  normalizing factor.
   * @return      arm_status function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_ARGUMENT_ERROR if <code>N</code> is not a supported transform length.
   */
  arm_status arm_dct4_init_q15(
  arm_dct4_instance_q15 * S,
  arm_rfft_instance_q15 * S_RFFT,
  arm_cfft_radix4_instance_q15 * S_CFFT,
  uint16_t N,
  uint16_t Nby2,
  q15_t normalize);


  /**
   * @brief Processing function for the Q15 DCT4/IDCT4.
   * @param[in]     S              points to an instance of the Q15 DCT4 structure.
   * @param[in]     pState         points to state buffer.
   * @param[in,out] pInlineBuffer  points to the in-place input and output buffer.
   */
  void arm_dct4_q15(
  const arm_dct4_instance_q15 * S,
  q15_t * pState,
  q15_t * pInlineBuffer);


  /**
   * @brief Floating-point vector addition.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_add_f32(
  float32_t * pSrcA,
  float32_t * pSrcB,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Q7 vector addition.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_add_q7(
  q7_t * pSrcA,
  q7_t * pSrcB,
  q7_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Q15 vector addition.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_add_q15(
  q15_t * pSrcA,
  q15_t * pSrcB,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Q31 vector addition.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_add_q31(
  q31_t * pSrcA,
  q31_t * pSrcB,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Floating-point vector subtraction.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_sub_f32(
  float32_t * pSrcA,
  float32_t * pSrcB,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Q7 vector subtraction.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_sub_q7(
  q7_t * pSrcA,
  q7_t * pSrcB,
  q7_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Q15 vector subtraction.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_sub_q15(
  q15_t * pSrcA,
  q15_t * pSrcB,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Q31 vector subtraction.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_sub_q31(
  q31_t * pSrcA,
  q31_t * pSrcB,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Multiplies a floating-point vector by a scalar.
   * @param[in]  pSrc       points to the input vector
   * @param[in]  scale      scale factor to be applied
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_scale_f32(
  float32_t * pSrc,
  float32_t scale,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Multiplies a Q7 vector by a scalar.
   * @param[in]  pSrc        points to the input vector
   * @param[in]  scaleFract  fractional portion of the scale value
   * @param[in]  shift       number of bits to shift the result by
   * @param[out] pDst        points to the output vector
   * @param[in]  blockSize   number of samples in the vector
   */
  void arm_scale_q7(
  q7_t * pSrc,
  q7_t scaleFract,
  int8_t shift,
  q7_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Multiplies a Q15 vector by a scalar.
   * @param[in]  pSrc        points to the input vector
   * @param[in]  scaleFract  fractional portion of the scale value
   * @param[in]  shift       number of bits to shift the result by
   * @param[out] pDst        points to the output vector
   * @param[in]  blockSize   number of samples in the vector
   */
  void arm_scale_q15(
  q15_t * pSrc,
  q15_t scaleFract,
  int8_t shift,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Multiplies a Q31 vector by a scalar.
   * @param[in]  pSrc        points to the input vector
   * @param[in]  scaleFract  fractional portion of the scale value
   * @param[in]  shift       number of bits to shift the result by
   * @param[out] pDst        points to the output vector
   * @param[in]  blockSize   number of samples in the vector
   */
  void arm_scale_q31(
  q31_t * pSrc,
  q31_t scaleFract,
  int8_t shift,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Q7 vector absolute value.
   * @param[in]  pSrc       points to the input buffer
   * @param[out] pDst       points to the output buffer
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_abs_q7(
  q7_t * pSrc,
  q7_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Floating-point vector absolute value.
   * @param[in]  pSrc       points to the input buffer
   * @param[out] pDst       points to the output buffer
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_abs_f32(
  float32_t * pSrc,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Q15 vector absolute value.
   * @param[in]  pSrc       points to the input buffer
   * @param[out] pDst       points to the output buffer
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_abs_q15(
  q15_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Q31 vector absolute value.
   * @param[in]  pSrc       points to the input buffer
   * @param[out] pDst       points to the output buffer
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_abs_q31(
  q31_t * pSrc,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Dot product of floating-point vectors.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[in]  blockSize  number of samples in each vector
   * @param[out] result     output result returned here
   */
  void arm_dot_prod_f32(
  float32_t * pSrcA,
  float32_t * pSrcB,
  uint32_t blockSize,
  float32_t * result);


  /**
   * @brief Dot product of Q7 vectors.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[in]  blockSize  number of samples in each vector
   * @param[out] result     output result returned here
   */
  void arm_dot_prod_q7(
  q7_t * pSrcA,
  q7_t * pSrcB,
  uint32_t blockSize,
  q31_t * result);


  /**
   * @brief Dot product of Q15 vectors.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[in]  blockSize  number of samples in each vector
   * @param[out] result     output result returned here
   */
  void arm_dot_prod_q15(
  q15_t * pSrcA,
  q15_t * pSrcB,
  uint32_t blockSize,
  q63_t * result);


  /**
   * @brief Dot product of Q31 vectors.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[in]  blockSize  number of samples in each vector
   * @param[out] result     output result returned here
   */
  void arm_dot_prod_q31(
  q31_t * pSrcA,
  q31_t * pSrcB,
  uint32_t blockSize,
  q63_t * result);


  /**
   * @brief  Shifts the elements of a Q7 vector a specified number of bits.
   * @param[in]  pSrc       points to the input vector
   * @param[in]  shiftBits  number of bits to shift.  A positive value shifts left; a negative value shifts right.
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_shift_q7(
  q7_t * pSrc,
  int8_t shiftBits,
  q7_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Shifts the elements of a Q15 vector a specified number of bits.
   * @param[in]  pSrc       points to the input vector
   * @param[in]  shiftBits  number of bits to shift.  A positive value shifts left; a negative value shifts right.
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_shift_q15(
  q15_t * pSrc,
  int8_t shiftBits,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Shifts the elements of a Q31 vector a specified number of bits.
   * @param[in]  pSrc       points to the input vector
   * @param[in]  shiftBits  number of bits to shift.  A positive value shifts left; a negative value shifts right.
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_shift_q31(
  q31_t * pSrc,
  int8_t shiftBits,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Adds a constant offset to a floating-point vector.
   * @param[in]  pSrc       points to the input vector
   * @param[in]  offset     is the offset to be added
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_offset_f32(
  float32_t * pSrc,
  float32_t offset,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Adds a constant offset to a Q7 vector.
   * @param[in]  pSrc       points to the input vector
   * @param[in]  offset     is the offset to be added
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_offset_q7(
  q7_t * pSrc,
  q7_t offset,
  q7_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Adds a constant offset to a Q15 vector.
   * @param[in]  pSrc       points to the input vector
   * @param[in]  offset     is the offset to be added
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_offset_q15(
  q15_t * pSrc,
  q15_t offset,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Adds a constant offset to a Q31 vector.
   * @param[in]  pSrc       points to the input vector
   * @param[in]  offset     is the offset to be added
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_offset_q31(
  q31_t * pSrc,
  q31_t offset,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Negates the elements of a floating-point vector.
   * @param[in]  pSrc       points to the input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_negate_f32(
  float32_t * pSrc,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Negates the elements of a Q7 vector.
   * @param[in]  pSrc       points to the input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_negate_q7(
  q7_t * pSrc,
  q7_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Negates the elements of a Q15 vector.
   * @param[in]  pSrc       points to the input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_negate_q15(
  q15_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Negates the elements of a Q31 vector.
   * @param[in]  pSrc       points to the input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_negate_q31(
  q31_t * pSrc,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Copies the elements of a floating-point vector.
   * @param[in]  pSrc       input pointer
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_copy_f32(
  float32_t * pSrc,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Copies the elements of a Q7 vector.
   * @param[in]  pSrc       input pointer
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_copy_q7(
  q7_t * pSrc,
  q7_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Copies the elements of a Q15 vector.
   * @param[in]  pSrc       input pointer
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_copy_q15(
  q15_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Copies the elements of a Q31 vector.
   * @param[in]  pSrc       input pointer
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_copy_q31(
  q31_t * pSrc,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Fills a constant value into a floating-point vector.
   * @param[in]  value      input value to be filled
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_fill_f32(
  float32_t value,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Fills a constant value into a Q7 vector.
   * @param[in]  value      input value to be filled
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_fill_q7(
  q7_t value,
  q7_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Fills a constant value into a Q15 vector.
   * @param[in]  value      input value to be filled
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_fill_q15(
  q15_t value,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Fills a constant value into a Q31 vector.
   * @param[in]  value      input value to be filled
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_fill_q31(
  q31_t value,
  q31_t * pDst,
  uint32_t blockSize);


/**
 * @brief Convolution of floating-point sequences.
 * @param[in]  pSrcA    points to the first input sequence.
 * @param[in]  srcALen  length of the first input sequence.
 * @param[in]  pSrcB    points to the second input sequence.
 * @param[in]  srcBLen  length of the second input sequence.
 * @param[out] pDst     points to the location where the output result is written.  Length srcALen+srcBLen-1.
 */
  void arm_conv_f32(
  float32_t * pSrcA,
  uint32_t srcALen,
  float32_t * pSrcB,
  uint32_t srcBLen,
  float32_t * pDst);


  /**
   * @brief Convolution of Q15 sequences.
   * @param[in]  pSrcA      points to the first input sequence.
   * @param[in]  srcALen    length of the first input sequence.
   * @param[in]  pSrcB      points to the second input sequence.
   * @param[in]  srcBLen    length of the second input sequence.
   * @param[out] pDst       points to the block of output data  Length srcALen+srcBLen-1.
   * @param[in]  pScratch1  points to scratch buffer of size max(srcALen, srcBLen) + 2*min(srcALen, srcBLen) - 2.
   * @param[in]  pScratch2  points to scratch buffer of size min(srcALen, srcBLen).
   */
  void arm_conv_opt_q15(
  q15_t * pSrcA,
  uint32_t srcALen,
  q15_t * pSrcB,
  uint32_t srcBLen,
  q15_t * pDst,
  q15_t * pScratch1,
  q15_t * pScratch2);


/**
 * @brief Convolution of Q15 sequences.
 * @param[in]  pSrcA    points to the first input sequence.
 * @param[in]  srcALen  length of the first input sequence.
 * @param[in]  pSrcB    points to the second input sequence.
 * @param[in]  srcBLen  length of the second input sequence.
 * @param[out] pDst     points to the location where the output result is written.  Length srcALen+srcBLen-1.
 */
  void arm_conv_q15(
  q15_t * pSrcA,
  uint32_t srcALen,
  q15_t * pSrcB,
  uint32_t srcBLen,
  q15_t * pDst);


  /**
   * @brief Convolution of Q15 sequences (fast version) for Cortex-M3 and Cortex-M4
   * @param[in]  pSrcA    points to the first input sequence.
   * @param[in]  srcALen  length of the first input sequence.
   * @param[in]  pSrcB    points to the second input sequence.
   * @param[in]  srcBLen  length of the second input sequence.
   * @param[out] pDst     points to the block of output data  Length srcALen+srcBLen-1.
   */
  void arm_conv_fast_q15(
          q15_t * pSrcA,
          uint32_t srcALen,
          q15_t * pSrcB,
          uint32_t srcBLen,
          q15_t * pDst);


  /**
   * @brief Convolution of Q15 sequences (fast version) for Cortex-M3 and Cortex-M4
   * @param[in]  pSrcA      points to the first input sequence.
   * @param[in]  srcALen    length of the first input sequence.
   * @param[in]  pSrcB      points to the second input sequence.
   * @param[in]  srcBLen    length of the second input sequence.
   * @param[out] pDst       points to the block of output data  Length srcALen+srcBLen-1.
   * @param[in]  pScratch1  points to scratch buffer of size max(srcALen, srcBLen) + 2*min(srcALen, srcBLen) - 2.
   * @param[in]  pScratch2  points to scratch buffer of size min(srcALen, srcBLen).
   */
  void arm_conv_fast_opt_q15(
  q15_t * pSrcA,
  uint32_t srcALen,
  q15_t * pSrcB,
  uint32_t srcBLen,
  q15_t * pDst,
  q15_t * pScratch1,
  q15_t * pScratch2);


  /**
   * @brief Convolution of Q31 sequences.
   * @param[in]  pSrcA    points to the first input sequence.
   * @param[in]  srcALen  length of the first input sequence.
   * @param[in]  pSrcB    points to the second input sequence.
   * @param[in]  srcBLen  length of the second input sequence.
   * @param[out] pDst     points to the block of output data  Length srcALen+srcBLen-1.
   */
  void arm_conv_q31(
  q31_t * pSrcA,
  uint32_t srcALen,
  q31_t * pSrcB,
  uint32_t srcBLen,
  q31_t * pDst);


  /**
   * @brief Convolution of Q31 sequences (fast version) for Cortex-M3 and Cortex-M4
   * @param[in]  pSrcA    points to the first input sequence.
   * @param[in]  srcALen  length of the first input sequence.
   * @param[in]  pSrcB    points to the second input sequence.
   * @param[in]  srcBLen  length of the second input sequence.
   * @param[out] pDst     points to the block of output data  Length srcALen+srcBLen-1.
   */
  void arm_conv_fast_q31(
  q31_t * pSrcA,
  uint32_t srcALen,
  q31_t * pSrcB,
  uint32_t srcBLen,
  q31_t * pDst);


    /**
   * @brief Convolution of Q7 sequences.
   * @param[in]  pSrcA      points to the first input sequence.
   * @param[in]  srcALen    length of the first input sequence.
   * @param[in]  pSrcB      points to the second input sequence.
   * @param[in]  srcBLen    length of the second input sequence.
   * @param[out] pDst       points to the block of output data  Length srcALen+srcBLen-1.
   * @param[in]  pScratch1  points to scratch buffer(of type q15_t) of size max(srcALen, srcBLen) + 2*min(srcALen, srcBLen) - 2.
   * @param[in]  pScratch2  points to scratch buffer (of type q15_t) of size min(srcALen, srcBLen).
   */
  void arm_conv_opt_q7(
  q7_t * pSrcA,
  uint32_t srcALen,
  q7_t * pSrcB,
  uint32_t srcBLen,
  q7_t * pDst,
  q15_t * pScratch1,
  q15_t * pScratch2);


  /**
   * @brief Convolution of Q7 sequences.
   * @param[in]  pSrcA    points to the first input sequence.
   * @param[in]  srcALen  length of the first input sequence.
   * @param[in]  pSrcB    points to the second input sequence.
   * @param[in]  srcBLen  length of the second input sequence.
   * @param[out] pDst     points to the block of output data  Length srcALen+srcBLen-1.
   */
  void arm_conv_q7(
  q7_t * pSrcA,
  uint32_t srcALen,
  q7_t * pSrcB,
  uint32_t srcBLen,
  q7_t * pDst);


  /**
   * @brief Partial convolution of floating-point sequences.
   * @param[in]  pSrcA       points to the first input sequence.
   * @param[in]  srcALen     length of the first input sequence.
   * @param[in]  pSrcB       points to the second input sequence.
   * @param[in]  srcBLen     length of the second input sequence.
   * @param[out] pDst        points to the block of output data
   * @param[in]  firstIndex  is the first output sample to start with.
   * @param[in]  numPoints   is the number of output points to be computed.
   * @return  Returns either ARM_MATH_SUCCESS if the function completed correctly or ARM_MATH_ARGUMENT_ERROR if the requested subset is not in the range [0 srcALen+srcBLen-2].
   */
  arm_status arm_conv_partial_f32(
  float32_t * pSrcA,
  uint32_t srcALen,
  float32_t * pSrcB,
  uint32_t srcBLen,
  float32_t * pDst,
  uint32_t firstIndex,
  uint32_t numPoints);


  /**
   * @brief Partial convolution of Q15 sequences.
   * @param[in]  pSrcA       points to the first input sequence.
   * @param[in]  srcALen     length of the first input sequence.
   * @param[in]  pSrcB       points to the second input sequence.
   * @param[in]  srcBLen     length of the second input sequence.
   * @param[out] pDst        points to the block of output data
   * @param[in]  firstIndex  is the first output sample to start with.
   * @param[in]  numPoints   is the number of output points to be computed.
   * @param[in]  pScratch1   points to scratch buffer of size max(srcALen, srcBLen) + 2*min(srcALen, srcBLen) - 2.
   * @param[in]  pScratch2   points to scratch buffer of size min(srcALen, srcBLen).
   * @return  Returns either ARM_MATH_SUCCESS if the function completed correctly or ARM_MATH_ARGUMENT_ERROR if the requested subset is not in the range [0 srcALen+srcBLen-2].
   */
  arm_status arm_conv_partial_opt_q15(
  q15_t * pSrcA,
  uint32_t srcALen,
  q15_t * pSrcB,
  uint32_t srcBLen,
  q15_t * pDst,
  uint32_t firstIndex,
  uint32_t numPoints,
  q15_t * pScratch1,
  q15_t * pScratch2);


  /**
   * @brief Partial convolution of Q15 sequences.
   * @param[in]  pSrcA       points to the first input sequence.
   * @param[in]  srcALen     length of the first input sequence.
   * @param[in]  pSrcB       points to the second input sequence.
   * @param[in]  srcBLen     length of the second input sequence.
   * @param[out] pDst        points to the block of output data
   * @param[in]  firstIndex  is the first output sample to start with.
   * @param[in]  numPoints   is the number of output points to be computed.
   * @return  Returns either ARM_MATH_SUCCESS if the function completed correctly or ARM_MATH_ARGUMENT_ERROR if the requested subset is not in the range [0 srcALen+srcBLen-2].
   */
  arm_status arm_conv_partial_q15(
  q15_t * pSrcA,
  uint32_t srcALen,
  q15_t * pSrcB,
  uint32_t srcBLen,
  q15_t * pDst,
  uint32_t firstIndex,
  uint32_t numPoints);


  /**
   * @brief Partial convolution of Q15 sequences (fast version) for Cortex-M3 and Cortex-M4
   * @param[in]  pSrcA       points to the first input sequence.
   * @param[in]  srcALen     length of the first input sequence.
   * @param[in]  pSrcB       points to the second input sequence.
   * @param[in]  srcBLen     length of the second input sequence.
   * @param[out] pDst        points to the block of output data
   * @param[in]  firstIndex  is the first output sample to start with.
   * @param[in]  numPoints   is the number of output points to be computed.
   * @return  Returns either ARM_MATH_SUCCESS if the function completed correctly or ARM_MATH_ARGUMENT_ERROR if the requested subset is not in the range [0 srcALen+srcBLen-2].
   */
  arm_status arm_conv_partial_fast_q15(
  q15_t * pSrcA,
  uint32_t srcALen,
  q15_t * pSrcB,
  uint32_t srcBLen,
  q15_t * pDst,
  uint32_t firstIndex,
  uint32_t numPoints);


  /**
   * @brief Partial convolution of Q15 sequences (fast version) for Cortex-M3 and Cortex-M4
   * @param[in]  pSrcA       points to the first input sequence.
   * @param[in]  srcALen     length of the first input sequence.
   * @param[in]  pSrcB       points to the second input sequence.
   * @param[in]  srcBLen     length of the second input sequence.
   * @param[out] pDst        points to the block of output data
   * @param[in]  firstIndex  is the first output sample to start with.
   * @param[in]  numPoints   is the number of output points to be computed.
   * @param[in]  pScratch1   points to scratch buffer of size max(srcALen, srcBLen) + 2*min(srcALen, srcBLen) - 2.
   * @param[in]  pScratch2   points to scratch buffer of size min(srcALen, srcBLen).
   * @return  Returns either ARM_MATH_SUCCESS if the function completed correctly or ARM_MATH_ARGUMENT_ERROR if the requested subset is not in the range [0 srcALen+srcBLen-2].
   */
  arm_status arm_conv_partial_fast_opt_q15(
  q15_t * pSrcA,
  uint32_t srcALen,
  q15_t * pSrcB,
  uint32_t srcBLen,
  q15_t * pDst,
  uint32_t firstIndex,
  uint32_t numPoints,
  q15_t * pScratch1,
  q15_t * pScratch2);


  /**
   * @brief Partial convolution of Q31 sequences.
   * @param[in]  pSrcA       points to the first input sequence.
   * @param[in]  srcALen     length of the first input sequence.
   * @param[in]  pSrcB       points to the second input sequence.
   * @param[in]  srcBLen     length of the second input sequence.
   * @param[out] pDst        points to the block of output data
   * @param[in]  firstIndex  is the first output sample to start with.
   * @param[in]  numPoints   is the number of output points to be computed.
   * @return  Returns either ARM_MATH_SUCCESS if the function completed correctly or ARM_MATH_ARGUMENT_ERROR if the requested subset is not in the range [0 srcALen+srcBLen-2].
   */
  arm_status arm_conv_partial_q31(
  q31_t * pSrcA,
  uint32_t srcALen,
  q31_t * pSrcB,
  uint32_t srcBLen,
  q31_t * pDst,
  uint32_t firstIndex,
  uint32_t numPoints);


  /**
   * @brief Partial convolution of Q31 sequences (fast version) for Cortex-M3 and Cortex-M4
   * @param[in]  pSrcA       points to the first input sequence.
   * @param[in]  srcALen     length of the first input sequence.
   * @param[in]  pSrcB       points to the second input sequence.
   * @param[in]  srcBLen     length of the second input sequence.
   * @param[out] pDst        points to the block of output data
   * @param[in]  firstIndex  is the first output sample to start with.
   * @param[in]  numPoints   is the number of output points to be computed.
   * @return  Returns either ARM_MATH_SUCCESS if the function completed correctly or ARM_MATH_ARGUMENT_ERROR if the requested subset is not in the range [0 srcALen+srcBLen-2].
   */
  arm_status arm_conv_partial_fast_q31(
  q31_t * pSrcA,
  uint32_t srcALen,
  q31_t * pSrcB,
  uint32_t srcBLen,
  q31_t * pDst,
  uint32_t firstIndex,
  uint32_t numPoints);


  /**
   * @brief Partial convolution of Q7 sequences
   * @param[in]  pSrcA       points to the first input sequence.
   * @param[in]  srcALen     length of the first input sequence.
   * @param[in]  pSrcB       points to the second input sequence.
   * @param[in]  srcBLen     length of the second input sequence.
   * @param[out] pDst        points to the block of output data
   * @param[in]  firstIndex  is the first output sample to start with.
   * @param[in]  numPoints   is the number of output points to be computed.
   * @param[in]  pScratch1   points to scratch buffer(of type q15_t) of size max(srcALen, srcBLen) + 2*min(srcALen, srcBLen) - 2.
   * @param[in]  pScratch2   points to scratch buffer (of type q15_t) of size min(srcALen, srcBLen).
   * @return  Returns either ARM_MATH_SUCCESS if the function completed correctly or ARM_MATH_ARGUMENT_ERROR if the requested subset is not in the range [0 srcALen+srcBLen-2].
   */
  arm_status arm_conv_partial_opt_q7(
  q7_t * pSrcA,
  uint32_t srcALen,
  q7_t * pSrcB,
  uint32_t srcBLen,
  q7_t * pDst,
  uint32_t firstIndex,
  uint32_t numPoints,
  q15_t * pScratch1,
  q15_t * pScratch2);


/**
   * @brief Partial convolution of Q7 sequences.
   * @param[in]  pSrcA       points to the first input sequence.
   * @param[in]  srcALen     length of the first input sequence.
   * @param[in]  pSrcB       points to the second input sequence.
   * @param[in]  srcBLen     length of the second input sequence.
   * @param[out] pDst        points to the block of output data
   * @param[in]  firstIndex  is the first output sample to start with.
   * @param[in]  numPoints   is the number of output points to be computed.
   * @return  Returns either ARM_MATH_SUCCESS if the function completed correctly or ARM_MATH_ARGUMENT_ERROR if the requested subset is not in the range [0 srcALen+srcBLen-2].
   */
  arm_status arm_conv_partial_q7(
  q7_t * pSrcA,
  uint32_t srcALen,
  q7_t * pSrcB,
  uint32_t srcBLen,
  q7_t * pDst,
  uint32_t firstIndex,
  uint32_t numPoints);


  /**
   * @brief Instance structure for the Q15 FIR decimator.
   */
  typedef struct
  {
    uint8_t M;                  /**< decimation factor. */
    uint16_t numTaps;           /**< number of coefficients in the filter. */
    q15_t *pCoeffs;             /**< points to the coefficient array. The array is of length numTaps.*/
    q15_t *pState;              /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
  } arm_fir_decimate_instance_q15;

  /**
   * @brief Instance structure for the Q31 FIR decimator.
   */
  typedef struct
  {
    uint8_t M;                  /**< decimation factor. */
    uint16_t numTaps;           /**< number of coefficients in the filter. */
    q31_t *pCoeffs;             /**< points to the coefficient array. The array is of length numTaps.*/
    q31_t *pState;              /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
  } arm_fir_decimate_instance_q31;

  /**
   * @brief Instance structure for the floating-point FIR decimator.
   */
  typedef struct
  {
    uint8_t M;                  /**< decimation factor. */
    uint16_t numTaps;           /**< number of coefficients in the filter. */
    float32_t *pCoeffs;         /**< points to the coefficient array. The array is of length numTaps.*/
    float32_t *pState;          /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
  } arm_fir_decimate_instance_f32;


  /**
   * @brief Processing function for the floating-point FIR decimator.
   * @param[in]  S          points to an instance of the floating-point FIR decimator structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of input samples to process per call.
   */
  void arm_fir_decimate_f32(
  const arm_fir_decimate_instance_f32 * S,
  float32_t * pSrc,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Initialization function for the floating-point FIR decimator.
   * @param[in,out] S          points to an instance of the floating-point FIR decimator structure.
   * @param[in]     numTaps    number of coefficients in the filter.
   * @param[in]     M          decimation factor.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of input samples to process per call.
   * @return    The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_LENGTH_ERROR if
   * <code>blockSize</code> is not a multiple of <code>M</code>.
   */
  arm_status arm_fir_decimate_init_f32(
  arm_fir_decimate_instance_f32 * S,
  uint16_t numTaps,
  uint8_t M,
  float32_t * pCoeffs,
  float32_t * pState,
  uint32_t blockSize);


  /**
   * @brief Processing function for the Q15 FIR decimator.
   * @param[in]  S          points to an instance of the Q15 FIR decimator structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of input samples to process per call.
   */
  void arm_fir_decimate_q15(
  const arm_fir_decimate_instance_q15 * S,
  q15_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Processing function for the Q15 FIR decimator (fast variant) for Cortex-M3 and Cortex-M4.
   * @param[in]  S          points to an instance of the Q15 FIR decimator structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of input samples to process per call.
   */
  void arm_fir_decimate_fast_q15(
  const arm_fir_decimate_instance_q15 * S,
  q15_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q15 FIR decimator.
   * @param[in,out] S          points to an instance of the Q15 FIR decimator structure.
   * @param[in]     numTaps    number of coefficients in the filter.
   * @param[in]     M          decimation factor.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of input samples to process per call.
   * @return    The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_LENGTH_ERROR if
   * <code>blockSize</code> is not a multiple of <code>M</code>.
   */
  arm_status arm_fir_decimate_init_q15(
  arm_fir_decimate_instance_q15 * S,
  uint16_t numTaps,
  uint8_t M,
  q15_t * pCoeffs,
  q15_t * pState,
  uint32_t blockSize);


  /**
   * @brief Processing function for the Q31 FIR decimator.
   * @param[in]  S     points to an instance of the Q31 FIR decimator structure.
   * @param[in]  pSrc  points to the block of input data.
   * @param[out] pDst  points to the block of output data
   * @param[in] blockSize number of input samples to process per call.
   */
  void arm_fir_decimate_q31(
  const arm_fir_decimate_instance_q31 * S,
  q31_t * pSrc,
  q31_t * pDst,
  uint32_t blockSize);

  /**
   * @brief Processing function for the Q31 FIR decimator (fast variant) for Cortex-M3 and Cortex-M4.
   * @param[in]  S          points to an instance of the Q31 FIR decimator structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of input samples to process per call.
   */
  void arm_fir_decimate_fast_q31(
  arm_fir_decimate_instance_q31 * S,
  q31_t * pSrc,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q31 FIR decimator.
   * @param[in,out] S          points to an instance of the Q31 FIR decimator structure.
   * @param[in]     numTaps    number of coefficients in the filter.
   * @param[in]     M          decimation factor.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of input samples to process per call.
   * @return    The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_LENGTH_ERROR if
   * <code>blockSize</code> is not a multiple of <code>M</code>.
   */
  arm_status arm_fir_decimate_init_q31(
  arm_fir_decimate_instance_q31 * S,
  uint16_t numTaps,
  uint8_t M,
  q31_t * pCoeffs,
  q31_t * pState,
  uint32_t blockSize);


  /**
   * @brief Instance structure for the Q15 FIR interpolator.
   */
  typedef struct
  {
    uint8_t L;                      /**< upsample factor. */
    uint16_t phaseLength;           /**< length of each polyphase filter component. */
    q15_t *pCoeffs;                 /**< points to the coefficient array. The array is of length L*phaseLength. */
    q15_t *pState;                  /**< points to the state variable array. The array is of length blockSize+phaseLength-1. */
  } arm_fir_interpolate_instance_q15;

  /**
   * @brief Instance structure for the Q31 FIR interpolator.
   */
  typedef struct
  {
    uint8_t L;                      /**< upsample factor. */
    uint16_t phaseLength;           /**< length of each polyphase filter component. */
    q31_t *pCoeffs;                 /**< points to the coefficient array. The array is of length L*phaseLength. */
    q31_t *pState;                  /**< points to the state variable array. The array is of length blockSize+phaseLength-1. */
  } arm_fir_interpolate_instance_q31;

  /**
   * @brief Instance structure for the floating-point FIR interpolator.
   */
  typedef struct
  {
    uint8_t L;                     /**< upsample factor. */
    uint16_t phaseLength;          /**< length of each polyphase filter component. */
    float32_t *pCoeffs;            /**< points to the coefficient array. The array is of length L*phaseLength. */
    float32_t *pState;             /**< points to the state variable array. The array is of length phaseLength+numTaps-1. */
  } arm_fir_interpolate_instance_f32;


  /**
   * @brief Processing function for the Q15 FIR interpolator.
   * @param[in]  S          points to an instance of the Q15 FIR interpolator structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of input samples to process per call.
   */
  void arm_fir_interpolate_q15(
  const arm_fir_interpolate_instance_q15 * S,
  q15_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q15 FIR interpolator.
   * @param[in,out] S          points to an instance of the Q15 FIR interpolator structure.
   * @param[in]     L          upsample factor.
   * @param[in]     numTaps    number of filter coefficients in the filter.
   * @param[in]     pCoeffs    points to the filter coefficient buffer.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of input samples to process per call.
   * @return        The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_LENGTH_ERROR if
   * the filter length <code>numTaps</code> is not a multiple of the interpolation factor <code>L</code>.
   */
  arm_status arm_fir_interpolate_init_q15(
  arm_fir_interpolate_instance_q15 * S,
  uint8_t L,
  uint16_t numTaps,
  q15_t * pCoeffs,
  q15_t * pState,
  uint32_t blockSize);


  /**
   * @brief Processing function for the Q31 FIR interpolator.
   * @param[in]  S          points to an instance of the Q15 FIR interpolator structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of input samples to process per call.
   */
  void arm_fir_interpolate_q31(
  const arm_fir_interpolate_instance_q31 * S,
  q31_t * pSrc,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q31 FIR interpolator.
   * @param[in,out] S          points to an instance of the Q31 FIR interpolator structure.
   * @param[in]     L          upsample factor.
   * @param[in]     numTaps    number of filter coefficients in the filter.
   * @param[in]     pCoeffs    points to the filter coefficient buffer.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of input samples to process per call.
   * @return        The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_LENGTH_ERROR if
   * the filter length <code>numTaps</code> is not a multiple of the interpolation factor <code>L</code>.
   */
  arm_status arm_fir_interpolate_init_q31(
  arm_fir_interpolate_instance_q31 * S,
  uint8_t L,
  uint16_t numTaps,
  q31_t * pCoeffs,
  q31_t * pState,
  uint32_t blockSize);


  /**
   * @brief Processing function for the floating-point FIR interpolator.
   * @param[in]  S          points to an instance of the floating-point FIR interpolator structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of input samples to process per call.
   */
  void arm_fir_interpolate_f32(
  const arm_fir_interpolate_instance_f32 * S,
  float32_t * pSrc,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Initialization function for the floating-point FIR interpolator.
   * @param[in,out] S          points to an instance of the floating-point FIR interpolator structure.
   * @param[in]     L          upsample factor.
   * @param[in]     numTaps    number of filter coefficients in the filter.
   * @param[in]     pCoeffs    points to the filter coefficient buffer.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     blockSize  number of input samples to process per call.
   * @return        The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_LENGTH_ERROR if
   * the filter length <code>numTaps</code> is not a multiple of the interpolation factor <code>L</code>.
   */
  arm_status arm_fir_interpolate_init_f32(
  arm_fir_interpolate_instance_f32 * S,
  uint8_t L,
  uint16_t numTaps,
  float32_t * pCoeffs,
  float32_t * pState,
  uint32_t blockSize);


  /**
   * @brief Instance structure for the high precision Q31 Biquad cascade filter.
   */
  typedef struct
  {
    uint8_t numStages;       /**< number of 2nd order stages in the filter.  Overall order is 2*numStages. */
    q63_t *pState;           /**< points to the array of state coefficients.  The array is of length 4*numStages. */
    q31_t *pCoeffs;          /**< points to the array of coefficients.  The array is of length 5*numStages. */
    uint8_t postShift;       /**< additional shift, in bits, applied to each output sample. */
  } arm_biquad_cas_df1_32x64_ins_q31;


  /**
   * @param[in]  S          points to an instance of the high precision Q31 Biquad cascade filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_biquad_cas_df1_32x64_q31(
  const arm_biquad_cas_df1_32x64_ins_q31 * S,
  q31_t * pSrc,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @param[in,out] S          points to an instance of the high precision Q31 Biquad cascade filter structure.
   * @param[in]     numStages  number of 2nd order stages in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     postShift  shift to be applied to the output. Varies according to the coefficients format
   */
  void arm_biquad_cas_df1_32x64_init_q31(
  arm_biquad_cas_df1_32x64_ins_q31 * S,
  uint8_t numStages,
  q31_t * pCoeffs,
  q63_t * pState,
  uint8_t postShift);


  /**
   * @brief Instance structure for the floating-point transposed direct form II Biquad cascade filter.
   */
  typedef struct
  {
    uint8_t numStages;         /**< number of 2nd order stages in the filter.  Overall order is 2*numStages. */
    float32_t *pState;         /**< points to the array of state coefficients.  The array is of length 2*numStages. */
    float32_t *pCoeffs;        /**< points to the array of coefficients.  The array is of length 5*numStages. */
  } arm_biquad_cascade_df2T_instance_f32;

  /**
   * @brief Instance structure for the floating-point transposed direct form II Biquad cascade filter.
   */
  typedef struct
  {
    uint8_t numStages;         /**< number of 2nd order stages in the filter.  Overall order is 2*numStages. */
    float32_t *pState;         /**< points to the array of state coefficients.  The array is of length 4*numStages. */
    float32_t *pCoeffs;        /**< points to the array of coefficients.  The array is of length 5*numStages. */
  } arm_biquad_cascade_stereo_df2T_instance_f32;

  /**
   * @brief Instance structure for the floating-point transposed direct form II Biquad cascade filter.
   */
  typedef struct
  {
    uint8_t numStages;         /**< number of 2nd order stages in the filter.  Overall order is 2*numStages. */
    float64_t *pState;         /**< points to the array of state coefficients.  The array is of length 2*numStages. */
    float64_t *pCoeffs;        /**< points to the array of coefficients.  The array is of length 5*numStages. */
  } arm_biquad_cascade_df2T_instance_f64;


  /**
   * @brief Processing function for the floating-point transposed direct form II Biquad cascade filter.
   * @param[in]  S          points to an instance of the filter data structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_biquad_cascade_df2T_f32(
  const arm_biquad_cascade_df2T_instance_f32 * S,
  float32_t * pSrc,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Processing function for the floating-point transposed direct form II Biquad cascade filter. 2 channels
   * @param[in]  S          points to an instance of the filter data structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_biquad_cascade_stereo_df2T_f32(
  const arm_biquad_cascade_stereo_df2T_instance_f32 * S,
  float32_t * pSrc,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Processing function for the floating-point transposed direct form II Biquad cascade filter.
   * @param[in]  S          points to an instance of the filter data structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_biquad_cascade_df2T_f64(
  const arm_biquad_cascade_df2T_instance_f64 * S,
  float64_t * pSrc,
  float64_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Initialization function for the floating-point transposed direct form II Biquad cascade filter.
   * @param[in,out] S          points to an instance of the filter data structure.
   * @param[in]     numStages  number of 2nd order stages in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   */
  void arm_biquad_cascade_df2T_init_f32(
  arm_biquad_cascade_df2T_instance_f32 * S,
  uint8_t numStages,
  float32_t * pCoeffs,
  float32_t * pState);


  /**
   * @brief  Initialization function for the floating-point transposed direct form II Biquad cascade filter.
   * @param[in,out] S          points to an instance of the filter data structure.
   * @param[in]     numStages  number of 2nd order stages in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   */
  void arm_biquad_cascade_stereo_df2T_init_f32(
  arm_biquad_cascade_stereo_df2T_instance_f32 * S,
  uint8_t numStages,
  float32_t * pCoeffs,
  float32_t * pState);


  /**
   * @brief  Initialization function for the floating-point transposed direct form II Biquad cascade filter.
   * @param[in,out] S          points to an instance of the filter data structure.
   * @param[in]     numStages  number of 2nd order stages in the filter.
   * @param[in]     pCoeffs    points to the filter coefficients.
   * @param[in]     pState     points to the state buffer.
   */
  void arm_biquad_cascade_df2T_init_f64(
  arm_biquad_cascade_df2T_instance_f64 * S,
  uint8_t numStages,
  float64_t * pCoeffs,
  float64_t * pState);


  /**
   * @brief Instance structure for the Q15 FIR lattice filter.
   */
  typedef struct
  {
    uint16_t numStages;                  /**< number of filter stages. */
    q15_t *pState;                       /**< points to the state variable array. The array is of length numStages. */
    q15_t *pCoeffs;                      /**< points to the coefficient array. The array is of length numStages. */
  } arm_fir_lattice_instance_q15;

  /**
   * @brief Instance structure for the Q31 FIR lattice filter.
   */
  typedef struct
  {
    uint16_t numStages;                  /**< number of filter stages. */
    q31_t *pState;                       /**< points to the state variable array. The array is of length numStages. */
    q31_t *pCoeffs;                      /**< points to the coefficient array. The array is of length numStages. */
  } arm_fir_lattice_instance_q31;

  /**
   * @brief Instance structure for the floating-point FIR lattice filter.
   */
  typedef struct
  {
    uint16_t numStages;                  /**< number of filter stages. */
    float32_t *pState;                   /**< points to the state variable array. The array is of length numStages. */
    float32_t *pCoeffs;                  /**< points to the coefficient array. The array is of length numStages. */
  } arm_fir_lattice_instance_f32;


  /**
   * @brief Initialization function for the Q15 FIR lattice filter.
   * @param[in] S          points to an instance of the Q15 FIR lattice structure.
   * @param[in] numStages  number of filter stages.
   * @param[in] pCoeffs    points to the coefficient buffer.  The array is of length numStages.
   * @param[in] pState     points to the state buffer.  The array is of length numStages.
   */
  void arm_fir_lattice_init_q15(
  arm_fir_lattice_instance_q15 * S,
  uint16_t numStages,
  q15_t * pCoeffs,
  q15_t * pState);


  /**
   * @brief Processing function for the Q15 FIR lattice filter.
   * @param[in]  S          points to an instance of the Q15 FIR lattice structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_fir_lattice_q15(
  const arm_fir_lattice_instance_q15 * S,
  q15_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Initialization function for the Q31 FIR lattice filter.
   * @param[in] S          points to an instance of the Q31 FIR lattice structure.
   * @param[in] numStages  number of filter stages.
   * @param[in] pCoeffs    points to the coefficient buffer.  The array is of length numStages.
   * @param[in] pState     points to the state buffer.   The array is of length numStages.
   */
  void arm_fir_lattice_init_q31(
  arm_fir_lattice_instance_q31 * S,
  uint16_t numStages,
  q31_t * pCoeffs,
  q31_t * pState);


  /**
   * @brief Processing function for the Q31 FIR lattice filter.
   * @param[in]  S          points to an instance of the Q31 FIR lattice structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_fir_lattice_q31(
  const arm_fir_lattice_instance_q31 * S,
  q31_t * pSrc,
  q31_t * pDst,
  uint32_t blockSize);


/**
 * @brief Initialization function for the floating-point FIR lattice filter.
 * @param[in] S          points to an instance of the floating-point FIR lattice structure.
 * @param[in] numStages  number of filter stages.
 * @param[in] pCoeffs    points to the coefficient buffer.  The array is of length numStages.
 * @param[in] pState     points to the state buffer.  The array is of length numStages.
 */
  void arm_fir_lattice_init_f32(
  arm_fir_lattice_instance_f32 * S,
  uint16_t numStages,
  float32_t * pCoeffs,
  float32_t * pState);


  /**
   * @brief Processing function for the floating-point FIR lattice filter.
   * @param[in]  S          points to an instance of the floating-point FIR lattice structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_fir_lattice_f32(
  const arm_fir_lattice_instance_f32 * S,
  float32_t * pSrc,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Instance structure for the Q15 IIR lattice filter.
   */
  typedef struct
  {
    uint16_t numStages;                  /**< number of stages in the filter. */
    q15_t *pState;                       /**< points to the state variable array. The array is of length numStages+blockSize. */
    q15_t *pkCoeffs;                     /**< points to the reflection coefficient array. The array is of length numStages. */
    q15_t *pvCoeffs;                     /**< points to the ladder coefficient array. The array is of length numStages+1. */
  } arm_iir_lattice_instance_q15;

  /**
   * @brief Instance structure for the Q31 IIR lattice filter.
   */
  typedef struct
  {
    uint16_t numStages;                  /**< number of stages in the filter. */
    q31_t *pState;                       /**< points to the state variable array. The array is of length numStages+blockSize. */
    q31_t *pkCoeffs;                     /**< points to the reflection coefficient array. The array is of length numStages. */
    q31_t *pvCoeffs;                     /**< points to the ladder coefficient array. The array is of length numStages+1. */
  } arm_iir_lattice_instance_q31;

  /**
   * @brief Instance structure for the floating-point IIR lattice filter.
   */
  typedef struct
  {
    uint16_t numStages;                  /**< number of stages in the filter. */
    float32_t *pState;                   /**< points to the state variable array. The array is of length numStages+blockSize. */
    float32_t *pkCoeffs;                 /**< points to the reflection coefficient array. The array is of length numStages. */
    float32_t *pvCoeffs;                 /**< points to the ladder coefficient array. The array is of length numStages+1. */
  } arm_iir_lattice_instance_f32;


  /**
   * @brief Processing function for the floating-point IIR lattice filter.
   * @param[in]  S          points to an instance of the floating-point IIR lattice structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_iir_lattice_f32(
  const arm_iir_lattice_instance_f32 * S,
  float32_t * pSrc,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Initialization function for the floating-point IIR lattice filter.
   * @param[in] S          points to an instance of the floating-point IIR lattice structure.
   * @param[in] numStages  number of stages in the filter.
   * @param[in] pkCoeffs   points to the reflection coefficient buffer.  The array is of length numStages.
   * @param[in] pvCoeffs   points to the ladder coefficient buffer.  The array is of length numStages+1.
   * @param[in] pState     points to the state buffer.  The array is of length numStages+blockSize-1.
   * @param[in] blockSize  number of samples to process.
   */
  void arm_iir_lattice_init_f32(
  arm_iir_lattice_instance_f32 * S,
  uint16_t numStages,
  float32_t * pkCoeffs,
  float32_t * pvCoeffs,
  float32_t * pState,
  uint32_t blockSize);


  /**
   * @brief Processing function for the Q31 IIR lattice filter.
   * @param[in]  S          points to an instance of the Q31 IIR lattice structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_iir_lattice_q31(
  const arm_iir_lattice_instance_q31 * S,
  q31_t * pSrc,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Initialization function for the Q31 IIR lattice filter.
   * @param[in] S          points to an instance of the Q31 IIR lattice structure.
   * @param[in] numStages  number of stages in the filter.
   * @param[in] pkCoeffs   points to the reflection coefficient buffer.  The array is of length numStages.
   * @param[in] pvCoeffs   points to the ladder coefficient buffer.  The array is of length numStages+1.
   * @param[in] pState     points to the state buffer.  The array is of length numStages+blockSize.
   * @param[in] blockSize  number of samples to process.
   */
  void arm_iir_lattice_init_q31(
  arm_iir_lattice_instance_q31 * S,
  uint16_t numStages,
  q31_t * pkCoeffs,
  q31_t * pvCoeffs,
  q31_t * pState,
  uint32_t blockSize);


  /**
   * @brief Processing function for the Q15 IIR lattice filter.
   * @param[in]  S          points to an instance of the Q15 IIR lattice structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[out] pDst       points to the block of output data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_iir_lattice_q15(
  const arm_iir_lattice_instance_q15 * S,
  q15_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize);


/**
 * @brief Initialization function for the Q15 IIR lattice filter.
 * @param[in] S          points to an instance of the fixed-point Q15 IIR lattice structure.
 * @param[in] numStages  number of stages in the filter.
 * @param[in] pkCoeffs   points to reflection coefficient buffer.  The array is of length numStages.
 * @param[in] pvCoeffs   points to ladder coefficient buffer.  The array is of length numStages+1.
 * @param[in] pState     points to state buffer.  The array is of length numStages+blockSize.
 * @param[in] blockSize  number of samples to process per call.
 */
  void arm_iir_lattice_init_q15(
  arm_iir_lattice_instance_q15 * S,
  uint16_t numStages,
  q15_t * pkCoeffs,
  q15_t * pvCoeffs,
  q15_t * pState,
  uint32_t blockSize);


  /**
   * @brief Instance structure for the floating-point LMS filter.
   */
  typedef struct
  {
    uint16_t numTaps;    /**< number of coefficients in the filter. */
    float32_t *pState;   /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    float32_t *pCoeffs;  /**< points to the coefficient array. The array is of length numTaps. */
    float32_t mu;        /**< step size that controls filter coefficient updates. */
  } arm_lms_instance_f32;


  /**
   * @brief Processing function for floating-point LMS filter.
   * @param[in]  S          points to an instance of the floating-point LMS filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[in]  pRef       points to the block of reference data.
   * @param[out] pOut       points to the block of output data.
   * @param[out] pErr       points to the block of error data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_lms_f32(
  const arm_lms_instance_f32 * S,
  float32_t * pSrc,
  float32_t * pRef,
  float32_t * pOut,
  float32_t * pErr,
  uint32_t blockSize);


  /**
   * @brief Initialization function for floating-point LMS filter.
   * @param[in] S          points to an instance of the floating-point LMS filter structure.
   * @param[in] numTaps    number of filter coefficients.
   * @param[in] pCoeffs    points to the coefficient buffer.
   * @param[in] pState     points to state buffer.
   * @param[in] mu         step size that controls filter coefficient updates.
   * @param[in] blockSize  number of samples to process.
   */
  void arm_lms_init_f32(
  arm_lms_instance_f32 * S,
  uint16_t numTaps,
  float32_t * pCoeffs,
  float32_t * pState,
  float32_t mu,
  uint32_t blockSize);


  /**
   * @brief Instance structure for the Q15 LMS filter.
   */
  typedef struct
  {
    uint16_t numTaps;    /**< number of coefficients in the filter. */
    q15_t *pState;       /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    q15_t *pCoeffs;      /**< points to the coefficient array. The array is of length numTaps. */
    q15_t mu;            /**< step size that controls filter coefficient updates. */
    uint32_t postShift;  /**< bit shift applied to coefficients. */
  } arm_lms_instance_q15;


  /**
   * @brief Initialization function for the Q15 LMS filter.
   * @param[in] S          points to an instance of the Q15 LMS filter structure.
   * @param[in] numTaps    number of filter coefficients.
   * @param[in] pCoeffs    points to the coefficient buffer.
   * @param[in] pState     points to the state buffer.
   * @param[in] mu         step size that controls filter coefficient updates.
   * @param[in] blockSize  number of samples to process.
   * @param[in] postShift  bit shift applied to coefficients.
   */
  void arm_lms_init_q15(
  arm_lms_instance_q15 * S,
  uint16_t numTaps,
  q15_t * pCoeffs,
  q15_t * pState,
  q15_t mu,
  uint32_t blockSize,
  uint32_t postShift);


  /**
   * @brief Processing function for Q15 LMS filter.
   * @param[in]  S          points to an instance of the Q15 LMS filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[in]  pRef       points to the block of reference data.
   * @param[out] pOut       points to the block of output data.
   * @param[out] pErr       points to the block of error data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_lms_q15(
  const arm_lms_instance_q15 * S,
  q15_t * pSrc,
  q15_t * pRef,
  q15_t * pOut,
  q15_t * pErr,
  uint32_t blockSize);


  /**
   * @brief Instance structure for the Q31 LMS filter.
   */
  typedef struct
  {
    uint16_t numTaps;    /**< number of coefficients in the filter. */
    q31_t *pState;       /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    q31_t *pCoeffs;      /**< points to the coefficient array. The array is of length numTaps. */
    q31_t mu;            /**< step size that controls filter coefficient updates. */
    uint32_t postShift;  /**< bit shift applied to coefficients. */
  } arm_lms_instance_q31;


  /**
   * @brief Processing function for Q31 LMS filter.
   * @param[in]  S          points to an instance of the Q15 LMS filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[in]  pRef       points to the block of reference data.
   * @param[out] pOut       points to the block of output data.
   * @param[out] pErr       points to the block of error data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_lms_q31(
  const arm_lms_instance_q31 * S,
  q31_t * pSrc,
  q31_t * pRef,
  q31_t * pOut,
  q31_t * pErr,
  uint32_t blockSize);


  /**
   * @brief Initialization function for Q31 LMS filter.
   * @param[in] S          points to an instance of the Q31 LMS filter structure.
   * @param[in] numTaps    number of filter coefficients.
   * @param[in] pCoeffs    points to coefficient buffer.
   * @param[in] pState     points to state buffer.
   * @param[in] mu         step size that controls filter coefficient updates.
   * @param[in] blockSize  number of samples to process.
   * @param[in] postShift  bit shift applied to coefficients.
   */
  void arm_lms_init_q31(
  arm_lms_instance_q31 * S,
  uint16_t numTaps,
  q31_t * pCoeffs,
  q31_t * pState,
  q31_t mu,
  uint32_t blockSize,
  uint32_t postShift);


  /**
   * @brief Instance structure for the floating-point normalized LMS filter.
   */
  typedef struct
  {
    uint16_t numTaps;     /**< number of coefficients in the filter. */
    float32_t *pState;    /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    float32_t *pCoeffs;   /**< points to the coefficient array. The array is of length numTaps. */
    float32_t mu;         /**< step size that control filter coefficient updates. */
    float32_t energy;     /**< saves previous frame energy. */
    float32_t x0;         /**< saves previous input sample. */
  } arm_lms_norm_instance_f32;


  /**
   * @brief Processing function for floating-point normalized LMS filter.
   * @param[in]  S          points to an instance of the floating-point normalized LMS filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[in]  pRef       points to the block of reference data.
   * @param[out] pOut       points to the block of output data.
   * @param[out] pErr       points to the block of error data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_lms_norm_f32(
  arm_lms_norm_instance_f32 * S,
  float32_t * pSrc,
  float32_t * pRef,
  float32_t * pOut,
  float32_t * pErr,
  uint32_t blockSize);


  /**
   * @brief Initialization function for floating-point normalized LMS filter.
   * @param[in] S          points to an instance of the floating-point LMS filter structure.
   * @param[in] numTaps    number of filter coefficients.
   * @param[in] pCoeffs    points to coefficient buffer.
   * @param[in] pState     points to state buffer.
   * @param[in] mu         step size that controls filter coefficient updates.
   * @param[in] blockSize  number of samples to process.
   */
  void arm_lms_norm_init_f32(
  arm_lms_norm_instance_f32 * S,
  uint16_t numTaps,
  float32_t * pCoeffs,
  float32_t * pState,
  float32_t mu,
  uint32_t blockSize);


  /**
   * @brief Instance structure for the Q31 normalized LMS filter.
   */
  typedef struct
  {
    uint16_t numTaps;     /**< number of coefficients in the filter. */
    q31_t *pState;        /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    q31_t *pCoeffs;       /**< points to the coefficient array. The array is of length numTaps. */
    q31_t mu;             /**< step size that controls filter coefficient updates. */
    uint8_t postShift;    /**< bit shift applied to coefficients. */
    q31_t *recipTable;    /**< points to the reciprocal initial value table. */
    q31_t energy;         /**< saves previous frame energy. */
    q31_t x0;             /**< saves previous input sample. */
  } arm_lms_norm_instance_q31;


  /**
   * @brief Processing function for Q31 normalized LMS filter.
   * @param[in]  S          points to an instance of the Q31 normalized LMS filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[in]  pRef       points to the block of reference data.
   * @param[out] pOut       points to the block of output data.
   * @param[out] pErr       points to the block of error data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_lms_norm_q31(
  arm_lms_norm_instance_q31 * S,
  q31_t * pSrc,
  q31_t * pRef,
  q31_t * pOut,
  q31_t * pErr,
  uint32_t blockSize);


  /**
   * @brief Initialization function for Q31 normalized LMS filter.
   * @param[in] S          points to an instance of the Q31 normalized LMS filter structure.
   * @param[in] numTaps    number of filter coefficients.
   * @param[in] pCoeffs    points to coefficient buffer.
   * @param[in] pState     points to state buffer.
   * @param[in] mu         step size that controls filter coefficient updates.
   * @param[in] blockSize  number of samples to process.
   * @param[in] postShift  bit shift applied to coefficients.
   */
  void arm_lms_norm_init_q31(
  arm_lms_norm_instance_q31 * S,
  uint16_t numTaps,
  q31_t * pCoeffs,
  q31_t * pState,
  q31_t mu,
  uint32_t blockSize,
  uint8_t postShift);


  /**
   * @brief Instance structure for the Q15 normalized LMS filter.
   */
  typedef struct
  {
    uint16_t numTaps;     /**< Number of coefficients in the filter. */
    q15_t *pState;        /**< points to the state variable array. The array is of length numTaps+blockSize-1. */
    q15_t *pCoeffs;       /**< points to the coefficient array. The array is of length numTaps. */
    q15_t mu;             /**< step size that controls filter coefficient updates. */
    uint8_t postShift;    /**< bit shift applied to coefficients. */
    q15_t *recipTable;    /**< Points to the reciprocal initial value table. */
    q15_t energy;         /**< saves previous frame energy. */
    q15_t x0;             /**< saves previous input sample. */
  } arm_lms_norm_instance_q15;


  /**
   * @brief Processing function for Q15 normalized LMS filter.
   * @param[in]  S          points to an instance of the Q15 normalized LMS filter structure.
   * @param[in]  pSrc       points to the block of input data.
   * @param[in]  pRef       points to the block of reference data.
   * @param[out] pOut       points to the block of output data.
   * @param[out] pErr       points to the block of error data.
   * @param[in]  blockSize  number of samples to process.
   */
  void arm_lms_norm_q15(
  arm_lms_norm_instance_q15 * S,
  q15_t * pSrc,
  q15_t * pRef,
  q15_t * pOut,
  q15_t * pErr,
  uint32_t blockSize);


  /**
   * @brief Initialization function for Q15 normalized LMS filter.
   * @param[in] S          points to an instance of the Q15 normalized LMS filter structure.
   * @param[in] numTaps    number of filter coefficients.
   * @param[in] pCoeffs    points to coefficient buffer.
   * @param[in] pState     points to state buffer.
   * @param[in] mu         step size that controls filter coefficient updates.
   * @param[in] blockSize  number of samples to process.
   * @param[in] postShift  bit shift applied to coefficients.
   */
  void arm_lms_norm_init_q15(
  arm_lms_norm_instance_q15 * S,
  uint16_t numTaps,
  q15_t * pCoeffs,
  q15_t * pState,
  q15_t mu,
  uint32_t blockSize,
  uint8_t postShift);


  /**
   * @brief Correlation of floating-point sequences.
   * @param[in]  pSrcA    points to the first input sequence.
   * @param[in]  srcALen  length of the first input sequence.
   * @param[in]  pSrcB    points to the second input sequence.
   * @param[in]  srcBLen  length of the second input sequence.
   * @param[out] pDst     points to the block of output data  Length 2 * max(srcALen, srcBLen) - 1.
   */
  void arm_correlate_f32(
  float32_t * pSrcA,
  uint32_t srcALen,
  float32_t * pSrcB,
  uint32_t srcBLen,
  float32_t * pDst);


   /**
   * @brief Correlation of Q15 sequences
   * @param[in]  pSrcA     points to the first input sequence.
   * @param[in]  srcALen   length of the first input sequence.
   * @param[in]  pSrcB     points to the second input sequence.
   * @param[in]  srcBLen   length of the second input sequence.
   * @param[out] pDst      points to the block of output data  Length 2 * max(srcALen, srcBLen) - 1.
   * @param[in]  pScratch  points to scratch buffer of size max(srcALen, srcBLen) + 2*min(srcALen, srcBLen) - 2.
   */
  void arm_correlate_opt_q15(
  q15_t * pSrcA,
  uint32_t srcALen,
  q15_t * pSrcB,
  uint32_t srcBLen,
  q15_t * pDst,
  q15_t * pScratch);


  /**
   * @brief Correlation of Q15 sequences.
   * @param[in]  pSrcA    points to the first input sequence.
   * @param[in]  srcALen  length of the first input sequence.
   * @param[in]  pSrcB    points to the second input sequence.
   * @param[in]  srcBLen  length of the second input sequence.
   * @param[out] pDst     points to the block of output data  Length 2 * max(srcALen, srcBLen) - 1.
   */

  void arm_correlate_q15(
  q15_t * pSrcA,
  uint32_t srcALen,
  q15_t * pSrcB,
  uint32_t srcBLen,
  q15_t * pDst);


  /**
   * @brief Correlation of Q15 sequences (fast version) for Cortex-M3 and Cortex-M4.
   * @param[in]  pSrcA    points to the first input sequence.
   * @param[in]  srcALen  length of the first input sequence.
   * @param[in]  pSrcB    points to the second input sequence.
   * @param[in]  srcBLen  length of the second input sequence.
   * @param[out] pDst     points to the block of output data  Length 2 * max(srcALen, srcBLen) - 1.
   */

  void arm_correlate_fast_q15(
  q15_t * pSrcA,
  uint32_t srcALen,
  q15_t * pSrcB,
  uint32_t srcBLen,
  q15_t * pDst);


  /**
   * @brief Correlation of Q15 sequences (fast version) for Cortex-M3 and Cortex-M4.
   * @param[in]  pSrcA     points to the first input sequence.
   * @param[in]  srcALen   length of the first input sequence.
   * @param[in]  pSrcB     points to the second input sequence.
   * @param[in]  srcBLen   length of the second input sequence.
   * @param[out] pDst      points to the block of output data  Length 2 * max(srcALen, srcBLen) - 1.
   * @param[in]  pScratch  points to scratch buffer of size max(srcALen, srcBLen) + 2*min(srcALen, srcBLen) - 2.
   */
  void arm_correlate_fast_opt_q15(
  q15_t * pSrcA,
  uint32_t srcALen,
  q15_t * pSrcB,
  uint32_t srcBLen,
  q15_t * pDst,
  q15_t * pScratch);


  /**
   * @brief Correlation of Q31 sequences.
   * @param[in]  pSrcA    points to the first input sequence.
   * @param[in]  srcALen  length of the first input sequence.
   * @param[in]  pSrcB    points to the second input sequence.
   * @param[in]  srcBLen  length of the second input sequence.
   * @param[out] pDst     points to the block of output data  Length 2 * max(srcALen, srcBLen) - 1.
   */
  void arm_correlate_q31(
  q31_t * pSrcA,
  uint32_t srcALen,
  q31_t * pSrcB,
  uint32_t srcBLen,
  q31_t * pDst);


  /**
   * @brief Correlation of Q31 sequences (fast version) for Cortex-M3 and Cortex-M4
   * @param[in]  pSrcA    points to the first input sequence.
   * @param[in]  srcALen  length of the first input sequence.
   * @param[in]  pSrcB    points to the second input sequence.
   * @param[in]  srcBLen  length of the second input sequence.
   * @param[out] pDst     points to the block of output data  Length 2 * max(srcALen, srcBLen) - 1.
   */
  void arm_correlate_fast_q31(
  q31_t * pSrcA,
  uint32_t srcALen,
  q31_t * pSrcB,
  uint32_t srcBLen,
  q31_t * pDst);


 /**
   * @brief Correlation of Q7 sequences.
   * @param[in]  pSrcA      points to the first input sequence.
   * @param[in]  srcALen    length of the first input sequence.
   * @param[in]  pSrcB      points to the second input sequence.
   * @param[in]  srcBLen    length of the second input sequence.
   * @param[out] pDst       points to the block of output data  Length 2 * max(srcALen, srcBLen) - 1.
   * @param[in]  pScratch1  points to scratch buffer(of type q15_t) of size max(srcALen, srcBLen) + 2*min(srcALen, srcBLen) - 2.
   * @param[in]  pScratch2  points to scratch buffer (of type q15_t) of size min(srcALen, srcBLen).
   */
  void arm_correlate_opt_q7(
  q7_t * pSrcA,
  uint32_t srcALen,
  q7_t * pSrcB,
  uint32_t srcBLen,
  q7_t * pDst,
  q15_t * pScratch1,
  q15_t * pScratch2);


  /**
   * @brief Correlation of Q7 sequences.
   * @param[in]  pSrcA    points to the first input sequence.
   * @param[in]  srcALen  length of the first input sequence.
   * @param[in]  pSrcB    points to the second input sequence.
   * @param[in]  srcBLen  length of the second input sequence.
   * @param[out] pDst     points to the block of output data  Length 2 * max(srcALen, srcBLen) - 1.
   */
  void arm_correlate_q7(
  q7_t * pSrcA,
  uint32_t srcALen,
  q7_t * pSrcB,
  uint32_t srcBLen,
  q7_t * pDst);


  /**
   * @brief Instance structure for the floating-point sparse FIR filter.
   */
  typedef struct
  {
    uint16_t numTaps;             /**< number of coefficients in the filter. */
    uint16_t stateIndex;          /**< state buffer index.  Points to the oldest sample in the state buffer. */
    float32_t *pState;            /**< points to the state buffer array. The array is of length maxDelay+blockSize-1. */
    float32_t *pCoeffs;           /**< points to the coefficient array. The array is of length numTaps.*/
    uint16_t maxDelay;            /**< maximum offset specified by the pTapDelay array. */
    int32_t *pTapDelay;           /**< points to the array of delay values.  The array is of length numTaps. */
  } arm_fir_sparse_instance_f32;

  /**
   * @brief Instance structure for the Q31 sparse FIR filter.
   */
  typedef struct
  {
    uint16_t numTaps;             /**< number of coefficients in the filter. */
    uint16_t stateIndex;          /**< state buffer index.  Points to the oldest sample in the state buffer. */
    q31_t *pState;                /**< points to the state buffer array. The array is of length maxDelay+blockSize-1. */
    q31_t *pCoeffs;               /**< points to the coefficient array. The array is of length numTaps.*/
    uint16_t maxDelay;            /**< maximum offset specified by the pTapDelay array. */
    int32_t *pTapDelay;           /**< points to the array of delay values.  The array is of length numTaps. */
  } arm_fir_sparse_instance_q31;

  /**
   * @brief Instance structure for the Q15 sparse FIR filter.
   */
  typedef struct
  {
    uint16_t numTaps;             /**< number of coefficients in the filter. */
    uint16_t stateIndex;          /**< state buffer index.  Points to the oldest sample in the state buffer. */
    q15_t *pState;                /**< points to the state buffer array. The array is of length maxDelay+blockSize-1. */
    q15_t *pCoeffs;               /**< points to the coefficient array. The array is of length numTaps.*/
    uint16_t maxDelay;            /**< maximum offset specified by the pTapDelay array. */
    int32_t *pTapDelay;           /**< points to the array of delay values.  The array is of length numTaps. */
  } arm_fir_sparse_instance_q15;

  /**
   * @brief Instance structure for the Q7 sparse FIR filter.
   */
  typedef struct
  {
    uint16_t numTaps;             /**< number of coefficients in the filter. */
    uint16_t stateIndex;          /**< state buffer index.  Points to the oldest sample in the state buffer. */
    q7_t *pState;                 /**< points to the state buffer array. The array is of length maxDelay+blockSize-1. */
    q7_t *pCoeffs;                /**< points to the coefficient array. The array is of length numTaps.*/
    uint16_t maxDelay;            /**< maximum offset specified by the pTapDelay array. */
    int32_t *pTapDelay;           /**< points to the array of delay values.  The array is of length numTaps. */
  } arm_fir_sparse_instance_q7;


  /**
   * @brief Processing function for the floating-point sparse FIR filter.
   * @param[in]  S           points to an instance of the floating-point sparse FIR structure.
   * @param[in]  pSrc        points to the block of input data.
   * @param[out] pDst        points to the block of output data
   * @param[in]  pScratchIn  points to a temporary buffer of size blockSize.
   * @param[in]  blockSize   number of input samples to process per call.
   */
  void arm_fir_sparse_f32(
  arm_fir_sparse_instance_f32 * S,
  float32_t * pSrc,
  float32_t * pDst,
  float32_t * pScratchIn,
  uint32_t blockSize);


  /**
   * @brief  Initialization function for the floating-point sparse FIR filter.
   * @param[in,out] S          points to an instance of the floating-point sparse FIR structure.
   * @param[in]     numTaps    number of nonzero coefficients in the filter.
   * @param[in]     pCoeffs    points to the array of filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     pTapDelay  points to the array of offset times.
   * @param[in]     maxDelay   maximum offset time supported.
   * @param[in]     blockSize  number of samples that will be processed per block.
   */
  void arm_fir_sparse_init_f32(
  arm_fir_sparse_instance_f32 * S,
  uint16_t numTaps,
  float32_t * pCoeffs,
  float32_t * pState,
  int32_t * pTapDelay,
  uint16_t maxDelay,
  uint32_t blockSize);


  /**
   * @brief Processing function for the Q31 sparse FIR filter.
   * @param[in]  S           points to an instance of the Q31 sparse FIR structure.
   * @param[in]  pSrc        points to the block of input data.
   * @param[out] pDst        points to the block of output data
   * @param[in]  pScratchIn  points to a temporary buffer of size blockSize.
   * @param[in]  blockSize   number of input samples to process per call.
   */
  void arm_fir_sparse_q31(
  arm_fir_sparse_instance_q31 * S,
  q31_t * pSrc,
  q31_t * pDst,
  q31_t * pScratchIn,
  uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q31 sparse FIR filter.
   * @param[in,out] S          points to an instance of the Q31 sparse FIR structure.
   * @param[in]     numTaps    number of nonzero coefficients in the filter.
   * @param[in]     pCoeffs    points to the array of filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     pTapDelay  points to the array of offset times.
   * @param[in]     maxDelay   maximum offset time supported.
   * @param[in]     blockSize  number of samples that will be processed per block.
   */
  void arm_fir_sparse_init_q31(
  arm_fir_sparse_instance_q31 * S,
  uint16_t numTaps,
  q31_t * pCoeffs,
  q31_t * pState,
  int32_t * pTapDelay,
  uint16_t maxDelay,
  uint32_t blockSize);


  /**
   * @brief Processing function for the Q15 sparse FIR filter.
   * @param[in]  S            points to an instance of the Q15 sparse FIR structure.
   * @param[in]  pSrc         points to the block of input data.
   * @param[out] pDst         points to the block of output data
   * @param[in]  pScratchIn   points to a temporary buffer of size blockSize.
   * @param[in]  pScratchOut  points to a temporary buffer of size blockSize.
   * @param[in]  blockSize    number of input samples to process per call.
   */
  void arm_fir_sparse_q15(
  arm_fir_sparse_instance_q15 * S,
  q15_t * pSrc,
  q15_t * pDst,
  q15_t * pScratchIn,
  q31_t * pScratchOut,
  uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q15 sparse FIR filter.
   * @param[in,out] S          points to an instance of the Q15 sparse FIR structure.
   * @param[in]     numTaps    number of nonzero coefficients in the filter.
   * @param[in]     pCoeffs    points to the array of filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     pTapDelay  points to the array of offset times.
   * @param[in]     maxDelay   maximum offset time supported.
   * @param[in]     blockSize  number of samples that will be processed per block.
   */
  void arm_fir_sparse_init_q15(
  arm_fir_sparse_instance_q15 * S,
  uint16_t numTaps,
  q15_t * pCoeffs,
  q15_t * pState,
  int32_t * pTapDelay,
  uint16_t maxDelay,
  uint32_t blockSize);


  /**
   * @brief Processing function for the Q7 sparse FIR filter.
   * @param[in]  S            points to an instance of the Q7 sparse FIR structure.
   * @param[in]  pSrc         points to the block of input data.
   * @param[out] pDst         points to the block of output data
   * @param[in]  pScratchIn   points to a temporary buffer of size blockSize.
   * @param[in]  pScratchOut  points to a temporary buffer of size blockSize.
   * @param[in]  blockSize    number of input samples to process per call.
   */
  void arm_fir_sparse_q7(
  arm_fir_sparse_instance_q7 * S,
  q7_t * pSrc,
  q7_t * pDst,
  q7_t * pScratchIn,
  q31_t * pScratchOut,
  uint32_t blockSize);


  /**
   * @brief  Initialization function for the Q7 sparse FIR filter.
   * @param[in,out] S          points to an instance of the Q7 sparse FIR structure.
   * @param[in]     numTaps    number of nonzero coefficients in the filter.
   * @param[in]     pCoeffs    points to the array of filter coefficients.
   * @param[in]     pState     points to the state buffer.
   * @param[in]     pTapDelay  points to the array of offset times.
   * @param[in]     maxDelay   maximum offset time supported.
   * @param[in]     blockSize  number of samples that will be processed per block.
   */
  void arm_fir_sparse_init_q7(
  arm_fir_sparse_instance_q7 * S,
  uint16_t numTaps,
  q7_t * pCoeffs,
  q7_t * pState,
  int32_t * pTapDelay,
  uint16_t maxDelay,
  uint32_t blockSize);


  /**
   * @brief  Floating-point sin_cos function.
   * @param[in]  theta   input value in degrees
   * @param[out] pSinVal  points to the processed sine output.
   * @param[out] pCosVal  points to the processed cos output.
   */
  void arm_sin_cos_f32(
  float32_t theta,
  float32_t * pSinVal,
  float32_t * pCosVal);


  /**
   * @brief  Q31 sin_cos function.
   * @param[in]  theta    scaled input value in degrees
   * @param[out] pSinVal  points to the processed sine output.
   * @param[out] pCosVal  points to the processed cosine output.
   */
  void arm_sin_cos_q31(
  q31_t theta,
  q31_t * pSinVal,
  q31_t * pCosVal);


  /**
   * @brief  Floating-point complex conjugate.
   * @param[in]  pSrc        points to the input vector
   * @param[out] pDst        points to the output vector
   * @param[in]  numSamples  number of complex samples in each vector
   */
  void arm_cmplx_conj_f32(
  float32_t * pSrc,
  float32_t * pDst,
  uint32_t numSamples);

  /**
   * @brief  Q31 complex conjugate.
   * @param[in]  pSrc        points to the input vector
   * @param[out] pDst        points to the output vector
   * @param[in]  numSamples  number of complex samples in each vector
   */
  void arm_cmplx_conj_q31(
  q31_t * pSrc,
  q31_t * pDst,
  uint32_t numSamples);


  /**
   * @brief  Q15 complex conjugate.
   * @param[in]  pSrc        points to the input vector
   * @param[out] pDst        points to the output vector
   * @param[in]  numSamples  number of complex samples in each vector
   */
  void arm_cmplx_conj_q15(
  q15_t * pSrc,
  q15_t * pDst,
  uint32_t numSamples);


  /**
   * @brief  Floating-point complex magnitude squared
   * @param[in]  pSrc        points to the complex input vector
   * @param[out] pDst        points to the real output vector
   * @param[in]  numSamples  number of complex samples in the input vector
   */
  void arm_cmplx_mag_squared_f32(
  float32_t * pSrc,
  float32_t * pDst,
  uint32_t numSamples);


  /**
   * @brief  Q31 complex magnitude squared
   * @param[in]  pSrc        points to the complex input vector
   * @param[out] pDst        points to the real output vector
   * @param[in]  numSamples  number of complex samples in the input vector
   */
  void arm_cmplx_mag_squared_q31(
  q31_t * pSrc,
  q31_t * pDst,
  uint32_t numSamples);


  /**
   * @brief  Q15 complex magnitude squared
   * @param[in]  pSrc        points to the complex input vector
   * @param[out] pDst        points to the real output vector
   * @param[in]  numSamples  number of complex samples in the input vector
   */
  void arm_cmplx_mag_squared_q15(
  q15_t * pSrc,
  q15_t * pDst,
  uint32_t numSamples);


 /**
   * @ingroup groupController
   */

  /**
   * @defgroup PID PID Motor Control
   *
   * A Proportional Integral Derivative (PID) controller is a generic feedback control
   * loop mechanism widely used in industrial control systems.
   * A PID controller is the most commonly used type of feedback controller.
   *
   * This set of functions implements (PID) controllers
   * for Q15, Q31, and floating-point data types.  The functions operate on a single sample
   * of data and each call to the function returns a single processed value.
   * <code>S</code> points to an instance of the PID control data structure.  <code>in</code>
   * is the input sample value. The functions return the output value.
   *
   * \par Algorithm:
   * <pre>
   *    y[n] = y[n-1] + A0 * x[n] + A1 * x[n-1] + A2 * x[n-2]
   *    A0 = Kp + Ki + Kd
   *    A1 = (-Kp ) - (2 * Kd )
   *    A2 = Kd  </pre>
   *
   * \par
   * where \c Kp is proportional constant, \c Ki is Integral constant and \c Kd is Derivative constant
   *
   * \par
   * \image html PID.gif "Proportional Integral Derivative Controller"
   *
   * \par
   * The PID controller calculates an "error" value as the difference between
   * the measured output and the reference input.
   * The controller attempts to minimize the error by adjusting the process control inputs.
   * The proportional value determines the reaction to the current error,
   * the integral value determines the reaction based on the sum of recent errors,
   * and the derivative value determines the reaction based on the rate at which the error has been changing.
   *
   * \par Instance Structure
   * The Gains A0, A1, A2 and state variables for a PID controller are stored together in an instance data structure.
   * A separate instance structure must be defined for each PID Controller.
   * There are separate instance structure declarations for each of the 3 supported data types.
   *
   * \par Reset Functions
   * There is also an associated reset function for each data type which clears the state array.
   *
   * \par Initialization Functions
   * There is also an associated initialization function for each data type.
   * The initialization function performs the following operations:
   * - Initializes the Gains A0, A1, A2 from Kp,Ki, Kd gains.
   * - Zeros out the values in the state buffer.
   *
   * \par
   * Instance structure cannot be placed into a const data section and it is recommended to use the initialization function.
   *
   * \par Fixed-Point Behavior
   * Care must be taken when using the fixed-point versions of the PID Controller functions.
   * In particular, the overflow and saturation behavior of the accumulator used in each function must be considered.
   * Refer to the function specific documentation below for usage guidelines.
   */

  /**
   * @addtogroup PID
   * @{
   */

  /**
   * @brief  Process function for the floating-point PID Control.
   * @param[in,out] S   is an instance of the floating-point PID Control structure
   * @param[in]     in  input sample to process
   * @return out processed output sample.
   */
  static __INLINE float32_t arm_pid_f32(
  arm_pid_instance_f32 * S,
  float32_t in)
  {
    float32_t out;

    /* y[n] = y[n-1] + A0 * x[n] + A1 * x[n-1] + A2 * x[n-2]  */
    out = (S->A0 * in) +
      (S->A1 * S->state[0]) + (S->A2 * S->state[1]) + (S->state[2]);

    /* Update state */
    S->state[1] = S->state[0];
    S->state[0] = in;
    S->state[2] = out;

    /* return to application */
    return (out);

  }

  /**
   * @brief  Process function for the Q31 PID Control.
   * @param[in,out] S  points to an instance of the Q31 PID Control structure
   * @param[in]     in  input sample to process
   * @return out processed output sample.
   *
   * <b>Scaling and Overflow Behavior:</b>
   * \par
   * The function is implemented using an internal 64-bit accumulator.
   * The accumulator has a 2.62 format and maintains full precision of the intermediate multiplication results but provides only a single guard bit.
   * Thus, if the accumulator result overflows it wraps around rather than clip.
   * In order to avoid overflows completely the input signal must be scaled down by 2 bits as there are four additions.
   * After all multiply-accumulates are performed, the 2.62 accumulator is truncated to 1.32 format and then saturated to 1.31 format.
   */
  static __INLINE q31_t arm_pid_q31(
  arm_pid_instance_q31 * S,
  q31_t in)
  {
    q63_t acc;
    q31_t out;

    /* acc = A0 * x[n]  */
    acc = (q63_t) S->A0 * in;

    /* acc += A1 * x[n-1] */
    acc += (q63_t) S->A1 * S->state[0];

    /* acc += A2 * x[n-2]  */
    acc += (q63_t) S->A2 * S->state[1];

    /* convert output to 1.31 format to add y[n-1] */
    out = (q31_t) (acc >> 31u);

    /* out += y[n-1] */
    out += S->state[2];

    /* Update state */
    S->state[1] = S->state[0];
    S->state[0] = in;
    S->state[2] = out;

    /* return to application */
    return (out);
  }


  /**
   * @brief  Process function for the Q15 PID Control.
   * @param[in,out] S   points to an instance of the Q15 PID Control structure
   * @param[in]     in  input sample to process
   * @return out processed output sample.
   *
   * <b>Scaling and Overflow Behavior:</b>
   * \par
   * The function is implemented using a 64-bit internal accumulator.
   * Both Gains and state variables are represented in 1.15 format and multiplications yield a 2.30 result.
   * The 2.30 intermediate results are accumulated in a 64-bit accumulator in 34.30 format.
   * There is no risk of internal overflow with this approach and the full precision of intermediate multiplications is preserved.
   * After all additions have been performed, the accumulator is truncated to 34.15 format by discarding low 15 bits.
   * Lastly, the accumulator is saturated to yield a result in 1.15 format.
   */
  static __INLINE q15_t arm_pid_q15(
  arm_pid_instance_q15 * S,
  q15_t in)
  {
    q63_t acc;
    q15_t out;

#ifndef ARM_MATH_CM0_FAMILY
    __SIMD32_TYPE *vstate;

    /* Implementation of PID controller */

    /* acc = A0 * x[n]  */
    acc = (q31_t) __SMUAD((uint32_t)S->A0, (uint32_t)in);

    /* acc += A1 * x[n-1] + A2 * x[n-2]  */
    vstate = __SIMD32_CONST(S->state);
    acc = (q63_t)__SMLALD((uint32_t)S->A1, (uint32_t)*vstate, (uint64_t)acc);
#else
    /* acc = A0 * x[n]  */
    acc = ((q31_t) S->A0) * in;

    /* acc += A1 * x[n-1] + A2 * x[n-2]  */
    acc += (q31_t) S->A1 * S->state[0];
    acc += (q31_t) S->A2 * S->state[1];
#endif

    /* acc += y[n-1] */
    acc += (q31_t) S->state[2] << 15;

    /* saturate the output */
    out = (q15_t) (__SSAT((acc >> 15), 16));

    /* Update state */
    S->state[1] = S->state[0];
    S->state[0] = in;
    S->state[2] = out;

    /* return to application */
    return (out);
  }

  /**
   * @} end of PID group
   */


  /**
   * @brief Floating-point matrix inverse.
   * @param[in]  src   points to the instance of the input floating-point matrix structure.
   * @param[out] dst   points to the instance of the output floating-point matrix structure.
   * @return The function returns ARM_MATH_SIZE_MISMATCH, if the dimensions do not match.
   * If the input matrix is singular (does not have an inverse), then the algorithm terminates and returns error status ARM_MATH_SINGULAR.
   */
  arm_status arm_mat_inverse_f32(
  const arm_matrix_instance_f32 * src,
  arm_matrix_instance_f32 * dst);


  /**
   * @brief Floating-point matrix inverse.
   * @param[in]  src   points to the instance of the input floating-point matrix structure.
   * @param[out] dst   points to the instance of the output floating-point matrix structure.
   * @return The function returns ARM_MATH_SIZE_MISMATCH, if the dimensions do not match.
   * If the input matrix is singular (does not have an inverse), then the algorithm terminates and returns error status ARM_MATH_SINGULAR.
   */
  arm_status arm_mat_inverse_f64(
  const arm_matrix_instance_f64 * src,
  arm_matrix_instance_f64 * dst);



  /**
   * @ingroup groupController
   */

  /**
   * @defgroup clarke Vector Clarke Transform
   * Forward Clarke transform converts the instantaneous stator phases into a two-coordinate time invariant vector.
   * Generally the Clarke transform uses three-phase currents <code>Ia, Ib and Ic</code> to calculate currents
   * in the two-phase orthogonal stator axis <code>Ialpha</code> and <code>Ibeta</code>.
   * When <code>Ialpha</code> is superposed with <code>Ia</code> as shown in the figure below
   * \image html clarke.gif Stator current space vector and its components in (a,b).
   * and <code>Ia + Ib + Ic = 0</code>, in this condition <code>Ialpha</code> and <code>Ibeta</code>
   * can be calculated using only <code>Ia</code> and <code>Ib</code>.
   *
   * The function operates on a single sample of data and each call to the function returns the processed output.
   * The library provides separate functions for Q31 and floating-point data types.
   * \par Algorithm
   * \image html clarkeFormula.gif
   * where <code>Ia</code> and <code>Ib</code> are the instantaneous stator phases and
   * <code>pIalpha</code> and <code>pIbeta</code> are the two coordinates of time invariant vector.
   * \par Fixed-Point Behavior
   * Care must be taken when using the Q31 version of the Clarke transform.
   * In particular, the overflow and saturation behavior of the accumulator used must be considered.
   * Refer to the function specific documentation below for usage guidelines.
   */

  /**
   * @addtogroup clarke
   * @{
   */

  /**
   *
   * @brief  Floating-point Clarke transform
   * @param[in]  Ia       input three-phase coordinate <code>a</code>
   * @param[in]  Ib       input three-phase coordinate <code>b</code>
   * @param[out] pIalpha  points to output two-phase orthogonal vector axis alpha
   * @param[out] pIbeta   points to output two-phase orthogonal vector axis beta
   */
  static __INLINE void arm_clarke_f32(
  float32_t Ia,
  float32_t Ib,
  float32_t * pIalpha,
  float32_t * pIbeta)
  {
    /* Calculate pIalpha using the equation, pIalpha = Ia */
    *pIalpha = Ia;

    /* Calculate pIbeta using the equation, pIbeta = (1/sqrt(3)) * Ia + (2/sqrt(3)) * Ib */
    *pIbeta = ((float32_t) 0.57735026919 * Ia + (float32_t) 1.15470053838 * Ib);
  }


  /**
   * @brief  Clarke transform for Q31 version
   * @param[in]  Ia       input three-phase coordinate <code>a</code>
   * @param[in]  Ib       input three-phase coordinate <code>b</code>
   * @param[out] pIalpha  points to output two-phase orthogonal vector axis alpha
   * @param[out] pIbeta   points to output two-phase orthogonal vector axis beta
   *
   * <b>Scaling and Overflow Behavior:</b>
   * \par
   * The function is implemented using an internal 32-bit accumulator.
   * The accumulator maintains 1.31 format by truncating lower 31 bits of the intermediate multiplication in 2.62 format.
   * There is saturation on the addition, hence there is no risk of overflow.
   */
  static __INLINE void arm_clarke_q31(
  q31_t Ia,
  q31_t Ib,
  q31_t * pIalpha,
  q31_t * pIbeta)
  {
    q31_t product1, product2;                    /* Temporary variables used to store intermediate results */

    /* Calculating pIalpha from Ia by equation pIalpha = Ia */
    *pIalpha = Ia;

    /* Intermediate product is calculated by (1/(sqrt(3)) * Ia) */
    product1 = (q31_t) (((q63_t) Ia * 0x24F34E8B) >> 30);

    /* Intermediate product is calculated by (2/sqrt(3) * Ib) */
    product2 = (q31_t) (((q63_t) Ib * 0x49E69D16) >> 30);

    /* pIbeta is calculated by adding the intermediate products */
    *pIbeta = __QADD(product1, product2);
  }

  /**
   * @} end of clarke group
   */

  /**
   * @brief  Converts the elements of the Q7 vector to Q31 vector.
   * @param[in]  pSrc       input pointer
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_q7_to_q31(
  q7_t * pSrc,
  q31_t * pDst,
  uint32_t blockSize);



  /**
   * @ingroup groupController
   */

  /**
   * @defgroup inv_clarke Vector Inverse Clarke Transform
   * Inverse Clarke transform converts the two-coordinate time invariant vector into instantaneous stator phases.
   *
   * The function operates on a single sample of data and each call to the function returns the processed output.
   * The library provides separate functions for Q31 and floating-point data types.
   * \par Algorithm
   * \image html clarkeInvFormula.gif
   * where <code>pIa</code> and <code>pIb</code> are the instantaneous stator phases and
   * <code>Ialpha</code> and <code>Ibeta</code> are the two coordinates of time invariant vector.
   * \par Fixed-Point Behavior
   * Care must be taken when using the Q31 version of the Clarke transform.
   * In particular, the overflow and saturation behavior of the accumulator used must be considered.
   * Refer to the function specific documentation below for usage guidelines.
   */

  /**
   * @addtogroup inv_clarke
   * @{
   */

   /**
   * @brief  Floating-point Inverse Clarke transform
   * @param[in]  Ialpha  input two-phase orthogonal vector axis alpha
   * @param[in]  Ibeta   input two-phase orthogonal vector axis beta
   * @param[out] pIa     points to output three-phase coordinate <code>a</code>
   * @param[out] pIb     points to output three-phase coordinate <code>b</code>
   */
  static __INLINE void arm_inv_clarke_f32(
  float32_t Ialpha,
  float32_t Ibeta,
  float32_t * pIa,
  float32_t * pIb)
  {
    /* Calculating pIa from Ialpha by equation pIa = Ialpha */
    *pIa = Ialpha;

    /* Calculating pIb from Ialpha and Ibeta by equation pIb = -(1/2) * Ialpha + (sqrt(3)/2) * Ibeta */
    *pIb = -0.5f * Ialpha + 0.8660254039f * Ibeta;
  }


  /**
   * @brief  Inverse Clarke transform for Q31 version
   * @param[in]  Ialpha  input two-phase orthogonal vector axis alpha
   * @param[in]  Ibeta   input two-phase orthogonal vector axis beta
   * @param[out] pIa     points to output three-phase coordinate <code>a</code>
   * @param[out] pIb     points to output three-phase coordinate <code>b</code>
   *
   * <b>Scaling and Overflow Behavior:</b>
   * \par
   * The function is implemented using an internal 32-bit accumulator.
   * The accumulator maintains 1.31 format by truncating lower 31 bits of the intermediate multiplication in 2.62 format.
   * There is saturation on the subtraction, hence there is no risk of overflow.
   */
  static __INLINE void arm_inv_clarke_q31(
  q31_t Ialpha,
  q31_t Ibeta,
  q31_t * pIa,
  q31_t * pIb)
  {
    q31_t product1, product2;                    /* Temporary variables used to store intermediate results */

    /* Calculating pIa from Ialpha by equation pIa = Ialpha */
    *pIa = Ialpha;

    /* Intermediate product is calculated by (1/(2*sqrt(3)) * Ia) */
    product1 = (q31_t) (((q63_t) (Ialpha) * (0x40000000)) >> 31);

    /* Intermediate product is calculated by (1/sqrt(3) * pIb) */
    product2 = (q31_t) (((q63_t) (Ibeta) * (0x6ED9EBA1)) >> 31);

    /* pIb is calculated by subtracting the products */
    *pIb = __QSUB(product2, product1);
  }

  /**
   * @} end of inv_clarke group
   */

  /**
   * @brief  Converts the elements of the Q7 vector to Q15 vector.
   * @param[in]  pSrc       input pointer
   * @param[out] pDst       output pointer
   * @param[in]  blockSize  number of samples to process
   */
  void arm_q7_to_q15(
  q7_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize);



  /**
   * @ingroup groupController
   */

  /**
   * @defgroup park Vector Park Transform
   *
   * Forward Park transform converts the input two-coordinate vector to flux and torque components.
   * The Park transform can be used to realize the transformation of the <code>Ialpha</code> and the <code>Ibeta</code> currents
   * from the stationary to the moving reference frame and control the spatial relationship between
   * the stator vector current and rotor flux vector.
   * If we consider the d axis aligned with the rotor flux, the diagram below shows the
   * current vector and the relationship from the two reference frames:
   * \image html park.gif "Stator current space vector and its component in (a,b) and in the d,q rotating reference frame"
   *
   * The function operates on a single sample of data and each call to the function returns the processed output.
   * The library provides separate functions for Q31 and floating-point data types.
   * \par Algorithm
   * \image html parkFormula.gif
   * where <code>Ialpha</code> and <code>Ibeta</code> are the stator vector components,
   * <code>pId</code> and <code>pIq</code> are rotor vector components and <code>cosVal</code> and <code>sinVal</code> are the
   * cosine and sine values of theta (rotor flux position).
   * \par Fixed-Point Behavior
   * Care must be taken when using the Q31 version of the Park transform.
   * In particular, the overflow and saturation behavior of the accumulator used must be considered.
   * Refer to the function specific documentation below for usage guidelines.
   */

  /**
   * @addtogroup park
   * @{
   */

  /**
   * @brief Floating-point Park transform
   * @param[in]  Ialpha  input two-phase vector coordinate alpha
   * @param[in]  Ibeta   input two-phase vector coordinate beta
   * @param[out] pId     points to output   rotor reference frame d
   * @param[out] pIq     points to output   rotor reference frame q
   * @param[in]  sinVal  sine value of rotation angle theta
   * @param[in]  cosVal  cosine value of rotation angle theta
   *
   * The function implements the forward Park transform.
   *
   */
  static __INLINE void arm_park_f32(
  float32_t Ialpha,
  float32_t Ibeta,
  float32_t * pId,
  float32_t * pIq,
  float32_t sinVal,
  float32_t cosVal)
  {
    /* Calculate pId using the equation, pId = Ialpha * cosVal + Ibeta * sinVal */
    *pId = Ialpha * cosVal + Ibeta * sinVal;

    /* Calculate pIq using the equation, pIq = - Ialpha * sinVal + Ibeta * cosVal */
    *pIq = -Ialpha * sinVal + Ibeta * cosVal;
  }


  /**
   * @brief  Park transform for Q31 version
   * @param[in]  Ialpha  input two-phase vector coordinate alpha
   * @param[in]  Ibeta   input two-phase vector coordinate beta
   * @param[out] pId     points to output rotor reference frame d
   * @param[out] pIq     points to output rotor reference frame q
   * @param[in]  sinVal  sine value of rotation angle theta
   * @param[in]  cosVal  cosine value of rotation angle theta
   *
   * <b>Scaling and Overflow Behavior:</b>
   * \par
   * The function is implemented using an internal 32-bit accumulator.
   * The accumulator maintains 1.31 format by truncating lower 31 bits of the intermediate multiplication in 2.62 format.
   * There is saturation on the addition and subtraction, hence there is no risk of overflow.
   */
  static __INLINE void arm_park_q31(
  q31_t Ialpha,
  q31_t Ibeta,
  q31_t * pId,
  q31_t * pIq,
  q31_t sinVal,
  q31_t cosVal)
  {
    q31_t product1, product2;                    /* Temporary variables used to store intermediate results */
    q31_t product3, product4;                    /* Temporary variables used to store intermediate results */

    /* Intermediate product is calculated by (Ialpha * cosVal) */
    product1 = (q31_t) (((q63_t) (Ialpha) * (cosVal)) >> 31);

    /* Intermediate product is calculated by (Ibeta * sinVal) */
    product2 = (q31_t) (((q63_t) (Ibeta) * (sinVal)) >> 31);


    /* Intermediate product is calculated by (Ialpha * sinVal) */
    product3 = (q31_t) (((q63_t) (Ialpha) * (sinVal)) >> 31);

    /* Intermediate product is calculated by (Ibeta * cosVal) */
    product4 = (q31_t) (((q63_t) (Ibeta) * (cosVal)) >> 31);

    /* Calculate pId by adding the two intermediate products 1 and 2 */
    *pId = __QADD(product1, product2);

    /* Calculate pIq by subtracting the two intermediate products 3 from 4 */
    *pIq = __QSUB(product4, product3);
  }

  /**
   * @} end of park group
   */

  /**
   * @brief  Converts the elements of the Q7 vector to floating-point vector.
   * @param[in]  pSrc       is input pointer
   * @param[out] pDst       is output pointer
   * @param[in]  blockSize  is the number of samples to process
   */
  void arm_q7_to_float(
  q7_t * pSrc,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @ingroup groupController
   */

  /**
   * @defgroup inv_park Vector Inverse Park transform
   * Inverse Park transform converts the input flux and torque components to two-coordinate vector.
   *
   * The function operates on a single sample of data and each call to the function returns the processed output.
   * The library provides separate functions for Q31 and floating-point data types.
   * \par Algorithm
   * \image html parkInvFormula.gif
   * where <code>pIalpha</code> and <code>pIbeta</code> are the stator vector components,
   * <code>Id</code> and <code>Iq</code> are rotor vector components and <code>cosVal</code> and <code>sinVal</code> are the
   * cosine and sine values of theta (rotor flux position).
   * \par Fixed-Point Behavior
   * Care must be taken when using the Q31 version of the Park transform.
   * In particular, the overflow and saturation behavior of the accumulator used must be considered.
   * Refer to the function specific documentation below for usage guidelines.
   */

  /**
   * @addtogroup inv_park
   * @{
   */

   /**
   * @brief  Floating-point Inverse Park transform
   * @param[in]  Id       input coordinate of rotor reference frame d
   * @param[in]  Iq       input coordinate of rotor reference frame q
   * @param[out] pIalpha  points to output two-phase orthogonal vector axis alpha
   * @param[out] pIbeta   points to output two-phase orthogonal vector axis beta
   * @param[in]  sinVal   sine value of rotation angle theta
   * @param[in]  cosVal   cosine value of rotation angle theta
   */
  static __INLINE void arm_inv_park_f32(
  float32_t Id,
  float32_t Iq,
  float32_t * pIalpha,
  float32_t * pIbeta,
  float32_t sinVal,
  float32_t cosVal)
  {
    /* Calculate pIalpha using the equation, pIalpha = Id * cosVal - Iq * sinVal */
    *pIalpha = Id * cosVal - Iq * sinVal;

    /* Calculate pIbeta using the equation, pIbeta = Id * sinVal + Iq * cosVal */
    *pIbeta = Id * sinVal + Iq * cosVal;
  }


  /**
   * @brief  Inverse Park transform for   Q31 version
   * @param[in]  Id       input coordinate of rotor reference frame d
   * @param[in]  Iq       input coordinate of rotor reference frame q
   * @param[out] pIalpha  points to output two-phase orthogonal vector axis alpha
   * @param[out] pIbeta   points to output two-phase orthogonal vector axis beta
   * @param[in]  sinVal   sine value of rotation angle theta
   * @param[in]  cosVal   cosine value of rotation angle theta
   *
   * <b>Scaling and Overflow Behavior:</b>
   * \par
   * The function is implemented using an internal 32-bit accumulator.
   * The accumulator maintains 1.31 format by truncating lower 31 bits of the intermediate multiplication in 2.62 format.
   * There is saturation on the addition, hence there is no risk of overflow.
   */
  static __INLINE void arm_inv_park_q31(
  q31_t Id,
  q31_t Iq,
  q31_t * pIalpha,
  q31_t * pIbeta,
  q31_t sinVal,
  q31_t cosVal)
  {
    q31_t product1, product2;                    /* Temporary variables used to store intermediate results */
    q31_t product3, product4;                    /* Temporary variables used to store intermediate results */

    /* Intermediate product is calculated by (Id * cosVal) */
    product1 = (q31_t) (((q63_t) (Id) * (cosVal)) >> 31);

    /* Intermediate product is calculated by (Iq * sinVal) */
    product2 = (q31_t) (((q63_t) (Iq) * (sinVal)) >> 31);


    /* Intermediate product is calculated by (Id * sinVal) */
    product3 = (q31_t) (((q63_t) (Id) * (sinVal)) >> 31);

    /* Intermediate product is calculated by (Iq * cosVal) */
    product4 = (q31_t) (((q63_t) (Iq) * (cosVal)) >> 31);

    /* Calculate pIalpha by using the two intermediate products 1 and 2 */
    *pIalpha = __QSUB(product1, product2);

    /* Calculate pIbeta by using the two intermediate products 3 and 4 */
    *pIbeta = __QADD(product4, product3);
  }

  /**
   * @} end of Inverse park group
   */


  /**
   * @brief  Converts the elements of the Q31 vector to floating-point vector.
   * @param[in]  pSrc       is input pointer
   * @param[out] pDst       is output pointer
   * @param[in]  blockSize  is the number of samples to process
   */
  void arm_q31_to_float(
  q31_t * pSrc,
  float32_t * pDst,
  uint32_t blockSize);

  /**
   * @ingroup groupInterpolation
   */

  /**
   * @defgroup LinearInterpolate Linear Interpolation
   *
   * Linear interpolation is a method of curve fitting using linear polynomials.
   * Linear interpolation works by effectively drawing a straight line between two neighboring samples and returning the appropriate point along that line
   *
   * \par
   * \image html LinearInterp.gif "Linear interpolation"
   *
   * \par
   * A  Linear Interpolate function calculates an output value(y), for the input(x)
   * using linear interpolation of the input values x0, x1( nearest input values) and the output values y0 and y1(nearest output values)
   *
   * \par Algorithm:
   * <pre>
   *       y = y0 + (x - x0) * ((y1 - y0)/(x1-x0))
   *       where x0, x1 are nearest values of input x
   *             y0, y1 are nearest values to output y
   * </pre>
   *
   * \par
   * This set of functions implements Linear interpolation process
   * for Q7, Q15, Q31, and floating-point data types.  The functions operate on a single
   * sample of data and each call to the function returns a single processed value.
   * <code>S</code> points to an instance of the Linear Interpolate function data structure.
   * <code>x</code> is the input sample value. The functions returns the output value.
   *
   * \par
   * if x is outside of the table boundary, Linear interpolation returns first value of the table
   * if x is below input range and returns last value of table if x is above range.
   */

  /**
   * @addtogroup LinearInterpolate
   * @{
   */

  /**
   * @brief  Process function for the floating-point Linear Interpolation Function.
   * @param[in,out] S  is an instance of the floating-point Linear Interpolation structure
   * @param[in]     x  input sample to process
   * @return y processed output sample.
   *
   */
  static __INLINE float32_t arm_linear_interp_f32(
  arm_linear_interp_instance_f32 * S,
  float32_t x)
  {
    float32_t y;
    float32_t x0, x1;                            /* Nearest input values */
    float32_t y0, y1;                            /* Nearest output values */
    float32_t xSpacing = S->xSpacing;            /* spacing between input values */
    int32_t i;                                   /* Index variable */
    float32_t *pYData = S->pYData;               /* pointer to output table */

    /* Calculation of index */
    i = (int32_t) ((x - S->x1) / xSpacing);

    if(i < 0)
    {
      /* Iniatilize output for below specified range as least output value of table */
      y = pYData[0];
    }
    else if((uint32_t)i >= S->nValues)
    {
      /* Iniatilize output for above specified range as last output value of table */
      y = pYData[S->nValues - 1];
    }
    else
    {
      /* Calculation of nearest input values */
      x0 = S->x1 +  i      * xSpacing;
      x1 = S->x1 + (i + 1) * xSpacing;

      /* Read of nearest output values */
      y0 = pYData[i];
      y1 = pYData[i + 1];

      /* Calculation of output */
      y = y0 + (x - x0) * ((y1 - y0) / (x1 - x0));

    }

    /* returns output value */
    return (y);
  }


   /**
   *
   * @brief  Process function for the Q31 Linear Interpolation Function.
   * @param[in] pYData   pointer to Q31 Linear Interpolation table
   * @param[in] x        input sample to process
   * @param[in] nValues  number of table values
   * @return y processed output sample.
   *
   * \par
   * Input sample <code>x</code> is in 12.20 format which contains 12 bits for table index and 20 bits for fractional part.
   * This function can support maximum of table size 2^12.
   *
   */
  static __INLINE q31_t arm_linear_interp_q31(
  q31_t * pYData,
  q31_t x,
  uint32_t nValues)
  {
    q31_t y;                                     /* output */
    q31_t y0, y1;                                /* Nearest output values */
    q31_t fract;                                 /* fractional part */
    int32_t index;                               /* Index to read nearest output values */

    /* Input is in 12.20 format */
    /* 12 bits for the table index */
    /* Index value calculation */
    index = ((x & (q31_t)0xFFF00000) >> 20);

    if(index >= (int32_t)(nValues - 1))
    {
      return (pYData[nValues - 1]);
    }
    else if(index < 0)
    {
      return (pYData[0]);
    }
    else
    {
      /* 20 bits for the fractional part */
      /* shift left by 11 to keep fract in 1.31 format */
      fract = (x & 0x000FFFFF) << 11;

      /* Read two nearest output values from the index in 1.31(q31) format */
      y0 = pYData[index];
      y1 = pYData[index + 1];

      /* Calculation of y0 * (1-fract) and y is in 2.30 format */
      y = ((q31_t) ((q63_t) y0 * (0x7FFFFFFF - fract) >> 32));

      /* Calculation of y0 * (1-fract) + y1 *fract and y is in 2.30 format */
      y += ((q31_t) (((q63_t) y1 * fract) >> 32));

      /* Convert y to 1.31 format */
      return (y << 1u);
    }
  }


  /**
   *
   * @brief  Process function for the Q15 Linear Interpolation Function.
   * @param[in] pYData   pointer to Q15 Linear Interpolation table
   * @param[in] x        input sample to process
   * @param[in] nValues  number of table values
   * @return y processed output sample.
   *
   * \par
   * Input sample <code>x</code> is in 12.20 format which contains 12 bits for table index and 20 bits for fractional part.
   * This function can support maximum of table size 2^12.
   *
   */
  static __INLINE q15_t arm_linear_interp_q15(
  q15_t * pYData,
  q31_t x,
  uint32_t nValues)
  {
    q63_t y;                                     /* output */
    q15_t y0, y1;                                /* Nearest output values */
    q31_t fract;                                 /* fractional part */
    int32_t index;                               /* Index to read nearest output values */

    /* Input is in 12.20 format */
    /* 12 bits for the table index */
    /* Index value calculation */
    index = ((x & (int32_t)0xFFF00000) >> 20);

    if(index >= (int32_t)(nValues - 1))
    {
      return (pYData[nValues - 1]);
    }
    else if(index < 0)
    {
      return (pYData[0]);
    }
    else
    {
      /* 20 bits for the fractional part */
      /* fract is in 12.20 format */
      fract = (x & 0x000FFFFF);

      /* Read two nearest output values from the index */
      y0 = pYData[index];
      y1 = pYData[index + 1];

      /* Calculation of y0 * (1-fract) and y is in 13.35 format */
      y = ((q63_t) y0 * (0xFFFFF - fract));

      /* Calculation of (y0 * (1-fract) + y1 * fract) and y is in 13.35 format */
      y += ((q63_t) y1 * (fract));

      /* convert y to 1.15 format */
      return (q15_t) (y >> 20);
    }
  }


  /**
   *
   * @brief  Process function for the Q7 Linear Interpolation Function.
   * @param[in] pYData   pointer to Q7 Linear Interpolation table
   * @param[in] x        input sample to process
   * @param[in] nValues  number of table values
   * @return y processed output sample.
   *
   * \par
   * Input sample <code>x</code> is in 12.20 format which contains 12 bits for table index and 20 bits for fractional part.
   * This function can support maximum of table size 2^12.
   */
  static __INLINE q7_t arm_linear_interp_q7(
  q7_t * pYData,
  q31_t x,
  uint32_t nValues)
  {
    q31_t y;                                     /* output */
    q7_t y0, y1;                                 /* Nearest output values */
    q31_t fract;                                 /* fractional part */
    uint32_t index;                              /* Index to read nearest output values */

    /* Input is in 12.20 format */
    /* 12 bits for the table index */
    /* Index value calculation */
    if (x < 0)
    {
      return (pYData[0]);
    }
    index = (x >> 20) & 0xfff;

    if(index >= (nValues - 1))
    {
      return (pYData[nValues - 1]);
    }
    else
    {
      /* 20 bits for the fractional part */
      /* fract is in 12.20 format */
      fract = (x & 0x000FFFFF);

      /* Read two nearest output values from the index and are in 1.7(q7) format */
      y0 = pYData[index];
      y1 = pYData[index + 1];

      /* Calculation of y0 * (1-fract ) and y is in 13.27(q27) format */
      y = ((y0 * (0xFFFFF - fract)));

      /* Calculation of y1 * fract + y0 * (1-fract) and y is in 13.27(q27) format */
      y += (y1 * fract);

      /* convert y to 1.7(q7) format */
      return (q7_t) (y >> 20);
     }
  }

  /**
   * @} end of LinearInterpolate group
   */

  /**
   * @brief  Fast approximation to the trigonometric sine function for floating-point data.
   * @param[in] x  input value in radians.
   * @return  sin(x).
   */
  float32_t arm_sin_f32(
  float32_t x);


  /**
   * @brief  Fast approximation to the trigonometric sine function for Q31 data.
   * @param[in] x  Scaled input value in radians.
   * @return  sin(x).
   */
  q31_t arm_sin_q31(
  q31_t x);


  /**
   * @brief  Fast approximation to the trigonometric sine function for Q15 data.
   * @param[in] x  Scaled input value in radians.
   * @return  sin(x).
   */
  q15_t arm_sin_q15(
  q15_t x);


  /**
   * @brief  Fast approximation to the trigonometric cosine function for floating-point data.
   * @param[in] x  input value in radians.
   * @return  cos(x).
   */
  float32_t arm_cos_f32(
  float32_t x);


  /**
   * @brief Fast approximation to the trigonometric cosine function for Q31 data.
   * @param[in] x  Scaled input value in radians.
   * @return  cos(x).
   */
  q31_t arm_cos_q31(
  q31_t x);


  /**
   * @brief  Fast approximation to the trigonometric cosine function for Q15 data.
   * @param[in] x  Scaled input value in radians.
   * @return  cos(x).
   */
  q15_t arm_cos_q15(
  q15_t x);


  /**
   * @ingroup groupFastMath
   */


  /**
   * @defgroup SQRT Square Root
   *
   * Computes the square root of a number.
   * There are separate functions for Q15, Q31, and floating-point data types.
   * The square root function is computed using the Newton-Raphson algorithm.
   * This is an iterative algorithm of the form:
   * <pre>
   *      x1 = x0 - f(x0)/f'(x0)
   * </pre>
   * where <code>x1</code> is the current estimate,
   * <code>x0</code> is the previous estimate, and
   * <code>f'(x0)</code> is the derivative of <code>f()</code> evaluated at <code>x0</code>.
   * For the square root function, the algorithm reduces to:
   * <pre>
   *     x0 = in/2                         [initial guess]
   *     x1 = 1/2 * ( x0 + in / x0)        [each iteration]
   * </pre>
   */


  /**
   * @addtogroup SQRT
   * @{
   */

  /**
   * @brief  Floating-point square root function.
   * @param[in]  in    input value.
   * @param[out] pOut  square root of input value.
   * @return The function returns ARM_MATH_SUCCESS if input value is positive value or ARM_MATH_ARGUMENT_ERROR if
   * <code>in</code> is negative value and returns zero output for negative values.
   */
  static __INLINE arm_status arm_sqrt_f32(
  float32_t in,
  float32_t * pOut)
  {
    if(in >= 0.0f)
    {

#if   (__FPU_USED == 1) && defined ( __CC_ARM   )
      *pOut = __sqrtf(in);
#elif (__FPU_USED == 1) && (defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050))
      *pOut = __builtin_sqrtf(in);
#elif (__FPU_USED == 1) && defined(__GNUC__)
      *pOut = __builtin_sqrtf(in);
#elif (__FPU_USED == 1) && defined ( __ICCARM__ ) && (__VER__ >= 6040000)
      __ASM("VSQRT.F32 %0,%1" : "=t"(*pOut) : "t"(in));
#else
      *pOut = sqrtf(in);
#endif

      return (ARM_MATH_SUCCESS);
    }
    else
    {
      *pOut = 0.0f;
      return (ARM_MATH_ARGUMENT_ERROR);
    }
  }


  /**
   * @brief Q31 square root function.
   * @param[in]  in    input value.  The range of the input value is [0 +1) or 0x00000000 to 0x7FFFFFFF.
   * @param[out] pOut  square root of input value.
   * @return The function returns ARM_MATH_SUCCESS if input value is positive value or ARM_MATH_ARGUMENT_ERROR if
   * <code>in</code> is negative value and returns zero output for negative values.
   */
  arm_status arm_sqrt_q31(
  q31_t in,
  q31_t * pOut);


  /**
   * @brief  Q15 square root function.
   * @param[in]  in    input value.  The range of the input value is [0 +1) or 0x0000 to 0x7FFF.
   * @param[out] pOut  square root of input value.
   * @return The function returns ARM_MATH_SUCCESS if input value is positive value or ARM_MATH_ARGUMENT_ERROR if
   * <code>in</code> is negative value and returns zero output for negative values.
   */
  arm_status arm_sqrt_q15(
  q15_t in,
  q15_t * pOut);

  /**
   * @} end of SQRT group
   */


  /**
   * @brief floating-point Circular write function.
   */
  static __INLINE void arm_circularWrite_f32(
  int32_t * circBuffer,
  int32_t L,
  uint16_t * writeOffset,
  int32_t bufferInc,
  const int32_t * src,
  int32_t srcInc,
  uint32_t blockSize)
  {
    uint32_t i = 0u;
    int32_t wOffset;

    /* Copy the value of Index pointer that points
     * to the current location where the input samples to be copied */
    wOffset = *writeOffset;

    /* Loop over the blockSize */
    i = blockSize;

    while(i > 0u)
    {
      /* copy the input sample to the circular buffer */
      circBuffer[wOffset] = *src;

      /* Update the input pointer */
      src += srcInc;

      /* Circularly update wOffset.  Watch out for positive and negative value */
      wOffset += bufferInc;
      if(wOffset >= L)
        wOffset -= L;

      /* Decrement the loop counter */
      i--;
    }

    /* Update the index pointer */
    *writeOffset = (uint16_t)wOffset;
  }



  /**
   * @brief floating-point Circular Read function.
   */
  static __INLINE void arm_circularRead_f32(
  int32_t * circBuffer,
  int32_t L,
  int32_t * readOffset,
  int32_t bufferInc,
  int32_t * dst,
  int32_t * dst_base,
  int32_t dst_length,
  int32_t dstInc,
  uint32_t blockSize)
  {
    uint32_t i = 0u;
    int32_t rOffset, dst_end;

    /* Copy the value of Index pointer that points
     * to the current location from where the input samples to be read */
    rOffset = *readOffset;
    dst_end = (int32_t) (dst_base + dst_length);

    /* Loop over the blockSize */
    i = blockSize;

    while(i > 0u)
    {
      /* copy the sample from the circular buffer to the destination buffer */
      *dst = circBuffer[rOffset];

      /* Update the input pointer */
      dst += dstInc;

      if(dst == (int32_t *) dst_end)
      {
        dst = dst_base;
      }

      /* Circularly update rOffset.  Watch out for positive and negative value  */
      rOffset += bufferInc;

      if(rOffset >= L)
      {
        rOffset -= L;
      }

      /* Decrement the loop counter */
      i--;
    }

    /* Update the index pointer */
    *readOffset = rOffset;
  }


  /**
   * @brief Q15 Circular write function.
   */
  static __INLINE void arm_circularWrite_q15(
  q15_t * circBuffer,
  int32_t L,
  uint16_t * writeOffset,
  int32_t bufferInc,
  const q15_t * src,
  int32_t srcInc,
  uint32_t blockSize)
  {
    uint32_t i = 0u;
    int32_t wOffset;

    /* Copy the value of Index pointer that points
     * to the current location where the input samples to be copied */
    wOffset = *writeOffset;

    /* Loop over the blockSize */
    i = blockSize;

    while(i > 0u)
    {
      /* copy the input sample to the circular buffer */
      circBuffer[wOffset] = *src;

      /* Update the input pointer */
      src += srcInc;

      /* Circularly update wOffset.  Watch out for positive and negative value */
      wOffset += bufferInc;
      if(wOffset >= L)
        wOffset -= L;

      /* Decrement the loop counter */
      i--;
    }

    /* Update the index pointer */
    *writeOffset = (uint16_t)wOffset;
  }


  /**
   * @brief Q15 Circular Read function.
   */
  static __INLINE void arm_circularRead_q15(
  q15_t * circBuffer,
  int32_t L,
  int32_t * readOffset,
  int32_t bufferInc,
  q15_t * dst,
  q15_t * dst_base,
  int32_t dst_length,
  int32_t dstInc,
  uint32_t blockSize)
  {
    uint32_t i = 0;
    int32_t rOffset, dst_end;

    /* Copy the value of Index pointer that points
     * to the current location from where the input samples to be read */
    rOffset = *readOffset;

    dst_end = (int32_t) (dst_base + dst_length);

    /* Loop over the blockSize */
    i = blockSize;

    while(i > 0u)
    {
      /* copy the sample from the circular buffer to the destination buffer */
      *dst = circBuffer[rOffset];

      /* Update the input pointer */
      dst += dstInc;

      if(dst == (q15_t *) dst_end)
      {
        dst = dst_base;
      }

      /* Circularly update wOffset.  Watch out for positive and negative value */
      rOffset += bufferInc;

      if(rOffset >= L)
      {
        rOffset -= L;
      }

      /* Decrement the loop counter */
      i--;
    }

    /* Update the index pointer */
    *readOffset = rOffset;
  }


  /**
   * @brief Q7 Circular write function.
   */
  static __INLINE void arm_circularWrite_q7(
  q7_t * circBuffer,
  int32_t L,
  uint16_t * writeOffset,
  int32_t bufferInc,
  const q7_t * src,
  int32_t srcInc,
  uint32_t blockSize)
  {
    uint32_t i = 0u;
    int32_t wOffset;

    /* Copy the value of Index pointer that points
     * to the current location where the input samples to be copied */
    wOffset = *writeOffset;

    /* Loop over the blockSize */
    i = blockSize;

    while(i > 0u)
    {
      /* copy the input sample to the circular buffer */
      circBuffer[wOffset] = *src;

      /* Update the input pointer */
      src += srcInc;

      /* Circularly update wOffset.  Watch out for positive and negative value */
      wOffset += bufferInc;
      if(wOffset >= L)
        wOffset -= L;

      /* Decrement the loop counter */
      i--;
    }

    /* Update the index pointer */
    *writeOffset = (uint16_t)wOffset;
  }


  /**
   * @brief Q7 Circular Read function.
   */
  static __INLINE void arm_circularRead_q7(
  q7_t * circBuffer,
  int32_t L,
  int32_t * readOffset,
  int32_t bufferInc,
  q7_t * dst,
  q7_t * dst_base,
  int32_t dst_length,
  int32_t dstInc,
  uint32_t blockSize)
  {
    uint32_t i = 0;
    int32_t rOffset, dst_end;

    /* Copy the value of Index pointer that points
     * to the current location from where the input samples to be read */
    rOffset = *readOffset;

    dst_end = (int32_t) (dst_base + dst_length);

    /* Loop over the blockSize */
    i = blockSize;

    while(i > 0u)
    {
      /* copy the sample from the circular buffer to the destination buffer */
      *dst = circBuffer[rOffset];

      /* Update the input pointer */
      dst += dstInc;

      if(dst == (q7_t *) dst_end)
      {
        dst = dst_base;
      }

      /* Circularly update rOffset.  Watch out for positive and negative value */
      rOffset += bufferInc;

      if(rOffset >= L)
      {
        rOffset -= L;
      }

      /* Decrement the loop counter */
      i--;
    }

    /* Update the index pointer */
    *readOffset = rOffset;
  }


  /**
   * @brief  Sum of the squares of the elements of a Q31 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_power_q31(
  q31_t * pSrc,
  uint32_t blockSize,
  q63_t * pResult);


  /**
   * @brief  Sum of the squares of the elements of a floating-point vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_power_f32(
  float32_t * pSrc,
  uint32_t blockSize,
  float32_t * pResult);


  /**
   * @brief  Sum of the squares of the elements of a Q15 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_power_q15(
  q15_t * pSrc,
  uint32_t blockSize,
  q63_t * pResult);


  /**
   * @brief  Sum of the squares of the elements of a Q7 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_power_q7(
  q7_t * pSrc,
  uint32_t blockSize,
  q31_t * pResult);


  /**
   * @brief  Mean value of a Q7 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_mean_q7(
  q7_t * pSrc,
  uint32_t blockSize,
  q7_t * pResult);


  /**
   * @brief  Mean value of a Q15 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_mean_q15(
  q15_t * pSrc,
  uint32_t blockSize,
  q15_t * pResult);


  /**
   * @brief  Mean value of a Q31 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_mean_q31(
  q31_t * pSrc,
  uint32_t blockSize,
  q31_t * pResult);


  /**
   * @brief  Mean value of a floating-point vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_mean_f32(
  float32_t * pSrc,
  uint32_t blockSize,
  float32_t * pResult);


  /**
   * @brief  Variance of the elements of a floating-point vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_var_f32(
  float32_t * pSrc,
  uint32_t blockSize,
  float32_t * pResult);


  /**
   * @brief  Variance of the elements of a Q31 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_var_q31(
  q31_t * pSrc,
  uint32_t blockSize,
  q31_t * pResult);


  /**
   * @brief  Variance of the elements of a Q15 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_var_q15(
  q15_t * pSrc,
  uint32_t blockSize,
  q15_t * pResult);


  /**
   * @brief  Root Mean Square of the elements of a floating-point vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_rms_f32(
  float32_t * pSrc,
  uint32_t blockSize,
  float32_t * pResult);


  /**
   * @brief  Root Mean Square of the elements of a Q31 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_rms_q31(
  q31_t * pSrc,
  uint32_t blockSize,
  q31_t * pResult);


  /**
   * @brief  Root Mean Square of the elements of a Q15 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_rms_q15(
  q15_t * pSrc,
  uint32_t blockSize,
  q15_t * pResult);


  /**
   * @brief  Standard deviation of the elements of a floating-point vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_std_f32(
  float32_t * pSrc,
  uint32_t blockSize,
  float32_t * pResult);


  /**
   * @brief  Standard deviation of the elements of a Q31 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_std_q31(
  q31_t * pSrc,
  uint32_t blockSize,
  q31_t * pResult);


  /**
   * @brief  Standard deviation of the elements of a Q15 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output value.
   */
  void arm_std_q15(
  q15_t * pSrc,
  uint32_t blockSize,
  q15_t * pResult);


  /**
   * @brief  Floating-point complex magnitude
   * @param[in]  pSrc        points to the complex input vector
   * @param[out] pDst        points to the real output vector
   * @param[in]  numSamples  number of complex samples in the input vector
   */
  void arm_cmplx_mag_f32(
  float32_t * pSrc,
  float32_t * pDst,
  uint32_t numSamples);


  /**
   * @brief  Q31 complex magnitude
   * @param[in]  pSrc        points to the complex input vector
   * @param[out] pDst        points to the real output vector
   * @param[in]  numSamples  number of complex samples in the input vector
   */
  void arm_cmplx_mag_q31(
  q31_t * pSrc,
  q31_t * pDst,
  uint32_t numSamples);


  /**
   * @brief  Q15 complex magnitude
   * @param[in]  pSrc        points to the complex input vector
   * @param[out] pDst        points to the real output vector
   * @param[in]  numSamples  number of complex samples in the input vector
   */
  void arm_cmplx_mag_q15(
  q15_t * pSrc,
  q15_t * pDst,
  uint32_t numSamples);


  /**
   * @brief  Q15 complex dot product
   * @param[in]  pSrcA       points to the first input vector
   * @param[in]  pSrcB       points to the second input vector
   * @param[in]  numSamples  number of complex samples in each vector
   * @param[out] realResult  real part of the result returned here
   * @param[out] imagResult  imaginary part of the result returned here
   */
  void arm_cmplx_dot_prod_q15(
  q15_t * pSrcA,
  q15_t * pSrcB,
  uint32_t numSamples,
  q31_t * realResult,
  q31_t * imagResult);


  /**
   * @brief  Q31 complex dot product
   * @param[in]  pSrcA       points to the first input vector
   * @param[in]  pSrcB       points to the second input vector
   * @param[in]  numSamples  number of complex samples in each vector
   * @param[out] realResult  real part of the result returned here
   * @param[out] imagResult  imaginary part of the result returned here
   */
  void arm_cmplx_dot_prod_q31(
  q31_t * pSrcA,
  q31_t * pSrcB,
  uint32_t numSamples,
  q63_t * realResult,
  q63_t * imagResult);


  /**
   * @brief  Floating-point complex dot product
   * @param[in]  pSrcA       points to the first input vector
   * @param[in]  pSrcB       points to the second input vector
   * @param[in]  numSamples  number of complex samples in each vector
   * @param[out] realResult  real part of the result returned here
   * @param[out] imagResult  imaginary part of the result returned here
   */
  void arm_cmplx_dot_prod_f32(
  float32_t * pSrcA,
  float32_t * pSrcB,
  uint32_t numSamples,
  float32_t * realResult,
  float32_t * imagResult);


  /**
   * @brief  Q15 complex-by-real multiplication
   * @param[in]  pSrcCmplx   points to the complex input vector
   * @param[in]  pSrcReal    points to the real input vector
   * @param[out] pCmplxDst   points to the complex output vector
   * @param[in]  numSamples  number of samples in each vector
   */
  void arm_cmplx_mult_real_q15(
  q15_t * pSrcCmplx,
  q15_t * pSrcReal,
  q15_t * pCmplxDst,
  uint32_t numSamples);


  /**
   * @brief  Q31 complex-by-real multiplication
   * @param[in]  pSrcCmplx   points to the complex input vector
   * @param[in]  pSrcReal    points to the real input vector
   * @param[out] pCmplxDst   points to the complex output vector
   * @param[in]  numSamples  number of samples in each vector
   */
  void arm_cmplx_mult_real_q31(
  q31_t * pSrcCmplx,
  q31_t * pSrcReal,
  q31_t * pCmplxDst,
  uint32_t numSamples);


  /**
   * @brief  Floating-point complex-by-real multiplication
   * @param[in]  pSrcCmplx   points to the complex input vector
   * @param[in]  pSrcReal    points to the real input vector
   * @param[out] pCmplxDst   points to the complex output vector
   * @param[in]  numSamples  number of samples in each vector
   */
  void arm_cmplx_mult_real_f32(
  float32_t * pSrcCmplx,
  float32_t * pSrcReal,
  float32_t * pCmplxDst,
  uint32_t numSamples);


  /**
   * @brief  Minimum value of a Q7 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] result     is output pointer
   * @param[in]  index      is the array index of the minimum value in the input buffer.
   */
  void arm_min_q7(
  q7_t * pSrc,
  uint32_t blockSize,
  q7_t * result,
  uint32_t * index);


  /**
   * @brief  Minimum value of a Q15 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output pointer
   * @param[in]  pIndex     is the array index of the minimum value in the input buffer.
   */
  void arm_min_q15(
  q15_t * pSrc,
  uint32_t blockSize,
  q15_t * pResult,
  uint32_t * pIndex);


  /**
   * @brief  Minimum value of a Q31 vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output pointer
   * @param[out] pIndex     is the array index of the minimum value in the input buffer.
   */
  void arm_min_q31(
  q31_t * pSrc,
  uint32_t blockSize,
  q31_t * pResult,
  uint32_t * pIndex);


  /**
   * @brief  Minimum value of a floating-point vector.
   * @param[in]  pSrc       is input pointer
   * @param[in]  blockSize  is the number of samples to process
   * @param[out] pResult    is output pointer
   * @param[out] pIndex     is the array index of the minimum value in the input buffer.
   */
  void arm_min_f32(
  float32_t * pSrc,
  uint32_t blockSize,
  float32_t * pResult,
  uint32_t * pIndex);


/**
 * @brief Maximum value of a Q7 vector.
 * @param[in]  pSrc       points to the input buffer
 * @param[in]  blockSize  length of the input vector
 * @param[out] pResult    maximum value returned here
 * @param[out] pIndex     index of maximum value returned here
 */
  void arm_max_q7(
  q7_t * pSrc,
  uint32_t blockSize,
  q7_t * pResult,
  uint32_t * pIndex);


/**
 * @brief Maximum value of a Q15 vector.
 * @param[in]  pSrc       points to the input buffer
 * @param[in]  blockSize  length of the input vector
 * @param[out] pResult    maximum value returned here
 * @param[out] pIndex     index of maximum value returned here
 */
  void arm_max_q15(
  q15_t * pSrc,
  uint32_t blockSize,
  q15_t * pResult,
  uint32_t * pIndex);


/**
 * @brief Maximum value of a Q31 vector.
 * @param[in]  pSrc       points to the input buffer
 * @param[in]  blockSize  length of the input vector
 * @param[out] pResult    maximum value returned here
 * @param[out] pIndex     index of maximum value returned here
 */
  void arm_max_q31(
  q31_t * pSrc,
  uint32_t blockSize,
  q31_t * pResult,
  uint32_t * pIndex);


/**
 * @brief Maximum value of a floating-point vector.
 * @param[in]  pSrc       points to the input buffer
 * @param[in]  blockSize  length of the input vector
 * @param[out] pResult    maximum value returned here
 * @param[out] pIndex     index of maximum value returned here
 */
  void arm_max_f32(
  float32_t * pSrc,
  uint32_t blockSize,
  float32_t * pResult,
  uint32_t * pIndex);


  /**
   * @brief  Q15 complex-by-complex multiplication
   * @param[in]  pSrcA       points to the first input vector
   * @param[in]  pSrcB       points to the second input vector
   * @param[out] pDst        points to the output vector
   * @param[in]  numSamples  number of complex samples in each vector
   */
  void arm_cmplx_mult_cmplx_q15(
  q15_t * pSrcA,
  q15_t * pSrcB,
  q15_t * pDst,
  uint32_t numSamples);


  /**
   * @brief  Q31 complex-by-complex multiplication
   * @param[in]  pSrcA       points to the first input vector
   * @param[in]  pSrcB       points to the second input vector
   * @param[out] pDst        points to the output vector
   * @param[in]  numSamples  number of complex samples in each vector
   */
  void arm_cmplx_mult_cmplx_q31(
  q31_t * pSrcA,
  q31_t * pSrcB,
  q31_t * pDst,
  uint32_t numSamples);


  /**
   * @brief  Floating-point complex-by-complex multiplication
   * @param[in]  pSrcA       points to the first input vector
   * @param[in]  pSrcB       points to the second input vector
   * @param[out] pDst        points to the output vector
   * @param[in]  numSamples  number of complex samples in each vector
   */
  void arm_cmplx_mult_cmplx_f32(
  float32_t * pSrcA,
  float32_t * pSrcB,
  float32_t * pDst,
  uint32_t numSamples);


  /**
   * @brief Converts the elements of the floating-point vector to Q31 vector.
   * @param[in]  pSrc       points to the floating-point input vector
   * @param[out] pDst       points to the Q31 output vector
   * @param[in]  blockSize  length of the input vector
   */
  void arm_float_to_q31(
  float32_t * pSrc,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Converts the elements of the floating-point vector to Q15 vector.
   * @param[in]  pSrc       points to the floating-point input vector
   * @param[out] pDst       points to the Q15 output vector
   * @param[in]  blockSize  length of the input vector
   */
  void arm_float_to_q15(
  float32_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief Converts the elements of the floating-point vector to Q7 vector.
   * @param[in]  pSrc       points to the floating-point input vector
   * @param[out] pDst       points to the Q7 output vector
   * @param[in]  blockSize  length of the input vector
   */
  void arm_float_to_q7(
  float32_t * pSrc,
  q7_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Converts the elements of the Q31 vector to Q15 vector.
   * @param[in]  pSrc       is input pointer
   * @param[out] pDst       is output pointer
   * @param[in]  blockSize  is the number of samples to process
   */
  void arm_q31_to_q15(
  q31_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Converts the elements of the Q31 vector to Q7 vector.
   * @param[in]  pSrc       is input pointer
   * @param[out] pDst       is output pointer
   * @param[in]  blockSize  is the number of samples to process
   */
  void arm_q31_to_q7(
  q31_t * pSrc,
  q7_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Converts the elements of the Q15 vector to floating-point vector.
   * @param[in]  pSrc       is input pointer
   * @param[out] pDst       is output pointer
   * @param[in]  blockSize  is the number of samples to process
   */
  void arm_q15_to_float(
  q15_t * pSrc,
  float32_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Converts the elements of the Q15 vector to Q31 vector.
   * @param[in]  pSrc       is input pointer
   * @param[out] pDst       is output pointer
   * @param[in]  blockSize  is the number of samples to process
   */
  void arm_q15_to_q31(
  q15_t * pSrc,
  q31_t * pDst,
  uint32_t blockSize);


  /**
   * @brief  Converts the elements of the Q15 vector to Q7 vector.
   * @param[in]  pSrc       is input pointer
   * @param[out] pDst       is output pointer
   * @param[in]  blockSize  is the number of samples to process
   */
  void arm_q15_to_q7(
  q15_t * pSrc,
  q7_t * pDst,
  uint32_t blockSize);


  /**
   * @ingroup groupInterpolation
   */

  /**
   * @defgroup BilinearInterpolate Bilinear Interpolation
   *
   * Bilinear interpolation is an extension of linear interpolation applied to a two dimensional grid.
   * The underlying function <code>f(x, y)</code> is sampled on a regular grid and the interpolation process
   * determines values between the grid points.
   * Bilinear interpolation is equivalent to two step linear interpolation, first in the x-dimension and then in the y-dimension.
   * Bilinear interpolation is often used in image processing to rescale images.
   * The CMSIS DSP library provides bilinear interpolation functions for Q7, Q15, Q31, and floating-point data types.
   *
   * <b>Algorithm</b>
   * \par
   * The instance structure used by the bilinear interpolation functions describes a two dimensional data table.
   * For floating-point, the instance structure is defined as:
   * <pre>
   *   typedef struct
   *   {
   *     uint16_t numRows;
   *     uint16_t numCols;
   *     float32_t *pData;
   * } arm_bilinear_interp_instance_f32;
   * </pre>
   *
   * \par
   * where <code>numRows</code> specifies the number of rows in the table;
   * <code>numCols</code> specifies the number of columns in the table;
   * and <code>pData</code> points to an array of size <code>numRows*numCols</code> values.
   * The data table <code>pTable</code> is organized in row order and the supplied data values fall on integer indexes.
   * That is, table element (x,y) is located at <code>pTable[x + y*numCols]</code> where x and y are integers.
   *
   * \par
   * Let <code>(x, y)</code> specify the desired interpolation point.  Then define:
   * <pre>
   *     XF = floor(x)
   *     YF = floor(y)
   * </pre>
   * \par
   * The interpolated output point is computed as:
   * <pre>
   *  f(x, y) = f(XF, YF) * (1-(x-XF)) * (1-(y-YF))
   *           + f(XF+1, YF) * (x-XF)*(1-(y-YF))
   *           + f(XF, YF+1) * (1-(x-XF))*(y-YF)
   *           + f(XF+1, YF+1) * (x-XF)*(y-YF)
   * </pre>
   * Note that the coordinates (x, y) contain integer and fractional components.
   * The integer components specify which portion of the table to use while the
   * fractional components control the interpolation processor.
   *
   * \par
   * if (x,y) are outside of the table boundary, Bilinear interpolation returns zero output.
   */

  /**
   * @addtogroup BilinearInterpolate
   * @{
   */


  /**
  *
  * @brief  Floating-point bilinear interpolation.
  * @param[in,out] S  points to an instance of the interpolation structure.
  * @param[in]     X  interpolation coordinate.
  * @param[in]     Y  interpolation coordinate.
  * @return out interpolated value.
  */
  static __INLINE float32_t arm_bilinear_interp_f32(
  const arm_bilinear_interp_instance_f32 * S,
  float32_t X,
  float32_t Y)
  {
    float32_t out;
    float32_t f00, f01, f10, f11;
    float32_t *pData = S->pData;
    int32_t xIndex, yIndex, index;
    float32_t xdiff, ydiff;
    float32_t b1, b2, b3, b4;

    xIndex = (int32_t) X;
    yIndex = (int32_t) Y;

    /* Care taken for table outside boundary */
    /* Returns zero output when values are outside table boundary */
    if(xIndex < 0 || xIndex > (S->numRows - 1) || yIndex < 0 || yIndex > (S->numCols - 1))
    {
      return (0);
    }

    /* Calculation of index for two nearest points in X-direction */
    index = (xIndex - 1) + (yIndex - 1) * S->numCols;


    /* Read two nearest points in X-direction */
    f00 = pData[index];
    f01 = pData[index + 1];

    /* Calculation of index for two nearest points in Y-direction */
    index = (xIndex - 1) + (yIndex) * S->numCols;


    /* Read two nearest points in Y-direction */
    f10 = pData[index];
    f11 = pData[index + 1];

    /* Calculation of intermediate values */
    b1 = f00;
    b2 = f01 - f00;
    b3 = f10 - f00;
    b4 = f00 - f01 - f10 + f11;

    /* Calculation of fractional part in X */
    xdiff = X - xIndex;

    /* Calculation of fractional part in Y */
    ydiff = Y - yIndex;

    /* Calculation of bi-linear interpolated output */
    out = b1 + b2 * xdiff + b3 * ydiff + b4 * xdiff * ydiff;

    /* return to application */
    return (out);
  }


  /**
  *
  * @brief  Q31 bilinear interpolation.
  * @param[in,out] S  points to an instance of the interpolation structure.
  * @param[in]     X  interpolation coordinate in 12.20 format.
  * @param[in]     Y  interpolation coordinate in 12.20 format.
  * @return out interpolated value.
  */
  static __INLINE q31_t arm_bilinear_interp_q31(
  arm_bilinear_interp_instance_q31 * S,
  q31_t X,
  q31_t Y)
  {
    q31_t out;                                   /* Temporary output */
    q31_t acc = 0;                               /* output */
    q31_t xfract, yfract;                        /* X, Y fractional parts */
    q31_t x1, x2, y1, y2;                        /* Nearest output values */
    int32_t rI, cI;                              /* Row and column indices */
    q31_t *pYData = S->pData;                    /* pointer to output table values */
    uint32_t nCols = S->numCols;                 /* num of rows */

    /* Input is in 12.20 format */
    /* 12 bits for the table index */
    /* Index value calculation */
    rI = ((X & (q31_t)0xFFF00000) >> 20);

    /* Input is in 12.20 format */
    /* 12 bits for the table index */
    /* Index value calculation */
    cI = ((Y & (q31_t)0xFFF00000) >> 20);

    /* Care taken for table outside boundary */
    /* Returns zero output when values are outside table boundary */
    if(rI < 0 || rI > (S->numRows - 1) || cI < 0 || cI > (S->numCols - 1))
    {
      return (0);
    }

    /* 20 bits for the fractional part */
    /* shift left xfract by 11 to keep 1.31 format */
    xfract = (X & 0x000FFFFF) << 11u;

    /* Read two nearest output values from the index */
    x1 = pYData[(rI) + (int32_t)nCols * (cI)    ];
    x2 = pYData[(rI) + (int32_t)nCols * (cI) + 1];

    /* 20 bits for the fractional part */
    /* shift left yfract by 11 to keep 1.31 format */
    yfract = (Y & 0x000FFFFF) << 11u;

    /* Read two nearest output values from the index */
    y1 = pYData[(rI) + (int32_t)nCols * (cI + 1)    ];
    y2 = pYData[(rI) + (int32_t)nCols * (cI + 1) + 1];

    /* Calculation of x1 * (1-xfract ) * (1-yfract) and acc is in 3.29(q29) format */
    out = ((q31_t) (((q63_t) x1  * (0x7FFFFFFF - xfract)) >> 32));
    acc = ((q31_t) (((q63_t) out * (0x7FFFFFFF - yfract)) >> 32));

    /* x2 * (xfract) * (1-yfract)  in 3.29(q29) and adding to acc */
    out = ((q31_t) ((q63_t) x2 * (0x7FFFFFFF - yfract) >> 32));
    acc += ((q31_t) ((q63_t) out * (xfract) >> 32));

    /* y1 * (1 - xfract) * (yfract)  in 3.29(q29) and adding to acc */
    out = ((q31_t) ((q63_t) y1 * (0x7FFFFFFF - xfract) >> 32));
    acc += ((q31_t) ((q63_t) out * (yfract) >> 32));

    /* y2 * (xfract) * (yfract)  in 3.29(q29) and adding to acc */
    out = ((q31_t) ((q63_t) y2 * (xfract) >> 32));
    acc += ((q31_t) ((q63_t) out * (yfract) >> 32));

    /* Convert acc to 1.31(q31) format */
    return ((q31_t)(acc << 2));
  }


  /**
  * @brief  Q15 bilinear interpolation.
  * @param[in,out] S  points to an instance of the interpolation structure.
  * @param[in]     X  interpolation coordinate in 12.20 format.
  * @param[in]     Y  interpolation coordinate in 12.20 format.
  * @return out interpolated value.
  */
  static __INLINE q15_t arm_bilinear_interp_q15(
  arm_bilinear_interp_instance_q15 * S,
  q31_t X,
  q31_t Y)
  {
    q63_t acc = 0;                               /* output */
    q31_t out;                                   /* Temporary output */
    q15_t x1, x2, y1, y2;                        /* Nearest output values */
    q31_t xfract, yfract;                        /* X, Y fractional parts */
    int32_t rI, cI;                              /* Row and column indices */
    q15_t *pYData = S->pData;                    /* pointer to output table values */
    uint32_t nCols = S->numCols;                 /* num of rows */

    /* Input is in 12.20 format */
    /* 12 bits for the table index */
    /* Index value calculation */
    rI = ((X & (q31_t)0xFFF00000) >> 20);

    /* Input is in 12.20 format */
    /* 12 bits for the table index */
    /* Index value calculation */
    cI = ((Y & (q31_t)0xFFF00000) >> 20);

    /* Care taken for table outside boundary */
    /* Returns zero output when values are outside table boundary */
    if(rI < 0 || rI > (S->numRows - 1) || cI < 0 || cI > (S->numCols - 1))
    {
      return (0);
    }

    /* 20 bits for the fractional part */
    /* xfract should be in 12.20 format */
    xfract = (X & 0x000FFFFF);

    /* Read two nearest output values from the index */
    x1 = pYData[((uint32_t)rI) + nCols * ((uint32_t)cI)    ];
    x2 = pYData[((uint32_t)rI) + nCols * ((uint32_t)cI) + 1];

    /* 20 bits for the fractional part */
    /* yfract should be in 12.20 format */
    yfract = (Y & 0x000FFFFF);

    /* Read two nearest output values from the index */
    y1 = pYData[((uint32_t)rI) + nCols * ((uint32_t)cI + 1)    ];
    y2 = pYData[((uint32_t)rI) + nCols * ((uint32_t)cI + 1) + 1];

    /* Calculation of x1 * (1-xfract ) * (1-yfract) and acc is in 13.51 format */

    /* x1 is in 1.15(q15), xfract in 12.20 format and out is in 13.35 format */
    /* convert 13.35 to 13.31 by right shifting  and out is in 1.31 */
    out = (q31_t) (((q63_t) x1 * (0xFFFFF - xfract)) >> 4u);
    acc = ((q63_t) out * (0xFFFFF - yfract));

    /* x2 * (xfract) * (1-yfract)  in 1.51 and adding to acc */
    out = (q31_t) (((q63_t) x2 * (0xFFFFF - yfract)) >> 4u);
    acc += ((q63_t) out * (xfract));

    /* y1 * (1 - xfract) * (yfract)  in 1.51 and adding to acc */
    out = (q31_t) (((q63_t) y1 * (0xFFFFF - xfract)) >> 4u);
    acc += ((q63_t) out * (yfract));

    /* y2 * (xfract) * (yfract)  in 1.51 and adding to acc */
    out = (q31_t) (((q63_t) y2 * (xfract)) >> 4u);
    acc += ((q63_t) out * (yfract));

    /* acc is in 13.51 format and down shift acc by 36 times */
    /* Convert out to 1.15 format */
    return ((q15_t)(acc >> 36));
  }


  /**
  * @brief  Q7 bilinear interpolation.
  * @param[in,out] S  points to an instance of the interpolation structure.
  * @param[in]     X  interpolation coordinate in 12.20 format.
  * @param[in]     Y  interpolation coordinate in 12.20 format.
  * @return out interpolated value.
  */
  static __INLINE q7_t arm_bilinear_interp_q7(
  arm_bilinear_interp_instance_q7 * S,
  q31_t X,
  q31_t Y)
  {
    q63_t acc = 0;                               /* output */
    q31_t out;                                   /* Temporary output */
    q31_t xfract, yfract;                        /* X, Y fractional parts */
    q7_t x1, x2, y1, y2;                         /* Nearest output values */
    int32_t rI, cI;                              /* Row and column indices */
    q7_t *pYData = S->pData;                     /* pointer to output table values */
    uint32_t nCols = S->numCols;                 /* num of rows */

    /* Input is in 12.20 format */
    /* 12 bits for the table index */
    /* Index value calculation */
    rI = ((X & (q31_t)0xFFF00000) >> 20);

    /* Input is in 12.20 format */
    /* 12 bits for the table index */
    /* Index value calculation */
    cI = ((Y & (q31_t)0xFFF00000) >> 20);

    /* Care taken for table outside boundary */
    /* Returns zero output when values are outside table boundary */
    if(rI < 0 || rI > (S->numRows - 1) || cI < 0 || cI > (S->numCols - 1))
    {
      return (0);
    }

    /* 20 bits for the fractional part */
    /* xfract should be in 12.20 format */
    xfract = (X & (q31_t)0x000FFFFF);

    /* Read two nearest output values from the index */
    x1 = pYData[((uint32_t)rI) + nCols * ((uint32_t)cI)    ];
    x2 = pYData[((uint32_t)rI) + nCols * ((uint32_t)cI) + 1];

    /* 20 bits for the fractional part */
    /* yfract should be in 12.20 format */
    yfract = (Y & (q31_t)0x000FFFFF);

    /* Read two nearest output values from the index */
    y1 = pYData[((uint32_t)rI) + nCols * ((uint32_t)cI + 1)    ];
    y2 = pYData[((uint32_t)rI) + nCols * ((uint32_t)cI + 1) + 1];

    /* Calculation of x1 * (1-xfract ) * (1-yfract) and acc is in 16.47 format */
    out = ((x1 * (0xFFFFF - xfract)));
    acc = (((q63_t) out * (0xFFFFF - yfract)));

    /* x2 * (xfract) * (1-yfract)  in 2.22 and adding to acc */
    out = ((x2 * (0xFFFFF - yfract)));
    acc += (((q63_t) out * (xfract)));

    /* y1 * (1 - xfract) * (yfract)  in 2.22 and adding to acc */
    out = ((y1 * (0xFFFFF - xfract)));
    acc += (((q63_t) out * (yfract)));

    /* y2 * (xfract) * (yfract)  in 2.22 and adding to acc */
    out = ((y2 * (yfract)));
    acc += (((q63_t) out * (xfract)));

    /* acc in 16.47 format and down shift by 40 to convert to 1.7 format */
    return ((q7_t)(acc >> 40));
  }

  /**
   * @} end of BilinearInterpolate group
   */


/* SMMLAR */
#define multAcc_32x32_keep32_R(a, x, y) \
    a = (q31_t) (((((q63_t) a) << 32) + ((q63_t) x * y) + 0x80000000LL ) >> 32)

/* SMMLSR */
#define multSub_32x32_keep32_R(a, x, y) \
    a = (q31_t) (((((q63_t) a) << 32) - ((q63_t) x * y) + 0x80000000LL ) >> 32)

/* SMMULR */
#define mult_32x32_keep32_R(a, x, y) \
    a = (q31_t) (((q63_t) x * y + 0x80000000LL ) >> 32)

/* SMMLA */
#define multAcc_32x32_keep32(a, x, y) \
    a += (q31_t) (((q63_t) x * y) >> 32)

/* SMMLS */
#define multSub_32x32_keep32(a, x, y) \
    a -= (q31_t) (((q63_t) x * y) >> 32)

/* SMMUL */
#define mult_32x32_keep32(a, x, y) \
    a = (q31_t) (((q63_t) x * y ) >> 32)


#if defined ( __CC_ARM )
  /* Enter low optimization region - place directly above function definition */
  #if defined( ARM_MATH_CM4 ) || defined( ARM_MATH_CM7)
    #define LOW_OPTIMIZATION_ENTER \
       _Pragma ("push")         \
       _Pragma ("O1")
  #else
    #define LOW_OPTIMIZATION_ENTER
  #endif

  /* Exit low optimization region - place directly after end of function definition */
  #if defined( ARM_MATH_CM4 ) || defined( ARM_MATH_CM7)
    #define LOW_OPTIMIZATION_EXIT \
       _Pragma ("pop")
  #else
    #define LOW_OPTIMIZATION_EXIT
  #endif

  /* Enter low optimization region - place directly above function definition */
  #define IAR_ONLY_LOW_OPTIMIZATION_ENTER

  /* Exit low optimization region - place directly after end of function definition */
  #define IAR_ONLY_LOW_OPTIMIZATION_EXIT

#elif defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)
  #define LOW_OPTIMIZATION_ENTER
  #define LOW_OPTIMIZATION_EXIT
  #define IAR_ONLY_LOW_OPTIMIZATION_ENTER
  #define IAR_ONLY_LOW_OPTIMIZATION_EXIT

#elif defined(__GNUC__)
  #define LOW_OPTIMIZATION_ENTER __attribute__(( optimize("-O1") ))
  #define LOW_OPTIMIZATION_EXIT
  #define IAR_ONLY_LOW_OPTIMIZATION_ENTER
  #define IAR_ONLY_LOW_OPTIMIZATION_EXIT

#elif defined(__ICCARM__)
  /* Enter low optimization region - place directly above function definition */
  #if defined( ARM_MATH_CM4 ) || defined( ARM_MATH_CM7)
    #define LOW_OPTIMIZATION_ENTER \
       _Pragma ("optimize=low")
  #else
    #define LOW_OPTIMIZATION_ENTER
  #endif

  /* Exit low optimization region - place directly after end of function definition */
  #define LOW_OPTIMIZATION_EXIT

  /* Enter low optimization region - place directly above function definition */
  #if defined( ARM_MATH_CM4 ) || defined( ARM_MATH_CM7)
    #define IAR_ONLY_LOW_OPTIMIZATION_ENTER \
       _Pragma ("optimize=low")
  #else
    #define IAR_ONLY_LOW_OPTIMIZATION_ENTER
  #endif

  /* Exit low optimization region - place directly after end of function definition */
  #define IAR_ONLY_LOW_OPTIMIZATION_EXIT

#elif defined(__CSMC__)
  #define LOW_OPTIMIZATION_ENTER
  #define LOW_OPTIMIZATION_EXIT
  #define IAR_ONLY_LOW_OPTIMIZATION_ENTER
  #define IAR_ONLY_LOW_OPTIMIZATION_EXIT

#elif defined(__TASKING__)
  #define LOW_OPTIMIZATION_ENTER
  #define LOW_OPTIMIZATION_EXIT
  #define IAR_ONLY_LOW_OPTIMIZATION_ENTER
  #define IAR_ONLY_LOW_OPTIMIZATION_EXIT

#endif


#ifdef   __cplusplus
}
#endif


#if defined ( __GNUC__ )
#pragma GCC diagnostic pop
#endif

#endif /* _ARM_MATH_H */

/**
 *
 * End of file.
 */
