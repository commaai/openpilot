/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_ROW_H_
#define INCLUDE_LIBYUV_ROW_H_

#include <stdlib.h>  // For malloc.

#include "libyuv/basic_types.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#define IS_ALIGNED(p, a) (!((uintptr_t)(p) & ((a) - 1)))

#define align_buffer_64(var, size)                                             \
  uint8* var##_mem = (uint8*)(malloc((size) + 63));               /* NOLINT */ \
  uint8* var = (uint8*)(((intptr_t)(var##_mem) + 63) & ~63)       /* NOLINT */

#define free_aligned_buffer_64(var) \
  free(var##_mem);  \
  var = 0

#if defined(__pnacl__) || defined(__CLR_VER) || \
    (defined(__i386__) && !defined(__SSE2__))
#define LIBYUV_DISABLE_X86
#endif
// MemorySanitizer does not support assembly code yet. http://crbug.com/344505
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#define LIBYUV_DISABLE_X86
#endif
#endif
// True if compiling for SSSE3 as a requirement.
#if defined(__SSSE3__) || (defined(_M_IX86_FP) && (_M_IX86_FP >= 3))
#define LIBYUV_SSSE3_ONLY
#endif

#if defined(__native_client__)
#define LIBYUV_DISABLE_NEON
#endif
// clang >= 3.5.0 required for Arm64.
#if defined(__clang__) && defined(__aarch64__) && !defined(LIBYUV_DISABLE_NEON)
#if (__clang_major__ < 3) || (__clang_major__ == 3 && (__clang_minor__ < 5))
#define LIBYUV_DISABLE_NEON
#endif  // clang >= 3.5
#endif  // __clang__

// GCC >= 4.7.0 required for AVX2.
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#if (__GNUC__ > 4) || (__GNUC__ == 4 && (__GNUC_MINOR__ >= 7))
#define GCC_HAS_AVX2 1
#endif  // GNUC >= 4.7
#endif  // __GNUC__

// clang >= 3.4.0 required for AVX2.
#if defined(__clang__) && (defined(__x86_64__) || defined(__i386__))
#if (__clang_major__ > 3) || (__clang_major__ == 3 && (__clang_minor__ >= 4))
#define CLANG_HAS_AVX2 1
#endif  // clang >= 3.4
#endif  // __clang__

// Visual C 2012 required for AVX2.
#if defined(_M_IX86) && !defined(__clang__) && \
    defined(_MSC_VER) && _MSC_VER >= 1700
#define VISUALC_HAS_AVX2 1
#endif  // VisualStudio >= 2012

// The following are available on all x86 platforms:
#if !defined(LIBYUV_DISABLE_X86) && \
    (defined(_M_IX86) || defined(__x86_64__) || defined(__i386__))
// Conversions:
#define HAS_ABGRTOUVROW_SSSE3
#define HAS_ABGRTOYROW_SSSE3
#define HAS_ARGB1555TOARGBROW_SSE2
#define HAS_ARGB4444TOARGBROW_SSE2
#define HAS_ARGBSETROW_X86
#define HAS_ARGBSHUFFLEROW_SSE2
#define HAS_ARGBSHUFFLEROW_SSSE3
#define HAS_ARGBTOARGB1555ROW_SSE2
#define HAS_ARGBTOARGB4444ROW_SSE2
#define HAS_ARGBTORAWROW_SSSE3
#define HAS_ARGBTORGB24ROW_SSSE3
#define HAS_ARGBTORGB565DITHERROW_SSE2
#define HAS_ARGBTORGB565ROW_SSE2
#define HAS_ARGBTOUV444ROW_SSSE3
#define HAS_ARGBTOUVJROW_SSSE3
#define HAS_ARGBTOUVROW_SSSE3
#define HAS_ARGBTOYJROW_SSSE3
#define HAS_ARGBTOYROW_SSSE3
#define HAS_ARGBEXTRACTALPHAROW_SSE2
#define HAS_BGRATOUVROW_SSSE3
#define HAS_BGRATOYROW_SSSE3
#define HAS_COPYROW_ERMS
#define HAS_COPYROW_SSE2
#define HAS_H422TOARGBROW_SSSE3
#define HAS_I400TOARGBROW_SSE2
#define HAS_I422TOARGB1555ROW_SSSE3
#define HAS_I422TOARGB4444ROW_SSSE3
#define HAS_I422TOARGBROW_SSSE3
#define HAS_I422TORGB24ROW_SSSE3
#define HAS_I422TORGB565ROW_SSSE3
#define HAS_I422TORGBAROW_SSSE3
#define HAS_I422TOUYVYROW_SSE2
#define HAS_I422TOYUY2ROW_SSE2
#define HAS_I444TOARGBROW_SSSE3
#define HAS_J400TOARGBROW_SSE2
#define HAS_J422TOARGBROW_SSSE3
#define HAS_MERGEUVROW_SSE2
#define HAS_MIRRORROW_SSSE3
#define HAS_MIRRORUVROW_SSSE3
#define HAS_NV12TOARGBROW_SSSE3
#define HAS_NV12TORGB565ROW_SSSE3
#define HAS_NV21TOARGBROW_SSSE3
#define HAS_RAWTOARGBROW_SSSE3
#define HAS_RAWTORGB24ROW_SSSE3
#define HAS_RAWTOYROW_SSSE3
#define HAS_RGB24TOARGBROW_SSSE3
#define HAS_RGB24TOYROW_SSSE3
#define HAS_RGB565TOARGBROW_SSE2
#define HAS_RGBATOUVROW_SSSE3
#define HAS_RGBATOYROW_SSSE3
#define HAS_SETROW_ERMS
#define HAS_SETROW_X86
#define HAS_SPLITUVROW_SSE2
#define HAS_UYVYTOARGBROW_SSSE3
#define HAS_UYVYTOUV422ROW_SSE2
#define HAS_UYVYTOUVROW_SSE2
#define HAS_UYVYTOYROW_SSE2
#define HAS_YUY2TOARGBROW_SSSE3
#define HAS_YUY2TOUV422ROW_SSE2
#define HAS_YUY2TOUVROW_SSE2
#define HAS_YUY2TOYROW_SSE2

// Effects:
#define HAS_ARGBADDROW_SSE2
#define HAS_ARGBAFFINEROW_SSE2
#define HAS_ARGBATTENUATEROW_SSSE3
#define HAS_ARGBBLENDROW_SSSE3
#define HAS_ARGBCOLORMATRIXROW_SSSE3
#define HAS_ARGBCOLORTABLEROW_X86
#define HAS_ARGBCOPYALPHAROW_SSE2
#define HAS_ARGBCOPYYTOALPHAROW_SSE2
#define HAS_ARGBGRAYROW_SSSE3
#define HAS_ARGBLUMACOLORTABLEROW_SSSE3
#define HAS_ARGBMIRRORROW_SSE2
#define HAS_ARGBMULTIPLYROW_SSE2
#define HAS_ARGBPOLYNOMIALROW_SSE2
#define HAS_ARGBQUANTIZEROW_SSE2
#define HAS_ARGBSEPIAROW_SSSE3
#define HAS_ARGBSHADEROW_SSE2
#define HAS_ARGBSUBTRACTROW_SSE2
#define HAS_ARGBUNATTENUATEROW_SSE2
#define HAS_BLENDPLANEROW_SSSE3
#define HAS_COMPUTECUMULATIVESUMROW_SSE2
#define HAS_CUMULATIVESUMTOAVERAGEROW_SSE2
#define HAS_INTERPOLATEROW_SSSE3
#define HAS_RGBCOLORTABLEROW_X86
#define HAS_SOBELROW_SSE2
#define HAS_SOBELTOPLANEROW_SSE2
#define HAS_SOBELXROW_SSE2
#define HAS_SOBELXYROW_SSE2
#define HAS_SOBELYROW_SSE2

// The following functions fail on gcc/clang 32 bit with fpic and framepointer.
// caveat: clangcl uses row_win.cc which works.
#if defined(NDEBUG) || !(defined(_DEBUG) && defined(__i386__)) || \
    !defined(__i386__) || defined(_MSC_VER)
// TODO(fbarchard): fix build error on x86 debug
// https://code.google.com/p/libyuv/issues/detail?id=524
#define HAS_I411TOARGBROW_SSSE3
// TODO(fbarchard): fix build error on android_full_debug=1
// https://code.google.com/p/libyuv/issues/detail?id=517
#define HAS_I422ALPHATOARGBROW_SSSE3
#endif
#endif

// The following are available on all x86 platforms, but
// require VS2012, clang 3.4 or gcc 4.7.
// The code supports NaCL but requires a new compiler and validator.
#if !defined(LIBYUV_DISABLE_X86) && (defined(VISUALC_HAS_AVX2) || \
    defined(CLANG_HAS_AVX2) || defined(GCC_HAS_AVX2))
#define HAS_ARGBCOPYALPHAROW_AVX2
#define HAS_ARGBCOPYYTOALPHAROW_AVX2
#define HAS_ARGBMIRRORROW_AVX2
#define HAS_ARGBPOLYNOMIALROW_AVX2
#define HAS_ARGBSHUFFLEROW_AVX2
#define HAS_ARGBTORGB565DITHERROW_AVX2
#define HAS_ARGBTOUVJROW_AVX2
#define HAS_ARGBTOUVROW_AVX2
#define HAS_ARGBTOYJROW_AVX2
#define HAS_ARGBTOYROW_AVX2
#define HAS_COPYROW_AVX
#define HAS_H422TOARGBROW_AVX2
#define HAS_I400TOARGBROW_AVX2
#if !(defined(_DEBUG) && defined(__i386__))
// TODO(fbarchard): fix build error on android_full_debug=1
// https://code.google.com/p/libyuv/issues/detail?id=517
#define HAS_I422ALPHATOARGBROW_AVX2
#endif
#define HAS_I411TOARGBROW_AVX2
#define HAS_I422TOARGB1555ROW_AVX2
#define HAS_I422TOARGB4444ROW_AVX2
#define HAS_I422TOARGBROW_AVX2
#define HAS_I422TORGB24ROW_AVX2
#define HAS_I422TORGB565ROW_AVX2
#define HAS_I422TORGBAROW_AVX2
#define HAS_I444TOARGBROW_AVX2
#define HAS_INTERPOLATEROW_AVX2
#define HAS_J422TOARGBROW_AVX2
#define HAS_MERGEUVROW_AVX2
#define HAS_MIRRORROW_AVX2
#define HAS_NV12TOARGBROW_AVX2
#define HAS_NV12TORGB565ROW_AVX2
#define HAS_NV21TOARGBROW_AVX2
#define HAS_SPLITUVROW_AVX2
#define HAS_UYVYTOARGBROW_AVX2
#define HAS_UYVYTOUV422ROW_AVX2
#define HAS_UYVYTOUVROW_AVX2
#define HAS_UYVYTOYROW_AVX2
#define HAS_YUY2TOARGBROW_AVX2
#define HAS_YUY2TOUV422ROW_AVX2
#define HAS_YUY2TOUVROW_AVX2
#define HAS_YUY2TOYROW_AVX2
#define HAS_HALFFLOATROW_AVX2

// Effects:
#define HAS_ARGBADDROW_AVX2
#define HAS_ARGBATTENUATEROW_AVX2
#define HAS_ARGBMULTIPLYROW_AVX2
#define HAS_ARGBSUBTRACTROW_AVX2
#define HAS_ARGBUNATTENUATEROW_AVX2
#define HAS_BLENDPLANEROW_AVX2
#endif

// The following are available for AVX2 Visual C and clangcl 32 bit:
// TODO(fbarchard): Port to gcc.
#if !defined(LIBYUV_DISABLE_X86) && defined(_M_IX86) && \
    (defined(VISUALC_HAS_AVX2) || defined(CLANG_HAS_AVX2))
#define HAS_ARGB1555TOARGBROW_AVX2
#define HAS_ARGB4444TOARGBROW_AVX2
#define HAS_ARGBTOARGB1555ROW_AVX2
#define HAS_ARGBTOARGB4444ROW_AVX2
#define HAS_ARGBTORGB565ROW_AVX2
#define HAS_J400TOARGBROW_AVX2
#define HAS_RGB565TOARGBROW_AVX2
#endif

// The following are also available on x64 Visual C.
#if !defined(LIBYUV_DISABLE_X86) && defined(_MSC_VER) && defined(_M_X64) && \
    (!defined(__clang__) || defined(__SSSE3__))
#define HAS_I422ALPHATOARGBROW_SSSE3
#define HAS_I422TOARGBROW_SSSE3
#endif

// The following are available on gcc x86 platforms:
// TODO(fbarchard): Port to Visual C.
#if !defined(LIBYUV_DISABLE_X86) && \
    (defined(__x86_64__) || (defined(__i386__) && !defined(_MSC_VER)))
#define HAS_HALFFLOATROW_SSE2
#endif

// The following are available on Neon platforms:
#if !defined(LIBYUV_DISABLE_NEON) && \
    (defined(__aarch64__) || defined(__ARM_NEON__) || defined(LIBYUV_NEON))
#define HAS_ABGRTOUVROW_NEON
#define HAS_ABGRTOYROW_NEON
#define HAS_ARGB1555TOARGBROW_NEON
#define HAS_ARGB1555TOUVROW_NEON
#define HAS_ARGB1555TOYROW_NEON
#define HAS_ARGB4444TOARGBROW_NEON
#define HAS_ARGB4444TOUVROW_NEON
#define HAS_ARGB4444TOYROW_NEON
#define HAS_ARGBSETROW_NEON
#define HAS_ARGBTOARGB1555ROW_NEON
#define HAS_ARGBTOARGB4444ROW_NEON
#define HAS_ARGBTORAWROW_NEON
#define HAS_ARGBTORGB24ROW_NEON
#define HAS_ARGBTORGB565DITHERROW_NEON
#define HAS_ARGBTORGB565ROW_NEON
#define HAS_ARGBTOUV411ROW_NEON
#define HAS_ARGBTOUV444ROW_NEON
#define HAS_ARGBTOUVJROW_NEON
#define HAS_ARGBTOUVROW_NEON
#define HAS_ARGBTOYJROW_NEON
#define HAS_ARGBTOYROW_NEON
#define HAS_ARGBEXTRACTALPHAROW_NEON
#define HAS_BGRATOUVROW_NEON
#define HAS_BGRATOYROW_NEON
#define HAS_COPYROW_NEON
#define HAS_I400TOARGBROW_NEON
#define HAS_I411TOARGBROW_NEON
#define HAS_I422ALPHATOARGBROW_NEON
#define HAS_I422TOARGB1555ROW_NEON
#define HAS_I422TOARGB4444ROW_NEON
#define HAS_I422TOARGBROW_NEON
#define HAS_I422TORGB24ROW_NEON
#define HAS_I422TORGB565ROW_NEON
#define HAS_I422TORGBAROW_NEON
#define HAS_I422TOUYVYROW_NEON
#define HAS_I422TOYUY2ROW_NEON
#define HAS_I444TOARGBROW_NEON
#define HAS_J400TOARGBROW_NEON
#define HAS_MERGEUVROW_NEON
#define HAS_MIRRORROW_NEON
#define HAS_MIRRORUVROW_NEON
#define HAS_NV12TOARGBROW_NEON
#define HAS_NV12TORGB565ROW_NEON
#define HAS_NV21TOARGBROW_NEON
#define HAS_RAWTOARGBROW_NEON
#define HAS_RAWTORGB24ROW_NEON
#define HAS_RAWTOUVROW_NEON
#define HAS_RAWTOYROW_NEON
#define HAS_RGB24TOARGBROW_NEON
#define HAS_RGB24TOUVROW_NEON
#define HAS_RGB24TOYROW_NEON
#define HAS_RGB565TOARGBROW_NEON
#define HAS_RGB565TOUVROW_NEON
#define HAS_RGB565TOYROW_NEON
#define HAS_RGBATOUVROW_NEON
#define HAS_RGBATOYROW_NEON
#define HAS_SETROW_NEON
#define HAS_SPLITUVROW_NEON
#define HAS_UYVYTOARGBROW_NEON
#define HAS_UYVYTOUV422ROW_NEON
#define HAS_UYVYTOUVROW_NEON
#define HAS_UYVYTOYROW_NEON
#define HAS_YUY2TOARGBROW_NEON
#define HAS_YUY2TOUV422ROW_NEON
#define HAS_YUY2TOUVROW_NEON
#define HAS_YUY2TOYROW_NEON

// Effects:
#define HAS_ARGBADDROW_NEON
#define HAS_ARGBATTENUATEROW_NEON
#define HAS_ARGBBLENDROW_NEON
#define HAS_ARGBCOLORMATRIXROW_NEON
#define HAS_ARGBGRAYROW_NEON
#define HAS_ARGBMIRRORROW_NEON
#define HAS_ARGBMULTIPLYROW_NEON
#define HAS_ARGBQUANTIZEROW_NEON
#define HAS_ARGBSEPIAROW_NEON
#define HAS_ARGBSHADEROW_NEON
#define HAS_ARGBSHUFFLEROW_NEON
#define HAS_ARGBSUBTRACTROW_NEON
#define HAS_INTERPOLATEROW_NEON
#define HAS_SOBELROW_NEON
#define HAS_SOBELTOPLANEROW_NEON
#define HAS_SOBELXROW_NEON
#define HAS_SOBELXYROW_NEON
#define HAS_SOBELYROW_NEON
#endif

// The following are available on Mips platforms:
#if !defined(LIBYUV_DISABLE_MIPS) && defined(__mips__) && \
    (_MIPS_SIM == _MIPS_SIM_ABI32) && (__mips_isa_rev < 6)
#define HAS_COPYROW_MIPS
#if defined(__mips_dsp) && (__mips_dsp_rev >= 2)
#define HAS_I422TOARGBROW_DSPR2
#define HAS_INTERPOLATEROW_DSPR2
#define HAS_MIRRORROW_DSPR2
#define HAS_MIRRORUVROW_DSPR2
#define HAS_SPLITUVROW_DSPR2
#endif
#endif

#if !defined(LIBYUV_DISABLE_MSA) && defined(__mips_msa)
#define HAS_MIRRORROW_MSA
#define HAS_ARGBMIRRORROW_MSA
#endif

#if defined(_MSC_VER) && !defined(__CLR_VER) && !defined(__clang__)
#if defined(VISUALC_HAS_AVX2)
#define SIMD_ALIGNED(var) __declspec(align(32)) var
#else
#define SIMD_ALIGNED(var) __declspec(align(16)) var
#endif
typedef __declspec(align(16)) int16 vec16[8];
typedef __declspec(align(16)) int32 vec32[4];
typedef __declspec(align(16)) int8 vec8[16];
typedef __declspec(align(16)) uint16 uvec16[8];
typedef __declspec(align(16)) uint32 uvec32[4];
typedef __declspec(align(16)) uint8 uvec8[16];
typedef __declspec(align(32)) int16 lvec16[16];
typedef __declspec(align(32)) int32 lvec32[8];
typedef __declspec(align(32)) int8 lvec8[32];
typedef __declspec(align(32)) uint16 ulvec16[16];
typedef __declspec(align(32)) uint32 ulvec32[8];
typedef __declspec(align(32)) uint8 ulvec8[32];
#elif !defined(__pnacl__) && (defined(__GNUC__) || defined(__clang__))
// Caveat GCC 4.2 to 4.7 have a known issue using vectors with const.
#if defined(CLANG_HAS_AVX2) || defined(GCC_HAS_AVX2)
#define SIMD_ALIGNED(var) var __attribute__((aligned(32)))
#else
#define SIMD_ALIGNED(var) var __attribute__((aligned(16)))
#endif
typedef int16 __attribute__((vector_size(16))) vec16;
typedef int32 __attribute__((vector_size(16))) vec32;
typedef int8 __attribute__((vector_size(16))) vec8;
typedef uint16 __attribute__((vector_size(16))) uvec16;
typedef uint32 __attribute__((vector_size(16))) uvec32;
typedef uint8 __attribute__((vector_size(16))) uvec8;
typedef int16 __attribute__((vector_size(32))) lvec16;
typedef int32 __attribute__((vector_size(32))) lvec32;
typedef int8 __attribute__((vector_size(32))) lvec8;
typedef uint16 __attribute__((vector_size(32))) ulvec16;
typedef uint32 __attribute__((vector_size(32))) ulvec32;
typedef uint8 __attribute__((vector_size(32))) ulvec8;
#else
#define SIMD_ALIGNED(var) var
typedef int16 vec16[8];
typedef int32 vec32[4];
typedef int8 vec8[16];
typedef uint16 uvec16[8];
typedef uint32 uvec32[4];
typedef uint8 uvec8[16];
typedef int16 lvec16[16];
typedef int32 lvec32[8];
typedef int8 lvec8[32];
typedef uint16 ulvec16[16];
typedef uint32 ulvec32[8];
typedef uint8 ulvec8[32];
#endif

#if defined(__aarch64__)
// This struct is for Arm64 color conversion.
struct YuvConstants {
  uvec16 kUVToRB;
  uvec16 kUVToRB2;
  uvec16 kUVToG;
  uvec16 kUVToG2;
  vec16 kUVBiasBGR;
  vec32 kYToRgb;
};
#elif defined(__arm__)
// This struct is for ArmV7 color conversion.
struct YuvConstants {
  uvec8 kUVToRB;
  uvec8 kUVToG;
  vec16 kUVBiasBGR;
  vec32 kYToRgb;
};
#else
// This struct is for Intel color conversion.
struct YuvConstants {
  int8 kUVToB[32];
  int8 kUVToG[32];
  int8 kUVToR[32];
  int16 kUVBiasB[16];
  int16 kUVBiasG[16];
  int16 kUVBiasR[16];
  int16 kYToRgb[16];
};

// Offsets into YuvConstants structure
#define KUVTOB   0
#define KUVTOG   32
#define KUVTOR   64
#define KUVBIASB 96
#define KUVBIASG 128
#define KUVBIASR 160
#define KYTORGB  192
#endif

// Conversion matrix for YUV to RGB
extern const struct YuvConstants SIMD_ALIGNED(kYuvI601Constants);  // BT.601
extern const struct YuvConstants SIMD_ALIGNED(kYuvJPEGConstants);  // JPeg
extern const struct YuvConstants SIMD_ALIGNED(kYuvH709Constants);  // BT.709

// Conversion matrix for YVU to BGR
extern const struct YuvConstants SIMD_ALIGNED(kYvuI601Constants);  // BT.601
extern const struct YuvConstants SIMD_ALIGNED(kYvuJPEGConstants);  // JPeg
extern const struct YuvConstants SIMD_ALIGNED(kYvuH709Constants);  // BT.709

#if defined(__APPLE__) || defined(__x86_64__) || defined(__llvm__)
#define OMITFP
#else
#define OMITFP __attribute__((optimize("omit-frame-pointer")))
#endif

// NaCL macros for GCC x86 and x64.
#if defined(__native_client__)
#define LABELALIGN ".p2align 5\n"
#else
#define LABELALIGN
#endif
#if defined(__native_client__) && defined(__x86_64__)
// r14 is used for MEMOP macros.
#define NACL_R14 "r14",
#define BUNDLELOCK ".bundle_lock\n"
#define BUNDLEUNLOCK ".bundle_unlock\n"
#define MEMACCESS(base) "%%nacl:(%%r15,%q" #base ")"
#define MEMACCESS2(offset, base) "%%nacl:" #offset "(%%r15,%q" #base ")"
#define MEMLEA(offset, base) #offset "(%q" #base ")"
#define MEMLEA3(offset, index, scale) \
    #offset "(,%q" #index "," #scale ")"
#define MEMLEA4(offset, base, index, scale) \
    #offset "(%q" #base ",%q" #index "," #scale ")"
#define MEMMOVESTRING(s, d) "%%nacl:(%q" #s "),%%nacl:(%q" #d "), %%r15"
#define MEMSTORESTRING(reg, d) "%%" #reg ",%%nacl:(%q" #d "), %%r15"
#define MEMOPREG(opcode, offset, base, index, scale, reg) \
    BUNDLELOCK \
    "lea " #offset "(%q" #base ",%q" #index "," #scale "),%%r14d\n" \
    #opcode " (%%r15,%%r14),%%" #reg "\n" \
    BUNDLEUNLOCK
#define MEMOPMEM(opcode, reg, offset, base, index, scale) \
    BUNDLELOCK \
    "lea " #offset "(%q" #base ",%q" #index "," #scale "),%%r14d\n" \
    #opcode " %%" #reg ",(%%r15,%%r14)\n" \
    BUNDLEUNLOCK
#define MEMOPARG(opcode, offset, base, index, scale, arg) \
    BUNDLELOCK \
    "lea " #offset "(%q" #base ",%q" #index "," #scale "),%%r14d\n" \
    #opcode " (%%r15,%%r14),%" #arg "\n" \
    BUNDLEUNLOCK
#define VMEMOPREG(opcode, offset, base, index, scale, reg1, reg2) \
    BUNDLELOCK \
    "lea " #offset "(%q" #base ",%q" #index "," #scale "),%%r14d\n" \
    #opcode " (%%r15,%%r14),%%" #reg1 ",%%" #reg2 "\n" \
    BUNDLEUNLOCK
#define VEXTOPMEM(op, sel, reg, offset, base, index, scale) \
    BUNDLELOCK \
    "lea " #offset "(%q" #base ",%q" #index "," #scale "),%%r14d\n" \
    #op " $" #sel ",%%" #reg ",(%%r15,%%r14)\n" \
    BUNDLEUNLOCK
#else  // defined(__native_client__) && defined(__x86_64__)
#define NACL_R14
#define BUNDLEALIGN
#define MEMACCESS(base) "(%" #base ")"
#define MEMACCESS2(offset, base) #offset "(%" #base ")"
#define MEMLEA(offset, base) #offset "(%" #base ")"
#define MEMLEA3(offset, index, scale) \
    #offset "(,%" #index "," #scale ")"
#define MEMLEA4(offset, base, index, scale) \
    #offset "(%" #base ",%" #index "," #scale ")"
#define MEMMOVESTRING(s, d)
#define MEMSTORESTRING(reg, d)
#define MEMOPREG(opcode, offset, base, index, scale, reg) \
    #opcode " " #offset "(%" #base ",%" #index "," #scale "),%%" #reg "\n"
#define MEMOPMEM(opcode, reg, offset, base, index, scale) \
    #opcode " %%" #reg ","#offset "(%" #base ",%" #index "," #scale ")\n"
#define MEMOPARG(opcode, offset, base, index, scale, arg) \
    #opcode " " #offset "(%" #base ",%" #index "," #scale "),%" #arg "\n"
#define VMEMOPREG(opcode, offset, base, index, scale, reg1, reg2) \
    #opcode " " #offset "(%" #base ",%" #index "," #scale "),%%" #reg1 ",%%" \
    #reg2 "\n"
#define VEXTOPMEM(op, sel, reg, offset, base, index, scale) \
    #op " $" #sel ",%%" #reg ","#offset "(%" #base ",%" #index "," #scale ")\n"
#endif  // defined(__native_client__) && defined(__x86_64__)

#if defined(__arm__) || defined(__aarch64__)
#undef MEMACCESS
#if defined(__native_client__)
#define MEMACCESS(base) ".p2align 3\nbic %" #base ", #0xc0000000\n"
#else
#define MEMACCESS(base)
#endif
#endif

void I444ToARGBRow_NEON(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width);
void I422ToARGBRow_NEON(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width);
void I422AlphaToARGBRow_NEON(const uint8* y_buf,
                             const uint8* u_buf,
                             const uint8* v_buf,
                             const uint8* a_buf,
                             uint8* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width);
void I422ToARGBRow_NEON(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width);
void I411ToARGBRow_NEON(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width);
void I422ToRGBARow_NEON(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_rgba,
                        const struct YuvConstants* yuvconstants,
                        int width);
void I422ToRGB24Row_NEON(const uint8* src_y,
                         const uint8* src_u,
                         const uint8* src_v,
                         uint8* dst_rgb24,
                         const struct YuvConstants* yuvconstants,
                         int width);
void I422ToRGB565Row_NEON(const uint8* src_y,
                          const uint8* src_u,
                          const uint8* src_v,
                          uint8* dst_rgb565,
                          const struct YuvConstants* yuvconstants,
                          int width);
void I422ToARGB1555Row_NEON(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_argb1555,
                            const struct YuvConstants* yuvconstants,
                            int width);
void I422ToARGB4444Row_NEON(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_argb4444,
                            const struct YuvConstants* yuvconstants,
                            int width);
void NV12ToARGBRow_NEON(const uint8* src_y,
                        const uint8* src_uv,
                        uint8* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width);
void NV12ToRGB565Row_NEON(const uint8* src_y,
                          const uint8* src_uv,
                          uint8* dst_rgb565,
                          const struct YuvConstants* yuvconstants,
                          int width);
void NV21ToARGBRow_NEON(const uint8* src_y,
                        const uint8* src_vu,
                        uint8* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width);
void YUY2ToARGBRow_NEON(const uint8* src_yuy2,
                        uint8* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width);
void UYVYToARGBRow_NEON(const uint8* src_uyvy,
                        uint8* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width);

void ARGBToYRow_AVX2(const uint8* src_argb, uint8* dst_y, int width);
void ARGBToYRow_Any_AVX2(const uint8* src_argb, uint8* dst_y, int width);
void ARGBToYRow_SSSE3(const uint8* src_argb, uint8* dst_y, int width);
void ARGBToYJRow_AVX2(const uint8* src_argb, uint8* dst_y, int width);
void ARGBToYJRow_Any_AVX2(const uint8* src_argb, uint8* dst_y, int width);
void ARGBToYJRow_SSSE3(const uint8* src_argb, uint8* dst_y, int width);
void BGRAToYRow_SSSE3(const uint8* src_bgra, uint8* dst_y, int width);
void ABGRToYRow_SSSE3(const uint8* src_abgr, uint8* dst_y, int width);
void RGBAToYRow_SSSE3(const uint8* src_rgba, uint8* dst_y, int width);
void RGB24ToYRow_SSSE3(const uint8* src_rgb24, uint8* dst_y, int width);
void RAWToYRow_SSSE3(const uint8* src_raw, uint8* dst_y, int width);
void ARGBToYRow_NEON(const uint8* src_argb, uint8* dst_y, int width);
void ARGBToYJRow_NEON(const uint8* src_argb, uint8* dst_y, int width);
void ARGBToUV444Row_NEON(const uint8* src_argb, uint8* dst_u, uint8* dst_v,
                         int width);
void ARGBToUV411Row_NEON(const uint8* src_argb, uint8* dst_u, uint8* dst_v,
                         int width);
void ARGBToUVRow_NEON(const uint8* src_argb, int src_stride_argb,
                      uint8* dst_u, uint8* dst_v, int width);
void ARGBToUVJRow_NEON(const uint8* src_argb, int src_stride_argb,
                       uint8* dst_u, uint8* dst_v, int width);
void BGRAToUVRow_NEON(const uint8* src_bgra, int src_stride_bgra,
                      uint8* dst_u, uint8* dst_v, int width);
void ABGRToUVRow_NEON(const uint8* src_abgr, int src_stride_abgr,
                      uint8* dst_u, uint8* dst_v, int width);
void RGBAToUVRow_NEON(const uint8* src_rgba, int src_stride_rgba,
                      uint8* dst_u, uint8* dst_v, int width);
void RGB24ToUVRow_NEON(const uint8* src_rgb24, int src_stride_rgb24,
                       uint8* dst_u, uint8* dst_v, int width);
void RAWToUVRow_NEON(const uint8* src_raw, int src_stride_raw,
                     uint8* dst_u, uint8* dst_v, int width);
void RGB565ToUVRow_NEON(const uint8* src_rgb565, int src_stride_rgb565,
                        uint8* dst_u, uint8* dst_v, int width);
void ARGB1555ToUVRow_NEON(const uint8* src_argb1555, int src_stride_argb1555,
                          uint8* dst_u, uint8* dst_v, int width);
void ARGB4444ToUVRow_NEON(const uint8* src_argb4444, int src_stride_argb4444,
                          uint8* dst_u, uint8* dst_v, int width);
void BGRAToYRow_NEON(const uint8* src_bgra, uint8* dst_y, int width);
void ABGRToYRow_NEON(const uint8* src_abgr, uint8* dst_y, int width);
void RGBAToYRow_NEON(const uint8* src_rgba, uint8* dst_y, int width);
void RGB24ToYRow_NEON(const uint8* src_rgb24, uint8* dst_y, int width);
void RAWToYRow_NEON(const uint8* src_raw, uint8* dst_y, int width);
void RGB565ToYRow_NEON(const uint8* src_rgb565, uint8* dst_y, int width);
void ARGB1555ToYRow_NEON(const uint8* src_argb1555, uint8* dst_y, int width);
void ARGB4444ToYRow_NEON(const uint8* src_argb4444, uint8* dst_y, int width);
void ARGBToYRow_C(const uint8* src_argb, uint8* dst_y, int width);
void ARGBToYJRow_C(const uint8* src_argb, uint8* dst_y, int width);
void BGRAToYRow_C(const uint8* src_bgra, uint8* dst_y, int width);
void ABGRToYRow_C(const uint8* src_abgr, uint8* dst_y, int width);
void RGBAToYRow_C(const uint8* src_rgba, uint8* dst_y, int width);
void RGB24ToYRow_C(const uint8* src_rgb24, uint8* dst_y, int width);
void RAWToYRow_C(const uint8* src_raw, uint8* dst_y, int width);
void RGB565ToYRow_C(const uint8* src_rgb565, uint8* dst_y, int width);
void ARGB1555ToYRow_C(const uint8* src_argb1555, uint8* dst_y, int width);
void ARGB4444ToYRow_C(const uint8* src_argb4444, uint8* dst_y, int width);
void ARGBToYRow_Any_SSSE3(const uint8* src_argb, uint8* dst_y, int width);
void ARGBToYJRow_Any_SSSE3(const uint8* src_argb, uint8* dst_y, int width);
void BGRAToYRow_Any_SSSE3(const uint8* src_bgra, uint8* dst_y, int width);
void ABGRToYRow_Any_SSSE3(const uint8* src_abgr, uint8* dst_y, int width);
void RGBAToYRow_Any_SSSE3(const uint8* src_rgba, uint8* dst_y, int width);
void RGB24ToYRow_Any_SSSE3(const uint8* src_rgb24, uint8* dst_y, int width);
void RAWToYRow_Any_SSSE3(const uint8* src_raw, uint8* dst_y, int width);
void ARGBToYRow_Any_NEON(const uint8* src_argb, uint8* dst_y, int width);
void ARGBToYJRow_Any_NEON(const uint8* src_argb, uint8* dst_y, int width);
void BGRAToYRow_Any_NEON(const uint8* src_bgra, uint8* dst_y, int width);
void ABGRToYRow_Any_NEON(const uint8* src_abgr, uint8* dst_y, int width);
void RGBAToYRow_Any_NEON(const uint8* src_rgba, uint8* dst_y, int width);
void RGB24ToYRow_Any_NEON(const uint8* src_rgb24, uint8* dst_y, int width);
void RAWToYRow_Any_NEON(const uint8* src_raw, uint8* dst_y, int width);
void RGB565ToYRow_Any_NEON(const uint8* src_rgb565, uint8* dst_y, int width);
void ARGB1555ToYRow_Any_NEON(const uint8* src_argb1555, uint8* dst_y,
                             int width);
void ARGB4444ToYRow_Any_NEON(const uint8* src_argb4444, uint8* dst_y,
                             int width);

void ARGBToUVRow_AVX2(const uint8* src_argb, int src_stride_argb,
                      uint8* dst_u, uint8* dst_v, int width);
void ARGBToUVJRow_AVX2(const uint8* src_argb, int src_stride_argb,
                       uint8* dst_u, uint8* dst_v, int width);
void ARGBToUVRow_SSSE3(const uint8* src_argb, int src_stride_argb,
                       uint8* dst_u, uint8* dst_v, int width);
void ARGBToUVJRow_SSSE3(const uint8* src_argb, int src_stride_argb,
                        uint8* dst_u, uint8* dst_v, int width);
void BGRAToUVRow_SSSE3(const uint8* src_bgra, int src_stride_bgra,
                       uint8* dst_u, uint8* dst_v, int width);
void ABGRToUVRow_SSSE3(const uint8* src_abgr, int src_stride_abgr,
                       uint8* dst_u, uint8* dst_v, int width);
void RGBAToUVRow_SSSE3(const uint8* src_rgba, int src_stride_rgba,
                       uint8* dst_u, uint8* dst_v, int width);
void ARGBToUVRow_Any_AVX2(const uint8* src_argb, int src_stride_argb,
                          uint8* dst_u, uint8* dst_v, int width);
void ARGBToUVJRow_Any_AVX2(const uint8* src_argb, int src_stride_argb,
                           uint8* dst_u, uint8* dst_v, int width);
void ARGBToUVRow_Any_SSSE3(const uint8* src_argb, int src_stride_argb,
                           uint8* dst_u, uint8* dst_v, int width);
void ARGBToUVJRow_Any_SSSE3(const uint8* src_argb, int src_stride_argb,
                            uint8* dst_u, uint8* dst_v, int width);
void BGRAToUVRow_Any_SSSE3(const uint8* src_bgra, int src_stride_bgra,
                           uint8* dst_u, uint8* dst_v, int width);
void ABGRToUVRow_Any_SSSE3(const uint8* src_abgr, int src_stride_abgr,
                           uint8* dst_u, uint8* dst_v, int width);
void RGBAToUVRow_Any_SSSE3(const uint8* src_rgba, int src_stride_rgba,
                           uint8* dst_u, uint8* dst_v, int width);
void ARGBToUV444Row_Any_NEON(const uint8* src_argb, uint8* dst_u, uint8* dst_v,
                             int width);
void ARGBToUV411Row_Any_NEON(const uint8* src_argb, uint8* dst_u, uint8* dst_v,
                             int width);
void ARGBToUVRow_Any_NEON(const uint8* src_argb, int src_stride_argb,
                          uint8* dst_u, uint8* dst_v, int width);
void ARGBToUVJRow_Any_NEON(const uint8* src_argb, int src_stride_argb,
                           uint8* dst_u, uint8* dst_v, int width);
void BGRAToUVRow_Any_NEON(const uint8* src_bgra, int src_stride_bgra,
                          uint8* dst_u, uint8* dst_v, int width);
void ABGRToUVRow_Any_NEON(const uint8* src_abgr, int src_stride_abgr,
                          uint8* dst_u, uint8* dst_v, int width);
void RGBAToUVRow_Any_NEON(const uint8* src_rgba, int src_stride_rgba,
                          uint8* dst_u, uint8* dst_v, int width);
void RGB24ToUVRow_Any_NEON(const uint8* src_rgb24, int src_stride_rgb24,
                           uint8* dst_u, uint8* dst_v, int width);
void RAWToUVRow_Any_NEON(const uint8* src_raw, int src_stride_raw,
                         uint8* dst_u, uint8* dst_v, int width);
void RGB565ToUVRow_Any_NEON(const uint8* src_rgb565, int src_stride_rgb565,
                            uint8* dst_u, uint8* dst_v, int width);
void ARGB1555ToUVRow_Any_NEON(const uint8* src_argb1555,
                              int src_stride_argb1555,
                              uint8* dst_u, uint8* dst_v, int width);
void ARGB4444ToUVRow_Any_NEON(const uint8* src_argb4444,
                              int src_stride_argb4444,
                              uint8* dst_u, uint8* dst_v, int width);
void ARGBToUVRow_C(const uint8* src_argb, int src_stride_argb,
                   uint8* dst_u, uint8* dst_v, int width);
void ARGBToUVJRow_C(const uint8* src_argb, int src_stride_argb,
                    uint8* dst_u, uint8* dst_v, int width);
void BGRAToUVRow_C(const uint8* src_bgra, int src_stride_bgra,
                   uint8* dst_u, uint8* dst_v, int width);
void ABGRToUVRow_C(const uint8* src_abgr, int src_stride_abgr,
                   uint8* dst_u, uint8* dst_v, int width);
void RGBAToUVRow_C(const uint8* src_rgba, int src_stride_rgba,
                   uint8* dst_u, uint8* dst_v, int width);
void RGB24ToUVRow_C(const uint8* src_rgb24, int src_stride_rgb24,
                    uint8* dst_u, uint8* dst_v, int width);
void RAWToUVRow_C(const uint8* src_raw, int src_stride_raw,
                  uint8* dst_u, uint8* dst_v, int width);
void RGB565ToUVRow_C(const uint8* src_rgb565, int src_stride_rgb565,
                     uint8* dst_u, uint8* dst_v, int width);
void ARGB1555ToUVRow_C(const uint8* src_argb1555, int src_stride_argb1555,
                       uint8* dst_u, uint8* dst_v, int width);
void ARGB4444ToUVRow_C(const uint8* src_argb4444, int src_stride_argb4444,
                       uint8* dst_u, uint8* dst_v, int width);

void ARGBToUV444Row_SSSE3(const uint8* src_argb,
                          uint8* dst_u, uint8* dst_v, int width);
void ARGBToUV444Row_Any_SSSE3(const uint8* src_argb,
                              uint8* dst_u, uint8* dst_v, int width);

void ARGBToUV444Row_C(const uint8* src_argb,
                      uint8* dst_u, uint8* dst_v, int width);
void ARGBToUV411Row_C(const uint8* src_argb,
                      uint8* dst_u, uint8* dst_v, int width);

void MirrorRow_AVX2(const uint8* src, uint8* dst, int width);
void MirrorRow_SSSE3(const uint8* src, uint8* dst, int width);
void MirrorRow_NEON(const uint8* src, uint8* dst, int width);
void MirrorRow_DSPR2(const uint8* src, uint8* dst, int width);
void MirrorRow_MSA(const uint8* src, uint8* dst, int width);
void MirrorRow_C(const uint8* src, uint8* dst, int width);
void MirrorRow_Any_AVX2(const uint8* src, uint8* dst, int width);
void MirrorRow_Any_SSSE3(const uint8* src, uint8* dst, int width);
void MirrorRow_Any_SSE2(const uint8* src, uint8* dst, int width);
void MirrorRow_Any_NEON(const uint8* src, uint8* dst, int width);
void MirrorRow_Any_MSA(const uint8* src, uint8* dst, int width);

void MirrorUVRow_SSSE3(const uint8* src_uv, uint8* dst_u, uint8* dst_v,
                       int width);
void MirrorUVRow_NEON(const uint8* src_uv, uint8* dst_u, uint8* dst_v,
                      int width);
void MirrorUVRow_DSPR2(const uint8* src_uv, uint8* dst_u, uint8* dst_v,
                       int width);
void MirrorUVRow_C(const uint8* src_uv, uint8* dst_u, uint8* dst_v, int width);

void ARGBMirrorRow_AVX2(const uint8* src, uint8* dst, int width);
void ARGBMirrorRow_SSE2(const uint8* src, uint8* dst, int width);
void ARGBMirrorRow_NEON(const uint8* src, uint8* dst, int width);
void ARGBMirrorRow_MSA(const uint8* src, uint8* dst, int width);
void ARGBMirrorRow_C(const uint8* src, uint8* dst, int width);
void ARGBMirrorRow_Any_AVX2(const uint8* src, uint8* dst, int width);
void ARGBMirrorRow_Any_SSE2(const uint8* src, uint8* dst, int width);
void ARGBMirrorRow_Any_NEON(const uint8* src, uint8* dst, int width);
void ARGBMirrorRow_Any_MSA(const uint8* src, uint8* dst, int width);

void SplitUVRow_C(const uint8* src_uv, uint8* dst_u, uint8* dst_v, int width);
void SplitUVRow_SSE2(const uint8* src_uv, uint8* dst_u, uint8* dst_v,
                     int width);
void SplitUVRow_AVX2(const uint8* src_uv, uint8* dst_u, uint8* dst_v,
                     int width);
void SplitUVRow_NEON(const uint8* src_uv, uint8* dst_u, uint8* dst_v,
                     int width);
void SplitUVRow_DSPR2(const uint8* src_uv, uint8* dst_u, uint8* dst_v,
                      int width);
void SplitUVRow_Any_SSE2(const uint8* src_uv, uint8* dst_u, uint8* dst_v,
                         int width);
void SplitUVRow_Any_AVX2(const uint8* src_uv, uint8* dst_u, uint8* dst_v,
                         int width);
void SplitUVRow_Any_NEON(const uint8* src_uv, uint8* dst_u, uint8* dst_v,
                         int width);
void SplitUVRow_Any_DSPR2(const uint8* src_uv, uint8* dst_u, uint8* dst_v,
                          int width);

void MergeUVRow_C(const uint8* src_u, const uint8* src_v, uint8* dst_uv,
                  int width);
void MergeUVRow_SSE2(const uint8* src_u, const uint8* src_v, uint8* dst_uv,
                     int width);
void MergeUVRow_AVX2(const uint8* src_u, const uint8* src_v, uint8* dst_uv,
                     int width);
void MergeUVRow_NEON(const uint8* src_u, const uint8* src_v, uint8* dst_uv,
                     int width);
void MergeUVRow_Any_SSE2(const uint8* src_u, const uint8* src_v, uint8* dst_uv,
                         int width);
void MergeUVRow_Any_AVX2(const uint8* src_u, const uint8* src_v, uint8* dst_uv,
                         int width);
void MergeUVRow_Any_NEON(const uint8* src_u, const uint8* src_v, uint8* dst_uv,
                         int width);

void CopyRow_SSE2(const uint8* src, uint8* dst, int count);
void CopyRow_AVX(const uint8* src, uint8* dst, int count);
void CopyRow_ERMS(const uint8* src, uint8* dst, int count);
void CopyRow_NEON(const uint8* src, uint8* dst, int count);
void CopyRow_MIPS(const uint8* src, uint8* dst, int count);
void CopyRow_C(const uint8* src, uint8* dst, int count);
void CopyRow_Any_SSE2(const uint8* src, uint8* dst, int count);
void CopyRow_Any_AVX(const uint8* src, uint8* dst, int count);
void CopyRow_Any_NEON(const uint8* src, uint8* dst, int count);

void CopyRow_16_C(const uint16* src, uint16* dst, int count);

void ARGBCopyAlphaRow_C(const uint8* src_argb, uint8* dst_argb, int width);
void ARGBCopyAlphaRow_SSE2(const uint8* src_argb, uint8* dst_argb, int width);
void ARGBCopyAlphaRow_AVX2(const uint8* src_argb, uint8* dst_argb, int width);
void ARGBCopyAlphaRow_Any_SSE2(const uint8* src_argb, uint8* dst_argb,
                               int width);
void ARGBCopyAlphaRow_Any_AVX2(const uint8* src_argb, uint8* dst_argb,
                               int width);

void ARGBExtractAlphaRow_C(const uint8* src_argb, uint8* dst_a, int width);
void ARGBExtractAlphaRow_SSE2(const uint8* src_argb, uint8* dst_a, int width);
void ARGBExtractAlphaRow_NEON(const uint8* src_argb, uint8* dst_a, int width);
void ARGBExtractAlphaRow_Any_SSE2(const uint8* src_argb, uint8* dst_a,
                                  int width);
void ARGBExtractAlphaRow_Any_NEON(const uint8* src_argb, uint8* dst_a,
                                  int width);

void ARGBCopyYToAlphaRow_C(const uint8* src_y, uint8* dst_argb, int width);
void ARGBCopyYToAlphaRow_SSE2(const uint8* src_y, uint8* dst_argb, int width);
void ARGBCopyYToAlphaRow_AVX2(const uint8* src_y, uint8* dst_argb, int width);
void ARGBCopyYToAlphaRow_Any_SSE2(const uint8* src_y, uint8* dst_argb,
                                  int width);
void ARGBCopyYToAlphaRow_Any_AVX2(const uint8* src_y, uint8* dst_argb,
                                  int width);

void SetRow_C(uint8* dst, uint8 v8, int count);
void SetRow_X86(uint8* dst, uint8 v8, int count);
void SetRow_ERMS(uint8* dst, uint8 v8, int count);
void SetRow_NEON(uint8* dst, uint8 v8, int count);
void SetRow_Any_X86(uint8* dst, uint8 v8, int count);
void SetRow_Any_NEON(uint8* dst, uint8 v8, int count);

void ARGBSetRow_C(uint8* dst_argb, uint32 v32, int count);
void ARGBSetRow_X86(uint8* dst_argb, uint32 v32, int count);
void ARGBSetRow_NEON(uint8* dst_argb, uint32 v32, int count);
void ARGBSetRow_Any_NEON(uint8* dst_argb, uint32 v32, int count);

// ARGBShufflers for BGRAToARGB etc.
void ARGBShuffleRow_C(const uint8* src_argb, uint8* dst_argb,
                      const uint8* shuffler, int width);
void ARGBShuffleRow_SSE2(const uint8* src_argb, uint8* dst_argb,
                         const uint8* shuffler, int width);
void ARGBShuffleRow_SSSE3(const uint8* src_argb, uint8* dst_argb,
                          const uint8* shuffler, int width);
void ARGBShuffleRow_AVX2(const uint8* src_argb, uint8* dst_argb,
                         const uint8* shuffler, int width);
void ARGBShuffleRow_NEON(const uint8* src_argb, uint8* dst_argb,
                         const uint8* shuffler, int width);
void ARGBShuffleRow_Any_SSE2(const uint8* src_argb, uint8* dst_argb,
                             const uint8* shuffler, int width);
void ARGBShuffleRow_Any_SSSE3(const uint8* src_argb, uint8* dst_argb,
                              const uint8* shuffler, int width);
void ARGBShuffleRow_Any_AVX2(const uint8* src_argb, uint8* dst_argb,
                             const uint8* shuffler, int width);
void ARGBShuffleRow_Any_NEON(const uint8* src_argb, uint8* dst_argb,
                             const uint8* shuffler, int width);

void RGB24ToARGBRow_SSSE3(const uint8* src_rgb24, uint8* dst_argb, int width);
void RAWToARGBRow_SSSE3(const uint8* src_raw, uint8* dst_argb, int width);
void RAWToRGB24Row_SSSE3(const uint8* src_raw, uint8* dst_rgb24, int width);
void RGB565ToARGBRow_SSE2(const uint8* src_rgb565, uint8* dst_argb, int width);
void ARGB1555ToARGBRow_SSE2(const uint8* src_argb1555, uint8* dst_argb,
                            int width);
void ARGB4444ToARGBRow_SSE2(const uint8* src_argb4444, uint8* dst_argb,
                            int width);
void RGB565ToARGBRow_AVX2(const uint8* src_rgb565, uint8* dst_argb, int width);
void ARGB1555ToARGBRow_AVX2(const uint8* src_argb1555, uint8* dst_argb,
                            int width);
void ARGB4444ToARGBRow_AVX2(const uint8* src_argb4444, uint8* dst_argb,
                            int width);

void RGB24ToARGBRow_NEON(const uint8* src_rgb24, uint8* dst_argb, int width);
void RAWToARGBRow_NEON(const uint8* src_raw, uint8* dst_argb, int width);
void RAWToRGB24Row_NEON(const uint8* src_raw, uint8* dst_rgb24, int width);
void RGB565ToARGBRow_NEON(const uint8* src_rgb565, uint8* dst_argb, int width);
void ARGB1555ToARGBRow_NEON(const uint8* src_argb1555, uint8* dst_argb,
                            int width);
void ARGB4444ToARGBRow_NEON(const uint8* src_argb4444, uint8* dst_argb,
                            int width);
void RGB24ToARGBRow_C(const uint8* src_rgb24, uint8* dst_argb, int width);
void RAWToARGBRow_C(const uint8* src_raw, uint8* dst_argb, int width);
void RAWToRGB24Row_C(const uint8* src_raw, uint8* dst_rgb24, int width);
void RGB565ToARGBRow_C(const uint8* src_rgb, uint8* dst_argb, int width);
void ARGB1555ToARGBRow_C(const uint8* src_argb, uint8* dst_argb, int width);
void ARGB4444ToARGBRow_C(const uint8* src_argb, uint8* dst_argb, int width);
void RGB24ToARGBRow_Any_SSSE3(const uint8* src_rgb24, uint8* dst_argb,
                              int width);
void RAWToARGBRow_Any_SSSE3(const uint8* src_raw, uint8* dst_argb, int width);
void RAWToRGB24Row_Any_SSSE3(const uint8* src_raw, uint8* dst_rgb24, int width);

void RGB565ToARGBRow_Any_SSE2(const uint8* src_rgb565, uint8* dst_argb,
                              int width);
void ARGB1555ToARGBRow_Any_SSE2(const uint8* src_argb1555, uint8* dst_argb,
                                int width);
void ARGB4444ToARGBRow_Any_SSE2(const uint8* src_argb4444, uint8* dst_argb,
                                int width);
void RGB565ToARGBRow_Any_AVX2(const uint8* src_rgb565, uint8* dst_argb,
                              int width);
void ARGB1555ToARGBRow_Any_AVX2(const uint8* src_argb1555, uint8* dst_argb,
                                int width);
void ARGB4444ToARGBRow_Any_AVX2(const uint8* src_argb4444, uint8* dst_argb,
                                int width);

void RGB24ToARGBRow_Any_NEON(const uint8* src_rgb24, uint8* dst_argb,
                             int width);
void RAWToARGBRow_Any_NEON(const uint8* src_raw, uint8* dst_argb, int width);
void RAWToRGB24Row_Any_NEON(const uint8* src_raw, uint8* dst_rgb24, int width);
void RGB565ToARGBRow_Any_NEON(const uint8* src_rgb565, uint8* dst_argb,
                              int width);
void ARGB1555ToARGBRow_Any_NEON(const uint8* src_argb1555, uint8* dst_argb,
                                int width);
void ARGB4444ToARGBRow_Any_NEON(const uint8* src_argb4444, uint8* dst_argb,
                                int width);

void ARGBToRGB24Row_SSSE3(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToRAWRow_SSSE3(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToRGB565Row_SSE2(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToARGB1555Row_SSE2(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToARGB4444Row_SSE2(const uint8* src_argb, uint8* dst_rgb, int width);

void ARGBToRGB565DitherRow_C(const uint8* src_argb, uint8* dst_rgb,
                             const uint32 dither4, int width);
void ARGBToRGB565DitherRow_SSE2(const uint8* src_argb, uint8* dst_rgb,
                                const uint32 dither4, int width);
void ARGBToRGB565DitherRow_AVX2(const uint8* src_argb, uint8* dst_rgb,
                                const uint32 dither4, int width);

void ARGBToRGB565Row_AVX2(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToARGB1555Row_AVX2(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToARGB4444Row_AVX2(const uint8* src_argb, uint8* dst_rgb, int width);

void ARGBToRGB24Row_NEON(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToRAWRow_NEON(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToRGB565Row_NEON(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToARGB1555Row_NEON(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToARGB4444Row_NEON(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToRGB565DitherRow_NEON(const uint8* src_argb, uint8* dst_rgb,
                                const uint32 dither4, int width);

void ARGBToRGBARow_C(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToRGB24Row_C(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToRAWRow_C(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToRGB565Row_C(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToARGB1555Row_C(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToARGB4444Row_C(const uint8* src_argb, uint8* dst_rgb, int width);

void J400ToARGBRow_SSE2(const uint8* src_y, uint8* dst_argb, int width);
void J400ToARGBRow_AVX2(const uint8* src_y, uint8* dst_argb, int width);
void J400ToARGBRow_NEON(const uint8* src_y, uint8* dst_argb, int width);
void J400ToARGBRow_C(const uint8* src_y, uint8* dst_argb, int width);
void J400ToARGBRow_Any_SSE2(const uint8* src_y, uint8* dst_argb, int width);
void J400ToARGBRow_Any_AVX2(const uint8* src_y, uint8* dst_argb, int width);
void J400ToARGBRow_Any_NEON(const uint8* src_y, uint8* dst_argb, int width);

void I444ToARGBRow_C(const uint8* src_y,
                     const uint8* src_u,
                     const uint8* src_v,
                     uint8* dst_argb,
                     const struct YuvConstants* yuvconstants,
                     int width);
void I422ToARGBRow_C(const uint8* src_y,
                     const uint8* src_u,
                     const uint8* src_v,
                     uint8* dst_argb,
                     const struct YuvConstants* yuvconstants,
                     int width);
void I422ToARGBRow_C(const uint8* src_y,
                     const uint8* src_u,
                     const uint8* src_v,
                     uint8* dst_argb,
                     const struct YuvConstants* yuvconstants,
                     int width);
void I422AlphaToARGBRow_C(const uint8* y_buf,
                          const uint8* u_buf,
                          const uint8* v_buf,
                          const uint8* a_buf,
                          uint8* dst_argb,
                          const struct YuvConstants* yuvconstants,
                          int width);
void I411ToARGBRow_C(const uint8* src_y,
                     const uint8* src_u,
                     const uint8* src_v,
                     uint8* dst_argb,
                     const struct YuvConstants* yuvconstants,
                     int width);
void NV12ToARGBRow_C(const uint8* src_y,
                     const uint8* src_uv,
                     uint8* dst_argb,
                     const struct YuvConstants* yuvconstants,
                     int width);
void NV12ToRGB565Row_C(const uint8* src_y,
                       const uint8* src_uv,
                       uint8* dst_argb,
                       const struct YuvConstants* yuvconstants,
                       int width);
void NV21ToARGBRow_C(const uint8* src_y,
                     const uint8* src_uv,
                     uint8* dst_argb,
                     const struct YuvConstants* yuvconstants,
                     int width);
void YUY2ToARGBRow_C(const uint8* src_yuy2,
                     uint8* dst_argb,
                     const struct YuvConstants* yuvconstants,
                     int width);
void UYVYToARGBRow_C(const uint8* src_uyvy,
                     uint8* dst_argb,
                     const struct YuvConstants* yuvconstants,
                     int width);
void I422ToRGBARow_C(const uint8* src_y,
                     const uint8* src_u,
                     const uint8* src_v,
                     uint8* dst_rgba,
                     const struct YuvConstants* yuvconstants,
                     int width);
void I422ToRGB24Row_C(const uint8* src_y,
                      const uint8* src_u,
                      const uint8* src_v,
                      uint8* dst_rgb24,
                      const struct YuvConstants* yuvconstants,
                      int width);
void I422ToARGB4444Row_C(const uint8* src_y,
                         const uint8* src_u,
                         const uint8* src_v,
                         uint8* dst_argb4444,
                         const struct YuvConstants* yuvconstants,
                         int width);
void I422ToARGB1555Row_C(const uint8* src_y,
                         const uint8* src_u,
                         const uint8* src_v,
                         uint8* dst_argb4444,
                         const struct YuvConstants* yuvconstants,
                         int width);
void I422ToRGB565Row_C(const uint8* src_y,
                       const uint8* src_u,
                       const uint8* src_v,
                       uint8* dst_rgb565,
                       const struct YuvConstants* yuvconstants,
                       int width);
void I422ToARGBRow_AVX2(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width);
void I422ToARGBRow_AVX2(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width);
void I422ToRGBARow_AVX2(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width);
void I444ToARGBRow_SSSE3(const uint8* src_y,
                         const uint8* src_u,
                         const uint8* src_v,
                         uint8* dst_argb,
                         const struct YuvConstants* yuvconstants,
                         int width);
void I444ToARGBRow_AVX2(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width);
void I444ToARGBRow_SSSE3(const uint8* src_y,
                         const uint8* src_u,
                         const uint8* src_v,
                         uint8* dst_argb,
                         const struct YuvConstants* yuvconstants,
                         int width);
void I444ToARGBRow_AVX2(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width);
void I422ToARGBRow_SSSE3(const uint8* src_y,
                         const uint8* src_u,
                         const uint8* src_v,
                         uint8* dst_argb,
                         const struct YuvConstants* yuvconstants,
                         int width);
void I422AlphaToARGBRow_SSSE3(const uint8* y_buf,
                              const uint8* u_buf,
                              const uint8* v_buf,
                              const uint8* a_buf,
                              uint8* dst_argb,
                              const struct YuvConstants* yuvconstants,
                              int width);
void I422AlphaToARGBRow_AVX2(const uint8* y_buf,
                             const uint8* u_buf,
                             const uint8* v_buf,
                             const uint8* a_buf,
                             uint8* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width);
void I422ToARGBRow_SSSE3(const uint8* src_y,
                         const uint8* src_u,
                         const uint8* src_v,
                         uint8* dst_argb,
                         const struct YuvConstants* yuvconstants,
                         int width);
void I411ToARGBRow_SSSE3(const uint8* src_y,
                         const uint8* src_u,
                         const uint8* src_v,
                         uint8* dst_argb,
                         const struct YuvConstants* yuvconstants,
                         int width);
void I411ToARGBRow_AVX2(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width);
void NV12ToARGBRow_SSSE3(const uint8* src_y,
                         const uint8* src_uv,
                         uint8* dst_argb,
                         const struct YuvConstants* yuvconstants,
                         int width);
void NV12ToARGBRow_AVX2(const uint8* src_y,
                        const uint8* src_uv,
                        uint8* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width);
void NV12ToRGB565Row_SSSE3(const uint8* src_y,
                           const uint8* src_uv,
                           uint8* dst_argb,
                           const struct YuvConstants* yuvconstants,
                           int width);
void NV12ToRGB565Row_AVX2(const uint8* src_y,
                          const uint8* src_uv,
                          uint8* dst_argb,
                          const struct YuvConstants* yuvconstants,
                          int width);
void NV21ToARGBRow_SSSE3(const uint8* src_y,
                         const uint8* src_uv,
                         uint8* dst_argb,
                         const struct YuvConstants* yuvconstants,
                         int width);
void NV21ToARGBRow_AVX2(const uint8* src_y,
                        const uint8* src_uv,
                        uint8* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width);
void YUY2ToARGBRow_SSSE3(const uint8* src_yuy2,
                         uint8* dst_argb,
                         const struct YuvConstants* yuvconstants,
                         int width);
void UYVYToARGBRow_SSSE3(const uint8* src_uyvy,
                         uint8* dst_argb,
                         const struct YuvConstants* yuvconstants,
                         int width);
void YUY2ToARGBRow_AVX2(const uint8* src_yuy2,
                        uint8* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width);
void UYVYToARGBRow_AVX2(const uint8* src_uyvy,
                        uint8* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width);
void I422ToRGBARow_SSSE3(const uint8* src_y,
                         const uint8* src_u,
                         const uint8* src_v,
                         uint8* dst_rgba,
                         const struct YuvConstants* yuvconstants,
                         int width);
void I422ToARGB4444Row_SSSE3(const uint8* src_y,
                             const uint8* src_u,
                             const uint8* src_v,
                             uint8* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width);
void I422ToARGB4444Row_AVX2(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width);
void I422ToARGB1555Row_SSSE3(const uint8* src_y,
                             const uint8* src_u,
                             const uint8* src_v,
                             uint8* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width);
void I422ToARGB1555Row_AVX2(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width);
void I422ToRGB565Row_SSSE3(const uint8* src_y,
                           const uint8* src_u,
                           const uint8* src_v,
                           uint8* dst_argb,
                           const struct YuvConstants* yuvconstants,
                           int width);
void I422ToRGB565Row_AVX2(const uint8* src_y,
                          const uint8* src_u,
                          const uint8* src_v,
                          uint8* dst_argb,
                          const struct YuvConstants* yuvconstants,
                          int width);
void I422ToRGB24Row_SSSE3(const uint8* src_y,
                          const uint8* src_u,
                          const uint8* src_v,
                          uint8* dst_rgb24,
                          const struct YuvConstants* yuvconstants,
                          int width);
void I422ToRGB24Row_AVX2(const uint8* src_y,
                         const uint8* src_u,
                         const uint8* src_v,
                         uint8* dst_rgb24,
                         const struct YuvConstants* yuvconstants,
                         int width);
void I422ToARGBRow_Any_AVX2(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width);
void I422ToRGBARow_Any_AVX2(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width);
void I444ToARGBRow_Any_SSSE3(const uint8* src_y,
                             const uint8* src_u,
                             const uint8* src_v,
                             uint8* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width);
void I444ToARGBRow_Any_AVX2(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width);
void I422ToARGBRow_Any_SSSE3(const uint8* src_y,
                             const uint8* src_u,
                             const uint8* src_v,
                             uint8* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width);
void I422AlphaToARGBRow_Any_SSSE3(const uint8* y_buf,
                                  const uint8* u_buf,
                                  const uint8* v_buf,
                                  const uint8* a_buf,
                                  uint8* dst_argb,
                                  const struct YuvConstants* yuvconstants,
                                  int width);
void I422AlphaToARGBRow_Any_AVX2(const uint8* y_buf,
                                 const uint8* u_buf,
                                 const uint8* v_buf,
                                 const uint8* a_buf,
                                 uint8* dst_argb,
                                 const struct YuvConstants* yuvconstants,
                                 int width);
void I411ToARGBRow_Any_SSSE3(const uint8* src_y,
                             const uint8* src_u,
                             const uint8* src_v,
                             uint8* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width);
void I411ToARGBRow_Any_AVX2(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width);
void NV12ToARGBRow_Any_SSSE3(const uint8* src_y,
                             const uint8* src_uv,
                             uint8* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width);
void NV12ToARGBRow_Any_AVX2(const uint8* src_y,
                            const uint8* src_uv,
                            uint8* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width);
void NV21ToARGBRow_Any_SSSE3(const uint8* src_y,
                             const uint8* src_vu,
                             uint8* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width);
void NV21ToARGBRow_Any_AVX2(const uint8* src_y,
                            const uint8* src_vu,
                            uint8* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width);
void NV12ToRGB565Row_Any_SSSE3(const uint8* src_y,
                               const uint8* src_uv,
                               uint8* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width);
void NV12ToRGB565Row_Any_AVX2(const uint8* src_y,
                              const uint8* src_uv,
                              uint8* dst_argb,
                              const struct YuvConstants* yuvconstants,
                              int width);
void YUY2ToARGBRow_Any_SSSE3(const uint8* src_yuy2,
                             uint8* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width);
void UYVYToARGBRow_Any_SSSE3(const uint8* src_uyvy,
                             uint8* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width);
void YUY2ToARGBRow_Any_AVX2(const uint8* src_yuy2,
                            uint8* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width);
void UYVYToARGBRow_Any_AVX2(const uint8* src_uyvy,
                            uint8* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width);
void I422ToRGBARow_Any_SSSE3(const uint8* src_y,
                             const uint8* src_u,
                             const uint8* src_v,
                             uint8* dst_rgba,
                             const struct YuvConstants* yuvconstants,
                             int width);
void I422ToARGB4444Row_Any_SSSE3(const uint8* src_y,
                                 const uint8* src_u,
                                 const uint8* src_v,
                                 uint8* dst_rgba,
                                 const struct YuvConstants* yuvconstants,
                                 int width);
void I422ToARGB4444Row_Any_AVX2(const uint8* src_y,
                                const uint8* src_u,
                                const uint8* src_v,
                                uint8* dst_rgba,
                                const struct YuvConstants* yuvconstants,
                                int width);
void I422ToARGB1555Row_Any_SSSE3(const uint8* src_y,
                                 const uint8* src_u,
                                 const uint8* src_v,
                                 uint8* dst_rgba,
                                 const struct YuvConstants* yuvconstants,
                                 int width);
void I422ToARGB1555Row_Any_AVX2(const uint8* src_y,
                                const uint8* src_u,
                                const uint8* src_v,
                                uint8* dst_rgba,
                                const struct YuvConstants* yuvconstants,
                                int width);
void I422ToRGB565Row_Any_SSSE3(const uint8* src_y,
                               const uint8* src_u,
                               const uint8* src_v,
                               uint8* dst_rgba,
                               const struct YuvConstants* yuvconstants,
                               int width);
void I422ToRGB565Row_Any_AVX2(const uint8* src_y,
                              const uint8* src_u,
                              const uint8* src_v,
                              uint8* dst_rgba,
                              const struct YuvConstants* yuvconstants,
                              int width);
void I422ToRGB24Row_Any_SSSE3(const uint8* src_y,
                              const uint8* src_u,
                              const uint8* src_v,
                              uint8* dst_argb,
                              const struct YuvConstants* yuvconstants,
                              int width);
void I422ToRGB24Row_Any_AVX2(const uint8* src_y,
                             const uint8* src_u,
                             const uint8* src_v,
                             uint8* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width);

void I400ToARGBRow_C(const uint8* src_y, uint8* dst_argb, int width);
void I400ToARGBRow_SSE2(const uint8* src_y, uint8* dst_argb, int width);
void I400ToARGBRow_AVX2(const uint8* src_y, uint8* dst_argb, int width);
void I400ToARGBRow_NEON(const uint8* src_y, uint8* dst_argb, int width);
void I400ToARGBRow_Any_SSE2(const uint8* src_y, uint8* dst_argb, int width);
void I400ToARGBRow_Any_AVX2(const uint8* src_y, uint8* dst_argb, int width);
void I400ToARGBRow_Any_NEON(const uint8* src_y, uint8* dst_argb, int width);

// ARGB preattenuated alpha blend.
void ARGBBlendRow_SSSE3(const uint8* src_argb, const uint8* src_argb1,
                        uint8* dst_argb, int width);
void ARGBBlendRow_NEON(const uint8* src_argb, const uint8* src_argb1,
                       uint8* dst_argb, int width);
void ARGBBlendRow_C(const uint8* src_argb, const uint8* src_argb1,
                    uint8* dst_argb, int width);

// Unattenuated planar alpha blend.
void BlendPlaneRow_SSSE3(const uint8* src0, const uint8* src1,
                         const uint8* alpha, uint8* dst, int width);
void BlendPlaneRow_Any_SSSE3(const uint8* src0, const uint8* src1,
                             const uint8* alpha, uint8* dst, int width);
void BlendPlaneRow_AVX2(const uint8* src0, const uint8* src1,
                        const uint8* alpha, uint8* dst, int width);
void BlendPlaneRow_Any_AVX2(const uint8* src0, const uint8* src1,
                            const uint8* alpha, uint8* dst, int width);
void BlendPlaneRow_C(const uint8* src0, const uint8* src1,
                     const uint8* alpha, uint8* dst, int width);

// ARGB multiply images. Same API as Blend, but these require
// pointer and width alignment for SSE2.
void ARGBMultiplyRow_C(const uint8* src_argb, const uint8* src_argb1,
                       uint8* dst_argb, int width);
void ARGBMultiplyRow_SSE2(const uint8* src_argb, const uint8* src_argb1,
                          uint8* dst_argb, int width);
void ARGBMultiplyRow_Any_SSE2(const uint8* src_argb, const uint8* src_argb1,
                              uint8* dst_argb, int width);
void ARGBMultiplyRow_AVX2(const uint8* src_argb, const uint8* src_argb1,
                          uint8* dst_argb, int width);
void ARGBMultiplyRow_Any_AVX2(const uint8* src_argb, const uint8* src_argb1,
                              uint8* dst_argb, int width);
void ARGBMultiplyRow_NEON(const uint8* src_argb, const uint8* src_argb1,
                          uint8* dst_argb, int width);
void ARGBMultiplyRow_Any_NEON(const uint8* src_argb, const uint8* src_argb1,
                              uint8* dst_argb, int width);

// ARGB add images.
void ARGBAddRow_C(const uint8* src_argb, const uint8* src_argb1,
                  uint8* dst_argb, int width);
void ARGBAddRow_SSE2(const uint8* src_argb, const uint8* src_argb1,
                     uint8* dst_argb, int width);
void ARGBAddRow_Any_SSE2(const uint8* src_argb, const uint8* src_argb1,
                         uint8* dst_argb, int width);
void ARGBAddRow_AVX2(const uint8* src_argb, const uint8* src_argb1,
                     uint8* dst_argb, int width);
void ARGBAddRow_Any_AVX2(const uint8* src_argb, const uint8* src_argb1,
                         uint8* dst_argb, int width);
void ARGBAddRow_NEON(const uint8* src_argb, const uint8* src_argb1,
                     uint8* dst_argb, int width);
void ARGBAddRow_Any_NEON(const uint8* src_argb, const uint8* src_argb1,
                         uint8* dst_argb, int width);

// ARGB subtract images. Same API as Blend, but these require
// pointer and width alignment for SSE2.
void ARGBSubtractRow_C(const uint8* src_argb, const uint8* src_argb1,
                       uint8* dst_argb, int width);
void ARGBSubtractRow_SSE2(const uint8* src_argb, const uint8* src_argb1,
                          uint8* dst_argb, int width);
void ARGBSubtractRow_Any_SSE2(const uint8* src_argb, const uint8* src_argb1,
                              uint8* dst_argb, int width);
void ARGBSubtractRow_AVX2(const uint8* src_argb, const uint8* src_argb1,
                          uint8* dst_argb, int width);
void ARGBSubtractRow_Any_AVX2(const uint8* src_argb, const uint8* src_argb1,
                              uint8* dst_argb, int width);
void ARGBSubtractRow_NEON(const uint8* src_argb, const uint8* src_argb1,
                          uint8* dst_argb, int width);
void ARGBSubtractRow_Any_NEON(const uint8* src_argb, const uint8* src_argb1,
                              uint8* dst_argb, int width);

void ARGBToRGB24Row_Any_SSSE3(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToRAWRow_Any_SSSE3(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToRGB565Row_Any_SSE2(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToARGB1555Row_Any_SSE2(const uint8* src_argb, uint8* dst_rgb,
                                int width);
void ARGBToARGB4444Row_Any_SSE2(const uint8* src_argb, uint8* dst_rgb,
                                int width);

void ARGBToRGB565DitherRow_Any_SSE2(const uint8* src_argb, uint8* dst_rgb,
                                    const uint32 dither4, int width);
void ARGBToRGB565DitherRow_Any_AVX2(const uint8* src_argb, uint8* dst_rgb,
                                    const uint32 dither4, int width);

void ARGBToRGB565Row_Any_AVX2(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToARGB1555Row_Any_AVX2(const uint8* src_argb, uint8* dst_rgb,
                                int width);
void ARGBToARGB4444Row_Any_AVX2(const uint8* src_argb, uint8* dst_rgb,
                                int width);

void ARGBToRGB24Row_Any_NEON(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToRAWRow_Any_NEON(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToRGB565Row_Any_NEON(const uint8* src_argb, uint8* dst_rgb, int width);
void ARGBToARGB1555Row_Any_NEON(const uint8* src_argb, uint8* dst_rgb,
                                int width);
void ARGBToARGB4444Row_Any_NEON(const uint8* src_argb, uint8* dst_rgb,
                                int width);
void ARGBToRGB565DitherRow_Any_NEON(const uint8* src_argb, uint8* dst_rgb,
                                    const uint32 dither4, int width);

void I444ToARGBRow_Any_NEON(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width);
void I422ToARGBRow_Any_NEON(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width);
void I422AlphaToARGBRow_Any_NEON(const uint8* src_y,
                                 const uint8* src_u,
                                 const uint8* src_v,
                                 const uint8* src_a,
                                 uint8* dst_argb,
                                 const struct YuvConstants* yuvconstants,
                                 int width);
void I411ToARGBRow_Any_NEON(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width);
void I422ToRGBARow_Any_NEON(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width);
void I422ToRGB24Row_Any_NEON(const uint8* src_y,
                             const uint8* src_u,
                             const uint8* src_v,
                             uint8* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width);
void I422ToARGB4444Row_Any_NEON(const uint8* src_y,
                                const uint8* src_u,
                                const uint8* src_v,
                                uint8* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width);
void I422ToARGB1555Row_Any_NEON(const uint8* src_y,
                                const uint8* src_u,
                                const uint8* src_v,
                                uint8* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width);
void I422ToRGB565Row_Any_NEON(const uint8* src_y,
                              const uint8* src_u,
                              const uint8* src_v,
                              uint8* dst_argb,
                              const struct YuvConstants* yuvconstants,
                              int width);
void NV12ToARGBRow_Any_NEON(const uint8* src_y,
                            const uint8* src_uv,
                            uint8* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width);
void NV21ToARGBRow_Any_NEON(const uint8* src_y,
                            const uint8* src_vu,
                            uint8* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width);
void NV12ToRGB565Row_Any_NEON(const uint8* src_y,
                              const uint8* src_uv,
                              uint8* dst_argb,
                              const struct YuvConstants* yuvconstants,
                              int width);
void YUY2ToARGBRow_Any_NEON(const uint8* src_yuy2,
                            uint8* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width);
void UYVYToARGBRow_Any_NEON(const uint8* src_uyvy,
                            uint8* dst_argb,
                            const struct YuvConstants* yuvconstants,
                            int width);
void I422ToARGBRow_DSPR2(const uint8* src_y,
                         const uint8* src_u,
                         const uint8* src_v,
                         uint8* dst_argb,
                         const struct YuvConstants* yuvconstants,
                         int width);
void I422ToARGBRow_DSPR2(const uint8* src_y,
                         const uint8* src_u,
                         const uint8* src_v,
                         uint8* dst_argb,
                         const struct YuvConstants* yuvconstants,
                         int width);

void YUY2ToYRow_AVX2(const uint8* src_yuy2, uint8* dst_y, int width);
void YUY2ToUVRow_AVX2(const uint8* src_yuy2, int stride_yuy2,
                      uint8* dst_u, uint8* dst_v, int width);
void YUY2ToUV422Row_AVX2(const uint8* src_yuy2,
                         uint8* dst_u, uint8* dst_v, int width);
void YUY2ToYRow_SSE2(const uint8* src_yuy2, uint8* dst_y, int width);
void YUY2ToUVRow_SSE2(const uint8* src_yuy2, int stride_yuy2,
                      uint8* dst_u, uint8* dst_v, int width);
void YUY2ToUV422Row_SSE2(const uint8* src_yuy2,
                         uint8* dst_u, uint8* dst_v, int width);
void YUY2ToYRow_NEON(const uint8* src_yuy2, uint8* dst_y, int width);
void YUY2ToUVRow_NEON(const uint8* src_yuy2, int stride_yuy2,
                      uint8* dst_u, uint8* dst_v, int width);
void YUY2ToUV422Row_NEON(const uint8* src_yuy2,
                         uint8* dst_u, uint8* dst_v, int width);
void YUY2ToYRow_C(const uint8* src_yuy2, uint8* dst_y, int width);
void YUY2ToUVRow_C(const uint8* src_yuy2, int stride_yuy2,
                   uint8* dst_u, uint8* dst_v, int width);
void YUY2ToUV422Row_C(const uint8* src_yuy2,
                      uint8* dst_u, uint8* dst_v, int width);
void YUY2ToYRow_Any_AVX2(const uint8* src_yuy2, uint8* dst_y, int width);
void YUY2ToUVRow_Any_AVX2(const uint8* src_yuy2, int stride_yuy2,
                          uint8* dst_u, uint8* dst_v, int width);
void YUY2ToUV422Row_Any_AVX2(const uint8* src_yuy2,
                             uint8* dst_u, uint8* dst_v, int width);
void YUY2ToYRow_Any_SSE2(const uint8* src_yuy2, uint8* dst_y, int width);
void YUY2ToUVRow_Any_SSE2(const uint8* src_yuy2, int stride_yuy2,
                          uint8* dst_u, uint8* dst_v, int width);
void YUY2ToUV422Row_Any_SSE2(const uint8* src_yuy2,
                             uint8* dst_u, uint8* dst_v, int width);
void YUY2ToYRow_Any_NEON(const uint8* src_yuy2, uint8* dst_y, int width);
void YUY2ToUVRow_Any_NEON(const uint8* src_yuy2, int stride_yuy2,
                          uint8* dst_u, uint8* dst_v, int width);
void YUY2ToUV422Row_Any_NEON(const uint8* src_yuy2,
                             uint8* dst_u, uint8* dst_v, int width);
void UYVYToYRow_AVX2(const uint8* src_uyvy, uint8* dst_y, int width);
void UYVYToUVRow_AVX2(const uint8* src_uyvy, int stride_uyvy,
                      uint8* dst_u, uint8* dst_v, int width);
void UYVYToUV422Row_AVX2(const uint8* src_uyvy,
                         uint8* dst_u, uint8* dst_v, int width);
void UYVYToYRow_SSE2(const uint8* src_uyvy, uint8* dst_y, int width);
void UYVYToUVRow_SSE2(const uint8* src_uyvy, int stride_uyvy,
                      uint8* dst_u, uint8* dst_v, int width);
void UYVYToUV422Row_SSE2(const uint8* src_uyvy,
                         uint8* dst_u, uint8* dst_v, int width);
void UYVYToYRow_AVX2(const uint8* src_uyvy, uint8* dst_y, int width);
void UYVYToUVRow_AVX2(const uint8* src_uyvy, int stride_uyvy,
                      uint8* dst_u, uint8* dst_v, int width);
void UYVYToUV422Row_AVX2(const uint8* src_uyvy,
                         uint8* dst_u, uint8* dst_v, int width);
void UYVYToYRow_NEON(const uint8* src_uyvy, uint8* dst_y, int width);
void UYVYToUVRow_NEON(const uint8* src_uyvy, int stride_uyvy,
                      uint8* dst_u, uint8* dst_v, int width);
void UYVYToUV422Row_NEON(const uint8* src_uyvy,
                         uint8* dst_u, uint8* dst_v, int width);

void UYVYToYRow_C(const uint8* src_uyvy, uint8* dst_y, int width);
void UYVYToUVRow_C(const uint8* src_uyvy, int stride_uyvy,
                   uint8* dst_u, uint8* dst_v, int width);
void UYVYToUV422Row_C(const uint8* src_uyvy,
                      uint8* dst_u, uint8* dst_v, int width);
void UYVYToYRow_Any_AVX2(const uint8* src_uyvy, uint8* dst_y, int width);
void UYVYToUVRow_Any_AVX2(const uint8* src_uyvy, int stride_uyvy,
                          uint8* dst_u, uint8* dst_v, int width);
void UYVYToUV422Row_Any_AVX2(const uint8* src_uyvy,
                             uint8* dst_u, uint8* dst_v, int width);
void UYVYToYRow_Any_SSE2(const uint8* src_uyvy, uint8* dst_y, int width);
void UYVYToUVRow_Any_SSE2(const uint8* src_uyvy, int stride_uyvy,
                          uint8* dst_u, uint8* dst_v, int width);
void UYVYToUV422Row_Any_SSE2(const uint8* src_uyvy,
                             uint8* dst_u, uint8* dst_v, int width);
void UYVYToYRow_Any_NEON(const uint8* src_uyvy, uint8* dst_y, int width);
void UYVYToUVRow_Any_NEON(const uint8* src_uyvy, int stride_uyvy,
                          uint8* dst_u, uint8* dst_v, int width);
void UYVYToUV422Row_Any_NEON(const uint8* src_uyvy,
                             uint8* dst_u, uint8* dst_v, int width);

void I422ToYUY2Row_C(const uint8* src_y,
                     const uint8* src_u,
                     const uint8* src_v,
                     uint8* dst_yuy2, int width);
void I422ToUYVYRow_C(const uint8* src_y,
                     const uint8* src_u,
                     const uint8* src_v,
                     uint8* dst_uyvy, int width);
void I422ToYUY2Row_SSE2(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_yuy2, int width);
void I422ToUYVYRow_SSE2(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_uyvy, int width);
void I422ToYUY2Row_Any_SSE2(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_yuy2, int width);
void I422ToUYVYRow_Any_SSE2(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_uyvy, int width);
void I422ToYUY2Row_NEON(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_yuy2, int width);
void I422ToUYVYRow_NEON(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_uyvy, int width);
void I422ToYUY2Row_Any_NEON(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_yuy2, int width);
void I422ToUYVYRow_Any_NEON(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_uyvy, int width);

// Effects related row functions.
void ARGBAttenuateRow_C(const uint8* src_argb, uint8* dst_argb, int width);
void ARGBAttenuateRow_SSSE3(const uint8* src_argb, uint8* dst_argb, int width);
void ARGBAttenuateRow_AVX2(const uint8* src_argb, uint8* dst_argb, int width);
void ARGBAttenuateRow_NEON(const uint8* src_argb, uint8* dst_argb, int width);
void ARGBAttenuateRow_Any_SSE2(const uint8* src_argb, uint8* dst_argb,
                               int width);
void ARGBAttenuateRow_Any_SSSE3(const uint8* src_argb, uint8* dst_argb,
                                int width);
void ARGBAttenuateRow_Any_AVX2(const uint8* src_argb, uint8* dst_argb,
                               int width);
void ARGBAttenuateRow_Any_NEON(const uint8* src_argb, uint8* dst_argb,
                               int width);

// Inverse table for unattenuate, shared by C and SSE2.
extern const uint32 fixed_invtbl8[256];
void ARGBUnattenuateRow_C(const uint8* src_argb, uint8* dst_argb, int width);
void ARGBUnattenuateRow_SSE2(const uint8* src_argb, uint8* dst_argb, int width);
void ARGBUnattenuateRow_AVX2(const uint8* src_argb, uint8* dst_argb, int width);
void ARGBUnattenuateRow_Any_SSE2(const uint8* src_argb, uint8* dst_argb,
                                 int width);
void ARGBUnattenuateRow_Any_AVX2(const uint8* src_argb, uint8* dst_argb,
                                 int width);

void ARGBGrayRow_C(const uint8* src_argb, uint8* dst_argb, int width);
void ARGBGrayRow_SSSE3(const uint8* src_argb, uint8* dst_argb, int width);
void ARGBGrayRow_NEON(const uint8* src_argb, uint8* dst_argb, int width);

void ARGBSepiaRow_C(uint8* dst_argb, int width);
void ARGBSepiaRow_SSSE3(uint8* dst_argb, int width);
void ARGBSepiaRow_NEON(uint8* dst_argb, int width);

void ARGBColorMatrixRow_C(const uint8* src_argb, uint8* dst_argb,
                          const int8* matrix_argb, int width);
void ARGBColorMatrixRow_SSSE3(const uint8* src_argb, uint8* dst_argb,
                              const int8* matrix_argb, int width);
void ARGBColorMatrixRow_NEON(const uint8* src_argb, uint8* dst_argb,
                             const int8* matrix_argb, int width);

void ARGBColorTableRow_C(uint8* dst_argb, const uint8* table_argb, int width);
void ARGBColorTableRow_X86(uint8* dst_argb, const uint8* table_argb, int width);

void RGBColorTableRow_C(uint8* dst_argb, const uint8* table_argb, int width);
void RGBColorTableRow_X86(uint8* dst_argb, const uint8* table_argb, int width);

void ARGBQuantizeRow_C(uint8* dst_argb, int scale, int interval_size,
                       int interval_offset, int width);
void ARGBQuantizeRow_SSE2(uint8* dst_argb, int scale, int interval_size,
                          int interval_offset, int width);
void ARGBQuantizeRow_NEON(uint8* dst_argb, int scale, int interval_size,
                          int interval_offset, int width);

void ARGBShadeRow_C(const uint8* src_argb, uint8* dst_argb, int width,
                    uint32 value);
void ARGBShadeRow_SSE2(const uint8* src_argb, uint8* dst_argb, int width,
                       uint32 value);
void ARGBShadeRow_NEON(const uint8* src_argb, uint8* dst_argb, int width,
                       uint32 value);

// Used for blur.
void CumulativeSumToAverageRow_SSE2(const int32* topleft, const int32* botleft,
                                    int width, int area, uint8* dst, int count);
void ComputeCumulativeSumRow_SSE2(const uint8* row, int32* cumsum,
                                  const int32* previous_cumsum, int width);

void CumulativeSumToAverageRow_C(const int32* topleft, const int32* botleft,
                                 int width, int area, uint8* dst, int count);
void ComputeCumulativeSumRow_C(const uint8* row, int32* cumsum,
                               const int32* previous_cumsum, int width);

LIBYUV_API
void ARGBAffineRow_C(const uint8* src_argb, int src_argb_stride,
                     uint8* dst_argb, const float* uv_dudv, int width);
LIBYUV_API
void ARGBAffineRow_SSE2(const uint8* src_argb, int src_argb_stride,
                        uint8* dst_argb, const float* uv_dudv, int width);

// Used for I420Scale, ARGBScale, and ARGBInterpolate.
void InterpolateRow_C(uint8* dst_ptr, const uint8* src_ptr,
                      ptrdiff_t src_stride_ptr,
                      int width, int source_y_fraction);
void InterpolateRow_SSSE3(uint8* dst_ptr, const uint8* src_ptr,
                          ptrdiff_t src_stride_ptr, int width,
                          int source_y_fraction);
void InterpolateRow_AVX2(uint8* dst_ptr, const uint8* src_ptr,
                         ptrdiff_t src_stride_ptr, int width,
                         int source_y_fraction);
void InterpolateRow_NEON(uint8* dst_ptr, const uint8* src_ptr,
                         ptrdiff_t src_stride_ptr, int width,
                         int source_y_fraction);
void InterpolateRow_DSPR2(uint8* dst_ptr, const uint8* src_ptr,
                          ptrdiff_t src_stride_ptr, int width,
                          int source_y_fraction);
void InterpolateRow_Any_NEON(uint8* dst_ptr, const uint8* src_ptr,
                             ptrdiff_t src_stride_ptr, int width,
                             int source_y_fraction);
void InterpolateRow_Any_SSSE3(uint8* dst_ptr, const uint8* src_ptr,
                              ptrdiff_t src_stride_ptr, int width,
                              int source_y_fraction);
void InterpolateRow_Any_AVX2(uint8* dst_ptr, const uint8* src_ptr,
                             ptrdiff_t src_stride_ptr, int width,
                             int source_y_fraction);
void InterpolateRow_Any_DSPR2(uint8* dst_ptr, const uint8* src_ptr,
                              ptrdiff_t src_stride_ptr, int width,
                              int source_y_fraction);

void InterpolateRow_16_C(uint16* dst_ptr, const uint16* src_ptr,
                         ptrdiff_t src_stride_ptr,
                         int width, int source_y_fraction);

// Sobel images.
void SobelXRow_C(const uint8* src_y0, const uint8* src_y1, const uint8* src_y2,
                 uint8* dst_sobelx, int width);
void SobelXRow_SSE2(const uint8* src_y0, const uint8* src_y1,
                    const uint8* src_y2, uint8* dst_sobelx, int width);
void SobelXRow_NEON(const uint8* src_y0, const uint8* src_y1,
                    const uint8* src_y2, uint8* dst_sobelx, int width);
void SobelYRow_C(const uint8* src_y0, const uint8* src_y1,
                 uint8* dst_sobely, int width);
void SobelYRow_SSE2(const uint8* src_y0, const uint8* src_y1,
                    uint8* dst_sobely, int width);
void SobelYRow_NEON(const uint8* src_y0, const uint8* src_y1,
                    uint8* dst_sobely, int width);
void SobelRow_C(const uint8* src_sobelx, const uint8* src_sobely,
                uint8* dst_argb, int width);
void SobelRow_SSE2(const uint8* src_sobelx, const uint8* src_sobely,
                   uint8* dst_argb, int width);
void SobelRow_NEON(const uint8* src_sobelx, const uint8* src_sobely,
                   uint8* dst_argb, int width);
void SobelToPlaneRow_C(const uint8* src_sobelx, const uint8* src_sobely,
                       uint8* dst_y, int width);
void SobelToPlaneRow_SSE2(const uint8* src_sobelx, const uint8* src_sobely,
                          uint8* dst_y, int width);
void SobelToPlaneRow_NEON(const uint8* src_sobelx, const uint8* src_sobely,
                          uint8* dst_y, int width);
void SobelXYRow_C(const uint8* src_sobelx, const uint8* src_sobely,
                  uint8* dst_argb, int width);
void SobelXYRow_SSE2(const uint8* src_sobelx, const uint8* src_sobely,
                     uint8* dst_argb, int width);
void SobelXYRow_NEON(const uint8* src_sobelx, const uint8* src_sobely,
                     uint8* dst_argb, int width);
void SobelRow_Any_SSE2(const uint8* src_sobelx, const uint8* src_sobely,
                       uint8* dst_argb, int width);
void SobelRow_Any_NEON(const uint8* src_sobelx, const uint8* src_sobely,
                       uint8* dst_argb, int width);
void SobelToPlaneRow_Any_SSE2(const uint8* src_sobelx, const uint8* src_sobely,
                              uint8* dst_y, int width);
void SobelToPlaneRow_Any_NEON(const uint8* src_sobelx, const uint8* src_sobely,
                              uint8* dst_y, int width);
void SobelXYRow_Any_SSE2(const uint8* src_sobelx, const uint8* src_sobely,
                         uint8* dst_argb, int width);
void SobelXYRow_Any_NEON(const uint8* src_sobelx, const uint8* src_sobely,
                         uint8* dst_argb, int width);

void ARGBPolynomialRow_C(const uint8* src_argb,
                         uint8* dst_argb, const float* poly,
                         int width);
void ARGBPolynomialRow_SSE2(const uint8* src_argb,
                            uint8* dst_argb, const float* poly,
                            int width);
void ARGBPolynomialRow_AVX2(const uint8* src_argb,
                            uint8* dst_argb, const float* poly,
                            int width);

// Scale and convert to half float.
void HalfFloatRow_C(const uint16* src, uint16* dst, float scale, int width);
void HalfFloatRow_AVX2(const uint16* src, uint16* dst, float scale, int width);
void HalfFloatRow_Any_AVX2(const uint16* src, uint16* dst, float scale,
                           int width);
void HalfFloatRow_SSE2(const uint16* src, uint16* dst, float scale, int width);
void HalfFloatRow_Any_SSE2(const uint16* src, uint16* dst, float scale,
                           int width);

void ARGBLumaColorTableRow_C(const uint8* src_argb, uint8* dst_argb, int width,
                             const uint8* luma, uint32 lumacoeff);
void ARGBLumaColorTableRow_SSSE3(const uint8* src_argb, uint8* dst_argb,
                                 int width,
                                 const uint8* luma, uint32 lumacoeff);

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_ROW_H_
