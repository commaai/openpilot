#ifndef TARGET_X64_INTEL_HASWELL
#define TARGET_X64_INTEL_HASWELL
#endif

#ifndef TARGET_NEED_FEATURE_AVX2
#define TARGET_NEED_FEATURE_AVX2 1
#endif

#ifndef TARGET_NEED_FEATURE_FMA
#define TARGET_NEED_FEATURE_FMA 1
#endif

#ifndef TARGET_NEED_FEATURE_SSE3
/* #undef TARGET_NEED_FEATURE_SSE3 */
#endif

#ifndef TARGET_NEED_FEATURE_AVX
/* #undef TARGET_NEED_FEATURE_AVX */
#endif

#ifndef TARGET_NEED_FEATURE_VFPv3
/* #undef TARGET_NEED_FEATURE_VFPv3 */
#endif

#ifndef TARGET_NEED_FEATURE_NEON
/* #undef TARGET_NEED_FEATURE_NEON */
#endif

#ifndef TARGET_NEED_FEATURE_VFPv4
/* #undef TARGET_NEED_FEATURE_VFPv4 */
#endif

#ifndef TARGET_NEED_FEATURE_NEONv2
/* #undef TARGET_NEED_FEATURE_NEONv2 */
#endif

#ifndef LA_HIGH_PERFORMANCE
#define LA_HIGH_PERFORMANCE
#endif

#ifndef MF_PANELMAJ
#define MF_PANELMAJ
#endif

#ifndef EXT_DEP
#define ON 1
#define OFF 0
#if ON==ON
#define EXT_DEP
#endif
#undef ON
#undef OFF
#endif

#ifndef BLAS_API
#define ON 1
#define OFF 0
#if OFF==ON
#define BLAS_API
#endif
#undef ON
#undef OFF
#endif

#ifndef FORTRAN_BLAS_API
#define ON 1
#define OFF 0
#if OFF==ON
#define FORTRAN_BLAS_API
#endif
#undef ON
#undef OFF
#endif
