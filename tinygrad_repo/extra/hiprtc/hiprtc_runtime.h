#pragma clang diagnostic ignored "-Weverything"
      
# 1 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/out/ubuntu-22.04/22.04/build/hip-on-rocclr/hipamd/src/hiprtc/hip_rtc_gen/hipRTC_header.h"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 845 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
# 1 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_runtime_wrapper.h" 1 3
# 33 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_runtime_wrapper.h" 3
extern "C" {
  __attribute__((__visibility__("default")))
  __attribute__((weak))
  __attribute__((noreturn))
  __attribute__((device)) void __cxa_pure_virtual(void) {
    __builtin_trap();
  }
  __attribute__((__visibility__("default")))
  __attribute__((weak))
  __attribute__((noreturn))
  __attribute__((device)) void __cxa_deleted_virtual(void) {
    __builtin_trap();
  }
}
# 57 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_runtime_wrapper.h" 3
typedef long unsigned int size_t;
# 74 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_runtime_wrapper.h" 3
typedef long unsigned int __hip_size_t;


extern "C" {



extern "C" __attribute__((device)) unsigned long long __ockl_dm_alloc(unsigned long long __size);
extern "C" __attribute__((device)) void __ockl_dm_dealloc(unsigned long long __addr);
# 95 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_runtime_wrapper.h" 3
__attribute__((weak)) inline __attribute__((device)) void *malloc(__hip_size_t __size) {
  return (void *) __ockl_dm_alloc(__size);
}
__attribute__((weak)) inline __attribute__((device)) void free(void *__ptr) {
  __ockl_dm_dealloc((unsigned long long)__ptr);
}
# 124 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_runtime_wrapper.h" 3
}


# 1 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_libdevice_declares.h" 1 3
# 14 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_libdevice_declares.h" 3
extern "C" {



__attribute__((device)) __attribute__((const)) float __ocml_acos_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_acosh_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_asin_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_asinh_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_atan2_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_atan_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_atanh_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_cbrt_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_ceil_f32(float);
__attribute__((device)) __attribute__((const)) __attribute__((device)) float __ocml_copysign_f32(float,
                                                                       float);
__attribute__((device)) float __ocml_cos_f32(float);
__attribute__((device)) float __ocml_native_cos_f32(float);
__attribute__((device)) __attribute__((pure)) __attribute__((device)) float __ocml_cosh_f32(float);
__attribute__((device)) float __ocml_cospi_f32(float);
__attribute__((device)) float __ocml_i0_f32(float);
__attribute__((device)) float __ocml_i1_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_erfc_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_erfcinv_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_erfcx_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_erf_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_erfinv_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_exp10_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_native_exp10_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_exp2_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_exp_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_native_exp_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_expm1_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_fabs_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_fdim_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_floor_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_fma_f32(float, float, float);
__attribute__((device)) __attribute__((const)) float __ocml_fmax_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_fmin_f32(float, float);
__attribute__((device)) __attribute__((const)) __attribute__((device)) float __ocml_fmod_f32(float,
                                                                   float);
__attribute__((device)) float __ocml_frexp_f32(float,
                                  __attribute__((address_space(5))) int *);
__attribute__((device)) __attribute__((const)) float __ocml_hypot_f32(float, float);
__attribute__((device)) __attribute__((const)) int __ocml_ilogb_f32(float);
__attribute__((device)) __attribute__((const)) int __ocml_isfinite_f32(float);
__attribute__((device)) __attribute__((const)) int __ocml_isinf_f32(float);
__attribute__((device)) __attribute__((const)) int __ocml_isnan_f32(float);
__attribute__((device)) float __ocml_j0_f32(float);
__attribute__((device)) float __ocml_j1_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_ldexp_f32(float, int);
__attribute__((device)) float __ocml_lgamma_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_log10_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_native_log10_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_log1p_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_log2_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_native_log2_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_logb_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_log_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_native_log_f32(float);
__attribute__((device)) float __ocml_modf_f32(float,
                                 __attribute__((address_space(5))) float *);
__attribute__((device)) __attribute__((const)) float __ocml_nearbyint_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_nextafter_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_len3_f32(float, float, float);
__attribute__((device)) __attribute__((const)) float __ocml_len4_f32(float, float, float,
                                                        float);
__attribute__((device)) __attribute__((pure)) float __ocml_ncdf_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_ncdfinv_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_pow_f32(float, float);
__attribute__((device)) __attribute__((pure)) float __ocml_pown_f32(float, int);
__attribute__((device)) __attribute__((pure)) float __ocml_rcbrt_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_remainder_f32(float, float);
__attribute__((device)) float __ocml_remquo_f32(float, float,
                                   __attribute__((address_space(5))) int *);
__attribute__((device)) __attribute__((const)) float __ocml_rhypot_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_rint_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_rlen3_f32(float, float, float);
__attribute__((device)) __attribute__((const)) float __ocml_rlen4_f32(float, float, float,
                                                         float);
__attribute__((device)) __attribute__((const)) float __ocml_round_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_rsqrt_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_scalb_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_scalbn_f32(float, int);
__attribute__((device)) __attribute__((const)) int __ocml_signbit_f32(float);
__attribute__((device)) float __ocml_sincos_f32(float,
                                   __attribute__((address_space(5))) float *);
__attribute__((device)) float __ocml_sincospi_f32(float,
                                     __attribute__((address_space(5))) float *);
__attribute__((device)) float __ocml_sin_f32(float);
__attribute__((device)) float __ocml_native_sin_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_sinh_f32(float);
__attribute__((device)) float __ocml_sinpi_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_sqrt_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_native_sqrt_f32(float);
__attribute__((device)) float __ocml_tan_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_tanh_f32(float);
__attribute__((device)) float __ocml_tgamma_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_trunc_f32(float);
__attribute__((device)) float __ocml_y0_f32(float);
__attribute__((device)) float __ocml_y1_f32(float);


__attribute__((device)) __attribute__((const)) float __ocml_add_rte_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_add_rtn_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_add_rtp_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_add_rtz_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_sub_rte_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_sub_rtn_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_sub_rtp_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_sub_rtz_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_mul_rte_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_mul_rtn_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_mul_rtp_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_mul_rtz_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_div_rte_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_div_rtn_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_div_rtp_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_div_rtz_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_sqrt_rte_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_sqrt_rtn_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_sqrt_rtp_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_sqrt_rtz_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_fma_rte_f32(float, float, float);
__attribute__((device)) __attribute__((const)) float __ocml_fma_rtn_f32(float, float, float);
__attribute__((device)) __attribute__((const)) float __ocml_fma_rtp_f32(float, float, float);
__attribute__((device)) __attribute__((const)) float __ocml_fma_rtz_f32(float, float, float);

__attribute__((device)) inline __attribute__((const)) float
__llvm_amdgcn_cos_f32(float __x) {
  return __builtin_amdgcn_cosf(__x);
}
__attribute__((device)) inline __attribute__((const)) float
__llvm_amdgcn_rcp_f32(float __x) {
  return __builtin_amdgcn_rcpf(__x);
}
__attribute__((device)) inline __attribute__((const)) float
__llvm_amdgcn_rsq_f32(float __x) {
  return __builtin_amdgcn_rsqf(__x);
}
__attribute__((device)) inline __attribute__((const)) float
__llvm_amdgcn_sin_f32(float __x) {
  return __builtin_amdgcn_sinf(__x);
}




__attribute__((device)) __attribute__((const)) double __ocml_acos_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_acosh_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_asin_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_asinh_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_atan2_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_atan_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_atanh_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_cbrt_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_ceil_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_copysign_f64(double, double);
__attribute__((device)) double __ocml_cos_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_cosh_f64(double);
__attribute__((device)) double __ocml_cospi_f64(double);
__attribute__((device)) double __ocml_i0_f64(double);
__attribute__((device)) double __ocml_i1_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_erfc_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_erfcinv_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_erfcx_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_erf_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_erfinv_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_exp10_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_exp2_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_exp_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_expm1_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_fabs_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_fdim_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_floor_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_fma_f64(double, double, double);
__attribute__((device)) __attribute__((const)) double __ocml_fmax_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_fmin_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_fmod_f64(double, double);
__attribute__((device)) double __ocml_frexp_f64(double,
                                   __attribute__((address_space(5))) int *);
__attribute__((device)) __attribute__((const)) double __ocml_hypot_f64(double, double);
__attribute__((device)) __attribute__((const)) int __ocml_ilogb_f64(double);
__attribute__((device)) __attribute__((const)) int __ocml_isfinite_f64(double);
__attribute__((device)) __attribute__((const)) int __ocml_isinf_f64(double);
__attribute__((device)) __attribute__((const)) int __ocml_isnan_f64(double);
__attribute__((device)) double __ocml_j0_f64(double);
__attribute__((device)) double __ocml_j1_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_ldexp_f64(double, int);
__attribute__((device)) double __ocml_lgamma_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_log10_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_log1p_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_log2_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_logb_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_log_f64(double);
__attribute__((device)) double __ocml_modf_f64(double,
                                  __attribute__((address_space(5))) double *);
__attribute__((device)) __attribute__((const)) double __ocml_nearbyint_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_nextafter_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_len3_f64(double, double,
                                                         double);
__attribute__((device)) __attribute__((const)) double __ocml_len4_f64(double, double, double,
                                                         double);
__attribute__((device)) __attribute__((pure)) double __ocml_ncdf_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_ncdfinv_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_pow_f64(double, double);
__attribute__((device)) __attribute__((pure)) double __ocml_pown_f64(double, int);
__attribute__((device)) __attribute__((pure)) double __ocml_rcbrt_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_remainder_f64(double, double);
__attribute__((device)) double __ocml_remquo_f64(double, double,
                                    __attribute__((address_space(5))) int *);
__attribute__((device)) __attribute__((const)) double __ocml_rhypot_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_rint_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_rlen3_f64(double, double,
                                                          double);
__attribute__((device)) __attribute__((const)) double __ocml_rlen4_f64(double, double,
                                                          double, double);
__attribute__((device)) __attribute__((const)) double __ocml_round_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_rsqrt_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_scalb_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_scalbn_f64(double, int);
__attribute__((device)) __attribute__((const)) int __ocml_signbit_f64(double);
__attribute__((device)) double __ocml_sincos_f64(double,
                                    __attribute__((address_space(5))) double *);
__attribute__((device)) double
__ocml_sincospi_f64(double, __attribute__((address_space(5))) double *);
__attribute__((device)) double __ocml_sin_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_sinh_f64(double);
__attribute__((device)) double __ocml_sinpi_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_sqrt_f64(double);
__attribute__((device)) double __ocml_tan_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_tanh_f64(double);
__attribute__((device)) double __ocml_tgamma_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_trunc_f64(double);
__attribute__((device)) double __ocml_y0_f64(double);
__attribute__((device)) double __ocml_y1_f64(double);


__attribute__((device)) __attribute__((const)) double __ocml_add_rte_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_add_rtn_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_add_rtp_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_add_rtz_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_sub_rte_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_sub_rtn_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_sub_rtp_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_sub_rtz_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_mul_rte_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_mul_rtn_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_mul_rtp_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_mul_rtz_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_div_rte_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_div_rtn_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_div_rtp_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_div_rtz_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_sqrt_rte_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_sqrt_rtn_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_sqrt_rtp_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_sqrt_rtz_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_fma_rte_f64(double, double,
                                                            double);
__attribute__((device)) __attribute__((const)) double __ocml_fma_rtn_f64(double, double,
                                                            double);
__attribute__((device)) __attribute__((const)) double __ocml_fma_rtp_f64(double, double,
                                                            double);
__attribute__((device)) __attribute__((const)) double __ocml_fma_rtz_f64(double, double,
                                                            double);

__attribute__((device)) inline __attribute__((const)) double
__llvm_amdgcn_rcp_f64(double __x) {
  return __builtin_amdgcn_rcp(__x);
}
__attribute__((device)) inline __attribute__((const)) double
__llvm_amdgcn_rsq_f64(double __x) {
  return __builtin_amdgcn_rsq(__x);
}

__attribute__((device)) __attribute__((const)) _Float16 __ocml_ceil_f16(_Float16);
__attribute__((device)) _Float16 __ocml_cos_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_cvtrtn_f16_f32(float);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_cvtrtp_f16_f32(float);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_cvtrtz_f16_f32(float);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_exp_f16(_Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_exp10_f16(_Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_exp2_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_floor_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_fma_f16(_Float16, _Float16,
                                                          _Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_fmax_f16(_Float16, _Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_fmin_f16(_Float16, _Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_fabs_f16(_Float16);
__attribute__((device)) __attribute__((const)) int __ocml_isinf_f16(_Float16);
__attribute__((device)) __attribute__((const)) int __ocml_isnan_f16(_Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_log_f16(_Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_log10_f16(_Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_log2_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __llvm_amdgcn_rcp_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_rint_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_rsqrt_f16(_Float16);
__attribute__((device)) _Float16 __ocml_sin_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_sqrt_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_trunc_f16(_Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_pown_f16(_Float16, int);

typedef _Float16 __2f16 __attribute__((ext_vector_type(2)));
typedef short __2i16 __attribute__((ext_vector_type(2)));


__attribute__((device)) __attribute__((const)) float __ockl_fdot2(__2f16 a, __2f16 b,
                                                     float c, bool s);




__attribute__((device)) __attribute__((const)) __2f16 __ocml_ceil_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_fabs_2f16(__2f16);
__attribute__((device)) __2f16 __ocml_cos_2f16(__2f16);
__attribute__((device)) __attribute__((pure)) __2f16 __ocml_exp_2f16(__2f16);
__attribute__((device)) __attribute__((pure)) __2f16 __ocml_exp10_2f16(__2f16);
__attribute__((device)) __attribute__((pure)) __2f16 __ocml_exp2_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_floor_2f16(__2f16);
__attribute__((device)) __attribute__((const))
__2f16 __ocml_fma_2f16(__2f16, __2f16, __2f16);
__attribute__((device)) __attribute__((const)) __2i16 __ocml_isinf_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2i16 __ocml_isnan_2f16(__2f16);
__attribute__((device)) __attribute__((pure)) __2f16 __ocml_log_2f16(__2f16);
__attribute__((device)) __attribute__((pure)) __2f16 __ocml_log10_2f16(__2f16);
__attribute__((device)) __attribute__((pure)) __2f16 __ocml_log2_2f16(__2f16);
__attribute__((device)) inline __2f16
__llvm_amdgcn_rcp_2f16(__2f16 __x)
{
  return (__2f16)(__llvm_amdgcn_rcp_f16(__x.x), __llvm_amdgcn_rcp_f16(__x.y));
}
__attribute__((device)) __attribute__((const)) __2f16 __ocml_rint_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_rsqrt_2f16(__2f16);
__attribute__((device)) __2f16 __ocml_sin_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_sqrt_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_trunc_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_pown_2f16(__2f16, __2i16);


}
# 128 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_runtime_wrapper.h" 2 3
# 1 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_math.h" 1 3
# 94 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_math.h" 3
static __attribute__((device)) inline __attribute__((always_inline))
long unsigned int __make_mantissa_base8(const char *__tagp __attribute__((nonnull))) {
  long unsigned int __r = 0;
  while (*__tagp != '\0') {
    char __tmp = *__tagp;

    if (__tmp >= '0' && __tmp <= '7')
      __r = (__r * 8u) + __tmp - '0';
    else
      return 0;

    ++__tagp;
  }

  return __r;
}

static __attribute__((device)) inline __attribute__((always_inline))
long unsigned int __make_mantissa_base10(const char *__tagp __attribute__((nonnull))) {
  long unsigned int __r = 0;
  while (*__tagp != '\0') {
    char __tmp = *__tagp;

    if (__tmp >= '0' && __tmp <= '9')
      __r = (__r * 10u) + __tmp - '0';
    else
      return 0;

    ++__tagp;
  }

  return __r;
}

static __attribute__((device)) inline __attribute__((always_inline))
long unsigned int __make_mantissa_base16(const char *__tagp __attribute__((nonnull))) {
  long unsigned int __r = 0;
  while (*__tagp != '\0') {
    char __tmp = *__tagp;

    if (__tmp >= '0' && __tmp <= '9')
      __r = (__r * 16u) + __tmp - '0';
    else if (__tmp >= 'a' && __tmp <= 'f')
      __r = (__r * 16u) + __tmp - 'a' + 10;
    else if (__tmp >= 'A' && __tmp <= 'F')
      __r = (__r * 16u) + __tmp - 'A' + 10;
    else
      return 0;

    ++__tagp;
  }

  return __r;
}

static __attribute__((device)) inline __attribute__((always_inline))
long unsigned int __make_mantissa(const char *__tagp __attribute__((nonnull))) {
  if (*__tagp == '0') {
    ++__tagp;

    if (*__tagp == 'x' || *__tagp == 'X')
      return __make_mantissa_base16(__tagp);
    else
      return __make_mantissa_base8(__tagp);
  }

  return __make_mantissa_base10(__tagp);
}



static __attribute__((device)) inline __attribute__((always_inline))
int abs(int __x) {
  int __sgn = __x >> (sizeof(int) * 8 - 1);
  return (__x ^ __sgn) - __sgn;
}
static __attribute__((device)) inline __attribute__((always_inline))
long labs(long __x) {
  long __sgn = __x >> (sizeof(long) * 8 - 1);
  return (__x ^ __sgn) - __sgn;
}
static __attribute__((device)) inline __attribute__((always_inline))
long long llabs(long long __x) {
  long long __sgn = __x >> (sizeof(long long) * 8 - 1);
  return (__x ^ __sgn) - __sgn;
}


static __attribute__((device)) inline __attribute__((always_inline))
float acosf(float __x) { return __ocml_acos_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float acoshf(float __x) { return __ocml_acosh_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float asinf(float __x) { return __ocml_asin_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float asinhf(float __x) { return __ocml_asinh_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float atan2f(float __x, float __y) { return __ocml_atan2_f32(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
float atanf(float __x) { return __ocml_atan_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float atanhf(float __x) { return __ocml_atanh_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float cbrtf(float __x) { return __ocml_cbrt_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float ceilf(float __x) { return __ocml_ceil_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float copysignf(float __x, float __y) { return __ocml_copysign_f32(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
float cosf(float __x) { return __ocml_cos_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float coshf(float __x) { return __ocml_cosh_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float cospif(float __x) { return __ocml_cospi_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float cyl_bessel_i0f(float __x) { return __ocml_i0_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float cyl_bessel_i1f(float __x) { return __ocml_i1_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float erfcf(float __x) { return __ocml_erfc_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float erfcinvf(float __x) { return __ocml_erfcinv_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float erfcxf(float __x) { return __ocml_erfcx_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float erff(float __x) { return __ocml_erf_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float erfinvf(float __x) { return __ocml_erfinv_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float exp10f(float __x) { return __ocml_exp10_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float exp2f(float __x) { return __ocml_exp2_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float expf(float __x) { return __ocml_exp_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float expm1f(float __x) { return __ocml_expm1_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float fabsf(float __x) { return __builtin_fabsf(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float fdimf(float __x, float __y) { return __ocml_fdim_f32(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
float fdividef(float __x, float __y) { return __x / __y; }

static __attribute__((device)) inline __attribute__((always_inline))
float floorf(float __x) { return __ocml_floor_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float fmaf(float __x, float __y, float __z) {
  return __ocml_fma_f32(__x, __y, __z);
}

static __attribute__((device)) inline __attribute__((always_inline))
float fmaxf(float __x, float __y) { return __ocml_fmax_f32(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
float fminf(float __x, float __y) { return __ocml_fmin_f32(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
float fmodf(float __x, float __y) { return __ocml_fmod_f32(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
float frexpf(float __x, int *__nptr) {
  int __tmp;



  float __r =
      __ocml_frexp_f32(__x, (__attribute__((address_space(5))) int *)&__tmp);
  *__nptr = __tmp;

  return __r;
}

static __attribute__((device)) inline __attribute__((always_inline))
float hypotf(float __x, float __y) { return __ocml_hypot_f32(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
int ilogbf(float __x) { return __ocml_ilogb_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
bool __finitef(float __x) { return __ocml_isfinite_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
bool __isinff(float __x) { return __ocml_isinf_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
bool __isnanf(float __x) { return __ocml_isnan_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float j0f(float __x) { return __ocml_j0_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float j1f(float __x) { return __ocml_j1_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float jnf(int __n, float __x) {



  if (__n == 0)
    return j0f(__x);
  if (__n == 1)
    return j1f(__x);

  float __x0 = j0f(__x);
  float __x1 = j1f(__x);
  for (int __i = 1; __i < __n; ++__i) {
    float __x2 = (2 * __i) / __x * __x1 - __x0;
    __x0 = __x1;
    __x1 = __x2;
  }

  return __x1;
}

static __attribute__((device)) inline __attribute__((always_inline))
float ldexpf(float __x, int __e) { return __ocml_ldexp_f32(__x, __e); }

static __attribute__((device)) inline __attribute__((always_inline))
float lgammaf(float __x) { return __ocml_lgamma_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
long long int llrintf(float __x) { return __ocml_rint_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
long long int llroundf(float __x) { return __ocml_round_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float log10f(float __x) { return __ocml_log10_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float log1pf(float __x) { return __ocml_log1p_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float log2f(float __x) { return __ocml_log2_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float log2fi(int __x) { return __ocml_log2_f32((float) __x); }

static __attribute__((device)) inline __attribute__((always_inline))
float logbf(float __x) { return __ocml_logb_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float logf(float __x) { return __ocml_log_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
long int lrintf(float __x) { return __ocml_rint_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
long int lroundf(float __x) { return __ocml_round_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float modff(float __x, float *__iptr) {
  float __tmp;



  float __r =
      __ocml_modf_f32(__x, (__attribute__((address_space(5))) float *)&__tmp);
  *__iptr = __tmp;
  return __r;
}



static __attribute__((device)) inline __attribute__((always_inline))
float nanf(const char *__tagp __attribute__((nonnull))) {
  union {
    float val;
    struct ieee_float {
      unsigned int mantissa : 22;
      unsigned int quiet : 1;
      unsigned int exponent : 8;
      unsigned int sign : 1;
    } bits;
  } __tmp;
  static_assert((sizeof(__tmp.val)) == (sizeof(__tmp.bits)), "");

  __tmp.bits.sign = 0u;
  __tmp.bits.exponent = ~0u;
  __tmp.bits.quiet = 1u;
  __tmp.bits.mantissa = __make_mantissa(__tagp);

  return __tmp.val;
}


static __attribute__((device)) inline __attribute__((always_inline))
float nearbyintf(float __x) { return __ocml_nearbyint_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float nextafterf(float __x, float __y) {
  return __ocml_nextafter_f32(__x, __y);
}

static __attribute__((device)) inline __attribute__((always_inline))
float norm3df(float __x, float __y, float __z) {
  return __ocml_len3_f32(__x, __y, __z);
}

static __attribute__((device)) inline __attribute__((always_inline))
float norm4df(float __x, float __y, float __z, float __w) {
  return __ocml_len4_f32(__x, __y, __z, __w);
}

static __attribute__((device)) inline __attribute__((always_inline))
float normcdff(float __x) { return __ocml_ncdf_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float normcdfinvf(float __x) { return __ocml_ncdfinv_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float normf(int __dim,
            const float *__a) {
  float __r = 0;
  while (__dim--) {
    __r += __a[0] * __a[0];
    ++__a;
  }

  return __ocml_sqrt_f32(__r);
}

static __attribute__((device)) inline __attribute__((always_inline))
float powf(float __x, float __y) { return __ocml_pow_f32(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
float powif(float __x, int __y) { return __ocml_pown_f32(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
int powii(int __base, int __exp) {
  if (__exp < 0 )
    return -1;
  int __result = 1;
  for (;;) {
    if (__exp & 1)
      __result *= __base;
    __exp >>= 1;
    if (!__exp)
      break;
    __base *= __base;
  }
  return __result;
}

static __attribute__((device)) inline __attribute__((always_inline))
float rcbrtf(float __x) { return __ocml_rcbrt_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float remainderf(float __x, float __y) {
  return __ocml_remainder_f32(__x, __y);
}

static __attribute__((device)) inline __attribute__((always_inline))
float remquof(float __x, float __y, int *__quo) {
  int __tmp;



  float __r = __ocml_remquo_f32(
      __x, __y, (__attribute__((address_space(5))) int *)&__tmp);
  *__quo = __tmp;

  return __r;
}

static __attribute__((device)) inline __attribute__((always_inline))
float rhypotf(float __x, float __y) { return __ocml_rhypot_f32(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
float rintf(float __x) { return __ocml_rint_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float rnorm3df(float __x, float __y, float __z) {
  return __ocml_rlen3_f32(__x, __y, __z);
}

static __attribute__((device)) inline __attribute__((always_inline))
float rnorm4df(float __x, float __y, float __z, float __w) {
  return __ocml_rlen4_f32(__x, __y, __z, __w);
}

static __attribute__((device)) inline __attribute__((always_inline))
float rnormf(int __dim,
             const float *__a) {
  float __r = 0;
  while (__dim--) {
    __r += __a[0] * __a[0];
    ++__a;
  }

  return __ocml_rsqrt_f32(__r);
}

static __attribute__((device)) inline __attribute__((always_inline))
float roundf(float __x) { return __ocml_round_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float rsqrtf(float __x) { return __ocml_rsqrt_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float scalblnf(float __x, long int __n) {
  return (__n < 9223372036854775807L) ? __ocml_scalbn_f32(__x, __n)
                         : __ocml_scalb_f32(__x, __n);
}

static __attribute__((device)) inline __attribute__((always_inline))
float scalbnf(float __x, int __n) { return __ocml_scalbn_f32(__x, __n); }

static __attribute__((device)) inline __attribute__((always_inline))
bool __signbitf(float __x) { return __ocml_signbit_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
void sincosf(float __x, float *__sinptr, float *__cosptr) {
  float __tmp;



  *__sinptr =
      __ocml_sincos_f32(__x, (__attribute__((address_space(5))) float *)&__tmp);
  *__cosptr = __tmp;
}

static __attribute__((device)) inline __attribute__((always_inline))
void sincospif(float __x, float *__sinptr, float *__cosptr) {
  float __tmp;



  *__sinptr = __ocml_sincospi_f32(
      __x, (__attribute__((address_space(5))) float *)&__tmp);
  *__cosptr = __tmp;
}

static __attribute__((device)) inline __attribute__((always_inline))
float sinf(float __x) { return __ocml_sin_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float sinhf(float __x) { return __ocml_sinh_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float sinpif(float __x) { return __ocml_sinpi_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float sqrtf(float __x) { return __ocml_sqrt_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float tanf(float __x) { return __ocml_tan_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float tanhf(float __x) { return __ocml_tanh_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float tgammaf(float __x) { return __ocml_tgamma_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float truncf(float __x) { return __ocml_trunc_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float y0f(float __x) { return __ocml_y0_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float y1f(float __x) { return __ocml_y1_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float ynf(int __n, float __x) {




  if (__n == 0)
    return y0f(__x);
  if (__n == 1)
    return y1f(__x);

  float __x0 = y0f(__x);
  float __x1 = y1f(__x);
  for (int __i = 1; __i < __n; ++__i) {
    float __x2 = (2 * __i) / __x * __x1 - __x0;
    __x0 = __x1;
    __x1 = __x2;
  }

  return __x1;
}



static __attribute__((device)) inline __attribute__((always_inline))
float __cosf(float __x) { return __ocml_native_cos_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float __exp10f(float __x) { return __ocml_native_exp10_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float __expf(float __x) { return __ocml_native_exp_f32(__x); }
# 627 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_math.h" 3
static __attribute__((device)) inline __attribute__((always_inline))
float __fadd_rn(float __x, float __y) { return __x + __y; }
# 641 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_math.h" 3
static __attribute__((device)) inline __attribute__((always_inline))
float __fdiv_rn(float __x, float __y) { return __x / __y; }


static __attribute__((device)) inline __attribute__((always_inline))
float __fdividef(float __x, float __y) { return __x / __y; }
# 666 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_math.h" 3
static __attribute__((device)) inline __attribute__((always_inline))
float __fmaf_rn(float __x, float __y, float __z) {
  return __ocml_fma_f32(__x, __y, __z);
}
# 682 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_math.h" 3
static __attribute__((device)) inline __attribute__((always_inline))
float __fmul_rn(float __x, float __y) { return __x * __y; }
# 696 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_math.h" 3
static __attribute__((device)) inline __attribute__((always_inline))
float __frcp_rn(float __x) { return 1.0f / __x; }


static __attribute__((device)) inline __attribute__((always_inline))
float __frsqrt_rn(float __x) { return __llvm_amdgcn_rsq_f32(__x); }
# 713 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_math.h" 3
static __attribute__((device)) inline __attribute__((always_inline))
float __fsqrt_rn(float __x) { return __ocml_native_sqrt_f32(__x); }
# 727 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_math.h" 3
static __attribute__((device)) inline __attribute__((always_inline))
float __fsub_rn(float __x, float __y) { return __x - __y; }


static __attribute__((device)) inline __attribute__((always_inline))
float __log10f(float __x) { return __ocml_native_log10_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float __log2f(float __x) { return __ocml_native_log2_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float __logf(float __x) { return __ocml_native_log_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float __powf(float __x, float __y) { return __ocml_pow_f32(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
float __saturatef(float __x) { return (__x < 0) ? 0 : ((__x > 1) ? 1 : __x); }

static __attribute__((device)) inline __attribute__((always_inline))
void __sincosf(float __x, float *__sinptr, float *__cosptr) {
  *__sinptr = __ocml_native_sin_f32(__x);
  *__cosptr = __ocml_native_cos_f32(__x);
}

static __attribute__((device)) inline __attribute__((always_inline))
float __sinf(float __x) { return __ocml_native_sin_f32(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
float __tanf(float __x) { return __ocml_tan_f32(__x); }




static __attribute__((device)) inline __attribute__((always_inline))
double acos(double __x) { return __ocml_acos_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double acosh(double __x) { return __ocml_acosh_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double asin(double __x) { return __ocml_asin_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double asinh(double __x) { return __ocml_asinh_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double atan(double __x) { return __ocml_atan_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double atan2(double __x, double __y) { return __ocml_atan2_f64(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
double atanh(double __x) { return __ocml_atanh_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double cbrt(double __x) { return __ocml_cbrt_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double ceil(double __x) { return __ocml_ceil_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double copysign(double __x, double __y) {
  return __ocml_copysign_f64(__x, __y);
}

static __attribute__((device)) inline __attribute__((always_inline))
double cos(double __x) { return __ocml_cos_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double cosh(double __x) { return __ocml_cosh_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double cospi(double __x) { return __ocml_cospi_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double cyl_bessel_i0(double __x) { return __ocml_i0_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double cyl_bessel_i1(double __x) { return __ocml_i1_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double erf(double __x) { return __ocml_erf_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double erfc(double __x) { return __ocml_erfc_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double erfcinv(double __x) { return __ocml_erfcinv_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double erfcx(double __x) { return __ocml_erfcx_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double erfinv(double __x) { return __ocml_erfinv_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double exp(double __x) { return __ocml_exp_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double exp10(double __x) { return __ocml_exp10_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double exp2(double __x) { return __ocml_exp2_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double expm1(double __x) { return __ocml_expm1_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double fabs(double __x) { return __builtin_fabs(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double fdim(double __x, double __y) { return __ocml_fdim_f64(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
double floor(double __x) { return __ocml_floor_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double fma(double __x, double __y, double __z) {
  return __ocml_fma_f64(__x, __y, __z);
}

static __attribute__((device)) inline __attribute__((always_inline))
double fmax(double __x, double __y) { return __ocml_fmax_f64(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
double fmin(double __x, double __y) { return __ocml_fmin_f64(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
double fmod(double __x, double __y) { return __ocml_fmod_f64(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
double frexp(double __x, int *__nptr) {
  int __tmp;



  double __r =
      __ocml_frexp_f64(__x, (__attribute__((address_space(5))) int *)&__tmp);
  *__nptr = __tmp;
  return __r;
}

static __attribute__((device)) inline __attribute__((always_inline))
double hypot(double __x, double __y) { return __ocml_hypot_f64(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
int ilogb(double __x) { return __ocml_ilogb_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
bool __finite(double __x) { return __ocml_isfinite_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
bool __isinf(double __x) { return __ocml_isinf_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
bool __isnan(double __x) { return __ocml_isnan_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double j0(double __x) { return __ocml_j0_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double j1(double __x) { return __ocml_j1_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double jn(int __n, double __x) {




  if (__n == 0)
    return j0(__x);
  if (__n == 1)
    return j1(__x);

  double __x0 = j0(__x);
  double __x1 = j1(__x);
  for (int __i = 1; __i < __n; ++__i) {
    double __x2 = (2 * __i) / __x * __x1 - __x0;
    __x0 = __x1;
    __x1 = __x2;
  }
  return __x1;
}

static __attribute__((device)) inline __attribute__((always_inline))
double ldexp(double __x, int __e) { return __ocml_ldexp_f64(__x, __e); }

static __attribute__((device)) inline __attribute__((always_inline))
double lgamma(double __x) { return __ocml_lgamma_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
long long int llrint(double __x) { return __ocml_rint_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
long long int llround(double __x) { return __ocml_round_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double log(double __x) { return __ocml_log_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double log10(double __x) { return __ocml_log10_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double log1p(double __x) { return __ocml_log1p_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double log2(double __x) { return __ocml_log2_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double logb(double __x) { return __ocml_logb_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
long int lrint(double __x) { return __ocml_rint_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
long int lround(double __x) { return __ocml_round_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double modf(double __x, double *__iptr) {
  double __tmp;



  double __r =
      __ocml_modf_f64(__x, (__attribute__((address_space(5))) double *)&__tmp);
  *__iptr = __tmp;

  return __r;
}



static __attribute__((device)) inline __attribute__((always_inline))
double nan(const char *__tagp) {

  union {
    double val;
    struct ieee_double {
      long unsigned int mantissa : 51;
      unsigned int quiet : 1;
      unsigned int exponent : 11;
      unsigned int sign : 1;
    } bits;
  } __tmp;
  static_assert((sizeof(__tmp.val)) == (sizeof(__tmp.bits)), "");

  __tmp.bits.sign = 0u;
  __tmp.bits.exponent = ~0u;
  __tmp.bits.quiet = 1u;
  __tmp.bits.mantissa = __make_mantissa(__tagp);

  return __tmp.val;






}


static __attribute__((device)) inline __attribute__((always_inline))
double nearbyint(double __x) { return __ocml_nearbyint_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double nextafter(double __x, double __y) {
  return __ocml_nextafter_f64(__x, __y);
}

static __attribute__((device)) inline __attribute__((always_inline))
double norm(int __dim,
            const double *__a) {
  double __r = 0;
  while (__dim--) {
    __r += __a[0] * __a[0];
    ++__a;
  }

  return __ocml_sqrt_f64(__r);
}

static __attribute__((device)) inline __attribute__((always_inline))
double norm3d(double __x, double __y, double __z) {
  return __ocml_len3_f64(__x, __y, __z);
}

static __attribute__((device)) inline __attribute__((always_inline))
double norm4d(double __x, double __y, double __z, double __w) {
  return __ocml_len4_f64(__x, __y, __z, __w);
}

static __attribute__((device)) inline __attribute__((always_inline))
double normcdf(double __x) { return __ocml_ncdf_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double normcdfinv(double __x) { return __ocml_ncdfinv_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double pow(double __x, double __y) { return __ocml_pow_f64(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
double powi(double __x, int __y) { return __ocml_pown_f64(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
double rcbrt(double __x) { return __ocml_rcbrt_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double remainder(double __x, double __y) {
  return __ocml_remainder_f64(__x, __y);
}

static __attribute__((device)) inline __attribute__((always_inline))
double remquo(double __x, double __y, int *__quo) {
  int __tmp;



  double __r = __ocml_remquo_f64(
      __x, __y, (__attribute__((address_space(5))) int *)&__tmp);
  *__quo = __tmp;

  return __r;
}

static __attribute__((device)) inline __attribute__((always_inline))
double rhypot(double __x, double __y) { return __ocml_rhypot_f64(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
double rint(double __x) { return __ocml_rint_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double rnorm(int __dim,
             const double *__a) {
  double __r = 0;
  while (__dim--) {
    __r += __a[0] * __a[0];
    ++__a;
  }

  return __ocml_rsqrt_f64(__r);
}

static __attribute__((device)) inline __attribute__((always_inline))
double rnorm3d(double __x, double __y, double __z) {
  return __ocml_rlen3_f64(__x, __y, __z);
}

static __attribute__((device)) inline __attribute__((always_inline))
double rnorm4d(double __x, double __y, double __z, double __w) {
  return __ocml_rlen4_f64(__x, __y, __z, __w);
}

static __attribute__((device)) inline __attribute__((always_inline))
double round(double __x) { return __ocml_round_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double rsqrt(double __x) { return __ocml_rsqrt_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double scalbln(double __x, long int __n) {
  return (__n < 9223372036854775807L) ? __ocml_scalbn_f64(__x, __n)
                         : __ocml_scalb_f64(__x, __n);
}
static __attribute__((device)) inline __attribute__((always_inline))
double scalbn(double __x, int __n) { return __ocml_scalbn_f64(__x, __n); }

static __attribute__((device)) inline __attribute__((always_inline))
bool __signbit(double __x) { return __ocml_signbit_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double sin(double __x) { return __ocml_sin_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
void sincos(double __x, double *__sinptr, double *__cosptr) {
  double __tmp;



  *__sinptr = __ocml_sincos_f64(
      __x, (__attribute__((address_space(5))) double *)&__tmp);
  *__cosptr = __tmp;
}

static __attribute__((device)) inline __attribute__((always_inline))
void sincospi(double __x, double *__sinptr, double *__cosptr) {
  double __tmp;



  *__sinptr = __ocml_sincospi_f64(
      __x, (__attribute__((address_space(5))) double *)&__tmp);
  *__cosptr = __tmp;
}

static __attribute__((device)) inline __attribute__((always_inline))
double sinh(double __x) { return __ocml_sinh_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double sinpi(double __x) { return __ocml_sinpi_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double sqrt(double __x) { return __ocml_sqrt_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double tan(double __x) { return __ocml_tan_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double tanh(double __x) { return __ocml_tanh_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double tgamma(double __x) { return __ocml_tgamma_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double trunc(double __x) { return __ocml_trunc_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double y0(double __x) { return __ocml_y0_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double y1(double __x) { return __ocml_y1_f64(__x); }

static __attribute__((device)) inline __attribute__((always_inline))
double yn(int __n, double __x) {




  if (__n == 0)
    return y0(__x);
  if (__n == 1)
    return y1(__x);

  double __x0 = y0(__x);
  double __x1 = y1(__x);
  for (int __i = 1; __i < __n; ++__i) {
    double __x2 = (2 * __i) / __x * __x1 - __x0;
    __x0 = __x1;
    __x1 = __x2;
  }

  return __x1;
}
# 1190 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_math.h" 3
static __attribute__((device)) inline __attribute__((always_inline))
double __dadd_rn(double __x, double __y) { return __x + __y; }
# 1212 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_math.h" 3
static __attribute__((device)) inline __attribute__((always_inline))
double __ddiv_rn(double __x, double __y) { return __x / __y; }
# 1234 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_math.h" 3
static __attribute__((device)) inline __attribute__((always_inline))
double __dmul_rn(double __x, double __y) { return __x * __y; }
# 1248 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_math.h" 3
static __attribute__((device)) inline __attribute__((always_inline))
double __drcp_rn(double __x) { return 1.0 / __x; }
# 1262 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_math.h" 3
static __attribute__((device)) inline __attribute__((always_inline))
double __dsqrt_rn(double __x) { return __ocml_sqrt_f64(__x); }
# 1284 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_math.h" 3
static __attribute__((device)) inline __attribute__((always_inline))
double __dsub_rn(double __x, double __y) { return __x - __y; }
# 1306 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_math.h" 3
static __attribute__((device)) inline __attribute__((always_inline))
double __fma_rn(double __x, double __y, double __z) {
  return __ocml_fma_f64(__x, __y, __z);
}
# 1325 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_math.h" 3
template <class T> static __attribute__((device)) inline __attribute__((always_inline)) T min(T __arg1, T __arg2) {
  return (__arg1 < __arg2) ? __arg1 : __arg2;
}

template <class T> static __attribute__((device)) inline __attribute__((always_inline)) T max(T __arg1, T __arg2) {
  return (__arg1 > __arg2) ? __arg1 : __arg2;
}


static __attribute__((device)) inline __attribute__((always_inline)) int min(int __arg1, int __arg2) {
  return (__arg1 < __arg2) ? __arg1 : __arg2;
}
static __attribute__((device)) inline __attribute__((always_inline)) int max(int __arg1, int __arg2) {
  return (__arg1 > __arg2) ? __arg1 : __arg2;
}

static __attribute__((device)) inline __attribute__((always_inline))
float max(float __x, float __y) { return fmaxf(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
double max(double __x, double __y) { return fmax(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
float min(float __x, float __y) { return fminf(__x, __y); }

static __attribute__((device)) inline __attribute__((always_inline))
double min(double __x, double __y) { return fmin(__x, __y); }
# 129 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_runtime_wrapper.h" 2 3
# 1 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_stdlib.h" 1 3
# 130 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_runtime_wrapper.h" 2 3


# 1 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_cmath.h" 1 3
# 41 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_cmath.h" 3
static __attribute__((device)) inline __attribute__((always_inline)) double abs(double __x) { return ::fabs(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float abs(float __x) { return ::fabsf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) long long abs(long long __n) { return ::llabs(__n); }
static __attribute__((device)) inline __attribute__((always_inline)) long abs(long __n) { return ::labs(__n); }
static __attribute__((device)) inline __attribute__((always_inline)) float fma(float __x, float __y, float __z) {
  return ::fmaf(__x, __y, __z);
}
# 61 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_cmath.h" 3
static __attribute__((device)) inline __attribute__((always_inline)) float frexp(float __arg, int *__exp) {
  return ::frexpf(__arg, __exp);
}
# 93 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_cmath.h" 3
static __attribute__((device)) inline __attribute__((always_inline)) bool isinf(float __x) { return ::__isinff(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) bool isinf(double __x) { return ::__isinf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) bool isfinite(float __x) { return ::__finitef(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) bool isfinite(double __x) { return ::__finite(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) bool isnan(float __x) { return ::__isnanf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) bool isnan(double __x) { return ::__isnan(__x); }





static __attribute__((device)) inline __attribute__((always_inline)) bool isgreater(float __x, float __y) {
  return __builtin_isgreater(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isgreater(double __x, double __y) {
  return __builtin_isgreater(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isgreaterequal(float __x, float __y) {
  return __builtin_isgreaterequal(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isgreaterequal(double __x, double __y) {
  return __builtin_isgreaterequal(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isless(float __x, float __y) {
  return __builtin_isless(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isless(double __x, double __y) {
  return __builtin_isless(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool islessequal(float __x, float __y) {
  return __builtin_islessequal(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool islessequal(double __x, double __y) {
  return __builtin_islessequal(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool islessgreater(float __x, float __y) {
  return __builtin_islessgreater(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool islessgreater(double __x, double __y) {
  return __builtin_islessgreater(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isnormal(float __x) {
  return __builtin_isnormal(__x);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isnormal(double __x) {
  return __builtin_isnormal(__x);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isunordered(float __x, float __y) {
  return __builtin_isunordered(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isunordered(double __x, double __y) {
  return __builtin_isunordered(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) float modf(float __x, float *__iptr) {
  return ::modff(__x, __iptr);
}
static __attribute__((device)) inline __attribute__((always_inline)) float pow(float __base, int __iexp) {
  return ::powif(__base, __iexp);
}
static __attribute__((device)) inline __attribute__((always_inline)) double pow(double __base, int __iexp) {
  return ::powi(__base, __iexp);
}
static __attribute__((device)) inline __attribute__((always_inline)) float remquo(float __x, float __y, int *__quo) {
  return ::remquof(__x, __y, __quo);
}
static __attribute__((device)) inline __attribute__((always_inline)) float scalbln(float __x, long int __n) {
  return ::scalblnf(__x, __n);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool signbit(float __x) { return ::__signbitf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) bool signbit(double __x) { return ::__signbit(__x); }






static __attribute__((device)) inline __attribute__((always_inline)) _Float16 fma(_Float16 __x, _Float16 __y,
                                      _Float16 __z) {
  return __ocml_fma_f16(__x, __y, __z);
}
static __attribute__((device)) inline __attribute__((always_inline)) _Float16 pow(_Float16 __base, int __iexp) {
  return __ocml_pown_f16(__base, __iexp);
}
# 202 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_cmath.h" 3
static __attribute__((device)) inline __attribute__((always_inline)) float acos(float __x) { return acosf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float acosh(float __x) { return acoshf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float asin(float __x) { return asinf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float asinh(float __x) { return asinhf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float atan(float __x) { return atanf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float atan2(float __x, float __y) { return atan2f(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float atanh(float __x) { return atanhf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float cbrt(float __x) { return cbrtf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float ceil(float __x) { return ceilf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float copysign(float __x, float __y) { return copysignf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float cos(float __x) { return cosf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float cosh(float __x) { return coshf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float erf(float __x) { return erff(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float erfc(float __x) { return erfcf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float exp(float __x) { return expf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float exp2(float __x) { return exp2f(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float expm1(float __x) { return expm1f(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float fabs(float __x) { return fabsf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float fdim(float __x, float __y) { return fdimf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float floor(float __x) { return floorf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float fmax(float __x, float __y) { return fmaxf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float fmin(float __x, float __y) { return fminf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float fmod(float __x, float __y) { return fmodf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float hypot(float __x, float __y) { return hypotf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) int ilogb(float __x) { return ilogbf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float ldexp(float __x, int __y) { return ldexpf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float lgamma(float __x) { return lgammaf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float log(float __x) { return logf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float log10(float __x) { return log10f(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float log1p(float __x) { return log1pf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float log2(float __x) { return log2f(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float logb(float __x) { return logbf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) long long llrint(float __x) { return llrintf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) long long llround(float __x) { return llroundf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) long lrint(float __x) { return lrintf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) long lround(float __x) { return lroundf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float nearbyint(float __x) { return nearbyintf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float nextafter(float __x, float __y) { return nextafterf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float pow(float __x, float __y) { return powf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float remainder(float __x, float __y) { return remainderf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float rint(float __x) { return rintf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float round(float __x) { return roundf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float scalbn(float __x, int __y) { return scalbnf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float sin(float __x) { return sinf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float sinh(float __x) { return sinhf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float sqrt(float __x) { return sqrtf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float tan(float __x) { return tanf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float tanh(float __x) { return tanhf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float tgamma(float __x) { return tgammaf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float trunc(float __x) { return truncf(__x); }
# 265 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_cmath.h" 3
template <bool __B, class __T = void> struct __hip_enable_if {};

template <class __T> struct __hip_enable_if<true, __T> { typedef __T type; };

namespace __hip {
template <class _Tp> struct is_integral {
  enum { value = 0 };
};
template <> struct is_integral<bool> {
  enum { value = 1 };
};
template <> struct is_integral<char> {
  enum { value = 1 };
};
template <> struct is_integral<signed char> {
  enum { value = 1 };
};
template <> struct is_integral<unsigned char> {
  enum { value = 1 };
};
template <> struct is_integral<wchar_t> {
  enum { value = 1 };
};
template <> struct is_integral<short> {
  enum { value = 1 };
};
template <> struct is_integral<unsigned short> {
  enum { value = 1 };
};
template <> struct is_integral<int> {
  enum { value = 1 };
};
template <> struct is_integral<unsigned int> {
  enum { value = 1 };
};
template <> struct is_integral<long> {
  enum { value = 1 };
};
template <> struct is_integral<unsigned long> {
  enum { value = 1 };
};
template <> struct is_integral<long long> {
  enum { value = 1 };
};
template <> struct is_integral<unsigned long long> {
  enum { value = 1 };
};


template <class _Tp> struct is_arithmetic {
  enum { value = 0 };
};
template <> struct is_arithmetic<bool> {
  enum { value = 1 };
};
template <> struct is_arithmetic<char> {
  enum { value = 1 };
};
template <> struct is_arithmetic<signed char> {
  enum { value = 1 };
};
template <> struct is_arithmetic<unsigned char> {
  enum { value = 1 };
};
template <> struct is_arithmetic<wchar_t> {
  enum { value = 1 };
};
template <> struct is_arithmetic<short> {
  enum { value = 1 };
};
template <> struct is_arithmetic<unsigned short> {
  enum { value = 1 };
};
template <> struct is_arithmetic<int> {
  enum { value = 1 };
};
template <> struct is_arithmetic<unsigned int> {
  enum { value = 1 };
};
template <> struct is_arithmetic<long> {
  enum { value = 1 };
};
template <> struct is_arithmetic<unsigned long> {
  enum { value = 1 };
};
template <> struct is_arithmetic<long long> {
  enum { value = 1 };
};
template <> struct is_arithmetic<unsigned long long> {
  enum { value = 1 };
};
template <> struct is_arithmetic<float> {
  enum { value = 1 };
};
template <> struct is_arithmetic<double> {
  enum { value = 1 };
};

struct true_type {
  static const __attribute__((constant)) bool value = true;
};
struct false_type {
  static const __attribute__((constant)) bool value = false;
};

template <typename __T, typename __U> struct is_same : public false_type {};
template <typename __T> struct is_same<__T, __T> : public true_type {};

template <typename __T> struct add_rvalue_reference { typedef __T &&type; };

template <typename __T> typename add_rvalue_reference<__T>::type declval();




template <class _Tp> struct __numeric_type {
  static void __test(...);
  static _Float16 __test(_Float16);
  static float __test(float);
  static double __test(char);
  static double __test(int);
  static double __test(unsigned);
  static double __test(long);
  static double __test(unsigned long);
  static double __test(long long);
  static double __test(unsigned long long);
  static double __test(double);

  static double __test(long double);

  typedef decltype(__test(declval<_Tp>())) type;
  static const bool value = !is_same<type, void>::value;
};

template <> struct __numeric_type<void> { static const bool value = true; };

template <class _A1, class _A2 = void, class _A3 = void,
          bool = __numeric_type<_A1>::value &&__numeric_type<_A2>::value
              &&__numeric_type<_A3>::value>
class __promote_imp {
public:
  static const bool value = false;
};

template <class _A1, class _A2, class _A3>
class __promote_imp<_A1, _A2, _A3, true> {
private:
  typedef typename __promote_imp<_A1>::type __type1;
  typedef typename __promote_imp<_A2>::type __type2;
  typedef typename __promote_imp<_A3>::type __type3;

public:
  typedef decltype(__type1() + __type2() + __type3()) type;
  static const bool value = true;
};

template <class _A1, class _A2> class __promote_imp<_A1, _A2, void, true> {
private:
  typedef typename __promote_imp<_A1>::type __type1;
  typedef typename __promote_imp<_A2>::type __type2;

public:
  typedef decltype(__type1() + __type2()) type;
  static const bool value = true;
};

template <class _A1> class __promote_imp<_A1, void, void, true> {
public:
  typedef typename __numeric_type<_A1>::type type;
  static const bool value = true;
};

template <class _A1, class _A2 = void, class _A3 = void>
class __promote : public __promote_imp<_A1, _A2, _A3> {};

}
# 478 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_cmath.h" 3
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type acos(__T __x) { return ::acos((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type acosh(__T __x) { return ::acosh((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type asin(__T __x) { return ::asin((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type asinh(__T __x) { return ::asinh((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type atan(__T __x) { return ::atan((double)__x); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if< __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, typename __hip::__promote<__T1, __T2>::type>::type atan2(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __result_type; return atan2((__result_type)__x, (__result_type)__y); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type atanh(__T __x) { return ::atanh((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type cbrt(__T __x) { return ::cbrt((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type ceil(__T __x) { return ::ceil((double)__x); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if< __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, typename __hip::__promote<__T1, __T2>::type>::type copysign(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __result_type; return copysign((__result_type)__x, (__result_type)__y); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type cos(__T __x) { return ::cos((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type cosh(__T __x) { return ::cosh((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type erf(__T __x) { return ::erf((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type erfc(__T __x) { return ::erfc((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type exp(__T __x) { return ::exp((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type exp2(__T __x) { return ::exp2((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type expm1(__T __x) { return ::expm1((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type fabs(__T __x) { return ::fabs((double)__x); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if< __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, typename __hip::__promote<__T1, __T2>::type>::type fdim(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __result_type; return fdim((__result_type)__x, (__result_type)__y); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type floor(__T __x) { return ::floor((double)__x); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if< __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, typename __hip::__promote<__T1, __T2>::type>::type fmax(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __result_type; return fmax((__result_type)__x, (__result_type)__y); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if< __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, typename __hip::__promote<__T1, __T2>::type>::type fmin(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __result_type; return fmin((__result_type)__x, (__result_type)__y); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if< __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, typename __hip::__promote<__T1, __T2>::type>::type fmod(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __result_type; return fmod((__result_type)__x, (__result_type)__y); }



template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if< __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, typename __hip::__promote<__T1, __T2>::type>::type hypot(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __result_type; return hypot((__result_type)__x, (__result_type)__y); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, int>::type ilogb(__T __x) { return ::ilogb((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, bool>::type isfinite(__T __x) { return ::isfinite((double)__x); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if< __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, typename __hip::__promote<__T1, __T2>::type>::type isgreater(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __result_type; return isgreater((__result_type)__x, (__result_type)__y); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if< __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, typename __hip::__promote<__T1, __T2>::type>::type isgreaterequal(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __result_type; return isgreaterequal((__result_type)__x, (__result_type)__y); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, bool>::type isinf(__T __x) { return ::isinf((double)__x); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if< __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, typename __hip::__promote<__T1, __T2>::type>::type isless(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __result_type; return isless((__result_type)__x, (__result_type)__y); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if< __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, typename __hip::__promote<__T1, __T2>::type>::type islessequal(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __result_type; return islessequal((__result_type)__x, (__result_type)__y); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if< __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, typename __hip::__promote<__T1, __T2>::type>::type islessgreater(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __result_type; return islessgreater((__result_type)__x, (__result_type)__y); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, bool>::type isnan(__T __x) { return ::isnan((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, bool>::type isnormal(__T __x) { return ::isnormal((double)__x); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if< __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, typename __hip::__promote<__T1, __T2>::type>::type isunordered(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __result_type; return isunordered((__result_type)__x, (__result_type)__y); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type lgamma(__T __x) { return ::lgamma((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type log(__T __x) { return ::log((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type log10(__T __x) { return ::log10((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type log1p(__T __x) { return ::log1p((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type log2(__T __x) { return ::log2((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type logb(__T __x) { return ::logb((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, long long>::type llrint(__T __x) { return ::llrint((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, long long>::type llround(__T __x) { return ::llround((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, long>::type lrint(__T __x) { return ::lrint((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, long>::type lround(__T __x) { return ::lround((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type nearbyint(__T __x) { return ::nearbyint((double)__x); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if< __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, typename __hip::__promote<__T1, __T2>::type>::type nextafter(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __result_type; return nextafter((__result_type)__x, (__result_type)__y); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if< __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, typename __hip::__promote<__T1, __T2>::type>::type pow(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __result_type; return pow((__result_type)__x, (__result_type)__y); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if< __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, typename __hip::__promote<__T1, __T2>::type>::type remainder(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __result_type; return remainder((__result_type)__x, (__result_type)__y); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type rint(__T __x) { return ::rint((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type round(__T __x) { return ::round((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, bool>::type signbit(__T __x) { return ::signbit((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type sin(__T __x) { return ::sin((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type sinh(__T __x) { return ::sinh((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type sqrt(__T __x) { return ::sqrt((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type tan(__T __x) { return ::tan((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type tanh(__T __x) { return ::tanh((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type tgamma(__T __x) { return ::tgamma((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type trunc(__T __x) { return ::trunc((double)__x); }


template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if< __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, typename __hip::__promote<__T1, __T2>::type>::type max(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __result_type; return max((__result_type)__x, (__result_type)__y); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if< __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, typename __hip::__promote<__T1, __T2>::type>::type min(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __result_type; return min((__result_type)__x, (__result_type)__y); }



template <typename __T1, typename __T2, typename __T3>
static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<
    __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value &&
        __hip::is_arithmetic<__T3>::value,
    typename __hip::__promote<__T1, __T2, __T3>::type>::type
fma(__T1 __x, __T2 __y, __T3 __z) {
  typedef typename __hip::__promote<__T1, __T2, __T3>::type __result_type;
  return ::fma((__result_type)__x, (__result_type)__y, (__result_type)__z);
}
# 568 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_cmath.h" 3
template <typename __T>
static __attribute__((device)) inline __attribute__((always_inline))
    typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type
    frexp(__T __x, int *__exp) {
  return ::frexp((double)__x, __exp);
}

template <typename __T>
static __attribute__((device)) inline __attribute__((always_inline))
    typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type
    ldexp(__T __x, int __exp) {
  return ::ldexp((double)__x, __exp);
}

template <typename __T>
static __attribute__((device)) inline __attribute__((always_inline))
    typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type
    modf(__T __x, double *__exp) {
  return ::modf((double)__x, __exp);
}


template <typename __T1, typename __T2>
static __attribute__((device)) inline __attribute__((always_inline))
    typename __hip_enable_if<__hip::is_arithmetic<__T1>::value &&
                                 __hip::is_arithmetic<__T2>::value,
                             typename __hip::__promote<__T1, __T2>::type>::type
    remquo(__T1 __x, __T2 __y, int *__quo) {
  typedef typename __hip::__promote<__T1, __T2>::type __result_type;
  return ::remquo((__result_type)__x, (__result_type)__y, __quo);
}
# 610 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_cmath.h" 3
template <typename __T>
static __attribute__((device)) inline __attribute__((always_inline))
    typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type
    scalbln(__T __x, long int __exp) {
  return ::scalbln((double)__x, __exp);
}

template <typename __T>
static __attribute__((device)) inline __attribute__((always_inline))
    typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type
    scalbn(__T __x, int __exp) {
  return ::scalbn((double)__x, __exp);
}
# 133 "/opt/rocm-6.0.0/lib/llvm/lib/clang/17.0.0/include/__clang_hip_runtime_wrapper.h" 2 3
# 2 "<built-in>" 2
# 1 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/out/ubuntu-22.04/22.04/build/hip-on-rocclr/hipamd/src/hiprtc/hip_rtc_gen/hipRTC_header.h" 2




# 1 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/hip-on-vdi/include/hip/hip_runtime.h" 1 3
# 58 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/hip-on-vdi/include/hip/hip_runtime.h" 3
# 1 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/out/ubuntu-22.04/22.04/build/hip-on-rocclr/hipamd/include/hip/hip_version.h" 1 3
# 59 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/hip-on-vdi/include/hip/hip_runtime.h" 2 3
# 1 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/hip-on-vdi/include/hip/hip_common.h" 1 3
# 27 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/hip-on-vdi/include/hip/hip_common.h" 3
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-macro-identifier"
# 97 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/hip-on-vdi/include/hip/hip_common.h" 3
#pragma clang diagnostic pop
# 60 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/hip-on-vdi/include/hip/hip_runtime.h" 2 3


# 1 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h" 1 3
# 32 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h" 3
# 1 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_common.h" 1 3
# 33 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h" 2 3
# 43 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h" 3
extern "C" {
# 54 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h" 3
const char* amd_dbgapi_get_build_name();
# 63 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h" 3
const char* amd_dbgapi_get_git_hash();
# 72 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h" 3
size_t amd_dbgapi_get_build_id();


}
# 92 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h" 3
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef signed int int32_t;
typedef signed long long int64_t;
namespace std {
using ::uint32_t;
using ::uint64_t;
using ::int32_t;
using ::int64_t;
}
# 124 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h" 3
# 1 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/hip_ldg.h" 1 3
# 27 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/hip_ldg.h" 3
# 1 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_vector_types.h" 1 3
# 31 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_vector_types.h" 3
# 1 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/host_defines.h" 1 3
# 38 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/host_defines.h" 3
namespace __hip_internal {
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long long int64_t;

template <class _Tp, _Tp __v> struct integral_constant {
  static constexpr const _Tp value = __v;
  typedef _Tp value_type;
  typedef integral_constant type;
  constexpr operator value_type() const { return value; }
  constexpr value_type operator()() const { return value; }
};
template <class _Tp, _Tp __v> constexpr const _Tp integral_constant<_Tp, __v>::value;

typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;

template <bool B> using bool_constant = integral_constant<bool, B>;
typedef bool_constant<true> true_type;
typedef bool_constant<false> false_type;

template <bool __B, class __T = void> struct enable_if {};
template <class __T> struct enable_if<true, __T> { typedef __T type; };

template<bool _B> struct true_or_false_type : public false_type {};
template<> struct true_or_false_type<true> : public true_type {};

template <class _Tp> struct is_integral : public false_type {};
template <> struct is_integral<bool> : public true_type {};
template <> struct is_integral<char> : public true_type {};
template <> struct is_integral<signed char> : public true_type {};
template <> struct is_integral<unsigned char> : public true_type {};
template <> struct is_integral<wchar_t> : public true_type {};
template <> struct is_integral<short> : public true_type {};
template <> struct is_integral<unsigned short> : public true_type {};
template <> struct is_integral<int> : public true_type {};
template <> struct is_integral<unsigned int> : public true_type {};
template <> struct is_integral<long> : public true_type {};
template <> struct is_integral<unsigned long> : public true_type {};
template <> struct is_integral<long long> : public true_type {};
template <> struct is_integral<unsigned long long> : public true_type {};

template <class _Tp> struct is_arithmetic : public false_type {};
template <> struct is_arithmetic<bool> : public true_type {};
template <> struct is_arithmetic<char> : public true_type {};
template <> struct is_arithmetic<signed char> : public true_type {};
template <> struct is_arithmetic<unsigned char> : public true_type {};
template <> struct is_arithmetic<wchar_t> : public true_type {};
template <> struct is_arithmetic<short> : public true_type {};
template <> struct is_arithmetic<unsigned short> : public true_type {};
template <> struct is_arithmetic<int> : public true_type {};
template <> struct is_arithmetic<unsigned int> : public true_type {};
template <> struct is_arithmetic<long> : public true_type {};
template <> struct is_arithmetic<unsigned long> : public true_type {};
template <> struct is_arithmetic<long long> : public true_type {};
template <> struct is_arithmetic<unsigned long long> : public true_type {};
template <> struct is_arithmetic<float> : public true_type {};
template <> struct is_arithmetic<double> : public true_type {};

template<typename _Tp> struct is_floating_point : public false_type {};
template<> struct is_floating_point<float> : public true_type {};
template<> struct is_floating_point<double> : public true_type {};
template<> struct is_floating_point<long double> : public true_type {};

template <typename __T, typename __U> struct is_same : public false_type {};
template <typename __T> struct is_same<__T, __T> : public true_type {};

template<typename _Tp, bool = is_arithmetic<_Tp>::value>
  struct is_signed : public false_type {};
template<typename _Tp>
  struct is_signed<_Tp, true> : public true_or_false_type<_Tp(-1) < _Tp(0)> {};

template<typename _CharT> struct char_traits;
template<typename _CharT, typename _Traits = char_traits<_CharT>> class basic_istream;
template<typename _CharT, typename _Traits = char_traits<_CharT>> class basic_ostream;
typedef basic_istream<char> istream;
typedef basic_ostream<char> ostream;

template<typename _Tp>
    struct is_standard_layout
    : public integral_constant<bool, __is_standard_layout(_Tp)>
    { };

template<typename _Tp>
    struct is_trivial
    : public integral_constant<bool, __is_trivial(_Tp)>
    { };
}
typedef __hip_internal::uint8_t __hip_uint8_t;
typedef __hip_internal::uint16_t __hip_uint16_t;
typedef __hip_internal::uint32_t __hip_uint32_t;
typedef __hip_internal::uint64_t __hip_uint64_t;
typedef __hip_internal::int8_t __hip_int8_t;
typedef __hip_internal::int16_t __hip_int16_t;
typedef __hip_internal::int32_t __hip_int32_t;
typedef __hip_internal::int64_t __hip_int64_t;
# 32 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_vector_types.h" 2 3
# 52 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_vector_types.h" 3
namespace std {
using ::size_t;

template <class _Tp, _Tp __v> struct integral_constant {
  static constexpr const _Tp value = __v;
  typedef _Tp value_type;
  typedef integral_constant type;
  constexpr operator value_type() const { return value; }
  constexpr value_type operator()() const { return value; }
};
template <class _Tp, _Tp __v> constexpr const _Tp integral_constant<_Tp, __v>::value;

typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;

template <bool B> using bool_constant = integral_constant<bool, B>;
typedef bool_constant<true> true_type;
typedef bool_constant<false> false_type;

template <bool __B, class __T = void> struct enable_if {};
template <class __T> struct enable_if<true, __T> { typedef __T type; };

template<bool _B> struct true_or_false_type : public false_type {};
template<> struct true_or_false_type<true> : public true_type {};

template <class _Tp> struct is_integral : public false_type {};
template <> struct is_integral<bool> : public true_type {};
template <> struct is_integral<char> : public true_type {};
template <> struct is_integral<signed char> : public true_type {};
template <> struct is_integral<unsigned char> : public true_type {};
template <> struct is_integral<wchar_t> : public true_type {};
template <> struct is_integral<short> : public true_type {};
template <> struct is_integral<unsigned short> : public true_type {};
template <> struct is_integral<int> : public true_type {};
template <> struct is_integral<unsigned int> : public true_type {};
template <> struct is_integral<long> : public true_type {};
template <> struct is_integral<unsigned long> : public true_type {};
template <> struct is_integral<long long> : public true_type {};
template <> struct is_integral<unsigned long long> : public true_type {};

template <class _Tp> struct is_arithmetic : public false_type {};
template <> struct is_arithmetic<bool> : public true_type {};
template <> struct is_arithmetic<char> : public true_type {};
template <> struct is_arithmetic<signed char> : public true_type {};
template <> struct is_arithmetic<unsigned char> : public true_type {};
template <> struct is_arithmetic<wchar_t> : public true_type {};
template <> struct is_arithmetic<short> : public true_type {};
template <> struct is_arithmetic<unsigned short> : public true_type {};
template <> struct is_arithmetic<int> : public true_type {};
template <> struct is_arithmetic<unsigned int> : public true_type {};
template <> struct is_arithmetic<long> : public true_type {};
template <> struct is_arithmetic<unsigned long> : public true_type {};
template <> struct is_arithmetic<long long> : public true_type {};
template <> struct is_arithmetic<unsigned long long> : public true_type {};
template <> struct is_arithmetic<float> : public true_type {};
template <> struct is_arithmetic<double> : public true_type {};

template<typename _Tp> struct is_floating_point : public false_type {};
template<> struct is_floating_point<float> : public true_type {};
template<> struct is_floating_point<double> : public true_type {};
template<> struct is_floating_point<long double> : public true_type {};

template <typename __T, typename __U> struct is_same : public false_type {};
template <typename __T> struct is_same<__T, __T> : public true_type {};

template<typename _Tp, bool = is_arithmetic<_Tp>::value>
  struct is_signed : public false_type {};
template<typename _Tp>
  struct is_signed<_Tp, true> : public true_or_false_type<_Tp(-1) < _Tp(0)> {};

template <class _T1, class _T2> struct is_convertible
  : public true_or_false_type<__is_convertible_to(_T1, _T2)> {};

template<typename _CharT> struct char_traits;
template<typename _CharT, typename _Traits = char_traits<_CharT>> class basic_istream;
template<typename _CharT, typename _Traits = char_traits<_CharT>> class basic_ostream;
typedef basic_istream<char> istream;
typedef basic_ostream<char> ostream;

template <typename __T> struct is_scalar : public integral_constant<bool, __is_scalar(__T)> {};
}


    namespace hip_impl {
        inline
        constexpr
        unsigned int next_pot(unsigned int x) {

         return 1u << (32u - __builtin_clz(x - 1u));
        }
    }

    template<typename T, unsigned int n> struct HIP_vector_base;

    template<typename T>
    struct HIP_vector_base<T, 1> {
        using Native_vec_ = T __attribute__((ext_vector_type(1)));

        union {
            Native_vec_ data;
            struct {
                T x;
            };
        };

        using value_type = T;

        __attribute__((device))
        HIP_vector_base() = default;
        __attribute__((device))
        explicit
        constexpr
        HIP_vector_base(T x_) noexcept : data{x_} {}
        __attribute__((device))
        constexpr
        HIP_vector_base(const HIP_vector_base&) = default;
        __attribute__((device))
        constexpr
        HIP_vector_base(HIP_vector_base&&) = default;
        __attribute__((device))
        ~HIP_vector_base() = default;
        __attribute__((device))
        HIP_vector_base& operator=(const HIP_vector_base&) = default;
    };

    template<typename T>
    struct HIP_vector_base<T, 2> {
        using Native_vec_ = T __attribute__((ext_vector_type(2)));

        union



        {
            Native_vec_ data;
            struct {
                T x;
                T y;
            };
        };

        using value_type = T;

        __attribute__((device))
        HIP_vector_base() = default;
        __attribute__((device))
        explicit
        constexpr
        HIP_vector_base(T x_) noexcept : data{x_, x_} {}
        __attribute__((device))
        constexpr
        HIP_vector_base(T x_, T y_) noexcept : data{x_, y_} {}
        __attribute__((device))
        constexpr
        HIP_vector_base(const HIP_vector_base&) = default;
        __attribute__((device))
        constexpr
        HIP_vector_base(HIP_vector_base&&) = default;
        __attribute__((device))
        ~HIP_vector_base() = default;
        __attribute__((device))
        HIP_vector_base& operator=(const HIP_vector_base&) = default;
    };

    template<typename T>
    struct HIP_vector_base<T, 3> {
        struct Native_vec_ {
            T d[3];

            __attribute__((device))
            Native_vec_() = default;

            __attribute__((device))
            explicit
            constexpr
            Native_vec_(T x_) noexcept : d{x_, x_, x_} {}
            __attribute__((device))
            constexpr
            Native_vec_(T x_, T y_, T z_) noexcept : d{x_, y_, z_} {}
            __attribute__((device))
            constexpr
            Native_vec_(const Native_vec_&) = default;
            __attribute__((device))
            constexpr
            Native_vec_(Native_vec_&&) = default;
            __attribute__((device))
            ~Native_vec_() = default;

            __attribute__((device))
            Native_vec_& operator=(const Native_vec_&) = default;
            __attribute__((device))
            Native_vec_& operator=(Native_vec_&&) = default;

            __attribute__((device))
            T& operator[](unsigned int idx) noexcept { return d[idx]; }
            __attribute__((device))
            T operator[](unsigned int idx) const noexcept { return d[idx]; }

            __attribute__((device))
            Native_vec_& operator+=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] += x_.d[i];
                return *this;
            }
            __attribute__((device))
            Native_vec_& operator-=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] -= x_.d[i];
                return *this;
            }

            __attribute__((device))
            Native_vec_& operator*=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] *= x_.d[i];
                return *this;
            }
            __attribute__((device))
            Native_vec_& operator/=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] /= x_.d[i];
                return *this;
            }

            template<
                typename U = T,
                typename std::enable_if<std::is_signed<U>{}>::type* = nullptr>
            __attribute__((device))
            Native_vec_ operator-() const noexcept
            {
                auto r{*this};
                for (auto&& x : r.d) x = -x;
                return r;
            }

            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __attribute__((device))
            Native_vec_ operator~() const noexcept
            {
                auto r{*this};
                for (auto&& x : r.d) x = ~x;
                return r;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __attribute__((device))
            Native_vec_& operator%=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] %= x_.d[i];
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __attribute__((device))
            Native_vec_& operator^=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] ^= x_.d[i];
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __attribute__((device))
            Native_vec_& operator|=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] |= x_.d[i];
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __attribute__((device))
            Native_vec_& operator&=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] &= x_.d[i];
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __attribute__((device))
            Native_vec_& operator>>=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] >>= x_.d[i];
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __attribute__((device))
            Native_vec_& operator<<=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] <<= x_.d[i];
                return *this;
            }






            using Vec3_cmp = int __attribute__((vector_size(4 * sizeof(int))));

            __attribute__((device))
            Vec3_cmp operator==(const Native_vec_& x_) const noexcept
            {
                return Vec3_cmp{d[0] == x_.d[0], d[1] == x_.d[1], d[2] == x_.d[2]};
            }
        };

        union {
            Native_vec_ data;
            struct {
                T x;
                T y;
                T z;
            };
        };

        using value_type = T;

        __attribute__((device))
        HIP_vector_base() = default;
        __attribute__((device))
        explicit
        constexpr
        HIP_vector_base(T x_) noexcept : data{x_, x_, x_} {}
        __attribute__((device))
        constexpr
        HIP_vector_base(T x_, T y_, T z_) noexcept : data{x_, y_, z_} {}
        __attribute__((device))
        constexpr
        HIP_vector_base(const HIP_vector_base&) = default;
        __attribute__((device))
        constexpr
        HIP_vector_base(HIP_vector_base&&) = default;
        __attribute__((device))
        ~HIP_vector_base() = default;

        __attribute__((device))
        HIP_vector_base& operator=(const HIP_vector_base&) = default;
        __attribute__((device))
        HIP_vector_base& operator=(HIP_vector_base&&) = default;
    };

    template<typename T>
    struct HIP_vector_base<T, 4> {
        using Native_vec_ = T __attribute__((ext_vector_type(4)));

        union



        {
            Native_vec_ data;
            struct {
                T x;
                T y;
                T z;
                T w;
            };
        };

        using value_type = T;

        __attribute__((device))
        HIP_vector_base() = default;
        __attribute__((device))
        explicit
        constexpr
        HIP_vector_base(T x_) noexcept : data{x_, x_, x_, x_} {}
        __attribute__((device))
        constexpr
        HIP_vector_base(T x_, T y_, T z_, T w_) noexcept : data{x_, y_, z_, w_} {}
        __attribute__((device))
        constexpr
        HIP_vector_base(const HIP_vector_base&) = default;
        __attribute__((device))
        constexpr
        HIP_vector_base(HIP_vector_base&&) = default;
        __attribute__((device))
        ~HIP_vector_base() = default;
        __attribute__((device))
        HIP_vector_base& operator=(const HIP_vector_base&) = default;
    };

    template<typename T, unsigned int rank>
    struct HIP_vector_type : public HIP_vector_base<T, rank> {
        using HIP_vector_base<T, rank>::data;
        using typename HIP_vector_base<T, rank>::Native_vec_;

        __attribute__((device))
        HIP_vector_type() = default;
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>::value>::type* = nullptr>
        __attribute__((device))
        explicit
        constexpr
        HIP_vector_type(U x_) noexcept
            : HIP_vector_base<T, rank>{static_cast<T>(x_)}
        {}
        template<
            typename... Us,
            typename std::enable_if<
                (rank > 1) && sizeof...(Us) == rank>::type* = nullptr>
        __attribute__((device))
        constexpr
        HIP_vector_type(Us... xs) noexcept
            : HIP_vector_base<T, rank>{static_cast<T>(xs)...}
        {}
        __attribute__((device))
        constexpr
        HIP_vector_type(const HIP_vector_type&) = default;
        __attribute__((device))
        constexpr
        HIP_vector_type(HIP_vector_type&&) = default;
        __attribute__((device))
        ~HIP_vector_type() = default;

        __attribute__((device))
        HIP_vector_type& operator=(const HIP_vector_type&) = default;
        __attribute__((device))
        HIP_vector_type& operator=(HIP_vector_type&&) = default;


        __attribute__((device))
        HIP_vector_type& operator++() noexcept
        {
            return *this += HIP_vector_type{1};
        }
        __attribute__((device))
        HIP_vector_type operator++(int) noexcept
        {
            auto tmp(*this);
            ++*this;
            return tmp;
        }

        __attribute__((device))
        HIP_vector_type& operator--() noexcept
        {
            return *this -= HIP_vector_type{1};
        }
        __attribute__((device))
        HIP_vector_type operator--(int) noexcept
        {
            auto tmp(*this);
            --*this;
            return tmp;
        }

        __attribute__((device))
        HIP_vector_type& operator+=(const HIP_vector_type& x) noexcept
        {
            data += x.data;
            return *this;
        }
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        __attribute__((device))
        HIP_vector_type& operator+=(U x) noexcept
        {
            return *this += HIP_vector_type{x};
        }

        __attribute__((device))
        HIP_vector_type& operator-=(const HIP_vector_type& x) noexcept
        {
            data -= x.data;
            return *this;
        }
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        __attribute__((device))
        HIP_vector_type& operator-=(U x) noexcept
        {
            return *this -= HIP_vector_type{x};
        }

        __attribute__((device))
        HIP_vector_type& operator*=(const HIP_vector_type& x) noexcept
        {
            data *= x.data;
            return *this;
        }

        friend __attribute__((device)) inline constexpr HIP_vector_type operator*(
        HIP_vector_type x, const HIP_vector_type& y) noexcept
        {
          return HIP_vector_type{ x } *= y;
        }

        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        __attribute__((device))
        HIP_vector_type& operator*=(U x) noexcept
        {
            return *this *= HIP_vector_type{x};
        }

        friend __attribute__((device)) inline constexpr HIP_vector_type operator/(
        HIP_vector_type x, const HIP_vector_type& y) noexcept
        {
          return HIP_vector_type{ x } /= y;
        }

        __attribute__((device))
        HIP_vector_type& operator/=(const HIP_vector_type& x) noexcept
        {
            data /= x.data;
            return *this;
        }
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        __attribute__((device))
        HIP_vector_type& operator/=(U x) noexcept
        {
            return *this /= HIP_vector_type{x};
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_signed<U>{}>::type* = nullptr>
        __attribute__((device))
        HIP_vector_type operator-() const noexcept
        {
            auto tmp(*this);
            tmp.data = -tmp.data;
            return tmp;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __attribute__((device))
        HIP_vector_type operator~() const noexcept
        {
            HIP_vector_type r{*this};
            r.data = ~r.data;
            return r;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __attribute__((device))
        HIP_vector_type& operator%=(const HIP_vector_type& x) noexcept
        {
            data %= x.data;
            return *this;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __attribute__((device))
        HIP_vector_type& operator^=(const HIP_vector_type& x) noexcept
        {
            data ^= x.data;
            return *this;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __attribute__((device))
        HIP_vector_type& operator|=(const HIP_vector_type& x) noexcept
        {
            data |= x.data;
            return *this;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __attribute__((device))
        HIP_vector_type& operator&=(const HIP_vector_type& x) noexcept
        {
            data &= x.data;
            return *this;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __attribute__((device))
        HIP_vector_type& operator>>=(const HIP_vector_type& x) noexcept
        {
            data >>= x.data;
            return *this;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __attribute__((device))
        HIP_vector_type& operator<<=(const HIP_vector_type& x) noexcept
        {
            data <<= x.data;
            return *this;
        }
    };

    template<typename T, unsigned int n>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator+(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} += y;
    }
    template<typename T, unsigned int n, typename U>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator+(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} += HIP_vector_type<T, n>{y};
    }
    template<typename T, unsigned int n, typename U>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator+(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} += y;
    }

    template<typename T, unsigned int n>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator-(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} -= y;
    }
    template<typename T, unsigned int n, typename U>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator-(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} -= HIP_vector_type<T, n>{y};
    }
    template<typename T, unsigned int n, typename U>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator-(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} -= y;
    }

    template<typename T, unsigned int n, typename U>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator*(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} *= HIP_vector_type<T, n>{y};
    }
    template<typename T, unsigned int n, typename U>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator*(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} *= y;
    }

    template<typename T, unsigned int n, typename U>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator/(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} /= HIP_vector_type<T, n>{y};
    }
    template<typename T, unsigned int n, typename U>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator/(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} /= y;
    }

    template<typename V>
    __attribute__((device))
    inline
    constexpr
    bool _hip_any_zero(const V& x, int n) noexcept
    {
        return
            (n == -1) ? true : ((x[n] == 0) ? false : _hip_any_zero(x, n - 1));
    }

    template<typename T, unsigned int n>
    __attribute__((device))
    inline
    constexpr
    bool operator==(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return _hip_any_zero(x.data == y.data, n - 1);
    }
    template<typename T, unsigned int n, typename U>
    __attribute__((device))
    inline
    constexpr
    bool operator==(const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return x == HIP_vector_type<T, n>{y};
    }
    template<typename T, unsigned int n, typename U>
    __attribute__((device))
    inline
    constexpr
    bool operator==(U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} == y;
    }

    template<typename T, unsigned int n>
    __attribute__((device))
    inline
    constexpr
    bool operator!=(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return !(x == y);
    }
    template<typename T, unsigned int n, typename U>
    __attribute__((device))
    inline
    constexpr
    bool operator!=(const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return !(x == y);
    }
    template<typename T, unsigned int n, typename U>
    __attribute__((device))
    inline
    constexpr
    bool operator!=(U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return !(x == y);
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator%(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} %= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator%(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} %= HIP_vector_type<T, n>{y};
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator%(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} %= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator^(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} ^= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator^(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} ^= HIP_vector_type<T, n>{y};
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator^(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} ^= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator|(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} |= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator|(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} |= HIP_vector_type<T, n>{y};
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator|(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} |= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator&(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} &= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator&(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} &= HIP_vector_type<T, n>{y};
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator&(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} &= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator>>(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} >>= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator>>(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} >>= HIP_vector_type<T, n>{y};
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator>>(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} >>= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator<<(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} <<= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator<<(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} <<= HIP_vector_type<T, n>{y};
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_arithmetic<U>::value>::type,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __attribute__((device))
    inline
    constexpr
    HIP_vector_type<T, n> operator<<(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} <<= y;
    }




    template <typename T, unsigned int rankT, typename U, unsigned int rankU>
    inline __attribute__((always_inline)) __attribute__((device)) typename std::enable_if<(rankT == 1 && rankU >= 1),
                                                            const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
      return HIP_vector_type<T, rankT>(static_cast<T>(u.x));
    };

    template <typename T, unsigned int rankT, typename U, unsigned int rankU>
    inline __attribute__((always_inline)) __attribute__((device)) typename std::enable_if<(rankT == 2 && rankU == 1),
                                                            const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
      return HIP_vector_type<T, rankT> (static_cast<T>(u.x), static_cast<T>(0));
    };

    template <typename T, unsigned int rankT, typename U, unsigned int rankU>
    inline __attribute__((always_inline)) __attribute__((device)) typename std::enable_if<(rankT == 2 && rankU >= 2),
                                                            const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
      return HIP_vector_type<T, rankT> (static_cast<T>(u.x), static_cast<T>(u.y));
    };

    template <typename T, unsigned int rankT, typename U, unsigned int rankU>
    inline __attribute__((always_inline)) __attribute__((device)) typename std::enable_if<(rankT == 4 && rankU == 1),
                                                            const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
      return HIP_vector_type<T, rankT> (static_cast<T>(u.x), static_cast<T>(0),
                                       static_cast<T>(0), static_cast<T>(0));
    };

    template <typename T, unsigned int rankT, typename U, unsigned int rankU>
    inline __attribute__((always_inline)) __attribute__((device)) typename std::enable_if<(rankT == 4 && rankU == 2),
                                                            const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
      return HIP_vector_type<T, rankT>(static_cast<T>(u.x), static_cast<T>(u.y),
                                       static_cast<T>(0), static_cast<T>(0));
    };

    template <typename T, unsigned int rankT, typename U, unsigned int rankU>
    inline __attribute__((always_inline)) __attribute__((device)) typename std::enable_if<(rankT == 4 && rankU == 4),
                                                            const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
      return HIP_vector_type<T, rankT> (static_cast<T>(u.x), static_cast<T>(u.y),
                                       static_cast<T>(u.z), static_cast<T>(u.w));
    };
# 1135 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_vector_types.h" 3
using uchar1 = HIP_vector_type<unsigned char, 1>; using uchar2 = HIP_vector_type<unsigned char, 2>; using uchar3 = HIP_vector_type<unsigned char, 3>; using uchar4 = HIP_vector_type<unsigned char, 4>;;
using char1 = HIP_vector_type<char, 1>; using char2 = HIP_vector_type<char, 2>; using char3 = HIP_vector_type<char, 3>; using char4 = HIP_vector_type<char, 4>;;
using ushort1 = HIP_vector_type<unsigned short, 1>; using ushort2 = HIP_vector_type<unsigned short, 2>; using ushort3 = HIP_vector_type<unsigned short, 3>; using ushort4 = HIP_vector_type<unsigned short, 4>;;
using short1 = HIP_vector_type<short, 1>; using short2 = HIP_vector_type<short, 2>; using short3 = HIP_vector_type<short, 3>; using short4 = HIP_vector_type<short, 4>;;
using uint1 = HIP_vector_type<unsigned int, 1>; using uint2 = HIP_vector_type<unsigned int, 2>; using uint3 = HIP_vector_type<unsigned int, 3>; using uint4 = HIP_vector_type<unsigned int, 4>;;
using int1 = HIP_vector_type<int, 1>; using int2 = HIP_vector_type<int, 2>; using int3 = HIP_vector_type<int, 3>; using int4 = HIP_vector_type<int, 4>;;
using ulong1 = HIP_vector_type<unsigned long, 1>; using ulong2 = HIP_vector_type<unsigned long, 2>; using ulong3 = HIP_vector_type<unsigned long, 3>; using ulong4 = HIP_vector_type<unsigned long, 4>;;
using long1 = HIP_vector_type<long, 1>; using long2 = HIP_vector_type<long, 2>; using long3 = HIP_vector_type<long, 3>; using long4 = HIP_vector_type<long, 4>;;
using ulonglong1 = HIP_vector_type<unsigned long long, 1>; using ulonglong2 = HIP_vector_type<unsigned long long, 2>; using ulonglong3 = HIP_vector_type<unsigned long long, 3>; using ulonglong4 = HIP_vector_type<unsigned long long, 4>;;
using longlong1 = HIP_vector_type<long long, 1>; using longlong2 = HIP_vector_type<long long, 2>; using longlong3 = HIP_vector_type<long long, 3>; using longlong4 = HIP_vector_type<long long, 4>;;
using float1 = HIP_vector_type<float, 1>; using float2 = HIP_vector_type<float, 2>; using float3 = HIP_vector_type<float, 3>; using float4 = HIP_vector_type<float, 4>;;
using double1 = HIP_vector_type<double, 1>; using double2 = HIP_vector_type<double, 2>; using double3 = HIP_vector_type<double, 3>; using double4 = HIP_vector_type<double, 4>;;
# 2117 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_vector_types.h" 3
static inline __attribute__((device)) uchar1 make_uchar1(unsigned char x) { uchar1 r{x}; return r; };
static inline __attribute__((device)) uchar2 make_uchar2(unsigned char x, unsigned char y) { uchar2 r{x, y}; return r; };
static inline __attribute__((device)) uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) { uchar3 r{x, y, z}; return r; };
static inline __attribute__((device)) uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) { uchar4 r{x, y, z, w}; return r; };

static inline __attribute__((device)) char1 make_char1(signed char x) { char1 r{x}; return r; };
static inline __attribute__((device)) char2 make_char2(signed char x, signed char y) { char2 r{x, y}; return r; };
static inline __attribute__((device)) char3 make_char3(signed char x, signed char y, signed char z) { char3 r{x, y, z}; return r; };
static inline __attribute__((device)) char4 make_char4(signed char x, signed char y, signed char z, signed char w) { char4 r{x, y, z, w}; return r; };

static inline __attribute__((device)) ushort1 make_ushort1(unsigned short x) { ushort1 r{x}; return r; };
static inline __attribute__((device)) ushort2 make_ushort2(unsigned short x, unsigned short y) { ushort2 r{x, y}; return r; };
static inline __attribute__((device)) ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) { ushort3 r{x, y, z}; return r; };
static inline __attribute__((device)) ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) { ushort4 r{x, y, z, w}; return r; };

static inline __attribute__((device)) short1 make_short1(signed short x) { short1 r{x}; return r; };
static inline __attribute__((device)) short2 make_short2(signed short x, signed short y) { short2 r{x, y}; return r; };
static inline __attribute__((device)) short3 make_short3(signed short x, signed short y, signed short z) { short3 r{x, y, z}; return r; };
static inline __attribute__((device)) short4 make_short4(signed short x, signed short y, signed short z, signed short w) { short4 r{x, y, z, w}; return r; };

static inline __attribute__((device)) uint1 make_uint1(unsigned int x) { uint1 r{x}; return r; };
static inline __attribute__((device)) uint2 make_uint2(unsigned int x, unsigned int y) { uint2 r{x, y}; return r; };
static inline __attribute__((device)) uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) { uint3 r{x, y, z}; return r; };
static inline __attribute__((device)) uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { uint4 r{x, y, z, w}; return r; };

static inline __attribute__((device)) int1 make_int1(signed int x) { int1 r{x}; return r; };
static inline __attribute__((device)) int2 make_int2(signed int x, signed int y) { int2 r{x, y}; return r; };
static inline __attribute__((device)) int3 make_int3(signed int x, signed int y, signed int z) { int3 r{x, y, z}; return r; };
static inline __attribute__((device)) int4 make_int4(signed int x, signed int y, signed int z, signed int w) { int4 r{x, y, z, w}; return r; };

static inline __attribute__((device)) float1 make_float1(float x) { float1 r{x}; return r; };
static inline __attribute__((device)) float2 make_float2(float x, float y) { float2 r{x, y}; return r; };
static inline __attribute__((device)) float3 make_float3(float x, float y, float z) { float3 r{x, y, z}; return r; };
static inline __attribute__((device)) float4 make_float4(float x, float y, float z, float w) { float4 r{x, y, z, w}; return r; };

static inline __attribute__((device)) double1 make_double1(double x) { double1 r{x}; return r; };
static inline __attribute__((device)) double2 make_double2(double x, double y) { double2 r{x, y}; return r; };
static inline __attribute__((device)) double3 make_double3(double x, double y, double z) { double3 r{x, y, z}; return r; };
static inline __attribute__((device)) double4 make_double4(double x, double y, double z, double w) { double4 r{x, y, z, w}; return r; };

static inline __attribute__((device)) ulong1 make_ulong1(unsigned long x) { ulong1 r{x}; return r; };
static inline __attribute__((device)) ulong2 make_ulong2(unsigned long x, unsigned long y) { ulong2 r{x, y}; return r; };
static inline __attribute__((device)) ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) { ulong3 r{x, y, z}; return r; };
static inline __attribute__((device)) ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) { ulong4 r{x, y, z, w}; return r; };

static inline __attribute__((device)) long1 make_long1(signed long x) { long1 r{x}; return r; };
static inline __attribute__((device)) long2 make_long2(signed long x, signed long y) { long2 r{x, y}; return r; };
static inline __attribute__((device)) long3 make_long3(signed long x, signed long y, signed long z) { long3 r{x, y, z}; return r; };
static inline __attribute__((device)) long4 make_long4(signed long x, signed long y, signed long z, signed long w) { long4 r{x, y, z, w}; return r; };

static inline __attribute__((device)) ulonglong1 make_ulonglong1(unsigned long long x) { ulonglong1 r{x}; return r; };
static inline __attribute__((device)) ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y) { ulonglong2 r{x, y}; return r; };
static inline __attribute__((device)) ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z) { ulonglong3 r{x, y, z}; return r; };
static inline __attribute__((device)) ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w) { ulonglong4 r{x, y, z, w}; return r; };

static inline __attribute__((device)) longlong1 make_longlong1(signed long long x) { longlong1 r{x}; return r; };
static inline __attribute__((device)) longlong2 make_longlong2(signed long long x, signed long long y) { longlong2 r{x, y}; return r; };
static inline __attribute__((device)) longlong3 make_longlong3(signed long long x, signed long long y, signed long long z) { longlong3 r{x, y, z}; return r; };
static inline __attribute__((device)) longlong4 make_longlong4(signed long long x, signed long long y, signed long long z, signed long long w) { longlong4 r{x, y, z, w}; return r; };
# 28 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/hip_ldg.h" 2 3


__attribute__((device)) inline static char __ldg(const char* ptr) { return *ptr; }

__attribute__((device)) inline static char2 __ldg(const char2* ptr) { return *ptr; }

__attribute__((device)) inline static char4 __ldg(const char4* ptr) { return *ptr; }

__attribute__((device)) inline static signed char __ldg(const signed char* ptr) { return ptr[0]; }

__attribute__((device)) inline static unsigned char __ldg(const unsigned char* ptr) { return ptr[0]; }


__attribute__((device)) inline static short __ldg(const short* ptr) { return ptr[0]; }

__attribute__((device)) inline static short2 __ldg(const short2* ptr) { return ptr[0]; }

__attribute__((device)) inline static short4 __ldg(const short4* ptr) { return ptr[0]; }

__attribute__((device)) inline static unsigned short __ldg(const unsigned short* ptr) { return ptr[0]; }


__attribute__((device)) inline static int __ldg(const int* ptr) { return ptr[0]; }

__attribute__((device)) inline static int2 __ldg(const int2* ptr) { return ptr[0]; }

__attribute__((device)) inline static int4 __ldg(const int4* ptr) { return ptr[0]; }

__attribute__((device)) inline static unsigned int __ldg(const unsigned int* ptr) { return ptr[0]; }


__attribute__((device)) inline static long __ldg(const long* ptr) { return ptr[0]; }

__attribute__((device)) inline static unsigned long __ldg(const unsigned long* ptr) { return ptr[0]; }


__attribute__((device)) inline static long long __ldg(const long long* ptr) { return ptr[0]; }

__attribute__((device)) inline static longlong2 __ldg(const longlong2* ptr) { return ptr[0]; }

__attribute__((device)) inline static unsigned long long __ldg(const unsigned long long* ptr) { return ptr[0]; }


__attribute__((device)) inline static uchar2 __ldg(const uchar2* ptr) { return ptr[0]; }

__attribute__((device)) inline static uchar4 __ldg(const uchar4* ptr) { return ptr[0]; }


__attribute__((device)) inline static ushort2 __ldg(const ushort2* ptr) { return ptr[0]; }


__attribute__((device)) inline static uint2 __ldg(const uint2* ptr) { return ptr[0]; }

__attribute__((device)) inline static uint4 __ldg(const uint4* ptr) { return ptr[0]; }


__attribute__((device)) inline static ulonglong2 __ldg(const ulonglong2* ptr) { return ptr[0]; }


__attribute__((device)) inline static float __ldg(const float* ptr) { return ptr[0]; }

__attribute__((device)) inline static float2 __ldg(const float2* ptr) { return ptr[0]; }

__attribute__((device)) inline static float4 __ldg(const float4* ptr) { return ptr[0]; }


__attribute__((device)) inline static double __ldg(const double* ptr) { return ptr[0]; }

__attribute__((device)) inline static double2 __ldg(const double2* ptr) { return ptr[0]; }
# 125 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h" 2 3
# 250 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h" 3
extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_size(unsigned int);
extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_num_groups(unsigned int);
struct __HIP_BlockIdx {
  __attribute__((device))
  std::uint32_t operator()(std::uint32_t x) const noexcept { return __ockl_get_group_id(x); }
};
struct __HIP_BlockDim {
  __attribute__((device))
  std::uint32_t operator()(std::uint32_t x) const noexcept {
    return __ockl_get_local_size(x);
  }
};
struct __HIP_GridDim {
  __attribute__((device))
  std::uint32_t operator()(std::uint32_t x) const noexcept {
    return __ockl_get_num_groups(x);
  }
};
struct __HIP_ThreadIdx {
  __attribute__((device))
  std::uint32_t operator()(std::uint32_t x) const noexcept {
    return __ockl_get_local_id(x);
  }
};


typedef struct dim3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;

    constexpr __attribute__((device)) dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1) : x(_x), y(_y), z(_z){};

} dim3;


extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_global_size(unsigned int);


template <typename F> struct __HIP_Coordinates {
    using R = decltype(F{}(0));

    struct __X {
    __attribute__((device)) operator R() const noexcept { return F{}(0); }
    __attribute__((device)) R operator+=(const R& rhs) { return F{}(0) + rhs; }
    };
    struct __Y {
    __attribute__((device)) operator R() const noexcept { return F{}(1); }
    __attribute__((device)) R operator+=(const R& rhs) { return F{}(1) + rhs; }
    };
    struct __Z {
    __attribute__((device)) operator R() const noexcept { return F{}(2); }
    __attribute__((device)) R operator+=(const R& rhs) { return F{}(2) + rhs; }
    };


    __attribute__((weak))

    __attribute__((device)) static constexpr __X x{};

    __attribute__((weak))

    __attribute__((device)) static constexpr __Y y{};

    __attribute__((weak))

    __attribute__((device)) static constexpr __Z z{};

    __attribute__((device)) operator dim3() const { return dim3(x, y, z); }
};

template <typename F>
constexpr typename __HIP_Coordinates<F>::__X __HIP_Coordinates<F>::x;
template <typename F>
constexpr typename __HIP_Coordinates<F>::__Y __HIP_Coordinates<F>::y;
template <typename F>
constexpr typename __HIP_Coordinates<F>::__Z __HIP_Coordinates<F>::z;

inline
__attribute__((device))
std::uint32_t operator*(__HIP_Coordinates<__HIP_GridDim>::__X,
                        __HIP_Coordinates<__HIP_BlockDim>::__X) noexcept {
  return __ockl_get_global_size(0);
}
inline
__attribute__((device))
std::uint32_t operator*(__HIP_Coordinates<__HIP_BlockDim>::__X,
                        __HIP_Coordinates<__HIP_GridDim>::__X) noexcept {
  return __ockl_get_global_size(0);
}
inline
__attribute__((device))
std::uint32_t operator*(__HIP_Coordinates<__HIP_GridDim>::__Y,
                        __HIP_Coordinates<__HIP_BlockDim>::__Y) noexcept {
  return __ockl_get_global_size(1);
}
inline
__attribute__((device))
std::uint32_t operator*(__HIP_Coordinates<__HIP_BlockDim>::__Y,
                        __HIP_Coordinates<__HIP_GridDim>::__Y) noexcept {
  return __ockl_get_global_size(1);
}
inline
__attribute__((device))
std::uint32_t operator*(__HIP_Coordinates<__HIP_GridDim>::__Z,
                        __HIP_Coordinates<__HIP_BlockDim>::__Z) noexcept {
  return __ockl_get_global_size(2);
}
inline
__attribute__((device))
std::uint32_t operator*(__HIP_Coordinates<__HIP_BlockDim>::__Z,
                        __HIP_Coordinates<__HIP_GridDim>::__Z) noexcept {
  return __ockl_get_global_size(2);
}

static constexpr __HIP_Coordinates<__HIP_BlockDim> blockDim{};
static constexpr __HIP_Coordinates<__HIP_BlockIdx> blockIdx{};
static constexpr __HIP_Coordinates<__HIP_GridDim> gridDim{};
static constexpr __HIP_Coordinates<__HIP_ThreadIdx> threadIdx{};


extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_id(unsigned int);




extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_group_id(unsigned int);




extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_size(unsigned int);




extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_num_groups(unsigned int);
# 63 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/hip-on-vdi/include/hip/hip_runtime.h" 2 3
# 73 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/hip-on-vdi/include/hip/hip_runtime.h" 3
# 1 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/hip-on-vdi/include/hip/hip_vector_types.h" 1 3
# 74 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/hip-on-vdi/include/hip/hip_runtime.h" 2 3
# 6 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/out/ubuntu-22.04/22.04/build/hip-on-rocclr/hipamd/src/hiprtc/hip_rtc_gen/hipRTC_header.h" 2
# 1 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/hip-on-vdi/include/hip/hip_bfloat16.h" 1 3
# 37 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/hip-on-vdi/include/hip/hip_bfloat16.h" 3
# 1 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_bfloat16.h" 1 3
# 55 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_bfloat16.h" 3
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
struct hip_bfloat16
{
    __hip_uint16_t data;

    enum truncate_t
    {
        truncate
    };

    __attribute__((device)) hip_bfloat16() = default;


    explicit __attribute__((device)) hip_bfloat16(float f)
        : data(float_to_bfloat16(f))
    {
    }

    explicit __attribute__((device)) hip_bfloat16(float f, truncate_t)
        : data(truncate_float_to_bfloat16(f))
    {
    }


    __attribute__((device)) operator float() const
    {
        union
        {
            uint32_t int32;
            float fp32;
        } u = {uint32_t(data) << 16};
        return u.fp32;
    }

    __attribute__((device)) hip_bfloat16 &operator=(const float& f)
    {
       data = float_to_bfloat16(f);
       return *this;
    }

    static __attribute__((device)) hip_bfloat16 round_to_bfloat16(float f)
    {
        hip_bfloat16 output;
        output.data = float_to_bfloat16(f);
        return output;
    }

    static __attribute__((device)) hip_bfloat16 round_to_bfloat16(float f, truncate_t)
    {
        hip_bfloat16 output;
        output.data = truncate_float_to_bfloat16(f);
        return output;
    }

private:
    static __attribute__((device)) __hip_uint16_t float_to_bfloat16(float f)
    {
        union
        {
            float fp32;
            uint32_t int32;
        } u = {f};
        if(~u.int32 & 0x7f800000)
        {
# 136 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_bfloat16.h" 3
            u.int32 += 0x7fff + ((u.int32 >> 16) & 1);
        }
        else if(u.int32 & 0xffff)
        {
# 148 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_bfloat16.h" 3
            u.int32 |= 0x10000;
        }
        return __hip_uint16_t(u.int32 >> 16);
    }


    static __attribute__((device)) __hip_uint16_t truncate_float_to_bfloat16(float f)
    {
        union
        {
            float fp32;
            uint32_t int32;
        } u = {f};
        return __hip_uint16_t(u.int32 >> 16) | (!(~u.int32 & 0x7f800000) && (u.int32 & 0xffff));
    }
};
#pragma clang diagnostic pop

typedef struct
{
    __hip_uint16_t data;
} hip_bfloat16_public;

static_assert(__hip_internal::is_standard_layout<hip_bfloat16>{},
              "hip_bfloat16 is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(__hip_internal::is_trivial<hip_bfloat16>{},
              "hip_bfloat16 is not a trivial type, and thus is "
              "incompatible with C.");
# 189 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/include/hip/amd_detail/amd_hip_bfloat16.h" 3
inline __attribute__((device)) hip_bfloat16 operator+(hip_bfloat16 a)
{
    return a;
}
inline __attribute__((device)) hip_bfloat16 operator-(hip_bfloat16 a)
{
    a.data ^= 0x8000;
    return a;
}
inline __attribute__((device)) hip_bfloat16 operator+(hip_bfloat16 a, hip_bfloat16 b)
{
    return hip_bfloat16(float(a) + float(b));
}
inline __attribute__((device)) hip_bfloat16 operator-(hip_bfloat16 a, hip_bfloat16 b)
{
    return hip_bfloat16(float(a) - float(b));
}
inline __attribute__((device)) hip_bfloat16 operator*(hip_bfloat16 a, hip_bfloat16 b)
{
    return hip_bfloat16(float(a) * float(b));
}
inline __attribute__((device)) hip_bfloat16 operator/(hip_bfloat16 a, hip_bfloat16 b)
{
    return hip_bfloat16(float(a) / float(b));
}
inline __attribute__((device)) bool operator<(hip_bfloat16 a, hip_bfloat16 b)
{
    return float(a) < float(b);
}
inline __attribute__((device)) bool operator==(hip_bfloat16 a, hip_bfloat16 b)
{
    return float(a) == float(b);
}
inline __attribute__((device)) bool operator>(hip_bfloat16 a, hip_bfloat16 b)
{
    return b < a;
}
inline __attribute__((device)) bool operator<=(hip_bfloat16 a, hip_bfloat16 b)
{
    return !(a > b);
}
inline __attribute__((device)) bool operator!=(hip_bfloat16 a, hip_bfloat16 b)
{
    return !(a == b);
}
inline __attribute__((device)) bool operator>=(hip_bfloat16 a, hip_bfloat16 b)
{
    return !(a < b);
}
inline __attribute__((device)) hip_bfloat16& operator+=(hip_bfloat16& a, hip_bfloat16 b)
{
    return a = a + b;
}
inline __attribute__((device)) hip_bfloat16& operator-=(hip_bfloat16& a, hip_bfloat16 b)
{
    return a = a - b;
}
inline __attribute__((device)) hip_bfloat16& operator*=(hip_bfloat16& a, hip_bfloat16 b)
{
    return a = a * b;
}
inline __attribute__((device)) hip_bfloat16& operator/=(hip_bfloat16& a, hip_bfloat16 b)
{
    return a = a / b;
}
inline __attribute__((device)) hip_bfloat16& operator++(hip_bfloat16& a)
{
    return a += hip_bfloat16(1.0f);
}
inline __attribute__((device)) hip_bfloat16& operator--(hip_bfloat16& a)
{
    return a -= hip_bfloat16(1.0f);
}
inline __attribute__((device)) hip_bfloat16 operator++(hip_bfloat16& a, int)
{
    hip_bfloat16 orig = a;
    ++a;
    return orig;
}
inline __attribute__((device)) hip_bfloat16 operator--(hip_bfloat16& a, int)
{
    hip_bfloat16 orig = a;
    --a;
    return orig;
}

namespace std
{
    constexpr __attribute__((device)) bool isinf(hip_bfloat16 a)
    {
        return !(~a.data & 0x7f80) && !(a.data & 0x7f);
    }
    constexpr __attribute__((device)) bool isnan(hip_bfloat16 a)
    {
        return !(~a.data & 0x7f80) && +(a.data & 0x7f);
    }
    constexpr __attribute__((device)) bool iszero(hip_bfloat16 a)
    {
        return !(a.data & 0x7fff);
    }
}
# 38 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/hip-on-vdi/include/hip/hip_bfloat16.h" 2 3
# 7 "/long_pathname_so_that_rpms_can_package_the_debug_info/src/out/ubuntu-22.04/22.04/build/hip-on-rocclr/hipamd/src/hiprtc/hip_rtc_gen/hipRTC_header.h" 2

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-id-macro"
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#pragma clang diagnostic ignored "-Wreserved-macro-identifier"
#pragma clang diagnostic ignored "-Wundef"
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __global__ __attribute__((global))
#define __constant__ __attribute__((constant))
#define __shared__ __attribute__((shared))
#define __align__(x) __attribute__((aligned(x)))
#if !defined(__has_feature) || !__has_feature(cuda_noinline_keyword)
#define __noinline__ __attribute__((noinline))
#endif
#define __forceinline__ inline __attribute__((always_inline))
#if __HIP_NO_IMAGE_SUPPORT
#define __hip_img_chk__ __attribute__((unavailable("The image/texture API not supported on the device")))
#else
#define __hip_img_chk__
#endif
#define launch_bounds_impl0(requiredMaxThreadsPerBlock)                                       \
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock)))
#define launch_bounds_impl1(requiredMaxThreadsPerBlock, minBlocksPerMultiprocessor)           \
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock),                \
                   amdgpu_waves_per_eu(minBlocksPerMultiprocessor)))
#define select_impl_(_1, _2, impl_, ...) impl_
#define __launch_bounds__(...)                                                                \
    select_impl_(__VA_ARGS__, launch_bounds_impl1, launch_bounds_impl0)(__VA_ARGS__)           
#define HIP_INCLUDE_HIP_HIP_RUNTIME_H
#define _HIP_BFLOAT16_H_
#define HIP_INCLUDE_HIP_MATH_FUNCTIONS_H
#define HIP_INCLUDE_HIP_HIP_VECTOR_TYPES_H
#if !__HIP_NO_STD_DEFS__
#if defined(__HIPRTC_PTRDIFF_T_IS_LONG_LONG__) && __HIPRTC_PTRDIFF_T_IS_LONG_LONG__==1
typedef long long ptrdiff_t;
#else
typedef __PTRDIFF_TYPE__ ptrdiff_t;
#endif
typedef long clock_t;
namespace std {
using ::ptrdiff_t;
using ::clock_t;
}
#endif // __HIP_NO_STD_DEFS__
#pragma clang diagnostic pop/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HIP_INCLUDE_HIP_HIP_COMMON_H
#define HIP_INCLUDE_HIP_HIP_COMMON_H

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-macro-identifier"
#endif
// Common code included at start of every hip file.
// Auto enable __HIP_PLATFORM_AMD__ if compiling on AMD platform
// Other compiler (GCC,ICC,etc) need to set one of these macros explicitly
#if defined(__clang__) && defined(__HIP__)
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#endif  // defined(__clang__) && defined(__HIP__)

// Auto enable __HIP_PLATFORM_NVIDIA__ if compiling with NVIDIA platform
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__) && !defined(__HIP__))
#ifndef __HIP_PLATFORM_NVIDIA__
#define __HIP_PLATFORM_NVIDIA__
#endif

#ifdef __CUDACC__
#define __HIPCC__
#endif

#endif  //__NVCC__

// Auto enable __HIP_DEVICE_COMPILE__ if compiled in HCC or NVCC device path
#if (defined(__HCC_ACCELERATOR__) && __HCC_ACCELERATOR__ != 0) ||                                  \
    (defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 0)
#define __HIP_DEVICE_COMPILE__ 1
#endif

#ifdef __GNUC__
#define HIP_PUBLIC_API              __attribute__ ((visibility ("default")))
#define HIP_INTERNAL_EXPORTED_API   __attribute__ ((visibility ("default")))
#else
#define HIP_PUBLIC_API
#define HIP_INTERNAL_EXPORTED_API 
#endif

#if __HIP_DEVICE_COMPILE__ == 0
// 32-bit Atomics
#define __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__ (0)
#define __HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__ (0)
#define __HIP_ARCH_HAS_SHARED_INT32_ATOMICS__ (0)
#define __HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH__ (0)
#define __HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__ (0)

// 64-bit Atomics
#define __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__ (0)
#define __HIP_ARCH_HAS_SHARED_INT64_ATOMICS__ (0)

// Doubles
#define __HIP_ARCH_HAS_DOUBLES__ (0)

// Warp cross-lane operations
#define __HIP_ARCH_HAS_WARP_VOTE__ (0)
#define __HIP_ARCH_HAS_WARP_BALLOT__ (0)
#define __HIP_ARCH_HAS_WARP_SHUFFLE__ (0)
#define __HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__ (0)

// Sync
#define __HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__ (0)
#define __HIP_ARCH_HAS_SYNC_THREAD_EXT__ (0)

// Misc
#define __HIP_ARCH_HAS_SURFACE_FUNCS__ (0)
#define __HIP_ARCH_HAS_3DGRID__ (0)
#define __HIP_ARCH_HAS_DYNAMIC_PARALLEL__ (0)
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HIP_INCLUDE_HIP_LIBRARY_TYPES_H
#define HIP_INCLUDE_HIP_LIBRARY_TYPES_H

#if !defined(__HIPCC_RTC__)
#include <hip/hip_common.h>
#endif

#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)

typedef enum hipDataType {
  HIP_R_32F   =  0,
  HIP_R_64F   =  1,
  HIP_R_16F   =  2,
  HIP_R_8I    =  3,
  HIP_C_32F   =  4,
  HIP_C_64F   =  5,
  HIP_C_16F   =  6,
  HIP_C_8I    =  7,
  HIP_R_8U    =  8,
  HIP_C_8U    =  9,
  HIP_R_32I   = 10,
  HIP_C_32I   = 11,
  HIP_R_32U   = 12,
  HIP_C_32U   = 13,
  HIP_R_16BF  = 14,
  HIP_C_16BF  = 15,
  HIP_R_4I    = 16,
  HIP_C_4I    = 17,
  HIP_R_4U    = 18,
  HIP_C_4U    = 19,
  HIP_R_16I   = 20,
  HIP_C_16I   = 21,
  HIP_R_16U   = 22,
  HIP_C_16U   = 23,
  HIP_R_64I   = 24,
  HIP_C_64I   = 25,
  HIP_R_64U   = 26,
  HIP_C_64U   = 27,
  // HIP specific Data Types
  HIP_R_8F_E4M3_FNUZ = 1000,
  HIP_R_8F_E5M2_FNUZ = 1001
} hipDataType;

typedef enum hipLibraryPropertyType {
  HIP_LIBRARY_MAJOR_VERSION,
  HIP_LIBRARY_MINOR_VERSION,
  HIP_LIBRARY_PATCH_LEVEL
} hipLibraryPropertyType;

#elif !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include "library_types.h"
#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

#endif
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HIP_INCLUDE_HIP_DRIVER_TYPES_H
#define HIP_INCLUDE_HIP_DRIVER_TYPES_H

#if !defined(__HIPCC_RTC__)
#include <hip/hip_common.h>
#endif

#if !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include "driver_types.h"
#elif defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)

#if !defined(__HIPCC_RTC__)
#ifndef __cplusplus
#include <stdbool.h>
#endif
#endif // !defined(__HIPCC_RTC__)
typedef void* hipDeviceptr_t;
typedef enum hipChannelFormatKind {
    hipChannelFormatKindSigned = 0,
    hipChannelFormatKindUnsigned = 1,
    hipChannelFormatKindFloat = 2,
    hipChannelFormatKindNone = 3
}hipChannelFormatKind;
typedef struct hipChannelFormatDesc {
    int x;
    int y;
    int z;
    int w;
    enum hipChannelFormatKind f;
}hipChannelFormatDesc;
#define HIP_TRSA_OVERRIDE_FORMAT 0x01
#define HIP_TRSF_READ_AS_INTEGER 0x01
#define HIP_TRSF_NORMALIZED_COORDINATES 0x02
#define HIP_TRSF_SRGB 0x10

typedef struct hipArray* hipArray_t;
typedef const struct hipArray* hipArray_const_t;
typedef enum hipArray_Format {
    HIP_AD_FORMAT_UNSIGNED_INT8 = 0x01,
    HIP_AD_FORMAT_UNSIGNED_INT16 = 0x02,
    HIP_AD_FORMAT_UNSIGNED_INT32 = 0x03,
    HIP_AD_FORMAT_SIGNED_INT8 = 0x08,
    HIP_AD_FORMAT_SIGNED_INT16 = 0x09,
    HIP_AD_FORMAT_SIGNED_INT32 = 0x0a,
    HIP_AD_FORMAT_HALF = 0x10,
    HIP_AD_FORMAT_FLOAT = 0x20
}hipArray_Format;
typedef struct HIP_ARRAY_DESCRIPTOR {
  size_t Width;
  size_t Height;
  enum hipArray_Format Format;
  unsigned int NumChannels;
}HIP_ARRAY_DESCRIPTOR;
typedef struct HIP_ARRAY3D_DESCRIPTOR {
  size_t Width;
  size_t Height;
  size_t Depth;
  enum hipArray_Format Format;
  unsigned int NumChannels;
  unsigned int Flags;
}HIP_ARRAY3D_DESCRIPTOR;
#if !defined(__HIPCC_RTC__)
typedef struct hip_Memcpy2D {
    size_t srcXInBytes;
    size_t srcY;
    hipMemoryType srcMemoryType;
    const void* srcHost;
    hipDeviceptr_t srcDevice;
    hipArray_t srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    hipMemoryType dstMemoryType;
    void* dstHost;
    hipDeviceptr_t dstDevice;
    hipArray_t dstArray;
    size_t dstPitch;
    size_t WidthInBytes;
    size_t Height;
} hip_Memcpy2D;
#endif // !defined(__HIPCC_RTC__)
typedef struct hipMipmappedArray {
  void* data;
  struct hipChannelFormatDesc desc;
  unsigned int type;
  unsigned int width;
  unsigned int height;
  unsigned int depth;
  unsigned int min_mipmap_level;
  unsigned int max_mipmap_level;
  unsigned int flags;
  enum hipArray_Format format;
  unsigned int num_channels;
} hipMipmappedArray;
typedef struct hipMipmappedArray* hipMipmappedArray_t;
typedef hipMipmappedArray_t hipmipmappedArray;
typedef const struct hipMipmappedArray* hipMipmappedArray_const_t;
/**
 * hip resource types
 */
typedef enum hipResourceType {
    hipResourceTypeArray = 0x00,
    hipResourceTypeMipmappedArray = 0x01,
    hipResourceTypeLinear = 0x02,
    hipResourceTypePitch2D = 0x03
}hipResourceType;
typedef enum HIPresourcetype_enum {
    HIP_RESOURCE_TYPE_ARRAY           = 0x00, /**< Array resoure */
    HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01, /**< Mipmapped array resource */
    HIP_RESOURCE_TYPE_LINEAR          = 0x02, /**< Linear resource */
    HIP_RESOURCE_TYPE_PITCH2D         = 0x03  /**< Pitch 2D resource */
} HIPresourcetype, hipResourcetype;
/**
 * hip address modes
 */
typedef enum HIPaddress_mode_enum {
    HIP_TR_ADDRESS_MODE_WRAP   = 0,
    HIP_TR_ADDRESS_MODE_CLAMP  = 1,
    HIP_TR_ADDRESS_MODE_MIRROR = 2,
    HIP_TR_ADDRESS_MODE_BORDER = 3
} HIPaddress_mode;
/**
 * hip filter modes
 */
typedef enum HIPfilter_mode_enum {
    HIP_TR_FILTER_MODE_POINT  = 0,
    HIP_TR_FILTER_MODE_LINEAR = 1
} HIPfilter_mode;
/**
 * Texture descriptor
 */
typedef struct HIP_TEXTURE_DESC_st {
    HIPaddress_mode addressMode[3];  /**< Address modes */
    HIPfilter_mode filterMode;       /**< Filter mode */
    unsigned int flags;              /**< Flags */
    unsigned int maxAnisotropy;      /**< Maximum anisotropy ratio */
    HIPfilter_mode mipmapFilterMode; /**< Mipmap filter mode */
    float mipmapLevelBias;           /**< Mipmap level bias */
    float minMipmapLevelClamp;       /**< Mipmap minimum level clamp */
    float maxMipmapLevelClamp;       /**< Mipmap maximum level clamp */
    float borderColor[4];            /**< Border Color */
    int reserved[12];
} HIP_TEXTURE_DESC;
/**
 * hip texture resource view formats
 */
typedef enum hipResourceViewFormat {
    hipResViewFormatNone = 0x00,
    hipResViewFormatUnsignedChar1 = 0x01,
    hipResViewFormatUnsignedChar2 = 0x02,
    hipResViewFormatUnsignedChar4 = 0x03,
    hipResViewFormatSignedChar1 = 0x04,
    hipResViewFormatSignedChar2 = 0x05,
    hipResViewFormatSignedChar4 = 0x06,
    hipResViewFormatUnsignedShort1 = 0x07,
    hipResViewFormatUnsignedShort2 = 0x08,
    hipResViewFormatUnsignedShort4 = 0x09,
    hipResViewFormatSignedShort1 = 0x0a,
    hipResViewFormatSignedShort2 = 0x0b,
    hipResViewFormatSignedShort4 = 0x0c,
    hipResViewFormatUnsignedInt1 = 0x0d,
    hipResViewFormatUnsignedInt2 = 0x0e,
    hipResViewFormatUnsignedInt4 = 0x0f,
    hipResViewFormatSignedInt1 = 0x10,
    hipResViewFormatSignedInt2 = 0x11,
    hipResViewFormatSignedInt4 = 0x12,
    hipResViewFormatHalf1 = 0x13,
    hipResViewFormatHalf2 = 0x14,
    hipResViewFormatHalf4 = 0x15,
    hipResViewFormatFloat1 = 0x16,
    hipResViewFormatFloat2 = 0x17,
    hipResViewFormatFloat4 = 0x18,
    hipResViewFormatUnsignedBlockCompressed1 = 0x19,
    hipResViewFormatUnsignedBlockCompressed2 = 0x1a,
    hipResViewFormatUnsignedBlockCompressed3 = 0x1b,
    hipResViewFormatUnsignedBlockCompressed4 = 0x1c,
    hipResViewFormatSignedBlockCompressed4 = 0x1d,
    hipResViewFormatUnsignedBlockCompressed5 = 0x1e,
    hipResViewFormatSignedBlockCompressed5 = 0x1f,
    hipResViewFormatUnsignedBlockCompressed6H = 0x20,
    hipResViewFormatSignedBlockCompressed6H = 0x21,
    hipResViewFormatUnsignedBlockCompressed7 = 0x22
}hipResourceViewFormat;
typedef enum HIPresourceViewFormat_enum
{
    HIP_RES_VIEW_FORMAT_NONE          = 0x00, /**< No resource view format (use underlying resource format) */
    HIP_RES_VIEW_FORMAT_UINT_1X8      = 0x01, /**< 1 channel unsigned 8-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_2X8      = 0x02, /**< 2 channel unsigned 8-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_4X8      = 0x03, /**< 4 channel unsigned 8-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_1X8      = 0x04, /**< 1 channel signed 8-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_2X8      = 0x05, /**< 2 channel signed 8-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_4X8      = 0x06, /**< 4 channel signed 8-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_1X16     = 0x07, /**< 1 channel unsigned 16-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_2X16     = 0x08, /**< 2 channel unsigned 16-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_4X16     = 0x09, /**< 4 channel unsigned 16-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_1X16     = 0x0a, /**< 1 channel signed 16-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_2X16     = 0x0b, /**< 2 channel signed 16-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_4X16     = 0x0c, /**< 4 channel signed 16-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_1X32     = 0x0d, /**< 1 channel unsigned 32-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_2X32     = 0x0e, /**< 2 channel unsigned 32-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_4X32     = 0x0f, /**< 4 channel unsigned 32-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_1X32     = 0x10, /**< 1 channel signed 32-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_2X32     = 0x11, /**< 2 channel signed 32-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_4X32     = 0x12, /**< 4 channel signed 32-bit integers */
    HIP_RES_VIEW_FORMAT_FLOAT_1X16    = 0x13, /**< 1 channel 16-bit floating point */
    HIP_RES_VIEW_FORMAT_FLOAT_2X16    = 0x14, /**< 2 channel 16-bit floating point */
    HIP_RES_VIEW_FORMAT_FLOAT_4X16    = 0x15, /**< 4 channel 16-bit floating point */
    HIP_RES_VIEW_FORMAT_FLOAT_1X32    = 0x16, /**< 1 channel 32-bit floating point */
    HIP_RES_VIEW_FORMAT_FLOAT_2X32    = 0x17, /**< 2 channel 32-bit floating point */
    HIP_RES_VIEW_FORMAT_FLOAT_4X32    = 0x18, /**< 4 channel 32-bit floating point */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC1  = 0x19, /**< Block compressed 1 */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC2  = 0x1a, /**< Block compressed 2 */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC3  = 0x1b, /**< Block compressed 3 */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC4  = 0x1c, /**< Block compressed 4 unsigned */
    HIP_RES_VIEW_FORMAT_SIGNED_BC4    = 0x1d, /**< Block compressed 4 signed */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC5  = 0x1e, /**< Block compressed 5 unsigned */
    HIP_RES_VIEW_FORMAT_SIGNED_BC5    = 0x1f, /**< Block compressed 5 signed */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H = 0x20, /**< Block compressed 6 unsigned half-float */
    HIP_RES_VIEW_FORMAT_SIGNED_BC6H   = 0x21, /**< Block compressed 6 signed half-float */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC7  = 0x22  /**< Block compressed 7 */
} HIPresourceViewFormat;
/**
 * HIP resource descriptor
 */
typedef struct hipResourceDesc {
    enum hipResourceType resType;
    union {
        struct {
            hipArray_t array;
        } array;
        struct {
            hipMipmappedArray_t mipmap;
        } mipmap;
        struct {
            void* devPtr;
            struct hipChannelFormatDesc desc;
            size_t sizeInBytes;
        } linear;
        struct {
            void* devPtr;
            struct hipChannelFormatDesc desc;
            size_t width;
            size_t height;
            size_t pitchInBytes;
        } pitch2D;
    } res;
}hipResourceDesc;
typedef struct HIP_RESOURCE_DESC_st
{
    HIPresourcetype resType;                     /**< Resource type */
    union {
        struct {
            hipArray_t hArray;                   /**< HIP array */
        } array;
        struct {
            hipMipmappedArray_t hMipmappedArray; /**< HIP mipmapped array */
        } mipmap;
        struct {
            hipDeviceptr_t devPtr;               /**< Device pointer */
            hipArray_Format format;              /**< Array format */
            unsigned int numChannels;            /**< Channels per array element */
            size_t sizeInBytes;                  /**< Size in bytes */
        } linear;
        struct {
            hipDeviceptr_t devPtr;               /**< Device pointer */
            hipArray_Format format;              /**< Array format */
            unsigned int numChannels;            /**< Channels per array element */
            size_t width;                        /**< Width of the array in elements */
            size_t height;                       /**< Height of the array in elements */
            size_t pitchInBytes;                 /**< Pitch between two rows in bytes */
        } pitch2D;
        struct {
            int reserved[32];
        } reserved;
    } res;
    unsigned int flags;                          /**< Flags (must be zero) */
} HIP_RESOURCE_DESC;
/**
 * hip resource view descriptor
 */
struct hipResourceViewDesc {
    enum hipResourceViewFormat format;
    size_t width;
    size_t height;
    size_t depth;
    unsigned int firstMipmapLevel;
    unsigned int lastMipmapLevel;
    unsigned int firstLayer;
    unsigned int lastLayer;
};
/**
 * Resource view descriptor
 */
typedef struct HIP_RESOURCE_VIEW_DESC_st
{
    HIPresourceViewFormat format;   /**< Resource view format */
    size_t width;                   /**< Width of the resource view */
    size_t height;                  /**< Height of the resource view */
    size_t depth;                   /**< Depth of the resource view */
    unsigned int firstMipmapLevel;  /**< First defined mipmap level */
    unsigned int lastMipmapLevel;   /**< Last defined mipmap level */
    unsigned int firstLayer;        /**< First layer index */
    unsigned int lastLayer;         /**< Last layer index */
    unsigned int reserved[16];
} HIP_RESOURCE_VIEW_DESC;
/**
 * Memory copy types
 *
 */
#if !defined(__HIPCC_RTC__)
typedef enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,      ///< Host-to-Host Copy
    hipMemcpyHostToDevice = 1,    ///< Host-to-Device Copy
    hipMemcpyDeviceToHost = 2,    ///< Device-to-Host Copy
    hipMemcpyDeviceToDevice = 3,  ///< Device-to-Device Copy
    hipMemcpyDefault =
        4  ///< Runtime will automatically determine copy-kind based on virtual addresses.
} hipMemcpyKind;
typedef struct hipPitchedPtr {
    void* ptr;
    size_t pitch;
    size_t xsize;
    size_t ysize;
}hipPitchedPtr;
typedef struct hipExtent {
    size_t width;  // Width in elements when referring to array memory, in bytes when referring to
                   // linear memory
    size_t height;
    size_t depth;
}hipExtent;
typedef struct hipPos {
    size_t x;
    size_t y;
    size_t z;
}hipPos;
typedef struct hipMemcpy3DParms {
    hipArray_t srcArray;
    struct hipPos srcPos;
    struct hipPitchedPtr srcPtr;
    hipArray_t dstArray;
    struct hipPos dstPos;
    struct hipPitchedPtr dstPtr;
    struct hipExtent extent;
    enum hipMemcpyKind kind;
} hipMemcpy3DParms;
typedef struct HIP_MEMCPY3D {
  size_t srcXInBytes;
  size_t srcY;
  size_t srcZ;
  size_t srcLOD;
  hipMemoryType srcMemoryType;
  const void* srcHost;
  hipDeviceptr_t srcDevice;
  hipArray_t srcArray;
  size_t srcPitch;
  size_t srcHeight;
  size_t dstXInBytes;
  size_t dstY;
  size_t dstZ;
  size_t dstLOD;
  hipMemoryType dstMemoryType;
  void* dstHost;
  hipDeviceptr_t dstDevice;
  hipArray_t dstArray;
  size_t dstPitch;
  size_t dstHeight;
  size_t WidthInBytes;
  size_t Height;
  size_t Depth;
} HIP_MEMCPY3D;
static inline struct hipPitchedPtr make_hipPitchedPtr(void* d, size_t p, size_t xsz,
                                                          size_t ysz) {
    struct hipPitchedPtr s;
    s.ptr = d;
    s.pitch = p;
    s.xsize = xsz;
    s.ysize = ysz;
    return s;
}
static inline struct hipPos make_hipPos(size_t x, size_t y, size_t z) {
    struct hipPos p;
    p.x = x;
    p.y = y;
    p.z = z;
    return p;
}
static inline struct hipExtent make_hipExtent(size_t w, size_t h, size_t d) {
    struct hipExtent e;
    e.width = w;
    e.height = h;
    e.depth = d;
    return e;
}
typedef enum hipFunction_attribute {
    HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
    HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
    HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
    HIP_FUNC_ATTRIBUTE_NUM_REGS,
    HIP_FUNC_ATTRIBUTE_PTX_VERSION,
    HIP_FUNC_ATTRIBUTE_BINARY_VERSION,
    HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA,
    HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
    HIP_FUNC_ATTRIBUTE_MAX
} hipFunction_attribute;

typedef enum hipPointer_attribute {
    HIP_POINTER_ATTRIBUTE_CONTEXT = 1,   ///< The context on which a pointer was allocated
                                         ///< @warning - not supported in HIP
    HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,   ///< memory type describing location of a pointer
    HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,///< address at which the pointer is allocated on device
    HIP_POINTER_ATTRIBUTE_HOST_POINTER,  ///< address at which the pointer is allocated on host
    HIP_POINTER_ATTRIBUTE_P2P_TOKENS,    ///< A pair of tokens for use with linux kernel interface
                                         ///< @warning - not supported in HIP
    HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,   ///< Synchronize every synchronous memory operation
                                         ///< initiated on this region
    HIP_POINTER_ATTRIBUTE_BUFFER_ID,     ///< Unique ID for an allocated memory region
    HIP_POINTER_ATTRIBUTE_IS_MANAGED,    ///< Indicates if the pointer points to managed memory
    HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,///< device ordinal of a device on which a pointer
                                         ///< was allocated or registered
    HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE, ///< if this pointer maps to an allocation
                                                     ///< that is suitable for hipIpcGetMemHandle
                                                     ///< @warning - not supported in HIP
    HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,///< Starting address for this requested pointer
    HIP_POINTER_ATTRIBUTE_RANGE_SIZE,      ///< Size of the address range for this requested pointer
    HIP_POINTER_ATTRIBUTE_MAPPED,          ///< tells if this pointer is in a valid address range
                                           ///< that is mapped to a backing allocation
    HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,///< Bitmask of allowed hipmemAllocationHandleType
                                           ///< for this allocation @warning - not supported in HIP
    HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE, ///< returns if the memory referenced by
                                           ///< this pointer can be used with the GPUDirect RDMA API
                                           ///< @warning - not supported in HIP
    HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS,    ///< Returns the access flags the device associated with
                                           ///< for the corresponding memory referenced by the ptr
    HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE   ///< Returns the mempool handle for the allocation if
                                           ///< it was allocated from a mempool
                                           ///< @warning - not supported in HIP
} hipPointer_attribute;

#endif // !defined(__HIPCC_RTC__)
#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif
#endif
/*
Copyright (c) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 *  @file  surface_types.h
 *  @brief Defines surface types for HIP runtime.
 */

#ifndef HIP_INCLUDE_HIP_SURFACE_TYPES_H
#define HIP_INCLUDE_HIP_SURFACE_TYPES_H

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-identifier"
#endif

#if !defined(__HIPCC_RTC__)
#include <hip/driver_types.h>
#endif

/**
 * An opaque value that represents a hip surface object
 */
struct __hip_surface;
typedef struct __hip_surface* hipSurfaceObject_t;

/**
 * hip surface reference
 */
struct surfaceReference {
    hipSurfaceObject_t surfaceObject;
};

/**
 * hip surface boundary modes
 */
enum hipSurfaceBoundaryMode {
    hipBoundaryModeZero = 0,
    hipBoundaryModeTrap = 1,
    hipBoundaryModeClamp = 2
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif /* !HIP_INCLUDE_HIP_SURFACE_TYPES_H */
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_CHANNEL_DESCRIPTOR_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_CHANNEL_DESCRIPTOR_H

#if !defined(__HIPCC_RTC__)
#include <hip/hip_common.h>
#include <hip/driver_types.h>
#include <hip/amd_detail/amd_hip_vector_types.h>
#endif

#ifdef __cplusplus

extern "C" HIP_PUBLIC_API
hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f);

static inline hipChannelFormatDesc hipCreateChannelDescHalf() {
    int e = (int)sizeof(unsigned short) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindFloat);
}

static inline hipChannelFormatDesc hipCreateChannelDescHalf1() {
    int e = (int)sizeof(unsigned short) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindFloat);
}

static inline hipChannelFormatDesc hipCreateChannelDescHalf2() {
    int e = (int)sizeof(unsigned short) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindFloat);
}

static inline hipChannelFormatDesc hipCreateChannelDescHalf4() {
    int e = (int)sizeof(unsigned short) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindFloat);
}

template <typename T>
static inline hipChannelFormatDesc hipCreateChannelDesc() {
    return hipCreateChannelDesc(0, 0, 0, 0, hipChannelFormatKindNone);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<char>() {
    int e = (int)sizeof(char) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<signed char>() {
    int e = (int)sizeof(signed char) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<unsigned char>() {
    int e = (int)sizeof(unsigned char) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<uchar1>() {
    int e = (int)sizeof(unsigned char) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<char1>() {
    int e = (int)sizeof(signed char) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<uchar2>() {
    int e = (int)sizeof(unsigned char) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<char2>() {
    int e = (int)sizeof(signed char) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindSigned);
}

#ifndef __GNUC__  // vector3 is the same as vector4
template <>
inline hipChannelFormatDesc hipCreateChannelDesc<uchar3>() {
    int e = (int)sizeof(unsigned char) * 8;
    return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<char3>() {
    int e = (int)sizeof(signed char) * 8;
    return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindSigned);
}
#endif

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<uchar4>() {
    int e = (int)sizeof(unsigned char) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<char4>() {
    int e = (int)sizeof(signed char) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<unsigned short>() {
    int e = (int)sizeof(unsigned short) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<signed short>() {
    int e = (int)sizeof(signed short) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<ushort1>() {
    int e = (int)sizeof(unsigned short) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<short1>() {
    int e = (int)sizeof(signed short) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<ushort2>() {
    int e = (int)sizeof(unsigned short) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<short2>() {
    int e = (int)sizeof(signed short) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindSigned);
}

#ifndef __GNUC__
template <>
inline hipChannelFormatDesc hipCreateChannelDesc<ushort3>() {
    int e = (int)sizeof(unsigned short) * 8;
    return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<short3>() {
    int e = (int)sizeof(signed short) * 8;
    return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindSigned);
}
#endif

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<ushort4>() {
    int e = (int)sizeof(unsigned short) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<short4>() {
    int e = (int)sizeof(signed short) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<unsigned int>() {
    int e = (int)sizeof(unsigned int) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<signed int>() {
    int e = (int)sizeof(signed int) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<uint1>() {
    int e = (int)sizeof(unsigned int) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<int1>() {
    int e = (int)sizeof(signed int) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<uint2>() {
    int e = (int)sizeof(unsigned int) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<int2>() {
    int e = (int)sizeof(signed int) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindSigned);
}

#ifndef __GNUC__
template <>
inline hipChannelFormatDesc hipCreateChannelDesc<uint3>() {
    int e = (int)sizeof(unsigned int) * 8;
    return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<int3>() {
    int e = (int)sizeof(signed int) * 8;
    return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindSigned);
}
#endif

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<uint4>() {
    int e = (int)sizeof(unsigned int) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<int4>() {
    int e = (int)sizeof(signed int) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<float>() {
    int e = (int)sizeof(float) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindFloat);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<float1>() {
    int e = (int)sizeof(float) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindFloat);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<float2>() {
    int e = (int)sizeof(float) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindFloat);
}

#ifndef __GNUC__
template <>
inline hipChannelFormatDesc hipCreateChannelDesc<float3>() {
    int e = (int)sizeof(float) * 8;
    return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindFloat);
}
#endif

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<float4>() {
    int e = (int)sizeof(float) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindFloat);
}

#if !defined(__LP64__)

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<unsigned long>() {
    int e = (int)sizeof(unsigned long) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<signed long>() {
    int e = (int)sizeof(signed long) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<ulong1>() {
    int e = (int)sizeof(unsigned long) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<long1>() {
    int e = (int)sizeof(signed long) * 8;
    return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<ulong2>() {
    int e = (int)sizeof(unsigned long) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<long2>() {
    int e = (int)sizeof(signed long) * 8;
    return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindSigned);
}

#ifndef __GNUC__
template <>
inline hipChannelFormatDesc hipCreateChannelDesc<ulong3>() {
    int e = (int)sizeof(unsigned long) * 8;
    return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<long3>() {
    int e = (int)sizeof(signed long) * 8;
    return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindSigned);
}
#endif

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<ulong4>() {
    int e = (int)sizeof(unsigned long) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindUnsigned);
}

template <>
inline hipChannelFormatDesc hipCreateChannelDesc<long4>() {
    int e = (int)sizeof(signed long) * 8;
    return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindSigned);
}
#endif /* !__LP64__ */

#else

struct hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w,
                                                 enum hipChannelFormatKind f);

#endif /* __cplusplus */

#endif /* !HIP_INCLUDE_HIP_AMD_DETAIL_CHANNEL_DESCRIPTOR_H */
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HIP_INCLUDE_HIP_TEXTURE_TYPES_H
#define HIP_INCLUDE_HIP_TEXTURE_TYPES_H

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-identifier"
#pragma clang diagnostic ignored "-Wreserved-macro-identifier"
#pragma clang diagnostic ignored "-Wc++98-compat"
#endif

#if !defined(__HIPCC_RTC__)
#include <hip/hip_common.h>
#endif

#if !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include "texture_types.h"
#elif defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
#if !defined(__HIPCC_RTC__)
#include <limits.h>
#include <hip/channel_descriptor.h>
#include <hip/driver_types.h>
#endif // !defined(__HIPCC_RTC__)

#define hipTextureType1D 0x01
#define hipTextureType2D 0x02
#define hipTextureType3D 0x03
#define hipTextureTypeCubemap 0x0C
#define hipTextureType1DLayered 0xF1
#define hipTextureType2DLayered 0xF2
#define hipTextureTypeCubemapLayered 0xFC

/**
 * Should be same as HSA_IMAGE_OBJECT_SIZE_DWORD/HSA_SAMPLER_OBJECT_SIZE_DWORD
 */
#define HIP_IMAGE_OBJECT_SIZE_DWORD 12
#define HIP_SAMPLER_OBJECT_SIZE_DWORD 8
#define HIP_SAMPLER_OBJECT_OFFSET_DWORD HIP_IMAGE_OBJECT_SIZE_DWORD
#define HIP_TEXTURE_OBJECT_SIZE_DWORD (HIP_IMAGE_OBJECT_SIZE_DWORD + HIP_SAMPLER_OBJECT_SIZE_DWORD)

/**
 * An opaque value that represents a hip texture object
 */
struct __hip_texture;
typedef struct __hip_texture* hipTextureObject_t;

/**
 * hip texture address modes
 */
enum hipTextureAddressMode {
    hipAddressModeWrap = 0,
    hipAddressModeClamp = 1,
    hipAddressModeMirror = 2,
    hipAddressModeBorder = 3
};

/**
 * hip texture filter modes
 */
enum hipTextureFilterMode { hipFilterModePoint = 0, hipFilterModeLinear = 1 };

/**
 * hip texture read modes
 */
enum hipTextureReadMode { hipReadModeElementType = 0, hipReadModeNormalizedFloat = 1 };

/**
 * hip texture reference
 */
typedef struct textureReference {
    int normalized;
    enum hipTextureReadMode readMode;// used only for driver API's
    enum hipTextureFilterMode filterMode;
    enum hipTextureAddressMode addressMode[3];  // Texture address mode for up to 3 dimensions
    struct hipChannelFormatDesc channelDesc;
    int sRGB;                    // Perform sRGB->linear conversion during texture read
    unsigned int maxAnisotropy;  // Limit to the anisotropy ratio
    enum hipTextureFilterMode mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;

    hipTextureObject_t textureObject;
    int numChannels;
    enum hipArray_Format format;
}textureReference;

/**
 * hip texture descriptor
 */
typedef struct hipTextureDesc {
    enum hipTextureAddressMode addressMode[3];  // Texture address mode for up to 3 dimensions
    enum hipTextureFilterMode filterMode;
    enum hipTextureReadMode readMode;
    int sRGB;  // Perform sRGB->linear conversion during texture read
    float borderColor[4];
    int normalizedCoords;
    unsigned int maxAnisotropy;
    enum hipTextureFilterMode mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
}hipTextureDesc;

#if __cplusplus

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
#if __HIP__
#define __HIP_TEXTURE_ATTRIB __attribute__((device_builtin_texture_type))
#else
#define __HIP_TEXTURE_ATTRIB
#endif

typedef textureReference* hipTexRef;

template <class T, int texType = hipTextureType1D,
          enum hipTextureReadMode mode = hipReadModeElementType>
struct __HIP_TEXTURE_ATTRIB texture : public textureReference {
    texture(int norm = 0, enum hipTextureFilterMode fMode = hipFilterModePoint,
            enum hipTextureAddressMode aMode = hipAddressModeClamp) {
        normalized = norm;
        readMode = mode;
        filterMode = fMode;
        addressMode[0] = aMode;
        addressMode[1] = aMode;
        addressMode[2] = aMode;
        channelDesc = hipCreateChannelDesc<T>();
        sRGB = 0;
        textureObject = nullptr;
        maxAnisotropy = 0;
        mipmapLevelBias = 0;
        minMipmapLevelClamp = 0;
        maxMipmapLevelClamp = 0;
    }

    texture(int norm, enum hipTextureFilterMode fMode, enum hipTextureAddressMode aMode,
            struct hipChannelFormatDesc desc) {
        normalized = norm;
        readMode = mode;
        filterMode = fMode;
        addressMode[0] = aMode;
        addressMode[1] = aMode;
        addressMode[2] = aMode;
        channelDesc = desc;
        sRGB = 0;
        textureObject = nullptr;
        maxAnisotropy = 0;
        mipmapLevelBias = 0;
        minMipmapLevelClamp = 0;
        maxMipmapLevelClamp = 0;
    }
};

#endif /* __cplusplus */

#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#if !defined(__HIPCC_RTC__)
#include <hip/hip_vector_types.h>
#endif

extern "C" {

#define ADDRESS_SPACE_CONSTANT __attribute__((address_space(4)))

__device__ float4::Native_vec_ __ockl_image_load_1D(unsigned int ADDRESS_SPACE_CONSTANT*i, int c);

__device__ float4::Native_vec_ __ockl_image_load_1Db(unsigned int ADDRESS_SPACE_CONSTANT*i, int c);

__device__ float4::Native_vec_ __ockl_image_load_1Da(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_load_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_load_2Da(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_load_3D(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_load_CM(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, int f);

__device__ float4::Native_vec_ __ockl_image_load_CMa(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, int f);

__device__ float4::Native_vec_ __ockl_image_load_lod_1D(unsigned int ADDRESS_SPACE_CONSTANT*i, int c, int l);

__device__ float4::Native_vec_ __ockl_image_load_lod_1Da(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, int l);

__device__ float4::Native_vec_ __ockl_image_load_lod_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, int l);

__device__ float4::Native_vec_ __ockl_image_load_lod_2Da(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, int l);

__device__ float4::Native_vec_ __ockl_image_load_lod_3D(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, int l);

__device__ float4::Native_vec_ __ockl_image_load_lod_CM(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, int f, int l);

__device__ float4::Native_vec_ __ockl_image_load_lod_CMa(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, int f, int l);

__device__ void __ockl_image_store_1D(unsigned int ADDRESS_SPACE_CONSTANT*i, int c, float4::Native_vec_ p);

__device__ void __ockl_image_store_1Da(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, float4::Native_vec_ p);

__device__ void __ockl_image_store_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, float4::Native_vec_ p);

__device__ void __ockl_image_store_2Da(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, float4::Native_vec_ p);

__device__ void __ockl_image_store_3D(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, float4::Native_vec_ p);

__device__ void __ockl_image_store_CM(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, int f, float4::Native_vec_ p);

__device__ void __ockl_image_store_CMa(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, int f, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_1D(unsigned int ADDRESS_SPACE_CONSTANT*i, int c, int l, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_1Da(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, int l, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, int l, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_2Da(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, int l, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_3D(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, int l, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_CM(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, int f, int l, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_CMa(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, int f, int l, float4::Native_vec_ p);

__device__ float4::Native_vec_ __ockl_image_sample_1D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float c);

__device__ float4::Native_vec_ __ockl_image_sample_1Da(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_sample_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_sample_2Da(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_sample_3D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_sample_CM(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_sample_CMa(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_sample_grad_1D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float c, float dx, float dy);

__device__ float4::Native_vec_ __ockl_image_sample_grad_1Da(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c, float dx, float dy);

__device__ float4::Native_vec_ __ockl_image_sample_grad_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c, float2::Native_vec_ dx, float2::Native_vec_ dy);

__device__ float4::Native_vec_ __ockl_image_sample_grad_2Da(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c, float2::Native_vec_ dx, float2::Native_vec_ dy);

__device__ float4::Native_vec_ __ockl_image_sample_grad_3D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c, float4::Native_vec_ dx, float4::Native_vec_ dy);

__device__ float4::Native_vec_ __ockl_image_sample_lod_1D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float c, float l);

__device__ float4::Native_vec_ __ockl_image_sample_lod_1Da(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c, float l);

__device__ float4::Native_vec_ __ockl_image_sample_lod_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c, float l);

__device__ float4::Native_vec_ __ockl_image_sample_lod_2Da(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c, float l);

__device__ float4::Native_vec_ __ockl_image_sample_lod_3D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c, float l);

__device__ float4::Native_vec_ __ockl_image_sample_lod_CM(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c, float l);

__device__ float4::Native_vec_ __ockl_image_sample_lod_CMa(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c, float l);

__device__ float4::Native_vec_ __ockl_image_gather4r_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_gather4g_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_gather4b_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_gather4a_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c);

__device__ int __ockl_image_channel_data_type_1D(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_1Da(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_1Db(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_2D(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_2Da(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_2Dad(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_2Dd(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_3D(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_CM(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_CMa(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_1D(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_1Da(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_1Db(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_2D(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_2Da(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_2Dad(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_2Dd(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_3D(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_CM(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_CMa(unsigned int ADDRESS_SPACE_CONSTANT* i);

}
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#if defined(__cplusplus)

#if !defined(__HIPCC_RTC__)
#include <hip/hip_vector_types.h>
#include <hip/hip_texture_types.h>
#include <hip/amd_detail/ockl_image.h>
#include <type_traits>
#endif // !defined(__HIPCC_RTC__)

#define TEXTURE_PARAMETERS_INIT                                                                     \
    unsigned int ADDRESS_SPACE_CONSTANT* i = (unsigned int ADDRESS_SPACE_CONSTANT*)t.textureObject; \
    unsigned int ADDRESS_SPACE_CONSTANT* s = i + HIP_SAMPLER_OBJECT_OFFSET_DWORD;

template<typename T>
struct __hip_is_tex_surf_scalar_channel_type
{
    static constexpr bool value =
        std::is_same<T, char>::value ||
        std::is_same<T, unsigned char>::value ||
        std::is_same<T, short>::value ||
        std::is_same<T, unsigned short>::value ||
        std::is_same<T, int>::value ||
        std::is_same<T, unsigned int>::value ||
        std::is_same<T, float>::value;
};

template<typename T>
struct __hip_is_tex_surf_channel_type
{
    static constexpr bool value =
        __hip_is_tex_surf_scalar_channel_type<T>::value;
};

template<
    typename T,
    unsigned int rank>
struct __hip_is_tex_surf_channel_type<HIP_vector_type<T, rank>>
{
    static constexpr bool value =
        __hip_is_tex_surf_scalar_channel_type<T>::value &&
        ((rank == 1) ||
         (rank == 2) ||
         (rank == 4));
};

template<typename T>
struct __hip_is_tex_normalized_channel_type
{
    static constexpr bool value =
        std::is_same<T, char>::value ||
        std::is_same<T, unsigned char>::value ||
        std::is_same<T, short>::value ||
        std::is_same<T, unsigned short>::value;
};

template<
    typename T,
    unsigned int rank>
struct __hip_is_tex_normalized_channel_type<HIP_vector_type<T, rank>>
{
    static constexpr bool value =
        __hip_is_tex_normalized_channel_type<T>::value &&
        ((rank == 1) ||
         (rank == 2) ||
         (rank == 4));
};

template <
    typename T,
    hipTextureReadMode readMode,
    typename Enable = void>
struct __hip_tex_ret
{
    static_assert(std::is_same<Enable, void>::value, "Invalid channel type!");
};

/*
 * Map from device function return U to scalar texture type T
 */
template<typename T, typename U>
__forceinline__ __device__
typename std::enable_if<
  __hip_is_tex_surf_scalar_channel_type<T>::value, const T>::type
__hipMapFrom(const U &u) {
  if constexpr (sizeof(T) < sizeof(float)) {
    union {
      U u;
      int i;
    } d = { u };
    return static_cast<T>(d.i);
  } else { // sizeof(T) == sizeof(float)
    union {
      U u;
      T t;
    } d = { u };
    return d.t;
  }
}

/*
 * Map from device function return U to vector texture type T
 */
template<typename T, typename U>
__forceinline__ __device__
typename std::enable_if<
  __hip_is_tex_surf_scalar_channel_type<typename T::value_type>::value, const T>::type
__hipMapFrom(const U &u) {
  if constexpr (sizeof(typename T::value_type) < sizeof(float)) {
    union {
      U u;
      int4 i4;
    } d = { u };
    return __hipMapVector<typename T::value_type, sizeof(T)/sizeof(typename T::value_type)>(d.i4);
  } else { // sizeof(typename T::value_type) == sizeof(float)
    union {
      U u;
      T t;
    } d = { u };
    return d.t;
  }
}

/*
 * Map from scalar texture type T to device function input U
 */
template<typename U, typename T>
__forceinline__ __device__
typename std::enable_if<
__hip_is_tex_surf_scalar_channel_type<T>::value, const U>::type
__hipMapTo(const T &t) {
  if constexpr (sizeof(T) < sizeof(float)) {
    union {
      U u;
      int i;
    } d = { 0 };
    d.i = static_cast<int>(t);
    return d.u;
  } else { // sizeof(T) == sizeof(float)
    union {
      U u;
      T t;
    } d = { 0 };
    d.t = t;
    return d.u;
  }
}

/*
 * Map from vector texture type T to device function input U
 */
template<typename U, typename T>
__forceinline__ __device__
typename std::enable_if<
  __hip_is_tex_surf_scalar_channel_type<typename T::value_type>::value, const U>::type
__hipMapTo(const T &t) {
  if constexpr (sizeof(typename T::value_type) < sizeof(float)) {
    union {
      U u;
      int4 i4;
    } d = { 0 };
    d.i4 = __hipMapVector<int, 4>(t);
    return d.u;
  } else { // sizeof(typename T::value_type) == sizeof(float)
    union {
      U u;
      T t;
    } d = { 0 };
    d.t = t;
    return d.u;
  }
}

template <
    typename T,
    hipTextureReadMode readMode>
using __hip_tex_ret_t = typename __hip_tex_ret<T, readMode, bool>::type;

template <typename T>
struct __hip_tex_ret<
    T,
    hipReadModeElementType,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value, bool>::type>
{
    using type = T;
};

template<
    typename T,
    unsigned int rank>
struct __hip_tex_ret<
    HIP_vector_type<T, rank>,
    hipReadModeElementType,
    typename std::enable_if<__hip_is_tex_surf_channel_type<HIP_vector_type<T, rank>>::value, bool>::type>
{
    using type = HIP_vector_type<__hip_tex_ret_t<T, hipReadModeElementType>, rank>;
};

template<typename T>
struct __hip_tex_ret<
    T,
    hipReadModeNormalizedFloat,
    typename std::enable_if<__hip_is_tex_normalized_channel_type<T>::value, bool>::type>
{
    using type = float;
};

template<
    typename T,
    unsigned int rank>
struct __hip_tex_ret<
    HIP_vector_type<T, rank>,
    hipReadModeNormalizedFloat,
    typename std::enable_if<__hip_is_tex_normalized_channel_type<HIP_vector_type<T, rank>>::value, bool>::type>
{
    using type = HIP_vector_type<__hip_tex_ret_t<T, hipReadModeNormalizedFloat>, rank>;
};


template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1Dfetch(texture<T, hipTextureType1D, readMode> t, int x)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_load_1Db(i, x);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1D(texture<T, hipTextureType1D, readMode> t, float x)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_1D(i, s, x);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex2D(texture<T, hipTextureType2D, readMode> t, float x, float y)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_2D(i, s, float2(x, y).data);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1DLayered(texture<T, hipTextureType1DLayered, readMode> t, float x, int layer)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex2DLayered(texture<T, hipTextureType2DLayered, readMode> t, float x, float y, int layer)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex3D(texture<T, hipTextureType3D, readMode> t, float x, float y, float z)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> texCubemap(texture<T, hipTextureTypeCubemap, readMode> t, float x, float y, float z)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_CM(i, s, float4(x, y, z, 0.0f).data);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1DLod(texture<T, hipTextureType1D, readMode> t, float x, float level)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_lod_1D(i, s, x, level);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex2DLod(texture<T, hipTextureType2D, readMode> t, float x, float y, float level)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1DLayeredLod(texture<T, hipTextureType1DLayered, readMode> t, float x, int layer, float level)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_lod_1Da(i, s, float2(x, layer).data, level);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex2DLayeredLod(texture<T, hipTextureType2DLayered, readMode> t, float x, float y, int layer, float level)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_lod_2Da(i, s, float4(x, y, layer, 0.0f).data, level);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex3DLod(texture<T, hipTextureType3D, readMode> t, float x, float y, float z, float level)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data, level);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> texCubemapLod(texture<T, hipTextureTypeCubemap, readMode> t, float x, float y, float z, float level)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_lod_CM(i, s, float4(x, y, z, 0.0f).data, level);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> texCubemapLayered(texture<T, hipTextureTypeCubemapLayered, readMode> t, float x, float y, float z, int layer)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_CMa(i, s, float4(x, y, z, layer).data);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> texCubemapLayeredLod(texture<T, hipTextureTypeCubemapLayered, readMode> t, float x, float y, float z, int layer, float level)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_lod_CMa(i, s, float4(x, y, z, layer).data, level);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> texCubemapGrad(texture<T, hipTextureTypeCubemap, readMode> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
    TEXTURE_PARAMETERS_INIT;
    // TODO missing in device libs.
    // auto tmp = __ockl_image_sample_grad_CM(i, s, float4(x, y, z, 0.0f).data, float4(dPdx.x, dPdx.y, dPdx.z, 0.0f).data, float4(dPdy.x, dPdy.y, dPdy.z, 0.0f).data);
    // return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
    return {};
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> texCubemapLayeredGrad(texture<T, hipTextureTypeCubemapLayered, readMode> t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy)
{
    TEXTURE_PARAMETERS_INIT;
    // TODO missing in device libs.
    // auto tmp = __ockl_image_sample_grad_CMa(i, s, float4(x, y, z, layer).data, float4(dPdx.x, dPdx.y, dPdx.z, 0.0f).data, float4(dPdy.x, dPdy.y, dPdy.z, 0.0f).data);
    // return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
    return {};
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1DGrad(texture<T, hipTextureType1D, readMode> t, float x, float dPdx, float dPdy)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_grad_1D(i, s, x, dPdx, dPdy);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex2DGrad(texture<T, hipTextureType2D, readMode> t, float x, float y, float2 dPdx, float2 dPdy)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_grad_2D(i, s, float2(x, y).data, float2(dPdx.x, dPdx.y).data,  float2(dPdy.x, dPdy.y).data);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1DLayeredGrad(texture<T, hipTextureType1DLayered, readMode> t, float x, int layer, float dPdx, float dPdy)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dPdx, dPdy);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex2DLayeredGrad(texture<T, hipTextureType2DLayered, readMode> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data, float2(dPdx.x, dPdx.y).data, float2(dPdy.x, dPdy.y).data);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex3DGrad(texture<T, hipTextureType3D, readMode> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{
    TEXTURE_PARAMETERS_INIT;
    auto tmp = __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data, float4(dPdx.x, dPdx.y, dPdx.z, 0.0f).data, float4(dPdy.x, dPdy.y, dPdy.z, 0.0f).data);
    return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <
    typename T,
    hipTextureReadMode readMode,
    typename Enable = void>
struct __hip_tex2dgather_ret
{
    static_assert(std::is_same<Enable, void>::value, "Invalid channel type!");
};

template <
    typename T,
    hipTextureReadMode readMode>
using __hip_tex2dgather_ret_t = typename __hip_tex2dgather_ret<T, readMode, bool>::type;

template <typename T>
struct __hip_tex2dgather_ret<
    T,
    hipReadModeElementType,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value, bool>::type>
{
    using type = HIP_vector_type<T, 4>;
};

template<
    typename T,
    unsigned int rank>
struct __hip_tex2dgather_ret<
    HIP_vector_type<T, rank>,
    hipReadModeElementType,
    typename std::enable_if<__hip_is_tex_surf_channel_type<HIP_vector_type<T, rank>>::value, bool>::type>
{
    using type = HIP_vector_type<T, 4>;
};

template <typename T>
struct __hip_tex2dgather_ret<
    T,
    hipReadModeNormalizedFloat,
    typename std::enable_if<__hip_is_tex_normalized_channel_type<T>::value, bool>::type>
{
    using type = float4;
};

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex2dgather_ret_t<T, readMode> tex2Dgather(texture<T, hipTextureType2D, readMode> t, float x, float y, int comp=0)
{
    TEXTURE_PARAMETERS_INIT;
    switch (comp) {
    case 1: {
        auto tmp = __ockl_image_gather4g_2D(i, s, float2(x, y).data);
        return __hipMapFrom<__hip_tex2dgather_ret_t<T, readMode>>(tmp);
    }
    case 2: {
        auto tmp = __ockl_image_gather4b_2D(i, s, float2(x, y).data);
        return __hipMapFrom<__hip_tex2dgather_ret_t<T, readMode>>(tmp);
    }
    case 3: {
        auto tmp = __ockl_image_gather4a_2D(i, s, float2(x, y).data);
        return __hipMapFrom<__hip_tex2dgather_ret_t<T, readMode>>(tmp);
    }
    default: {
        auto tmp = __ockl_image_gather4r_2D(i, s, float2(x, y).data);
        return __hipMapFrom<__hip_tex2dgather_ret_t<T, readMode>>(tmp);
    }
    }
    return {};
}

#endif
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#if defined(__cplusplus)

#if !defined(__HIPCC_RTC__)
#include <hip/hip_vector_types.h>
#include <hip/hip_texture_types.h>
#include <hip/amd_detail/texture_fetch_functions.h>
#include <hip/amd_detail/ockl_image.h>
#include <type_traits>
#endif // !defined(__HIPCC_RTC__)

#define TEXTURE_OBJECT_PARAMETERS_INIT                                                            \
    unsigned int ADDRESS_SPACE_CONSTANT* i = (unsigned int ADDRESS_SPACE_CONSTANT*)textureObject; \
    unsigned int ADDRESS_SPACE_CONSTANT* s = i + HIP_SAMPLER_OBJECT_OFFSET_DWORD;

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1Dfetch(hipTextureObject_t textureObject, int x)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_load_1Db(i, x);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1Dfetch(T *ptr, hipTextureObject_t textureObject, int x)
{
    *ptr = tex1Dfetch<T>(textureObject, x);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1D(hipTextureObject_t textureObject, float x)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_1D(i, s, x);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1D(T *ptr, hipTextureObject_t textureObject, float x)
{
    *ptr = tex1D<T>(textureObject, x);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2D(hipTextureObject_t textureObject, float x, float y)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_2D(i, s, float2(x, y).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2D(T *ptr, hipTextureObject_t textureObject, float x, float y)
{
    *ptr = tex2D<T>(textureObject, x, y);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex3D(hipTextureObject_t textureObject, float x, float y, float z)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex3D(T *ptr, hipTextureObject_t textureObject, float x, float y, float z)
{
    *ptr = tex3D<T>(textureObject, x, y, z);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1DLayered(hipTextureObject_t textureObject, float x, int layer)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1DLayered(T *ptr, hipTextureObject_t textureObject, float x, int layer)
{
    *ptr = tex1DLayered<T>(textureObject, x, layer);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2DLayered(hipTextureObject_t textureObject, float x, float y, int layer)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2DLayered(T *ptr, hipTextureObject_t textureObject, float x, float y, int layer)
{
    *ptr = tex1DLayered<T>(textureObject, x, y, layer);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__  T texCubemap(hipTextureObject_t textureObject, float x, float y, float z)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_CM(i, s, float4(x, y, z, 0.0f).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void texCubemap(T *ptr, hipTextureObject_t textureObject, float x, float y, float z)
{
    *ptr = texCubemap<T>(textureObject, x, y, z);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T texCubemapLayered(hipTextureObject_t textureObject, float x, float y, float z, int layer)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_CMa(i, s, float4(x, y, z, layer).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void texCubemapLayered(T *ptr, hipTextureObject_t textureObject, float x, float y, float z, int layer)
{
    *ptr = texCubemapLayered<T>(textureObject, x, y, z, layer);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2Dgather(hipTextureObject_t textureObject, float x, float y, int comp = 0)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    switch (comp) {
    case 1: {
        auto tmp = __ockl_image_gather4r_2D(i, s, float2(x, y).data);
        return __hipMapFrom<T>(tmp);
        break;
    }
    case 2: {
        auto tmp = __ockl_image_gather4g_2D(i, s, float2(x, y).data);
        return __hipMapFrom<T>(tmp);
        break;
    }
    case 3: {
        auto tmp = __ockl_image_gather4b_2D(i, s, float2(x, y).data);
        return __hipMapFrom<T>(tmp);
        break;
    }
    default: {
        auto tmp = __ockl_image_gather4a_2D(i, s, float2(x, y).data);
        return __hipMapFrom<T>(tmp);
        break;
    }
    }
    return {};
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2Dgather(T *ptr, hipTextureObject_t textureObject, float x, float y, int comp = 0)
{
    *ptr = texCubemapLayered<T>(textureObject, x, y, comp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1DLod(hipTextureObject_t textureObject, float x, float level)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_lod_1D(i, s, x, level);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1DLod(T *ptr, hipTextureObject_t textureObject, float x, float level)
{
    *ptr = tex1DLod<T>(textureObject, x, level);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2DLod(hipTextureObject_t textureObject, float x, float y, float level)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2DLod(T *ptr, hipTextureObject_t textureObject, float x, float y, float level)
{
    *ptr = tex2DLod<T>(textureObject, x, y, level);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex3DLod(hipTextureObject_t textureObject, float x, float y, float z, float level)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data, level);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex3DLod(T *ptr, hipTextureObject_t textureObject, float x, float y, float z, float level)
{
    *ptr = tex3DLod<T>(textureObject, x, y, z, level);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1DLayeredLod(hipTextureObject_t textureObject, float x, int layer, float level)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1DLayeredLod(T *ptr, hipTextureObject_t textureObject, float x, int layer, float level)
{
    *ptr = tex1DLayeredLod<T>(textureObject, x, layer, level);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__  T tex2DLayeredLod(hipTextureObject_t textureObject, float x, float y, int layer, float level)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2DLayeredLod(T *ptr, hipTextureObject_t textureObject, float x, float y, int layer, float level)
{
    *ptr = tex2DLayeredLod<T>(textureObject, x, y, layer, level);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T texCubemapLod(hipTextureObject_t textureObject, float x, float y, float z, float level)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_lod_CM(i, s, float4(x, y, z, 0.0f).data, level);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void texCubemapLod(T *ptr, hipTextureObject_t textureObject, float x, float y, float z, float level)
{
    *ptr = texCubemapLod<T>(textureObject, x, y, z, level);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T texCubemapGrad(hipTextureObject_t textureObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    // TODO missing in device libs.
    // auto tmp = __ockl_image_sample_grad_CM(i, s, float4(x, y, z, 0.0f).data, float4(dPdx.x, dPdx.y, dPdx.z, 0.0f).data, float4(dPdy.x, dPdy.y, dPdy.z, 0.0f).data);
    // return __hipMapFrom<T>(tmp);
    return {};
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void texCubemapGrad(T *ptr, hipTextureObject_t textureObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
    *ptr = texCubemapGrad<T>(textureObject, x, y, z, dPdx, dPdy);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T texCubemapLayeredLod(hipTextureObject_t textureObject, float x, float y, float z, int layer, float level)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_lod_CMa(i, s, float4(x, y, z, layer).data, level);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void texCubemapLayeredLod(T *ptr, hipTextureObject_t textureObject, float x, float y, float z, int layer, float level)
{
    *ptr = texCubemapLayeredLod<T>(textureObject, x, y, z, layer, level);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1DGrad(hipTextureObject_t textureObject, float x, float dPdx, float dPdy)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_grad_1D(i, s, x, dPdx, dPdy);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1DGrad(T *ptr, hipTextureObject_t textureObject, float x, float dPdx, float dPdy)
{
    *ptr = tex1DGrad<T>(textureObject, x, dPdx, dPdy);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2DGrad(hipTextureObject_t textureObject, float x, float y, float2 dPdx, float2 dPdy)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_grad_2D(i, s, float2(x, y).data, float2(dPdx.x, dPdx.y).data,  float2(dPdy.x, dPdy.y).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2DGrad(T *ptr, hipTextureObject_t textureObject, float x, float y, float2 dPdx, float2 dPdy)
{
    *ptr = tex2DGrad<T>(textureObject, x, y, dPdx, dPdy);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex3DGrad(hipTextureObject_t textureObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data, float4(dPdx.x, dPdx.y, dPdx.z, 0.0f).data, float4(dPdy.x, dPdy.y, dPdy.z, 0.0f).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex3DGrad(T *ptr, hipTextureObject_t textureObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
    *ptr = tex3DGrad<T>(textureObject, x, y, z, dPdx, dPdy);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1DLayeredGrad(hipTextureObject_t textureObject, float x, int layer, float dPdx, float dPdy)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dPdx, dPdy);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1DLayeredGrad(T *ptr, hipTextureObject_t textureObject, float x, int layer, float dPdx, float dPdy)
{
    *ptr = tex1DLayeredGrad<T>(textureObject, x, layer, dPdx, dPdy);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2DLayeredGrad(hipTextureObject_t textureObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data, float2(dPdx.x, dPdx.y).data, float2(dPdy.x, dPdy.y).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2DLayeredGrad(T *ptr, hipTextureObject_t textureObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
    *ptr = tex2DLayeredGrad<T>(textureObject, x, y, layer, dPdx, dPdy);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__  T texCubemapLayeredGrad(hipTextureObject_t textureObject, float x, float y, float z, int layer, float4 dPdx, float4 dPdy)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    // TODO missing in device libs.
    // auto tmp = __ockl_image_sample_grad_CMa(i, s, float4(x, y, z, layer).data, float4(dPdx.x, dPdx.y, dPdx.z, 0.0f).data, float4(dPdy.x, dPdy.y, dPdy.z, 0.0f).data);
    // return __hipMapFrom<T>(tmp);
    return {};
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void texCubemapLayeredGrad(T *ptr, hipTextureObject_t textureObject, float x, float y, float z, int layer, float4 dPdx, float4 dPdy)
{
    *ptr = texCubemapLayeredGrad<T>(textureObject, x, y, z, layer, dPdx, dPdy);
}

#endif
/*
Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_SURFACE_FUNCTIONS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_SURFACE_FUNCTIONS_H

#if defined(__cplusplus)

#if !defined(__HIPCC_RTC__)
#include <hip/surface_types.h>
#include <hip/hip_vector_types.h>
#include <hip/amd_detail/texture_fetch_functions.h>
#include <hip/amd_detail/ockl_image.h>
#endif

#if defined(__HIPCC_RTC__)
#define __HOST_DEVICE__ __device__
#else
#define __HOST_DEVICE__ __host__ __device__
#endif

#define __HIP_SURFACE_OBJECT_PARAMETERS_INIT                                                            \
    unsigned int ADDRESS_SPACE_CONSTANT* i = (unsigned int ADDRESS_SPACE_CONSTANT*)surfObj; 

// CUDA is using byte address, need map to pixel address for HIP
static __HOST_DEVICE__ __forceinline__ int __hipGetPixelAddr(int x, int format, int order) {
    /*
    * use below format index to generate format LUT
      typedef enum {
        HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8 = 0,
        HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16 = 1,
        HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8 = 2,
        HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16 = 3,
        HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24 = 4,
        HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555 = 5,
        HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565 = 6,
        HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010 = 7,
        HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8 = 8,
        HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16 = 9,
        HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32 = 10,
        HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8 = 11,
        HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16 = 12,
        HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32 = 13,
        HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT = 14,
        HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT = 15
      } hsa_ext_image_channel_type_t;
    */
    static const int FormatLUT[] = { 0, 1, 0, 1, 3, 1, 1, 1, 0, 1, 2, 0, 1, 2, 1, 2 };
    x = FormatLUT[format] == 3 ? x / FormatLUT[format] : x >> FormatLUT[format];

    /*
    * use below order index to generate order LUT
      typedef enum {
        HSA_EXT_IMAGE_CHANNEL_ORDER_A = 0,
        HSA_EXT_IMAGE_CHANNEL_ORDER_R = 1,
        HSA_EXT_IMAGE_CHANNEL_ORDER_RX = 2,
        HSA_EXT_IMAGE_CHANNEL_ORDER_RG = 3,
        HSA_EXT_IMAGE_CHANNEL_ORDER_RGX = 4,
        HSA_EXT_IMAGE_CHANNEL_ORDER_RA = 5,
        HSA_EXT_IMAGE_CHANNEL_ORDER_RGB = 6,
        HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX = 7,
        HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA = 8,
        HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA = 9,
        HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB = 10,
        HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR = 11,
        HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB = 12,
        HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX = 13,
        HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA = 14,
        HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA = 15,
        HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY = 16,
        HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE = 17,
        HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH = 18,
        HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL = 19
      } hsa_ext_image_channel_order_t;
    */
    static const int OrderLUT[] = { 0, 0, 1, 1, 3, 1, 3, 2, 2, 2, 2, 2, 3, 2, 2, 2, 0, 0, 0, 0 };
    return x = OrderLUT[order] == 3 ? x / OrderLUT[order] : x >> OrderLUT[order];
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf1Dread(T* data, hipSurfaceObject_t surfObj, int x,
        int boundaryMode = hipBoundaryModeZero) {
    __HIP_SURFACE_OBJECT_PARAMETERS_INIT
    x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_1D(i), __ockl_image_channel_order_1D(i));
    auto tmp = __ockl_image_load_1D(i, x);
    *data = __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf1Dwrite(T data, hipSurfaceObject_t surfObj, int x) {
    __HIP_SURFACE_OBJECT_PARAMETERS_INIT
    x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_1D(i), __ockl_image_channel_order_1D(i));
    auto tmp = __hipMapTo<float4::Native_vec_>(data);
    __ockl_image_store_1D(i, x, tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf2Dread(T* data, hipSurfaceObject_t surfObj, int x, int y) {
    __HIP_SURFACE_OBJECT_PARAMETERS_INIT
    x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
    auto tmp = __ockl_image_load_2D(i, int2(x, y).data);
    *data = __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf2Dwrite(T data, hipSurfaceObject_t surfObj, int x, int y) {
    __HIP_SURFACE_OBJECT_PARAMETERS_INIT
    x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
    auto tmp = __hipMapTo<float4::Native_vec_>(data);
    __ockl_image_store_2D(i, int2(x, y).data, tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf3Dread(T* data, hipSurfaceObject_t surfObj, int x, int y, int z) {
    __HIP_SURFACE_OBJECT_PARAMETERS_INIT
    x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_3D(i), __ockl_image_channel_order_3D(i));
    auto tmp = __ockl_image_load_3D(i, int4(x, y, z, 0).data);
    *data = __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf3Dwrite(T data, hipSurfaceObject_t surfObj, int x, int y, int z) {
    __HIP_SURFACE_OBJECT_PARAMETERS_INIT
    x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_3D(i), __ockl_image_channel_order_3D(i));
    auto tmp = __hipMapTo<float4::Native_vec_>(data);
    __ockl_image_store_3D(i, int4(x, y, z, 0).data, tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf1DLayeredread(T* data, hipSurfaceObject_t surfObj, int x, int layer) {
    __HIP_SURFACE_OBJECT_PARAMETERS_INIT
    x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_1D(i), __ockl_image_channel_order_1D(i));
    auto tmp = __ockl_image_load_lod_1D(i, x, layer);
    *data = __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf1DLayeredwrite(T data, hipSurfaceObject_t surfObj, int x, int layer) {
    __HIP_SURFACE_OBJECT_PARAMETERS_INIT
    x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_1D(i), __ockl_image_channel_order_1D(i));
    auto tmp = __hipMapTo<float4::Native_vec_>(data);
    __ockl_image_store_lod_1D(i, x, layer, tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf2DLayeredread(T* data, hipSurfaceObject_t surfObj, int x, int y, int layer) {
    __HIP_SURFACE_OBJECT_PARAMETERS_INIT
    x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
    auto tmp = __ockl_image_load_lod_2D(i, int2(x, y).data, layer);
    *data = __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf2DLayeredwrite(T data, hipSurfaceObject_t surfObj, int x, int y, int layer) {
    __HIP_SURFACE_OBJECT_PARAMETERS_INIT
    x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
    auto tmp = __hipMapTo<float4::Native_vec_>(data);
    __ockl_image_store_lod_2D(i, int2(x, y).data, layer, tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surfCubemapread(T* data, hipSurfaceObject_t surfObj, int x, int y, int face) {
    __HIP_SURFACE_OBJECT_PARAMETERS_INIT
    x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
    auto tmp = __ockl_image_load_CM(i, int2(x, y).data, face);
    *data = __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surfCubemapwrite(T data, hipSurfaceObject_t surfObj, int x, int y, int face) {
    __HIP_SURFACE_OBJECT_PARAMETERS_INIT
    x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
    auto tmp = __hipMapTo<float4::Native_vec_>(data);
    __ockl_image_store_CM(i, int2(x, y).data, face, tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surfCubemapLayeredread(T* data, hipSurfaceObject_t surfObj, int x, int y, int face,
        int layer) {
    __HIP_SURFACE_OBJECT_PARAMETERS_INIT
    x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
    auto tmp = __ockl_image_load_lod_CM(i, int2(x, y).data, face, layer);
    *data = __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surfCubemapLayeredwrite(T* data, hipSurfaceObject_t surfObj, int x, int y, int face,
        int layer) {
    __HIP_SURFACE_OBJECT_PARAMETERS_INIT
    x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
    auto tmp = __hipMapTo<float4::Native_vec_>(data);
    __ockl_image_store_lod_CM(i, int2(x, y).data, face, layer, tmp);
}

#endif

#endif
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COMPLEX_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COMPLEX_H

#if !defined(__HIPCC_RTC__)
#include "hip/amd_detail/amd_hip_vector_types.h"
#endif

#if defined(__HIPCC_RTC__)
#define __HOST_DEVICE__ __device__
#else
#define __HOST_DEVICE__ __host__ __device__
// TODO: Clang has a bug which allows device functions to call std functions
// when std functions are introduced into default namespace by using statement.
// math.h may be included after this bug is fixed.
#if __cplusplus
#include <cmath>
#else
#include "math.h"
#endif
#endif // !defined(__HIPCC_RTC__)

#if __cplusplus
#define COMPLEX_NEG_OP_OVERLOAD(type)                                                              \
    __HOST_DEVICE__ static inline type operator-(const type& op) {                             \
        type ret;                                                                                  \
        ret.x = -op.x;                                                                             \
        ret.y = -op.y;                                                                             \
        return ret;                                                                                \
    }

#define COMPLEX_EQ_OP_OVERLOAD(type)                                                               \
    __HOST_DEVICE__ static inline bool operator==(const type& lhs, const type& rhs) {          \
        return lhs.x == rhs.x && lhs.y == rhs.y;                                                   \
    }

#define COMPLEX_NE_OP_OVERLOAD(type)                                                               \
    __HOST_DEVICE__ static inline bool operator!=(const type& lhs, const type& rhs) {          \
        return !(lhs == rhs);                                                                      \
    }

#define COMPLEX_ADD_OP_OVERLOAD(type)                                                              \
    __HOST_DEVICE__ static inline type operator+(const type& lhs, const type& rhs) {           \
        type ret;                                                                                  \
        ret.x = lhs.x + rhs.x;                                                                     \
        ret.y = lhs.y + rhs.y;                                                                     \
        return ret;                                                                                \
    }

#define COMPLEX_SUB_OP_OVERLOAD(type)                                                              \
    __HOST_DEVICE__ static inline type operator-(const type& lhs, const type& rhs) {           \
        type ret;                                                                                  \
        ret.x = lhs.x - rhs.x;                                                                     \
        ret.y = lhs.y - rhs.y;                                                                     \
        return ret;                                                                                \
    }

#define COMPLEX_MUL_OP_OVERLOAD(type)                                                              \
    __HOST_DEVICE__ static inline type operator*(const type& lhs, const type& rhs) {           \
        type ret;                                                                                  \
        ret.x = lhs.x * rhs.x - lhs.y * rhs.y;                                                     \
        ret.y = lhs.x * rhs.y + lhs.y * rhs.x;                                                     \
        return ret;                                                                                \
    }

#define COMPLEX_DIV_OP_OVERLOAD(type)                                                              \
    __HOST_DEVICE__ static inline type operator/(const type& lhs, const type& rhs) {           \
        type ret;                                                                                  \
        ret.x = (lhs.x * rhs.x + lhs.y * rhs.y);                                                   \
        ret.y = (rhs.x * lhs.y - lhs.x * rhs.y);                                                   \
        ret.x = ret.x / (rhs.x * rhs.x + rhs.y * rhs.y);                                           \
        ret.y = ret.y / (rhs.x * rhs.x + rhs.y * rhs.y);                                           \
        return ret;                                                                                \
    }

#define COMPLEX_ADD_PREOP_OVERLOAD(type)                                                           \
    __HOST_DEVICE__ static inline type& operator+=(type& lhs, const type& rhs) {               \
        lhs.x += rhs.x;                                                                            \
        lhs.y += rhs.y;                                                                            \
        return lhs;                                                                                \
    }

#define COMPLEX_SUB_PREOP_OVERLOAD(type)                                                           \
    __HOST_DEVICE__ static inline type& operator-=(type& lhs, const type& rhs) {               \
        lhs.x -= rhs.x;                                                                            \
        lhs.y -= rhs.y;                                                                            \
        return lhs;                                                                                \
    }

#define COMPLEX_MUL_PREOP_OVERLOAD(type)                                                            \
    __HOST_DEVICE__ static inline type& operator*=(type& lhs, const type& rhs) {                    \
        type temp{lhs};                                                                             \
        lhs.x = rhs.x * temp.x - rhs.y * temp.y;                                                    \
        lhs.y = rhs.y * temp.x + rhs.x * temp.y;                                                    \
        return lhs;                                                                                 \
    }

#define COMPLEX_DIV_PREOP_OVERLOAD(type)                                                           \
    __HOST_DEVICE__ static inline type& operator/=(type& lhs, const type& rhs) {                   \
        type temp;                                                                                 \
        temp.x = (lhs.x*rhs.x + lhs.y * rhs.y) / (rhs.x*rhs.x + rhs.y*rhs.y);                      \
        temp.y = (lhs.y * rhs.x - lhs.x * rhs.y) / (rhs.x*rhs.x + rhs.y*rhs.y);                    \
        lhs = temp;                                                                                \
        return lhs;                                                                                \
    }

#define COMPLEX_SCALAR_PRODUCT(type, type1)                                                        \
    __HOST_DEVICE__ static inline type operator*(const type& lhs, type1 rhs) {                 \
        type ret;                                                                                  \
        ret.x = lhs.x * rhs;                                                                       \
        ret.y = lhs.y * rhs;                                                                       \
        return ret;                                                                                \
    }

#endif

typedef float2 hipFloatComplex;

__HOST_DEVICE__ static inline float hipCrealf(hipFloatComplex z) { return z.x; }

__HOST_DEVICE__ static inline float hipCimagf(hipFloatComplex z) { return z.y; }

__HOST_DEVICE__ static inline hipFloatComplex make_hipFloatComplex(float a, float b) {
    hipFloatComplex z;
    z.x = a;
    z.y = b;
    return z;
}

__HOST_DEVICE__ static inline hipFloatComplex hipConjf(hipFloatComplex z) {
    hipFloatComplex ret;
    ret.x = z.x;
    ret.y = -z.y;
    return ret;
}

__HOST_DEVICE__ static inline float hipCsqabsf(hipFloatComplex z) {
    return z.x * z.x + z.y * z.y;
}

__HOST_DEVICE__ static inline hipFloatComplex hipCaddf(hipFloatComplex p, hipFloatComplex q) {
    return make_hipFloatComplex(p.x + q.x, p.y + q.y);
}

__HOST_DEVICE__ static inline hipFloatComplex hipCsubf(hipFloatComplex p, hipFloatComplex q) {
    return make_hipFloatComplex(p.x - q.x, p.y - q.y);
}

__HOST_DEVICE__ static inline hipFloatComplex hipCmulf(hipFloatComplex p, hipFloatComplex q) {
    return make_hipFloatComplex(p.x * q.x - p.y * q.y, p.y * q.x + p.x * q.y);
}

__HOST_DEVICE__ static inline hipFloatComplex hipCdivf(hipFloatComplex p, hipFloatComplex q) {
    float sqabs = hipCsqabsf(q);
    hipFloatComplex ret;
    ret.x = (p.x * q.x + p.y * q.y) / sqabs;
    ret.y = (p.y * q.x - p.x * q.y) / sqabs;
    return ret;
}

__HOST_DEVICE__ static inline float hipCabsf(hipFloatComplex z) { return sqrtf(hipCsqabsf(z)); }


typedef double2 hipDoubleComplex;

__HOST_DEVICE__ static inline double hipCreal(hipDoubleComplex z) { return z.x; }

__HOST_DEVICE__ static inline double hipCimag(hipDoubleComplex z) { return z.y; }

__HOST_DEVICE__ static inline hipDoubleComplex make_hipDoubleComplex(double a, double b) {
    hipDoubleComplex z;
    z.x = a;
    z.y = b;
    return z;
}

__HOST_DEVICE__ static inline hipDoubleComplex hipConj(hipDoubleComplex z) {
    hipDoubleComplex ret;
    ret.x = z.x;
    ret.y = -z.y;
    return ret;
}

__HOST_DEVICE__ static inline double hipCsqabs(hipDoubleComplex z) {
    return z.x * z.x + z.y * z.y;
}

__HOST_DEVICE__ static inline hipDoubleComplex hipCadd(hipDoubleComplex p, hipDoubleComplex q) {
    return make_hipDoubleComplex(p.x + q.x, p.y + q.y);
}

__HOST_DEVICE__ static inline hipDoubleComplex hipCsub(hipDoubleComplex p, hipDoubleComplex q) {
    return make_hipDoubleComplex(p.x - q.x, p.y - q.y);
}

__HOST_DEVICE__ static inline hipDoubleComplex hipCmul(hipDoubleComplex p, hipDoubleComplex q) {
    return make_hipDoubleComplex(p.x * q.x - p.y * q.y, p.y * q.x + p.x * q.y);
}

__HOST_DEVICE__ static inline hipDoubleComplex hipCdiv(hipDoubleComplex p, hipDoubleComplex q) {
    double sqabs = hipCsqabs(q);
    hipDoubleComplex ret;
    ret.x = (p.x * q.x + p.y * q.y) / sqabs;
    ret.y = (p.y * q.x - p.x * q.y) / sqabs;
    return ret;
}

__HOST_DEVICE__ static inline double hipCabs(hipDoubleComplex z) { return sqrt(hipCsqabs(z)); }


#if __cplusplus

COMPLEX_NEG_OP_OVERLOAD(hipFloatComplex)
COMPLEX_EQ_OP_OVERLOAD(hipFloatComplex)
COMPLEX_NE_OP_OVERLOAD(hipFloatComplex)
COMPLEX_ADD_OP_OVERLOAD(hipFloatComplex)
COMPLEX_SUB_OP_OVERLOAD(hipFloatComplex)
COMPLEX_MUL_OP_OVERLOAD(hipFloatComplex)
COMPLEX_DIV_OP_OVERLOAD(hipFloatComplex)
COMPLEX_ADD_PREOP_OVERLOAD(hipFloatComplex)
COMPLEX_SUB_PREOP_OVERLOAD(hipFloatComplex)
COMPLEX_MUL_PREOP_OVERLOAD(hipFloatComplex)
COMPLEX_DIV_PREOP_OVERLOAD(hipFloatComplex)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, unsigned short)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, signed short)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, unsigned int)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, signed int)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, float)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, unsigned long)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, signed long)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, double)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, signed long long)
COMPLEX_SCALAR_PRODUCT(hipFloatComplex, unsigned long long)

COMPLEX_NEG_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_EQ_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_NE_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_ADD_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_SUB_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_MUL_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_DIV_OP_OVERLOAD(hipDoubleComplex)
COMPLEX_ADD_PREOP_OVERLOAD(hipDoubleComplex)
COMPLEX_SUB_PREOP_OVERLOAD(hipDoubleComplex)
COMPLEX_MUL_PREOP_OVERLOAD(hipDoubleComplex)
COMPLEX_DIV_PREOP_OVERLOAD(hipDoubleComplex)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, unsigned short)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, signed short)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, unsigned int)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, signed int)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, float)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, unsigned long)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, signed long)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, double)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, signed long long)
COMPLEX_SCALAR_PRODUCT(hipDoubleComplex, unsigned long long)

#endif


typedef hipFloatComplex hipComplex;

__HOST_DEVICE__ static inline hipComplex make_hipComplex(float x, float y) {
    return make_hipFloatComplex(x, y);
}

__HOST_DEVICE__ static inline hipFloatComplex hipComplexDoubleToFloat(hipDoubleComplex z) {
    return make_hipFloatComplex((float)z.x, (float)z.y);
}

__HOST_DEVICE__ static inline hipDoubleComplex hipComplexFloatToDouble(hipFloatComplex z) {
    return make_hipDoubleComplex((double)z.x, (double)z.y);
}

__HOST_DEVICE__ static inline hipComplex hipCfmaf(hipComplex p, hipComplex q, hipComplex r) {
    float real = (p.x * q.x) + r.x;
    float imag = (q.x * p.y) + r.y;

    real = -(p.y * q.y) + real;
    imag = (p.x * q.y) + imag;

    return make_hipComplex(real, imag);
}

__HOST_DEVICE__ static inline hipDoubleComplex hipCfma(hipDoubleComplex p, hipDoubleComplex q,
                                                           hipDoubleComplex r) {
    double real = (p.x * q.x) + r.x;
    double imag = (q.x * p.y) + r.y;

    real = -(p.y * q.y) + real;
    imag = (p.x * q.y) + imag;

    return make_hipDoubleComplex(real, imag);
}

#endif //HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COMPLEX_H
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#ifndef AMD_HIP_MATH_CONSTANTS_H
#define AMD_HIP_MATH_CONSTANTS_H

// single precision constants
#define HIP_INF_F            __int_as_float(0x7f800000U)
#define HIP_NAN_F            __int_as_float(0x7fffffffU)
#define HIP_MIN_DENORM_F     __int_as_float(0x00000001U)
#define HIP_MAX_NORMAL_F     __int_as_float(0x7f7fffffU)
#define HIP_NEG_ZERO_F       __int_as_float(0x80000000U)
#define HIP_ZERO_F           0.0F
#define HIP_ONE_F            1.0F
#define HIP_SQRT_HALF_F      0.707106781F
#define HIP_SQRT_HALF_HI_F   0.707106781F
#define HIP_SQRT_HALF_LO_F   1.210161749e-08F
#define HIP_SQRT_TWO_F       1.414213562F
#define HIP_THIRD_F          0.333333333F
#define HIP_PIO4_F           0.785398163F
#define HIP_PIO2_F           1.570796327F
#define HIP_3PIO4_F          2.356194490F
#define HIP_2_OVER_PI_F      0.636619772F
#define HIP_SQRT_2_OVER_PI_F 0.797884561F
#define HIP_PI_F             3.141592654F
#define HIP_L2E_F            1.442695041F
#define HIP_L2T_F            3.321928094F
#define HIP_LG2_F            0.301029996F
#define HIP_LGE_F            0.434294482F
#define HIP_LN2_F            0.693147181F
#define HIP_LNT_F            2.302585093F
#define HIP_LNPI_F           1.144729886F
#define HIP_TWO_TO_M126_F    1.175494351e-38F
#define HIP_TWO_TO_126_F     8.507059173e37F
#define HIP_NORM_HUGE_F      3.402823466e38F
#define HIP_TWO_TO_23_F      8388608.0F
#define HIP_TWO_TO_24_F      16777216.0F
#define HIP_TWO_TO_31_F      2147483648.0F
#define HIP_TWO_TO_32_F      4294967296.0F
#define HIP_REMQUO_BITS_F    3U
#define HIP_REMQUO_MASK_F    (~((~0U)<<HIP_REMQUO_BITS_F))
#define HIP_TRIG_PLOSS_F     105615.0F

// double precision constants
#define HIP_INF              __longlong_as_double(0x7ff0000000000000ULL)
#define HIP_NAN              __longlong_as_double(0xfff8000000000000ULL)
#define HIP_NEG_ZERO         __longlong_as_double(0x8000000000000000ULL)
#define HIP_MIN_DENORM       __longlong_as_double(0x0000000000000001ULL)
#define HIP_ZERO             0.0
#define HIP_ONE              1.0
#define HIP_SQRT_TWO         1.4142135623730951e+0
#define HIP_SQRT_HALF        7.0710678118654757e-1
#define HIP_SQRT_HALF_HI     7.0710678118654757e-1
#define HIP_SQRT_HALF_LO   (-4.8336466567264567e-17)
#define HIP_THIRD            3.3333333333333333e-1
#define HIP_TWOTHIRD         6.6666666666666667e-1
#define HIP_PIO4             7.8539816339744828e-1
#define HIP_PIO4_HI          7.8539816339744828e-1
#define HIP_PIO4_LO          3.0616169978683830e-17
#define HIP_PIO2             1.5707963267948966e+0
#define HIP_PIO2_HI          1.5707963267948966e+0
#define HIP_PIO2_LO          6.1232339957367660e-17
#define HIP_3PIO4            2.3561944901923448e+0
#define HIP_2_OVER_PI        6.3661977236758138e-1
#define HIP_PI               3.1415926535897931e+0
#define HIP_PI_HI            3.1415926535897931e+0
#define HIP_PI_LO            1.2246467991473532e-16
#define HIP_SQRT_2PI         2.5066282746310007e+0
#define HIP_SQRT_2PI_HI      2.5066282746310007e+0
#define HIP_SQRT_2PI_LO    (-1.8328579980459167e-16)
#define HIP_SQRT_PIO2        1.2533141373155003e+0
#define HIP_SQRT_PIO2_HI     1.2533141373155003e+0
#define HIP_SQRT_PIO2_LO   (-9.1642899902295834e-17)
#define HIP_SQRT_2OPI        7.9788456080286536e-1
#define HIP_L2E              1.4426950408889634e+0
#define HIP_L2E_HI           1.4426950408889634e+0
#define HIP_L2E_LO           2.0355273740931033e-17
#define HIP_L2T              3.3219280948873622e+0
#define HIP_LG2              3.0102999566398120e-1
#define HIP_LG2_HI           3.0102999566398120e-1
#define HIP_LG2_LO         (-2.8037281277851704e-18)
#define HIP_LGE              4.3429448190325182e-1
#define HIP_LGE_HI           4.3429448190325182e-1
#define HIP_LGE_LO           1.09831965021676510e-17
#define HIP_LN2              6.9314718055994529e-1
#define HIP_LN2_HI           6.9314718055994529e-1
#define HIP_LN2_LO           2.3190468138462996e-17
#define HIP_LNT              2.3025850929940459e+0
#define HIP_LNT_HI           2.3025850929940459e+0
#define HIP_LNT_LO         (-2.1707562233822494e-16)
#define HIP_LNPI             1.1447298858494002e+0
#define HIP_LN2_X_1024       7.0978271289338397e+2
#define HIP_LN2_X_1025       7.1047586007394398e+2
#define HIP_LN2_X_1075       7.4513321910194122e+2
#define HIP_LG2_X_1024       3.0825471555991675e+2
#define HIP_LG2_X_1075       3.2360724533877976e+2
#define HIP_TWO_TO_23        8388608.0
#define HIP_TWO_TO_52        4503599627370496.0
#define HIP_TWO_TO_53        9007199254740992.0
#define HIP_TWO_TO_54        18014398509481984.0
#define HIP_TWO_TO_M54       5.5511151231257827e-17
#define HIP_TWO_TO_M1022     2.22507385850720140e-308
#define HIP_TRIG_PLOSS       2147483648.0
#define HIP_DBL2INT_CVT      6755399441055744.0

#endif
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#if !defined(__HIPCC_RTC__)
#include "host_defines.h"
#include "amd_hip_vector_types.h"  // For Native_vec_
#endif

#if defined(__cplusplus)
    extern "C" {
#endif

// DOT FUNCTIONS
#if defined(__clang__) && defined(__HIP__)
__device__
__attribute__((const))
int __ockl_sdot2(
    HIP_vector_base<short, 2>::Native_vec_,
    HIP_vector_base<short, 2>::Native_vec_,
    int, bool);

__device__
__attribute__((const))
unsigned int __ockl_udot2(
    HIP_vector_base<unsigned short, 2>::Native_vec_,
    HIP_vector_base<unsigned short, 2>::Native_vec_,
    unsigned int, bool);

__device__
__attribute__((const))
int __ockl_sdot4(
    HIP_vector_base<char, 4>::Native_vec_,
    HIP_vector_base<char, 4>::Native_vec_,
    int, bool);

__device__
__attribute__((const))
unsigned int __ockl_udot4(
    HIP_vector_base<unsigned char, 4>::Native_vec_,
    HIP_vector_base<unsigned char, 4>::Native_vec_,
    unsigned int, bool);

__device__
__attribute__((const))
int __ockl_sdot8(int, int, int, bool);

__device__
__attribute__((const))
unsigned int __ockl_udot8(unsigned int, unsigned int, unsigned int, bool);
#endif

#if !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
// BEGIN FLOAT
__device__
__attribute__((const))
float __ocml_acos_f32(float);
__device__
__attribute__((pure))
float __ocml_acosh_f32(float);
__device__
__attribute__((const))
float __ocml_asin_f32(float);
__device__
__attribute__((pure))
float __ocml_asinh_f32(float);
__device__
__attribute__((const))
float __ocml_atan2_f32(float, float);
__device__
__attribute__((const))
float __ocml_atan_f32(float);
__device__
__attribute__((pure))
float __ocml_atanh_f32(float);
__device__
__attribute__((pure))
float __ocml_cbrt_f32(float);
__device__
__attribute__((const))
float __ocml_ceil_f32(float);
__device__
__attribute__((const))
__device__
float __ocml_copysign_f32(float, float);
__device__
float __ocml_cos_f32(float);
__device__
float __ocml_native_cos_f32(float);
__device__
__attribute__((pure))
__device__
float __ocml_cosh_f32(float);
__device__
float __ocml_cospi_f32(float);
__device__
float __ocml_i0_f32(float);
__device__
float __ocml_i1_f32(float);
__device__
__attribute__((pure))
float __ocml_erfc_f32(float);
__device__
__attribute__((pure))
float __ocml_erfcinv_f32(float);
__device__
__attribute__((pure))
float __ocml_erfcx_f32(float);
__device__
__attribute__((pure))
float __ocml_erf_f32(float);
__device__
__attribute__((pure))
float __ocml_erfinv_f32(float);
__device__
__attribute__((pure))
float __ocml_exp10_f32(float);
__device__
__attribute__((pure))
float __ocml_native_exp10_f32(float);
__device__
__attribute__((pure))
float __ocml_exp2_f32(float);
__device__
__attribute__((pure))
float __ocml_exp_f32(float);
__device__
__attribute__((pure))
float __ocml_native_exp_f32(float);
__device__
__attribute__((pure))
float __ocml_expm1_f32(float);
__device__
__attribute__((const))
float __ocml_fabs_f32(float);
__device__
__attribute__((const))
float __ocml_fdim_f32(float, float);
__device__
__attribute__((const))
float __ocml_floor_f32(float);
__device__
__attribute__((const))
float __ocml_fma_f32(float, float, float);
__device__
__attribute__((const))
float __ocml_fmax_f32(float, float);
__device__
__attribute__((const))
float __ocml_fmin_f32(float, float);
__device__
__attribute__((const))
__device__
float __ocml_fmod_f32(float, float);
__device__
float __ocml_frexp_f32(float, __attribute__((address_space(5))) int*);
__device__
__attribute__((const))
float __ocml_hypot_f32(float, float);
__device__
__attribute__((const))
int __ocml_ilogb_f32(float);
__device__
__attribute__((const))
int __ocml_isfinite_f32(float);
__device__
__attribute__((const))
int __ocml_isinf_f32(float);
__device__
__attribute__((const))
int __ocml_isnan_f32(float);
__device__
float __ocml_j0_f32(float);
__device__
float __ocml_j1_f32(float);
__device__
__attribute__((const))
float __ocml_ldexp_f32(float, int);
__device__
float __ocml_lgamma_f32(float);
__device__
__attribute__((pure))
float __ocml_log10_f32(float);
__device__
__attribute__((pure))
float __ocml_native_log10_f32(float);
__device__
__attribute__((pure))
float __ocml_log1p_f32(float);
__device__
__attribute__((pure))
float __ocml_log2_f32(float);
__device__
__attribute__((pure))
float __ocml_native_log2_f32(float);
__device__
__attribute__((const))
float __ocml_logb_f32(float);
__device__
__attribute__((pure))
float __ocml_log_f32(float);
__device__
__attribute__((pure))
float __ocml_native_log_f32(float);
__device__
float __ocml_modf_f32(float, __attribute__((address_space(5))) float*);
__device__
__attribute__((const))
float __ocml_nearbyint_f32(float);
__device__
__attribute__((const))
float __ocml_nextafter_f32(float, float);
__device__
__attribute__((const))
float __ocml_len3_f32(float, float, float);
__device__
__attribute__((const))
float __ocml_len4_f32(float, float, float, float);
__device__
__attribute__((pure))
float __ocml_ncdf_f32(float);
__device__
__attribute__((pure))
float __ocml_ncdfinv_f32(float);
__device__
__attribute__((pure))
float __ocml_pow_f32(float, float);
__device__
__attribute__((pure))
float __ocml_pown_f32(float, int);
__device__
__attribute__((pure))
float __ocml_rcbrt_f32(float);
__device__
__attribute__((const))
float __ocml_remainder_f32(float, float);
__device__
float __ocml_remquo_f32(float, float, __attribute__((address_space(5))) int*);
__device__
__attribute__((const))
float __ocml_rhypot_f32(float, float);
__device__
__attribute__((const))
float __ocml_rint_f32(float);
__device__
__attribute__((const))
float __ocml_rlen3_f32(float, float, float);
__device__
__attribute__((const))
float __ocml_rlen4_f32(float, float, float, float);
__device__
__attribute__((const))
float __ocml_round_f32(float);
__device__
__attribute__((pure))
float __ocml_rsqrt_f32(float);
__device__
__attribute__((const))
float __ocml_scalb_f32(float, float);
__device__
__attribute__((const))
float __ocml_scalbn_f32(float, int);
__device__
__attribute__((const))
int __ocml_signbit_f32(float);
__device__
float __ocml_sincos_f32(float, __attribute__((address_space(5))) float*);
__device__
float __ocml_sincospi_f32(float, __attribute__((address_space(5))) float*);
__device__
float __ocml_sin_f32(float);
__device__
float __ocml_native_sin_f32(float);
__device__
__attribute__((pure))
float __ocml_sinh_f32(float);
__device__
float __ocml_sinpi_f32(float);
__device__
__attribute__((const))
float __ocml_sqrt_f32(float);
__device__
__attribute__((const))
float __ocml_native_sqrt_f32(float);
__device__
float __ocml_tan_f32(float);
__device__
__attribute__((pure))
float __ocml_tanh_f32(float);
__device__
float __ocml_tgamma_f32(float);
__device__
__attribute__((const))
float __ocml_trunc_f32(float);
__device__
float __ocml_y0_f32(float);
__device__
float __ocml_y1_f32(float);

// BEGIN INTRINSICS
__device__
__attribute__((const))
float __ocml_add_rte_f32(float, float);
__device__
__attribute__((const))
float __ocml_add_rtn_f32(float, float);
__device__
__attribute__((const))
float __ocml_add_rtp_f32(float, float);
__device__
__attribute__((const))
float __ocml_add_rtz_f32(float, float);
__device__
__attribute__((const))
float __ocml_sub_rte_f32(float, float);
__device__
__attribute__((const))
float __ocml_sub_rtn_f32(float, float);
__device__
__attribute__((const))
float __ocml_sub_rtp_f32(float, float);
__device__
__attribute__((const))
float __ocml_sub_rtz_f32(float, float);
__device__
__attribute__((const))
float __ocml_mul_rte_f32(float, float);
__device__
__attribute__((const))
float __ocml_mul_rtn_f32(float, float);
__device__
__attribute__((const))
float __ocml_mul_rtp_f32(float, float);
__device__
__attribute__((const))
float __ocml_mul_rtz_f32(float, float);
__device__
__attribute__((const))
float __ocml_div_rte_f32(float, float);
__device__
__attribute__((const))
float __ocml_div_rtn_f32(float, float);
__device__
__attribute__((const))
float __ocml_div_rtp_f32(float, float);
__device__
__attribute__((const))
float __ocml_div_rtz_f32(float, float);
__device__
__attribute__((const))
float __ocml_sqrt_rte_f32(float);
__device__
__attribute__((const))
float __ocml_sqrt_rtn_f32(float);
__device__
__attribute__((const))
float __ocml_sqrt_rtp_f32(float);
__device__
__attribute__((const))
float __ocml_sqrt_rtz_f32(float);
__device__
__attribute__((const))
float __ocml_fma_rte_f32(float, float, float);
__device__
__attribute__((const))
float __ocml_fma_rtn_f32(float, float, float);
__device__
__attribute__((const))
float __ocml_fma_rtp_f32(float, float, float);
__device__
__attribute__((const))
float __ocml_fma_rtz_f32(float, float, float);
// END INTRINSICS
// END FLOAT

// BEGIN DOUBLE
__device__
__attribute__((const))
double __ocml_acos_f64(double);
__device__
__attribute__((pure))
double __ocml_acosh_f64(double);
__device__
__attribute__((const))
double __ocml_asin_f64(double);
__device__
__attribute__((pure))
double __ocml_asinh_f64(double);
__device__
__attribute__((const))
double __ocml_atan2_f64(double, double);
__device__
__attribute__((const))
double __ocml_atan_f64(double);
__device__
__attribute__((pure))
double __ocml_atanh_f64(double);
__device__
__attribute__((pure))
double __ocml_cbrt_f64(double);
__device__
__attribute__((const))
double __ocml_ceil_f64(double);
__device__
__attribute__((const))
double __ocml_copysign_f64(double, double);
__device__
double __ocml_cos_f64(double);
__device__
__attribute__((pure))
double __ocml_cosh_f64(double);
__device__
double __ocml_cospi_f64(double);
__device__
double __ocml_i0_f64(double);
__device__
double __ocml_i1_f64(double);
__device__
__attribute__((pure))
double __ocml_erfc_f64(double);
__device__
__attribute__((pure))
double __ocml_erfcinv_f64(double);
__device__
__attribute__((pure))
double __ocml_erfcx_f64(double);
__device__
__attribute__((pure))
double __ocml_erf_f64(double);
__device__
__attribute__((pure))
double __ocml_erfinv_f64(double);
__device__
__attribute__((pure))
double __ocml_exp10_f64(double);
__device__
__attribute__((pure))
double __ocml_exp2_f64(double);
__device__
__attribute__((pure))
double __ocml_exp_f64(double);
__device__
__attribute__((pure))
double __ocml_expm1_f64(double);
__device__
__attribute__((const))
double __ocml_fabs_f64(double);
__device__
__attribute__((const))
double __ocml_fdim_f64(double, double);
__device__
__attribute__((const))
double __ocml_floor_f64(double);
__device__
__attribute__((const))
double __ocml_fma_f64(double, double, double);
__device__
__attribute__((const))
double __ocml_fmax_f64(double, double);
__device__
__attribute__((const))
double __ocml_fmin_f64(double, double);
__device__
__attribute__((const))
double __ocml_fmod_f64(double, double);
__device__
double __ocml_frexp_f64(double, __attribute__((address_space(5))) int*);
__device__
__attribute__((const))
double __ocml_hypot_f64(double, double);
__device__
__attribute__((const))
int __ocml_ilogb_f64(double);
__device__
__attribute__((const))
int __ocml_isfinite_f64(double);
__device__
__attribute__((const))
int __ocml_isinf_f64(double);
__device__
__attribute__((const))
int __ocml_isnan_f64(double);
__device__
double __ocml_j0_f64(double);
__device__
double __ocml_j1_f64(double);
__device__
__attribute__((const))
double __ocml_ldexp_f64(double, int);
__device__
double __ocml_lgamma_f64(double);
__device__
__attribute__((pure))
double __ocml_log10_f64(double);
__device__
__attribute__((pure))
double __ocml_log1p_f64(double);
__device__
__attribute__((pure))
double __ocml_log2_f64(double);
__device__
__attribute__((const))
double __ocml_logb_f64(double);
__device__
__attribute__((pure))
double __ocml_log_f64(double);
__device__
double __ocml_modf_f64(double, __attribute__((address_space(5))) double*);
__device__
__attribute__((const))
double __ocml_nearbyint_f64(double);
__device__
__attribute__((const))
double __ocml_nextafter_f64(double, double);
__device__
__attribute__((const))
double __ocml_len3_f64(double, double, double);
__device__
__attribute__((const))
double __ocml_len4_f64(double, double, double, double);
__device__
__attribute__((pure))
double __ocml_ncdf_f64(double);
__device__
__attribute__((pure))
double __ocml_ncdfinv_f64(double);
__device__
__attribute__((pure))
double __ocml_pow_f64(double, double);
__device__
__attribute__((pure))
double __ocml_pown_f64(double, int);
__device__
__attribute__((pure))
double __ocml_rcbrt_f64(double);
__device__
__attribute__((const))
double __ocml_remainder_f64(double, double);
__device__
double __ocml_remquo_f64(
    double, double, __attribute__((address_space(5))) int*);
__device__
__attribute__((const))
double __ocml_rhypot_f64(double, double);
__device__
__attribute__((const))
double __ocml_rint_f64(double);
__device__
__attribute__((const))
double __ocml_rlen3_f64(double, double, double);
__device__
__attribute__((const))
double __ocml_rlen4_f64(double, double, double, double);
__device__
__attribute__((const))
double __ocml_round_f64(double);
__device__
__attribute__((pure))
double __ocml_rsqrt_f64(double);
__device__
__attribute__((const))
double __ocml_scalb_f64(double, double);
__device__
__attribute__((const))
double __ocml_scalbn_f64(double, int);
__device__
__attribute__((const))
int __ocml_signbit_f64(double);
__device__
double __ocml_sincos_f64(double, __attribute__((address_space(5))) double*);
__device__
double __ocml_sincospi_f64(double, __attribute__((address_space(5))) double*);
__device__
double __ocml_sin_f64(double);
__device__
__attribute__((pure))
double __ocml_sinh_f64(double);
__device__
double __ocml_sinpi_f64(double);
__device__
__attribute__((const))
double __ocml_sqrt_f64(double);
__device__
double __ocml_tan_f64(double);
__device__
__attribute__((pure))
double __ocml_tanh_f64(double);
__device__
double __ocml_tgamma_f64(double);
__device__
__attribute__((const))
double __ocml_trunc_f64(double);
__device__
double __ocml_y0_f64(double);
__device__
double __ocml_y1_f64(double);

// BEGIN INTRINSICS
__device__
__attribute__((const))
double __ocml_add_rte_f64(double, double);
__device__
__attribute__((const))
double __ocml_add_rtn_f64(double, double);
__device__
__attribute__((const))
double __ocml_add_rtp_f64(double, double);
__device__
__attribute__((const))
double __ocml_add_rtz_f64(double, double);
__device__
__attribute__((const))
double __ocml_sub_rte_f64(double, double);
__device__
__attribute__((const))
double __ocml_sub_rtn_f64(double, double);
__device__
__attribute__((const))
double __ocml_sub_rtp_f64(double, double);
__device__
__attribute__((const))
double __ocml_sub_rtz_f64(double, double);
__device__
__attribute__((const))
double __ocml_mul_rte_f64(double, double);
__device__
__attribute__((const))
double __ocml_mul_rtn_f64(double, double);
__device__
__attribute__((const))
double __ocml_mul_rtp_f64(double, double);
__device__
__attribute__((const))
double __ocml_mul_rtz_f64(double, double);
__device__
__attribute__((const))
double __ocml_div_rte_f64(double, double);
__device__
__attribute__((const))
double __ocml_div_rtn_f64(double, double);
__device__
__attribute__((const))
double __ocml_div_rtp_f64(double, double);
__device__
__attribute__((const))
double __ocml_div_rtz_f64(double, double);
__device__
__attribute__((const))
double __ocml_sqrt_rte_f64(double);
__device__
__attribute__((const))
double __ocml_sqrt_rtn_f64(double);
__device__
__attribute__((const))
double __ocml_sqrt_rtp_f64(double);
__device__
__attribute__((const))
double __ocml_sqrt_rtz_f64(double);
__device__
__attribute__((const))
double __ocml_fma_rte_f64(double, double, double);
__device__
__attribute__((const))
double __ocml_fma_rtn_f64(double, double, double);
__device__
__attribute__((const))
double __ocml_fma_rtp_f64(double, double, double);
__device__
__attribute__((const))
double __ocml_fma_rtz_f64(double, double, double);
// END INTRINSICS
// END DOUBLE

#endif // !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__

#if defined(__cplusplus)
    } // extern "C"
#endif
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 *  @file  amd_detail/device_library_decls.h
 *  @brief Contains declarations for types and functions in device library.
 *         Uses int64_t and uint64_t instead of long, long long, unsigned
 *         long and unsigned long long types for device library API
 *         declarations.
 */

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_DEVICE_LIBRARY_DECLS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_DEVICE_LIBRARY_DECLS_H

#if !defined(__HIPCC_RTC__)
#include "hip/amd_detail/host_defines.h"
#endif

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long long ullong;

extern "C" __device__ __attribute__((const)) bool __ockl_wfany_i32(int);
extern "C" __device__ __attribute__((const)) bool __ockl_wfall_i32(int);
extern "C" __device__ uint __ockl_activelane_u32(void);

extern "C" __device__ __attribute__((const)) uint __ockl_mul24_u32(uint, uint);
extern "C" __device__ __attribute__((const)) int __ockl_mul24_i32(int, int);
extern "C" __device__ __attribute__((const)) uint __ockl_mul_hi_u32(uint, uint);
extern "C" __device__ __attribute__((const)) int __ockl_mul_hi_i32(int, int);
extern "C" __device__ __attribute__((const)) uint __ockl_sadd_u32(uint, uint, uint);

extern "C" __device__ __attribute__((const)) uchar __ockl_clz_u8(uchar);
extern "C" __device__ __attribute__((const)) ushort __ockl_clz_u16(ushort);
extern "C" __device__ __attribute__((const)) uint __ockl_clz_u32(uint);
extern "C" __device__ __attribute__((const)) uint64_t __ockl_clz_u64(uint64_t);

extern "C" __device__ __attribute__((const)) float __ocml_floor_f32(float);
extern "C" __device__ __attribute__((const)) float __ocml_rint_f32(float);
extern "C" __device__ __attribute__((const)) float __ocml_ceil_f32(float);
extern "C" __device__ __attribute__((const)) float __ocml_trunc_f32(float);

extern "C" __device__ __attribute__((const)) float __ocml_fmin_f32(float, float);
extern "C" __device__ __attribute__((const)) float __ocml_fmax_f32(float, float);

extern "C" __device__ __attribute__((const)) float __ocml_cvtrtn_f32_f64(double);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtp_f32_f64(double);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtz_f32_f64(double);

extern "C" __device__ __attribute__((const)) _Float16 __ocml_cvtrtn_f16_f32(float);
extern "C" __device__ __attribute__((const)) _Float16 __ocml_cvtrtp_f16_f32(float);
extern "C" __device__ __attribute__((const)) _Float16 __ocml_cvtrtz_f16_f32(float);

extern "C" __device__ __attribute__((const)) float __ocml_cvtrtn_f32_s32(int);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtp_f32_s32(int);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtz_f32_s32(int);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtn_f32_u32(uint32_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtp_f32_u32(uint32_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtz_f32_u32(uint32_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtn_f32_s64(int64_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtp_f32_s64(int64_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtz_f32_s64(int64_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtn_f32_u64(uint64_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtp_f32_u64(uint64_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtz_f32_u64(uint64_t);
extern "C" __device__ __attribute__((const)) double __ocml_cvtrtn_f64_s64(int64_t);
extern "C" __device__ __attribute__((const)) double __ocml_cvtrtp_f64_s64(int64_t);
extern "C" __device__ __attribute__((const)) double __ocml_cvtrtz_f64_s64(int64_t);
extern "C" __device__ __attribute__((const)) double __ocml_cvtrtn_f64_u64(uint64_t);
extern "C" __device__ __attribute__((const)) double __ocml_cvtrtp_f64_u64(uint64_t);
extern "C" __device__ __attribute__((const)) double __ocml_cvtrtz_f64_u64(uint64_t);

extern "C" __device__ __attribute__((convergent)) void __ockl_gws_init(uint nwm1, uint rid);
extern "C" __device__ __attribute__((convergent)) void __ockl_gws_barrier(uint nwm1, uint rid);

extern "C" __device__ __attribute__((const)) uint32_t __ockl_lane_u32();
extern "C" __device__ __attribute__((const)) int __ockl_grid_is_valid(void);
extern "C" __device__ __attribute__((convergent)) void __ockl_grid_sync(void);
extern "C" __device__ __attribute__((const)) uint __ockl_multi_grid_num_grids(void);
extern "C" __device__ __attribute__((const)) uint __ockl_multi_grid_grid_rank(void);
extern "C" __device__ __attribute__((const)) uint __ockl_multi_grid_size(void);
extern "C" __device__ __attribute__((const)) uint __ockl_multi_grid_thread_rank(void);
extern "C" __device__ __attribute__((const)) int __ockl_multi_grid_is_valid(void);
extern "C" __device__ __attribute__((convergent)) void __ockl_multi_grid_sync(void);

extern "C" __device__ void __ockl_atomic_add_noret_f32(float*, float);

extern "C" __device__ __attribute__((convergent)) int __ockl_wgred_add_i32(int a);
extern "C" __device__ __attribute__((convergent)) int __ockl_wgred_and_i32(int a);
extern "C" __device__ __attribute__((convergent)) int __ockl_wgred_or_i32(int a);

extern "C" __device__ uint64_t __ockl_fprintf_stderr_begin();
extern "C" __device__ uint64_t __ockl_fprintf_append_args(uint64_t msg_desc, uint32_t num_args,
                                                          uint64_t value0, uint64_t value1,
                                                          uint64_t value2, uint64_t value3,
                                                          uint64_t value4, uint64_t value5,
                                                          uint64_t value6, uint32_t is_last);
extern "C" __device__ uint64_t __ockl_fprintf_append_string_n(uint64_t msg_desc, const char* data,
                                                              uint64_t length, uint32_t is_last);

// Introduce local address space
#define __local __attribute__((address_space(3)))

#ifdef __HIP_DEVICE_COMPILE__
__device__ inline static __local void* __to_local(unsigned x) { return (__local void*)x; }
#endif //__HIP_DEVICE_COMPILE__

// Using hip.amdgcn.bc - sync threads
#define __CLK_LOCAL_MEM_FENCE    0x01
typedef unsigned __cl_mem_fence_flags;

#endif
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_DEVICE_FUNCTIONS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_DEVICE_FUNCTIONS_H

#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/amd_hip_common.h>
#include "host_defines.h"
#include "math_fwd.h"
#include <hip/hip_runtime_api.h>
#include <stddef.h>
#include <hip/hip_vector_types.h>
#include <hip/amd_detail/device_library_decls.h>
#endif // !defined(__HIPCC_RTC__)

#if defined(__clang__) && defined(__HIP__)
extern "C" __device__ int printf(const char *fmt, ...);
#else
template <typename... All>
static inline __device__ void printf(const char* format, All... all) {}
#endif // __HIP_CLANG_ONLY__

extern "C" __device__ unsigned long long __ockl_steadyctr_u64();

/*
Integer Intrinsics
*/

// integer intrinsic function __poc __clz __ffs __brev
__device__ static inline unsigned int __popc(unsigned int input) {
    return __builtin_popcount(input);
}
__device__ static inline unsigned int __popcll(unsigned long long int input) {
    return __builtin_popcountll(input);
}

__device__ static inline int __clz(int input) {
    return __ockl_clz_u32((uint)input);
}

__device__ static inline int __clzll(long long int input) {
    return __ockl_clz_u64((uint64_t)input);
}

__device__ static inline unsigned int __ffs(unsigned int input) {
    return ( input == 0 ? -1 : __builtin_ctz(input) ) + 1;
}

__device__ static inline unsigned int __ffsll(unsigned long long int input) {
    return ( input == 0 ? -1 : __builtin_ctzll(input) ) + 1;
}

__device__ static inline unsigned int __ffs(int input) {
    return ( input == 0 ? -1 : __builtin_ctz(input) ) + 1;
}

__device__ static inline unsigned int __ffsll(long long int input) {
    return ( input == 0 ? -1 : __builtin_ctzll(input) ) + 1;
}

// Given a 32/64-bit value exec mask and an integer value base (between 0 and WAVEFRONT_SIZE),
// find the n-th (given by offset) set bit in the exec mask from the base bit, and return the bit position.
// If not found, return -1.
__device__  static int32_t __fns64(uint64_t mask, uint32_t base, int32_t offset) {
  uint64_t temp_mask = mask;
  int32_t temp_offset = offset;

  if (offset == 0) {
    temp_mask &= (1 << base);
    temp_offset = 1;
  }
  else if (offset < 0) {
    temp_mask = __builtin_bitreverse64(mask);
    base = 63 - base;
    temp_offset = -offset;
  }

  temp_mask = temp_mask & ((~0ULL) << base);
  if (__builtin_popcountll(temp_mask) < temp_offset)
    return -1;
  int32_t total = 0;
  for (int i = 0x20; i > 0; i >>= 1) {
    uint64_t temp_mask_lo = temp_mask & ((1ULL << i) - 1);
    int32_t pcnt = __builtin_popcountll(temp_mask_lo);
    if (pcnt < temp_offset) {
      temp_mask = temp_mask >> i;
      temp_offset -= pcnt;
      total += i;
    }
    else {
      temp_mask = temp_mask_lo;
    }
  }
  if (offset < 0)
    return 63 - total;
  else
    return total;
}

__device__ static int32_t __fns32(uint64_t mask, uint32_t base, int32_t offset) {
  uint64_t temp_mask = mask;
  int32_t temp_offset = offset;
  if (offset == 0) {
    temp_mask &= (1 << base);
    temp_offset = 1;
  }
  else if (offset < 0) {
    temp_mask = __builtin_bitreverse64(mask);
    base = 63 - base;
    temp_offset = -offset;
  }
  temp_mask = temp_mask & ((~0ULL) << base);
  if (__builtin_popcountll(temp_mask) < temp_offset)
    return -1;
  int32_t total = 0;
  for (int i = 0x20; i > 0; i >>= 1) {
    uint64_t temp_mask_lo = temp_mask & ((1ULL << i) - 1);
    int32_t pcnt = __builtin_popcountll(temp_mask_lo);
    if (pcnt < temp_offset) {
      temp_mask = temp_mask >> i;
      temp_offset -= pcnt;
      total += i;
    }
    else {
      temp_mask = temp_mask_lo;
    }
  }
  if (offset < 0)
    return 63 - total;
  else
    return total;
}
__device__ static inline unsigned int __brev(unsigned int input) {
    return __builtin_bitreverse32(input);
}

__device__ static inline unsigned long long int __brevll(unsigned long long int input) {
    return __builtin_bitreverse64(input);
}

__device__ static inline unsigned int __lastbit_u32_u64(uint64_t input) {
    return input == 0 ? -1 : __builtin_ctzl(input);
}

__device__ static inline unsigned int __bitextract_u32(unsigned int src0, unsigned int src1, unsigned int src2) {
    uint32_t offset = src1 & 31;
    uint32_t width = src2 & 31;
    return width == 0 ? 0 : (src0 << (32 - offset - width)) >> (32 - width);
}

__device__ static inline uint64_t __bitextract_u64(uint64_t src0, unsigned int src1, unsigned int src2) {
    uint64_t offset = src1 & 63;
    uint64_t width = src2 & 63;
    return width == 0 ? 0 : (src0 << (64 - offset - width)) >> (64 - width);
}

__device__ static inline unsigned int __bitinsert_u32(unsigned int src0, unsigned int src1, unsigned int src2, unsigned int src3) {
    uint32_t offset = src2 & 31;
    uint32_t width = src3 & 31;
    uint32_t mask = (1 << width) - 1;
    return ((src0 & ~(mask << offset)) | ((src1 & mask) << offset));
}

__device__ static inline uint64_t __bitinsert_u64(uint64_t src0, uint64_t src1, unsigned int src2, unsigned int src3) {
    uint64_t offset = src2 & 63;
    uint64_t width = src3 & 63;
    uint64_t mask = (1ULL << width) - 1;
    return ((src0 & ~(mask << offset)) | ((src1 & mask) << offset));
}

__device__ inline unsigned int __funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift)
{
    uint32_t mask_shift = shift & 31;
    return mask_shift == 0 ? hi : __builtin_amdgcn_alignbit(hi, lo, 32 - mask_shift);
}

__device__ inline unsigned int __funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift)
{
    uint32_t min_shift = shift >= 32 ? 32 : shift;
    return min_shift == 0 ? hi : __builtin_amdgcn_alignbit(hi, lo, 32 - min_shift);
}

__device__ inline unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift)
{
    return __builtin_amdgcn_alignbit(hi, lo, shift);
}

__device__ inline unsigned int __funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift)
{
    return shift >= 32 ? hi : __builtin_amdgcn_alignbit(hi, lo, shift);
}

__device__ static unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s);
__device__ static unsigned int __hadd(int x, int y);
__device__ static int __mul24(int x, int y);
__device__ static long long int __mul64hi(long long int x, long long int y);
__device__ static int __mulhi(int x, int y);
__device__ static int __rhadd(int x, int y);
__device__ static unsigned int __sad(int x, int y,unsigned int z);
__device__ static unsigned int __uhadd(unsigned int x, unsigned int y);
__device__ static int __umul24(unsigned int x, unsigned int y);
__device__ static unsigned long long int __umul64hi(unsigned long long int x, unsigned long long int y);
__device__ static unsigned int __umulhi(unsigned int x, unsigned int y);
__device__ static unsigned int __urhadd(unsigned int x, unsigned int y);
__device__ static unsigned int __usad(unsigned int x, unsigned int y, unsigned int z);

struct ucharHolder {
    union {
        unsigned char c[4];
        unsigned int ui;
    };
} __attribute__((aligned(4)));

struct uchar2Holder {
    union {
        unsigned int ui[2];
        unsigned char c[8];
    };
} __attribute__((aligned(8)));

__device__
static inline unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s) {
    struct uchar2Holder cHoldVal;
    struct ucharHolder cHoldKey;
    cHoldKey.ui = s;
    cHoldVal.ui[0] = x;
    cHoldVal.ui[1] = y;
    unsigned int result;
    result = cHoldVal.c[cHoldKey.c[0] & 0x07];
    result += (cHoldVal.c[(cHoldKey.c[0] & 0x70) >> 4] << 8);
    result += (cHoldVal.c[cHoldKey.c[1] & 0x07] << 16);
    result += (cHoldVal.c[(cHoldKey.c[1] & 0x70) >> 4] << 24);
    return result;
}

__device__ static inline unsigned int __hadd(int x, int y) {
    int z = x + y;
    int sign = z & 0x8000000;
    int value = z & 0x7FFFFFFF;
    return ((value) >> 1 || sign);
}

__device__ static inline int __mul24(int x, int y) {
    return __ockl_mul24_i32(x, y);
}

__device__ static inline long long __mul64hi(long long int x, long long int y) {
    ulong x0 = (ulong)x & 0xffffffffUL;
    long x1 = x >> 32;
    ulong y0 = (ulong)y & 0xffffffffUL;
    long y1 = y >> 32;
    ulong z0 = x0*y0;
    long t = x1*y0 + (z0 >> 32);
    long z1 = t & 0xffffffffL;
    long z2 = t >> 32;
    z1 = x0*y1 + z1;
    return x1*y1 + z2 + (z1 >> 32);
}

__device__ static inline int __mulhi(int x, int y) {
    return __ockl_mul_hi_i32(x, y);
}

__device__ static inline int __rhadd(int x, int y) {
    int z = x + y + 1;
    int sign = z & 0x8000000;
    int value = z & 0x7FFFFFFF;
    return ((value) >> 1 || sign);
}
__device__ static inline unsigned int __sad(int x, int y, unsigned int z) {
    return x > y ? x - y + z : y - x + z;
}
__device__ static inline unsigned int __uhadd(unsigned int x, unsigned int y) {
    return (x + y) >> 1;
}
__device__ static inline int __umul24(unsigned int x, unsigned int y) {
    return __ockl_mul24_u32(x, y);
}

__device__
static inline unsigned long long __umul64hi(unsigned long long int x, unsigned long long int y) {
    ulong x0 = x & 0xffffffffUL;
    ulong x1 = x >> 32;
    ulong y0 = y & 0xffffffffUL;
    ulong y1 = y >> 32;
    ulong z0 = x0*y0;
    ulong t = x1*y0 + (z0 >> 32);
    ulong z1 = t & 0xffffffffUL;
    ulong z2 = t >> 32;
    z1 = x0*y1 + z1;
    return x1*y1 + z2 + (z1 >> 32);
}

__device__ static inline unsigned int __umulhi(unsigned int x, unsigned int y) {
    return __ockl_mul_hi_u32(x, y);
}
__device__ static inline unsigned int __urhadd(unsigned int x, unsigned int y) {
    return (x + y + 1) >> 1;
}
__device__ static inline unsigned int __usad(unsigned int x, unsigned int y, unsigned int z) {
    return __ockl_sadd_u32(x, y, z);
}

__device__ static inline unsigned int __lane_id() {
    return  __builtin_amdgcn_mbcnt_hi(
        -1, __builtin_amdgcn_mbcnt_lo(-1, 0));
}

__device__
static inline unsigned int __mbcnt_lo(unsigned int x, unsigned int y) {return __builtin_amdgcn_mbcnt_lo(x,y);};

__device__
static inline unsigned int __mbcnt_hi(unsigned int x, unsigned int y) {return __builtin_amdgcn_mbcnt_hi(x,y);};

/*
HIP specific device functions
*/

#if !defined(__HIPCC_RTC__)
#include "amd_warp_functions.h"
#endif

#define MASK1 0x00ff00ff
#define MASK2 0xff00ff00

__device__ static inline char4 __hip_hc_add8pk(char4 in1, char4 in2) {
    char4 out;
    unsigned one1 = in1.w & MASK1;
    unsigned one2 = in2.w & MASK1;
    out.w = (one1 + one2) & MASK1;
    one1 = in1.w & MASK2;
    one2 = in2.w & MASK2;
    out.w = out.w | ((one1 + one2) & MASK2);
    return out;
}

__device__ static inline char4 __hip_hc_sub8pk(char4 in1, char4 in2) {
    char4 out;
    unsigned one1 = in1.w & MASK1;
    unsigned one2 = in2.w & MASK1;
    out.w = (one1 - one2) & MASK1;
    one1 = in1.w & MASK2;
    one2 = in2.w & MASK2;
    out.w = out.w | ((one1 - one2) & MASK2);
    return out;
}

__device__ static inline char4 __hip_hc_mul8pk(char4 in1, char4 in2) {
    char4 out;
    unsigned one1 = in1.w & MASK1;
    unsigned one2 = in2.w & MASK1;
    out.w = (one1 * one2) & MASK1;
    one1 = in1.w & MASK2;
    one2 = in2.w & MASK2;
    out.w = out.w | ((one1 * one2) & MASK2);
    return out;
}

__device__ static inline float __double2float_rd(double x) {
    return __ocml_cvtrtn_f32_f64(x);
}
__device__ static inline float __double2float_rn(double x) { return x; }
__device__ static inline float __double2float_ru(double x) {
    return __ocml_cvtrtp_f32_f64(x);
}
__device__ static inline float __double2float_rz(double x) {
    return __ocml_cvtrtz_f32_f64(x);
}

__device__ static inline int __double2hiint(double x) {
    static_assert(sizeof(double) == 2 * sizeof(int), "");

    int tmp[2];
    __builtin_memcpy(tmp, &x, sizeof(tmp));

    return tmp[1];
}
__device__ static inline int __double2loint(double x) {
    static_assert(sizeof(double) == 2 * sizeof(int), "");

    int tmp[2];
    __builtin_memcpy(tmp, &x, sizeof(tmp));

    return tmp[0];
}

__device__ static inline int __double2int_rd(double x) { return (int)__ocml_floor_f64(x); }
__device__ static inline int __double2int_rn(double x) { return (int)__ocml_rint_f64(x); }
__device__ static inline int __double2int_ru(double x) { return (int)__ocml_ceil_f64(x); }
__device__ static inline int __double2int_rz(double x) { return (int)x; }

__device__ static inline long long int __double2ll_rd(double x) {
  return (long long)__ocml_floor_f64(x);
}
__device__ static inline long long int __double2ll_rn(double x) {
  return (long long)__ocml_rint_f64(x);
}
__device__ static inline long long int __double2ll_ru(double x) {
  return (long long)__ocml_ceil_f64(x);
}
__device__ static inline long long int __double2ll_rz(double x) { return (long long)x; }

__device__ static inline unsigned int __double2uint_rd(double x) {
  return (unsigned int)__ocml_floor_f64(x);
}
__device__ static inline unsigned int __double2uint_rn(double x) {
  return (unsigned int)__ocml_rint_f64(x);
}
__device__ static inline unsigned int __double2uint_ru(double x) {
  return (unsigned int)__ocml_ceil_f64(x);
}
__device__ static inline unsigned int __double2uint_rz(double x) { return (unsigned int)x; }

__device__ static inline unsigned long long int __double2ull_rd(double x) {
  return (unsigned long long int)__ocml_floor_f64(x);
}
__device__ static inline unsigned long long int __double2ull_rn(double x) {
  return (unsigned long long int)__ocml_rint_f64(x);
}
__device__ static inline unsigned long long int __double2ull_ru(double x) {
  return (unsigned long long int)__ocml_ceil_f64(x);
}
__device__ static inline unsigned long long int __double2ull_rz(double x) {
  return (unsigned long long int)x;
}
__device__ static inline long long int __double_as_longlong(double x) {
    static_assert(sizeof(long long) == sizeof(double), "");

    long long tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

/*
__device__ unsigned short __float2half_rn(float x);
__device__ float __half2float(unsigned short);

The above device function are not a valid .
Use
__device__ __half __float2half_rn(float x);
__device__ float __half2float(__half);
from hip_fp16.h

CUDA implements half as unsigned short whereas, HIP doesn't.

*/

__device__ static inline int __float2int_rd(float x) { return (int)__ocml_floor_f32(x); }
__device__ static inline int __float2int_rn(float x) { return (int)__ocml_rint_f32(x); }
__device__ static inline int __float2int_ru(float x) { return (int)__ocml_ceil_f32(x); }
__device__ static inline int __float2int_rz(float x) { return (int)__ocml_trunc_f32(x); }

__device__ static inline long long int __float2ll_rd(float x) {
  return (long long int)__ocml_floor_f32(x);
}
__device__ static inline long long int __float2ll_rn(float x) {
  return (long long int)__ocml_rint_f32(x);
}
__device__ static inline long long int __float2ll_ru(float x) {
  return (long long int)__ocml_ceil_f32(x);
}
__device__ static inline long long int __float2ll_rz(float x) { return (long long int)x; }

__device__ static inline unsigned int __float2uint_rd(float x) {
  return (unsigned int)__ocml_floor_f32(x);
}
__device__ static inline unsigned int __float2uint_rn(float x) {
  return (unsigned int)__ocml_rint_f32(x);
}
__device__ static inline unsigned int __float2uint_ru(float x) {
  return (unsigned int)__ocml_ceil_f32(x);
}
__device__ static inline unsigned int __float2uint_rz(float x) { return (unsigned int)x; }

__device__ static inline unsigned long long int __float2ull_rd(float x) {
  return (unsigned long long int)__ocml_floor_f32(x);
}
__device__ static inline unsigned long long int __float2ull_rn(float x) {
  return (unsigned long long int)__ocml_rint_f32(x);
}
__device__ static inline unsigned long long int __float2ull_ru(float x) {
  return (unsigned long long int)__ocml_ceil_f32(x);
}
__device__ static inline unsigned long long int __float2ull_rz(float x) {
  return (unsigned long long int)x;
}

__device__ static inline int __float_as_int(float x) {
    static_assert(sizeof(int) == sizeof(float), "");

    int tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

__device__ static inline unsigned int __float_as_uint(float x) {
    static_assert(sizeof(unsigned int) == sizeof(float), "");

    unsigned int tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

__device__ static inline double __hiloint2double(int hi, int lo) {
    static_assert(sizeof(double) == sizeof(uint64_t), "");

    uint64_t tmp0 = (static_cast<uint64_t>(hi) << 32ull) | static_cast<uint32_t>(lo);
    double tmp1;
    __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));

    return tmp1;
}

__device__ static inline double __int2double_rn(int x) { return (double)x; }

__device__ static inline float __int2float_rd(int x) {
    return __ocml_cvtrtn_f32_s32(x);
}
__device__ static inline float __int2float_rn(int x) { return (float)x; }
__device__ static inline float __int2float_ru(int x) {
    return __ocml_cvtrtp_f32_s32(x);
}
__device__ static inline float __int2float_rz(int x) {
    return __ocml_cvtrtz_f32_s32(x);
}

__device__ static inline float __int_as_float(int x) {
    static_assert(sizeof(float) == sizeof(int), "");

    float tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

__device__ static inline double __ll2double_rd(long long int x) {
    return __ocml_cvtrtn_f64_s64(x);
}
__device__ static inline double __ll2double_rn(long long int x) { return (double)x; }
__device__ static inline double __ll2double_ru(long long int x) {
    return __ocml_cvtrtp_f64_s64(x);
}
__device__ static inline double __ll2double_rz(long long int x) {
    return __ocml_cvtrtz_f64_s64(x);
}

__device__ static inline float __ll2float_rd(long long int x) {
    return __ocml_cvtrtn_f32_s64(x);
}
__device__ static inline float __ll2float_rn(long long int x) { return (float)x; }
__device__ static inline float __ll2float_ru(long long int x) {
    return __ocml_cvtrtp_f32_s64(x);
}
__device__ static inline float __ll2float_rz(long long int x) {
    return __ocml_cvtrtz_f32_s64(x);
}

__device__ static inline double __longlong_as_double(long long int x) {
    static_assert(sizeof(double) == sizeof(long long), "");

    double tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

__device__ static inline double __uint2double_rn(unsigned int x) { return (double)x; }

__device__ static inline float __uint2float_rd(unsigned int x) {
    return __ocml_cvtrtn_f32_u32(x);
}
__device__ static inline float __uint2float_rn(unsigned int x) { return (float)x; }
__device__ static inline float __uint2float_ru(unsigned int x) {
    return __ocml_cvtrtp_f32_u32(x);
}
__device__ static inline float __uint2float_rz(unsigned int x) {
    return __ocml_cvtrtz_f32_u32(x);
}

__device__ static inline float __uint_as_float(unsigned int x) {
   static_assert(sizeof(float) == sizeof(unsigned int), "");

    float tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));

    return tmp;
}

__device__ static inline double __ull2double_rd(unsigned long long int x) {
    return __ocml_cvtrtn_f64_u64(x);
}
__device__ static inline double __ull2double_rn(unsigned long long int x) { return (double)x; }
__device__ static inline double __ull2double_ru(unsigned long long int x) {
    return __ocml_cvtrtp_f64_u64(x);
}
__device__ static inline double __ull2double_rz(unsigned long long int x) {
    return __ocml_cvtrtz_f64_u64(x);
}

__device__ static inline float __ull2float_rd(unsigned long long int x) {
    return __ocml_cvtrtn_f32_u64(x);
}
__device__ static inline float __ull2float_rn(unsigned long long int x) { return (float)x; }
__device__ static inline float __ull2float_ru(unsigned long long int x) {
    return __ocml_cvtrtp_f32_u64(x);
}
__device__ static inline float __ull2float_rz(unsigned long long int x) {
    return __ocml_cvtrtz_f32_u64(x);
}

#if defined(__clang__) && defined(__HIP__)

// Clock functions
__device__ long long int __clock64();
__device__ long long int __clock();
__device__ long long int clock64();
__device__ long long int clock();
__device__ long long int wall_clock64();
// hip.amdgcn.bc - named sync
__device__ void __named_sync();

#ifdef __HIP_DEVICE_COMPILE__

// Clock function to return GPU core cycle count.
// GPU can change its core clock frequency at runtime. The maximum frequency can be queried
// through hipDeviceAttributeClockRate attribute.
__device__
inline  __attribute((always_inline))
long long int __clock64() {
#if __has_builtin(__builtin_amdgcn_s_memtime)
  // Exists on gfx8, gfx9, gfx10.1, gfx10.2, gfx10.3
  return (long long int) __builtin_amdgcn_s_memtime();
#else
  // Subject to change when better solution available
  return (long long int) __builtin_readcyclecounter();
#endif
}

__device__
inline __attribute((always_inline))
long long int  __clock() { return __clock64(); }

// Clock function to return wall clock count at a constant frequency that can be queried
// through hipDeviceAttributeWallClockRate attribute.
__device__
inline  __attribute__((always_inline))
long long int wall_clock64() {
  return (long long int) __ockl_steadyctr_u64();
}

__device__
inline  __attribute__((always_inline))
long long int clock64() { return __clock64(); }

__device__
inline __attribute__((always_inline))
long long int  clock() { return __clock(); }

// hip.amdgcn.bc - named sync
__device__
inline
void __named_sync() { __builtin_amdgcn_s_barrier(); }

#endif // __HIP_DEVICE_COMPILE__

// warp vote function __all __any __ballot
__device__
inline
int __all(int predicate) {
    return __ockl_wfall_i32(predicate);
}

__device__
inline
int __any(int predicate) {
    return __ockl_wfany_i32(predicate);
}

// XXX from llvm/include/llvm/IR/InstrTypes.h
#define ICMP_NE 33

__device__
inline
unsigned long long int __ballot(int predicate) {
    return __builtin_amdgcn_uicmp(predicate, 0, ICMP_NE);
}

__device__
inline
unsigned long long int __ballot64(int predicate) {
    return __builtin_amdgcn_uicmp(predicate, 0, ICMP_NE);
}

// hip.amdgcn.bc - lanemask
__device__
inline
uint64_t  __lanemask_gt()
{
    uint32_t lane = __ockl_lane_u32();
    if (lane == 63)
      return 0;
    uint64_t ballot = __ballot64(1);
    uint64_t mask = (~((uint64_t)0)) << (lane + 1);
    return mask & ballot;
}

__device__
inline
uint64_t __lanemask_lt()
{
    uint32_t lane = __ockl_lane_u32();
    int64_t ballot = __ballot64(1);
    uint64_t mask = ((uint64_t)1 << lane) - (uint64_t)1;
    return mask & ballot;
}

__device__
inline
uint64_t  __lanemask_eq()
{
    uint32_t lane = __ockl_lane_u32();
    int64_t mask = ((uint64_t)1 << lane);
    return mask;
}


__device__ inline void* __local_to_generic(void* p) { return p; }

#ifdef __HIP_DEVICE_COMPILE__
__device__
inline
void* __get_dynamicgroupbaseptr()
{
    // Get group segment base pointer.
    return (char*)__local_to_generic((void*)__to_local(__builtin_amdgcn_groupstaticsize()));
}
#else
__device__
void* __get_dynamicgroupbaseptr();
#endif // __HIP_DEVICE_COMPILE__

__device__
inline
void *__amdgcn_get_dynamicgroupbaseptr() {
    return __get_dynamicgroupbaseptr();
}

// Memory Fence Functions
__device__
inline
static void __threadfence()
{
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
}

__device__
inline
static void __threadfence_block()
{
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup");
}

__device__
inline
static void __threadfence_system()
{
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "");
}

// abort
__device__
inline
__attribute__((weak))
void abort() {
    return __builtin_trap();
}

// The noinline attribute helps encapsulate the printf expansion,
// which otherwise has a performance impact just by increasing the
// size of the calling function. Additionally, the weak attribute
// allows the function to exist as a global although its definition is
// included in every compilation unit.
#if defined(_WIN32) || defined(_WIN64)
extern "C" __device__ __attribute__((noinline)) __attribute__((weak))
void _wassert(const wchar_t *_msg, const wchar_t *_file, unsigned _line) {
    // FIXME: Need `wchar_t` support to generate assertion message.
    __builtin_trap();
}
#else /* defined(_WIN32) || defined(_WIN64) */
extern "C" __device__ __attribute__((noinline)) __attribute__((weak))
void __assert_fail(const char *assertion,
                   const char *file,
                   unsigned int line,
                   const char *function)
{
  const char fmt[] = "%s:%u: %s: Device-side assertion `%s' failed.\n";

  // strlen is not available as a built-in yet, so we create our own
  // loop in a macro. With a string literal argument, the compiler
  // usually manages to replace the loop with a constant.
  //
  // The macro does not check for null pointer, since all the string
  // arguments are defined to be constant literals when called from
  // the assert() macro.
  //
  // NOTE: The loop below includes the null terminator in the length
  // as required by append_string_n().
#define __hip_get_string_length(LEN, STR)       \
  do {                                          \
    const char *tmp = STR;                      \
    while (*tmp++);                             \
    LEN = tmp - STR;                            \
  } while (0)

  auto msg = __ockl_fprintf_stderr_begin();
  int len = 0;
  __hip_get_string_length(len, fmt);
  msg = __ockl_fprintf_append_string_n(msg, fmt, len, 0);
  __hip_get_string_length(len, file);
  msg = __ockl_fprintf_append_string_n(msg, file, len, 0);
  msg = __ockl_fprintf_append_args(msg, 1, line, 0, 0, 0, 0, 0, 0, 0);
  __hip_get_string_length(len, function);
  msg = __ockl_fprintf_append_string_n(msg, function, len, 0);
  __hip_get_string_length(len, assertion);
  __ockl_fprintf_append_string_n(msg, assertion, len, /* is_last = */ 1);

#undef __hip_get_string_length

  __builtin_trap();
}

extern "C" __device__ __attribute__((noinline)) __attribute__((weak))
void __assertfail()
{
    // ignore all the args for now.
    __builtin_trap();
}
#endif /* defined(_WIN32) || defined(_WIN64) */

__device__ inline static void __work_group_barrier(__cl_mem_fence_flags flags) {
    if (flags) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    } else {
        __builtin_amdgcn_s_barrier();
    }
}

__device__
inline
static void __barrier(int n)
{
  __work_group_barrier((__cl_mem_fence_flags)n);
}

__device__
inline
__attribute__((convergent))
void __syncthreads()
{
  __barrier(__CLK_LOCAL_MEM_FENCE);
}

__device__
inline
__attribute__((convergent))
int __syncthreads_count(int predicate)
{
  return __ockl_wgred_add_i32(!!predicate);
}

__device__
inline
__attribute__((convergent))
int __syncthreads_and(int predicate)
{
  return __ockl_wgred_and_i32(!!predicate);
}

__device__
inline
__attribute__((convergent))
int __syncthreads_or(int predicate)
{
  return __ockl_wgred_or_i32(!!predicate);
}

// hip.amdgcn.bc - device routine
/*
  HW_ID Register bit structure for RDNA2 & RDNA3
  WAVE_ID     4:0     Wave id within the SIMD.
  SIMD_ID     9:8     SIMD_ID within the WGP: [0] = row, [1] = column.
  WGP_ID      13:10   Physical WGP ID.
  SA_ID       16      Shader Array ID
  SE_ID       20:18   Shader Engine the wave is assigned to for gfx11
  SE_ID       19:18   Shader Engine the wave is assigned to for gfx10
  DP_RATE     31:29   Number of double-precision float units per SIMD

  HW_ID Register bit structure for GCN and CDNA
  WAVE_ID     3:0     Wave buffer slot number. 0-9.
  SIMD_ID     5:4     SIMD which the wave is assigned to within the CU.
  PIPE_ID     7:6     Pipeline from which the wave was dispatched.
  CU_ID       11:8    Compute Unit the wave is assigned to.
  SH_ID       12      Shader Array (within an SE) the wave is assigned to.
  SE_ID       15:13   Shader Engine the wave is assigned to for gfx908, gfx90a, gfx940-942
              14:13   Shader Engine the wave is assigned to for Vega.
  TG_ID       19:16   Thread-group ID
  VM_ID       23:20   Virtual Memory ID
  QUEUE_ID    26:24   Queue from which this wave was dispatched.
  STATE_ID    29:27   State ID (graphics only, not compute).
  ME_ID       31:30   Micro-engine ID.

  XCC_ID Register bit structure for gfx940
  XCC_ID      3:0     XCC the wave is assigned to.
 */

#if (defined (__GFX10__) || defined (__GFX11__))
  #define HW_ID               23
#else
  #define HW_ID               4
#endif

#if (defined(__GFX10__) || defined(__GFX11__))
  #define HW_ID_WGP_ID_SIZE   4
  #define HW_ID_WGP_ID_OFFSET 10
#else
  #define HW_ID_CU_ID_SIZE    4
  #define HW_ID_CU_ID_OFFSET  8
#endif

#if (defined(__gfx908__) || defined(__gfx90a__) || \
     defined(__GFX11__))
  #define HW_ID_SE_ID_SIZE    3
#else //4 SEs/XCC for gfx940-942
  #define HW_ID_SE_ID_SIZE    2
#endif
#if (defined(__GFX10__) || defined(__GFX11__))
  #define HW_ID_SE_ID_OFFSET  18
  #define HW_ID_SA_ID_OFFSET  16
  #define HW_ID_SA_ID_SIZE    1
#else
  #define HW_ID_SE_ID_OFFSET  13
#endif

#if (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
  #define XCC_ID                   20
  #define XCC_ID_XCC_ID_SIZE       4
  #define XCC_ID_XCC_ID_OFFSET     0
#endif

#if (!defined(__HIP_NO_IMAGE_SUPPORT) && \
    (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)))
  #define __HIP_NO_IMAGE_SUPPORT   1
#endif

/*
   Encoding of parameter bitmask
   HW_ID        5:0     HW_ID
   OFFSET       10:6    Range: 0..31
   SIZE         15:11   Range: 1..32
 */

#define GETREG_IMMED(SZ,OFF,REG) (((SZ) << 11) | ((OFF) << 6) | (REG))

/*
  __smid returns the wave's assigned Compute Unit and Shader Engine.
  The Compute Unit, CU_ID returned in bits 3:0, and Shader Engine, SE_ID in bits 5:4.
  Note: the results vary over time.
  SZ minus 1 since SIZE is 1-based.
*/
__device__
inline
unsigned __smid(void)
{
    unsigned se_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_SE_ID_SIZE-1, HW_ID_SE_ID_OFFSET, HW_ID));
    #if (defined(__GFX10__) || defined(__GFX11__))
      unsigned wgp_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_WGP_ID_SIZE - 1, HW_ID_WGP_ID_OFFSET, HW_ID));
      unsigned sa_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_SA_ID_SIZE - 1, HW_ID_SA_ID_OFFSET, HW_ID));
    #else
      #if (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
      unsigned xcc_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(XCC_ID_XCC_ID_SIZE - 1, XCC_ID_XCC_ID_OFFSET, XCC_ID));
      #endif
      unsigned cu_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_CU_ID_SIZE - 1, HW_ID_CU_ID_OFFSET, HW_ID));
    #endif
    #if (defined(__GFX10__) || defined(__GFX11__))
      unsigned temp = se_id;
      temp = (temp << HW_ID_SA_ID_SIZE) | sa_id;
      temp = (temp << HW_ID_WGP_ID_SIZE) | wgp_id;
      return temp;
      //TODO : CU Mode impl
    #elif (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
      unsigned temp = xcc_id;
      temp = (temp << HW_ID_SE_ID_SIZE) | se_id;
      temp = (temp << HW_ID_CU_ID_SIZE) | cu_id;
      return temp;
    #else
      return (se_id << HW_ID_CU_ID_SIZE) + cu_id;
    #endif
}

/**
 * Map HIP_DYNAMIC_SHARED to "extern __shared__" for compatibility with old HIP applications
 * To be removed in a future release.
 */
#define HIP_DYNAMIC_SHARED(type, var) extern __shared__ type var[];
#define HIP_DYNAMIC_SHARED_ATTRIBUTE

#endif //defined(__clang__) && defined(__HIP__)


// loop unrolling
static inline __device__ void* __hip_hc_memcpy(void* dst, const void* src, size_t size) {
    auto dstPtr = static_cast<unsigned char*>(dst);
    auto srcPtr = static_cast<const unsigned char*>(src);

    while (size >= 4u) {
        dstPtr[0] = srcPtr[0];
        dstPtr[1] = srcPtr[1];
        dstPtr[2] = srcPtr[2];
        dstPtr[3] = srcPtr[3];

        size -= 4u;
        srcPtr += 4u;
        dstPtr += 4u;
    }
    switch (size) {
        case 3:
            dstPtr[2] = srcPtr[2];
        case 2:
            dstPtr[1] = srcPtr[1];
        case 1:
            dstPtr[0] = srcPtr[0];
    }

    return dst;
}

static inline __device__ void* __hip_hc_memset(void* dst, unsigned char val, size_t size) {
    auto dstPtr = static_cast<unsigned char*>(dst);

    while (size >= 4u) {
        dstPtr[0] = val;
        dstPtr[1] = val;
        dstPtr[2] = val;
        dstPtr[3] = val;

        size -= 4u;
        dstPtr += 4u;
    }
    switch (size) {
        case 3:
            dstPtr[2] = val;
        case 2:
            dstPtr[1] = val;
        case 1:
            dstPtr[0] = val;
    }

    return dst;
}
#ifndef __OPENMP_AMDGCN__
static inline __device__ void* memcpy(void* dst, const void* src, size_t size) {
    return __hip_hc_memcpy(dst, src, size);
}

static inline __device__ void* memset(void* ptr, int val, size_t size) {
    unsigned char val8 = static_cast<unsigned char>(val);
    return __hip_hc_memset(ptr, val8, size);
}
#endif // !__OPENMP_AMDGCN__

#endif
/*
Copyright (c) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_WARP_FUNCTIONS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_WARP_FUNCTIONS_H

__device__ static inline unsigned __hip_ds_bpermute(int index, unsigned src) {
    union { int i; unsigned u; float f; } tmp; tmp.u = src;
    tmp.i = __builtin_amdgcn_ds_bpermute(index, tmp.i);
    return tmp.u;
}

__device__ static inline float __hip_ds_bpermutef(int index, float src) {
    union { int i; unsigned u; float f; } tmp; tmp.f = src;
    tmp.i = __builtin_amdgcn_ds_bpermute(index, tmp.i);
    return tmp.f;
}

__device__ static inline unsigned __hip_ds_permute(int index, unsigned src) {
    union { int i; unsigned u; float f; } tmp; tmp.u = src;
    tmp.i = __builtin_amdgcn_ds_permute(index, tmp.i);
    return tmp.u;
}

__device__ static inline float __hip_ds_permutef(int index, float src) {
    union { int i; unsigned u; float f; } tmp; tmp.f = src;
    tmp.i = __builtin_amdgcn_ds_permute(index, tmp.i);
    return tmp.f;
}

#define __hip_ds_swizzle(src, pattern)  __hip_ds_swizzle_N<(pattern)>((src))
#define __hip_ds_swizzlef(src, pattern) __hip_ds_swizzlef_N<(pattern)>((src))

template <int pattern>
__device__ static inline unsigned __hip_ds_swizzle_N(unsigned int src) {
    union { int i; unsigned u; float f; } tmp; tmp.u = src;
    tmp.i = __builtin_amdgcn_ds_swizzle(tmp.i, pattern);
    return tmp.u;
}

template <int pattern>
__device__ static inline float __hip_ds_swizzlef_N(float src) {
    union { int i; unsigned u; float f; } tmp; tmp.f = src;
    tmp.i = __builtin_amdgcn_ds_swizzle(tmp.i, pattern);
    return tmp.f;
}

#define __hip_move_dpp(src, dpp_ctrl, row_mask, bank_mask, bound_ctrl) \
  __hip_move_dpp_N<(dpp_ctrl), (row_mask), (bank_mask), (bound_ctrl)>((src))

template <int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl>
__device__ static inline int __hip_move_dpp_N(int src) {
    return __builtin_amdgcn_mov_dpp(src, dpp_ctrl, row_mask, bank_mask,
                                    bound_ctrl);
}

static constexpr int warpSize = __AMDGCN_WAVEFRONT_SIZE;

__device__
inline
int __shfl(int var, int src_lane, int width = warpSize) {
    int self = __lane_id();
    int index = (src_lane & (width - 1)) + (self & ~(width-1));
    return __builtin_amdgcn_ds_bpermute(index<<2, var);
}
__device__
inline
unsigned int __shfl(unsigned int var, int src_lane, int width = warpSize) {
     union { int i; unsigned u; float f; } tmp; tmp.u = var;
    tmp.i = __shfl(tmp.i, src_lane, width);
    return tmp.u;
}
__device__
inline
float __shfl(float var, int src_lane, int width = warpSize) {
    union { int i; unsigned u; float f; } tmp; tmp.f = var;
    tmp.i = __shfl(tmp.i, src_lane, width);
    return tmp.f;
}
__device__
inline
double __shfl(double var, int src_lane, int width = warpSize) {
    static_assert(sizeof(double) == 2 * sizeof(int), "");
    static_assert(sizeof(double) == sizeof(uint64_t), "");

    int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl(tmp[0], src_lane, width);
    tmp[1] = __shfl(tmp[1], src_lane, width);

    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    double tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
}
__device__
inline
long __shfl(long var, int src_lane, int width = warpSize)
{
    #ifndef _MSC_VER
    static_assert(sizeof(long) == 2 * sizeof(int), "");
    static_assert(sizeof(long) == sizeof(uint64_t), "");

    int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl(tmp[0], src_lane, width);
    tmp[1] = __shfl(tmp[1], src_lane, width);

    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    long tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
    #else
    static_assert(sizeof(long) == sizeof(int), "");
    return static_cast<long>(__shfl(static_cast<int>(var), src_lane, width));
    #endif
}
__device__
inline
unsigned long __shfl(unsigned long var, int src_lane, int width = warpSize) {
    #ifndef _MSC_VER
    static_assert(sizeof(unsigned long) == 2 * sizeof(unsigned int), "");
    static_assert(sizeof(unsigned long) == sizeof(uint64_t), "");

    unsigned int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl(tmp[0], src_lane, width);
    tmp[1] = __shfl(tmp[1], src_lane, width);

    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    unsigned long tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
    #else
    static_assert(sizeof(unsigned long) == sizeof(unsigned int), "");
    return static_cast<unsigned long>(__shfl(static_cast<unsigned int>(var), src_lane, width));
    #endif
}
__device__
inline
long long __shfl(long long var, int src_lane, int width = warpSize)
{
    static_assert(sizeof(long long) == 2 * sizeof(int), "");
    static_assert(sizeof(long long) == sizeof(uint64_t), "");

    int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl(tmp[0], src_lane, width);
    tmp[1] = __shfl(tmp[1], src_lane, width);

    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    long long tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
}
__device__
inline
unsigned long long __shfl(unsigned long long var, int src_lane, int width = warpSize) {
    static_assert(sizeof(unsigned long long) == 2 * sizeof(unsigned int), "");
    static_assert(sizeof(unsigned long long) == sizeof(uint64_t), "");

    unsigned int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl(tmp[0], src_lane, width);
    tmp[1] = __shfl(tmp[1], src_lane, width);

    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    unsigned long long tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
}

__device__
inline
int __shfl_up(int var, unsigned int lane_delta, int width = warpSize) {
    int self = __lane_id();
    int index = self - lane_delta;
    index = (index < (self & ~(width-1)))?self:index;
    return __builtin_amdgcn_ds_bpermute(index<<2, var);
}
__device__
inline
unsigned int __shfl_up(unsigned int var, unsigned int lane_delta, int width = warpSize) {
    union { int i; unsigned u; float f; } tmp; tmp.u = var;
    tmp.i = __shfl_up(tmp.i, lane_delta, width);
    return tmp.u;
}
__device__
inline
float __shfl_up(float var, unsigned int lane_delta, int width = warpSize) {
    union { int i; unsigned u; float f; } tmp; tmp.f = var;
    tmp.i = __shfl_up(tmp.i, lane_delta, width);
    return tmp.f;
}
__device__
inline
double __shfl_up(double var, unsigned int lane_delta, int width = warpSize) {
    static_assert(sizeof(double) == 2 * sizeof(int), "");
    static_assert(sizeof(double) == sizeof(uint64_t), "");

    int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl_up(tmp[0], lane_delta, width);
    tmp[1] = __shfl_up(tmp[1], lane_delta, width);

    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    double tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
}
__device__
inline
long __shfl_up(long var, unsigned int lane_delta, int width = warpSize)
{
    #ifndef _MSC_VER
    static_assert(sizeof(long) == 2 * sizeof(int), "");
    static_assert(sizeof(long) == sizeof(uint64_t), "");

    int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl_up(tmp[0], lane_delta, width);
    tmp[1] = __shfl_up(tmp[1], lane_delta, width);

    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    long tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
    #else
    static_assert(sizeof(long) == sizeof(int), "");
    return static_cast<long>(__shfl_up(static_cast<int>(var), lane_delta, width));
    #endif
}

__device__
inline
unsigned long __shfl_up(unsigned long var, unsigned int lane_delta, int width = warpSize)
{
    #ifndef _MSC_VER
    static_assert(sizeof(unsigned long) == 2 * sizeof(unsigned int), "");
    static_assert(sizeof(unsigned long) == sizeof(uint64_t), "");

    unsigned int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl_up(tmp[0], lane_delta, width);
    tmp[1] = __shfl_up(tmp[1], lane_delta, width);

    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    unsigned long tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
    #else
    static_assert(sizeof(unsigned long) == sizeof(unsigned int), "");
    return static_cast<unsigned long>(__shfl_up(static_cast<unsigned int>(var), lane_delta, width));
    #endif
}

__device__
inline
long long __shfl_up(long long var, unsigned int lane_delta, int width = warpSize)
{
    static_assert(sizeof(long long) == 2 * sizeof(int), "");
    static_assert(sizeof(long long) == sizeof(uint64_t), "");
    int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl_up(tmp[0], lane_delta, width);
    tmp[1] = __shfl_up(tmp[1], lane_delta, width);
    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    long long tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
}

__device__
inline
unsigned long long __shfl_up(unsigned long long var, unsigned int lane_delta, int width = warpSize)
{
    static_assert(sizeof(unsigned long long) == 2 * sizeof(unsigned int), "");
    static_assert(sizeof(unsigned long long) == sizeof(uint64_t), "");
    unsigned int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl_up(tmp[0], lane_delta, width);
    tmp[1] = __shfl_up(tmp[1], lane_delta, width);
    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    unsigned long long tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
}

__device__
inline
int __shfl_down(int var, unsigned int lane_delta, int width = warpSize) {
    int self = __lane_id();
    int index = self + lane_delta;
    index = (int)((self&(width-1))+lane_delta) >= width?self:index;
    return __builtin_amdgcn_ds_bpermute(index<<2, var);
}
__device__
inline
unsigned int __shfl_down(unsigned int var, unsigned int lane_delta, int width = warpSize) {
    union { int i; unsigned u; float f; } tmp; tmp.u = var;
    tmp.i = __shfl_down(tmp.i, lane_delta, width);
    return tmp.u;
}
__device__
inline
float __shfl_down(float var, unsigned int lane_delta, int width = warpSize) {
    union { int i; unsigned u; float f; } tmp; tmp.f = var;
    tmp.i = __shfl_down(tmp.i, lane_delta, width);
    return tmp.f;
}
__device__
inline
double __shfl_down(double var, unsigned int lane_delta, int width = warpSize) {
    static_assert(sizeof(double) == 2 * sizeof(int), "");
    static_assert(sizeof(double) == sizeof(uint64_t), "");

    int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl_down(tmp[0], lane_delta, width);
    tmp[1] = __shfl_down(tmp[1], lane_delta, width);

    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    double tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
}
__device__
inline
long __shfl_down(long var, unsigned int lane_delta, int width = warpSize)
{
    #ifndef _MSC_VER
    static_assert(sizeof(long) == 2 * sizeof(int), "");
    static_assert(sizeof(long) == sizeof(uint64_t), "");

    int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl_down(tmp[0], lane_delta, width);
    tmp[1] = __shfl_down(tmp[1], lane_delta, width);

    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    long tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
    #else
    static_assert(sizeof(long) == sizeof(int), "");
    return static_cast<long>(__shfl_down(static_cast<int>(var), lane_delta, width));
    #endif
}
__device__
inline
unsigned long __shfl_down(unsigned long var, unsigned int lane_delta, int width = warpSize)
{
    #ifndef _MSC_VER
    static_assert(sizeof(unsigned long) == 2 * sizeof(unsigned int), "");
    static_assert(sizeof(unsigned long) == sizeof(uint64_t), "");

    unsigned int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl_down(tmp[0], lane_delta, width);
    tmp[1] = __shfl_down(tmp[1], lane_delta, width);

    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    unsigned long tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
    #else
    static_assert(sizeof(unsigned long) == sizeof(unsigned int), "");
    return static_cast<unsigned long>(__shfl_down(static_cast<unsigned int>(var), lane_delta, width));
    #endif
}
__device__
inline
long long __shfl_down(long long var, unsigned int lane_delta, int width = warpSize)
{
    static_assert(sizeof(long long) == 2 * sizeof(int), "");
    static_assert(sizeof(long long) == sizeof(uint64_t), "");
    int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl_down(tmp[0], lane_delta, width);
    tmp[1] = __shfl_down(tmp[1], lane_delta, width);
    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    long long tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
}
__device__
inline
unsigned long long __shfl_down(unsigned long long var, unsigned int lane_delta, int width = warpSize)
{
    static_assert(sizeof(unsigned long long) == 2 * sizeof(unsigned int), "");
    static_assert(sizeof(unsigned long long) == sizeof(uint64_t), "");
    unsigned int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl_down(tmp[0], lane_delta, width);
    tmp[1] = __shfl_down(tmp[1], lane_delta, width);
    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    unsigned long long tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
}

__device__
inline
int __shfl_xor(int var, int lane_mask, int width = warpSize) {
    int self = __lane_id();
    int index = self^lane_mask;
    index = index >= ((self+width)&~(width-1))?self:index;
    return __builtin_amdgcn_ds_bpermute(index<<2, var);
}
__device__
inline
unsigned int __shfl_xor(unsigned int var, int lane_mask, int width = warpSize) {
    union { int i; unsigned u; float f; } tmp; tmp.u = var;
    tmp.i = __shfl_xor(tmp.i, lane_mask, width);
    return tmp.u;
}
__device__
inline
float __shfl_xor(float var, int lane_mask, int width = warpSize) {
    union { int i; unsigned u; float f; } tmp; tmp.f = var;
    tmp.i = __shfl_xor(tmp.i, lane_mask, width);
    return tmp.f;
}
__device__
inline
double __shfl_xor(double var, int lane_mask, int width = warpSize) {
    static_assert(sizeof(double) == 2 * sizeof(int), "");
    static_assert(sizeof(double) == sizeof(uint64_t), "");

    int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl_xor(tmp[0], lane_mask, width);
    tmp[1] = __shfl_xor(tmp[1], lane_mask, width);

    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    double tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
}
__device__
inline
long __shfl_xor(long var, int lane_mask, int width = warpSize)
{
    #ifndef _MSC_VER
    static_assert(sizeof(long) == 2 * sizeof(int), "");
    static_assert(sizeof(long) == sizeof(uint64_t), "");

    int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl_xor(tmp[0], lane_mask, width);
    tmp[1] = __shfl_xor(tmp[1], lane_mask, width);

    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    long tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
    #else
    static_assert(sizeof(long) == sizeof(int), "");
    return static_cast<long>(__shfl_xor(static_cast<int>(var), lane_mask, width));
    #endif
}
__device__
inline
unsigned long __shfl_xor(unsigned long var, int lane_mask, int width = warpSize)
{
    #ifndef _MSC_VER
    static_assert(sizeof(unsigned long) == 2 * sizeof(unsigned int), "");
    static_assert(sizeof(unsigned long) == sizeof(uint64_t), "");

    unsigned int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl_xor(tmp[0], lane_mask, width);
    tmp[1] = __shfl_xor(tmp[1], lane_mask, width);

    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    unsigned long tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
    #else
    static_assert(sizeof(unsigned long) == sizeof(unsigned int), "");
    return static_cast<unsigned long>(__shfl_xor(static_cast<unsigned int>(var), lane_mask, width));
    #endif
}
__device__
inline
long long __shfl_xor(long long var, int lane_mask, int width = warpSize)
{
    static_assert(sizeof(long long) == 2 * sizeof(int), "");
    static_assert(sizeof(long long) == sizeof(uint64_t), "");
    int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl_xor(tmp[0], lane_mask, width);
    tmp[1] = __shfl_xor(tmp[1], lane_mask, width);
    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    long long tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
}
__device__
inline
unsigned long long __shfl_xor(unsigned long long var, int lane_mask, int width = warpSize)
{
    static_assert(sizeof(unsigned long long) == 2 * sizeof(unsigned int), "");
    static_assert(sizeof(unsigned long long) == sizeof(uint64_t), "");
    unsigned int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl_xor(tmp[0], lane_mask, width);
    tmp[1] = __shfl_xor(tmp[1], lane_mask, width);
    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    unsigned long long tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
}

#endif
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 *  @file  amd_detail/hip_cooperative_groups_helper.h
 *
 *  @brief Device side implementation of cooperative group feature.
 *
 *  Defines helper constructs and APIs which aid the types and device API
 *  wrappers defined within `amd_detail/hip_cooperative_groups.h`.
 */
#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_HELPER_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_HELPER_H

#if __cplusplus
#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/amd_hip_runtime.h> // threadId, blockId
#include <hip/amd_detail/amd_device_functions.h>
#endif
#if !defined(__align__)
#define __align__(x) __attribute__((aligned(x)))
#endif

#if !defined(__CG_QUALIFIER__)
#define __CG_QUALIFIER__ __device__ __forceinline__
#endif

#if !defined(__CG_STATIC_QUALIFIER__)
#define __CG_STATIC_QUALIFIER__ __device__ static __forceinline__
#endif

#if !defined(_CG_STATIC_CONST_DECL_)
#define _CG_STATIC_CONST_DECL_ static constexpr
#endif

#if __AMDGCN_WAVEFRONT_SIZE == 32
using lane_mask = unsigned int;
#else
using lane_mask = unsigned long long int;
#endif

namespace cooperative_groups {

/* Global scope */
template <unsigned int size>
using is_power_of_2 = std::integral_constant<bool, (size & (size - 1)) == 0>;

template <unsigned int size>
using is_valid_wavefront = std::integral_constant<bool, (size <= __AMDGCN_WAVEFRONT_SIZE)>;

template <unsigned int size>
using is_valid_tile_size =
    std::integral_constant<bool, is_power_of_2<size>::value && is_valid_wavefront<size>::value>;

template <typename T>
using is_valid_type =
    std::integral_constant<bool, std::is_integral<T>::value || std::is_floating_point<T>::value>;

namespace internal {

/**
* @brief Enums representing different cooperative group types
* @note  This enum is only applicable on Linux.
*
 */
typedef enum {
  cg_invalid,
  cg_multi_grid,
  cg_grid,
  cg_workgroup,
  cg_tiled_group,
  cg_coalesced_group
} group_type;
/**
 *  @ingroup CooperativeG
 *  @{
 *  This section describes the cooperative groups functions of HIP runtime API.
 *  
 *  The cooperative groups provides flexible thread parallel programming algorithms, threads
 *  cooperate and share data to perform collective computations.
 *
 *  @note  Cooperative groups feature is implemented on Linux, under developement
 *  on Windows.
 *
 */
/**
 *
 * @brief  Functionalities related to multi-grid cooperative group type
 * @note  The following cooperative groups functions are only applicable on Linux.
 *
 */
namespace multi_grid {

__CG_STATIC_QUALIFIER__ uint32_t num_grids() {
  return static_cast<uint32_t>(__ockl_multi_grid_num_grids()); }

__CG_STATIC_QUALIFIER__ uint32_t grid_rank() {
  return static_cast<uint32_t>(__ockl_multi_grid_grid_rank()); }

__CG_STATIC_QUALIFIER__ uint32_t size() { return static_cast<uint32_t>(__ockl_multi_grid_size()); }

__CG_STATIC_QUALIFIER__ uint32_t thread_rank() {
  return static_cast<uint32_t>(__ockl_multi_grid_thread_rank()); }

__CG_STATIC_QUALIFIER__ bool is_valid() { return static_cast<bool>(__ockl_multi_grid_is_valid()); }

__CG_STATIC_QUALIFIER__ void sync() { __ockl_multi_grid_sync(); }

}  // namespace multi_grid

/**
 *  @brief Functionalities related to grid cooperative group type
 *  @note  The following cooperative groups functions are only applicable on Linux.
 */
namespace grid {

__CG_STATIC_QUALIFIER__ uint32_t size() {
  return static_cast<uint32_t>((blockDim.z * gridDim.z) * (blockDim.y * gridDim.y) *
                    (blockDim.x * gridDim.x));
}

__CG_STATIC_QUALIFIER__ uint32_t thread_rank() {
  // Compute global id of the workgroup to which the current thread belongs to
  uint32_t blkIdx = static_cast<uint32_t>((blockIdx.z * gridDim.y * gridDim.x) +
                               (blockIdx.y * gridDim.x) + (blockIdx.x));

  // Compute total number of threads being passed to reach current workgroup
  // within grid
  uint32_t num_threads_till_current_workgroup =
      static_cast<uint32_t>(blkIdx * (blockDim.x * blockDim.y * blockDim.z));

  // Compute thread local rank within current workgroup
  uint32_t local_thread_rank = static_cast<uint32_t>((threadIdx.z * blockDim.y * blockDim.x) +
                                          (threadIdx.y * blockDim.x) + (threadIdx.x));

  return (num_threads_till_current_workgroup + local_thread_rank);
}

__CG_STATIC_QUALIFIER__ bool is_valid() { return static_cast<bool>(__ockl_grid_is_valid()); }

__CG_STATIC_QUALIFIER__ void sync() { __ockl_grid_sync(); }

}  // namespace grid

/**
 *  @brief Functionalities related to `workgroup` (thread_block in CUDA terminology)
 *  cooperative group type
 *  @note  The following cooperative groups functions are only applicable on Linux.
 */
namespace workgroup {

__CG_STATIC_QUALIFIER__ dim3 group_index() {
  return (dim3(static_cast<uint32_t>(blockIdx.x), static_cast<uint32_t>(blockIdx.y),
               static_cast<uint32_t>(blockIdx.z)));
}

__CG_STATIC_QUALIFIER__ dim3 thread_index() {
  return (dim3(static_cast<uint32_t>(threadIdx.x), static_cast<uint32_t>(threadIdx.y),
               static_cast<uint32_t>(threadIdx.z)));
}

__CG_STATIC_QUALIFIER__ uint32_t size() {
  return (static_cast<uint32_t>(blockDim.x * blockDim.y * blockDim.z));
}

__CG_STATIC_QUALIFIER__ uint32_t thread_rank() {
  return (static_cast<uint32_t>((threadIdx.z * blockDim.y * blockDim.x) +
                     (threadIdx.y * blockDim.x) + (threadIdx.x)));
}

__CG_STATIC_QUALIFIER__ bool is_valid() {
  return true;
}

__CG_STATIC_QUALIFIER__ void sync() { __syncthreads(); }

__CG_STATIC_QUALIFIER__ dim3 block_dim() {
  return (dim3(static_cast<uint32_t>(blockDim.x), static_cast<uint32_t>(blockDim.y),
          static_cast<uint32_t>(blockDim.z)));
}

}  // namespace workgroup

namespace tiled_group {

// enforce ordering for memory intructions
__CG_STATIC_QUALIFIER__ void sync() { __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent"); }

}  // namespace tiled_group

namespace coalesced_group {

// enforce ordering for memory intructions
__CG_STATIC_QUALIFIER__ void sync() { __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent"); }

// Masked bit count
//
// For each thread, this function returns the number of active threads which
// have i-th bit of x set and come before the current thread.
__CG_STATIC_QUALIFIER__ unsigned int masked_bit_count(lane_mask x, unsigned int add = 0) {
  unsigned int counter=0;
    #if __AMDGCN_WAVEFRONT_SIZE == 32
      counter = __builtin_amdgcn_mbcnt_lo(x, add);
    #else
      counter = __builtin_amdgcn_mbcnt_lo(static_cast<lane_mask>(x), add);
      counter = __builtin_amdgcn_mbcnt_hi(static_cast<lane_mask>(x >> 32), counter);
    #endif

    return counter;
}

}  // namespace coalesced_group


}  // namespace internal

}  // namespace cooperative_groups
/**
*  @}
*/

#endif  // __cplusplus
#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_HELPER_H
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 *  @file  amd_detail/hip_cooperative_groups.h
 *
 *  @brief Device side implementation of `Cooperative Group` feature.
 *
 *  Defines new types and device API wrappers related to `Cooperative Group`
 *  feature, which the programmer can directly use in his kernel(s) in order to
 *  make use of this feature.
 */
#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_H

#if __cplusplus
#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/hip_cooperative_groups_helper.h>
#endif

#define __hip_abort()                                                                              \
  { abort(); }
#if defined(NDEBUG)
#define __hip_assert(COND)
#else
#define __hip_assert(COND)                                                                         \
  {                                                                                                \
    if (!COND) {                                                                                   \
      __hip_abort();                                                                               \
    }                                                                                              \
  }
#endif

namespace cooperative_groups {

/** @brief The base type of all cooperative group types
 *
 *  \details Holds the key properties of a constructed cooperative group types
 *           object, like the group type, its size, etc
 *
 *  @note  Cooperative groups feature is implemented on Linux, under developement
 *  on Windows.
 */
class thread_group {
 protected:
  uint32_t _type;  // thread_group type
  uint32_t _size;  // total number of threads in the tread_group
  uint64_t _mask;  // Lanemask for coalesced and tiled partitioned group types,
                   // LSB represents lane 0, and MSB represents lane 63

  // Construct a thread group, and set thread group type and other essential
  // thread group properties. This generic thread group is directly constructed
  // only when the group is supposed to contain only the calling the thread
  // (throurh the API - `this_thread()`), and in all other cases, this thread
  // group object is a sub-object of some other derived thread group object
  __CG_QUALIFIER__ thread_group(internal::group_type type, uint32_t size = static_cast<uint64_t>(0),
                                uint64_t mask = static_cast<uint64_t>(0)) {
    _type = type;
    _size = size;
    _mask = mask;
  }

  struct _tiled_info {
    bool is_tiled;
    unsigned int size;
    unsigned int meta_group_rank;
    unsigned int meta_group_size;
  };

  struct _coalesced_info {
    lane_mask member_mask;
    unsigned int size;
    struct _tiled_info tiled_info;
  } coalesced_info;

  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent,
                                                       unsigned int tile_size);
  friend class thread_block;

 public:
  // Total number of threads in the thread group, and this serves the purpose
  // for all derived cooperative group types since their `size` is directly
  // saved during the construction
  __CG_QUALIFIER__ uint32_t size() const { return _size; }
  __CG_QUALIFIER__ unsigned int cg_type() const { return _type; }
  // Rank of the calling thread within [0, size())
  __CG_QUALIFIER__ uint32_t thread_rank() const;
  // Is this cooperative group type valid?
  __CG_QUALIFIER__ bool is_valid() const;
  // synchronize the threads in the thread group
  __CG_QUALIFIER__ void sync() const;
};
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup CooperativeG Cooperative Groups
 *  @ingroup API
 *  @{
 *  This section describes the cooperative groups functions of HIP runtime API.
 *  
 *  The cooperative groups provides flexible thread parallel programming algorithms, threads
 *  cooperate and share data to perform collective computations.
 *
 *  @note  Cooperative groups feature is implemented on Linux, under developement
 *  on Windows.
 *
 */
/** \brief The multi-grid cooperative group type
 *
 *  \details Represents an inter-device cooperative group type where the
 *           participating threads within the group spans across multple
 *           devices, running the (same) kernel on these devices
 * @note  The multi-grid cooperative group type is implemented on Linux, under developement
 *  on Windows.
 */
class multi_grid_group : public thread_group {
  // Only these friend functions are allowed to construct an object of this class
  // and access its resources
  friend __CG_QUALIFIER__ multi_grid_group this_multi_grid();

 protected:
  // Construct mutli-grid thread group (through the API this_multi_grid())
  explicit __CG_QUALIFIER__ multi_grid_group(uint32_t size)
      : thread_group(internal::cg_multi_grid, size) {}

 public:
  // Number of invocations participating in this multi-grid group. In other
  // words, the number of GPUs
  __CG_QUALIFIER__ uint32_t num_grids() { return internal::multi_grid::num_grids(); }
  // Rank of this invocation. In other words, an ID number within the range
  // [0, num_grids()) of the GPU, this kernel is running on
  __CG_QUALIFIER__ uint32_t grid_rank() { return internal::multi_grid::grid_rank(); }
  __CG_QUALIFIER__ uint32_t thread_rank() const { return internal::multi_grid::thread_rank(); }
  __CG_QUALIFIER__ bool is_valid() const { return internal::multi_grid::is_valid(); }
  __CG_QUALIFIER__ void sync() const { internal::multi_grid::sync(); }
};

/** @brief User exposed API interface to construct multi-grid cooperative
 *         group type object - `multi_grid_group`
 *
 *  \details User is not allowed to directly construct an object of type
 *           `multi_grid_group`. Instead, he should construct it through this
 *           API function
 *  @note  This multi-grid cooperative API type is implemented on Linux, under developement
 *  on Windows.
 */
__CG_QUALIFIER__ multi_grid_group this_multi_grid() {
  return multi_grid_group(internal::multi_grid::size());
}

/** @brief The grid cooperative group type
 *
 *  \details Represents an inter-workgroup cooperative group type where the
 *           participating threads within the group spans across multiple
 *           workgroups running the (same) kernel on the same device
 *  @note  This is implemented on Linux, under developement
 *  on Windows.
 */
class grid_group : public thread_group {
  // Only these friend functions are allowed to construct an object of this class
  // and access its resources
  friend __CG_QUALIFIER__ grid_group this_grid();

 protected:
  // Construct grid thread group (through the API this_grid())
  explicit __CG_QUALIFIER__ grid_group(uint32_t size) : thread_group(internal::cg_grid, size) {}

 public:
  __CG_QUALIFIER__ uint32_t thread_rank() const { return internal::grid::thread_rank(); }
  __CG_QUALIFIER__ bool is_valid() const { return internal::grid::is_valid(); }
  __CG_QUALIFIER__ void sync() const { internal::grid::sync(); }
};

/** @brief User exposed API interface to construct grid cooperative group type
 *         object - `grid_group`
 *
 *  \details User is not allowed to directly construct an object of type
 *           `multi_grid_group`. Instead, he should construct it through this
 *           API function
 *  @note  This function is implemented on Linux, under developement
 *  on Windows.
 */
__CG_QUALIFIER__ grid_group this_grid() { return grid_group(internal::grid::size()); }

/** @brief   The workgroup (thread-block in CUDA terminology) cooperative group
 *           type
 *
 *  \details Represents an intra-workgroup cooperative group type where the
 *           participating threads within the group are exactly the same threads
 *           which are participated in the currently executing `workgroup`
 *  @note  This is implemented on Linux, under developement
 *  on Windows.
 */
class thread_block : public thread_group {
  // Only these friend functions are allowed to construct an object of thi
  // class and access its resources
  friend __CG_QUALIFIER__ thread_block this_thread_block();
  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent,
                                                       unsigned int tile_size);
  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_block& parent,
                                                       unsigned int tile_size);
 protected:
  // Construct a workgroup thread group (through the API this_thread_block())
  explicit __CG_QUALIFIER__ thread_block(uint32_t size)
      : thread_group(internal::cg_workgroup, size) {}

  __CG_QUALIFIER__ thread_group new_tiled_group(unsigned int tile_size) const {
    const bool pow2 = ((tile_size & (tile_size - 1)) == 0);
    // Invalid tile size, assert
    if (!tile_size || (tile_size > __AMDGCN_WAVEFRONT_SIZE) || !pow2) {
      __hip_assert(false && "invalid tile size")
    }

    thread_group tiledGroup = thread_group(internal::cg_tiled_group, tile_size);
    tiledGroup.coalesced_info.tiled_info.size = tile_size;
    tiledGroup.coalesced_info.tiled_info.is_tiled = true;
    tiledGroup.coalesced_info.tiled_info.meta_group_rank = thread_rank() / tile_size;
    tiledGroup.coalesced_info.tiled_info.meta_group_size = (size() + tile_size - 1) / tile_size;
    return tiledGroup;
  }

 public:
  // 3-dimensional block index within the grid
  __CG_STATIC_QUALIFIER__ dim3 group_index() { return internal::workgroup::group_index(); }
  // 3-dimensional thread index within the block
  __CG_STATIC_QUALIFIER__ dim3 thread_index() { return internal::workgroup::thread_index(); }
  __CG_STATIC_QUALIFIER__ uint32_t thread_rank() { return internal::workgroup::thread_rank(); }
  __CG_STATIC_QUALIFIER__ uint32_t size() { return internal::workgroup::size(); }
  __CG_STATIC_QUALIFIER__ bool is_valid() { return internal::workgroup::is_valid(); }
  __CG_STATIC_QUALIFIER__ void sync() { internal::workgroup::sync(); }
  __CG_QUALIFIER__ dim3 group_dim() { return internal::workgroup::block_dim(); }
};

/** \brief   User exposed API interface to construct workgroup cooperative
 *           group type object - `thread_block`.
 *
 *  \details User is not allowed to directly construct an object of type
 *           `thread_block`. Instead, he should construct it through this API
 *           function.
 *  @note  This function is implemented on Linux, under developement
 *  on Windows.
 */
__CG_QUALIFIER__ thread_block this_thread_block() {
  return thread_block(internal::workgroup::size());
}

/** \brief   The tiled_group cooperative group type
 *
 *  \details Represents one tiled thread group in a wavefront.
 *           This group type also supports sub-wave level intrinsics.
 *  @note  This is implemented on Linux, under developement
 *  on Windows.
 */

class tiled_group : public thread_group {
 private:
  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent,
                                                       unsigned int tile_size);
  friend __CG_QUALIFIER__ tiled_group tiled_partition(const tiled_group& parent,
                                                      unsigned int tile_size);

  __CG_QUALIFIER__ tiled_group new_tiled_group(unsigned int tile_size) const {
    const bool pow2 = ((tile_size & (tile_size - 1)) == 0);

    if (!tile_size || (tile_size > __AMDGCN_WAVEFRONT_SIZE) || !pow2) {
      __hip_assert(false && "invalid tile size")
    }

    if (size() <= tile_size) {
      return *this;
    }

    tiled_group tiledGroup = tiled_group(tile_size);
    tiledGroup.coalesced_info.tiled_info.is_tiled = true;
    return tiledGroup;
  }

 protected:
  explicit __CG_QUALIFIER__ tiled_group(unsigned int tileSize)
      : thread_group(internal::cg_tiled_group, tileSize) {
    coalesced_info.tiled_info.size = tileSize;
    coalesced_info.tiled_info.is_tiled = true;
  }

 public:
  __CG_QUALIFIER__ unsigned int size() const { return (coalesced_info.tiled_info.size); }

  __CG_QUALIFIER__ unsigned int thread_rank() const {
    return (internal::workgroup::thread_rank() & (coalesced_info.tiled_info.size - 1));
  }

  __CG_QUALIFIER__ void sync() const {
    internal::tiled_group::sync();
  }
};

/** \brief   The coalesced_group cooperative group type
 *
 *  \details Represents a active thread group in a wavefront.
 *           This group type also supports sub-wave level intrinsics.
 *  @note  This is implemented on Linux, under developement
 *  on Windows.
 */
class coalesced_group : public thread_group {
 private:
  friend __CG_QUALIFIER__ coalesced_group coalesced_threads();
  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent, unsigned int tile_size);
  friend __CG_QUALIFIER__ coalesced_group tiled_partition(const coalesced_group& parent, unsigned int tile_size);

  __CG_QUALIFIER__ coalesced_group new_tiled_group(unsigned int tile_size) const {
    const bool pow2 = ((tile_size & (tile_size - 1)) == 0);

    if (!tile_size || (tile_size > size()) || !pow2) {
      return coalesced_group(0);
    }

    // If a tiled group is passed to be partitioned further into a coalesced_group.
    // prepare a mask for further partitioning it so that it stays coalesced.
    if (coalesced_info.tiled_info.is_tiled) {
      unsigned int base_offset = (thread_rank() & (~(tile_size - 1)));
      unsigned int masklength = min(static_cast<unsigned int>(size()) - base_offset, tile_size);
      lane_mask member_mask = static_cast<lane_mask>(-1) >> (__AMDGCN_WAVEFRONT_SIZE - masklength);

      member_mask <<= (__lane_id() & ~(tile_size - 1));
      coalesced_group coalesced_tile = coalesced_group(member_mask);
      coalesced_tile.coalesced_info.tiled_info.is_tiled = true;
      coalesced_tile.coalesced_info.tiled_info.meta_group_rank = thread_rank() / tile_size;
      coalesced_tile.coalesced_info.tiled_info.meta_group_size = size() / tile_size;
      return coalesced_tile;
    }
    // Here the parent coalesced_group is not partitioned.
    else {
      lane_mask member_mask = 0;
      unsigned int tile_rank = 0;
      int lanes_to_skip = ((thread_rank()) / tile_size) * tile_size;

      for (unsigned int i = 0; i < __AMDGCN_WAVEFRONT_SIZE; i++) {
        lane_mask active = coalesced_info.member_mask & (1 << i);
        // Make sure the lane is active
        if (active) {
          if (lanes_to_skip <= 0 && tile_rank < tile_size) {
             // Prepare a member_mask that is appropriate for a tile
            member_mask |= active;
            tile_rank++;
          }
          lanes_to_skip--;
        }
      }
      coalesced_group coalesced_tile = coalesced_group(member_mask);
      coalesced_tile.coalesced_info.tiled_info.meta_group_rank = thread_rank() / tile_size;
      coalesced_tile.coalesced_info.tiled_info.meta_group_size =
                                                      (size() + tile_size - 1) / tile_size;
      return coalesced_tile;
    }
     return coalesced_group(0);
  }

 protected:
 // Constructor
  explicit __CG_QUALIFIER__ coalesced_group(lane_mask member_mask)
      : thread_group(internal::cg_coalesced_group) {
    coalesced_info.member_mask = member_mask; // Which threads are active
    coalesced_info.size = __popcll(coalesced_info.member_mask); // How many threads are active
    coalesced_info.tiled_info.is_tiled = false; // Not a partitioned group
    coalesced_info.tiled_info.meta_group_rank = 0;
    coalesced_info.tiled_info.meta_group_size = 1;
  }

 public:
   __CG_QUALIFIER__ unsigned int size() const {
     return coalesced_info.size;
   }

   __CG_QUALIFIER__ unsigned int thread_rank() const {
     return internal::coalesced_group::masked_bit_count(coalesced_info.member_mask);
    }

   __CG_QUALIFIER__ void sync() const {
       internal::coalesced_group::sync();
    }

   __CG_QUALIFIER__ unsigned int meta_group_rank() const {
       return coalesced_info.tiled_info.meta_group_rank;
    }

   __CG_QUALIFIER__ unsigned int meta_group_size() const {
       return coalesced_info.tiled_info.meta_group_size;
   }

  template <class T>
  __CG_QUALIFIER__ T shfl(T var, int srcRank) const {
    static_assert(is_valid_type<T>::value, "Neither an integer or float type.");

    srcRank = srcRank % static_cast<int>(size());

    int lane = (size() == __AMDGCN_WAVEFRONT_SIZE) ? srcRank
             : (__AMDGCN_WAVEFRONT_SIZE == 64)     ? __fns64(coalesced_info.member_mask, 0, (srcRank + 1))
                                          : __fns32(coalesced_info.member_mask, 0, (srcRank + 1));

    return __shfl(var, lane, __AMDGCN_WAVEFRONT_SIZE);
  }

  template <class T>
  __CG_QUALIFIER__ T shfl_down(T var, unsigned int lane_delta) const {
    static_assert(is_valid_type<T>::value, "Neither an integer or float type.");

    // Note: The cuda implementation appears to use the remainder of lane_delta
    // and WARP_SIZE as the shift value rather than lane_delta itself.
    // This is not described in the documentation and is not done here.

    if (size() == __AMDGCN_WAVEFRONT_SIZE) {
      return __shfl_down(var, lane_delta, __AMDGCN_WAVEFRONT_SIZE);
    }

    int lane;
    if (__AMDGCN_WAVEFRONT_SIZE == 64) {
      lane = __fns64(coalesced_info.member_mask, __lane_id(), lane_delta + 1);
    }
    else {
      lane = __fns32(coalesced_info.member_mask, __lane_id(), lane_delta + 1);
    }

    if (lane == -1) {
      lane = __lane_id();
    }

    return __shfl(var, lane, __AMDGCN_WAVEFRONT_SIZE);
  }

  template <class T>
  __CG_QUALIFIER__ T shfl_up(T var, unsigned int lane_delta) const {
    static_assert(is_valid_type<T>::value, "Neither an integer or float type.");

    // Note: The cuda implementation appears to use the remainder of lane_delta
    // and WARP_SIZE as the shift value rather than lane_delta itself.
    // This is not described in the documentation and is not done here.

    if (size() == __AMDGCN_WAVEFRONT_SIZE) {
      return __shfl_up(var, lane_delta, __AMDGCN_WAVEFRONT_SIZE);
    }

    int lane;
    if (__AMDGCN_WAVEFRONT_SIZE == 64) {
      lane = __fns64(coalesced_info.member_mask, __lane_id(), -(lane_delta + 1));
    }
    else if (__AMDGCN_WAVEFRONT_SIZE == 32) {
      lane = __fns32(coalesced_info.member_mask, __lane_id(), -(lane_delta + 1));
    }

    if (lane == -1) {
      lane = __lane_id();
    }

    return __shfl(var, lane, __AMDGCN_WAVEFRONT_SIZE);
  }
};

/** \brief   User exposed API to create coalesced groups.
 *
 *  \details A collective operation that groups  all active lanes into a new thread group.
 *  @note  This function is implemented on Linux, under developement
 *  on Windows.
 */

__CG_QUALIFIER__ coalesced_group coalesced_threads() {
    return cooperative_groups::coalesced_group(__builtin_amdgcn_read_exec());
}

/**
 *  Implemenation of all publicly exposed base class APIs
 *  @note  This function is implemented on Linux, under developement
 *  on Windows.
 */
__CG_QUALIFIER__ uint32_t thread_group::thread_rank() const {
  switch (this->_type) {
    case internal::cg_multi_grid: {
      return (static_cast<const multi_grid_group*>(this)->thread_rank());
    }
    case internal::cg_grid: {
      return (static_cast<const grid_group*>(this)->thread_rank());
    }
    case internal::cg_workgroup: {
      return (static_cast<const thread_block*>(this)->thread_rank());
    }
    case internal::cg_tiled_group: {
      return (static_cast<const tiled_group*>(this)->thread_rank());
    }
    case internal::cg_coalesced_group: {
      return (static_cast<const coalesced_group*>(this)->thread_rank());
    }
    default: {
      __hip_assert(false && "invalid cooperative group type")
      return -1;
    }
  }
}
/**
 *  Implemenation of all publicly exposed thread group API
 *  @note  This function is implemented on Linux, under developement
 *  on Windows.
 */
__CG_QUALIFIER__ bool thread_group::is_valid() const {
  switch (this->_type) {
    case internal::cg_multi_grid: {
      return (static_cast<const multi_grid_group*>(this)->is_valid());
    }
    case internal::cg_grid: {
      return (static_cast<const grid_group*>(this)->is_valid());
    }
    case internal::cg_workgroup: {
      return (static_cast<const thread_block*>(this)->is_valid());
    }
    case internal::cg_tiled_group: {
      return (static_cast<const tiled_group*>(this)->is_valid());
    }
    case internal::cg_coalesced_group: {
      return (static_cast<const coalesced_group*>(this)->is_valid());
    }
    default: {
      __hip_assert(false && "invalid cooperative group type")
      return false;
    }
  }
}
/**
 *  Implemenation of all publicly exposed thread group sync API
 *  @note  This function is implemented on Linux, under developement
 *  on Windows.
 */
__CG_QUALIFIER__ void thread_group::sync() const {
  switch (this->_type) {
    case internal::cg_multi_grid: {
      static_cast<const multi_grid_group*>(this)->sync();
      break;
    }
    case internal::cg_grid: {
      static_cast<const grid_group*>(this)->sync();
      break;
    }
    case internal::cg_workgroup: {
      static_cast<const thread_block*>(this)->sync();
      break;
    }
    case internal::cg_tiled_group: {
      static_cast<const tiled_group*>(this)->sync();
      break;
    }
    case internal::cg_coalesced_group: {
      static_cast<const coalesced_group*>(this)->sync();
      break;
    }
    default: {
      __hip_assert(false && "invalid cooperative group type")
    }
  }
}

/**
 *  Implemenation of publicly exposed `wrapper` API on top of basic cooperative
 *  group type APIs
 *  @note  This function is implemented on Linux, under developement
 *  on Windows.
 */
template <class CGTy> __CG_QUALIFIER__ uint32_t group_size(CGTy const& g) { return g.size(); }
/**
 *  Implemenation of publicly exposed `wrapper` API on top of basic cooperative
 *  group type APIs
 *  @note  This function is implemented on Linux, under developement
 *  on Windows.
 */
template <class CGTy> __CG_QUALIFIER__ uint32_t thread_rank(CGTy const& g) {
  return g.thread_rank();
}
/**
 *  Implemenation of publicly exposed `wrapper` API on top of basic cooperative
 *  group type APIs
 *  @note  This function is implemented on Linux, under developement
 *  on Windows.
 */
template <class CGTy> __CG_QUALIFIER__ bool is_valid(CGTy const& g) { return g.is_valid(); }
/**
 *  Implemenation of publicly exposed `wrapper` API on top of basic cooperative
 *  group type APIs
 *  @note  This function is implemented on Linux, under developement
 *  on Windows.
 */
template <class CGTy> __CG_QUALIFIER__ void sync(CGTy const& g) { g.sync(); }
/**
 * template class tile_base
 *  @note  This class is implemented on Linux, under developement
 *  on Windows.
 */
template <unsigned int tileSize> class tile_base {
 protected:
  _CG_STATIC_CONST_DECL_ unsigned int numThreads = tileSize;

 public:
  // Rank of the thread within this tile
  _CG_STATIC_CONST_DECL_ unsigned int thread_rank() {
    return (internal::workgroup::thread_rank() & (numThreads - 1));
  }

  // Number of threads within this tile
  __CG_STATIC_QUALIFIER__ unsigned int size() { return numThreads; }
};
/**
 * template class thread_block_tile_base
 *  @note  This class is implemented on Linux, under developement
 *  on Windows.
 */
template <unsigned int size> class thread_block_tile_base : public tile_base<size> {
  static_assert(is_valid_tile_size<size>::value,
                "Tile size is either not a power of 2 or greater than the wavefront size");
  using tile_base<size>::numThreads;

 public:
  __CG_STATIC_QUALIFIER__ void sync() {
    internal::tiled_group::sync();
  }

  template <class T> __CG_QUALIFIER__ T shfl(T var, int srcRank) const {
    static_assert(is_valid_type<T>::value, "Neither an integer or float type.");
    return (__shfl(var, srcRank, numThreads));
  }

  template <class T> __CG_QUALIFIER__ T shfl_down(T var, unsigned int lane_delta) const {
    static_assert(is_valid_type<T>::value, "Neither an integer or float type.");
    return (__shfl_down(var, lane_delta, numThreads));
  }

  template <class T> __CG_QUALIFIER__ T shfl_up(T var, unsigned int lane_delta) const {
    static_assert(is_valid_type<T>::value, "Neither an integer or float type.");
    return (__shfl_up(var, lane_delta, numThreads));
  }

  template <class T> __CG_QUALIFIER__ T shfl_xor(T var, unsigned int laneMask) const {
    static_assert(is_valid_type<T>::value, "Neither an integer or float type.");
    return (__shfl_xor(var, laneMask, numThreads));
  }
};
/** \brief   User exposed API that captures the state of the parent group pre-partition
 */
template <unsigned int tileSize, typename ParentCGTy>
class parent_group_info {
public:
  // Returns the linear rank of the group within the set of tiles partitioned
  // from a parent group (bounded by meta_group_size)
  __CG_STATIC_QUALIFIER__ unsigned int meta_group_rank() {
    return ParentCGTy::thread_rank() / tileSize;
  }

  // Returns the number of groups created when the parent group was partitioned.
  __CG_STATIC_QUALIFIER__ unsigned int meta_group_size() {
    return (ParentCGTy::size() + tileSize - 1) / tileSize;
  }
};

/** \brief   Group type - thread_block_tile
 *
 *  \details  Represents one tile of thread group.
 *  @note  This type is implemented on Linux, under developement
 *  on Windows.
 */
template <unsigned int tileSize, class ParentCGTy>
class thread_block_tile_type : public thread_block_tile_base<tileSize>,
                               public tiled_group,
                               public parent_group_info<tileSize, ParentCGTy> {
  _CG_STATIC_CONST_DECL_ unsigned int numThreads = tileSize;
  protected:
    __CG_QUALIFIER__ thread_block_tile_type() : tiled_group(numThreads) {
      coalesced_info.tiled_info.size = numThreads;
      coalesced_info.tiled_info.is_tiled = true;
    }
};

// Partial template specialization
template <unsigned int tileSize>
class thread_block_tile_type<tileSize, void> : public thread_block_tile_base<tileSize>,
                               public tiled_group
                             {
  _CG_STATIC_CONST_DECL_ unsigned int numThreads = tileSize;

  typedef thread_block_tile_base<numThreads> tbtBase;

 protected:

    __CG_QUALIFIER__ thread_block_tile_type(unsigned int meta_group_rank, unsigned int meta_group_size)
        : tiled_group(numThreads) {
    coalesced_info.tiled_info.size = numThreads;
    coalesced_info.tiled_info.is_tiled = true;
    coalesced_info.tiled_info.meta_group_rank = meta_group_rank;
    coalesced_info.tiled_info.meta_group_size = meta_group_size;
  }

 public:
  using tbtBase::size;
  using tbtBase::sync;
  using tbtBase::thread_rank;

  __CG_QUALIFIER__ unsigned int meta_group_rank() const {
    return coalesced_info.tiled_info.meta_group_rank;
  }

  __CG_QUALIFIER__ unsigned int meta_group_size() const {
    return coalesced_info.tiled_info.meta_group_size;
  }
// end of operative group
/**
* @}
*/
};


/** \brief   User exposed API to partition groups.
 *
 *  \details A collective operation that partitions the parent group into a one-dimensional,
 *           row-major, tiling of subgroups.
 */

__CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent, unsigned int tile_size) {
  if (parent.cg_type() == internal::cg_tiled_group) {
    const tiled_group* cg = static_cast<const tiled_group*>(&parent);
    return cg->new_tiled_group(tile_size);
  }
  else if(parent.cg_type() == internal::cg_coalesced_group) {
    const coalesced_group* cg = static_cast<const coalesced_group*>(&parent);
    return cg->new_tiled_group(tile_size);
  }
  else {
    const thread_block* tb = static_cast<const thread_block*>(&parent);
    return tb->new_tiled_group(tile_size);
  }
}

// Thread block type overload
__CG_QUALIFIER__ thread_group tiled_partition(const thread_block& parent, unsigned int tile_size) {
  return (parent.new_tiled_group(tile_size));
}

__CG_QUALIFIER__ tiled_group tiled_partition(const tiled_group& parent, unsigned int tile_size) {
  return (parent.new_tiled_group(tile_size));
}

// If a coalesced group is passed to be partitioned, it should remain coalesced
__CG_QUALIFIER__ coalesced_group tiled_partition(const coalesced_group& parent, unsigned int tile_size) {
    return (parent.new_tiled_group(tile_size));
}

template <unsigned int size, class ParentCGTy> class thread_block_tile;

namespace impl {
template <unsigned int size, class ParentCGTy> class thread_block_tile_internal;

template <unsigned int size, class ParentCGTy>
class thread_block_tile_internal : public thread_block_tile_type<size, ParentCGTy> {
 protected:
  template <unsigned int tbtSize, class tbtParentT>
  __CG_QUALIFIER__ thread_block_tile_internal(
      const thread_block_tile_internal<tbtSize, tbtParentT>& g)
      : thread_block_tile_type<size, ParentCGTy>(g.meta_group_rank(), g.meta_group_size()) {}

  __CG_QUALIFIER__ thread_block_tile_internal(const thread_block& g)
      : thread_block_tile_type<size, ParentCGTy>() {}
};
}  // namespace impl

template <unsigned int size, class ParentCGTy>
class thread_block_tile : public impl::thread_block_tile_internal<size, ParentCGTy> {
 protected:
  __CG_QUALIFIER__ thread_block_tile(const ParentCGTy& g)
      : impl::thread_block_tile_internal<size, ParentCGTy>(g) {}

 public:
  __CG_QUALIFIER__ operator thread_block_tile<size, void>() const {
    return thread_block_tile<size, void>(*this);
  }
};


template <unsigned int size>
class thread_block_tile<size, void> : public impl::thread_block_tile_internal<size, void> {
  template <unsigned int, class ParentCGTy> friend class thread_block_tile;

 protected:
 public:
  template <class ParentCGTy>
  __CG_QUALIFIER__ thread_block_tile(const thread_block_tile<size, ParentCGTy>& g)
      : impl::thread_block_tile_internal<size, void>(g) {}
};

template <unsigned int size, class ParentCGTy = void> class thread_block_tile;

namespace impl {
template <unsigned int size, class ParentCGTy> struct tiled_partition_internal;

template <unsigned int size>
struct tiled_partition_internal<size, thread_block> : public thread_block_tile<size, thread_block> {
  __CG_QUALIFIER__ tiled_partition_internal(const thread_block& g)
      : thread_block_tile<size, thread_block>(g) {}
};

}  // namespace impl

/** \brief   User exposed API to partition groups.
 *
 *  \details  This constructs a templated class derieved from thread_group.
 *            The template defines tile size of the new thread group at compile time.
 */
template <unsigned int size, class ParentCGTy>
__CG_QUALIFIER__ thread_block_tile<size, ParentCGTy> tiled_partition(const ParentCGTy& g) {
  static_assert(is_valid_tile_size<size>::value,
                "Tiled partition with size > wavefront size. Currently not supported ");
  return impl::tiled_partition_internal<size, ParentCGTy>(g);
}
}  // namespace cooperative_groups

#endif  // __cplusplus
#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_H
/*
Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#ifdef __cplusplus

/**
 * @brief Unsafe floating point rmw atomic add.
 *
 * Performs a relaxed read-modify-write floating point atomic add with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated to have the original value plus \p value
 *
 * @note This operation currently only performs different operations for
 * the gfx90a target. Other devices continue to use safe atomics.
 *
 * It can be used to generate code that uses fast hardware floating point atomic
 * operations which may handle rounding and subnormal values differently than
 * non-atomic floating point operations.
 *
 * The operation is not always safe and can have undefined behavior unless
 * following condition are met:
 *
 * - \p addr is at least 4 bytes aligned
 * - If \p addr is a global segment address, it is in a coarse grain allocation.
 * Passing in global segment addresses in fine grain allocations will result in
 * undefined behavior and is not supported.
 *
 * @param [in,out] addr Pointer to value to be increment by \p value.
 * @param [in] value Value by \p addr is to be incremented.
 * @return Original value contained in \p addr.
 */
__device__ inline float unsafeAtomicAdd(float* addr, float value) {
#if defined(__gfx90a__) &&                                                   \
    __has_builtin(__builtin_amdgcn_is_shared) &&                               \
    __has_builtin(__builtin_amdgcn_is_private) &&                              \
    __has_builtin(__builtin_amdgcn_ds_atomic_fadd_f32) &&                      \
    __has_builtin(__builtin_amdgcn_global_atomic_fadd_f32)
  if (__builtin_amdgcn_is_shared(
        (const __attribute__((address_space(0))) void*)addr))
    return __builtin_amdgcn_ds_atomic_fadd_f32(addr, value);
  else if (__builtin_amdgcn_is_private(
              (const __attribute__((address_space(0))) void*)addr)) {
    float temp = *addr;
    *addr = temp + value;
    return temp;
  }
  else
    return __builtin_amdgcn_global_atomic_fadd_f32(addr, value);
#elif __has_builtin(__hip_atomic_fetch_add)
  return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
  return __atomic_fetch_add(addr, value, __ATOMIC_RELAXED);
#endif
}

/**
 * @brief Unsafe floating point rmw atomic max.
 *
 * Performs a relaxed read-modify-write floating point atomic max with
 * device memory scope. The original value at \p addr is returned and
 * the value at \p addr is replaced by \p val if greater.
 *
 * @note This operation is currently identical to that performed by
 * atomicMax and is included for completeness.
 *
 * @param [in,out] addr Pointer to value to be updated
 * @param [in] val Value used to update the value at \p addr.
 * @return Original value contained in \p addr.
 */
__device__ inline float unsafeAtomicMax(float* addr, float val) {
  #if __has_builtin(__hip_atomic_load) && \
      __has_builtin(__hip_atomic_compare_exchange_strong)
  float value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  bool done = false;
  while (!done && value < val) {
    done = __hip_atomic_compare_exchange_strong(addr, &value, val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  return value;
  #else
  unsigned int *uaddr = (unsigned int *)addr;
  unsigned int value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __uint_as_float(value) < val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __float_as_uint(val), false,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __uint_as_float(value);
  #endif
}

/**
 * @brief Unsafe floating point rmw atomic min.
 *
 * Performs a relaxed read-modify-write floating point atomic min with
 * device memory scope. The original value at \p addr is returned and
 * the value at \p addr is replaced by \p val if lesser.
 *
 * @note This operation is currently identical to that performed by
 * atomicMin and is included for completeness.
 *
 * @param [in,out] addr Pointer to value to be updated
 * @param [in] val Value used to update the value at \p addr.
 * @return Original value contained in \p addr.
 */
__device__ inline float unsafeAtomicMin(float* addr, float val) {
  #if __has_builtin(__hip_atomic_load) && \
      __has_builtin(__hip_atomic_compare_exchange_strong)
  float value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  bool done = false;
  while (!done && value > val) {
    done = __hip_atomic_compare_exchange_strong(addr, &value, val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  return value;
  #else
  unsigned int *uaddr = (unsigned int *)addr;
  unsigned int value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __uint_as_float(value) > val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __float_as_uint(val), false,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __uint_as_float(value);
  #endif
}

/**
 * @brief Unsafe double precision rmw atomic add.
 *
 * Performs a relaxed read-modify-write double precision atomic add with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated to have the original value plus \p value
 *
 * @note This operation currently only performs different operations for
 * the gfx90a target. Other devices continue to use safe atomics.
 *
 * It can be used to generate code that uses fast hardware floating point atomic
 * operations which may handle rounding and subnormal values differently than
 * non-atomic floating point operations.
 *
 * The operation is not always safe and can have undefined behavior unless
 * following condition are met:
 *
 * - \p addr is at least 8 byte aligned
 * - If \p addr is a global segment address, it is in a coarse grain allocation.
 * Passing in global segment addresses in fine grain allocations will result in
 * undefined behavior and are not supported.
 *
 * @param [in,out] addr Pointer to value to be updated.
 * @param [in] value Value by \p addr is to be incremented.
 * @return Original value contained in \p addr.
 */
__device__ inline double unsafeAtomicAdd(double* addr, double value) {
#if defined(__gfx90a__) && __has_builtin(__builtin_amdgcn_flat_atomic_fadd_f64)
  return __builtin_amdgcn_flat_atomic_fadd_f64(addr, value);
#elif defined (__hip_atomic_fetch_add)
  return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
  return __atomic_fetch_add(addr, value, __ATOMIC_RELAXED);
#endif
}

/**
 * @brief Unsafe double precision rmw atomic max.
 *
 * Performs a relaxed read-modify-write double precision atomic max with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated with \p val if greater.
 *
 * @note This operation currently only performs different operations for
 * the gfx90a target. Other devices continue to use safe atomics.
 *
 * It can be used to generate code that uses fast hardware floating point atomic
 * operations which may handle rounding and subnormal values differently than
 * non-atomic floating point operations.
 *
 * The operation is not always safe and can have undefined behavior unless
 * following condition are met:
 *
 * - \p addr is at least 8 byte aligned
 * - If \p addr is a global segment address, it is in a coarse grain allocation.
 * Passing in global segment addresses in fine grain allocations will result in
 * undefined behavior and are not supported.
 *
 * @param [in,out] addr Pointer to value to be updated.
 * @param [in] val Value used to updated the contents at \p addr
 * @return Original value contained at \p addr.
 */
__device__ inline double unsafeAtomicMax(double* addr, double val) {
#if (defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)) &&  \
    __has_builtin(__builtin_amdgcn_flat_atomic_fmax_f64)
  return __builtin_amdgcn_flat_atomic_fmax_f64(addr, val);
#else
  #if __has_builtin(__hip_atomic_load) && \
      __has_builtin(__hip_atomic_compare_exchange_strong)
  double value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  bool done = false;
  while (!done && value < val) {
    done = __hip_atomic_compare_exchange_strong(addr, &value, val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  return value;
  #else
  unsigned long long *uaddr = (unsigned long long *)addr;
  unsigned long long value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __longlong_as_double(value) < val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __double_as_longlong(val), false,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __longlong_as_double(value);
  #endif
#endif
}

/**
 * @brief Unsafe double precision rmw atomic min.
 *
 * Performs a relaxed read-modify-write double precision atomic min with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated with \p val if lesser.
 *
 * @note This operation currently only performs different operations for
 * the gfx90a target. Other devices continue to use safe atomics.
 *
 * It can be used to generate code that uses fast hardware floating point atomic
 * operations which may handle rounding and subnormal values differently than
 * non-atomic floating point operations.
 *
 * The operation is not always safe and can have undefined behavior unless
 * following condition are met:
 *
 * - \p addr is at least 8 byte aligned
 * - If \p addr is a global segment address, it is in a coarse grain allocation.
 * Passing in global segment addresses in fine grain allocations will result in
 * undefined behavior and are not supported.
 *
 * @param [in,out] addr Pointer to value to be updated.
 * @param [in] val Value used to updated the contents at \p addr
 * @return Original value contained at \p addr.
 */
__device__ inline double unsafeAtomicMin(double* addr, double val) {
#if (defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)) &&  \
    __has_builtin(__builtin_amdgcn_flat_atomic_fmin_f64)
  return __builtin_amdgcn_flat_atomic_fmin_f64(addr, val);
#else
  #if __has_builtin(__hip_atomic_load) && \
      __has_builtin(__hip_atomic_compare_exchange_strong)
  double value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  bool done = false;
  while (!done && value > val) {
    done = __hip_atomic_compare_exchange_strong(addr, &value, val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  return value;
  #else
  unsigned long long *uaddr = (unsigned long long *)addr;
  unsigned long long value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __longlong_as_double(value) > val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __double_as_longlong(val), false,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __longlong_as_double(value);
  #endif
#endif
}

/**
 * @brief Safe floating point rmw atomic add.
 *
 * Performs a relaxed read-modify-write floating point atomic add with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated to have the original value plus \p value
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be increment by \p value.
 * @param [in] value Value by \p addr is to be incremented.
 * @return Original value contained in \p addr.
 */
__device__ inline float safeAtomicAdd(float* addr, float value) {
#if defined(__gfx908__) || defined(__gfx941__)                                \
    || ((defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx942__))   \
         && !__has_builtin(__hip_atomic_fetch_add))
  // On gfx908, we can generate unsafe FP32 atomic add that does not follow all
  // IEEE rules when -munsafe-fp-atomics is passed. Do a CAS loop emulation instead.
  // On gfx941, we can generate unsafe FP32 atomic add that may not always happen atomically,
  // so we need to force a CAS loop emulation to ensure safety.
  // On gfx90a, gfx940 and gfx942 if we do not have the __hip_atomic_fetch_add builtin, we
  // need to force a CAS loop here.
  float old_val;
#if __has_builtin(__hip_atomic_load)
  old_val = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else // !__has_builtin(__hip_atomic_load)
  old_val = __uint_as_float(__atomic_load_n(reinterpret_cast<unsigned int*>(addr), __ATOMIC_RELAXED));
#endif // __has_builtin(__hip_atomic_load)
  float expected, temp;
  do {
    temp = expected = old_val;
#if __has_builtin(__hip_atomic_compare_exchange_strong)
    __hip_atomic_compare_exchange_strong(addr, &expected, old_val + value, __ATOMIC_RELAXED,
                                         __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else // !__has_builtin(__hip_atomic_compare_exchange_strong)
    __atomic_compare_exchange_n(addr, &expected, old_val + value, false,
                                __ATOMIC_RELAXED, __ATOMIC_RELAXED);
#endif // __has_builtin(__hip_atomic_compare_exchange_strong)
    old_val = expected;
  } while (__float_as_uint(temp) != __float_as_uint(old_val));
  return old_val;
#elif defined(__gfx90a__)
  // On gfx90a, with the __hip_atomic_fetch_add builtin, relaxed system-scope
  // atomics will produce safe CAS loops, but are otherwise not different than
  // agent-scope atomics. This logic is only applicable for gfx90a, and should
  // not be assumed on other architectures.
  return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#elif __has_builtin(__hip_atomic_fetch_add)
  return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
  return __atomic_fetch_add(addr, value, __ATOMIC_RELAXED);
#endif
}

/**
 * @brief Safe floating point rmw atomic max.
 *
 * Performs a relaxed read-modify-write floating point atomic max with
 * device memory scope. The original value at \p addr is returned and
 * the value at \p addr is replaced by \p val if greater.
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be updated
 * @param [in] val Value used to update the value at \p addr.
 * @return Original value contained in \p addr.
 */
__device__ inline float safeAtomicMax(float* addr, float val) {
  #if __has_builtin(__hip_atomic_load) && \
      __has_builtin(__hip_atomic_compare_exchange_strong)
  float value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  bool done = false;
  while (!done && value < val) {
    done = __hip_atomic_compare_exchange_strong(addr, &value, val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  return value;
  #else
  unsigned int *uaddr = (unsigned int *)addr;
  unsigned int value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __uint_as_float(value) < val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __float_as_uint(val), false,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __uint_as_float(value);
  #endif
}

/**
 * @brief Safe floating point rmw atomic min.
 *
 * Performs a relaxed read-modify-write floating point atomic min with
 * device memory scope. The original value at \p addr is returned and
 * the value at \p addr is replaced by \p val if lesser.
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be updated
 * @param [in] val Value used to update the value at \p addr.
 * @return Original value contained in \p addr.
 */
__device__ inline float safeAtomicMin(float* addr, float val) {
  #if __has_builtin(__hip_atomic_load) && \
      __has_builtin(__hip_atomic_compare_exchange_strong)
  float value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  bool done = false;
  while (!done && value > val) {
    done = __hip_atomic_compare_exchange_strong(addr, &value, val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  return value;
  #else
  unsigned int *uaddr = (unsigned int *)addr;
  unsigned int value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __uint_as_float(value) > val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __float_as_uint(val), false,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __uint_as_float(value);
  #endif
}

/**
 * @brief Safe double precision rmw atomic add.
 *
 * Performs a relaxed read-modify-write double precision atomic add with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated to have the original value plus \p value
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be increment by \p value.
 * @param [in] value Value by \p addr is to be incremented.
 * @return Original value contained in \p addr.
 */
__device__ inline double safeAtomicAdd(double* addr, double value) {
#if defined(__gfx90a__) &&  __has_builtin(__hip_atomic_fetch_add)
  // On gfx90a, with the __hip_atomic_fetch_add builtin, relaxed system-scope
  // atomics will produce safe CAS loops, but are otherwise not different than
  // agent-scope atomics. This logic is only applicable for gfx90a, and should
  // not be assumed on other architectures.
  return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#elif defined(__gfx90a__)
  // On gfx90a, if we do not have the __hip_atomic_fetch_add builtin, we need to
  // force a CAS loop here.
  double old_val;
#if __has_builtin(__hip_atomic_load)
  old_val = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else // !__has_builtin(__hip_atomic_load)
  old_val = __longlong_as_double(__atomic_load_n(reinterpret_cast<unsigned long long*>(addr), __ATOMIC_RELAXED));
#endif // __has_builtin(__hip_atomic_load)
  double expected, temp;
  do {
    temp = expected = old_val;
#if __has_builtin(__hip_atomic_compare_exchange_strong)
    __hip_atomic_compare_exchange_strong(addr, &expected, old_val + value, __ATOMIC_RELAXED,
                                         __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else // !__has_builtin(__hip_atomic_compare_exchange_strong)
    __atomic_compare_exchange_n(addr, &expected, old_val + value, false,
                                __ATOMIC_RELAXED, __ATOMIC_RELAXED);
#endif // __has_builtin(__hip_atomic_compare_exchange_strong)
    old_val = expected;
  } while (__double_as_longlong(temp) != __double_as_longlong(old_val));
  return old_val;
#else // !defined(__gfx90a__)
#if __has_builtin(__hip_atomic_fetch_add)
  return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else  // !__has_builtin(__hip_atomic_fetch_add)
  return __atomic_fetch_add(addr, value, __ATOMIC_RELAXED);
#endif // __has_builtin(__hip_atomic_fetch_add)
#endif
}

/**
 * @brief Safe double precision rmw atomic max.
 *
 * Performs a relaxed read-modify-write double precision atomic max with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated with \p val if greater.
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be updated.
 * @param [in] val Value used to updated the contents at \p addr
 * @return Original value contained at \p addr.
 */
__device__ inline double safeAtomicMax(double* addr, double val) {
  #if __has_builtin(__builtin_amdgcn_is_private)
  if (__builtin_amdgcn_is_private(
          (const __attribute__((address_space(0))) void*)addr)) {
    double old = *addr;
    *addr = __builtin_fmax(old, val);
    return old;
  } else {
  #endif
  #if __has_builtin(__hip_atomic_load) && \
      __has_builtin(__hip_atomic_compare_exchange_strong)
  double value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  bool done = false;
  while (!done && value < val) {
    done = __hip_atomic_compare_exchange_strong(addr, &value, val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  return value;
  #else
  unsigned long long *uaddr = (unsigned long long *)addr;
  unsigned long long value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __longlong_as_double(value) < val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __double_as_longlong(val), false,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __longlong_as_double(value);
  #endif
  #if __has_builtin(__builtin_amdgcn_is_private)
  }
  #endif
}

/**
 * @brief Safe double precision rmw atomic min.
 *
 * Performs a relaxed read-modify-write double precision atomic min with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated with \p val if lesser.
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be updated.
 * @param [in] val Value used to updated the contents at \p addr
 * @return Original value contained at \p addr.
 */
__device__ inline double safeAtomicMin(double* addr, double val) {
  #if __has_builtin(__builtin_amdgcn_is_private)
  if (__builtin_amdgcn_is_private(
           (const __attribute__((address_space(0))) void*)addr)) {
    double old = *addr;
    *addr = __builtin_fmin(old, val);
    return old;
  } else {
  #endif
  #if __has_builtin(__hip_atomic_load) && \
      __has_builtin(__hip_atomic_compare_exchange_strong)
  double value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  bool done = false;
  while (!done && value > val) {
    done = __hip_atomic_compare_exchange_strong(addr, &value, val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  return value;
  #else
  unsigned long long *uaddr = (unsigned long long *)addr;
  unsigned long long value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __longlong_as_double(value) > val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __double_as_longlong(val), false,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __longlong_as_double(value);
  #endif
  #if __has_builtin(__builtin_amdgcn_is_private)
  }
  #endif
}

#endif
/*
Copyright (c) 2015 - Present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#if !defined(__HIPCC_RTC__)
#include "amd_device_functions.h"
#endif

#if __has_builtin(__hip_atomic_compare_exchange_strong)

template<bool B, typename T, typename F> struct Cond_t;

template<typename T, typename F> struct Cond_t<true, T, F> { using type = T; };
template<typename T, typename F> struct Cond_t<false, T, F> { using type = F; };

#if !__HIP_DEVICE_COMPILE__
//TODO: Remove this after compiler pre-defines the following Macros.
#define __HIP_MEMORY_SCOPE_SINGLETHREAD 1
#define __HIP_MEMORY_SCOPE_WAVEFRONT 2
#define __HIP_MEMORY_SCOPE_WORKGROUP 3
#define __HIP_MEMORY_SCOPE_AGENT 4
#define __HIP_MEMORY_SCOPE_SYSTEM 5
#endif

#if !defined(__HIPCC_RTC__)
#include "amd_hip_unsafe_atomics.h"
#endif

// Atomic expanders
template<
  int mem_order = __ATOMIC_SEQ_CST,
  int mem_scope= __HIP_MEMORY_SCOPE_SYSTEM,
  typename T,
  typename Op,
  typename F>
inline
__attribute__((always_inline, device))
T hip_cas_expander(T* p, T x, Op op, F f) noexcept
{
  using FP = __attribute__((address_space(0))) const void*;

  __device__
  extern bool is_shared_workaround(FP) asm("llvm.amdgcn.is.shared");

  if (is_shared_workaround((FP)p))
    return f();

  using U = typename Cond_t<
    sizeof(T) == sizeof(unsigned int), unsigned int, unsigned long long>::type;

  auto q = reinterpret_cast<U*>(p);

  U tmp0{__hip_atomic_load(q, mem_order, mem_scope)};
  U tmp1;
  do {
    tmp1 = tmp0;

    op(reinterpret_cast<T&>(tmp1), x);
  } while (!__hip_atomic_compare_exchange_strong(q, &tmp0, tmp1, mem_order,
                                                 mem_order, mem_scope));

  return reinterpret_cast<const T&>(tmp0);
}

template<
  int mem_order = __ATOMIC_SEQ_CST,
  int mem_scope= __HIP_MEMORY_SCOPE_SYSTEM,
  typename T,
  typename Cmp,
  typename F>
inline
__attribute__((always_inline, device))
T hip_cas_extrema_expander(T* p, T x, Cmp cmp, F f) noexcept
{
  using FP = __attribute__((address_space(0))) const void*;

  __device__
  extern bool is_shared_workaround(FP) asm("llvm.amdgcn.is.shared");

  if (is_shared_workaround((FP)p))
    return f();

  using U = typename Cond_t<
    sizeof(T) == sizeof(unsigned int), unsigned int, unsigned long long>::type;

  auto q = reinterpret_cast<U*>(p);

  U tmp{__hip_atomic_load(q, mem_order, mem_scope)};
  while (cmp(x, reinterpret_cast<const T&>(tmp)) &&
         !__hip_atomic_compare_exchange_strong(q, &tmp, x, mem_order, mem_order,
                                               mem_scope));

  return reinterpret_cast<const T&>(tmp);
}

__device__
inline
int atomicCAS(int* address, int compare, int val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
    return compare;
}

__device__
inline
int atomicCAS_system(int* address, int compare, int val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
    return compare;
}

__device__
inline
unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__
inline
unsigned int atomicCAS_system(unsigned int* address, unsigned int compare, unsigned int val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  return compare;
}

__device__
inline
unsigned long atomicCAS(unsigned long* address, unsigned long compare, unsigned long val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__
inline
unsigned long atomicCAS_system(unsigned long* address, unsigned long compare, unsigned long val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  return compare;
}

__device__
inline
unsigned long long atomicCAS(unsigned long long* address, unsigned long long compare,
                             unsigned long long val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__
inline
unsigned long long atomicCAS_system(unsigned long long* address, unsigned long long compare,
                                    unsigned long long val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  return compare;
}

__device__
inline
float atomicCAS(float* address, float compare, float val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
    return compare;
}

__device__
inline
float atomicCAS_system(float* address, float compare, float val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
    return compare;
}

__device__
inline
double atomicCAS(double* address, double compare, double val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
    return compare;
}

__device__
inline
double atomicCAS_system(double* address, double compare, double val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
    return compare;
}

__device__
inline
int atomicAdd(int* address, int val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
int atomicAdd_system(int* address, int val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned int atomicAdd(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned int atomicAdd_system(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned long atomicAdd(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long atomicAdd_system(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned long long atomicAdd(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long long atomicAdd_system(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
float atomicAdd(float* address, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicAdd(address, val);
#else
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif
}

__device__
inline
float atomicAdd_system(float* address, float val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

#if !defined(__HIPCC_RTC__)
DEPRECATED("use atomicAdd instead")
#endif // !defined(__HIPCC_RTC__)
__device__
inline
void atomicAddNoRet(float* address, float val)
{
    __ockl_atomic_add_noret_f32(address, val);
}

__device__
inline
double atomicAdd(double* address, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicAdd(address, val);
#else
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif
}

__device__
inline
double atomicAdd_system(double* address, double val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
int atomicSub(int* address, int val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
int atomicSub_system(int* address, int val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned int atomicSub(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned int atomicSub_system(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned long atomicSub(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long atomicSub_system(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned long long atomicSub(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long long atomicSub_system(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
float atomicSub(float* address, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicAdd(address, -val);
#else
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif
}

__device__
inline
float atomicSub_system(float* address, float val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
double atomicSub(double* address, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicAdd(address, -val);
#else
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif
}

__device__
inline
double atomicSub_system(double* address, double val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
int atomicExch(int* address, int val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
int atomicExch_system(int* address, int val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned int atomicExch(unsigned int* address, unsigned int val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned int atomicExch_system(unsigned int* address, unsigned int val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned long atomicExch(unsigned long* address, unsigned long val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long atomicExch_system(unsigned long* address, unsigned long val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned long long atomicExch(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long long atomicExch_system(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
float atomicExch(float* address, float val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
float atomicExch_system(float* address, float val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
double atomicExch(double* address, double val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
double atomicExch_system(double* address, double val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
int atomicMin(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](int x, int y) { return x < y; }, [=]() {
      return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
int atomicMin_system(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](int x, int y) { return x < y; }, [=]() {
      return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned int atomicMin(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](unsigned int x, unsigned int y) { return x < y; }, [=]() {
      return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__

}

__device__
inline
unsigned int atomicMin_system(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](unsigned int x, unsigned int y) { return x < y; }, [=]() {
      return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicMin(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address,
    val,
    [](unsigned long x, unsigned long y) { return x < y; },
    [=]() {
    return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long atomicMin_system(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address,
    val,
    [](unsigned long x, unsigned long y) { return x < y; },
    [=]() {
    return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicMin(unsigned long long* address, unsigned long long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address,
    val,
    [](unsigned long long x, unsigned long long y) { return x < y; },
    [=]() {
    return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicMin_system(unsigned long long* address, unsigned long long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address,
    val,
    [](unsigned long long x, unsigned long long y) { return x < y; },
    [=]() {
    return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
long long atomicMin(long long* address, long long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
      address, val, [](long long x, long long y) { return x < y; },
      [=]() {
        return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif  // __gfx941__
}

__device__
inline
long long atomicMin_system(long long* address, long long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
      address, val, [](long long x, long long y) { return x < y; },
      [=]() {
        return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif  // __gfx941__
}

__device__
inline
float atomicMin(float* addr, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMin(addr, val);
#else
  #if __has_builtin(__hip_atomic_load) && \
      __has_builtin(__hip_atomic_compare_exchange_strong)
  float value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  bool done = false;
  while (!done && value > val) {
    done = __hip_atomic_compare_exchange_strong(addr, &value, val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  return value;
  #else
  unsigned int *uaddr = (unsigned int *)addr;
  unsigned int value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __uint_as_float(value) > val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __float_as_uint(val), false,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __uint_as_float(value);
  #endif
#endif
}

__device__
inline
float atomicMin_system(float* address, float val) {
  unsigned int* uaddr { reinterpret_cast<unsigned int*>(address) };
  #if __has_builtin(__hip_atomic_load)
    unsigned int tmp {__hip_atomic_load(uaddr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)};
  #else
    unsigned int tmp {__atomic_load_n(uaddr, __ATOMIC_RELAXED)};
  #endif
  float value = __uint_as_float(tmp);

  while (val < value) {
    value = atomicCAS_system(address, value, val);
  }

  return value;
}

__device__
inline
double atomicMin(double* addr, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMin(addr, val);
#else
  #if __has_builtin(__hip_atomic_load) && \
      __has_builtin(__hip_atomic_compare_exchange_strong)
  double value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  bool done = false;
  while (!done && value > val) {
    done = __hip_atomic_compare_exchange_strong(addr, &value, val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  return value;
  #else
  unsigned long long *uaddr = (unsigned long long *)addr;
  unsigned long long value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __longlong_as_double(value) > val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __double_as_longlong(val), false,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __longlong_as_double(value);
  #endif
#endif
}

__device__
inline
double atomicMin_system(double* address, double val) {
  unsigned long long* uaddr { reinterpret_cast<unsigned long long*>(address) };
  #if __has_builtin(__hip_atomic_load)
    unsigned long long tmp {__hip_atomic_load(uaddr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)};
  #else
    unsigned long long tmp {__atomic_load_n(uaddr, __ATOMIC_RELAXED)};
  #endif
  double value = __longlong_as_double(tmp);

  while (val < value) {
    value = atomicCAS_system(address, value, val);
  }

  return value;
}

__device__
inline
int atomicMax(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](int x, int y) { return y < x; }, [=]() {
      return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
int atomicMax_system(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](int x, int y) { return y < x; }, [=]() {
      return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned int atomicMax(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](unsigned int x, unsigned int y) { return y < x; }, [=]() {
      return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned int atomicMax_system(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](unsigned int x, unsigned int y) { return y < x; }, [=]() {
      return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long atomicMax(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address,
    val,
    [](unsigned long x, unsigned long y) { return y < x; },
    [=]() {
    return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long atomicMax_system(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address,
    val,
    [](unsigned long x, unsigned long y) { return y < x; },
    [=]() {
    return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicMax(unsigned long long* address, unsigned long long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address,
    val,
    [](unsigned long long x, unsigned long long y) { return y < x; },
    [=]() {
      return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicMax_system(unsigned long long* address, unsigned long long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address,
    val,
    [](unsigned long long x, unsigned long long y) { return y < x; },
    [=]() {
      return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
long long atomicMax(long long* address, long long val) {
  #if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
      address, val, [](long long x, long long y) { return y < x; },
      [=]() {
        return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
long long atomicMax_system(long long* address, long long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
      address, val, [](long long x, long long y) { return y < x; },
      [=]() {
        return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif  // __gfx941__
}

__device__
inline
float atomicMax(float* addr, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMax(addr, val);
#else
  #if __has_builtin(__hip_atomic_load) && \
      __has_builtin(__hip_atomic_compare_exchange_strong)
  float value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  bool done = false;
  while (!done && value < val) {
    done = __hip_atomic_compare_exchange_strong(addr, &value, val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  return value;
  #else
  unsigned int *uaddr = (unsigned int *)addr;
  unsigned int value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __uint_as_float(value) < val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __float_as_uint(val), false,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __uint_as_float(value);
  #endif
#endif
}

__device__
inline
float atomicMax_system(float* address, float val) {
  unsigned int* uaddr { reinterpret_cast<unsigned int*>(address) };
  #if __has_builtin(__hip_atomic_load)
    unsigned int tmp {__hip_atomic_load(uaddr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)};
  #else
    unsigned int tmp {__atomic_load_n(uaddr, __ATOMIC_RELAXED)};
  #endif
  float value = __uint_as_float(tmp);

  while (value < val) {
    value = atomicCAS_system(address, value, val);
  }

  return value;
}

__device__
inline
double atomicMax(double* addr, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMax(addr, val);
#else
  #if __has_builtin(__hip_atomic_load) && \
      __has_builtin(__hip_atomic_compare_exchange_strong)
  double value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  bool done = false;
  while (!done && value < val) {
    done = __hip_atomic_compare_exchange_strong(addr, &value, val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  return value;
  #else
  unsigned long long *uaddr = (unsigned long long *)addr;
  unsigned long long value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __longlong_as_double(value) < val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __double_as_longlong(val), false,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __longlong_as_double(value);
  #endif
#endif
}

__device__
inline
double atomicMax_system(double* address, double val) {
  unsigned long long* uaddr { reinterpret_cast<unsigned long long*>(address) };
  #if __has_builtin(__hip_atomic_load)
    unsigned long long tmp {__hip_atomic_load(uaddr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)};
  #else
    unsigned long long tmp {__atomic_load_n(uaddr, __ATOMIC_RELAXED)};
  #endif
  double value = __longlong_as_double(tmp);

  while (value < val) {
      value = atomicCAS_system(address, value, val);
  }

  return value;
}

__device__
inline
unsigned int atomicInc(unsigned int* address, unsigned int val)
{
#if defined(__gfx941__)
  __device__
  extern
  unsigned int __builtin_amdgcn_atomic_inc(
    unsigned int*,
    unsigned int,
    unsigned int,
    unsigned int,
    bool) __asm("llvm.amdgcn.atomic.inc.i32.p0i32");

  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address,
    val,
    [](unsigned int& x, unsigned int y) { x = (x >= y) ? 0 : (x + 1); },
    [=]() {
    return
      __builtin_amdgcn_atomic_inc(address, val, __ATOMIC_RELAXED, 1, false);
  });
#else
    return __builtin_amdgcn_atomic_inc32(address, val, __ATOMIC_RELAXED, "agent");
#endif // __gfx941__

}

__device__
inline
unsigned int atomicDec(unsigned int* address, unsigned int val)
{
#if defined(__gfx941__)
  __device__
  extern
  unsigned int __builtin_amdgcn_atomic_dec(
    unsigned int*,
    unsigned int,
    unsigned int,
    unsigned int,
    bool) __asm("llvm.amdgcn.atomic.dec.i32.p0i32");

  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address,
    val,
    [](unsigned int& x, unsigned int y) { x = (!x || x > y) ? y : (x - 1); },
    [=]() {
    return
      __builtin_amdgcn_atomic_dec(address, val, __ATOMIC_RELAXED, 1, false);
  });
#else
  return __builtin_amdgcn_atomic_dec32(address, val, __ATOMIC_RELAXED, "agent");
#endif // __gfx941__
    
}

__device__
inline
int atomicAnd(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](int& x, int y) { x &= y; }, [=]() {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
int atomicAnd_system(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](int& x, int y) { x &= y; }, [=]() {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned int atomicAnd(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](unsigned int& x, unsigned int y) { x &= y; }, [=]() {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned int atomicAnd_system(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](unsigned int& x, unsigned int y) { x &= y; }, [=]() {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long atomicAnd(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](unsigned long& x, unsigned long y) { x &= y; }, [=]() {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long atomicAnd_system(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](unsigned long& x, unsigned long y) { x &= y; }, [=]() {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicAnd(unsigned long long* address, unsigned long long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address,
    val,
    [](unsigned long long& x, unsigned long long y) { x &= y; },
    [=]() {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicAnd_system(unsigned long long* address, unsigned long long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address,
    val,
    [](unsigned long long& x, unsigned long long y) { x &= y; },
    [=]() {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
int atomicOr(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](int& x, int y) { x |= y; }, [=]() {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
int atomicOr_system(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](int& x, int y) { x |= y; }, [=]() {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned int atomicOr(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](unsigned int& x, unsigned int y) { x |= y; }, [=]() {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned int atomicOr_system(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](unsigned int& x, unsigned int y) { x |= y; }, [=]() {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long atomicOr(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](unsigned long& x, unsigned long y) { x |= y; }, [=]() {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long atomicOr_system(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](unsigned long& x, unsigned long y) { x |= y; }, [=]() {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicOr(unsigned long long* address, unsigned long long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address,
    val,
    [](unsigned long long& x, unsigned long long y) { x |= y; },
    [=]() {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicOr_system(unsigned long long* address, unsigned long long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address,
    val,
    [](unsigned long long& x, unsigned long long y) { x |= y; },
    [=]() {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
int atomicXor(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](int& x, int y) { x ^= y; }, [=]() {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
int atomicXor_system(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](int& x, int y) { x ^= y; }, [=]() {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned int atomicXor(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](unsigned int& x, unsigned int y) { x ^= y; }, [=]() {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned int atomicXor_system(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](unsigned int& x, unsigned int y) { x ^= y; }, [=]() {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long atomicXor(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](unsigned long& x, unsigned long y) { x ^= y; }, [=]() {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long atomicXor_system(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](unsigned long& x, unsigned long y) { x ^= y; }, [=]() {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicXor(unsigned long long* address, unsigned long long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address,
    val,
    [](unsigned long long& x, unsigned long long y) { x ^= y; },
    [=]() {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicXor_system(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

#else // __hip_atomic_compare_exchange_strong

__device__
inline
int atomicCAS(int* address, int compare, int val)
{
    __atomic_compare_exchange_n(
        address, &compare, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);

    return compare;
}
__device__
inline
unsigned int atomicCAS(
    unsigned int* address, unsigned int compare, unsigned int val)
{
    __atomic_compare_exchange_n(
        address, &compare, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);

    return compare;
}
__device__
inline
unsigned long long atomicCAS(
    unsigned long long* address,
    unsigned long long compare,
    unsigned long long val)
{
    __atomic_compare_exchange_n(
        address, &compare, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);

    return compare;
}

__device__
inline
int atomicAdd(int* address, int val)
{
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicAdd(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicAdd(
    unsigned long long* address, unsigned long long val)
{
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
}
__device__
inline
float atomicAdd(float* address, float val)
{
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
    return unsafeAtomicAdd(address, val);
#else
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
#endif
}

#if !defined(__HIPCC_RTC__)
DEPRECATED("use atomicAdd instead")
#endif // !defined(__HIPCC_RTC__)
__device__
inline
void atomicAddNoRet(float* address, float val)
{
    __ockl_atomic_add_noret_f32(address, val);
}

__device__
inline
double atomicAdd(double* address, double val)
{
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
    return unsafeAtomicAdd(address, val);
#else
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
#endif
}

__device__
inline
int atomicSub(int* address, int val)
{
    return __atomic_fetch_sub(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicSub(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_sub(address, val, __ATOMIC_RELAXED);
}

__device__
inline
int atomicExch(int* address, int val)
{
    return __atomic_exchange_n(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicExch(unsigned int* address, unsigned int val)
{
    return __atomic_exchange_n(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicExch(unsigned long long* address, unsigned long long val)
{
    return __atomic_exchange_n(address, val, __ATOMIC_RELAXED);
}
__device__
inline
float atomicExch(float* address, float val)
{
    return __uint_as_float(__atomic_exchange_n(
        reinterpret_cast<unsigned int*>(address),
        __float_as_uint(val),
        __ATOMIC_RELAXED));
}

__device__
inline
int atomicMin(int* address, int val)
{
    return __atomic_fetch_min(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicMin(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_min(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicMin(
    unsigned long long* address, unsigned long long val)
{
    unsigned long long tmp{__atomic_load_n(address, __ATOMIC_RELAXED)};
    while (val < tmp) {
        const auto tmp1 = __atomic_load_n(address, __ATOMIC_RELAXED);

        if (tmp1 != tmp) { tmp = tmp1; continue; }

        tmp = atomicCAS(address, tmp, val);
    }

    return tmp;
}
__device__ inline long long atomicMin(long long* address, long long val) {
    long long tmp{__atomic_load_n(address, __ATOMIC_RELAXED)};
    while (val < tmp) {
        const auto tmp1 = __atomic_load_n(address, __ATOMIC_RELAXED);

        if (tmp1 != tmp) {
          tmp = tmp1;
          continue;
        }

        tmp = atomicCAS(address, tmp, val);
    }
    return tmp;
}

__device__
inline
int atomicMax(int* address, int val)
{
    return __atomic_fetch_max(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicMax(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_max(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicMax(
    unsigned long long* address, unsigned long long val)
{
    unsigned long long tmp{__atomic_load_n(address, __ATOMIC_RELAXED)};
    while (tmp < val) {
        const auto tmp1 = __atomic_load_n(address, __ATOMIC_RELAXED);

        if (tmp1 != tmp) { tmp = tmp1; continue; }

        tmp = atomicCAS(address, tmp, val);
    }

    return tmp;
}
__device__ inline long long atomicMax(long long* address, long long val) {
    long long tmp{__atomic_load_n(address, __ATOMIC_RELAXED)};
    while (tmp < val) {
        const auto tmp1 = __atomic_load_n(address, __ATOMIC_RELAXED);

        if (tmp1 != tmp) {
          tmp = tmp1;
          continue;
        }

        tmp = atomicCAS(address, tmp, val);
    }
    return tmp;
}

__device__
inline
unsigned int atomicInc(unsigned int* address, unsigned int val)
{
  return __builtin_amdgcn_atomic_inc32(address, val, __ATOMIC_RELAXED, "agent");
}

__device__
inline
unsigned int atomicDec(unsigned int* address, unsigned int val)
{
  return __builtin_amdgcn_atomic_dec32(address, val, __ATOMIC_RELAXED, "agent");
}

__device__
inline
int atomicAnd(int* address, int val)
{
    return __atomic_fetch_and(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicAnd(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_and(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicAnd(
    unsigned long long* address, unsigned long long val)
{
    return __atomic_fetch_and(address, val, __ATOMIC_RELAXED);
}

__device__
inline
int atomicOr(int* address, int val)
{
    return __atomic_fetch_or(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicOr(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_or(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicOr(
    unsigned long long* address, unsigned long long val)
{
    return __atomic_fetch_or(address, val, __ATOMIC_RELAXED);
}

__device__
inline
int atomicXor(int* address, int val)
{
    return __atomic_fetch_xor(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicXor(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_xor(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicXor(
    unsigned long long* address, unsigned long long val)
{
    return __atomic_fetch_xor(address, val, __ATOMIC_RELAXED);
}

#endif // __hip_atomic_compare_exchange_strong
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#if !defined(__HIPCC_RTC__)
#include "host_defines.h"
#include "amd_hip_vector_types.h"  // For Native_vec_
#endif

#if defined(__cplusplus)
    extern "C" {
#endif

// DOT FUNCTIONS
#if defined(__clang__) && defined(__HIP__)
__device__
__attribute__((const))
int __ockl_sdot2(
    HIP_vector_base<short, 2>::Native_vec_,
    HIP_vector_base<short, 2>::Native_vec_,
    int, bool);

__device__
__attribute__((const))
unsigned int __ockl_udot2(
    HIP_vector_base<unsigned short, 2>::Native_vec_,
    HIP_vector_base<unsigned short, 2>::Native_vec_,
    unsigned int, bool);

__device__
__attribute__((const))
int __ockl_sdot4(
    HIP_vector_base<char, 4>::Native_vec_,
    HIP_vector_base<char, 4>::Native_vec_,
    int, bool);

__device__
__attribute__((const))
unsigned int __ockl_udot4(
    HIP_vector_base<unsigned char, 4>::Native_vec_,
    HIP_vector_base<unsigned char, 4>::Native_vec_,
    unsigned int, bool);

__device__
__attribute__((const))
int __ockl_sdot8(int, int, int, bool);

__device__
__attribute__((const))
unsigned int __ockl_udot8(unsigned int, unsigned int, unsigned int, bool);
#endif

#if !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
// BEGIN FLOAT
__device__
__attribute__((const))
float __ocml_acos_f32(float);
__device__
__attribute__((pure))
float __ocml_acosh_f32(float);
__device__
__attribute__((const))
float __ocml_asin_f32(float);
__device__
__attribute__((pure))
float __ocml_asinh_f32(float);
__device__
__attribute__((const))
float __ocml_atan2_f32(float, float);
__device__
__attribute__((const))
float __ocml_atan_f32(float);
__device__
__attribute__((pure))
float __ocml_atanh_f32(float);
__device__
__attribute__((pure))
float __ocml_cbrt_f32(float);
__device__
__attribute__((const))
float __ocml_ceil_f32(float);
__device__
__attribute__((const))
__device__
float __ocml_copysign_f32(float, float);
__device__
float __ocml_cos_f32(float);
__device__
float __ocml_native_cos_f32(float);
__device__
__attribute__((pure))
__device__
float __ocml_cosh_f32(float);
__device__
float __ocml_cospi_f32(float);
__device__
float __ocml_i0_f32(float);
__device__
float __ocml_i1_f32(float);
__device__
__attribute__((pure))
float __ocml_erfc_f32(float);
__device__
__attribute__((pure))
float __ocml_erfcinv_f32(float);
__device__
__attribute__((pure))
float __ocml_erfcx_f32(float);
__device__
__attribute__((pure))
float __ocml_erf_f32(float);
__device__
__attribute__((pure))
float __ocml_erfinv_f32(float);
__device__
__attribute__((pure))
float __ocml_exp10_f32(float);
__device__
__attribute__((pure))
float __ocml_native_exp10_f32(float);
__device__
__attribute__((pure))
float __ocml_exp2_f32(float);
__device__
__attribute__((pure))
float __ocml_exp_f32(float);
__device__
__attribute__((pure))
float __ocml_native_exp_f32(float);
__device__
__attribute__((pure))
float __ocml_expm1_f32(float);
__device__
__attribute__((const))
float __ocml_fabs_f32(float);
__device__
__attribute__((const))
float __ocml_fdim_f32(float, float);
__device__
__attribute__((const))
float __ocml_floor_f32(float);
__device__
__attribute__((const))
float __ocml_fma_f32(float, float, float);
__device__
__attribute__((const))
float __ocml_fmax_f32(float, float);
__device__
__attribute__((const))
float __ocml_fmin_f32(float, float);
__device__
__attribute__((const))
__device__
float __ocml_fmod_f32(float, float);
__device__
float __ocml_frexp_f32(float, __attribute__((address_space(5))) int*);
__device__
__attribute__((const))
float __ocml_hypot_f32(float, float);
__device__
__attribute__((const))
int __ocml_ilogb_f32(float);
__device__
__attribute__((const))
int __ocml_isfinite_f32(float);
__device__
__attribute__((const))
int __ocml_isinf_f32(float);
__device__
__attribute__((const))
int __ocml_isnan_f32(float);
__device__
float __ocml_j0_f32(float);
__device__
float __ocml_j1_f32(float);
__device__
__attribute__((const))
float __ocml_ldexp_f32(float, int);
__device__
float __ocml_lgamma_f32(float);
__device__
__attribute__((pure))
float __ocml_log10_f32(float);
__device__
__attribute__((pure))
float __ocml_native_log10_f32(float);
__device__
__attribute__((pure))
float __ocml_log1p_f32(float);
__device__
__attribute__((pure))
float __ocml_log2_f32(float);
__device__
__attribute__((pure))
float __ocml_native_log2_f32(float);
__device__
__attribute__((const))
float __ocml_logb_f32(float);
__device__
__attribute__((pure))
float __ocml_log_f32(float);
__device__
__attribute__((pure))
float __ocml_native_log_f32(float);
__device__
float __ocml_modf_f32(float, __attribute__((address_space(5))) float*);
__device__
__attribute__((const))
float __ocml_nearbyint_f32(float);
__device__
__attribute__((const))
float __ocml_nextafter_f32(float, float);
__device__
__attribute__((const))
float __ocml_len3_f32(float, float, float);
__device__
__attribute__((const))
float __ocml_len4_f32(float, float, float, float);
__device__
__attribute__((pure))
float __ocml_ncdf_f32(float);
__device__
__attribute__((pure))
float __ocml_ncdfinv_f32(float);
__device__
__attribute__((pure))
float __ocml_pow_f32(float, float);
__device__
__attribute__((pure))
float __ocml_pown_f32(float, int);
__device__
__attribute__((pure))
float __ocml_rcbrt_f32(float);
__device__
__attribute__((const))
float __ocml_remainder_f32(float, float);
__device__
float __ocml_remquo_f32(float, float, __attribute__((address_space(5))) int*);
__device__
__attribute__((const))
float __ocml_rhypot_f32(float, float);
__device__
__attribute__((const))
float __ocml_rint_f32(float);
__device__
__attribute__((const))
float __ocml_rlen3_f32(float, float, float);
__device__
__attribute__((const))
float __ocml_rlen4_f32(float, float, float, float);
__device__
__attribute__((const))
float __ocml_round_f32(float);
__device__
__attribute__((pure))
float __ocml_rsqrt_f32(float);
__device__
__attribute__((const))
float __ocml_scalb_f32(float, float);
__device__
__attribute__((const))
float __ocml_scalbn_f32(float, int);
__device__
__attribute__((const))
int __ocml_signbit_f32(float);
__device__
float __ocml_sincos_f32(float, __attribute__((address_space(5))) float*);
__device__
float __ocml_sincospi_f32(float, __attribute__((address_space(5))) float*);
__device__
float __ocml_sin_f32(float);
__device__
float __ocml_native_sin_f32(float);
__device__
__attribute__((pure))
float __ocml_sinh_f32(float);
__device__
float __ocml_sinpi_f32(float);
__device__
__attribute__((const))
float __ocml_sqrt_f32(float);
__device__
__attribute__((const))
float __ocml_native_sqrt_f32(float);
__device__
float __ocml_tan_f32(float);
__device__
__attribute__((pure))
float __ocml_tanh_f32(float);
__device__
float __ocml_tgamma_f32(float);
__device__
__attribute__((const))
float __ocml_trunc_f32(float);
__device__
float __ocml_y0_f32(float);
__device__
float __ocml_y1_f32(float);

// BEGIN INTRINSICS
__device__
__attribute__((const))
float __ocml_add_rte_f32(float, float);
__device__
__attribute__((const))
float __ocml_add_rtn_f32(float, float);
__device__
__attribute__((const))
float __ocml_add_rtp_f32(float, float);
__device__
__attribute__((const))
float __ocml_add_rtz_f32(float, float);
__device__
__attribute__((const))
float __ocml_sub_rte_f32(float, float);
__device__
__attribute__((const))
float __ocml_sub_rtn_f32(float, float);
__device__
__attribute__((const))
float __ocml_sub_rtp_f32(float, float);
__device__
__attribute__((const))
float __ocml_sub_rtz_f32(float, float);
__device__
__attribute__((const))
float __ocml_mul_rte_f32(float, float);
__device__
__attribute__((const))
float __ocml_mul_rtn_f32(float, float);
__device__
__attribute__((const))
float __ocml_mul_rtp_f32(float, float);
__device__
__attribute__((const))
float __ocml_mul_rtz_f32(float, float);
__device__
__attribute__((const))
float __ocml_div_rte_f32(float, float);
__device__
__attribute__((const))
float __ocml_div_rtn_f32(float, float);
__device__
__attribute__((const))
float __ocml_div_rtp_f32(float, float);
__device__
__attribute__((const))
float __ocml_div_rtz_f32(float, float);
__device__
__attribute__((const))
float __ocml_sqrt_rte_f32(float);
__device__
__attribute__((const))
float __ocml_sqrt_rtn_f32(float);
__device__
__attribute__((const))
float __ocml_sqrt_rtp_f32(float);
__device__
__attribute__((const))
float __ocml_sqrt_rtz_f32(float);
__device__
__attribute__((const))
float __ocml_fma_rte_f32(float, float, float);
__device__
__attribute__((const))
float __ocml_fma_rtn_f32(float, float, float);
__device__
__attribute__((const))
float __ocml_fma_rtp_f32(float, float, float);
__device__
__attribute__((const))
float __ocml_fma_rtz_f32(float, float, float);
// END INTRINSICS
// END FLOAT

// BEGIN DOUBLE
__device__
__attribute__((const))
double __ocml_acos_f64(double);
__device__
__attribute__((pure))
double __ocml_acosh_f64(double);
__device__
__attribute__((const))
double __ocml_asin_f64(double);
__device__
__attribute__((pure))
double __ocml_asinh_f64(double);
__device__
__attribute__((const))
double __ocml_atan2_f64(double, double);
__device__
__attribute__((const))
double __ocml_atan_f64(double);
__device__
__attribute__((pure))
double __ocml_atanh_f64(double);
__device__
__attribute__((pure))
double __ocml_cbrt_f64(double);
__device__
__attribute__((const))
double __ocml_ceil_f64(double);
__device__
__attribute__((const))
double __ocml_copysign_f64(double, double);
__device__
double __ocml_cos_f64(double);
__device__
__attribute__((pure))
double __ocml_cosh_f64(double);
__device__
double __ocml_cospi_f64(double);
__device__
double __ocml_i0_f64(double);
__device__
double __ocml_i1_f64(double);
__device__
__attribute__((pure))
double __ocml_erfc_f64(double);
__device__
__attribute__((pure))
double __ocml_erfcinv_f64(double);
__device__
__attribute__((pure))
double __ocml_erfcx_f64(double);
__device__
__attribute__((pure))
double __ocml_erf_f64(double);
__device__
__attribute__((pure))
double __ocml_erfinv_f64(double);
__device__
__attribute__((pure))
double __ocml_exp10_f64(double);
__device__
__attribute__((pure))
double __ocml_exp2_f64(double);
__device__
__attribute__((pure))
double __ocml_exp_f64(double);
__device__
__attribute__((pure))
double __ocml_expm1_f64(double);
__device__
__attribute__((const))
double __ocml_fabs_f64(double);
__device__
__attribute__((const))
double __ocml_fdim_f64(double, double);
__device__
__attribute__((const))
double __ocml_floor_f64(double);
__device__
__attribute__((const))
double __ocml_fma_f64(double, double, double);
__device__
__attribute__((const))
double __ocml_fmax_f64(double, double);
__device__
__attribute__((const))
double __ocml_fmin_f64(double, double);
__device__
__attribute__((const))
double __ocml_fmod_f64(double, double);
__device__
double __ocml_frexp_f64(double, __attribute__((address_space(5))) int*);
__device__
__attribute__((const))
double __ocml_hypot_f64(double, double);
__device__
__attribute__((const))
int __ocml_ilogb_f64(double);
__device__
__attribute__((const))
int __ocml_isfinite_f64(double);
__device__
__attribute__((const))
int __ocml_isinf_f64(double);
__device__
__attribute__((const))
int __ocml_isnan_f64(double);
__device__
double __ocml_j0_f64(double);
__device__
double __ocml_j1_f64(double);
__device__
__attribute__((const))
double __ocml_ldexp_f64(double, int);
__device__
double __ocml_lgamma_f64(double);
__device__
__attribute__((pure))
double __ocml_log10_f64(double);
__device__
__attribute__((pure))
double __ocml_log1p_f64(double);
__device__
__attribute__((pure))
double __ocml_log2_f64(double);
__device__
__attribute__((const))
double __ocml_logb_f64(double);
__device__
__attribute__((pure))
double __ocml_log_f64(double);
__device__
double __ocml_modf_f64(double, __attribute__((address_space(5))) double*);
__device__
__attribute__((const))
double __ocml_nearbyint_f64(double);
__device__
__attribute__((const))
double __ocml_nextafter_f64(double, double);
__device__
__attribute__((const))
double __ocml_len3_f64(double, double, double);
__device__
__attribute__((const))
double __ocml_len4_f64(double, double, double, double);
__device__
__attribute__((pure))
double __ocml_ncdf_f64(double);
__device__
__attribute__((pure))
double __ocml_ncdfinv_f64(double);
__device__
__attribute__((pure))
double __ocml_pow_f64(double, double);
__device__
__attribute__((pure))
double __ocml_pown_f64(double, int);
__device__
__attribute__((pure))
double __ocml_rcbrt_f64(double);
__device__
__attribute__((const))
double __ocml_remainder_f64(double, double);
__device__
double __ocml_remquo_f64(
    double, double, __attribute__((address_space(5))) int*);
__device__
__attribute__((const))
double __ocml_rhypot_f64(double, double);
__device__
__attribute__((const))
double __ocml_rint_f64(double);
__device__
__attribute__((const))
double __ocml_rlen3_f64(double, double, double);
__device__
__attribute__((const))
double __ocml_rlen4_f64(double, double, double, double);
__device__
__attribute__((const))
double __ocml_round_f64(double);
__device__
__attribute__((pure))
double __ocml_rsqrt_f64(double);
__device__
__attribute__((const))
double __ocml_scalb_f64(double, double);
__device__
__attribute__((const))
double __ocml_scalbn_f64(double, int);
__device__
__attribute__((const))
int __ocml_signbit_f64(double);
__device__
double __ocml_sincos_f64(double, __attribute__((address_space(5))) double*);
__device__
double __ocml_sincospi_f64(double, __attribute__((address_space(5))) double*);
__device__
double __ocml_sin_f64(double);
__device__
__attribute__((pure))
double __ocml_sinh_f64(double);
__device__
double __ocml_sinpi_f64(double);
__device__
__attribute__((const))
double __ocml_sqrt_f64(double);
__device__
double __ocml_tan_f64(double);
__device__
__attribute__((pure))
double __ocml_tanh_f64(double);
__device__
double __ocml_tgamma_f64(double);
__device__
__attribute__((const))
double __ocml_trunc_f64(double);
__device__
double __ocml_y0_f64(double);
__device__
double __ocml_y1_f64(double);

// BEGIN INTRINSICS
__device__
__attribute__((const))
double __ocml_add_rte_f64(double, double);
__device__
__attribute__((const))
double __ocml_add_rtn_f64(double, double);
__device__
__attribute__((const))
double __ocml_add_rtp_f64(double, double);
__device__
__attribute__((const))
double __ocml_add_rtz_f64(double, double);
__device__
__attribute__((const))
double __ocml_sub_rte_f64(double, double);
__device__
__attribute__((const))
double __ocml_sub_rtn_f64(double, double);
__device__
__attribute__((const))
double __ocml_sub_rtp_f64(double, double);
__device__
__attribute__((const))
double __ocml_sub_rtz_f64(double, double);
__device__
__attribute__((const))
double __ocml_mul_rte_f64(double, double);
__device__
__attribute__((const))
double __ocml_mul_rtn_f64(double, double);
__device__
__attribute__((const))
double __ocml_mul_rtp_f64(double, double);
__device__
__attribute__((const))
double __ocml_mul_rtz_f64(double, double);
__device__
__attribute__((const))
double __ocml_div_rte_f64(double, double);
__device__
__attribute__((const))
double __ocml_div_rtn_f64(double, double);
__device__
__attribute__((const))
double __ocml_div_rtp_f64(double, double);
__device__
__attribute__((const))
double __ocml_div_rtz_f64(double, double);
__device__
__attribute__((const))
double __ocml_sqrt_rte_f64(double);
__device__
__attribute__((const))
double __ocml_sqrt_rtn_f64(double);
__device__
__attribute__((const))
double __ocml_sqrt_rtp_f64(double);
__device__
__attribute__((const))
double __ocml_sqrt_rtz_f64(double);
__device__
__attribute__((const))
double __ocml_fma_rte_f64(double, double, double);
__device__
__attribute__((const))
double __ocml_fma_rtn_f64(double, double, double);
__device__
__attribute__((const))
double __ocml_fma_rtp_f64(double, double, double);
__device__
__attribute__((const))
double __ocml_fma_rtz_f64(double, double, double);
// END INTRINSICS
// END DOUBLE

#endif // !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__

#if defined(__cplusplus)
    } // extern "C"
#endif
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

// /*
// Half Math Functions
// */
#if !defined(__HIPCC_RTC__)
#include "host_defines.h"
#endif
#ifndef __CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
extern "C"
{
    __device__ __attribute__((const)) _Float16 __ocml_ceil_f16(_Float16);
    __device__ _Float16 __ocml_cos_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_exp_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_exp10_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_exp2_f16(_Float16);
    __device__ __attribute__((const)) _Float16 __ocml_floor_f16(_Float16);
    __device__ __attribute__((const))
    _Float16 __ocml_fma_f16(_Float16, _Float16, _Float16);
    __device__ __attribute__((const)) _Float16 __ocml_fabs_f16(_Float16);
    __device__ __attribute__((const)) int __ocml_isinf_f16(_Float16);
    __device__ __attribute__((const)) int __ocml_isnan_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_log_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_log10_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_log2_f16(_Float16);
    __device__ __attribute__((pure)) _Float16 __ocml_pown_f16(_Float16, int);
    __device__ __attribute__((const)) _Float16 __ocml_rint_f16(_Float16);
    __device__ __attribute__((const)) _Float16 __ocml_rsqrt_f16(_Float16);
    __device__ _Float16 __ocml_sin_f16(_Float16);
    __device__ __attribute__((const)) _Float16 __ocml_sqrt_f16(_Float16);
    __device__ __attribute__((const)) _Float16 __ocml_trunc_f16(_Float16);
    __device__ __attribute__((const)) _Float16 __ocml_fmax_f16(_Float16, _Float16);
    __device__ __attribute__((const)) _Float16 __ocml_fmin_f16(_Float16, _Float16);

    typedef _Float16 __2f16 __attribute__((ext_vector_type(2)));
    typedef short __2i16 __attribute__((ext_vector_type(2)));

    #if defined(__clang__) && defined(__HIP__)
    __device__ __attribute__((const)) float __ockl_fdot2(__2f16 a, __2f16 b, float c, bool s);
    #endif

    __device__ __attribute__((const)) __2f16 __ocml_ceil_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_fabs_2f16(__2f16);
    __device__ __2f16 __ocml_cos_2f16(__2f16);
    __device__ __attribute__((pure)) __2f16 __ocml_exp_2f16(__2f16);
    __device__ __attribute__((pure)) __2f16 __ocml_exp10_2f16(__2f16);
    __device__ __attribute__((pure)) __2f16 __ocml_exp2_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_floor_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_fma_2f16(__2f16, __2f16, __2f16);
    __device__ __attribute__((const)) __2i16 __ocml_isinf_2f16(__2f16);
    __device__ __attribute__((const)) __2i16 __ocml_isnan_2f16(__2f16);
    __device__ __attribute__((pure)) __2f16 __ocml_log_2f16(__2f16);
    __device__ __attribute__((pure)) __2f16 __ocml_log10_2f16(__2f16);
    __device__ __attribute__((pure)) __2f16 __ocml_log2_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_rint_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_rsqrt_2f16(__2f16);
    __device__ __2f16 __ocml_sin_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_sqrt_2f16(__2f16);
    __device__ __attribute__((const)) __2f16 __ocml_trunc_2f16(__2f16);

    __device__ __attribute__((const)) _Float16 __ocml_cvtrtn_f16_f32(float);
    __device__ __attribute__((const)) _Float16 __ocml_cvtrtp_f16_f32(float);
    __device__ __attribute__((const)) _Float16 __ocml_cvtrtz_f16_f32(float);

}
#endif // !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
//TODO: remove these after they get into clang header __clang_hip_libdevice_declares.h'
extern "C" {
    __device__ __attribute__((const)) _Float16 __ocml_fmax_f16(_Float16, _Float16);
    __device__ __attribute__((const)) _Float16 __ocml_fmin_f16(_Float16, _Float16);
    __device__ __attribute__((const)) _Float16 __ocml_cvtrtn_f16_f32(float);
    __device__ __attribute__((const)) _Float16 __ocml_cvtrtp_f16_f32(float);
    __device__ __attribute__((const)) _Float16 __ocml_cvtrtz_f16_f32(float);
}
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once
#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP16_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP16_H

#if defined(__HIPCC_RTC__)
  #define __HOST_DEVICE__ __device__
#else
  #define __HOST_DEVICE__ __host__ __device__
  #include <hip/amd_detail/amd_hip_common.h>
  #include "hip/amd_detail/host_defines.h"
  #include <assert.h>
  #if defined(__cplusplus)
    #include <algorithm>
    #include <type_traits>
    #include <utility>
#endif
#endif // !defined(__HIPCC_RTC__)

#if defined(__clang__) && defined(__HIP__)
    typedef _Float16 _Float16_2 __attribute__((ext_vector_type(2)));

    struct __half_raw {
        union {
            static_assert(sizeof(_Float16) == sizeof(unsigned short), "");

            _Float16 data;
            unsigned short x;
        };
    };

    struct __half2_raw {
        union {
            static_assert(sizeof(_Float16_2) == sizeof(unsigned short[2]), "");

            struct {
                __half_raw x;
                __half_raw y;
            };
            _Float16_2 data;
        };
    };

    #if defined(__cplusplus)
      #if !defined(__HIPCC_RTC__)
        #include "hip_fp16_math_fwd.h"
        #include "amd_hip_vector_types.h"
        #include "host_defines.h"
        #include "amd_device_functions.h"
        #include "amd_warp_functions.h"
      #endif
        namespace std
        {
            template<> struct is_floating_point<_Float16> : std::true_type {};
        }

        template<bool cond, typename T = void>
        using Enable_if_t = typename std::enable_if<cond, T>::type;

        // BEGIN STRUCT __HALF
        struct __half {
        protected:
            union {
                static_assert(sizeof(_Float16) == sizeof(unsigned short), "");

                _Float16 data;
                unsigned short __x;
            };
        public:
            // CREATORS
            __HOST_DEVICE__
            __half() = default;
            __HOST_DEVICE__
            __half(const __half_raw& x) : data{x.data} {}
            #if !defined(__HIP_NO_HALF_CONVERSIONS__)
                __HOST_DEVICE__
                __half(decltype(data) x) : data{x} {}
                template<
                    typename T,
                    Enable_if_t<std::is_floating_point<T>{}>* = nullptr>
                __HOST_DEVICE__
                __half(T x) : data{static_cast<_Float16>(x)} {}
            #endif
            __HOST_DEVICE__
            __half(const __half&) = default;
            __HOST_DEVICE__
            __half(__half&&) = default;
            __HOST_DEVICE__
            ~__half() = default;

            // CREATORS - DEVICE ONLY
            #if !defined(__HIP_NO_HALF_CONVERSIONS__)
                template<
                    typename T, Enable_if_t<std::is_integral<T>{}>* = nullptr>
                __HOST_DEVICE__
                __half(T x) : data{static_cast<_Float16>(x)} {}
            #endif

            // MANIPULATORS
            __HOST_DEVICE__
            __half& operator=(const __half&) = default;
            __HOST_DEVICE__
            __half& operator=(__half&&) = default;
            __HOST_DEVICE__
            __half& operator=(const __half_raw& x)
            {
                data = x.data;
                return *this;
            }
            __HOST_DEVICE__
            volatile __half& operator=(const __half_raw& x) volatile
            {
                data = x.data;
                return *this;
            }
            volatile __half& operator=(const volatile __half_raw& x) volatile
            {
                data = x.data;
                return *this;
            }
            __half& operator=(__half_raw&& x)
            {
                data = x.data;
                return *this;
            }
            volatile __half& operator=(__half_raw&& x) volatile
            {
                data = x.data;
                return *this;
            }
            volatile __half& operator=(volatile __half_raw&& x) volatile
            {
                data = x.data;
                return *this;
            }
            #if !defined(__HIP_NO_HALF_CONVERSIONS__)
                template<
                    typename T,
                    Enable_if_t<std::is_floating_point<T>{}>* = nullptr>
                __HOST_DEVICE__
                __half& operator=(T x)
                {
                    data = static_cast<_Float16>(x);
                    return *this;
                }
            #endif

            // MANIPULATORS - DEVICE ONLY
            #if !defined(__HIP_NO_HALF_CONVERSIONS__)
                template<
                    typename T, Enable_if_t<std::is_integral<T>{}>* = nullptr>
                __device__
                __half& operator=(T x)
                {
                    data = static_cast<_Float16>(x);
                    return *this;
                }
            #endif

            #if !defined(__HIP_NO_HALF_OPERATORS__)
                __device__
                __half& operator+=(const __half& x)
                {
                    data += x.data;
                    return *this;
                }
                __device__
                __half& operator-=(const __half& x)
                {
                    data -= x.data;
                    return *this;
                }
                __device__
                __half& operator*=(const __half& x)
                {
                    data *= x.data;
                    return *this;
                }
                __device__
                __half& operator/=(const __half& x)
                {
                    data /= x.data;
                    return *this;
                }
                __device__
                __half& operator++() { ++data; return *this; }
                __device__
                __half operator++(int)
                {
                    __half tmp{*this};
                    ++*this;
                    return tmp;
                }
                __device__
                __half& operator--() { --data; return *this; }
                __device__
                __half operator--(int)
                {
                    __half tmp{*this};
                    --*this;
                    return tmp;
                }
            #endif

            // ACCESSORS
            #if !defined(__HIP_NO_HALF_CONVERSIONS__)
                template<
                    typename T,
                    Enable_if_t<std::is_floating_point<T>{}>* = nullptr>
                __HOST_DEVICE__
                operator T() const { return data; }
            #endif
            __HOST_DEVICE__
            operator __half_raw() const { return __half_raw{data}; }
            __HOST_DEVICE__
            operator __half_raw() const volatile
            {
                return __half_raw{data};
            }

            #if !defined(__HIP_NO_HALF_CONVERSIONS__)
                template<
                    typename T, Enable_if_t<std::is_integral<T>{}>* = nullptr>
                __HOST_DEVICE__
                operator T() const { return data; }
            #endif

            #if !defined(__HIP_NO_HALF_OPERATORS__)
                __device__
                __half operator+() const { return *this; }
                __device__
                __half operator-() const
                {
                    __half tmp{*this};
                    tmp.data = -tmp.data;
                    return tmp;
                }
            #endif

            // FRIENDS
            #if !defined(__HIP_NO_HALF_OPERATORS__)
                friend
                inline
                __device__
                __half operator+(const __half& x, const __half& y)
                {
                    return __half{x} += y;
                }
                friend
                inline
                __device__
                __half operator-(const __half& x, const __half& y)
                {
                    return __half{x} -= y;
                }
                friend
                inline
                __device__
                __half operator*(const __half& x, const __half& y)
                {
                    return __half{x} *= y;
                }
                friend
                inline
                __device__
                __half operator/(const __half& x, const __half& y)
                {
                    return __half{x} /= y;
                }
                friend
                inline
                __device__
                bool operator==(const __half& x, const __half& y)
                {
                    return x.data == y.data;
                }
                friend
                inline
                __device__
                bool operator!=(const __half& x, const __half& y)
                {
                    return !(x == y);
                }
                friend
                inline
                __device__
                bool operator<(const __half& x, const __half& y)
                {
                    return x.data < y.data;
                }
                friend
                inline
                __device__
                bool operator>(const __half& x, const __half& y)
                {
                    return y.data < x.data;
                }
                friend
                inline
                __device__
                bool operator<=(const __half& x, const __half& y)
                {
                    return !(y < x);
                }
                friend
                inline
                __device__
                bool operator>=(const __half& x, const __half& y)
                {
                    return !(x < y);
                }
            #endif // !defined(__HIP_NO_HALF_OPERATORS__)
        };
        // END STRUCT __HALF

        // BEGIN STRUCT __HALF2
        struct __half2 {
        public:
            union {
                static_assert(
                    sizeof(_Float16_2) == sizeof(unsigned short[2]), "");

                struct {
                    __half x;
                    __half y;
                };
                _Float16_2 data;
            };

            // CREATORS
            __HOST_DEVICE__
            __half2() = default;
            __HOST_DEVICE__
            __half2(const __half2_raw& xx) : data{xx.data} {}
            __HOST_DEVICE__
            __half2(decltype(data) xx) : data{xx} {}
            __HOST_DEVICE__
            __half2(const __half& xx, const __half& yy)
                :
                data{static_cast<__half_raw>(xx).data,
                     static_cast<__half_raw>(yy).data}
            {}
            __HOST_DEVICE__
            __half2(const __half2&) = default;
            __HOST_DEVICE__
            __half2(__half2&&) = default;
            __HOST_DEVICE__
            ~__half2() = default;

            // MANIPULATORS
            __HOST_DEVICE__
            __half2& operator=(const __half2&) = default;
            __HOST_DEVICE__
            __half2& operator=(__half2&&) = default;
            __HOST_DEVICE__
            __half2& operator=(const __half2_raw& xx)
            {
                data = xx.data;
                return *this;
            }

            // MANIPULATORS - DEVICE ONLY
            #if !defined(__HIP_NO_HALF_OPERATORS__)
                __device__
                __half2& operator+=(const __half2& xx)
                {
                    data += xx.data;
                    return *this;
                }
                __device__
                __half2& operator-=(const __half2& xx)
                {
                    data -= xx.data;
                    return *this;
                }
                __device__
                __half2& operator*=(const __half2& xx)
                {
                    data *= xx.data;
                    return *this;
                }
                __device__
                __half2& operator/=(const __half2& xx)
                {
                    data /= xx.data;
                    return *this;
                }
                __device__
                __half2& operator++() { return *this += _Float16_2{1, 1}; }
                __device__
                __half2 operator++(int)
                {
                    __half2 tmp{*this};
                    ++*this;
                    return tmp;
                }
                __device__
                __half2& operator--() { return *this -= _Float16_2{1, 1}; }
                __device__
                __half2 operator--(int)
                {
                    __half2 tmp{*this};
                    --*this;
                    return tmp;
                }
            #endif

            // ACCESSORS
            __HOST_DEVICE__
            operator decltype(data)() const { return data; }
            __HOST_DEVICE__
            operator __half2_raw() const {
              __half2_raw r;
              r.data = data;
              return r;
            }

            // ACCESSORS - DEVICE ONLY
            #if !defined(__HIP_NO_HALF_OPERATORS__)
                __device__
                __half2 operator+() const { return *this; }
                __device__
                __half2 operator-() const
                {
                    __half2 tmp{*this};
                    tmp.data = -tmp.data;
                    return tmp;
                }
            #endif

            // FRIENDS
            #if !defined(__HIP_NO_HALF_OPERATORS__)
                friend
                inline
                __device__
                __half2 operator+(const __half2& xx, const __half2& yy)
                {
                    return __half2{xx} += yy;
                }
                friend
                inline
                __device__
                __half2 operator-(const __half2& xx, const __half2& yy)
                {
                    return __half2{xx} -= yy;
                }
                friend
                inline
                __device__
                __half2 operator*(const __half2& xx, const __half2& yy)
                {
                    return __half2{xx} *= yy;
                }
                friend
                inline
                __device__
                __half2 operator/(const __half2& xx, const __half2& yy)
                {
                    return __half2{xx} /= yy;
                }
                friend
                inline
                __device__
                bool operator==(const __half2& xx, const __half2& yy)
                {
                    auto r = xx.data == yy.data;
                    return r.x != 0 && r.y != 0;
                }
                friend
                inline
                __device__
                bool operator!=(const __half2& xx, const __half2& yy)
                {
                    return !(xx == yy);
                }
                friend
                inline
                __device__
                bool operator<(const __half2& xx, const __half2& yy)
                {
                    auto r = xx.data < yy.data;
                    return r.x != 0 && r.y != 0;
                }
                friend
                inline
                __device__
                bool operator>(const __half2& xx, const __half2& yy)
                {
                    return yy < xx;
                }
                friend
                inline
                __device__
                bool operator<=(const __half2& xx, const __half2& yy)
                {
                    return !(yy < xx);
                }
                friend
                inline
                __device__
                bool operator>=(const __half2& xx, const __half2& yy)
                {
                    return !(xx < yy);
                }
            #endif // !defined(__HIP_NO_HALF_OPERATORS__)
        };
        // END STRUCT __HALF2

        namespace
        {
            inline
            __HOST_DEVICE__
            __half2 make_half2(__half x, __half y)
            {
                return __half2{x, y};
            }

            inline
            __HOST_DEVICE__
            __half __low2half(__half2 x)
            {
                return __half{__half_raw{static_cast<__half2_raw>(x).data.x}};
            }

            inline
            __HOST_DEVICE__
            __half __high2half(__half2 x)
            {
                return __half{__half_raw{static_cast<__half2_raw>(x).data.y}};
            }

            inline
            __HOST_DEVICE__
            __half2 __half2half2(__half x)
            {
                return __half2{x, x};
            }

            inline
            __HOST_DEVICE__
            __half2 __halves2half2(__half x, __half y)
            {
                return __half2{x, y};
            }

            inline
            __HOST_DEVICE__
            __half2 __low2half2(__half2 x)
            {
                return __half2{
                    _Float16_2{
                        static_cast<__half2_raw>(x).data.x,
                        static_cast<__half2_raw>(x).data.x}};
            }

            inline
            __HOST_DEVICE__
            __half2 __high2half2(__half2 x)
            {
                return __half2{
                    _Float16_2{
                        static_cast<__half2_raw>(x).data.y,
                        static_cast<__half2_raw>(x).data.y}};
            }

            inline
            __HOST_DEVICE__
            __half2 __lows2half2(__half2 x, __half2 y)
            {
                return __half2{
                    _Float16_2{
                        static_cast<__half2_raw>(x).data.x,
                        static_cast<__half2_raw>(y).data.x}};
            }

            inline
            __HOST_DEVICE__
            __half2 __highs2half2(__half2 x, __half2 y)
            {
                return __half2{
                    _Float16_2{
                        static_cast<__half2_raw>(x).data.y,
                        static_cast<__half2_raw>(y).data.y}};
            }

            inline
            __HOST_DEVICE__
            __half2 __lowhigh2highlow(__half2 x)
            {
                return __half2{
                    _Float16_2{
                        static_cast<__half2_raw>(x).data.y,
                        static_cast<__half2_raw>(x).data.x}};
            }

            // Bitcasts
            inline
            __device__
            short __half_as_short(__half x)
            {
                return static_cast<__half_raw>(x).x;
            }

            inline
            __device__
            unsigned short __half_as_ushort(__half x)
            {
                return static_cast<__half_raw>(x).x;
            }

            inline
            __device__
            __half __short_as_half(short x)
            {
                __half_raw r; r.x = x;
                return r;
            }

            inline
            __device__
            __half __ushort_as_half(unsigned short x)
            {
                __half_raw r; r.x = x;
                return r;
            }

            // float -> half | half2
            inline
            __HOST_DEVICE__
            __half __float2half(float x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __HOST_DEVICE__
            __half __float2half_rn(float x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            #if !defined(__HIPCC_RTC__)
            // TODO: rounding behaviour is not correct for host functions.
            inline
            __host__
            __half __float2half_rz(float x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __host__
            __half __float2half_rd(float x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __host__
            __half __float2half_ru(float x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            #endif
            inline
            __device__
            __half __float2half_rz(float x)
            {
                return __half_raw{__ocml_cvtrtz_f16_f32(x)};
            }
            inline
            __device__
            __half __float2half_rd(float x)
            {
                return __half_raw{__ocml_cvtrtn_f16_f32(x)};
            }
            inline
            __device__
            __half __float2half_ru(float x)
            {
                return __half_raw{__ocml_cvtrtp_f16_f32(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 __float2half2_rn(float x)
            {
                return __half2{
                    _Float16_2{
                        static_cast<_Float16>(x), static_cast<_Float16>(x)}};
            }
            inline
            __HOST_DEVICE__
            __half2 __floats2half2_rn(float x, float y)
            {
                return __half2{_Float16_2{
                    static_cast<_Float16>(x), static_cast<_Float16>(y)}};
            }
            inline
            __HOST_DEVICE__
            __half2 __float22half2_rn(float2 x)
            {
                return __floats2half2_rn(x.x, x.y);
            }

            // half | half2 -> float
            inline
            __HOST_DEVICE__
            float __half2float(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __HOST_DEVICE__
            float __low2float(__half2 x)
            {
                return static_cast<__half2_raw>(x).data.x;
            }
            inline
            __HOST_DEVICE__
            float __high2float(__half2 x)
            {
                return static_cast<__half2_raw>(x).data.y;
            }
            inline
            __HOST_DEVICE__
            float2 __half22float2(__half2 x)
            {
                return make_float2(
                    static_cast<__half2_raw>(x).data.x,
                    static_cast<__half2_raw>(x).data.y);
            }

            // half -> int
            inline
            __device__
            int __half2int_rn(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            int __half2int_rz(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            int __half2int_rd(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            int __half2int_ru(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }

            // int -> half
            inline
            __device__
            __half __int2half_rn(int x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __int2half_rz(int x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __int2half_rd(int x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __int2half_ru(int x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }

            // half -> short
            inline
            __device__
            short __half2short_rn(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            short __half2short_rz(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            short __half2short_rd(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            short __half2short_ru(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }

            // short -> half
            inline
            __device__
            __half __short2half_rn(short x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __short2half_rz(short x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __short2half_rd(short x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __short2half_ru(short x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }

            // half -> long long
            inline
            __device__
            long long __half2ll_rn(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            long long __half2ll_rz(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            long long __half2ll_rd(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            long long __half2ll_ru(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }

            // long long -> half
            inline
            __device__
            __half __ll2half_rn(long long x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __ll2half_rz(long long x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __ll2half_rd(long long x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __ll2half_ru(long long x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }

            // half -> unsigned int
            inline
            __device__
            unsigned int __half2uint_rn(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            unsigned int __half2uint_rz(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            unsigned int __half2uint_rd(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            unsigned int __half2uint_ru(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }

            // unsigned int -> half
            inline
            __device__
            __half __uint2half_rn(unsigned int x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __uint2half_rz(unsigned int x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __uint2half_rd(unsigned int x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __uint2half_ru(unsigned int x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }

            // half -> unsigned short
            inline
            __device__
            unsigned short __half2ushort_rn(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            unsigned short __half2ushort_rz(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            unsigned short __half2ushort_rd(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            unsigned short __half2ushort_ru(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }

            // unsigned short -> half
            inline
            __device__
            __half __ushort2half_rn(unsigned short x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __ushort2half_rz(unsigned short x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __ushort2half_rd(unsigned short x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __ushort2half_ru(unsigned short x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }

            // half -> unsigned long long
            inline
            __device__
            unsigned long long __half2ull_rn(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            unsigned long long __half2ull_rz(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            unsigned long long __half2ull_rd(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            unsigned long long __half2ull_ru(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }

            // unsigned long long -> half
            inline
            __device__
            __half __ull2half_rn(unsigned long long x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __ull2half_rz(unsigned long long x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __ull2half_rd(unsigned long long x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __ull2half_ru(unsigned long long x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }

            // Load primitives
            inline
            __device__
            __half __ldg(const __half* ptr) { return *ptr; }
            inline
            __device__
            __half __ldcg(const __half* ptr) { return *ptr; }
            inline
            __device__
            __half __ldca(const __half* ptr) { return *ptr; }
            inline
            __device__
            __half __ldcs(const __half* ptr) { return *ptr; }

            inline
            __HOST_DEVICE__
            __half2 __ldg(const __half2* ptr) { return *ptr; }
            inline
            __HOST_DEVICE__
            __half2 __ldcg(const __half2* ptr) { return *ptr; }
            inline
            __HOST_DEVICE__
            __half2 __ldca(const __half2* ptr) { return *ptr; }
            inline
            __HOST_DEVICE__
            __half2 __ldcs(const __half2* ptr) { return *ptr; }

            // Relations
            inline
            __device__
            bool __heq(__half x, __half y)
            {
                return static_cast<__half_raw>(x).data ==
                    static_cast<__half_raw>(y).data;
            }
            inline
            __device__
            bool __hne(__half x, __half y)
            {
                return static_cast<__half_raw>(x).data !=
                    static_cast<__half_raw>(y).data;
            }
            inline
            __device__
            bool __hle(__half x, __half y)
            {
                return static_cast<__half_raw>(x).data <=
                    static_cast<__half_raw>(y).data;
            }
            inline
            __device__
            bool __hge(__half x, __half y)
            {
                return static_cast<__half_raw>(x).data >=
                    static_cast<__half_raw>(y).data;
            }
            inline
            __device__
            bool __hlt(__half x, __half y)
            {
                return static_cast<__half_raw>(x).data <
                    static_cast<__half_raw>(y).data;
            }
            inline
            __device__
            bool __hgt(__half x, __half y)
            {
                return static_cast<__half_raw>(x).data >
                    static_cast<__half_raw>(y).data;
            }
            inline __device__
            bool __hequ(__half x, __half y) {
                return !(static_cast<__half_raw>(x).data < static_cast<__half_raw>(y).data) &&
                    !(static_cast<__half_raw>(x).data > static_cast<__half_raw>(y).data);
            }
            inline __device__
            bool __hneu(__half x, __half y) {
                return !(static_cast<__half_raw>(x).data == static_cast<__half_raw>(y).data);
            }
            inline __device__
            bool __hleu(__half x, __half y) {
                return !(static_cast<__half_raw>(x).data > static_cast<__half_raw>(y).data);
            }
            inline
            __device__
            bool __hgeu(__half x, __half y) {
                return !(static_cast<__half_raw>(x).data < static_cast<__half_raw>(y).data);
            }
            inline
            __device__
            bool __hltu(__half x, __half y) {
                return !(static_cast<__half_raw>(x).data >= static_cast<__half_raw>(y).data);
            }
            inline
            __device__
            bool __hgtu(__half x, __half y) {
                return !(static_cast<__half_raw>(x).data <= static_cast<__half_raw>(y).data);
            }

            inline
            __HOST_DEVICE__
            __half2 __heq2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(x).data ==
                    static_cast<__half2_raw>(y).data;
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hne2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(x).data !=
                    static_cast<__half2_raw>(y).data;
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hle2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(x).data <=
                    static_cast<__half2_raw>(y).data;
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hge2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(x).data >=
                    static_cast<__half2_raw>(y).data;
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hlt2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(x).data <
                    static_cast<__half2_raw>(y).data;
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hgt2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(x).data >
                    static_cast<__half2_raw>(y).data;
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline __HOST_DEVICE__
            __half2 __hequ2(__half2 x, __half2 y) {
                auto r = !(static_cast<__half2_raw>(x).data < static_cast<__half2_raw>(y).data) &&
                    !(static_cast<__half2_raw>(x).data > static_cast<__half2_raw>(y).data);
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hneu2(__half2 x, __half2 y) {
                auto r = !(static_cast<__half2_raw>(x).data == static_cast<__half2_raw>(y).data);
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hleu2(__half2 x, __half2 y) {
                auto r = !(static_cast<__half2_raw>(x).data > static_cast<__half2_raw>(y).data);
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hgeu2(__half2 x, __half2 y) {
                auto r = !(static_cast<__half2_raw>(x).data < static_cast<__half2_raw>(y).data);
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hltu2(__half2 x, __half2 y) {
                auto r = !(static_cast<__half2_raw>(x).data >= static_cast<__half2_raw>(y).data);
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hgtu2(__half2 x, __half2 y) {
                auto r = !(static_cast<__half2_raw>(x).data <= static_cast<__half2_raw>(y).data);
                return __builtin_convertvector(-r, _Float16_2);
            }

            inline
            __HOST_DEVICE__
            bool __hbeq2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(__heq2(x, y));
                return r.data.x != 0 && r.data.y != 0;
            }
            inline
            __HOST_DEVICE__
            bool __hbne2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(__hne2(x, y));
                return r.data.x != 0 && r.data.y != 0;
            }
            inline
            __HOST_DEVICE__
            bool __hble2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(__hle2(x, y));
                return r.data.x != 0 && r.data.y != 0;
            }
            inline
            __HOST_DEVICE__
            bool __hbge2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(__hge2(x, y));
                return r.data.x != 0 && r.data.y != 0;
            }
            inline
            __HOST_DEVICE__
            bool __hblt2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(__hlt2(x, y));
                return r.data.x != 0 && r.data.y != 0;
            }
            inline
            __HOST_DEVICE__
            bool __hbgt2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(__hgt2(x, y));
                return r.data.x != 0 && r.data.y != 0;
            }
            inline
            __HOST_DEVICE__
            bool __hbequ2(__half2 x, __half2 y) { return __hbeq2(x, y); }
            inline
            __HOST_DEVICE__
            bool __hbneu2(__half2 x, __half2 y) { return __hbne2(x, y); }
            inline
            __HOST_DEVICE__
            bool __hbleu2(__half2 x, __half2 y) { return __hble2(x, y); }
            inline
            __HOST_DEVICE__
            bool __hbgeu2(__half2 x, __half2 y) { return __hbge2(x, y); }
            inline
            __HOST_DEVICE__
            bool __hbltu2(__half2 x, __half2 y) { return __hblt2(x, y); }
            inline
            __HOST_DEVICE__
            bool __hbgtu2(__half2 x, __half2 y) { return __hbgt2(x, y); }
            inline
            __device__
            __half __hmax(const __half x, const __half y) {
              return __half_raw{__ocml_fmax_f16(static_cast<__half_raw>(x).data,
                                   static_cast<__half_raw>(y).data)};
            }
            inline
            __device__
            __half __hmax_nan(const __half x, const __half y) {
                if(__ocml_isnan_f16(static_cast<__half_raw>(x).data)) {
                  return x;
                } else if (__ocml_isnan_f16(static_cast<__half_raw>(y).data)) {
                  return y;
                }
                return __hmax(x, y);
            }
            inline
            __device__
            __half __hmin(const __half x, const __half y) {
              return __half_raw{__ocml_fmin_f16(static_cast<__half_raw>(x).data,
                                   static_cast<__half_raw>(y).data)};
            }
            inline
            __device__
            __half __hmin_nan(const __half x, const __half y) {
                if(__ocml_isnan_f16(static_cast<__half_raw>(x).data)) {
                  return x;
                } else if (__ocml_isnan_f16(static_cast<__half_raw>(y).data)) {
                  return y;
                }
                return __hmin(x, y);
            }

            // Arithmetic
            inline
            __device__
            __half __clamp_01(__half x)
            {
                auto r = static_cast<__half_raw>(x);

                if (__hlt(x, __half_raw{0})) return __half_raw{0};
                if (__hlt(__half_raw{1}, x)) return __half_raw{1};
                return r;
            }

            inline
            __device__
            __half __hadd(__half x, __half y)
            {
                return __half_raw{
                    static_cast<__half_raw>(x).data +
                    static_cast<__half_raw>(y).data};
            }
	    inline
	    __device__
	    __half __habs(__half x)
	    {
	        return __half_raw{
		    __ocml_fabs_f16(static_cast<__half_raw>(x).data)};
	    }
            inline
            __device__
            __half __hsub(__half x, __half y)
            {
                return __half_raw{
                    static_cast<__half_raw>(x).data -
                    static_cast<__half_raw>(y).data};
            }
            inline
            __device__
            __half __hmul(__half x, __half y)
            {
                return __half_raw{
                    static_cast<__half_raw>(x).data *
                    static_cast<__half_raw>(y).data};
            }
            inline
            __device__
            __half __hadd_sat(__half x, __half y)
            {
                return __clamp_01(__hadd(x, y));
            }
            inline
            __device__
            __half __hsub_sat(__half x, __half y)
            {
                return __clamp_01(__hsub(x, y));
            }
            inline
            __device__
            __half __hmul_sat(__half x, __half y)
            {
                return __clamp_01(__hmul(x, y));
            }
            inline
            __device__
            __half __hfma(__half x, __half y, __half z)
            {
                return __half_raw{__ocml_fma_f16(
                    static_cast<__half_raw>(x).data,
                    static_cast<__half_raw>(y).data,
                    static_cast<__half_raw>(z).data)};
            }
            inline
            __device__
            __half __hfma_sat(__half x, __half y, __half z)
            {
                return __clamp_01(__hfma(x, y, z));
            }
            inline
            __device__
            __half __hdiv(__half x, __half y)
            {
                return __half_raw{
                    static_cast<__half_raw>(x).data /
                    static_cast<__half_raw>(y).data};
            }

            inline
            __HOST_DEVICE__
            __half2 __hadd2(__half2 x, __half2 y)
            {
                return __half2{
                    static_cast<__half2_raw>(x).data +
                    static_cast<__half2_raw>(y).data};
            }
	    inline
	    __HOST_DEVICE__
	    __half2 __habs2(__half2 x)
	    {
	        return __half2{
		    __ocml_fabs_2f16(static_cast<__half2_raw>(x).data)};
	    }
            inline
            __HOST_DEVICE__
            __half2 __hsub2(__half2 x, __half2 y)
            {
                return __half2{
                    static_cast<__half2_raw>(x).data -
                    static_cast<__half2_raw>(y).data};
            }
            inline
            __HOST_DEVICE__
            __half2 __hmul2(__half2 x, __half2 y)
            {
                return __half2{
                    static_cast<__half2_raw>(x).data *
                    static_cast<__half2_raw>(y).data};
            }
            inline
            __HOST_DEVICE__
            __half2 __hadd2_sat(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(__hadd2(x, y));
                return __half2{
                    __clamp_01(__half_raw{r.data.x}),
                    __clamp_01(__half_raw{r.data.y})};
            }
            inline
            __HOST_DEVICE__
            __half2 __hsub2_sat(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(__hsub2(x, y));
                return __half2{
                    __clamp_01(__half_raw{r.data.x}),
                    __clamp_01(__half_raw{r.data.y})};
            }
            inline
            __HOST_DEVICE__
            __half2 __hmul2_sat(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(__hmul2(x, y));
                return __half2{
                    __clamp_01(__half_raw{r.data.x}),
                    __clamp_01(__half_raw{r.data.y})};
            }
            inline
            __HOST_DEVICE__
            __half2 __hfma2(__half2 x, __half2 y, __half2 z)
            {
                return __half2{__ocml_fma_2f16(x, y, z)};
            }
            inline
            __HOST_DEVICE__
            __half2 __hfma2_sat(__half2 x, __half2 y, __half2 z)
            {
                auto r = static_cast<__half2_raw>(__hfma2(x, y, z));
                return __half2{
                    __clamp_01(__half_raw{r.data.x}),
                    __clamp_01(__half_raw{r.data.y})};
            }
            inline
            __HOST_DEVICE__
            __half2 __h2div(__half2 x, __half2 y)
            {
                return __half2{
                    static_cast<__half2_raw>(x).data /
                    static_cast<__half2_raw>(y).data};
            }

            // Math functions
            #if defined(__clang__) && defined(__HIP__)
            inline
            __device__
            float amd_mixed_dot(__half2 a, __half2 b, float c, bool saturate) {
                return __ockl_fdot2(static_cast<__half2_raw>(a).data,
                                    static_cast<__half2_raw>(b).data,
                                    c, saturate);
            }
            #endif
            inline
            __device__
            __half htrunc(__half x)
            {
                return __half_raw{
                    __ocml_trunc_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hceil(__half x)
            {
                return __half_raw{
                    __ocml_ceil_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hfloor(__half x)
            {
                return __half_raw{
                   __ocml_floor_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hrint(__half x)
            {
                return __half_raw{
                    __ocml_rint_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hsin(__half x)
            {
                return __half_raw{
                    __ocml_sin_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hcos(__half x)
            {
                return __half_raw{
                    __ocml_cos_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hexp(__half x)
            {
                return __half_raw{
                    __ocml_exp_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hexp2(__half x)
            {
                return __half_raw{
                    __ocml_exp2_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hexp10(__half x)
            {
                return __half_raw{
                    __ocml_exp10_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hlog2(__half x)
            {
                return __half_raw{
                    __ocml_log2_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hlog(__half x)
            {
                return __half_raw{
                    __ocml_log_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hlog10(__half x)
            {
                return __half_raw{
                    __ocml_log10_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hrcp(__half x)
            {
                return __half_raw{
                    static_cast<_Float16>(1.0f) /static_cast<__half_raw>(x).data};
            }
            inline
            __device__
            __half hrsqrt(__half x)
            {
                return __half_raw{
                    __ocml_rsqrt_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hsqrt(__half x)
            {
                return __half_raw{
                    __ocml_sqrt_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            bool __hisinf(__half x)
            {
                return __ocml_isinf_f16(static_cast<__half_raw>(x).data);
            }
            inline
            __device__
            bool __hisnan(__half x)
            {
                return __ocml_isnan_f16(static_cast<__half_raw>(x).data);
            }
            inline
            __device__
            __half __hneg(__half x)
            {
                return __half_raw{-static_cast<__half_raw>(x).data};
            }

            inline
            __HOST_DEVICE__
            __half2 h2trunc(__half2 x)
            {
                return __half2{__ocml_trunc_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2ceil(__half2 x)
            {
                return __half2{__ocml_ceil_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2floor(__half2 x)
            {
                return __half2{__ocml_floor_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2rint(__half2 x)
            {
                return __half2{__ocml_rint_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2sin(__half2 x)
            {
                return __half2{__ocml_sin_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2cos(__half2 x)
            {
                return __half2{__ocml_cos_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2exp(__half2 x)
            {
                return __half2{__ocml_exp_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2exp2(__half2 x)
            {
                return __half2{__ocml_exp2_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2exp10(__half2 x)
            {
                return __half2{__ocml_exp10_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2log2(__half2 x)
            {
                return __half2{__ocml_log2_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2log(__half2 x) { return __ocml_log_2f16(x); }
            inline
            __HOST_DEVICE__
            __half2 h2log10(__half2 x) { return __ocml_log10_2f16(x); }
            inline
            __HOST_DEVICE__
            __half2 h2rcp(__half2 x) {
                return _Float16_2{
                    _Float16_2{static_cast<_Float16>(1.0f), static_cast<_Float16>(1.0f)} / x.data};
            }
            inline
            __HOST_DEVICE__
            __half2 h2rsqrt(__half2 x) { return __ocml_rsqrt_2f16(x); }
            inline
            __HOST_DEVICE__
            __half2 h2sqrt(__half2 x) { return __ocml_sqrt_2f16(x); }
            inline
            __HOST_DEVICE__
            __half2 __hisinf2(__half2 x)
            {
                auto r = __ocml_isinf_2f16(x);
                return __half2{_Float16_2{
                    static_cast<_Float16>(r.x), static_cast<_Float16>(r.y)}};
            }
            inline
            __HOST_DEVICE__
            __half2 __hisnan2(__half2 x)
            {
                auto r = __ocml_isnan_2f16(x);
                return __half2{_Float16_2{
                    static_cast<_Float16>(r.x), static_cast<_Float16>(r.y)}};
            }
            inline
            __HOST_DEVICE__
            __half2 __hneg2(__half2 x)
            {
                return __half2{-static_cast<__half2_raw>(x).data};
            }
        } // Anonymous namespace.

        #if !defined(HIP_NO_HALF)
            using half = __half;
            using half2 = __half2;
        #endif
        __device__
        inline
        __half __shfl(__half var, int src_lane, int width = warpSize) {
           union { int i; __half h; } tmp; tmp.h = var;
           tmp.i = __shfl(tmp.i, src_lane, width);
           return tmp.h;
        }
        __device__
        inline
        __half2 __shfl(__half2 var, int src_lane, int width = warpSize) {
           union { int i; __half2 h; } tmp; tmp.h = var;
           tmp.i = __shfl(tmp.i, src_lane, width);
           return tmp.h;
        }
        __device__
        inline
        __half __shfl_up(__half var, unsigned int lane_delta, int width = warpSize) {
           union { int i; __half h; } tmp; tmp.h = var;
           tmp.i = __shfl_up(tmp.i, lane_delta, width);
           return tmp.h;
        }
        __device__
        inline
         __half2 __shfl_up(__half2 var, unsigned int lane_delta, int width = warpSize) {
            union { int i; __half2 h; } tmp; tmp.h = var;
            tmp.i = __shfl_up(tmp.i, lane_delta, width);
            return tmp.h;
         }
         __device__
         inline
         __half __shfl_down(__half var, unsigned int lane_delta, int width = warpSize) {
            union { int i; __half h; } tmp; tmp.h = var;
            tmp.i = __shfl_down(tmp.i, lane_delta, width);
            return tmp.h;
         }
         __device__
         inline
         __half2 __shfl_down(__half2 var, unsigned int lane_delta, int width = warpSize) {
            union { int i; __half2 h; } tmp; tmp.h = var;
            tmp.i = __shfl_down(tmp.i, lane_delta, width);
            return tmp.h;
         }
         __device__
         inline
         __half __shfl_xor(__half var,  int lane_mask, int width = warpSize) {
            union { int i; __half h; } tmp; tmp.h = var;
            tmp.i = __shfl_xor(tmp.i, lane_mask, width);
            return tmp.h;
         }
         __device__
         inline
          __half2 __shfl_xor(__half2 var,  int lane_mask, int width = warpSize) {
             union { int i; __half2 h; } tmp; tmp.h = var;
             tmp.i = __shfl_xor(tmp.i, lane_mask, width);
             return tmp.h;
         }
    #endif // defined(__cplusplus)
#elif defined(__GNUC__)
    #if !defined(__HIPCC_RTC__)
      #include "hip_fp16_gcc.h"
    #endif
#endif // !defined(__clang__) && defined(__GNUC__)

#endif // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP16_H
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#if !defined(__HIPCC_RTC__)
#include "hip_fp16_math_fwd.h"
#include "amd_hip_vector_types.h"
#include "math_fwd.h"

#include <hip/amd_detail/host_defines.h>

#include <algorithm>
// assert.h is only for the host version of assert.
// The device version of assert is implemented in hip/amd_detail/hip_runtime.h.
// Users should include hip_runtime.h for the device version of assert.
#if !__HIP_DEVICE_COMPILE__
#include <assert.h>
#endif
#include <limits.h>
#include <limits>
#include <stdint.h>
#endif // !defined(__HIPCC_RTC__)

#if _LIBCPP_VERSION && __HIP__
namespace std {
template <>
struct __numeric_type<_Float16>
{
   static _Float16 __test(_Float16);

   typedef _Float16 type;
   static const bool value = true;
};
}
#endif // _LIBCPP_VERSION

#pragma push_macro("__DEVICE__")
#pragma push_macro("__RETURN_TYPE")

#define __DEVICE__ static __device__
#define __RETURN_TYPE bool

// DOT FUNCTIONS
#if __HIP_CLANG_ONLY__
__DEVICE__
inline
int amd_mixed_dot(short2 a, short2 b, int c, bool saturate) {
    return __ockl_sdot2(a.data, b.data, c, saturate);
}
__DEVICE__
inline
uint amd_mixed_dot(ushort2 a, ushort2 b, uint c, bool saturate) {
    return __ockl_udot2(a.data, b.data, c, saturate);
}
__DEVICE__
inline
int amd_mixed_dot(char4 a, char4 b, int c, bool saturate) {
    return __ockl_sdot4(a.data, b.data, c, saturate);
}
__DEVICE__
inline
uint amd_mixed_dot(uchar4 a, uchar4 b, uint c, bool saturate) {
    return __ockl_udot4(a.data, b.data, c, saturate);
}
__DEVICE__
inline
int amd_mixed_dot(int a, int b, int c, bool saturate) {
    return __ockl_sdot8(a, b, c, saturate);
}
__DEVICE__
inline
uint amd_mixed_dot(uint a, uint b, uint c, bool saturate) {
    return __ockl_udot8(a, b, c, saturate);
}
#endif

#pragma pop_macro("__DEVICE__")
#pragma pop_macro("__RETURN_TYPE")
// For backward compatibility.
// There are HIP applications e.g. TensorFlow, expecting __HIP_ARCH_* macros
// defined after including math_functions.h.
#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/amd_hip_runtime.h>
#endif
