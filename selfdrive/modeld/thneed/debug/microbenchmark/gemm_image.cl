// https://github.com/moskewcz/boda/issues/13

//#define USE_FP16

#ifdef USE_FP16
  #define xtype half4
  #define read_imagep read_imageh
  #define write_imagep write_imageh
#else
  #define xtype float4
  #define read_imagep read_imagef
  #define write_imagep write_imagef
#endif

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void gemm(const int M, const int N, const int K,
  read_only image2d_t A, 
  read_only image2d_t B, 
  write_only image2d_t C)
{
  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE |
                        CLK_ADDRESS_CLAMP           |
                        CLK_FILTER_NEAREST;

  xtype c_r[4] = {0,0,0,0};
  xtype a_r[4], b_r[4];

  int const a_off_thr = get_global_id(0);
  int const b_off_thr = get_global_id(1);

  int2 a_samp = {0, a_off_thr};
  int2 b_samp = {0, b_off_thr};

  for (short k = 0; k < K/4; k++) {
    for (short i = 0; i < 4; ++i) {
      a_r[i] = read_imagep(A, smp, a_samp);
      b_r[i] = read_imagep(B, smp, b_samp);
      ++a_samp.x;
      ++b_samp.x;
    }

    for (short i = 0; i < 4; ++i) {
      float4 ov = c_r[i];

      ov.x += a_r[i].x * b_r[0].x;
      ov.x += a_r[i].y * b_r[0].y;
      ov.x += a_r[i].z * b_r[0].z;
      ov.x += a_r[i].w * b_r[0].w;

      ov.y += a_r[i].x * b_r[1].x;
      ov.y += a_r[i].y * b_r[1].y;
      ov.y += a_r[i].z * b_r[1].z;
      ov.y += a_r[i].w * b_r[1].w;

      ov.z += a_r[i].x * b_r[2].x;
      ov.z += a_r[i].y * b_r[2].y;
      ov.z += a_r[i].z * b_r[2].z;
      ov.z += a_r[i].w * b_r[2].w;

      ov.w += a_r[i].x * b_r[3].x;
      ov.w += a_r[i].y * b_r[3].y;
      ov.w += a_r[i].z * b_r[3].z;
      ov.w += a_r[i].w * b_r[3].w;

      c_r[i] = ov;
    }
  }

  int2 c_samp = {a_off_thr, b_off_thr*4};
  for (short i = 0; i < 4; i++) {
    write_imagep(C, c_samp, c_r[i]);
    ++c_samp.y;
  }
}

