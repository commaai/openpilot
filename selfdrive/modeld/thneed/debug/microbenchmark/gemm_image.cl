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
__kernel void gemm(
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

  int2 a_samp = {a_off_thr, 0};
  int2 b_samp = {b_off_thr, 0};

  for (short k = 0; k < 1024; k+=4) {
    for (short i = 0; i < 4; ++i) {
      a_r[i] = read_imagep(A, smp, a_samp);
      ++a_samp.y;
      b_r[i] = read_imagep(B, smp, b_samp);
      ++b_samp.y;
    }

    for (short i = 0; i < 4; ++i) {
      xtype ov = c_r[i];
      ov += a_r[0].x * b_r[0];
      ov += a_r[1].y * b_r[1];
      ov += a_r[2].z * b_r[2];
      ov += a_r[3].w * b_r[3];
      c_r[i] = ov;
    }
  }

  int2 c_samp = {a_off_thr, b_off_thr*4};
  for (short i = 0; i < 4; i++) {
    write_imagep(C, c_samp, c_r[i]);
    ++c_samp.y;
  }
}

