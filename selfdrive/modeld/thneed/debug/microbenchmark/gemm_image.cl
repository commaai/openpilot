// https://github.com/moskewcz/boda/issues/13

//#define USE_FP16

#ifdef USE_FP16
	#define up(x) x
	#define down(x) x
	#define xtype half8
  #define read_imagep read_imageh
  #define write_imagep write_imageh
#else
	#define up(x) convert_float8(x)
	#define down(x) convert_half8(x)
	#define xtype float8
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

  xtype c_r[8] = {0,0,0,0,0,0,0,0};
  xtype a_r;
  xtype b_r;

  int const a_off_thr = get_global_id(0);
  int const b_off_thr = get_global_id(1);

  for( int k = 0; k < 1024; k += 1 ) {
    int2 a_samp = {a_off_thr*2+0, k};
    a_r.lo = read_imagep(A, smp, a_samp);
    ++a_samp.x;
    a_r.hi = read_imagep(A, smp, a_samp);

    int2 b_samp = {b_off_thr*2+0, k};
    b_r.lo = read_imagep(B, smp, b_samp);
    ++b_samp.x;
    b_r.hi = read_imagep(B, smp, b_samp);

    c_r[0] += a_r.s0*b_r;
    c_r[1] += a_r.s1*b_r;
    c_r[2] += a_r.s2*b_r;
    c_r[3] += a_r.s3*b_r;
    c_r[4] += a_r.s4*b_r;
    c_r[5] += a_r.s5*b_r;
    c_r[6] += a_r.s6*b_r;
    c_r[7] += a_r.s7*b_r;
  }

  for (int i = 0; i < 8; i++) {
    int2 c_samp = {a_off_thr*2, b_off_thr*8 + i};
    write_imagep(C, c_samp, c_r[i].lo);
    c_samp.x += 1;
    write_imagep(C, c_samp, c_r[i].hi);
  }
}

