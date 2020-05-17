// https://github.com/moskewcz/boda/issues/13

#define USE_FP16

#ifdef USE_FP16
	#define up(x) x
	#define down(x) x
	#define xtype half8
#else
	#define up(x) convert_float8(x)
	#define down(x) convert_half8(x)
	#define xtype float8
#endif

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void gemm( global const half* a, global const half* b, global half* c )
{
  xtype c_r[8] = {0,0,0,0,0,0,0,0};
  xtype a_r;
  xtype b_r;

  int const a_off_thr = get_global_id(0)/128;
  int const b_off_thr = get_global_id(0)%128;

  int a_off = a_off_thr;
  int b_off = b_off_thr;
  for( int k = 0; k < 1024; k += 1 ) {
    a_r = up(((global const half8*)a)[a_off+0]);
    b_r = up(((global const half8*)b)[b_off+0]);
    c_r[0] += a_r.s0*b_r;
    c_r[1] += a_r.s1*b_r;
    c_r[2] += a_r.s2*b_r;
    c_r[3] += a_r.s3*b_r;
    c_r[4] += a_r.s4*b_r;
    c_r[5] += a_r.s5*b_r;
    c_r[6] += a_r.s6*b_r;
    c_r[7] += a_r.s7*b_r;
    a_off += 128;
    b_off += 128;
  }

  int c_off = (get_global_id(0)/128)*1024 + (get_global_id(0)%128);
	for (int i = 0; i < 8; i++) {
		((global half8*)c)[c_off] = c_r[i];
		c_off += 128;
	}
}

