// https://github.com/moskewcz/boda/issues/13

#define USE_FP16

#ifdef USE_FP16
  #define up(x) x
  #define down(x) x
  #define xtype half8
  #define skip 128
#else
  #define up(x) convert_float8(x)
  #define down(x) convert_half8(x)
  #define xtype float8
  #define skip 128
#endif

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void gemm(const int M, const int N, const int K,
                   global const half8* a, global const half8* b, global half8* c )
{
  xtype c_r[8] = {0,0,0,0,0,0,0,0};

  int const a_off_thr = get_global_id(0);
  int const b_off_thr = get_global_id(1);

  int a_off = a_off_thr;
  int b_off = b_off_thr;
  for( int k = 0; k < 1024; k += 1 ) {
    xtype a_r = up(a[a_off]);
    xtype b_r = up(b[b_off]);

    c_r[0] += a_r.s0*b_r;
    c_r[1] += a_r.s1*b_r;
    c_r[2] += a_r.s2*b_r;
    c_r[3] += a_r.s3*b_r;
    c_r[4] += a_r.s4*b_r;
    c_r[5] += a_r.s5*b_r;
    c_r[6] += a_r.s6*b_r;
    c_r[7] += a_r.s7*b_r;

    a_off += skip;
    b_off += skip;
  }

  int c_off = a_off_thr*1024 + b_off_thr;
  for (int i = 0; i < 8; i++) {
    c[c_off] = down(c_r[i]);
    c_off += skip;
  }
}

