// https://github.com/moskewcz/boda/issues/13

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void gemm( global const half* a, global const half* b, global half* c )
{
  half8 c_r[8] = {0,0,0,0,0,0,0,0};
  half8 a_r;
  half8 b_r;

  int const a_off_thr = get_global_id(0)/128;
  int const b_off_thr = get_global_id(0)%128;

  int a_off = a_off_thr;
  int b_off = b_off_thr;
  for( int k = 0; k < 1024; k += 1 ) {
    a_r = ((global const half8*)a)[a_off+0];
    b_r = ((global const half8*)b)[b_off+0];
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
  int c_off = (get_global_id(0)/128)*1024*8 + (get_global_id(0)%128)*8;
  vstore8(c_r[0], 0, c+c_off);
  vstore8(c_r[1], 0, c+c_off+1024);
  vstore8(c_r[2], 0, c+c_off+1024*2);
  vstore8(c_r[3], 0, c+c_off+1024*3);
  vstore8(c_r[4], 0, c+c_off+1024*4);
  vstore8(c_r[5], 0, c+c_off+1024*5);
  vstore8(c_r[6], 0, c+c_off+1024*6);
  vstore8(c_r[7], 0, c+c_off+1024*7);
}

