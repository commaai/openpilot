// https://github.com/moskewcz/boda/issues/13

#define up(x) x
#define down(x) x
#define xtype half8
#define read_imagep read_imageh
#define write_imagep write_imageh

/*#define up(x) convert_float8(x)
#define down(x) convert_half8(x)
#define xtype float8*/

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
  half8 a_r;
  half8 b_r;

  int const a_off_thr = get_global_id(0)/128;
  int const b_off_thr = get_global_id(0)%128;

  for( int k = 0; k < 1024; k += 1 ) {
    int2 a_samp = {a_off_thr*2+0, k};
    a_r.lo = read_imagep(A, smp, a_samp);
    ++a_samp.x;
    a_r.hi = read_imagep(A, smp, a_samp);

    int2 b_samp = {b_off_thr*2+0, k};
    b_r.lo = read_imagep(B, smp, b_samp);
    ++b_samp.x;
    b_r.hi = read_imagep(B, smp, b_samp);

    c_r[0] += up(a_r.s0*b_r);
    c_r[1] += up(a_r.s1*b_r);
    c_r[2] += up(a_r.s2*b_r);
    c_r[3] += up(a_r.s3*b_r);
    c_r[4] += up(a_r.s4*b_r);
    c_r[5] += up(a_r.s5*b_r);
    c_r[6] += up(a_r.s6*b_r);
    c_r[7] += up(a_r.s7*b_r);
  }

  int2 c_samp = {a_off_thr, b_off_thr*8};
  for (int i = 0; i < 8; i++) {
    write_imagep(C, c_samp, c_r[i].hi);
    ++c_samp.y;
  }

  /*int c_off = (get_global_id(0)/128)*1024*8 + (get_global_id(0)%128)*8;
  vstore8(down(c_r[0]), 0, c+c_off);
  vstore8(down(c_r[1]), 0, c+c_off+1024);
  vstore8(down(c_r[2]), 0, c+c_off+1024*2);
  vstore8(down(c_r[3]), 0, c+c_off+1024*3);
  vstore8(down(c_r[4]), 0, c+c_off+1024*4);
  vstore8(down(c_r[5]), 0, c+c_off+1024*5);
  vstore8(down(c_r[6]), 0, c+c_off+1024*6);
  vstore8(down(c_r[7]), 0, c+c_off+1024*7);*/
}

