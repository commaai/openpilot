#define UV_SIZE ((TRANSFORMED_WIDTH/2)*(TRANSFORMED_HEIGHT/2))

__kernel void loadys(__global uchar8 const * const Y,
                     __global float * out)
{
    const int gid = get_global_id(0);
    const int ois = gid * 8;
    const int oy = ois / TRANSFORMED_WIDTH;
    const int ox = ois % TRANSFORMED_WIDTH;

    const uchar8 ys = Y[gid];

    // y = (x - 128) / 128
    const float8 ysf = (convert_float8(ys) - 128.f) * 0.0078125f;

    // 02
    // 13

    __global float* outy0;
    __global float* outy1;
    if ((oy & 1) == 0) {
      outy0 = out; //y0
      outy1 = out + UV_SIZE*2; //y2
    } else {
      outy0 = out + UV_SIZE; //y1
      outy1 = out + UV_SIZE*3; //y3
    }

    vstore4(ysf.s0246, 0, outy0 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2);
    vstore4(ysf.s1357, 0, outy1 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2);
}

__kernel void loaduv(__global uchar8 const * const in,
                     __global float8 * out,
                     int out_offset)
{
  const int gid = get_global_id(0);
  const uchar8 inv = in[gid];

  // y = (x - 128) / 128
  const float8 outv  = (convert_float8(inv) - 128.f) * 0.0078125f;
  out[gid + out_offset / 8] = outv;
}
