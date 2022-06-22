#define RGB_TO_Y(r, g, b) ((((mul24(b, 13) + mul24(g, 65) + mul24(r, 33)) + 64) >> 7) + 16)
#define RGB_TO_U(r, g, b) ((mul24(b, 56) - mul24(g, 37) - mul24(r, 19) + 0x8080) >> 8)
#define RGB_TO_V(r, g, b) ((mul24(r, 56) - mul24(g, 47) - mul24(b, 9) + 0x8080) >> 8)
#define AVERAGE(x, y, z, w) ((convert_ushort(x) + convert_ushort(y) + convert_ushort(z) + convert_ushort(w) + 1) >> 1)

inline void convert_2_ys(__global uchar * out_yuv, int yi, const uchar8 rgbs1) {
  uchar2 yy = (uchar2)(
    RGB_TO_Y(rgbs1.s2, rgbs1.s1, rgbs1.s0),
    RGB_TO_Y(rgbs1.s5, rgbs1.s4, rgbs1.s3)
  );
#ifdef CL_DEBUG
  if(yi >= RGB_SIZE)
    printf("Y vector2 overflow, %d > %d\n", yi, RGB_SIZE);
#endif
  vstore2(yy, 0, out_yuv + yi);
}

inline void convert_4_ys(__global uchar * out_yuv, int yi, const uchar8 rgbs1, const uchar8 rgbs3) {
  const uchar4 yy = (uchar4)(
    RGB_TO_Y(rgbs1.s2, rgbs1.s1, rgbs1.s0),
    RGB_TO_Y(rgbs1.s5, rgbs1.s4, rgbs1.s3),
    RGB_TO_Y(rgbs3.s0, rgbs1.s7, rgbs1.s6),
    RGB_TO_Y(rgbs3.s3, rgbs3.s2, rgbs3.s1)
  );
#ifdef CL_DEBUG
  if(yi > RGB_SIZE - 4)
    printf("Y vector4 overflow, %d > %d\n", yi, RGB_SIZE - 4);
#endif
  vstore4(yy, 0, out_yuv + yi);
}

inline void convert_uv(__global uchar * out_yuv, int ui, int vi,
                    const uchar8 rgbs1, const uchar8 rgbs2) {
  // U & V: average of 2x2 pixels square
  const short ab = AVERAGE(rgbs1.s0, rgbs1.s3, rgbs2.s0, rgbs2.s3);
  const short ag = AVERAGE(rgbs1.s1, rgbs1.s4, rgbs2.s1, rgbs2.s4);
  const short ar = AVERAGE(rgbs1.s2, rgbs1.s5, rgbs2.s2, rgbs2.s5);
#ifdef CL_DEBUG
  if(ui >= RGB_SIZE  + RGB_SIZE / 4)
    printf("U overflow, %d >= %d\n", ui, RGB_SIZE  + RGB_SIZE / 4);
  if(vi >= RGB_SIZE  + RGB_SIZE / 2)
    printf("V overflow, %d >= %d\n", vi, RGB_SIZE  + RGB_SIZE / 2);
#endif
  out_yuv[ui] = RGB_TO_U(ar, ag, ab);
  out_yuv[vi] = RGB_TO_V(ar, ag, ab);
}

inline void convert_2_uvs(__global uchar * out_yuv, int ui, int vi,
                    const uchar8 rgbs1, const uchar8 rgbs2, const uchar8 rgbs3, const uchar8 rgbs4) {
  // U & V: average of 2x2 pixels square
  const short ab1 = AVERAGE(rgbs1.s0, rgbs1.s3, rgbs2.s0, rgbs2.s3);
  const short ag1 = AVERAGE(rgbs1.s1, rgbs1.s4, rgbs2.s1, rgbs2.s4);
  const short ar1 = AVERAGE(rgbs1.s2, rgbs1.s5, rgbs2.s2, rgbs2.s5);
  const short ab2 = AVERAGE(rgbs1.s6, rgbs3.s1, rgbs2.s6, rgbs4.s1);
  const short ag2 = AVERAGE(rgbs1.s7, rgbs3.s2, rgbs2.s7, rgbs4.s2);
  const short ar2 = AVERAGE(rgbs3.s0, rgbs3.s3, rgbs4.s0, rgbs4.s3);
  uchar2 u2 = (uchar2)(
    RGB_TO_U(ar1, ag1, ab1),
    RGB_TO_U(ar2, ag2, ab2)
  );
  uchar2 v2 = (uchar2)(
    RGB_TO_V(ar1, ag1, ab1),
    RGB_TO_V(ar2, ag2, ab2)
  );
#ifdef CL_DEBUG1
  if(ui > RGB_SIZE  + RGB_SIZE / 4 - 2)
    printf("U 2 overflow, %d >= %d\n", ui, RGB_SIZE  + RGB_SIZE / 4 - 2);
  if(vi > RGB_SIZE  + RGB_SIZE / 2 - 2)
    printf("V 2 overflow, %d >= %d\n", vi, RGB_SIZE  + RGB_SIZE / 2 - 2);
#endif
  vstore2(u2, 0, out_yuv + ui);
  vstore2(v2, 0, out_yuv + vi);
}

__kernel void rgb_to_yuv(__global uchar const * const rgb,
                    __global uchar * out_yuv)
{
  const int dx = get_global_id(0);
  const int dy = get_global_id(1);
  const int col = mul24(dx, 4); // Current column in rgb image
  const int row = mul24(dy, 4); // Current row in rgb image
  const int bgri_start = mad24(row, RGB_STRIDE, mul24(col, 3)); // Start offset of rgb data being converted
  const int yi_start = mad24(row,  WIDTH, col); // Start offset in the target yuv buffer
  int ui = mad24(row / 2, UV_WIDTH, RGB_SIZE + col / 2);
  int vi = mad24(row / 2 , UV_WIDTH, RGB_SIZE + UV_WIDTH * UV_HEIGHT + col / 2);
  int num_col = min(WIDTH - col, 4);
  int num_row = min(HEIGHT - row, 4);
  if(num_row == 4) {
    const uchar8 rgbs0_0 = vload8(0, rgb + bgri_start);
    const uchar8 rgbs0_1 = vload8(0, rgb + bgri_start + 8);
    const uchar8 rgbs1_0 = vload8(0, rgb + bgri_start + RGB_STRIDE);
    const uchar8 rgbs1_1 = vload8(0, rgb + bgri_start + RGB_STRIDE + 8);
    const uchar8 rgbs2_0 = vload8(0, rgb + bgri_start + RGB_STRIDE * 2);
    const uchar8 rgbs2_1 = vload8(0, rgb + bgri_start + RGB_STRIDE * 2 + 8);
    const uchar8 rgbs3_0 = vload8(0, rgb + bgri_start + RGB_STRIDE * 3);
    const uchar8 rgbs3_1 = vload8(0, rgb + bgri_start + RGB_STRIDE * 3 + 8);
    if(num_col == 4) {
      convert_4_ys(out_yuv, yi_start, rgbs0_0, rgbs0_1);
      convert_4_ys(out_yuv, yi_start + WIDTH, rgbs1_0, rgbs1_1);
      convert_4_ys(out_yuv, yi_start + WIDTH * 2, rgbs2_0, rgbs2_1);
      convert_4_ys(out_yuv, yi_start + WIDTH * 3, rgbs3_0, rgbs3_1);
      convert_2_uvs(out_yuv, ui, vi, rgbs0_0, rgbs1_0, rgbs0_1, rgbs1_1);
      convert_2_uvs(out_yuv, ui + UV_WIDTH, vi + UV_WIDTH, rgbs2_0, rgbs3_0, rgbs2_1, rgbs3_1);
    } else if(num_col == 2) {
      convert_2_ys(out_yuv, yi_start, rgbs0_0);
      convert_2_ys(out_yuv, yi_start + WIDTH, rgbs1_0);
      convert_2_ys(out_yuv, yi_start + WIDTH * 2, rgbs2_0);
      convert_2_ys(out_yuv, yi_start + WIDTH * 3, rgbs3_0);
      convert_uv(out_yuv, ui, vi, rgbs0_0, rgbs1_0);
      convert_uv(out_yuv, ui + UV_WIDTH, vi + UV_WIDTH, rgbs2_0, rgbs3_0);
    }
  } else {
    const uchar8 rgbs0_0 = vload8(0, rgb + bgri_start);
    const uchar8 rgbs0_1 = vload8(0, rgb + bgri_start + 8);
    const uchar8 rgbs1_0 = vload8(0, rgb + bgri_start + RGB_STRIDE);
    const uchar8 rgbs1_1 = vload8(0, rgb + bgri_start + RGB_STRIDE + 8);
    if(num_col == 4) {
      convert_4_ys(out_yuv, yi_start, rgbs0_0, rgbs0_1);
      convert_4_ys(out_yuv, yi_start + WIDTH, rgbs1_0, rgbs1_1);
      convert_2_uvs(out_yuv, ui, vi, rgbs0_0, rgbs1_0, rgbs0_1, rgbs1_1);
    } else if(num_col == 2) {
      convert_2_ys(out_yuv, yi_start, rgbs0_0);
      convert_2_ys(out_yuv, yi_start + WIDTH, rgbs1_0);
      convert_uv(out_yuv, ui, vi, rgbs0_0, rgbs1_0);
    }
  }
}
