#define Y_WIDTH RGB_WIDTH
#define Y_HEIGHT RGB_HEIGHT
#define UV_WIDTH RGB_WIDTH / 2
#define UV_HEIGHT RGB_HEIGHT / 2
#define HALF_UV_WIDTH UV_WIDTH / 2
#define U_OFFSET Y_WIDTH * Y_HEIGHT
#define V_OFFSET Y_WIDTH * Y_HEIGHT + UV_WIDTH * UV_HEIGHT

#define RGB_TO_Y(r, g, b) ((((mul24(b, 13) + mul24(g, 65) + mul24(r, 33)) + 64) >> 7) + 16)
#define RGB_TO_U(r, g, b) ((mul24(b, 56) - mul24(g, 37) - mul24(r, 19) + 0x8080) >> 8)
#define RGB_TO_V(r, g, b) ((mul24(r, 56) - mul24(g, 47) - mul24(b, 9) + 0x8080) >> 8)

const __constant float3 color_correction[3] = {
  // Matrix from WBraw -> sRGBD65 (normalized)
  (float3)( 1.62393627, -0.2092988,  0.00119886),
  (float3)(-0.45734315,  1.5534676, -0.59296798),
  (float3)(-0.16659312, -0.3441688,  1.59176912),
};

inline float3 color_correct(float3 x) {
  // white balance of daylight
  x /= (float3)(0.4609375, 1.0, 0.546875);
  x = clamp(x, 0.0, 1.0);

  // fix up the colors
  float3 ret = x.x * color_correction[0];
  ret += x.y * color_correction[1];
  ret += x.z * color_correction[2];
  return ret;
}

inline float3 srgb_gamma(float3 p) {
  // go all out and add an sRGB gamma curve
  const float3 ph = (1.0f + 0.055f)*pow(p, 1/2.4f) - 0.055f;
	const float3 pl = p*12.92f;
	return select(ph, pl, islessequal(p, 0.0031308f));
}

__constant int dpcm_lookup[512] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, 935, 951, 967, 983, 999, 1015, 1031, 1047, 1063, 1079, 1095, 1111, 1127, 1143, 1159, 1175, 1191, 1207, 1223, 1239, 1255, 1271, 1287, 1303, 1319, 1335, 1351, 1367, 1383, 1399, 1415, 1431, -935, -951, -967, -983, -999, -1015, -1031, -1047, -1063, -1079, -1095, -1111, -1127, -1143, -1159, -1175, -1191, -1207, -1223, -1239, -1255, -1271, -1287, -1303, -1319, -1335, -1351, -1367, -1383, -1399, -1415, -1431, 419, 427, 435, 443, 451, 459, 467, 475, 483, 491, 499, 507, 515, 523, 531, 539, 547, 555, 563, 571, 579, 587, 595, 603, 611, 619, 627, 635, 643, 651, 659, 667, 675, 683, 691, 699, 707, 715, 723, 731, 739, 747, 755, 763, 771, 779, 787, 795, 803, 811, 819, 827, 835, 843, 851, 859, 867, 875, 883, 891, 899, 907, 915, 923, -419, -427, -435, -443, -451, -459, -467, -475, -483, -491, -499, -507, -515, -523, -531, -539, -547, -555, -563, -571, -579, -587, -595, -603, -611, -619, -627, -635, -643, -651, -659, -667, -675, -683, -691, -699, -707, -715, -723, -731, -739, -747, -755, -763, -771, -779, -787, -795, -803, -811, -819, -827, -835, -843, -851, -859, -867, -875, -883, -891, -899, -907, -915, -923, 161, 165, 169, 173, 177, 181, 185, 189, 193, 197, 201, 205, 209, 213, 217, 221, 225, 229, 233, 237, 241, 245, 249, 253, 257, 261, 265, 269, 273, 277, 281, 285, 289, 293, 297, 301, 305, 309, 313, 317, 321, 325, 329, 333, 337, 341, 345, 349, 353, 357, 361, 365, 369, 373, 377, 381, 385, 389, 393, 397, 401, 405, 409, 413, -161, -165, -169, -173, -177, -181, -185, -189, -193, -197, -201, -205, -209, -213, -217, -221, -225, -229, -233, -237, -241, -245, -249, -253, -257, -261, -265, -269, -273, -277, -281, -285, -289, -293, -297, -301, -305, -309, -313, -317, -321, -325, -329, -333, -337, -341, -345, -349, -353, -357, -361, -365, -369, -373, -377, -381, -385, -389, -393, -397, -401, -405, -409, -413, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, -32, -34, -36, -38, -40, -42, -44, -46, -48, -50, -52, -54, -56, -58, -60, -62, -64, -66, -68, -70, -72, -74, -76, -78, -80, -82, -84, -86, -88, -90, -92, -94, -96, -98, -100, -102, -104, -106, -108, -110, -112, -114, -116, -118, -120, -122, -124, -126, -128, -130, -132, -134, -136, -138, -140, -142, -144, -146, -148, -150, -152, -154, -156, -158};

inline uint4 decompress(uint4 p, uint4 pl) {
  uint4 r1 = (pl + (uint4)(dpcm_lookup[p.s0], dpcm_lookup[p.s1], dpcm_lookup[p.s2], dpcm_lookup[p.s3]));
  uint4 r2 = ((p-0x200)<<5) | 0xF;
  r2 += select((uint4)(0,0,0,0), (uint4)(1,1,1,1), r2 <= pl);
  return select(r2, r1, p < 0x200);
}

__kernel void debayer10(__global uchar const * const in,
                        __global uchar * out, float digital_gain, __local uchar * us, __local uchar * vs)
{
  const int rgby_y = get_global_id(0);
  const int lid = get_local_id(0);
  const int uv_y = rgby_y / 2;
  const int frame_y = rgby_y * 2;

  uint4 pint_last;
  uchar3 rgbs[2];

  for (int uv_x = 0; uv_x < UV_WIDTH; uv_x++) {
    const int frame_x = mul24(uv_x, 5);

    // TODO: why doesn't this work for the frontview
    /*const uchar8 v1 = vload8(0, &in[frame_y * FRAME_STRIDE + frame_x]);
    const uchar ex1 = v1.s4;
    const uchar8 v2 = vload8(0, &in[(frame_y+1) * FRAME_STRIDE + frame_x]);
    const uchar ex2 = v2.s4;*/

    const uchar4 v1 = vload4(0, &in[mad24(frame_y, FRAME_STRIDE, frame_x)]);
    const uchar ex1 = in[mad24(frame_y, FRAME_STRIDE, frame_x + 4)];
    const uchar4 v2 = vload4(0, &in[mad24(frame_y + 1, FRAME_STRIDE, frame_x)]);
    const uchar ex2 = in[mad24(frame_y + 1, FRAME_STRIDE, frame_x + 4)];

    uint4 pinta[2];
    pinta[0] = (uint4)(
      (((uint)v1.s0 << 2) + ( (ex1 >> 0) & 3)),
      (((uint)v1.s1 << 2) + ( (ex1 >> 2) & 3)),
      (((uint)v2.s0 << 2) + ( (ex2 >> 0) & 3)),
      (((uint)v2.s1 << 2) + ( (ex2 >> 2) & 3)));
    pinta[1] = (uint4)(
      (((uint)v1.s2 << 2) + ( (ex1 >> 4) & 3)),
      (((uint)v1.s3 << 2) + ( (ex1 >> 6) & 3)),
      (((uint)v2.s2 << 2) + ( (ex2 >> 4) & 3)),
      (((uint)v2.s3 << 2) + ( (ex2 >> 6) & 3)));

    #pragma unroll
    for (int px = 0; px < 2; px++) {
      const int rgby_x = mad24(uv_x, 2, px);
      uint4 pint = pinta[px];

#if HDR
      // decompress HDR
      pint = (rgby_x == 0) ? ((pint<<4) | 8) : decompress(pint, pint_last);
      pint_last = pint;
#endif

      float4 p = convert_float4(pint);

      // 64 is the black level of the sensor, remove
      // (changed to 56 for HDR)
      const float black_level = 56.0f;
      // TODO: switch to max here?
      p = (p - black_level);

      // correct vignetting (no pow function?)
      // see https://www.eecis.udel.edu/~jye/lab_research/09/JiUp.pdf the A (4th order)
      const float r = ((rgby_y - RGB_HEIGHT/2)*(rgby_y - RGB_HEIGHT/2) + (rgby_x - RGB_WIDTH/2)*(rgby_x - RGB_WIDTH/2));
      const float fake_f = 700.0f;    // should be 910, but this fits...
      const float lil_a = (1.0f + r/(fake_f*fake_f));
      p = p * lil_a * lil_a;

      // rescale to 1.0
#if HDR
      p /= (16384.0f-black_level);
#else
      p /= (1024.0f-black_level);
#endif

      // digital gain
      p *= digital_gain;

      // use both green channels
#if BAYER_FLIP == 3
      float3 c1 = (float3)(p.s3, (p.s1+p.s2)/2.0f, p.s0);
#elif BAYER_FLIP == 2
      float3 c1 = (float3)(p.s2, (p.s0+p.s3)/2.0f, p.s1);
#elif BAYER_FLIP == 1
      float3 c1 = (float3)(p.s1, (p.s0+p.s3)/2.0f, p.s2);
#elif BAYER_FLIP == 0
      float3 c1 = (float3)(p.s0, (p.s1+p.s2)/2.0f, p.s3);
#endif

      // color correction
      c1 = color_correct(c1);

#if HDR
      // srgb gamma isn't right for YUV, so it's disabled for now
      c1 = srgb_gamma(c1);
#endif

      // output BGR
      rgbs[px] = convert_uchar3_sat(c1.zyx * 255.0f);
    }

    const int yi_start = mad24(rgby_y, RGB_WIDTH, uv_x * 2);
    uchar2 yy = (uchar2)(
      RGB_TO_Y(rgbs[0].s2, rgbs[0].s1, rgbs[0].s0),
      RGB_TO_Y(rgbs[1].s2, rgbs[1].s1, rgbs[1].s0)
    );
    vstore2(yy, 0, out + yi_start);

    uchar3 argb = hadd(rgbs[0], rgbs[1]);
    us[mad24(lid, UV_WIDTH, uv_x)] = RGB_TO_U(argb.s0, argb.s1, argb.s2);
    vs[mad24(lid, UV_WIDTH, uv_x)] = RGB_TO_V(argb.s0, argb.s1, argb.s2);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid & 1) {
    for (int uv_x = 0; uv_x < HALF_UV_WIDTH; uv_x++) {
      int ui = mad24(uv_y, UV_WIDTH, U_OFFSET + uv_x);
      int vi = mad24(uv_y, UV_WIDTH, V_OFFSET + uv_x);

      out[vi] = hadd(us[mad24(lid-1, UV_WIDTH, uv_x)], us[mad24(lid, UV_WIDTH, uv_x)]);
      out[ui] = hadd(vs[mad24(lid-1, UV_WIDTH, uv_x)], vs[mad24(lid, UV_WIDTH, uv_x)]);
    }
  } else {
    for (int uv_x = HALF_UV_WIDTH; uv_x < UV_WIDTH; uv_x++) {
      int ui = mad24(uv_y, UV_WIDTH, U_OFFSET + uv_x);
      int vi = mad24(uv_y, UV_WIDTH, V_OFFSET + uv_x);

      out[vi] = hadd(us[mad24(lid, UV_WIDTH, uv_x)], us[mad24(lid+1, UV_WIDTH, uv_x)]);
      out[ui] = hadd(vs[mad24(lid, UV_WIDTH, uv_x)], vs[mad24(lid+1, UV_WIDTH, uv_x)]);
    }
  }
}
