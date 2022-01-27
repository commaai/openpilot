#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define UV_WIDTH RGB_WIDTH / 2
#define UV_HEIGHT RGB_HEIGHT / 2
#define U_OFFSET RGB_WIDTH * RGB_HEIGHT
#define V_OFFSET RGB_WIDTH * RGB_HEIGHT + UV_WIDTH * UV_HEIGHT

#define RGB_TO_Y(r, g, b) ((((mul24(b, 13) + mul24(g, 65) + mul24(r, 33)) + 64) >> 7) + 16)
#define RGB_TO_U(r, g, b) ((mul24(b, 56) - mul24(g, 37) - mul24(r, 19) + 0x8080) >> 8)
#define RGB_TO_V(r, g, b) ((mul24(r, 56) - mul24(g, 47) - mul24(b, 9) + 0x8080) >> 8)
#define AVERAGE(x, y, z, w) ((convert_ushort(x) + convert_ushort(y) + convert_ushort(z) + convert_ushort(w) + 1) >> 1)

#define CP 0.01h
#define CPK 0.75h
#define CPB 0.125h
#define RK 8.0h  // 9 - 100*CP

const half black_level = 42.0h;

const __constant half3 color_correction[3] = {
  // post wb CCM
  (half3)(1.82717181, -0.31231438, 0.07307673),
  (half3)(-0.5743977, 1.36858544, -0.53183455),
  (half3)(-0.25277411, -0.05627105, 1.45875782),
};


inline half3 mf(half3 x) {
  return x > CP ?
    (RK * (x-CP) * (1.0h-(CPK*CP+CPB)) * (1.0h+1.0h/(RK*(1.0h-CP))) / (1.0h+RK*(x-CP))) + CPK*CP + CPB :
    (RK * (x-CP) * (CPK*CP+CPB) * (1.0h+1.0h/(RK*CP)) / (1.0h-RK*(x-CP))) + CPK*CP + CPB;
}

inline uchar3 color_correct(half3 rgb) {
  half3 ret = rgb.x * color_correction[0];
  ret += rgb.y * color_correction[1];
  ret += rgb.z * color_correction[2];
  ret = mf(ret);
  return convert_uchar3_sat(ret * 255.0h);
}

half val_from_10(const uchar * const source, int gx, int gy) {
  // parse 10bit
  const int start = gy * FRAME_STRIDE + (5 * (gx / 4));
  const int offset = gx % 4;
  const uint major = (uint)source[start + offset] << 2;
  const uint minor = (source[start + 4] >> (2 * offset)) & 3;
  half pv = (half)(major + minor);

  // normalize
  pv = max(0.0h, pv - black_level);
  pv *= 0.00101833h; // /= (1024.0f - black_level);

  // correct vignetting
  if (CAM_NUM == 1) { // fcamera
    gx = (gx - RGB_WIDTH/2);
    gy = (gy - RGB_HEIGHT/2);
    float r = gx*gx + gy*gy;
    half s;
    if (r < 62500) {
      s = (half)(1.0f + 0.0000008f*r);
    } else if (r < 490000) {
      s = (half)(0.9625f + 0.0000014f*r);
    } else if (r < 1102500) {
      s = (half)(1.26434f + 0.0000000000016f*r*r);
    } else {
      s = (half)(0.53503625f + 0.0000000000022f*r*r);
    }
    pv = s * pv;
  }

  pv = clamp(0.0h, 1.0h, pv);
  return pv;
}

half4 vals_from_10(const uchar * const source, const int gx, const int gy) {
  // parse 4x 10bit, requires gx % 4 == 0
  const int start = mad24(gy, FRAME_STRIDE, (5 * (gx / 4)));

  const uchar4 l_in = vload4(0, source + start);
  const uchar l_pad = source[start + 4];

  half4 pvs;
  pvs.s0 = (half)(((uint)l_in.s0 << 2) + ((l_pad >> 0) & 3));
  pvs.s1 = (half)(((uint)l_in.s1 << 2) + ((l_pad >> 2) & 3));
  pvs.s2 = (half)(((uint)l_in.s2 << 2) + ((l_pad >> 4) & 3));
  pvs.s3 = (half)(((uint)l_in.s3 << 2) + ((l_pad >> 6) & 3));

  // normalize
  pvs = max(0.0h, pvs - black_level);
  pvs *= 0.00101833h; // /= (1024.0f - black_level);

  // correct vignetting
  if (CAM_NUM == 1) { // fcamera
    const float r = ((gx+1)-(RGB_WIDTH/2))*((gx+2)-(RGB_WIDTH/2)) + (gy-(RGB_HEIGHT/2))*(gy-(RGB_HEIGHT/2));
    if (r < 62500) {
      pvs *= 1.0f + 0.0000008f * r;
    } else if (r < 490000) {
      pvs *= 0.9625f + 0.0000014f*r;
    } else if (r < 1102500) {
      pvs *= 1.26434f + 0.0000000000016f*r*r;
    } else {
      pvs *= 0.53503625f + 0.0000000000022f*r*r;
    }
  }

  pvs = clamp(0.0h, 1.0h, pvs);
  return pvs;
}

inline half get_k(half a, half b, half c, half d) {
  // get_k(va.s0, vb.s1, va.s2, vb.s1);
  return 2.0h - (fabs(a - b) + fabs(c - d));
}

__kernel void debayer10(__global const uchar * const in, __global uchar * out, __local half * cached) {
  const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);

  const int lid_x = get_local_id(0);
  const int lid_y = get_local_id(1);

  const int localRowLen = mad24(get_local_size(0), 2, 2); // 2 padding
  const int localColLen = mad24(get_local_size(1), 2, 2);

  const int y_global = mad24(gid_y, 2, gid_x & 1);
  const int x_global = (gid_x / 2) * 4;

  const int y_local = mad24(lid_y, 2, lid_x & 1) + 1;
  const int x_local = (lid_x / 2) * 4 + 1;

  vstore4(vals_from_10(in, x_global, y_global), 0, cached + mad24(y_local, localRowLen, x_local));

  if (lid_y == 0 && (lid_x & 1) == 0) {
    vstore4(vals_from_10(in, x_global, y_global - 1), 0, cached + mad24(y_local - 1, localRowLen, x_local));
  } else if (lid_y == get_local_size(1) - 1 && lid_x & 1) {
    vstore4(vals_from_10(in, x_global, y_global + 1), 0, cached + mad24(y_local + 1, localRowLen, x_local));
  }

  if (lid_x <= 1) {
    cached[mad24(y_local, localRowLen, x_local - 1)] = val_from_10(in, x_global - 1, y_global);
  } else if (lid_x >= get_local_size(0) - 2) {
    cached[mad24(y_local, localRowLen, x_local + 4)] = val_from_10(in, x_global + 4, y_global);
  }

  // sync
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid_x == 0 && lid_y == 0) {
    cached[0] = val_from_10(in, x_global - 1, y_global - 1);
  } else if (lid_x == get_local_size(0) - 1 && lid_y == 0) {
    cached[localRowLen - 1] = val_from_10(in, x_global + 4, y_global - 2);
  } else if (lid_x == 0 && lid_y == get_local_size(1) - 1) {
    cached[mul24(localColLen - 1, localRowLen)] = val_from_10(in, x_global - 1, y_global + 2);
  } else if (lid_x == get_local_size(0) - 1 && lid_y == get_local_size(1) - 1) {
    cached[mad24(localColLen, localRowLen, -1)] = val_from_10(in, x_global + 4, y_global + 1);
  }

  half3 rgb;
  uchar3 rgb_out[4];

  const half4 va = vload4(0, cached + mad24(lid_y * 2, localRowLen, lid_x * 2));
  const half4 vb = vload4(0, cached + mad24(lid_y * 2 + 1, localRowLen, lid_x * 2));
  const half4 vc = vload4(0, cached + mad24(lid_y * 2 + 2, localRowLen, lid_x * 2));
  const half4 vd = vload4(0, cached + mad24(lid_y * 2 + 3, localRowLen, lid_x * 2));

  // a simplified version of https://opensignalprocessingjournal.com/contents/volumes/V6/TOSIGPJ-6-1/TOSIGPJ-6-1.pdf

  const half k01 = get_k(va.s0, vb.s1, va.s2, vb.s1);
  const half k02 = get_k(va.s2, vb.s1, vc.s2, vb.s1);
  const half k03 = get_k(vc.s0, vb.s1, vc.s2, vb.s1);
  const half k04 = get_k(va.s0, vb.s1, vc.s0, vb.s1);
  rgb.x = (k02*vb.s2+k04*vb.s0)/(k02+k04); // R_G1
  rgb.y = vb.s1; // G1(R)
  rgb.z = (k01*va.s1+k03*vc.s1)/(k01+k03); // B_G1
  rgb_out[0] = color_correct(clamp(0.0h, 1.0h, rgb));

  const half k11 = get_k(va.s1, vc.s1, va.s3, vc.s3);
  const half k12 = get_k(va.s2, vb.s1, vb.s3, vc.s2);
  const half k13 = get_k(va.s1, va.s3, vc.s1, vc.s3);
  const half k14 = get_k(va.s2, vb.s3, vc.s2, vb.s1);
  rgb.x = vb.s2; // R
  rgb.y = (k11*(va.s2+vc.s2)*0.5+k13*(vb.s3+vb.s1)*0.5)/(k11+k13); // G_R
  rgb.z = (k12*(va.s3+vc.s1)*0.5+k14*(va.s1+vc.s3)*0.5)/(k12+k14); // B_R
  rgb_out[1] = color_correct(clamp(0.0h, 1.0h, rgb));

  const half k21 = get_k(vb.s0, vd.s0, vb.s2, vd.s2);
  const half k22 = get_k(vb.s1, vc.s0, vc.s2, vd.s1);
  const half k23 = get_k(vb.s0, vb.s2, vd.s0, vd.s2);
  const half k24 = get_k(vb.s1, vc.s2, vd.s1, vc.s0);
  rgb.x = (k22*(vb.s2+vd.s0)*0.5+k24*(vb.s0+vd.s2)*0.5)/(k22+k24); // R_B
  rgb.y = (k21*(vb.s1+vd.s1)*0.5+k23*(vc.s2+vc.s0)*0.5)/(k21+k23); // G_B
  rgb.z = vc.s1; // B
  rgb_out[2] = color_correct(clamp(0.0h, 1.0h, rgb));

  const half k31 = get_k(vb.s1, vc.s2, vb.s3, vc.s2);
  const half k32 = get_k(vb.s3, vc.s2, vd.s3, vc.s2);
  const half k33 = get_k(vd.s1, vc.s2, vd.s3, vc.s2);
  const half k34 = get_k(vb.s1, vc.s2, vd.s1, vc.s2);
  rgb.x = (k31*vb.s2+k33*vd.s2)/(k31+k33); // R_G2
  rgb.y = vc.s2; // G2(B)
  rgb.z = (k32*vc.s3+k34*vc.s1)/(k32+k34); // B_G2
  rgb_out[3] = color_correct(clamp(0.0h, 1.0h, rgb));

  // write ys
  for (int i = 0; i < 2; i++) {
    const int y_global = mad24(gid_y, 2, i);
    const int yi_start = mad24(y_global, RGB_WIDTH, gid_x * 2);
    uchar2 yy = (uchar2)(
      RGB_TO_Y(rgb_out[0].s2, rgb_out[0].s1, rgb_out[0].s0),
      RGB_TO_Y(rgb_out[1].s2, rgb_out[1].s1, rgb_out[1].s0)
    );
    vstore2(yy, 0, out + yi_start);
  }

  // write uvs
  const short ar = AVERAGE(rgb_out[0].s0, rgb_out[1].s0, rgb_out[2].s0, rgb_out[3].s0);
  const short ag = AVERAGE(rgb_out[0].s1, rgb_out[1].s1, rgb_out[2].s1, rgb_out[3].s1);
  const short ab = AVERAGE(rgb_out[0].s2, rgb_out[1].s2, rgb_out[2].s2, rgb_out[3].s2);
  out[U_OFFSET + mad24(gid_y, UV_WIDTH, gid_x)] = RGB_TO_U(ar, ag, ab);
  out[V_OFFSET + mad24(gid_y, UV_WIDTH, gid_x)] = RGB_TO_V(ar, ag, ab);
}
