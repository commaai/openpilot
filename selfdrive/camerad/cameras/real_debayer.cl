#ifdef HALF_AS_FLOAT
#define half float
#define half2 float2
#define half3 float3
#define half4 float4
#else
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define UV_WIDTH RGB_WIDTH / 2
#define UV_HEIGHT RGB_HEIGHT / 2
#define U_OFFSET RGB_WIDTH * RGB_HEIGHT
#define V_OFFSET RGB_WIDTH * RGB_HEIGHT + UV_WIDTH * UV_HEIGHT

#define RGB_TO_Y(r, g, b) ((((mul24(b, 13) + mul24(g, 65) + mul24(r, 33)) + 64) >> 7) + 16)
#define RGB_TO_U(r, g, b) ((mul24(b, 56) - mul24(g, 37) - mul24(r, 19) + 0x8080) >> 8)
#define RGB_TO_V(r, g, b) ((mul24(r, 56) - mul24(g, 47) - mul24(b, 9) + 0x8080) >> 8)
#define AVERAGE(x, y, z, w) ((convert_ushort(x) + convert_ushort(y) + convert_ushort(z) + convert_ushort(w) + 1) >> 1)

// post wb CCM
const __constant half3 color_correction_0 = (half3)(1.82717181, -0.31231438, 0.07307673);
const __constant half3 color_correction_1 = (half3)(-0.5743977, 1.36858544, -0.53183455);
const __constant half3 color_correction_2 = (half3)(-0.25277411, -0.05627105, 1.45875782);

// tone mapping params
const half gamma_k = 0.75;
const half gamma_b = 0.125;
const half mp = 0.01; // ideally midpoint should be adaptive
const half rk = 9 - 100*mp;

inline half3 gamma_apply(half3 x) {
  // poly approximation for s curve
  return (x > mp) ?
    ((rk * (x-mp) * (1-(gamma_k*mp+gamma_b)) * (1+1/(rk*(1-mp))) / (1+rk*(x-mp))) + gamma_k*mp + gamma_b) :
    ((rk * (x-mp) * (gamma_k*mp+gamma_b) * (1+1/(rk*mp)) / (1-rk*(x-mp))) + gamma_k*mp + gamma_b);
}

inline half3 color_correct(half3 rgb) {
  half3 ret = (half)rgb.x * color_correction_0;
  ret += (half)rgb.y * color_correction_1;
  ret += (half)rgb.z * color_correction_2;
  return gamma_apply(ret);
}

inline half get_vignetting_s(float r) {
  if (r < 62500) {
    return (half)(1.0f + 0.0000008f*r);
  } else if (r < 490000) {
    return (half)(0.9625f + 0.0000014f*r);
  } else if (r < 1102500) {
    return (half)(1.26434f + 0.0000000000016f*r*r);
  } else {
    return (half)(0.53503625f + 0.0000000000022f*r*r);
  }
}

inline half val_from_10(const uchar * source, int gx, int gy, half black_level) {
  // parse 12bit
  int start = gy * FRAME_STRIDE + (3 * (gx / 2)) + (FRAME_STRIDE * FRAME_OFFSET);
  int offset = gx % 2;
  uint major = (uint)source[start + offset] << 4;
  uint minor = (source[start + 2] >> (4 * offset)) & 0xf;
  half pv = ((half)(major + minor)) / 4.0;

  // normalize
  pv = max((half)0.0, pv - black_level);
  pv /= (1024.0 - black_level);

  // correct vignetting
  if (CAM_NUM == 1) { // fcamera
    gx = (gx - RGB_WIDTH/2);
    gy = (gy - RGB_HEIGHT/2);
    pv *= get_vignetting_s(gx*gx + gy*gy);
  }

  pv = clamp(pv, (half)0.0, (half)1.0);
  return pv;
}

inline half get_k(half a, half b, half c, half d) {
  return 2.0 - (fabs(a - b) + fabs(c - d));
}

__kernel void debayer10(const __global uchar * in,
                        __global uchar * out,
                        __local half * cached,
                        float black_level
                       )
{
  const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);

  const int lid_x = get_local_id(0);
  const int lid_y = get_local_id(1);

  const int localRowLen = mad24(get_local_size(0), 2, 2); // 2 padding
  const int localColLen = mad24(get_local_size(1), 2, 2);

  const int x_global = mul24(gid_x, 2);
  const int y_global = mul24(gid_y, 2);

  const int x_local = mad24(lid_x, 2, 1);
  const int y_local = mad24(lid_y, 2, 1);

  const int x_global_mod = (gid_x == 0 || gid_x == get_global_size(0) - 1) ? -1: 1;
  const int y_global_mod = (gid_y == 0 || gid_y == get_global_size(1) - 1) ? -1: 1;

  int localColOffset = 0;
  int globalColOffset;

  cached[mad24(y_local + 0, localRowLen, x_local + 0)] = val_from_10(in, x_global + 0, y_global + 0, black_level);
  cached[mad24(y_local + 0, localRowLen, x_local + 1)] = val_from_10(in, x_global + 1, y_global + 0, black_level);
  cached[mad24(y_local + 1, localRowLen, x_local + 0)] = val_from_10(in, x_global + 0, y_global + 1, black_level);
  cached[mad24(y_local + 1, localRowLen, x_local + 1)] = val_from_10(in, x_global + 1, y_global + 1, black_level);

  if (lid_x == 0) {  // left edge
    localColOffset = -1;
    globalColOffset = -x_global_mod;
    cached[mad24(y_local + 0, localRowLen, x_local - 1)] = val_from_10(in, x_global - x_global_mod, y_global + 0, black_level);
    cached[mad24(y_local + 1, localRowLen, x_local - 1)] = val_from_10(in, x_global - x_global_mod, y_global + 1, black_level);
  } else if (lid_x == get_local_size(0) - 1) {  // right edge
    localColOffset = 2;
    globalColOffset = x_global_mod + 1;
    cached[mad24(y_local + 0, localRowLen, x_local + 2)] = val_from_10(in, x_global + x_global_mod + 1, y_global + 0, black_level);
    cached[mad24(y_local + 1, localRowLen, x_local + 2)] = val_from_10(in, x_global + x_global_mod + 1, y_global + 1, black_level);
  }

  if (lid_y == 0) {  // top row
    cached[mad24(y_local - 1, localRowLen, x_local + 0)] = val_from_10(in, x_global + 0, y_global - y_global_mod, black_level);
    cached[mad24(y_local - 1, localRowLen, x_local + 1)] = val_from_10(in, x_global + 1, y_global - y_global_mod, black_level);
    if (localColOffset != 0) {  // cache corners
      cached[mad24(y_local - 1, localRowLen, x_local + localColOffset)] = val_from_10(in, x_global + globalColOffset, y_global - y_global_mod, black_level);
    }
  } else if (lid_y == get_local_size(1) - 1) {  // bottom row
    cached[mad24(y_local + 2, localRowLen, x_local + 0)] = val_from_10(in, x_global + 0, y_global + y_global_mod + 1, black_level);
    cached[mad24(y_local + 2, localRowLen, x_local + 1)] = val_from_10(in, x_global + 1, y_global + y_global_mod + 1, black_level);
    if (localColOffset != 0) {  // cache corners
      cached[mad24(y_local + 2, localRowLen, x_local + localColOffset)] = val_from_10(in, x_global + globalColOffset, y_global + y_global_mod + 1, black_level);
    }
  }

  // sync
  barrier(CLK_LOCAL_MEM_FENCE);

  half3 rgb;
  uchar3 rgb_out[4];

  const half4 va = vload4(0, cached + mad24(lid_y * 2 + 0, localRowLen, lid_x * 2));
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
  rgb_out[0] = convert_uchar3_sat(color_correct(clamp(rgb, 0.0, 1.0)) * 255.0);

  const half k11 = get_k(va.s1, vc.s1, va.s3, vc.s3);
  const half k12 = get_k(va.s2, vb.s1, vb.s3, vc.s2);
  const half k13 = get_k(va.s1, va.s3, vc.s1, vc.s3);
  const half k14 = get_k(va.s2, vb.s3, vc.s2, vb.s1);
  rgb.x = vb.s2; // R
  rgb.y = (k11*(va.s2+vc.s2)*0.5+k13*(vb.s3+vb.s1)*0.5)/(k11+k13); // G_R
  rgb.z = (k12*(va.s3+vc.s1)*0.5+k14*(va.s1+vc.s3)*0.5)/(k12+k14); // B_R
  rgb_out[1] = convert_uchar3_sat(color_correct(clamp(rgb, 0.0, 1.0)) * 255.0);

  const half k21 = get_k(vb.s0, vd.s0, vb.s2, vd.s2);
  const half k22 = get_k(vb.s1, vc.s0, vc.s2, vd.s1);
  const half k23 = get_k(vb.s0, vb.s2, vd.s0, vd.s2);
  const half k24 = get_k(vb.s1, vc.s2, vd.s1, vc.s0);
  rgb.x = (k22*(vb.s2+vd.s0)*0.5+k24*(vb.s0+vd.s2)*0.5)/(k22+k24); // R_B
  rgb.y = (k21*(vb.s1+vd.s1)*0.5+k23*(vc.s2+vc.s0)*0.5)/(k21+k23); // G_B
  rgb.z = vc.s1; // B
  rgb_out[2] = convert_uchar3_sat(color_correct(clamp(rgb, 0.0, 1.0)) * 255.0);

  const half k31 = get_k(vb.s1, vc.s2, vb.s3, vc.s2);
  const half k32 = get_k(vb.s3, vc.s2, vd.s3, vc.s2);
  const half k33 = get_k(vd.s1, vc.s2, vd.s3, vc.s2);
  const half k34 = get_k(vb.s1, vc.s2, vd.s1, vc.s2);
  rgb.x = (k31*vb.s2+k33*vd.s2)/(k31+k33); // R_G2
  rgb.y = vc.s2; // G2(B)
  rgb.z = (k32*vc.s3+k34*vc.s1)/(k32+k34); // B_G2
  rgb_out[3] = convert_uchar3_sat(color_correct(clamp(rgb, 0.0, 1.0)) * 255.0);

  // write ys
  uchar2 yy = (uchar2)(
    RGB_TO_Y(rgb_out[0].s0, rgb_out[0].s1, rgb_out[0].s2),
    RGB_TO_Y(rgb_out[1].s0, rgb_out[1].s1, rgb_out[1].s2)
  );
  vstore2(yy, 0, out + mad24(gid_y * 2, RGB_WIDTH, gid_x * 2));
  yy = (uchar2)(
    RGB_TO_Y(rgb_out[2].s0, rgb_out[2].s1, rgb_out[2].s2),
    RGB_TO_Y(rgb_out[3].s0, rgb_out[3].s1, rgb_out[3].s2)
  );
  vstore2(yy, 0, out + mad24(gid_y * 2 + 1, RGB_WIDTH, gid_x * 2));

  // write uvs
  const short ar = AVERAGE(rgb_out[0].s0, rgb_out[1].s0, rgb_out[2].s0, rgb_out[3].s0);
  const short ag = AVERAGE(rgb_out[0].s1, rgb_out[1].s1, rgb_out[2].s1, rgb_out[3].s1);
  const short ab = AVERAGE(rgb_out[0].s2, rgb_out[1].s2, rgb_out[2].s2, rgb_out[3].s2);
  out[U_OFFSET + mad24(gid_y, UV_WIDTH, gid_x)] = RGB_TO_U(ar, ag, ab);
  out[V_OFFSET + mad24(gid_y, UV_WIDTH, gid_x)] = RGB_TO_V(ar, ag, ab);
}
