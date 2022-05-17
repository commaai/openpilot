#define UV_WIDTH RGB_WIDTH / 2
#define UV_HEIGHT RGB_HEIGHT / 2
#define U_OFFSET RGB_WIDTH * RGB_HEIGHT
#define V_OFFSET RGB_WIDTH * RGB_HEIGHT + UV_WIDTH * UV_HEIGHT

#define RGB_TO_Y(r, g, b) ((((mul24(b, 13) + mul24(g, 65) + mul24(r, 33)) + 64) >> 7) + 16)
#define RGB_TO_U(r, g, b) ((mul24(b, 56) - mul24(g, 37) - mul24(r, 19) + 0x8080) >> 8)
#define RGB_TO_V(r, g, b) ((mul24(r, 56) - mul24(g, 47) - mul24(b, 9) + 0x8080) >> 8)
#define AVERAGE(x, y, z, w) ((convert_ushort(x) + convert_ushort(y) + convert_ushort(z) + convert_ushort(w) + 1) >> 1)

float3 color_correct(float3 rgb) {
  // color correction
  float3 x = rgb.x * (float3)(1.82717181, -0.31231438, 0.07307673);
  x += rgb.y * (float3)(-0.5743977, 1.36858544, -0.53183455);
  x += rgb.z * (float3)(-0.25277411, -0.05627105, 1.45875782);

  // tone mapping params
  const float gamma_k = 0.75;
  const float gamma_b = 0.125;
  const float mp = 0.01; // ideally midpoint should be adaptive
  const float rk = 9 - 100*mp;

  // poly approximation for s curve
  return (x > mp) ?
    ((rk * (x-mp) * (1-(gamma_k*mp+gamma_b)) * (1+1/(rk*(1-mp))) / (1+rk*(x-mp))) + gamma_k*mp + gamma_b) :
    ((rk * (x-mp) * (gamma_k*mp+gamma_b) * (1+1/(rk*mp)) / (1-rk*(x-mp))) + gamma_k*mp + gamma_b);
}

float get_vignetting_s(float r) {
  if (r < 62500) {
    return (1.0f + 0.0000008f*r);
  } else if (r < 490000) {
    return (0.9625f + 0.0000014f*r);
  } else if (r < 1102500) {
    return (1.26434f + 0.0000000000016f*r*r);
  } else {
    return (0.53503625f + 0.0000000000022f*r*r);
  }
}

ushort val_from_12(const uchar * source, int gx, int gy) {
  // parse 12bit
  int start = gy * FRAME_STRIDE + (3 * (gx / 2)) + (FRAME_STRIDE * FRAME_OFFSET);
  int offset = gx % 2;
  uint major = (uint)source[start + offset] << 4;
  uint minor = (source[start + 2] >> (4 * offset)) & 0xf;
  return (major + minor);
}

float4 precolor_correct(ushort4 pvs, int gx, int gy) {
  // normalize
  float4 pv = max(0.0, convert_float4(pvs) - 168.0) / (4096.0 - 168.0);

  // correct vignetting
  #if VIGNETTING
    gx = (gx - RGB_WIDTH/2);
    gy = (gy - RGB_HEIGHT/2);
    pv *= get_vignetting_s(gx*gx + gy*gy);
  #endif

  return clamp(pv, 0.0, 1.0);
}

float get_k(float a, float b, float c, float d) {
  return 2.0 - (fabs(a - b) + fabs(c - d));
}

__kernel void debayer10(const __global uchar * in, __global uchar * out)
{
  const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);

  float3 rgb;
  uchar3 rgb_out[4];

  // load 4x4 conv data
  ushort4 vas, vbs, vcs, vds;

  const int gid_x2 = gid_x*2;
  const int gid_x2m1 = gid_x == 0 ? gid_x2+1 : gid_x2-1;
  const int gid_x2p2 = gid_x == (RGB_WIDTH-2) ? gid_x2+0 : gid_x2+2;
  const int gid_y2 = gid_y*2;
  const int gid_y2m1 = gid_y == 0 ? gid_y2+1 : gid_y2-1;
  const int gid_y2p2 = gid_y == (RGB_HEIGHT-2) ? gid_y2+0 : gid_y2+2;

  vas.s0 = val_from_12(in, gid_x2m1, gid_y2m1);
  vas.s1 = val_from_12(in, gid_x2+0, gid_y2m1);
  vas.s2 = val_from_12(in, gid_x2+1, gid_y2m1);
  vas.s3 = val_from_12(in, gid_x2p2, gid_y2m1);

  vbs.s0 = val_from_12(in, gid_x2m1, gid_y*2+0);
  vbs.s1 = val_from_12(in, gid_x2+0, gid_y*2+0); // G(R)
  vbs.s2 = val_from_12(in, gid_x2+1, gid_y*2+0); // R
  vbs.s3 = val_from_12(in, gid_x2p2, gid_y*2+0);

  vcs.s0 = val_from_12(in, gid_x2m1, gid_y*2+1);
  vcs.s1 = val_from_12(in, gid_x2+0, gid_y*2+1); // B
  vcs.s2 = val_from_12(in, gid_x2+1, gid_y*2+1); // G(B)
  vcs.s3 = val_from_12(in, gid_x2p2, gid_y*2+1);

  vds.s0 = val_from_12(in, gid_x2m1, gid_y2p2);
  vds.s1 = val_from_12(in, gid_x2+0, gid_y2p2);
  vds.s2 = val_from_12(in, gid_x2+1, gid_y2p2);
  vds.s3 = val_from_12(in, gid_x2p2, gid_y2p2);

  float4 va, vb, vc, vd;
  va = precolor_correct(vas, gid_x2, gid_y2);
  vb = precolor_correct(vbs, gid_x2, gid_y2);
  vc = precolor_correct(vcs, gid_x2, gid_y2);
  vd = precolor_correct(vds, gid_x2, gid_y2);

  // a simplified version of https://opensignalprocessingjournal.com/contents/volumes/V6/TOSIGPJ-6-1/TOSIGPJ-6-1.pdf
  const float k01 = get_k(va.s0, vb.s1, va.s2, vb.s1);
  const float k02 = get_k(va.s2, vb.s1, vc.s2, vb.s1);
  const float k03 = get_k(vc.s0, vb.s1, vc.s2, vb.s1);
  const float k04 = get_k(va.s0, vb.s1, vc.s0, vb.s1);
  rgb.x = (k02*vb.s2+k04*vb.s0)/(k02+k04); // R_G1
  rgb.y = vb.s1; // G1(R)
  rgb.z = (k01*va.s1+k03*vc.s1)/(k01+k03); // B_G1
  rgb_out[0] = convert_uchar3_sat(color_correct(clamp(rgb, 0.0, 1.0)) * 255.0);

  const float k11 = get_k(va.s1, vc.s1, va.s3, vc.s3);
  const float k12 = get_k(va.s2, vb.s1, vb.s3, vc.s2);
  const float k13 = get_k(va.s1, va.s3, vc.s1, vc.s3);
  const float k14 = get_k(va.s2, vb.s3, vc.s2, vb.s1);
  rgb.x = vb.s2; // R
  rgb.y = (k11*(va.s2+vc.s2)*0.5+k13*(vb.s3+vb.s1)*0.5)/(k11+k13); // G_R
  rgb.z = (k12*(va.s3+vc.s1)*0.5+k14*(va.s1+vc.s3)*0.5)/(k12+k14); // B_R
  rgb_out[1] = convert_uchar3_sat(color_correct(clamp(rgb, 0.0, 1.0)) * 255.0);

  const float k21 = get_k(vb.s0, vd.s0, vb.s2, vd.s2);
  const float k22 = get_k(vb.s1, vc.s0, vc.s2, vd.s1);
  const float k23 = get_k(vb.s0, vb.s2, vd.s0, vd.s2);
  const float k24 = get_k(vb.s1, vc.s2, vd.s1, vc.s0);
  rgb.x = (k22*(vb.s2+vd.s0)*0.5+k24*(vb.s0+vd.s2)*0.5)/(k22+k24); // R_B
  rgb.y = (k21*(vb.s1+vd.s1)*0.5+k23*(vc.s2+vc.s0)*0.5)/(k21+k23); // G_B
  rgb.z = vc.s1; // B
  rgb_out[2] = convert_uchar3_sat(color_correct(clamp(rgb, 0.0, 1.0)) * 255.0);

  const float k31 = get_k(vb.s1, vc.s2, vb.s3, vc.s2);
  const float k32 = get_k(vb.s3, vc.s2, vd.s3, vc.s2);
  const float k33 = get_k(vd.s1, vc.s2, vd.s3, vc.s2);
  const float k34 = get_k(vb.s1, vc.s2, vd.s1, vc.s2);
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
