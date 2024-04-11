#include "ar0231_cl.h"
#include "ox03c10_cl.h"
#include "os04c10_cl.h"

#define UV_WIDTH RGB_WIDTH / 2
#define UV_HEIGHT RGB_HEIGHT / 2

#define RGB_TO_Y(r, g, b) ((((mul24(b, 13) + mul24(g, 65) + mul24(r, 33)) + 64) >> 7) + 16)
#define RGB_TO_U(r, g, b) ((mul24(b, 56) - mul24(g, 37) - mul24(r, 19) + 0x8080) >> 8)
#define RGB_TO_V(r, g, b) ((mul24(r, 56) - mul24(g, 47) - mul24(b, 9) + 0x8080) >> 8)
#define AVERAGE(x, y, z, w) ((convert_ushort(x) + convert_ushort(y) + convert_ushort(z) + convert_ushort(w) + 1) >> 1)

#if defined(BGGR)
  #define ROW_READ_ORDER (int[]){3, 2, 1, 0}
  #define RGB_WRITE_ORDER (int[]){2, 3, 0, 1}
#else
  #define ROW_READ_ORDER (int[]){0, 1, 2, 3}
  #define RGB_WRITE_ORDER (int[]){0, 1, 2, 3}
#endif

float get_vignetting_s(float r) {
  #if IS_OS
    r = r / 2.2545f;
  #endif
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

float4 val4_from_12(uchar8 pvs, float gain) {
  uint4 parsed = (uint4)(((uint)pvs.s0<<4) + (pvs.s1>>4),  // is from the previous 10 bit
                         ((uint)pvs.s2<<4) + (pvs.s4&0xF),
                         ((uint)pvs.s3<<4) + (pvs.s4>>4),
                         ((uint)pvs.s5<<4) + (pvs.s7&0xF));
  #if IS_OX
  // PWL
  //float4 pv = (convert_float4(parsed) - 64.0) / (4096.0 - 64.0);
  float4 pv = {ox03c10_lut[parsed.s0], ox03c10_lut[parsed.s1], ox03c10_lut[parsed.s2], ox03c10_lut[parsed.s3]};

  // it's a 24 bit signal, center in the middle 8 bits
  return clamp(pv*gain*256.0, 0.0, 1.0);
  #else // AR
  // normalize and scale
  float4 pv = (convert_float4(parsed) - 168.0) / (4096.0 - 168.0);
  return clamp(pv*gain, 0.0, 1.0);
  #endif

}

float4 val4_from_10(uchar8 pvs, uchar ext, bool aligned, float gain) {
  uint4 parsed;
  if (aligned) {
    parsed = (uint4)(((uint)pvs.s0 << 2) + (pvs.s1 & 0b00000011),
                     ((uint)pvs.s2 << 2) + ((pvs.s6 & 0b11000000) / 64),
                     ((uint)pvs.s3 << 2) + ((pvs.s6 & 0b00110000) / 16),
                     ((uint)pvs.s4 << 2) + ((pvs.s6 & 0b00001100) / 4));
  } else {
    parsed = (uint4)(((uint)pvs.s0 << 2) + ((pvs.s3 & 0b00110000) / 16),
                     ((uint)pvs.s1 << 2) + ((pvs.s3 & 0b00001100) / 4),
                     ((uint)pvs.s2 << 2) + ((pvs.s3 & 0b00000011)),
                     ((uint)pvs.s4 << 2) + ((ext & 0b11000000) / 64));
  }

  float4 pv = (convert_float4(parsed) - 64.0) / (1024.0 - 64.0);
  return clamp(pv*gain, 0.0, 1.0);
}

float combine_pvs(float lv, float sv, int expo) {
  float svc = fmax(sv * expo, 61376.0);
  float svd = sv * fmin(expo, 8.0) / 8;

  if (expo > 64) {
    if (lv < 959) {
      return lv / (65536.0 - 64.0);
    } else {
      return (svc / 64) / (65536.0 - 64.0);
    }
  } else {
    if (lv > 32) {
      return (lv * 64 / fmax(expo, 8.0)) / (65536.0 - 64.0);
    } else {
      return svd / (65536.0 - 64.0);
    }
  }
}

float4 val4_from_10x2(uchar8 long_pvs, uchar long_ext, uchar8 short_pvs, uchar short_ext, bool aligned, float gain, int expo) {
  int8 parsed;
  if (aligned) {
    parsed = (int8)(((int)long_pvs.s0 << 2) + (long_pvs.s1 & 0b00000011),
                     ((int)long_pvs.s2 << 2) + ((long_pvs.s6 & 0b11000000) / 64),
                     ((int)long_pvs.s3 << 2) + ((long_pvs.s6 & 0b00110000) / 16),
                     ((int)long_pvs.s4 << 2) + ((long_pvs.s6 & 0b00001100) / 4),
                     ((int)short_pvs.s0 << 2) + (short_pvs.s1 & 0b00000011),
                     ((int)short_pvs.s2 << 2) + ((short_pvs.s6 & 0b11000000) / 64),
                     ((int)short_pvs.s3 << 2) + ((short_pvs.s6 & 0b00110000) / 16),
                     ((int)short_pvs.s4 << 2) + ((short_pvs.s6 & 0b00001100) / 4));
  } else {
    parsed = (int8)(((int)long_pvs.s0 << 2) + ((long_pvs.s3 & 0b00110000) / 16),
                     ((int)long_pvs.s1 << 2) + ((long_pvs.s3 & 0b00001100) / 4),
                     ((int)long_pvs.s2 << 2) + ((long_pvs.s3 & 0b00000011)),
                     ((int)long_pvs.s4 << 2) + ((long_ext & 0b11000000) / 64),
                     ((int)short_pvs.s0 << 2) + ((short_pvs.s3 & 0b00110000) / 16),
                     ((int)short_pvs.s1 << 2) + ((short_pvs.s3 & 0b00001100) / 4),
                     ((int)short_pvs.s2 << 2) + ((short_pvs.s3 & 0b00000011)),
                     ((int)short_pvs.s4 << 2) + ((short_ext & 0b11000000) / 64));
  }

  float8 pf = convert_float8(parsed - 64);
  float4 pv;
  pv.s0 = combine_pvs(pf.s0, pf.s4, expo);
  pv.s1 = combine_pvs(pf.s1, pf.s5, expo);
  pv.s2 = combine_pvs(pf.s2, pf.s6, expo);
  pv.s3 = combine_pvs(pf.s3, pf.s7, expo);
  return clamp(pv*gain, 0.0, 1.0);
}

float get_k(float a, float b, float c, float d) {
  return 2.0 - (fabs(a - b) + fabs(c - d));
}

__kernel void process_raw(const __global uchar * in, __global uchar * out, int expo_time)
{
  const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);

  const int row_before_offset = (gid_y == 0) ? 2 : 0;
  const int row_after_offset = (gid_y == (RGB_HEIGHT/2 - 1)) ? 1 : 3;

  float3 rgb;
  uchar3 rgb_out[4];

  int start_idx;
  #if IS_10BIT
    bool aligned10;
    if (gid_x % 2 == 0) {
      aligned10 = true;
      start_idx = (2 * gid_y - 1) * FRAME_STRIDE + (5 * gid_x / 2 - 2) + (FRAME_STRIDE * FRAME_OFFSET);
    } else {
      aligned10 = false;
      start_idx = (2 * gid_y - 1) * FRAME_STRIDE + (5 * (gid_x - 1) / 2 + 1) + (FRAME_STRIDE * FRAME_OFFSET);
    }
  #else
    start_idx = (2 * gid_y - 1) * FRAME_STRIDE + (3 * gid_x - 2) + (FRAME_STRIDE * FRAME_OFFSET);
  #endif

  // read in 8x4 chars
  uchar8 dat[4];
  dat[0] = vload8(0, in + start_idx + FRAME_STRIDE*row_before_offset);
  if (gid_x == 0 && gid_y == 0) {
    // this wasn't a problem due to extra rows
    dat[1] = vload8(0, in + start_idx + FRAME_STRIDE*1 + 2);
    dat[1] = (uchar8)(0, 0, dat[1].s0, dat[1].s1, dat[1].s2, dat[1].s3, dat[1].s4, dat[1].s5);
  } else {
    dat[1] = vload8(0, in + start_idx + FRAME_STRIDE*1);
  }
  dat[2] = vload8(0, in + start_idx + FRAME_STRIDE*2);
  dat[3] = vload8(0, in + start_idx + FRAME_STRIDE*row_after_offset);
  #if IS_HDR
    uchar8 short_dat[4];
    short_dat[0] = vload8(0, in + start_idx + FRAME_STRIDE*(row_before_offset+HDR_OFFSET/2) + FRAME_STRIDE/2);
    short_dat[1] = vload8(0, in + start_idx + FRAME_STRIDE*(1+HDR_OFFSET/2) + FRAME_STRIDE/2);
    short_dat[2] = vload8(0, in + start_idx + FRAME_STRIDE*(2+HDR_OFFSET/2) + FRAME_STRIDE/2);
    short_dat[3] = vload8(0, in + start_idx + FRAME_STRIDE*(row_after_offset+HDR_OFFSET/2) + FRAME_STRIDE/2);
  #endif

  // need extra bit for 10-bit
  #if IS_10BIT
    uchar extra[4];
    if (!aligned10) {
      extra[0] = in[start_idx + FRAME_STRIDE*row_before_offset + 8];
      extra[1] = in[start_idx + FRAME_STRIDE*1 + 8];
      extra[2] = in[start_idx + FRAME_STRIDE*2 + 8];
      extra[3] = in[start_idx + FRAME_STRIDE*row_after_offset + 8];
		}
    #if IS_HDR
      uchar short_extra[4];
      if (!aligned10) {
        short_extra[0] = in[start_idx + FRAME_STRIDE*(row_before_offset+HDR_OFFSET/2) + FRAME_STRIDE/2 + 8];
        short_extra[1] = in[start_idx + FRAME_STRIDE*(1+HDR_OFFSET/2) + FRAME_STRIDE/2 + 8];
        short_extra[2] = in[start_idx + FRAME_STRIDE*(2+HDR_OFFSET/2) + FRAME_STRIDE/2 + 8];
        short_extra[3] = in[start_idx + FRAME_STRIDE*(row_after_offset+HDR_OFFSET/2) + FRAME_STRIDE/2 + 8];
      }
    #endif
  #endif

  // correct vignetting
  #if VIGNETTING
    int gx = (gid_x*2 - RGB_WIDTH/2);
    int gy = (gid_y*2 - RGB_HEIGHT/2);
    const float gain = get_vignetting_s(gx*gx + gy*gy);
  #else
    const float gain = 1.0;
  #endif

  float4 v_rows[4];
  // parse into floats
  #if IS_10BIT
    #if IS_HDR
      v_rows[ROW_READ_ORDER[0]] = val4_from_10x2(dat[0], extra[0], short_dat[0], short_extra[0], aligned10, gain, expo_time);
      v_rows[ROW_READ_ORDER[1]] = val4_from_10x2(dat[1], extra[1], short_dat[1], short_extra[1], aligned10, gain, expo_time);
      v_rows[ROW_READ_ORDER[2]] = val4_from_10x2(dat[2], extra[2], short_dat[2], short_extra[2], aligned10, gain, expo_time);
      v_rows[ROW_READ_ORDER[3]] = val4_from_10x2(dat[3], extra[3], short_dat[3], short_extra[3], aligned10, gain, expo_time);
    #else
      v_rows[ROW_READ_ORDER[0]] = val4_from_10(dat[0], extra[0], aligned10, gain);
      v_rows[ROW_READ_ORDER[1]] = val4_from_10(dat[1], extra[1], aligned10, gain);
      v_rows[ROW_READ_ORDER[2]] = val4_from_10(dat[2], extra[2], aligned10, gain);
      v_rows[ROW_READ_ORDER[3]] = val4_from_10(dat[3], extra[3], aligned10, gain);
    #endif
  #else
    v_rows[ROW_READ_ORDER[0]] = val4_from_12(dat[0], gain);
    v_rows[ROW_READ_ORDER[1]] = val4_from_12(dat[1], gain);
    v_rows[ROW_READ_ORDER[2]] = val4_from_12(dat[2], gain);
    v_rows[ROW_READ_ORDER[3]] = val4_from_12(dat[3], gain);
  #endif

  // mirror padding
  if (gid_x == 0) {
    v_rows[0].s0 = v_rows[0].s2;
    v_rows[1].s0 = v_rows[1].s2;
    v_rows[2].s0 = v_rows[2].s2;
    v_rows[3].s0 = v_rows[3].s2;
  } else if (gid_x == RGB_WIDTH/2 - 1) {
    v_rows[0].s3 = v_rows[0].s1;
    v_rows[1].s3 = v_rows[1].s1;
    v_rows[2].s3 = v_rows[2].s1;
    v_rows[3].s3 = v_rows[3].s1;
  }

  // a simplified version of https://opensignalprocessingjournal.com/contents/volumes/V6/TOSIGPJ-6-1/TOSIGPJ-6-1.pdf
  const float k01 = get_k(v_rows[0].s0, v_rows[1].s1, v_rows[0].s2, v_rows[1].s1);
  const float k02 = get_k(v_rows[0].s2, v_rows[1].s1, v_rows[2].s2, v_rows[1].s1);
  const float k03 = get_k(v_rows[2].s0, v_rows[1].s1, v_rows[2].s2, v_rows[1].s1);
  const float k04 = get_k(v_rows[0].s0, v_rows[1].s1, v_rows[2].s0, v_rows[1].s1);
  rgb.x = (k02*v_rows[1].s2+k04*v_rows[1].s0)/(k02+k04); // R_G1
  rgb.y = v_rows[1].s1; // G1(R)
  rgb.z = (k01*v_rows[0].s1+k03*v_rows[2].s1)/(k01+k03); // B_G1
  rgb_out[RGB_WRITE_ORDER[0]] = convert_uchar3_sat(apply_gamma(color_correct(clamp(rgb, 0.0, 1.0)), expo_time) * 255.0);

  const float k11 = get_k(v_rows[0].s1, v_rows[2].s1, v_rows[0].s3, v_rows[2].s3);
  const float k12 = get_k(v_rows[0].s2, v_rows[1].s1, v_rows[1].s3, v_rows[2].s2);
  const float k13 = get_k(v_rows[0].s1, v_rows[0].s3, v_rows[2].s1, v_rows[2].s3);
  const float k14 = get_k(v_rows[0].s2, v_rows[1].s3, v_rows[2].s2, v_rows[1].s1);
  rgb.x = v_rows[1].s2; // R
  rgb.y = (k11*(v_rows[0].s2+v_rows[2].s2)*0.5+k13*(v_rows[1].s3+v_rows[1].s1)*0.5)/(k11+k13); // G_R
  rgb.z = (k12*(v_rows[0].s3+v_rows[2].s1)*0.5+k14*(v_rows[0].s1+v_rows[2].s3)*0.5)/(k12+k14); // B_R
  rgb_out[RGB_WRITE_ORDER[1]] = convert_uchar3_sat(apply_gamma(color_correct(clamp(rgb, 0.0, 1.0)), expo_time) * 255.0);

  const float k21 = get_k(v_rows[1].s0, v_rows[3].s0, v_rows[1].s2, v_rows[3].s2);
  const float k22 = get_k(v_rows[1].s1, v_rows[2].s0, v_rows[2].s2, v_rows[3].s1);
  const float k23 = get_k(v_rows[1].s0, v_rows[1].s2, v_rows[3].s0, v_rows[3].s2);
  const float k24 = get_k(v_rows[1].s1, v_rows[2].s2, v_rows[3].s1, v_rows[2].s0);
  rgb.x = (k22*(v_rows[1].s2+v_rows[3].s0)*0.5+k24*(v_rows[1].s0+v_rows[3].s2)*0.5)/(k22+k24); // R_B
  rgb.y = (k21*(v_rows[1].s1+v_rows[3].s1)*0.5+k23*(v_rows[2].s2+v_rows[2].s0)*0.5)/(k21+k23); // G_B
  rgb.z = v_rows[2].s1; // B
  rgb_out[RGB_WRITE_ORDER[2]] = convert_uchar3_sat(apply_gamma(color_correct(clamp(rgb, 0.0, 1.0)), expo_time) * 255.0);

  const float k31 = get_k(v_rows[1].s1, v_rows[2].s2, v_rows[1].s3, v_rows[2].s2);
  const float k32 = get_k(v_rows[1].s3, v_rows[2].s2, v_rows[3].s3, v_rows[2].s2);
  const float k33 = get_k(v_rows[3].s1, v_rows[2].s2, v_rows[3].s3, v_rows[2].s2);
  const float k34 = get_k(v_rows[1].s1, v_rows[2].s2, v_rows[3].s1, v_rows[2].s2);
  rgb.x = (k31*v_rows[1].s2+k33*v_rows[3].s2)/(k31+k33); // R_G2
  rgb.y = v_rows[2].s2; // G2(B)
  rgb.z = (k32*v_rows[2].s3+k34*v_rows[2].s1)/(k32+k34); // B_G2
  rgb_out[RGB_WRITE_ORDER[3]] = convert_uchar3_sat(apply_gamma(color_correct(clamp(rgb, 0.0, 1.0)), expo_time) * 255.0);

  // write ys
  uchar2 yy = (uchar2)(
    RGB_TO_Y(rgb_out[0].s0, rgb_out[0].s1, rgb_out[0].s2),
    RGB_TO_Y(rgb_out[1].s0, rgb_out[1].s1, rgb_out[1].s2)
  );
  vstore2(yy, 0, out + mad24(gid_y * 2, YUV_STRIDE, gid_x * 2));
  yy = (uchar2)(
    RGB_TO_Y(rgb_out[2].s0, rgb_out[2].s1, rgb_out[2].s2),
    RGB_TO_Y(rgb_out[3].s0, rgb_out[3].s1, rgb_out[3].s2)
  );
  vstore2(yy, 0, out + mad24(gid_y * 2 + 1, YUV_STRIDE, gid_x * 2));

  // write uvs
  const short ar = AVERAGE(rgb_out[0].s0, rgb_out[1].s0, rgb_out[2].s0, rgb_out[3].s0);
  const short ag = AVERAGE(rgb_out[0].s1, rgb_out[1].s1, rgb_out[2].s1, rgb_out[3].s1);
  const short ab = AVERAGE(rgb_out[0].s2, rgb_out[1].s2, rgb_out[2].s2, rgb_out[3].s2);
  uchar2 uv = (uchar2)(
    RGB_TO_U(ar, ag, ab),
    RGB_TO_V(ar, ag, ab)
  );
  vstore2(uv, 0, out + UV_OFFSET + mad24(gid_y, YUV_STRIDE, gid_x * 2));
}
