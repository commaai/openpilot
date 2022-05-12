#ifdef HALF_AS_FLOAT
#define half float
#define half3 float3
#else
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

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
  half pv = (half)((major + minor)/4);

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
  const int x_global = get_global_id(0);
  const int y_global = get_global_id(1);

  const int x_local = get_local_id(0);
  const int y_local = get_local_id(1);

  const int localRowLen = 2 + get_local_size(0); // 2 padding
  const int localColLen = 2 + get_local_size(1);

  const int localOffset = (y_local + 1) * localRowLen + x_local + 1;

  int out_idx = 3 * x_global + 3 * y_global * RGB_WIDTH;

  // cache padding
  int localColOffset = -1;
  int globalColOffset;

  const int x_global_mod = (x_global == 0 || x_global == RGB_WIDTH - 1) ? -1: 1;
  const int y_global_mod = (y_global == 0 || y_global == RGB_HEIGHT - 1) ? -1: 1;

  half pv = val_from_10(in, x_global, y_global, black_level);
  cached[localOffset] = pv;

  // cache padding
  if (x_local < 1) {
    localColOffset = x_local;
    globalColOffset = -1;
    cached[(y_local + 1) * localRowLen + x_local] = val_from_10(in, x_global-x_global_mod, y_global, black_level);
  } else if (x_local >= get_local_size(0) - 1) {
    localColOffset = x_local + 2;
    globalColOffset = 1;
    cached[localOffset + 1] = val_from_10(in, x_global+x_global_mod, y_global, black_level);
  }

  if (y_local < 1) {
    cached[y_local * localRowLen + x_local + 1] = val_from_10(in, x_global, y_global-y_global_mod, black_level);
    if (localColOffset != -1) {
      cached[y_local * localRowLen + localColOffset] = val_from_10(in, x_global+(x_global_mod*globalColOffset), y_global-y_global_mod, black_level);
    }
  } else if (y_local >= get_local_size(1) - 1) {
    cached[(y_local + 2) * localRowLen + x_local + 1] = val_from_10(in, x_global, y_global+y_global_mod, black_level);
    if (localColOffset != -1) {
      cached[(y_local + 2) * localRowLen + localColOffset] = val_from_10(in, x_global+(x_global_mod*globalColOffset), y_global+y_global_mod, black_level);
    }
  }

  // sync
  barrier(CLK_LOCAL_MEM_FENCE);

  half d1 = cached[localOffset - localRowLen - 1];
  half d2 = cached[localOffset - localRowLen + 1];
  half d3 = cached[localOffset + localRowLen - 1];
  half d4 = cached[localOffset + localRowLen + 1];
  half n1 = cached[localOffset - localRowLen];
  half n2 = cached[localOffset + 1];
  half n3 = cached[localOffset + localRowLen];
  half n4 = cached[localOffset - 1];

  half3 rgb;

  // a simplified version of https://opensignalprocessingjournal.com/contents/volumes/V6/TOSIGPJ-6-1/TOSIGPJ-6-1.pdf
  if (x_global % 2 == 0) {
    if (y_global % 2 == 0) {
      rgb.y = pv; // G1(R)
      half k1 = get_k(d1, pv, d2, pv);
      half k2 = get_k(d2, pv, d4, pv);
      half k3 = get_k(d3, pv, d4, pv);
      half k4 = get_k(d1, pv, d3, pv);
      // R_G1
      rgb.x = (k2*n2+k4*n4)/(k2+k4);
      // B_G1
      rgb.z = (k1*n1+k3*n3)/(k1+k3);
    } else {
      rgb.z = pv; // B
      half k1 = get_k(d1, d3, d2, d4);
      half k2 = get_k(n1, n4, n2, n3);
      half k3 = get_k(d1, d2, d3, d4);
      half k4 = get_k(n1, n2, n3, n4);
      // G_B
      rgb.y = (k1*(n1+n3)*0.5+k3*(n2+n4)*0.5)/(k1+k3);
      // R_B
      rgb.x = (k2*(d2+d3)*0.5+k4*(d1+d4)*0.5)/(k2+k4);
    }
  } else {
    if (y_global % 2 == 0) {
      rgb.x = pv; // R
      half k1 = get_k(d1, d3, d2, d4);
      half k2 = get_k(n1, n4, n2, n3);
      half k3 = get_k(d1, d2, d3, d4);
      half k4 = get_k(n1, n2, n3, n4);
      // G_R
      rgb.y = (k1*(n1+n3)*0.5+k3*(n2+n4)*0.5)/(k1+k3);
      // B_R
      rgb.z = (k2*(d2+d3)*0.5+k4*(d1+d4)*0.5)/(k2+k4);
    } else {
      rgb.y = pv; // G2(B)
      half k1 = get_k(d1, pv, d2, pv);
      half k2 = get_k(d2, pv, d4, pv);
      half k3 = get_k(d3, pv, d4, pv);
      half k4 = get_k(d1, pv, d3, pv);
      // R_G2
      rgb.x = (k1*n1+k3*n3)/(k1+k3);
      // B_G2
      rgb.z = (k2*n2+k4*n4)/(k2+k4);
    }
  }

  uchar3 rgbc = convert_uchar3_sat(color_correct(clamp(rgb, (half)0.0, (half)1.0)) * 255.0);
  out[out_idx + 0] = rgbc.z;
  out[out_idx + 1] = rgbc.y;
  out[out_idx + 2] = rgbc.x;
}
