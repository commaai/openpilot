#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef HALF_AS_FLOAT  // pc doesnt support fp16
typedef float  HALF;
typedef float3 HALF3;
#else
typedef half  HALF;
typedef half3 HALF3;
#endif

// post wb CCM
const __constant HALF3 color_correction_0 = (HALF3)(1.82717181, -0.31231438, 0.07307673);
const __constant HALF3 color_correction_1 = (HALF3)(-0.5743977, 1.36858544, -0.53183455);
const __constant HALF3 color_correction_2 = (HALF3)(-0.25277411, -0.05627105, 1.45875782);

// tone mapping params
const HALF cpk = 0.75;
const HALF cpb = 0.125;
const HALF cpxk = 0.0025;
const HALF cpxb = 0.01;

HALF mf(HALF x, HALF cp) {
  HALF rk = 9 - 100*cp;
  if (x > cp) {
    return (rk * (x-cp) * (1-(cpk*cp+cpb)) * (1+1/(rk*(1-cp))) / (1+rk*(x-cp))) + cpk*cp + cpb;
  } else if (x < cp) {
    return (rk * (x-cp) * (cpk*cp+cpb) * (1+1/(rk*cp)) / (1-rk*(x-cp))) + cpk*cp + cpb;
  } else {
    return x;
  }
}

HALF3 color_correct(HALF3 rgb) {
  HALF3 ret = (0,0,0);
  HALF cpx = 0.01;
  ret += (HALF)rgb.x * color_correction_0;
  ret += (HALF)rgb.y * color_correction_1;
  ret += (HALF)rgb.z * color_correction_2;
  ret.x = mf(ret.x, cpx);
  ret.y = mf(ret.y, cpx);
  ret.z = mf(ret.z, cpx);
  ret = clamp(0.0, 255.0, ret*255.0);
  return ret;
}

inline HALF val_from_10(const uchar * source, int gx, int gy, HALF black_level) {
  // parse 12bit
  int start = gy * FRAME_STRIDE + (3 * (gx / 2));
  int offset = gx % 2;
  uint major = (uint)source[start + offset] << 4;
  uint minor = (source[start + 2] >> (4 * offset)) & 0xf;
  HALF pv = (HALF)((major + minor)/4);

  // normalize
  pv = max(0.0, pv - black_level);
  pv /= (1024.0f - black_level);

  // correct vignetting
  if (CAM_NUM == 1) { // fcamera
    gx = (gx - RGB_WIDTH/2);
    gy = (gy - RGB_HEIGHT/2);
    float r = gx*gx + gy*gy;
    HALF s;
    if (r < 62500) {
      s = (HALF)(1.0f + 0.0000008f*r);
    } else if (r < 490000) {
      s = (HALF)(0.9625f + 0.0000014f*r);
    } else if (r < 1102500) {
      s = (HALF)(1.26434f + 0.0000000000016f*r*r);
    } else {
      s = (HALF)(0.53503625f + 0.0000000000022f*r*r);
    }
    pv = s * pv;
  }

  pv = clamp(0.0, 1.0, pv);
  return pv;
}

HALF fabs_diff(HALF x, HALF y) {
  return fabs(x-y);
}

HALF phi(HALF x) {
  // detection funtion
  return 2 - x;
  // if (x > 1) {
  //   return 1 / x;
  // } else {
  //   return 2 - x;
  // }
}

__kernel void debayer10(const __global uchar * in,
                        __global uchar * out,
                        __local HALF * cached,
                        float black_level
                       )
{
  const int x_global = get_global_id(0);
  const int y_global = get_global_id(1);

  const int localRowLen = 2 + get_local_size(0); // 2 padding
  const int x_local = get_local_id(0); // 0-15
  const int y_local = get_local_id(1); // 0-15
  const int localOffset = (y_local + 1) * localRowLen + x_local + 1; // max 18x18-1

  int out_idx = 3 * x_global + 3 * y_global * RGB_WIDTH;

  HALF pv = val_from_10(in, x_global, y_global, black_level);
  cached[localOffset] = pv;

  // cache padding
  int localColOffset = -1;
  int globalColOffset = -1;

  // cache padding
  if (x_global >= 1 && x_local < 1) {
    localColOffset = x_local;
    globalColOffset = -1;
    cached[(y_local + 1) * localRowLen + x_local] = val_from_10(in, x_global-1, y_global, black_level);
  } else if (x_global < RGB_WIDTH - 1 && x_local >= get_local_size(0) - 1) {
    localColOffset = x_local + 2;
    globalColOffset = 1;
    cached[localOffset + 1] = val_from_10(in, x_global+1, y_global, black_level);
  }

  if (y_global >= 1 && y_local < 1) {
    cached[y_local * localRowLen + x_local + 1] = val_from_10(in, x_global, y_global-1, black_level);
    if (localColOffset != -1) {
      cached[y_local * localRowLen + localColOffset] = val_from_10(in, x_global+globalColOffset, y_global-1, black_level);
    }
  } else if (y_global < RGB_HEIGHT - 1 && y_local >= get_local_size(1) - 1) {
    cached[(y_local + 2) * localRowLen + x_local + 1] = val_from_10(in, x_global, y_global+1, black_level);
    if (localColOffset != -1) {
      cached[(y_local + 2) * localRowLen + localColOffset] = val_from_10(in, x_global+globalColOffset, y_global+1, black_level);
    }
  }

  // don't care
  if (x_global < 1 || x_global >= RGB_WIDTH - 1 || y_global < 1 || y_global >= RGB_HEIGHT - 1) {
    return;
  }

  // sync
  barrier(CLK_LOCAL_MEM_FENCE);

  HALF d1 = cached[localOffset - localRowLen - 1];
  HALF d2 = cached[localOffset - localRowLen + 1];
  HALF d3 = cached[localOffset + localRowLen - 1];
  HALF d4 = cached[localOffset + localRowLen + 1];
  HALF n1 = cached[localOffset - localRowLen];
  HALF n2 = cached[localOffset + 1];
  HALF n3 = cached[localOffset + localRowLen];
  HALF n4 = cached[localOffset - 1];

  HALF3 rgb;

  // a simplified version of https://opensignalprocessingjournal.com/contents/volumes/V6/TOSIGPJ-6-1/TOSIGPJ-6-1.pdf
  if (x_global % 2 == 0) {
    if (y_global % 2 == 0) {
      rgb.y = pv; // G1(R)
      HALF k1 = phi(fabs_diff(d1, pv) + fabs_diff(d2, pv));
      HALF k2 = phi(fabs_diff(d2, pv) + fabs_diff(d4, pv));
      HALF k3 = phi(fabs_diff(d3, pv) + fabs_diff(d4, pv));
      HALF k4 = phi(fabs_diff(d1, pv) + fabs_diff(d3, pv));
      // R_G1
      rgb.x = (k2*n2+k4*n4)/(k2+k4);
      // B_G1
      rgb.z = (k1*n1+k3*n3)/(k1+k3);
    } else {
      rgb.z = pv; // B
      HALF k1 = phi(fabs_diff(d1, d3) + fabs_diff(d2, d4));
      HALF k2 = phi(fabs_diff(n1, n4) + fabs_diff(n2, n3));
      HALF k3 = phi(fabs_diff(d1, d2) + fabs_diff(d3, d4));
      HALF k4 = phi(fabs_diff(n1, n2) + fabs_diff(n3, n4));
      // G_B
      rgb.y = (k1*(n1+n3)*0.5+k3*(n2+n4)*0.5)/(k1+k3);
      // R_B
      rgb.x = (k2*(d2+d3)*0.5+k4*(d1+d4)*0.5)/(k2+k4);
    }
  } else {
    if (y_global % 2 == 0) {
      rgb.x = pv; // R
      HALF k1 = phi(fabs_diff(d1, d3) + fabs_diff(d2, d4));
      HALF k2 = phi(fabs_diff(n1, n4) + fabs_diff(n2, n3));
      HALF k3 = phi(fabs_diff(d1, d2) + fabs_diff(d3, d4));
      HALF k4 = phi(fabs_diff(n1, n2) + fabs_diff(n3, n4));
      // G_R
      rgb.y = (k1*(n1+n3)*0.5+k3*(n2+n4)*0.5)/(k1+k3);
      // B_R
      rgb.z = (k2*(d2+d3)*0.5+k4*(d1+d4)*0.5)/(k2+k4);
    } else {
      rgb.y = pv; // G2(B)
      HALF k1 = phi(fabs_diff(d1, pv) + fabs_diff(d2, pv));
      HALF k2 = phi(fabs_diff(d2, pv) + fabs_diff(d4, pv));
      HALF k3 = phi(fabs_diff(d3, pv) + fabs_diff(d4, pv));
      HALF k4 = phi(fabs_diff(d1, pv) + fabs_diff(d3, pv));
      // R_G2
      rgb.x = (k1*n1+k3*n3)/(k1+k3);
      // B_G2
      rgb.z = (k2*n2+k4*n4)/(k2+k4);
    }
  }

  rgb = clamp(0.0, 1.0, rgb);
  rgb = color_correct(rgb);

  out[out_idx + 0] = (uchar)(rgb.z);
  out[out_idx + 1] = (uchar)(rgb.y);
  out[out_idx + 2] = (uchar)(rgb.x);
}
