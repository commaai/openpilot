const __constant float3 color_correction[3] = {
  // post wb CCM
  (float3)(1.44602146, -0.24727126, -0.0403062),
  (float3)(-0.37658179, 1.26329038, -0.45978396),
  (float3)(-0.06943967, -0.01601912, 1.50009016),
};

float3 color_correct(float r, float g, float b) {
  float3 ret = (0,0,0);

  ret += r * color_correction[0];
  ret += g * color_correction[1];
  ret += b * color_correction[2];
  ret = max(0.0, min(1.0, ret));
  return ret;
}

uint int_from_10(const uchar * source, uint start, uint offset) {
  // source: source
  // start: starting address of 0
  // offset: 0 - 3
  uint major = (uint)source[start + offset] << 2;
  uint minor = (source[start + 4] >> (2 * offset)) & 3;
  return major + minor;
}

float to_normal(uint x, int gx, int gy) {
  float pv = (float)(x);
  const float black_level = 42.0;
  pv = max(0.0, pv - black_level);
  pv /= (1024.0f - black_level);
  if (CAM_NUM == 1) { // fcamera
    gx = (gx - RGB_WIDTH/2);
    gy = (gy - RGB_HEIGHT/2);
    float r = pow(gx*gx + gy*gy, 0.825);
    float s = 1 / (1-0.00000733*r);
    pv = s * pv;
  }
  pv = 20*pv / (1.0f + 20*pv); // reinhard
  return pv;
}

__kernel void debayer10(const __global uchar * in,
                        __global uchar * out,
                        __local float * cached
                       )
{
  const int x_global = get_global_id(0);
  const int y_global = get_global_id(1);

  // const int globalOffset = ;

  const int localRowLen = 2 + get_local_size(0); // 2 padding
  const int x_local = get_local_id(0);
  const int y_local = get_local_id(1);

  const int localOffset = (y_local + 1) * localRowLen + x_local + 1;

  // cache local pixels first
  // saves memory access and avoids repeated normalization
  uint globalStart_10 = y_global * FRAME_STRIDE + (5 * (x_global / 4));
  uint offset_10 = x_global % 4;
  uint raw_val = int_from_10(in, globalStart_10, offset_10);
  cached[localOffset] = to_normal(raw_val, x_global, y_global);

  // edges
  if (x_global < 1 || x_global > RGB_WIDTH - 2 || y_global < 1 || y_global > RGB_HEIGHT - 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    return;
  } else {
    int localColOffset = -1;
    int globalColOffset = -1;

    // cache padding
    if (x_local < 1) {
      localColOffset = x_local;
      globalColOffset = -1;
      cached[(y_local + 1) * localRowLen + x_local] = to_normal(int_from_10(in, y_global * FRAME_STRIDE + (5 * ((x_global-1) / 4)), (offset_10 + 3) % 4), x_global, y_global);
    } else if (x_local >= get_local_size(0) - 1) {
      localColOffset = x_local + 2;
      globalColOffset = 1;
      cached[localOffset + 1] = to_normal(int_from_10(in, y_global * FRAME_STRIDE + (5 * ((x_global+1) / 4)), (offset_10 + 1) % 4), x_global, y_global);
    }

    if (y_local < 1) {
      cached[y_local * localRowLen + x_local + 1] = to_normal(int_from_10(in, globalStart_10 - FRAME_STRIDE, offset_10), x_global, y_global);
      if (localColOffset != -1) {
        cached[y_local * localRowLen + localColOffset] = to_normal(int_from_10(in, (y_global-1) * FRAME_STRIDE + (5 * ((x_global+globalColOffset) / 4)), (offset_10+4+globalColOffset) % 4), x_global, y_global);
      }
    } else if (y_local >= get_local_size(1) - 1) {
      cached[(y_local + 2) * localRowLen + x_local + 1] = to_normal(int_from_10(in, globalStart_10 + FRAME_STRIDE, offset_10), x_global, y_global);
      if (localColOffset != -1) {
        cached[(y_local + 2) * localRowLen + localColOffset] = to_normal(int_from_10(in, (y_global+1) * FRAME_STRIDE + (5 * ((x_global+globalColOffset) / 4)), (offset_10+4+globalColOffset) % 4), x_global, y_global);
      }
    }

    // sync
    barrier(CLK_LOCAL_MEM_FENCE);

    // perform debayer
    float r;
    float g;
    float b;

    if (x_global % 2 == 0) {
      if (y_global % 2 == 0) { // G1
        r = (cached[localOffset - 1] + cached[localOffset + 1]) / 2.0f;
        g = (cached[localOffset] + cached[localOffset + localRowLen + 1]) / 2.0f;
        b = (cached[localOffset - localRowLen] + cached[localOffset + localRowLen]) / 2.0f;
      } else { // B
        r = (cached[localOffset - localRowLen - 1] + cached[localOffset - localRowLen + 1] + cached[localOffset + localRowLen - 1] + cached[localOffset + localRowLen + 1]) / 4.0f;
        g = (cached[localOffset - localRowLen] + cached[localOffset + localRowLen] + cached[localOffset - 1] + cached[localOffset + 1]) / 4.0f;
        b = cached[localOffset];
      }
    } else {
      if (y_global % 2 == 0) { // R
        r = cached[localOffset];
        g = (cached[localOffset - localRowLen] + cached[localOffset + localRowLen] + cached[localOffset - 1] + cached[localOffset + 1]) / 4.0f;
        b = (cached[localOffset - localRowLen - 1] + cached[localOffset - localRowLen + 1] + cached[localOffset + localRowLen - 1] + cached[localOffset + localRowLen + 1]) / 4.0f;
      } else { // G2
        r = (cached[localOffset - localRowLen] + cached[localOffset + localRowLen]) / 2.0f;
        g = (cached[localOffset] + cached[localOffset - localRowLen - 1]) / 2.0f;
        b = (cached[localOffset - 1] + cached[localOffset + 1]) / 2.0f;
      }
    }

    float3 rgb = color_correct(r, g, b);
    // rgb = srgb_gamma(rgb); 

    // BGR output
    out[3 * x_global + 3 * y_global * RGB_WIDTH + 0] = (uchar)(255.0f * rgb.z);
    out[3 * x_global + 3 * y_global * RGB_WIDTH + 1] = (uchar)(255.0f * rgb.y);
    out[3 * x_global + 3 * y_global * RGB_WIDTH + 2] = (uchar)(255.0f * rgb.x);
  }
}
