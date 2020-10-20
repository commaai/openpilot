#pragma OPENCL EXTENSION cl_khr_fp16 : enable

const __constant half3 color_correction[3] = {
  // post wb CCM
  (half3)(1.44602146, -0.24727126, -0.0403062),
  (half3)(-0.37658179, 1.26329038, -0.45978396),
  (half3)(-0.06943967, -0.01601912, 1.50009016),
};

const half black_level = 42.0;

uchar3 color_correct(half3 rgb) {
  half3 ret16 = (0,0,0);/*

  ret16 += rgb.x * color_correction[0];
  ret16 += rgb.y * color_correction[1];
  ret16 += rgb.z * color_correction[2];*/
  ret16 = max(0.0, min(255.0, ret16));
  return (uchar3)(ret16.x, ret16.y, ret16.z);
}

// --- conversion from 5xui8 to 4xf16 ---
half4 f16_from_10_batch(const uchar * source, int start) {
  // source: source
  // start: starting address
  uchar4 majors_raw;
  uchar minors_raw;
  majors_raw = vload4(start, source);
  minors_raw = source[start + 4];
  half4 pv = (half4)(majors_raw.x << 2 + (minors_raw) & 3,
                     majors_raw.y << 2 + (minors_raw >> 2) & 3,
                     majors_raw.z << 2 + (minors_raw >> 4) & 3,
                     majors_raw.w << 2 + (minors_raw >> 6) & 3);
  pv = max(0.0h, pv - black_level);
  pv /= (1024.0h - black_level);
  pv = 255.0h*20*pv / (1.0h + 20*pv); // reinhard
  return pv;
}
// --- ---

// --- conversion function for some standalone edge pixels ---
half f16_from_10_one(const uchar * source, int start, int offset) {
  // source: source
  // start: starting address of 0
  // offset: 0 - 3
  ushort major = (ushort)source[start + offset] << 2;
  ushort minor = (source[start + 4] >> (2 * offset)) & 3;
  half pv = (half)(major + minor);
  pv = max(0.0h, pv - black_level);
  pv /= (1024.0h - black_level);
  pv = 255.0h*20*pv / (1.0h + 20*pv); // reinhard
  return pv;
}
// --- ---

__kernel void debayer10(const __global uchar * in,
                        __global uchar * out,
                        __local half * cached
                       )
{
  const int x_global = get_global_id(0);

  bool to_cache = true;
  ushort offset_10 = x_global % 4;
  // caching is done in batches of 4
  if (offset_10 > 0) {
    to_cache = false;
  }

  const int y_global = get_global_id(1);
  const int localRowLen = get_local_size(0); // cache padding=trash speed
  const int x_local = get_local_id(0);
  const int y_local = get_local_id(1);

  const int localOffset = (y_local) * localRowLen + x_local;
  const uint globalStart_10 = y_global * FRAME_STRIDE + (5 * (x_global / 4));

  // --- cache local pixels ---
  if (to_cache) {
    half4 this = f16_from_10_batch(in, y_global * FRAME_STRIDE + (5 * (x_global / 4)));
    vstore4(this, localOffset, cached);
  }

  // global edges, don't bother
  if (x_global < 1 || x_global > RGB_WIDTH - 2 || y_global < 1 || y_global > RGB_WIDTH - 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    return;
  }

  // sync
  barrier(CLK_LOCAL_MEM_FENCE);

  // --- perform debayer ---

  half3 rgb;
  bool is_top = y_local==0;
  bool is_bot = y_local==get_local_size(1) - 1;
  bool is_left = x_local==0;
  bool is_right = x_local==localRowLen - 1;

  if (x_global % 2 == 0) {
    if (y_global % 2 == 0) { // G1
      rgb = (((is_left?f16_from_10_one(in,y_global*FRAME_STRIDE+(5*((x_global-1)/4)),(offset_10+3)%4):cached[localOffset - 1]) +
              (is_right?f16_from_10_one(in,y_global*FRAME_STRIDE+(5*((x_global+1)/4)),(offset_10+1)%4):cached[localOffset + 1])) * color_correction[0]/ 2.0h,
             (cached[localOffset] +
              ((is_bot*is_right)?f16_from_10_one(in,(y_global+1)*FRAME_STRIDE+(5*((x_global+1)/4)),(offset_10+1)%4):cached[localOffset + localRowLen + 1]))* color_correction[0] / 2.0h,
             ((is_top?f16_from_10_one(in,(y_global-1)*FRAME_STRIDE+(5*((x_global)/4)),(offset_10)%4):cached[localOffset - localRowLen]) +
              (is_bot?f16_from_10_one(in,(y_global+1)*FRAME_STRIDE+(5*((x_global)/4)),(offset_10)%4):cached[localOffset + localRowLen]))* color_correction[0] / 2.0h);
    } else { // B
      rgb = ((((is_top*is_left)?f16_from_10_one(in,(y_global-1)*FRAME_STRIDE+(5*((x_global-1)/4)),(offset_10+3)%4):cached[localOffset - localRowLen - 1]) +
              ((is_top*is_right)?f16_from_10_one(in,(y_global-1)*FRAME_STRIDE+(5*((x_global+1)/4)),(offset_10+1)%4):cached[localOffset - localRowLen + 1]) +
              ((is_bot*is_left)?f16_from_10_one(in,(y_global+1)*FRAME_STRIDE+(5*((x_global-1)/4)),(offset_10+3)%4):cached[localOffset + localRowLen - 1]) +
              ((is_bot*is_right)?f16_from_10_one(in,(y_global+1)*FRAME_STRIDE+(5*((x_global+1)/4)),(offset_10+1)%4):cached[localOffset + localRowLen + 1])) * color_correction[0]/ 4.0h,
             ((is_top?f16_from_10_one(in,(y_global-1)*FRAME_STRIDE+(5*((x_global)/4)),(offset_10)%4):cached[localOffset - localRowLen]) +
              (is_bot?f16_from_10_one(in,(y_global+1)*FRAME_STRIDE+(5*((x_global)/4)),(offset_10)%4):cached[localOffset + localRowLen]) +
              (is_left?f16_from_10_one(in,y_global*FRAME_STRIDE+(5*((x_global-1)/4)),(offset_10+3)%4):cached[localOffset - 1]) +
              (is_right?f16_from_10_one(in,y_global*FRAME_STRIDE+(5*((x_global+1)/4)),(offset_10+1)%4):cached[localOffset + 1])) * color_correction[0]/ 4.0h,
             cached[localOffset]* color_correction[0]);
    }
  } else {
    if (y_global % 2 == 0) { // R
      rgb = (cached[localOffset],
            ((is_top?f16_from_10_one(in,(y_global-1)*FRAME_STRIDE+(5*((x_global)/4)),(offset_10)%4):cached[localOffset - localRowLen]) +
              (is_bot?f16_from_10_one(in,(y_global+1)*FRAME_STRIDE+(5*((x_global)/4)),(offset_10)%4):cached[localOffset + localRowLen]) +
              (is_left?f16_from_10_one(in,y_global*FRAME_STRIDE+(5*((x_global-1)/4)),(offset_10+3)%4):cached[localOffset - 1]) +
              (is_right?f16_from_10_one(in,y_global*FRAME_STRIDE+(5*((x_global+1)/4)),(offset_10+1)%4):cached[localOffset + 1])) * color_correction[0]/ 4.0h,
            (((is_top*is_left)?f16_from_10_one(in,(y_global-1)*FRAME_STRIDE+(5*((x_global-1)/4)),(offset_10+3)%4):cached[localOffset - localRowLen - 1]) +
              ((is_top*is_right)?f16_from_10_one(in,(y_global-1)*FRAME_STRIDE+(5*((x_global+1)/4)),(offset_10+1)%4):cached[localOffset - localRowLen + 1]) +
              ((is_bot*is_left)?f16_from_10_one(in,(y_global+1)*FRAME_STRIDE+(5*((x_global-1)/4)),(offset_10+3)%4):cached[localOffset + localRowLen - 1]) +
              ((is_bot*is_right)?f16_from_10_one(in,(y_global+1)*FRAME_STRIDE+(5*((x_global+1)/4)),(offset_10+1)%4):cached[localOffset + localRowLen + 1])) * color_correction[0]/ 4.0h);
    } else { // G2
      rgb = (((is_top?f16_from_10_one(in,(y_global-1)*FRAME_STRIDE+(5*((x_global)/4)),(offset_10)%4):cached[localOffset - localRowLen]) +
              (is_bot?f16_from_10_one(in,(y_global+1)*FRAME_STRIDE+(5*((x_global)/4)),(offset_10)%4):cached[localOffset + localRowLen])) * color_correction[0]/ 2.0h,
             (cached[localOffset]* color_correction[0] +
             ((is_top*is_left)?f16_from_10_one(in,(y_global-1)*FRAME_STRIDE+(5*((x_global-1)/4)),(offset_10+3)%4):cached[localOffset - localRowLen - 1])) * color_correction[0]/ 2.0h,
             ((is_left?f16_from_10_one(in,y_global*FRAME_STRIDE+(5*((x_global-1)/4)),(offset_10+3)%4):cached[localOffset - 1]) +
              (is_right?f16_from_10_one(in,y_global*FRAME_STRIDE+(5*((x_global+1)/4)),(offset_10+1)%4):cached[localOffset + 1])) * color_correction[0]/ 2.0h);
    }
  }

  // BGR output
  vstore3(color_correct(rgb), 3 * x_global + 3 * y_global * RGB_WIDTH, out);
}
