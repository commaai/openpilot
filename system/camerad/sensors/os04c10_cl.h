#if IS_OS

#define BGGR
#define HDR_COMBINE

float3 color_correct(float3 rgb) {
  float3 corrected = rgb.x * (float3)(1.55361989, -0.268894615, -0.000593219);
  corrected += rgb.y * (float3)(-0.421217301, 1.51883144, -0.69760146);
  corrected += rgb.z * (float3)(-0.132402589, -0.249936825, 1.69819468);
  return corrected;
}

float3 apply_gamma(float3 rgb, int expo_time) {
  float s = log2((float)expo_time);
  if (s < 6) {s = fmin(12.0 - s, 9.0);}
  // log function adaptive to number of bits
  return clamp(log(1 + rgb*65472.0) * (0.48*s*s - 12.92*s + 115.0) - (1.08*s*s - 29.2*s + 260.0), 0.0, 255.0) / 255.0;
}

#endif