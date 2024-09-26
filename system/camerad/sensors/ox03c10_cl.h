#if SENSOR_ID == 2

#define BIT_DEPTH 12
#define BLACK_LVL 64
#define VIGNETTE_RSZ 1.0f

float4 ox_lut_func(int4 parsed) {
  return (exp(convert_float4(parsed) / 271.0) - 1.0) * 2.73845678e-07;
}

float4 normalize_pv(int4 parsed, float vignette_factor) {
  // PWL
  float4 pv = ox_lut_func(parsed);
  return clamp(pv*vignette_factor*256.0, 0.0, 1.0);
}

float3 color_correct(float3 rgb) {
  float3 corrected = rgb.x * (float3)(1.5664815, -0.29808738, -0.03973474);
  corrected += rgb.y * (float3)(-0.48672447, 1.41914433, -0.40295248);
  corrected += rgb.z * (float3)(-0.07975703, -0.12105695, 1.44268722);
  return corrected;
}

float3 apply_gamma(float3 rgb, int expo_time) {
  return -0.507089*exp(-12.54124638*rgb) + 0.9655*powr(rgb, 0.5) - 0.472597*rgb + 0.507089;
}

#endif
