#if SENSOR_ID == 3

#define BGGR

#define BIT_DEPTH 12
#define PV_MAX 4096
#define BLACK_LVL 64
#define VIGNETTE_RSZ 2.2545f

float4 normalize_pv(int4 parsed, float vignette_factor) {
  float4 pv = (convert_float4(parsed) - BLACK_LVL) / (PV_MAX - BLACK_LVL);
  return clamp(pv*vignette_factor, 0.0, 1.0);
}

float3 color_correct(float3 rgb) {
  float3 corrected = rgb.x * (float3)(1.5664815, -0.29808738, -0.03973474);
  corrected += rgb.y * (float3)(-0.48672447, 1.41914433, -0.40295248);
  corrected += rgb.z * (float3)(-0.07975703, -0.12105695, 1.44268722);
  return corrected;
}

float3 apply_gamma(float3 rgb, int expo_time) {
  return powr(rgb, 0.7);
}

#endif
