#if SENSOR_ID == 1

#define VIGNETTE_PROFILE_8DT0MM

#define BIT_DEPTH 12
#define PV_MAX 4096
#define BLACK_LVL 168

float4 normalize_pv(int4 parsed, float vignette_factor) {
  float4 pv = (convert_float4(parsed) - BLACK_LVL) / (PV_MAX - BLACK_LVL);
  return clamp(pv*vignette_factor, 0.0, 1.0);
}

float3 color_correct(float3 rgb) {
  float3 corrected = rgb.x * (float3)(1.82717181, -0.31231438, 0.07307673);
  corrected += rgb.y * (float3)(-0.5743977, 1.36858544, -0.53183455);
  corrected += rgb.z * (float3)(-0.25277411, -0.05627105, 1.45875782);
  return corrected;
}

float3 apply_gamma(float3 rgb, int expo_time) {
  // tone mapping params
  const float gamma_k = 0.75;
  const float gamma_b = 0.125;
  const float mp = 0.01; // ideally midpoint should be adaptive
  const float rk = 9 - 100*mp;

  // poly approximation for s curve
  return (rgb > mp) ?
    ((rk * (rgb-mp) * (1-(gamma_k*mp+gamma_b)) * (1+1/(rk*(1-mp))) / (1+rk*(rgb-mp))) + gamma_k*mp + gamma_b) :
    ((rk * (rgb-mp) * (gamma_k*mp+gamma_b) * (1+1/(rk*mp)) / (1-rk*(rgb-mp))) + gamma_k*mp + gamma_b);
}

#endif
