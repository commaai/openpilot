typedef unsigned long size_t;
typedef __bf16 bf16;
extern "C" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);

// Keep the generated GPTOSS reduction's 16 ordered 1024-token partials.
// Prefetching changes only load scheduling; all FP32 additions stay in order.
extern "C" __attribute__((global, amdgpu_flat_work_group_size(64,64)))
void gptoss_wo_bias_partial(float* out, const bf16* in) {
  int d = __ockl_get_group_id(0)*64 + __ockl_get_local_id(0);
  int p = __ockl_get_group_id(1);
  float acc = 0;
  for (int r=0; r<1024; r+=64) {
    float x[64];
    #pragma unroll
    for (int j=0; j<64; j++) x[j] = (float)in[(p*1024+r+j)*2880+d];
    #pragma unroll
    for (int j=0; j<64; j++) acc = acc+x[j];
  }
  out[p*2880+d] = acc;
}
