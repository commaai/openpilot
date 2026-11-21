typedef unsigned long size_t;
typedef __bf16 bf16;
extern "C" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);

// The local BF16 boundary must precede the existing data-parallel aggregation.
extern "C" __attribute__((global, amdgpu_flat_work_group_size(64,64)))
void gptoss_wo_bias_final(bf16* out, const float* in) {
  int d = __ockl_get_group_id(0)*64 + __ockl_get_local_id(0);
  float acc=0;
  #pragma unroll
  for (int p=0;p<16;p++) acc=acc+in[p*2880+d];
  out[d]=(bf16)acc;
}
