#include <cuda_bf16.h>
#include <cuda_fp8.h>
__device__ __forceinline__ void mma(float* c, unsigned* a, unsigned* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#if SPLITS == 1
using output_t = __nv_bfloat16;
#else
using output_t = float;
#endif
extern "C" __global__ __launch_bounds__(128) void alpamayo_gemm(
  output_t* out, const unsigned char* a, const unsigned char* b, const float* ascale, const float* bscale, const __nv_bfloat16* bias) {
  __shared__ __align__(16) unsigned char sa[64*80], sb[64*80];
  int tid=threadIdx.x, warp=tid/32, lane=tid%32;
  int split=blockIdx.x%SPLITS, block=blockIdx.x/SPLITS;
  int bm=block/((N+63)/64)*64, bn=block%((N+63)/64)*64;
  int span=(K+SPLITS*64-1)/(SPLITS*64)*64;
  int wm=warp/2*32, wn=warp%2*32, group=lane/4, t=lane%4;
  float c[2][4][4] = {};
  for(int k=split*span;k<min(K,(split+1)*span);k+=64) {
    #pragma unroll
    for(int z=0;z<2;z++) {
      int i=tid+z*128, row=i/4, col=(i%4)*16;
      unsigned aa=__cvta_generic_to_shared(sa+row*80+col), bb=__cvta_generic_to_shared(sb+row*80+col);
      int acount=bm+row<M && k+col<K ? 16 : 0, bcount=bn+row<N && k+col<K ? 16 : 0;
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(aa), "l"(a+(bm+row)*K+k+col), "r"(acount));
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(bb), "l"(b+(bn+row)*K+k+col), "r"(bcount));
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;");
    __syncthreads();
    #pragma unroll
    for(int kk=0;kk<64;kk+=32) {
      unsigned ar[2][4], br[4][2];
      #pragma unroll
      for(int i=0;i<2;i++) {
        ar[i][0]=*(unsigned*)(sa+(wm+i*16+group)*80+kk+t*4);
        ar[i][1]=*(unsigned*)(sa+(wm+i*16+group+8)*80+kk+t*4);
        ar[i][2]=*(unsigned*)(sa+(wm+i*16+group)*80+kk+t*4+16);
        ar[i][3]=*(unsigned*)(sa+(wm+i*16+group+8)*80+kk+t*4+16);
      }
      #pragma unroll
      for(int j=0;j<4;j++) {
        br[j][0]=*(unsigned*)(sb+(wn+j*8+group)*80+kk+t*4);
        br[j][1]=*(unsigned*)(sb+(wn+j*8+group)*80+kk+t*4+16);
      }
      #pragma unroll
      for(int i=0;i<2;i++) {
        #pragma unroll
        for(int j=0;j<4;j++) mma(c[i][j], ar[i], br[j]);
      }
    }
    __syncthreads();
  }
  float scale=ascale[0]*bscale[0];
  #pragma unroll
  for(int i=0;i<2;i++) {
    #pragma unroll
    for(int j=0;j<4;j++) {
      #pragma unroll
      for(int z=0;z<4;z++) {
        int row=bm+wm+i*16+group+(z/2)*8, col=bn+wn+j*8+t*2+z%2;
        int index=TRANSPOSE ? ((row/SEQUENCE*HEADS+col/HEAD_DIM)*SEQUENCE+row%SEQUENCE)*HEAD_DIM+col%HEAD_DIM : row*N+col;
        if(row<M && col<N) out[split*M*N+index]=output_t(c[i][j][z]*scale+(BIAS && split==0 ? float(bias[col]) : 0));
      }
    }
  }
}
