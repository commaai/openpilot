// clang -Ofast -Wno-unused-result -march=native matmul.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float b52[786432];
float b49[196608];
float h_0_mlp_c_fc_weight[2359296];
float h_0_mlp_c_fc_bias[3072];

void matmul_forward(float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}


void r_256_3072_768(float* restrict data0, const float* restrict data1, const float* restrict data2, const float* restrict data3) {
  for (int ridx0 = 0; ridx0 < 256; ridx0++) {
    for (int ridx1 = 0; ridx1 < 3072; ridx1++) {
      float acc0 = 0.0f;
      float val0 = data3[ridx1];
      for (int ridx2 = 0; ridx2 < 768; ridx2++) {
        float val1 = data1[(ridx0*768)+ridx2];
        float val2 = data2[(ridx1*768)+ridx2];
        acc0 = ((val1*val2)+acc0);
      }
      data0[(ridx0*3072)+ridx1] = (acc0+val0);
    }
  }
}


int main() {
  for (int i = 0; i < 5; i++) {
    struct timespec t1, t2, t3;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    r_256_3072_768(b52, b49, h_0_mlp_c_fc_weight, h_0_mlp_c_fc_bias);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    matmul_forward(b52, b49, h_0_mlp_c_fc_weight, h_0_mlp_c_fc_bias, 4, 64, 768, 3072);
    clock_gettime(CLOCK_MONOTONIC, &t3);
    double time_gen = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;
    double time_real = (t3.tv_sec - t2.tv_sec) + (t3.tv_nsec - t2.tv_nsec) / 1e9;
    printf("%.2f ms gen vs %.2f ms reference\n", time_gen*1e3, time_real*1e3);
  }
}

