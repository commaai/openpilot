#include <assert.h>
#include <math.h>
#include "commonmodel.h"
#include "common/clutil.h"
#include "common/mat.h"
#include "common/timing.h"

void softmax(const float* input, float* output, size_t len) {
  float max_val = -FLT_MAX;
  for(int i = 0; i < len; i++) {
    const float v = input[i];
    if( v > max_val ) {
      max_val = v;
    }
  }

  float denominator = 0;
  for(int i = 0; i < len; i++) {
    float const v = input[i];
    float const v_exp = expf(v - max_val);
    denominator += v_exp;
    output[i] = v_exp;
  }

  const float inv_denominator = 1. / denominator;
  for(int i = 0; i < len; i++) {
    output[i] *= inv_denominator;
  }

}

float sigmoid(float input) {
  return 1 / (1 + expf(-input));
}

float softplus(float input) {
  return log1p(expf(input));
}
