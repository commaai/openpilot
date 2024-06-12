// clang++ -O2 repro.cc && ./a.out

#include <sched.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

static inline double millis_since_boot() {
  struct timespec t;
  clock_gettime(CLOCK_BOOTTIME, &t);
  return t.tv_sec * 1000.0 + t.tv_nsec * 1e-6;
}

#define MODEL_WIDTH 320
#define MODEL_HEIGHT 640

// null function still breaks it
#define input_lambda(x) x

// this is copied from models/dmonitoring.cc, and is the code that triggers the issue
void inner(uint8_t *resized_buf, float *net_input_buf) {
  int resized_width = MODEL_WIDTH;
  int resized_height = MODEL_HEIGHT;

  // one shot conversion, O(n) anyway
  // yuvframe2tensor, normalize
  for (int r = 0; r < MODEL_HEIGHT/2; r++) {
    for (int c = 0; c < MODEL_WIDTH/2; c++) {
      // Y_ul
      net_input_buf[(c*MODEL_HEIGHT/2) + r] = input_lambda(resized_buf[(2*r*resized_width) + (2*c)]);
      // Y_ur
      net_input_buf[(c*MODEL_HEIGHT/2) + r + (2*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = input_lambda(resized_buf[(2*r*resized_width) + (2*c+1)]);
      // Y_dl
      net_input_buf[(c*MODEL_HEIGHT/2) + r + ((MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = input_lambda(resized_buf[(2*r*resized_width+1) + (2*c)]);
      // Y_dr
      net_input_buf[(c*MODEL_HEIGHT/2) + r + (3*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = input_lambda(resized_buf[(2*r*resized_width+1) + (2*c+1)]);
      // U
      net_input_buf[(c*MODEL_HEIGHT/2) + r + (4*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = input_lambda(resized_buf[(resized_width*resized_height) + (r*resized_width/2) + c]);
      // V
      net_input_buf[(c*MODEL_HEIGHT/2) + r + (5*(MODEL_WIDTH/2)*(MODEL_HEIGHT/2))] = input_lambda(resized_buf[(resized_width*resized_height) + ((resized_width/2)*(resized_height/2)) + (r*resized_width/2) + c]);
    }
  }
}

float trial() {
  int resized_width = MODEL_WIDTH;
  int resized_height = MODEL_HEIGHT;

  int yuv_buf_len = (MODEL_WIDTH/2) * (MODEL_HEIGHT/2) * 6; // Y|u|v -> y|y|y|y|u|v

  // allocate the buffers
  uint8_t *resized_buf = (uint8_t*)malloc(resized_width*resized_height*3/2);
  float *net_input_buf = (float*)malloc(yuv_buf_len*sizeof(float));
  printf("allocate -- %p 0x%x -- %p 0x%lx\n", resized_buf, resized_width*resized_height*3/2, net_input_buf, yuv_buf_len*sizeof(float));

  // test for bad buffers
  static int CNT = 20;
  float avg = 0.0;
  for (int i = 0; i < CNT; i++) {
    double s4 = millis_since_boot();
    inner(resized_buf, net_input_buf);
    double s5 = millis_since_boot();
    avg += s5-s4;
  }
  avg /= CNT;

  // once it's bad, it's reliably bad
  if (avg > 10) {
    printf("HIT %f\n", avg);
    printf("BAD\n");

    for (int i = 0; i < 200; i++) {
      double s4 = millis_since_boot();
      inner(resized_buf, net_input_buf);
      double s5 = millis_since_boot();
      printf("%.2f   ", s5-s4);
    }
    printf("\n");

    exit(0);
  }

  // don't free so we get a different buffer each time
  //free(resized_buf);
  //free(net_input_buf);

  return avg;
}

int main() {
  while (true) {
    float ret = trial();
    printf("got %f\n", ret);
  }
}

