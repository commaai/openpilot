// gcc -O2 waste.c -lpthread -owaste
// gcc -O2 waste.c -lpthread -owaste -DMEM

#define _GNU_SOURCE
#include <stdio.h>
#include <math.h>
#include <sched.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <arm_neon.h>
#include <sys/sysinfo.h>
#include "../common/timing.h"

int get_nprocs(void);
double *ttime, *oout;

void waste(int pid) {
  cpu_set_t my_set;
  CPU_ZERO(&my_set);
  CPU_SET(pid, &my_set);
  int ret = sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
  printf("set affinity to %d: %d\n", pid, ret);

  // 128 MB
  float32x4_t *tmp = (float32x4_t *)malloc(0x800000*sizeof(float32x4_t));

  // comment out the memset for CPU only and not RAM
  // otherwise we need this to avoid the zero page
#ifdef MEM
  memset(tmp, 0xaa, 0x800000*sizeof(float32x4_t));
#endif

  float32x4_t out;

  double sec = seconds_since_boot();
  while (1) {
    for (int i = 0; i < 0x10; i++) {
      for (int j = 0; j < 0x800000; j+=0x20) {
        out = vmlaq_f32(out, tmp[j+0], tmp[j+1]);
        out = vmlaq_f32(out, tmp[j+2], tmp[j+3]);
        out = vmlaq_f32(out, tmp[j+4], tmp[j+5]);
        out = vmlaq_f32(out, tmp[j+6], tmp[j+7]);
        out = vmlaq_f32(out, tmp[j+8], tmp[j+9]);
        out = vmlaq_f32(out, tmp[j+10], tmp[j+11]);
        out = vmlaq_f32(out, tmp[j+12], tmp[j+13]);
        out = vmlaq_f32(out, tmp[j+14], tmp[j+15]);
        out = vmlaq_f32(out, tmp[j+16], tmp[j+17]);
        out = vmlaq_f32(out, tmp[j+18], tmp[j+19]);
        out = vmlaq_f32(out, tmp[j+20], tmp[j+21]);
        out = vmlaq_f32(out, tmp[j+22], tmp[j+23]);
        out = vmlaq_f32(out, tmp[j+24], tmp[j+25]);
        out = vmlaq_f32(out, tmp[j+26], tmp[j+27]);
        out = vmlaq_f32(out, tmp[j+28], tmp[j+29]);
        out = vmlaq_f32(out, tmp[j+30], tmp[j+31]);
      }
    }
    double nsec = seconds_since_boot();
    ttime[pid] = nsec-sec;
    oout[pid] = out[0] + out[1] + out[2] + out[3];
    sec = nsec;
  }
}

int main() {
  int CORES = get_nprocs();
  ttime = (double *)malloc(CORES*sizeof(double));
  oout = (double *)malloc(CORES*sizeof(double));

  pthread_t waster[CORES];
  for (long i = 0; i < CORES; i++) {
    ttime[i] = NAN;
    pthread_create(&waster[i], NULL, (void *(*)(void *))waste, (void*)i);
  }
  while (1) {
    double avg = 0.0;
    double iavg = 0.0;
    for (int i = 0; i < CORES; i++) {
      avg += ttime[i];
      iavg += 1/ttime[i];
      printf("%4.2f ", ttime[i]);
    }
    double mb_per_sec = (16.*0x800000/(1024*1024))*sizeof(float32x4_t)*iavg;
    printf("-- %4.2f -- %.2f MB/s \n", avg/CORES, mb_per_sec);
    sleep(1);
  }
}

