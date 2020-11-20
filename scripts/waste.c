// gcc -O2 waste.c -lpthread -owaste
// gcc -O2 waste.c -lpthread -owaste -DMEM

#define _GNU_SOURCE
#include <stdio.h>
#include <math.h>
#include <sched.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <arm_neon.h>
#include <sys/sysinfo.h>
#include "../selfdrive/common/timing.h"

int get_nprocs(void);
double *ttime, *oout;

void waste(int pid) {
  cpu_set_t my_set;
  CPU_ZERO(&my_set);
  CPU_SET(pid, &my_set);
  int ret = sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
  printf("set affinity to %d: %d\n", pid, ret);

  // 256 MB
  float32x4_t *tmp = (float32x4_t *)malloc(0x1000000*sizeof(float32x4_t));

  // comment out the memset for CPU only and not RAM
  // otherwise we need this to avoid the zero page
#ifdef MEM
  memset(tmp, 0xaa, 0x1000000*sizeof(float32x4_t));
#endif

  float32x4_t out;

  double sec = seconds_since_boot();
  while (1) {
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 0x1000000; j+=2) {
        out = vmlaq_f32(out, tmp[j], tmp[j+1]);
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
    for (int i = 0; i < CORES; i++) {
      avg += ttime[i];
      printf("%4.2f ", ttime[i]);
    }
    avg /= CORES;
    double mb_per_sec = (8.*0x1000000/(1024*1024))*sizeof(float32x4_t)*CORES*(1/avg);
    printf("-- %4.2f -- %.2f MB/s \n", avg, mb_per_sec);
    sleep(1);
  }
}

