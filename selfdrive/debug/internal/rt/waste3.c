#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <arm_neon.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include "../../common/util.h"
#include "../../common/timing.h"

#define CORES 3
double ttime[CORES];
double oout[CORES];

void waste(int core) {
  prctl(PR_SET_NAME, (unsigned long)"waste", 0, 0, 0);

  cpu_set_t my_set;
  CPU_ZERO(&my_set);
  CPU_SET(core, &my_set);
  int ret = sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
  printf("set affinity to %d: %d\n", core, ret);

  //struct sched_param sa;
  //memset(&sa, 0, sizeof(sa));
  //sa.sched_priority = 51;
  //sched_setscheduler(syscall(SYS_gettid), SCHED_FIFO, &sa);

  float32x4_t *tmp = (float32x4_t *)malloc(0x1000008*sizeof(float32x4_t));
  float32x4_t out;

  uint64_t i = 0;
  double sec = seconds_since_boot();
  while(1) {
    int j;
    for (j = 0; j < 0x1000000; j++) {
      out = vmlaq_f32(out, tmp[j], tmp[j+1]);
    }
    if (i == 0x8) {
      double nsec = seconds_since_boot();
      ttime[core] = nsec-sec;
      oout[core] = out[0] + out[1] + out[2] + out[3];
      i = 0;
      sec = nsec;
    }
    i++;
  }
}

int main() {
  pthread_t waster[CORES];
  for (int i = 0 ; i < CORES; i++) {
    pthread_create(&waster[i], NULL, waste, (void*)i);
  }
  while (1) {
    for (int i = 0 ; i < CORES; i++) {
      printf("%.2f ", ttime[i]);
    }
    printf("\n");
    sleep(1);
  }
}

