// single:  clang -O2 -march=native gemm.c
// multi:   clang -O2 -march=native gemm.c -DNTHREADS=32 -lpthread
#define _GNU_SOURCE

// https://en.wikichip.org/wiki/amd/microarchitectures/zen_2
#include <stdint.h>
#include <time.h>
#include <sched.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <pthread.h>
#include <unistd.h>
#include <stdatomic.h>

//#define DEBUG

#ifdef DEBUG
  #define N 8
#endif

#ifndef N
  // NOTE: if you change this you have to rerun gemm.py
  #define N 512
#endif

#ifndef NTHREADS
  #define NTHREADS 1
#endif

// aligned?
float A[N*N] __attribute__ ((aligned (64)));
float B[N*N] __attribute__ ((aligned (64)));
float C[N*N] __attribute__ ((aligned (64)));
float val[N*N] __attribute__ ((aligned (64)));

__m256 *Am = (__m256*)A;
__m256 *Bm = (__m256*)B;
__m256 *Cm = (__m256*)C;

uint64_t nanos() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec*1000000000 + (uint64_t)start.tv_nsec;
}

float Bf[N*N] __attribute__ ((aligned (64)));
__m256 *Bfm = (__m256*)Bf;

#define BLOCK 8
#define BLOCK_Y 4
#define BLOCK_X 2
void matmul(int sy, int ey) {
  // 136.77 GFLOPS on single core numpy
  // 4.9 GHz is max boost for 5950X
  // 32 FLOPS/cycle (16 FMAs, aka 2x 8 single wide / 32 byte FMAs)
  // theoretical max is 156.8 GFLOPS, we see 150
  // multicore theo max = 2508.8 GFLOPS, we see 1501.434299

  // Bf = (y/8, k, 8)
  for (int y = sy; y < ey; y+=BLOCK_Y) {
    for (int x = 0; x < N; x+=BLOCK*BLOCK_X) {

      __m256 acc[BLOCK_Y][BLOCK_X] = {};
      for (int k = 0; k < N; k++) {
        for (int iy = 0; iy < BLOCK_Y; iy++) {
          __m256 ta = _mm256_broadcast_ss(&A[(y+iy)*N + k]);
          for (int ix = 0; ix < BLOCK_X; ix++) {
            acc[iy][ix] = _mm256_fmadd_ps(ta, Bfm[((x+ix*BLOCK)*N + k*8)/8], acc[iy][ix]);
          }
        }
      }

      for (int iy = 0; iy < BLOCK_Y; iy++) {
        for (int ix = 0; ix < BLOCK_X; ix++) {
          Cm[((y+iy)*N + x + ix * BLOCK)/8] = acc[iy][ix];
        }
      }
    }
  }
}

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
atomic_int nready = 0;
atomic_int ndone = 0;

void *matmul_thread(void *n) {
  int k = (int)(int64_t)n;
  int sy = (N/NTHREADS) * k;
  int ey = (N/NTHREADS) * (k+1);

  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(k,&set);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);

  nready++;

  // gotta have main lock once to signal start
  pthread_mutex_lock(&lock);
  pthread_mutex_unlock(&lock);

  matmul(sy, ey);

  // we done
  ndone++;
  return NULL;
}

int main() {
  printf("hello with %d threads\n", NTHREADS);

#ifdef DEBUG
  for (int i = 0; i < N*N; i++) A[i] = i;
  for (int i = 0; i < N*N; i++) B[i] = i;
#else
  FILE *f = fopen("/tmp/matmul", "rb");
  if (f == NULL) {
    printf("please pregenerate python /tmp/matmul file\n");
    return -1;
  }
  fread(A, 1, sizeof(float)*N*N, f);
  fread(B, 1, sizeof(float)*N*N, f);
  fread(val, 1, sizeof(float)*N*N, f);
  fclose(f);
#endif

  // preswizzle
  for (int y = 0; y < N; y+=8) {
    for (int x = 0; x < N; x++) {
      for (int iy = 0; iy < 8; iy++) {
        Bf[y*N + x*8 + iy] = B[(y+iy)*N + x];
      }
    }
  }

  for (int i = 0; i < 10; i++) {
    memset(C, 0, N*N*sizeof(float));

#if NTHREADS != 1
    nready = 0;
    ndone = 0;
    pthread_mutex_lock(&lock);
    pthread_t threads[NTHREADS];
    for (int j = 0; j < NTHREADS; j++) {
      pthread_create(&threads[j], NULL, matmul_thread, (void *)(uint64_t)j);
    }
    while (nready != NTHREADS) usleep(1);
#endif

    uint64_t start = nanos();
#if NTHREADS == 1
    matmul(0, N);
#else
    // unlocking mutex starts threads
    pthread_mutex_unlock(&lock);
    while (ndone != NTHREADS) usleep(1);
#endif
    uint64_t end = nanos();

#if NTHREADS != 1
    for (int j = 0; j < NTHREADS; j++) {
      pthread_join(threads[j], NULL);
    }
#endif

    double gflop = (2.0*N*N*N)*1e-9;
    double s = (end-start)*1e-9;
    printf("%f GFLOP/S -- %.2f ms\n", gflop/s, s*1e3);

    // hack around throttling
    //if (i%4 == 0) sleep(1);
  }

#ifdef DEBUG
  for (int i = 0; i < N*N; i++) {
    if (i%N == 0 && i != 0) printf("\n");
    printf("%f ", C[i]);
  }
  printf("\n");
#else
  for (int k = 0; k < N*N; k++) {
    if (fabsf(C[k] - val[k]) > 1e-3) {
      printf("MISMATCH AT %d, %f != %f\n", k, C[k], val[k]);
      return -1;
    }
  }
  printf("match\n");
#endif

  return 0;
}
