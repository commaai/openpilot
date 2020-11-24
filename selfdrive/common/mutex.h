#ifndef COMMON_MUTEX_H
#define COMMON_MUTEX_H

#include <pthread.h>

static inline void mutex_init_reentrant(pthread_mutex_t *mutex) {
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(mutex, &attr);
}

#endif
