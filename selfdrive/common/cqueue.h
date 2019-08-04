#ifndef COMMON_CQUEUE_H
#define COMMON_CQUEUE_H

#include <sys/queue.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// a blocking queue

typedef struct QueueEntry {
  TAILQ_ENTRY(QueueEntry) entries;
  void *data;
} QueueEntry;

typedef struct Queue {
  TAILQ_HEAD(queue, QueueEntry) q;
  pthread_mutex_t lock;
  pthread_cond_t cv;
} Queue;

void queue_init(Queue *q);
void* queue_pop(Queue *q);
void* queue_try_pop(Queue *q);
void queue_push(Queue *q, void *data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
