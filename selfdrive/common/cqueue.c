#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cqueue.h"

void queue_init(Queue *q) {
  memset(q, 0, sizeof(*q));
  TAILQ_INIT(&q->q);
  pthread_mutex_init(&q->lock, NULL);
  pthread_cond_init(&q->cv, NULL);
}

void* queue_pop(Queue *q) {
  pthread_mutex_lock(&q->lock);
  while (TAILQ_EMPTY(&q->q)) {
    pthread_cond_wait(&q->cv, &q->lock);
  }
  QueueEntry *entry = TAILQ_FIRST(&q->q);
  TAILQ_REMOVE(&q->q, entry, entries);
  pthread_mutex_unlock(&q->lock);

  void* r = entry->data;
  free(entry);
  return r;
}

void* queue_try_pop(Queue *q) {
  pthread_mutex_lock(&q->lock);

  void* r = NULL;
  if (!TAILQ_EMPTY(&q->q)) {
    QueueEntry *entry = TAILQ_FIRST(&q->q);
    TAILQ_REMOVE(&q->q, entry, entries);
    r = entry->data;
    free(entry);
  }

  pthread_mutex_unlock(&q->lock);
  return r;
}

void queue_push(Queue *q, void *data) {
  QueueEntry *entry = calloc(1, sizeof(QueueEntry));
  assert(entry);
  entry->data = data;

  pthread_mutex_lock(&q->lock);
  TAILQ_INSERT_TAIL(&q->q, entry, entries);
  pthread_cond_signal(&q->cv);
  pthread_mutex_unlock(&q->lock);
}
