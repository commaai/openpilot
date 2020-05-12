#ifndef BUFFERING_H
#define BUFFERING_H

#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// Tripple buffering helper

typedef struct TBuffer {
    pthread_mutex_t lock;
    pthread_cond_t cv;
    int efd;

    bool* reading;
    int pending_idx;

    int num_bufs;
    const char* name;

    void (*release_cb)(void* c, int idx);
    void *cb_cookie;

    bool stopped;
} TBuffer;

// num_bufs must be at least the number of buffers that can be acquired simultaniously plus two
void tbuffer_init(TBuffer *tb, int num_bufs, const char* name);

void tbuffer_init2(TBuffer *tb, int num_bufs, const char* name,
  void (*release_cb)(void* c, int idx),
  void* cb_cookie);

// returns an eventfd that signals if a buffer is ready and tbuffer_acquire shouldn't to block.
// useful to polling on multiple tbuffers.
int tbuffer_efd(TBuffer *tb);

// Chooses a buffer that's not reading or pending
int tbuffer_select(TBuffer *tb);

// Called when the writer is done with their buffer
//  - Wakes up the reader if it's waiting
//  - releases the pending buffer if the reader's too slow
void tbuffer_dispatch(TBuffer *tb, int idx);

// Called when the reader wants a new buffer, will return -1 when stopped
int tbuffer_acquire(TBuffer *tb);

// Called when the reader is done with their buffer
void tbuffer_release(TBuffer *tb, int idx);

void tbuffer_release_all(TBuffer *tb);

void tbuffer_stop(TBuffer *tb);




// pool: buffer pool + queue thing...

#define POOL_MAX_TBUFS 8
#define POOL_MAX_QUEUES 8

typedef struct Pool Pool;

typedef struct PoolQueue {
  pthread_mutex_t lock;
  pthread_cond_t cv;
  Pool* pool;
  bool inited;
  bool stopped;
  int efd;
  int num_bufs;
  int num;
  int head, tail;
  int* idx;
} PoolQueue;

int poolq_pop(PoolQueue *s);
int poolq_efd(PoolQueue *s);
void poolq_release(PoolQueue *c, int idx);

typedef struct Pool {
  pthread_mutex_t lock;
  bool stopped;
  int num_bufs;
  int counter;

  int* ts;
  int* refcnt;

  void (*release_cb)(void* c, int idx);
  void *cb_cookie;

  int num_tbufs;
  TBuffer tbufs[POOL_MAX_TBUFS];
  PoolQueue queues[POOL_MAX_QUEUES];
} Pool;

void pool_init(Pool *s, int num_bufs);
void pool_init2(Pool *s, int num_bufs,
  void (*release_cb)(void* c, int idx), void* cb_cookie);

TBuffer* pool_get_tbuffer(Pool *s);

PoolQueue* pool_get_queue(Pool *s);
void pool_release_queue(PoolQueue *q);

int pool_select(Pool *s);
void pool_push(Pool *s, int idx);
void pool_acquire(Pool *s, int idx);
void pool_release(Pool *s, int idx);
void pool_stop(Pool *s);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif
