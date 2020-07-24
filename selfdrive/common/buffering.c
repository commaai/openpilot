#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>

#include "common/efd.h"

#include "buffering.h"

void tbuffer_init(TBuffer *tb, int num_bufs, const char* name) {
  assert(num_bufs >= 3);

  memset(tb, 0, sizeof(TBuffer));
  tb->reading = (bool*)calloc(num_bufs, sizeof(bool));
  assert(tb->reading);
  tb->pending_idx = -1;
  tb->num_bufs = num_bufs;
  tb->name = name;

  pthread_mutex_init(&tb->lock, NULL);
  pthread_cond_init(&tb->cv, NULL);
  tb->efd = efd_init();
  assert(tb->efd >= 0);
}

void tbuffer_init2(TBuffer *tb, int num_bufs, const char* name,
                   void (*release_cb)(void* c, int idx),
                   void* cb_cookie) {

  tbuffer_init(tb, num_bufs, name);

  tb->release_cb = release_cb;
  tb->cb_cookie = cb_cookie;
}

int tbuffer_efd(TBuffer *tb) {
  return tb->efd;
}

int tbuffer_select(TBuffer *tb) {
  pthread_mutex_lock(&tb->lock);

  int i;
  for (i=0; i<tb->num_bufs; i++) {
    if (!tb->reading[i] && i != tb->pending_idx) {
      break;
    }
  }
  assert(i < tb->num_bufs);

  pthread_mutex_unlock(&tb->lock);
  return i;
}

void tbuffer_dispatch(TBuffer *tb, int idx) {
  pthread_mutex_lock(&tb->lock);

  if (tb->pending_idx != -1) {
    //printf("tbuffer (%s) dropped!\n", tb->name ? tb->name : "?");
    if (tb->release_cb) {
      tb->release_cb(tb->cb_cookie, tb->pending_idx);
    }
    tb->pending_idx = -1;
  }

  tb->pending_idx = idx;

  efd_write(tb->efd);
  pthread_cond_signal(&tb->cv);

  pthread_mutex_unlock(&tb->lock);
}

int tbuffer_acquire(TBuffer *tb) {
  pthread_mutex_lock(&tb->lock);

  if (tb->stopped) {
    pthread_mutex_unlock(&tb->lock);
    return -1;
  }

  while (tb->pending_idx == -1) {
    pthread_cond_wait(&tb->cv, &tb->lock);

    if (tb->stopped) {
      pthread_mutex_unlock(&tb->lock);
      return -1;
    }
  }

  efd_clear(tb->efd);

  int ret = tb->pending_idx;
  assert(ret < tb->num_bufs);

  tb->reading[ret] = true;
  tb->pending_idx = -1;

  pthread_mutex_unlock(&tb->lock);

  return ret;
}

static void tbuffer_release_locked(TBuffer *tb, int idx) {
  assert(idx < tb->num_bufs);
  if (!tb->reading[idx]) {
    printf("!! releasing tbuffer we aren't reading %d\n", idx);
  }

  if (tb->release_cb) {
    tb->release_cb(tb->cb_cookie, idx);
  }

  tb->reading[idx] = false;
}

void tbuffer_release(TBuffer *tb, int idx) {
  pthread_mutex_lock(&tb->lock);
  tbuffer_release_locked(tb, idx);
  pthread_mutex_unlock(&tb->lock);
}

void tbuffer_release_all(TBuffer *tb) {
  pthread_mutex_lock(&tb->lock);
  for (int i=0; i<tb->num_bufs; i++) {
    if (tb->reading[i]) {
      tbuffer_release_locked(tb, i);
    }
  }
  pthread_mutex_unlock(&tb->lock);
}

void tbuffer_stop(TBuffer *tb) {
  pthread_mutex_lock(&tb->lock);
  tb->stopped = true;
  efd_write(tb->efd);
  pthread_cond_signal(&tb->cv);
  pthread_mutex_unlock(&tb->lock);
}











void pool_init(Pool *s, int num_bufs) {
  assert(num_bufs > 3);

  memset(s, 0, sizeof(*s));
  s->num_bufs = num_bufs;

  s->refcnt = (int*)calloc(num_bufs, sizeof(int));
  s->ts = (int*)calloc(num_bufs, sizeof(int));

  s->counter = 1;

  pthread_mutex_init(&s->lock, NULL);
}

void pool_init2(Pool *s, int num_bufs,
  void (*release_cb)(void* c, int idx), void* cb_cookie) {

  pool_init(s, num_bufs);
  s->cb_cookie = cb_cookie;
  s->release_cb = release_cb;

}


void pool_acquire(Pool *s, int idx) {
  pthread_mutex_lock(&s->lock);

  assert(idx >= 0 && idx < s->num_bufs);

  s->refcnt[idx]++;

  pthread_mutex_unlock(&s->lock);
}

static void pool_release_locked(Pool *s, int idx) {
  // printf("release %d refcnt %d\n", idx, s->refcnt[idx]);

  assert(idx >= 0 && idx < s->num_bufs);

  assert(s->refcnt[idx] > 0);
  s->refcnt[idx]--;

  // printf("release %d -> %d, %p\n", idx, s->refcnt[idx], s->release_cb);
  if (s->refcnt[idx] == 0 && s->release_cb) {
    // printf("call %p\b", s->release_cb);
    s->release_cb(s->cb_cookie, idx);
  }
}

void pool_release(Pool *s, int idx) {
  pthread_mutex_lock(&s->lock);
  pool_release_locked(s, idx);
  pthread_mutex_unlock(&s->lock);
}

TBuffer* pool_get_tbuffer(Pool *s) {
  pthread_mutex_lock(&s->lock);

  assert(s->num_tbufs < POOL_MAX_TBUFS);
  TBuffer* tbuf = &s->tbufs[s->num_tbufs];
  s->num_tbufs++;
  tbuffer_init2(tbuf, s->num_bufs,
                "pool", (void (*)(void *, int))pool_release, s);

  bool stopped = s->stopped;
  pthread_mutex_unlock(&s->lock);

  // Stop the tbuffer so we can return a valid object.
  // We must stop here because the pool_stop may have already been called,
  // in which case tbuffer_stop may never be called again.
  if (stopped) {
    tbuffer_stop(tbuf);
  }
  return tbuf;
}

PoolQueue* pool_get_queue(Pool *s) {
  pthread_mutex_lock(&s->lock);

  int i;
  for (i = 0; i < POOL_MAX_QUEUES; i++) {
    if (!s->queues[i].inited) {
      break;
    }
  }
  assert(i < POOL_MAX_QUEUES);

  PoolQueue *c = &s->queues[i];
  memset(c, 0, sizeof(*c));

  c->pool = s;
  c->inited = true;

  c->efd = efd_init();
  assert(c->efd >= 0);

  c->num_bufs = s->num_bufs;
  c->num = c->num_bufs+1;
  c->idx = (int*)malloc(sizeof(int)*c->num);
  memset(c->idx, -1, sizeof(int)*c->num);

  pthread_mutex_init(&c->lock, NULL);
  pthread_cond_init(&c->cv, NULL);

  pthread_mutex_unlock(&s->lock);
  return c;
}

void pool_release_queue(PoolQueue *c) {
  Pool *s = c->pool;

  pthread_mutex_lock(&s->lock);
  pthread_mutex_lock(&c->lock);

  for (int i=0; i<c->num; i++) {
    if (c->idx[i] != -1) {
      pool_release_locked(s, c->idx[i]);
    }
  }

  close(c->efd);
  free(c->idx);

  c->inited = false;

  pthread_mutex_unlock(&c->lock);

  pthread_mutex_destroy(&c->lock);
  pthread_cond_destroy(&c->cv);

  pthread_mutex_unlock(&s->lock);
}

int pool_select(Pool *s) {
  pthread_mutex_lock(&s->lock);

  int i;
  for (i=0; i<s->num_bufs; i++) {
    if (s->refcnt[i] == 0) {
      break;
    }
  }

  if (i >= s->num_bufs) {
    // overwrite the oldest
    // still being using in a queue or tbuffer :/

    int min_k = 0;
    int min_ts = s->ts[0];
    for (int k=1; k<s->num_bufs; k++) {
      if (s->ts[k] < min_ts) {
        min_ts = s->ts[k];
        min_k = k;
      }
    }
    i = min_k;
    printf("pool is full! evicted %d\n", min_k);

    // might be really bad if the user is doing pointery stuff
    if (s->release_cb) {
      s->release_cb(s->cb_cookie, min_k);
    }
  }

  s->refcnt[i]++;

  s->ts[i] = s->counter;
  s->counter++;

  pthread_mutex_unlock(&s->lock);

  return i;
}

void pool_push(Pool *s, int idx) {
  pthread_mutex_lock(&s->lock);

  // printf("push %d head %d tail %d\n", idx, s->head, s->tail);

  assert(idx >= 0 && idx < s->num_bufs);

  s->ts[idx] = s->counter;
  s->counter++;

  assert(s->refcnt[idx] > 0);
  s->refcnt[idx]--; //push is a implcit release

  int num_tbufs = s->num_tbufs;
  s->refcnt[idx] += num_tbufs;

  // dispatch pool queues
  for (int i=0; i<POOL_MAX_QUEUES; i++) {
    PoolQueue *c = &s->queues[i];
    if (!c->inited) continue;

    pthread_mutex_lock(&c->lock);
    if (((c->head+1) % c->num) == c->tail) {
      // queue is full. skip for now
      pthread_mutex_unlock(&c->lock);
      continue;
    }

    s->refcnt[idx]++;

    c->idx[c->head] = idx;
    c->head = (c->head+1) % c->num;
    assert(c->head != c->tail);
    pthread_mutex_unlock(&c->lock);

    efd_write(c->efd);
    pthread_cond_signal(&c->cv);
  }

  pthread_mutex_unlock(&s->lock);

  for (int i=0; i<num_tbufs; i++) {
    tbuffer_dispatch(&s->tbufs[i], idx);
  }
}

int poolq_pop(PoolQueue *c) {
  pthread_mutex_lock(&c->lock);

  if (c->stopped) {
    pthread_mutex_unlock(&c->lock);
    return -1;
  }

  while (c->head == c->tail) {
    pthread_cond_wait(&c->cv, &c->lock);

    if (c->stopped) {
      pthread_mutex_unlock(&c->lock);
      return -1;
    }
  }

  // printf("pop head %d tail %d\n", s->head, s->tail);

  assert(c->head != c->tail);

  int r = c->idx[c->tail];
  c->idx[c->tail] = -1;
  c->tail = (c->tail+1) % c->num;

  // queue event is level triggered
  if (c->head == c->tail) {
    efd_clear(c->efd);
  }

  // printf("pop %d head %d tail %d\n", r, s->head, s->tail);

  assert(r >= 0 && r < c->num_bufs);

  pthread_mutex_unlock(&c->lock);

  return r;
}

int poolq_efd(PoolQueue *c) {
  return c->efd;
}

void poolq_release(PoolQueue *c, int idx) {
  pool_release(c->pool, idx);
}

void pool_stop(Pool *s) {
  for (int i=0; i<s->num_tbufs; i++) {
    tbuffer_stop(&s->tbufs[i]);
  }

  pthread_mutex_lock(&s->lock);
  s->stopped = true;
  for (int i=0; i<POOL_MAX_QUEUES; i++) {
    PoolQueue *c = &s->queues[i];
    if (!c->inited) continue;

    pthread_mutex_lock(&c->lock);
    c->stopped = true;
    pthread_mutex_unlock(&c->lock);
    efd_write(c->efd);
    pthread_cond_signal(&c->cv);
  }
  pthread_mutex_unlock(&s->lock);
}
