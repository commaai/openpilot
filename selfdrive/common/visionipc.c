#include "visionipc.h"

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/un.h>
#include <unistd.h>

#include "ipc.h"

typedef struct VisionPacketWire {
  int type;
  VisionPacketData d;
} VisionPacketWire;

int vipc_connect() {
  return ipc_connect(VIPC_SOCKET_PATH);
}

int vipc_recv(int fd, VisionPacket *out_p) {
  VisionPacketWire p = {0};
  VisionPacket p2 = {0};
  int ret = ipc_sendrecv_with_fds(false, fd, &p, sizeof(p), (int*)p2.fds, VIPC_MAX_FDS, &p2.num_fds);
  if (ret < 0) {
    printf("vipc_recv err: %s\n", strerror(errno));
  } else {
    p2.type = p.type;
    p2.d = p.d;
    *out_p = p2;
  }
  //printf("%d = vipc_recv(%d, %d): %d %d %d %zu\n", ret, fd, p2.num_fds, out_p->d.stream_bufs.type, out_p->d.stream_bufs.width, out_p->d.stream_bufs.height, out_p->d.stream_bufs.buf_len);
  return ret;
}

int vipc_send(int fd, const VisionPacket *p2) {
  assert(p2->num_fds <= VIPC_MAX_FDS);

  VisionPacketWire p = {
    .type = p2->type,
    .d = p2->d,
  };
  int ret = ipc_sendrecv_with_fds(true, fd, (void*)&p, sizeof(p), (int*)p2->fds, p2->num_fds, NULL);
  //printf("%d = vipc_send(%d, %d): %d %d %d %zu\n", ret, fd, p2->num_fds, p2->d.stream_bufs.type, p2->d.stream_bufs.width, p2->d.stream_bufs.height, p2->d.stream_bufs.buf_len);
  return ret;
}

void vipc_bufs_load(VIPCBuf *bufs, const VisionStreamBufs *stream_bufs,
                     int num_fds, const int* fds) {
  for (int i=0; i<num_fds; i++) {
    bufs[i].fd = fds[i];
    bufs[i].len = stream_bufs->buf_len;
    bufs[i].addr = mmap(NULL, bufs[i].len, PROT_READ, MAP_SHARED, bufs[i].fd, 0);
    // printf("b %d %zu -> %p\n", bufs[i].fd, bufs[i].len, bufs[i].addr);
    assert(bufs[i].addr != MAP_FAILED);
  }
}

int visionstream_init(VisionStream *s, VisionStreamType type, bool tbuffer, VisionStreamBufs *out_bufs_info) {

  memset(s, 0, sizeof(*s));

  s->last_idx = -1;

  s->ipc_fd = vipc_connect();
  if (s->ipc_fd < 0) return -1;

  VisionPacket p = {
    .type = VIPC_STREAM_SUBSCRIBE,
    .d = { .stream_sub = {
      .type = type,
      .tbuffer = tbuffer,
    }, },
  };
  VisionPacket rp;
  if (vipc_send(s->ipc_fd, &p) < 0 ||  vipc_recv(s->ipc_fd, &rp) <= 0) {
    close(s->ipc_fd);
    s->ipc_fd = -1;
    return -1;
  }

  assert(rp.type == VIPC_STREAM_BUFS);
  assert(rp.d.stream_bufs.type == type);

  s->bufs_info = rp.d.stream_bufs;

  s->num_bufs = rp.num_fds;
  s->bufs = calloc(s->num_bufs, sizeof(VIPCBuf));
  assert(s->bufs);

  vipc_bufs_load(s->bufs, &rp.d.stream_bufs, s->num_bufs, rp.fds);

  if (out_bufs_info) {
    *out_bufs_info = s->bufs_info;
  }

  return 0;
}

static int visionstream_release(VisionStream *s) {
  int err = 1;
  if (s->last_idx >= 0) {
    VisionPacket rep = {
      .type = VIPC_STREAM_RELEASE,
      .d = { .stream_rel = {
        .type = s->last_type,
        .idx = s->last_idx,
      }}
    };
    err = vipc_send(s->ipc_fd, &rep);
    s->last_idx = -1;
  }
  return err;
}

VIPCBuf* visionstream_get(VisionStream *s, VIPCBufExtra *out_extra) {
  VisionPacket rp;
  if (vipc_recv(s->ipc_fd, &rp) <= 0) {
    return NULL;
  }
  assert(rp.type == VIPC_STREAM_ACQUIRE);

  if (visionstream_release(s) <= 0) {
    return NULL;
  }

  s->last_type = rp.d.stream_acq.type;
  s->last_idx = rp.d.stream_acq.idx;
  assert(s->last_idx < s->num_bufs);

  if (out_extra) {
    *out_extra = rp.d.stream_acq.extra;
  }

  return &s->bufs[s->last_idx];
}

void visionstream_destroy(VisionStream *s) {
  visionstream_release(s);

  for (int i=0; i<s->num_bufs; i++) {
    if (s->bufs[i].addr) {
      munmap(s->bufs[i].addr, s->bufs[i].len);
      s->bufs[i].addr = NULL;
      close(s->bufs[i].fd);
    }
  }
  if (s->bufs) {
    free(s->bufs);
    s->bufs = NULL;
  }
  if (s->ipc_fd >= 0) {
    close(s->ipc_fd);
    s->ipc_fd = -1;
  }
}