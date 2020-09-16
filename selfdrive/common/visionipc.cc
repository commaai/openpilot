#include "visionipc.h"

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/un.h>
#include <unistd.h>
#include <poll.h>
#include "ipc.h"
#include "common/swaglog.h"

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

bool VisionStream::connect(VisionStreamType type, bool tbuffer) {
  last_idx = -1;

  ipc_fd = vipc_connect();
  if (ipc_fd < 0) return false;

  VisionPacket p = {
    .type = VIPC_STREAM_SUBSCRIBE,
    .d = { .stream_sub = {
      .type = type,
      .tbuffer = tbuffer,
    }, },
  };
  VisionPacket rp;
  if (vipc_send(ipc_fd, &p) < 0 || vipc_recv(ipc_fd, &rp) <= 0) {
    close(ipc_fd);
    ipc_fd = -1;
    return false;
  }

  assert(rp.type == VIPC_STREAM_BUFS);
  assert(rp.d.stream_bufs.type == type);

  bufs_info = rp.d.stream_bufs;

  num_bufs = rp.num_fds;
  bufs.reset(new VIPCBuf[num_bufs]);

  for (int i=0; i<num_bufs; i++) {
    bufs[i].fd = rp.fds[i];
    bufs[i].len = bufs_info.buf_len;
    bufs[i].addr = mmap(NULL, bufs[i].len, PROT_READ, MAP_SHARED, bufs[i].fd, 0);
    // printf("b %d %zu -> %p\n", bufs[i].fd, bufs[i].len, bufs[i].addr);
    assert(bufs[i].addr != MAP_FAILED);
  }

  last_type = type;
  return true;
}

VIPCBuf* VisionStream::recv(VIPCBufExtra *out_extra) {
  VisionPacket rp;
  if (vipc_recv(ipc_fd, &rp) <= 0) {
    return nullptr;
  }
  assert(rp.type == VIPC_STREAM_ACQUIRE);

  if (!release()) {
    return nullptr;
  }
  
  const auto &stream_acq = rp.d.stream_acq;
  last_type = stream_acq.type;
  last_idx = stream_acq.idx;
  assert(last_idx < num_bufs);

  if (out_extra) {
    *out_extra = stream_acq.extra;
  }
  return &bufs[last_idx];
}

VIPCBuf* VisionStream::acquire(VisionStreamType type, bool tbuffer, VIPCBufExtra *out_extra, bool use_poll) {
  if (ipc_fd == -1) {
    if (!connect(type, tbuffer)) {
      LOGW("visionstream connect failed");
      usleep(100000);
      return nullptr;
    }
    LOGW("connected with buffer size: %d", bufs_info.buf_len);
  }

  VIPCBuf *buf = recv(out_extra);
  if (buf == nullptr) {
    LOGW("visionstream get failed");
    disconnect();
  }
  return buf;
}

bool VisionStream::release() {
  if (last_idx >= 0) {
    VisionPacket rep = {
        .type = VIPC_STREAM_RELEASE,
        .d = {.stream_rel = {
                  .type = last_type,
                  .idx = last_idx,
              }}};
    int err = vipc_send(ipc_fd, &rep);
    last_idx = -1;
    return (err > 0);
  }
  return true;
}

void VisionStream::disconnect() {
  release();

  for (int i=0; i<num_bufs; i++) {
    if (bufs[i].addr) {
      munmap(bufs[i].addr, bufs[i].len);
      bufs[i].addr = nullptr;
      close(bufs[i].fd);
    }
  }
  if (ipc_fd >= 0) {
    close(ipc_fd);
    ipc_fd = -1;
  }
}

VisionStream::~VisionStream() {
  disconnect();
}
