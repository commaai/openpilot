#include <thread>
#include <stdio.h>
#include <signal.h>
#include <poll.h>
#include <assert.h>
#include <unistd.h>
#include <sys/socket.h>

#if defined(QCOM) && !defined(QCOM_REPLAY)
#include "cameras/camera_qcom.h"
#elif QCOM2
#include "cameras/camera_qcom2.h"
#elif WEBCAM
#include "cameras/camera_webcam.h"
#else
#include "cameras/camera_frame_stream.h"
#endif

#include <libyuv.h>

#include "clutil.h"
#include "common/ipc.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/util.h"
#include "common/visionipc.h"

#define MAX_CLIENTS 6

volatile sig_atomic_t do_exit = 0;

static void set_do_exit(int sig) {
  do_exit = 1;
}

struct VisionState;

struct VisionClientState {
  VisionState *s;
  int fd;
  pthread_t thread_handle;
  bool running;
};

struct VisionClientStreamState {
  bool subscribed;
  int bufs_outstanding;
  bool tb;
  TBuffer* tbuffer;
  PoolQueue* queue;
};

struct VisionState {
  MultiCameraState cameras;
  pthread_mutex_t clients_lock;
  VisionClientState clients[MAX_CLIENTS];
};

static CameraBuf *get_camerabuf_by_type(VisionState *s, VisionStreamType type) {
  assert(type >= 0 && type < VISION_STREAM_MAX);
  if (type == VISION_STREAM_RGB_BACK || type == VISION_STREAM_YUV) {
    return &s->cameras.rear.buf;
  } else if (type == VISION_STREAM_RGB_FRONT || type == VISION_STREAM_YUV_FRONT) {
    return &s->cameras.front.buf;
  }
#ifdef QCOM2
  return &s->cameras.wide.buf;
#endif
  assert(0);
}

// visionserver
void* visionserver_client_thread(void* arg) {
  int err;
  VisionClientState *client = (VisionClientState*)arg;
  VisionState *s = client->s;
  int fd = client->fd;

  set_thread_name("clientthread");

  VisionClientStreamState streams[VISION_STREAM_MAX] = {{0}};

  LOGW("client start fd %d", fd);

  while (true) {
    struct pollfd polls[1+VISION_STREAM_MAX] = {{0}};
    polls[0].fd = fd;
    polls[0].events = POLLIN;

    int poll_to_stream[1+VISION_STREAM_MAX] = {0};
    int num_polls = 1;
    for (int i=0; i<VISION_STREAM_MAX; i++) {
      if (!streams[i].subscribed) continue;
      polls[num_polls].events = POLLIN;
      if (streams[i].bufs_outstanding >= 2) {
        continue;
      }
      if (streams[i].tb) {
        polls[num_polls].fd = tbuffer_efd(streams[i].tbuffer);
      } else {
        polls[num_polls].fd = poolq_efd(streams[i].queue);
      }
      poll_to_stream[num_polls] = i;
      num_polls++;
    }
    int ret = poll(polls, num_polls, -1);
    if (ret < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      LOGE("poll failed (%d - %d)", ret, errno);
      break;
    }
    if (do_exit) break;
    if (polls[0].revents) {
      VisionPacket p;
      err = vipc_recv(fd, &p);
      if (err <= 0) {
        break;
      } else if (p.type == VIPC_STREAM_SUBSCRIBE) {
        VisionStreamType stream_type = p.d.stream_sub.type;
        VisionPacket rep = {
          .type = VIPC_STREAM_BUFS,
          .d = { .stream_bufs = { .type = stream_type }, },
        };

        VisionClientStreamState *stream = &streams[stream_type];
        stream->tb = p.d.stream_sub.tbuffer;

        VisionStreamBufs *stream_bufs = &rep.d.stream_bufs;
        CameraBuf *b = get_camerabuf_by_type(s, stream_type);
        if (stream_type == VISION_STREAM_RGB_BACK ||
            stream_type == VISION_STREAM_RGB_FRONT ||
            stream_type == VISION_STREAM_RGB_WIDE) {
          stream_bufs->width = b->rgb_width;
          stream_bufs->height = b->rgb_height;
          stream_bufs->stride = b->rgb_stride;
          stream_bufs->buf_len = b->rgb_bufs[0].len;
          rep.num_fds = UI_BUF_COUNT;
          for (int i = 0; i < rep.num_fds; i++) {
            rep.fds[i] = b->rgb_bufs[i].fd;
          }
          if (stream->tb) {
            stream->tbuffer = &b->ui_tb;
          } else {
            assert(false);
          }
        } else {
          stream_bufs->width = b->yuv_width;
          stream_bufs->height = b->yuv_height;
          stream_bufs->stride = b->yuv_width;
          stream_bufs->buf_len = b->yuv_buf_size;
          rep.num_fds = YUV_COUNT;
          for (int i = 0; i < rep.num_fds; i++) {
            rep.fds[i] = b->yuv_ion[i].fd;
          }
          if (stream->tb) {
            stream->tbuffer = b->yuv_tb;
          } else {
            stream->queue = pool_get_queue(&b->yuv_pool);
          }
        }
        vipc_send(fd, &rep);
        streams[stream_type].subscribed = true;
      } else if (p.type == VIPC_STREAM_RELEASE) {
        int si = p.d.stream_rel.type;
        assert(si < VISION_STREAM_MAX);
        if (streams[si].tb) {
          tbuffer_release(streams[si].tbuffer, p.d.stream_rel.idx);
        } else {
          poolq_release(streams[si].queue, p.d.stream_rel.idx);
        }
        streams[p.d.stream_rel.type].bufs_outstanding--;
      } else {
        assert(false);
      }
    } else {
      int stream_i = VISION_STREAM_MAX;
      for (int i=1; i<num_polls; i++) {
        int si = poll_to_stream[i];
        if (!streams[si].subscribed) continue;
        if (polls[i].revents) {
          stream_i = si;
          break;
        }
      }
      if (stream_i < VISION_STREAM_MAX) {
        streams[stream_i].bufs_outstanding++;
        int idx;
        if (streams[stream_i].tb) {
          idx = tbuffer_acquire(streams[stream_i].tbuffer);
        } else {
          idx = poolq_pop(streams[stream_i].queue);
        }
        if (idx < 0) {
          break;
        }
        VisionPacket rep = {
          .type = VIPC_STREAM_ACQUIRE,
          .d = {.stream_acq = {
            .type = (VisionStreamType)stream_i,
            .idx = idx,
          }},
        };
        if (stream_i == VISION_STREAM_YUV ||
            stream_i == VISION_STREAM_YUV_FRONT ||
            stream_i == VISION_STREAM_YUV_WIDE) {
          CameraBuf *b = get_camerabuf_by_type(s, (VisionStreamType)stream_i);
          rep.d.stream_acq.extra.frame_id = b->yuv_metas[idx].frame_id;
          rep.d.stream_acq.extra.timestamp_eof = b->yuv_metas[idx].timestamp_eof;
        }
        vipc_send(fd, &rep);
      }
    }
  }

  LOGW("client end fd %d", fd);

  for (int i=0; i<VISION_STREAM_MAX; i++) {
    if (!streams[i].subscribed) continue;
    if (streams[i].tb) {
      tbuffer_release_all(streams[i].tbuffer);
    } else {
      pool_release_queue(streams[i].queue);
    }
  }

  close(fd);

  pthread_mutex_lock(&s->clients_lock);
  client->running = false;
  pthread_mutex_unlock(&s->clients_lock);

  return NULL;
}

void* visionserver_thread(void* arg) {
  int err;
  VisionState *s = (VisionState*)arg;

  set_thread_name("visionserver");

  int sock = ipc_bind(VIPC_SOCKET_PATH);
  while (!do_exit) {
    struct pollfd polls[1] = {{0}};
    polls[0].fd = sock;
    polls[0].events = POLLIN;

    int ret = poll(polls, ARRAYSIZE(polls), 1000);
    if (ret < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      LOGE("poll failed (%d - %d)", ret, errno);
      break;
    }
    if (do_exit) break;
    if (!polls[0].revents) {
      continue;
    }

    int fd = accept(sock, NULL, NULL);
    assert(fd >= 0);

    pthread_mutex_lock(&s->clients_lock);

    int client_idx = 0;
    for (; client_idx < MAX_CLIENTS; client_idx++) {
      if (!s->clients[client_idx].running) break;
    }

    if (client_idx >= MAX_CLIENTS) {
      LOG("ignoring visionserver connection, max clients connected");
      close(fd);

      pthread_mutex_unlock(&s->clients_lock);
      continue;
    }

    VisionClientState *client = &s->clients[client_idx];
    client->s = s;
    client->fd = fd;
    client->running = true;

    err = pthread_create(&client->thread_handle, NULL,
                         visionserver_client_thread, client);
    assert(err == 0);

    pthread_mutex_unlock(&s->clients_lock);
  }

  for (int i=0; i<MAX_CLIENTS; i++) {
    pthread_mutex_lock(&s->clients_lock);
    bool running = s->clients[i].running;
    pthread_mutex_unlock(&s->clients_lock);
    if (running) {
      err = pthread_join(s->clients[i].thread_handle, NULL);
      assert(err == 0);
    }
  }

  close(sock);

  return NULL;
}

void party(cl_device_id device_id, cl_context context) {
  VisionState state = {};
  VisionState *s = &state;
  
  cameras_init(&s->cameras, device_id, context);
  cameras_open(&s->cameras);

  std::thread server_thread(visionserver_thread, s);
  
  // priority for cameras
  int err = set_realtime_priority(51);
  LOG("setpriority returns %d", err);

  cameras_run(&s->cameras);

  server_thread.join();
}

int main(int argc, char *argv[]) {
  set_realtime_priority(51);
#if defined(QCOM)
  set_core_affinity(2);
#elif defined(QCOM2)
  set_core_affinity(6);
#endif

  signal(SIGINT, (sighandler_t)set_do_exit);
  signal(SIGTERM, (sighandler_t)set_do_exit);

  int err;
  clu_init();
  cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  assert(err == 0);

  party(device_id, context);

  clReleaseContext(context);
}
