#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <assert.h>
#include <pthread.h>

#include <zmq.h>
#include <libusb.h>

#include <capnp/serialize.h>
#include "cereal/gen/cpp/log.capnp.h"

#include "common/swaglog.h"
#include "common/timing.h"

int do_exit = 0;

libusb_context *ctx = NULL;
pthread_mutex_t usb_lock;

enum board_options {
  DO_HEALTH_CHECK = 1,
  ENABLE_SW_GMLAN = 2,
  OPTIONAL = 4
};

struct board_config {
  // distinguish between boards by different pids
  uint16_t board_pid;
  int options;
  // figure out destination board by target can src
  int src_offset;
  libusb_device_handle *dev_handle;
};

// implicit dependency on boards not reporting
// can src above 15
const int src_offset_shift = 4;

board_config boards[] = {{0xddcc, DO_HEALTH_CHECK, 0, NULL}};

const int num_boards = sizeof(boards) / sizeof(boards[0]);

bool spoofing_started = false;
bool fake_send = false;

// double the FIFO size
#define RECV_SIZE (0x1000)
#define TIMEOUT 0

bool usb_connect() {
  libusb_device_handle *dev_handle;
  int err;

  for (int i = 0; i < num_boards; i++) {
    board_config *config = boards + i;
    bool required = !(config->options & OPTIONAL);
    bool gmlan = !!(config->options & ENABLE_SW_GMLAN);
    uint16_t pid = config->board_pid;

    dev_handle = libusb_open_device_with_vid_pid(ctx, 0xbbaa, pid);
    if (dev_handle != NULL) {
      err = libusb_set_configuration(dev_handle, 1);
      if (err == 0) {
        err = libusb_claim_interface(dev_handle, 0);
      }
      if (err == 0 && gmlan) {
        // enable single-wire GMLAN if needed
        err = libusb_control_transfer(dev_handle, 0xc0, 0xdb, 1, 0, NULL, 0, TIMEOUT);
      }
      if (err != 0) {
        LOGE("failed to connect to board %x, err %d", pid, err);
        libusb_close(dev_handle);
        dev_handle = NULL;
      }
    }

    if (config->dev_handle != NULL) {
      libusb_close(config->dev_handle);
    }
    config->dev_handle = dev_handle;
    config->src_offset = i << src_offset_shift;

    if (dev_handle == NULL) {
      if (required) {
        return false;
      } else {
        // This board is optional, ignore
        continue;
      }
    }
  }

  return true;
}

void usb_retry_connect() {
  LOG("attempting to connect");
  while (!usb_connect()) { usleep(100*1000); }
  LOGW("connected to board");
}

void handle_usb_issue(int err, const char func[]) {
  LOGE("usb error %d \"%s\" in %s", err, libusb_strerror((enum libusb_error)err), func);
  if (err == -4) {
    LOGE("lost connection");
    usb_retry_connect();
  }
  // TODO: check other errors, is simply retrying okay?
}

void can_recv(void *s) {
  libusb_device_handle *dev_handle;
  int err;
  uint32_t data[RECV_SIZE/4 * num_boards];
  int board_msgs[num_boards];
  uint8_t *board_data;
  int recv = 0;
  int board_recv;

  // do recv
  pthread_mutex_lock(&usb_lock);

  for (int i = 0; i < num_boards; i++) {
    dev_handle = boards[i].dev_handle;
    if (dev_handle == NULL) {
      // skip over an optional board
      continue;
    }

    board_data = (uint8_t*)data + recv;

    do {
      err = libusb_bulk_transfer(dev_handle, 0x81, board_data, RECV_SIZE, &board_recv, TIMEOUT);
      if (err != 0) { handle_usb_issue(err, __func__); }
      if (err == -8) { LOGE("overflow got 0x%x", board_recv); };

      // timeout is okay to exit, recv still happened
      if (err == -7) { break; }
    } while(err != 0);

    recv += board_recv;

    // number of messages received from
    // i-th and all prior boards
    board_msgs[i] = recv / 0x10;
  }

  pthread_mutex_unlock(&usb_lock);

  // return if length is 0
  if (recv <= 0) {
    return;
  }

  // create message
  capnp::MallocMessageBuilder msg;
  cereal::Event::Builder event = msg.initRoot<cereal::Event>();
  event.setLogMonoTime(nanos_since_boot());

  int num_msg = recv / 0x10;
  auto canData = event.initCan(num_msg);

  // populate message
  int board_num = 0;
  for (int i = 0; i < num_msg; i++) {
    while (i >= board_msgs[board_num]) {
      board_num++;
    }
    int src_offset = boards[board_num].src_offset;

    uint32_t address;
    if (data[i*4] & 4) {
      // extended
      address = data[i*4] >> 3;
    } else {
      // normal
      address = data[i*4] >> 21;
    }
    canData[i].setAddress(address);
    canData[i].setBusTime(data[i*4+1] >> 16);
    int len = data[i*4+1]&0xF;
    canData[i].setDat(kj::arrayPtr((uint8_t*)&data[i*4+2], len));
    canData[i].setSrc(src_offset + ((data[i*4+1] >> 4) & 0xf));
  }

  // send to can
  auto words = capnp::messageToFlatArray(msg);
  auto bytes = words.asBytes();
  zmq_send(s, bytes.begin(), bytes.size(), 0); 
}

void can_health(void *s) {
  int cnt;

  // copied from board/main.c
  struct health {
    uint32_t voltage;
    uint32_t current;
    uint8_t started;
    uint8_t controls_allowed;
    uint8_t gas_interceptor_detected;
    uint8_t started_signal_detected;
  } health;

  // pick a first health check board
  libusb_device_handle *dev_handle = NULL;
  for (int i = 0; i < num_boards; i++) {
    board_config *config = boards + i;
    if ((config->options & DO_HEALTH_CHECK) &&
      config->dev_handle != NULL) {
      dev_handle = config->dev_handle;
      break;
    }
  }
  if (dev_handle == NULL) {
    LOGE("health check board is unavailable");
    return;
  }

  // recv from board
  pthread_mutex_lock(&usb_lock);

  do {
    cnt = libusb_control_transfer(dev_handle, 0xc0, 0xd2, 0, 0, (unsigned char*)&health, sizeof(health), TIMEOUT);
    if (cnt != sizeof(health)) { handle_usb_issue(cnt, __func__); }
  } while(cnt != sizeof(health));

  pthread_mutex_unlock(&usb_lock);

  // create message
  capnp::MallocMessageBuilder msg;
  cereal::Event::Builder event = msg.initRoot<cereal::Event>();
  event.setLogMonoTime(nanos_since_boot());
  auto healthData = event.initHealth();

  // set fields
  healthData.setVoltage(health.voltage);
  healthData.setCurrent(health.current);
  if (spoofing_started) {
    healthData.setStarted(1);
  } else {
    healthData.setStarted(health.started);
  }
  healthData.setControlsAllowed(health.controls_allowed);
  healthData.setGasInterceptorDetected(health.gas_interceptor_detected);
  healthData.setStartedSignalDetected(health.started_signal_detected);

  // send to health
  auto words = capnp::messageToFlatArray(msg);
  auto bytes = words.asBytes();
  zmq_send(s, bytes.begin(), bytes.size(), 0); 
}


void can_send(void *s) {
  int err;

  // recv from sendcan
  zmq_msg_t msg;
  zmq_msg_init(&msg);
  err = zmq_msg_recv(&msg, s, 0);
  assert(err >= 0);

  // format for board, make copy due to alignment issues, will be freed on out of scope
  auto amsg = kj::heapArray<capnp::word>((zmq_msg_size(&msg) / sizeof(capnp::word)) + 1);
  memcpy(amsg.begin(), zmq_msg_data(&msg), zmq_msg_size(&msg));

  capnp::FlatArrayMessageReader cmsg(amsg);
  cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();
  int msg_count = event.getCan().size();
  if (msg_count == 0)
    return;

  // overallocate to not to deal with sorting messages
  uint32_t *send = (uint32_t*)malloc(msg_count * 0x10 * num_boards);
  memset(send, 0, msg_count * 0x10 * num_boards);
  uint32_t *board_send[num_boards];
  int board_msgs[num_boards];

  for (int i = 0; i < num_boards; i++) {
    // per-board send buffer, preserves
    // original message order
    board_send[i] = send + i * msg_count * 4;
    // number of messages in per-board buffer
    board_msgs[i] = 0;
  }

  for (int i = 0; i < msg_count; i++) {
    auto cmsg = event.getSendcan()[i];

    // route message to specific board based on src
    int src = cmsg.getSrc();
    int board_num = src >> src_offset_shift;
    src -= board_num << src_offset_shift;
    uint32_t *bsend = board_send[board_num];
    int j = 4 * (board_msgs[board_num]++);

    int transmit = 1;
    int extended = 4;
    if (cmsg.getAddress() >= 0x800) {
      // extended
      bsend[j] = (cmsg.getAddress() << 3) | transmit | extended;
    } else {
      // normal
      bsend[j] = (cmsg.getAddress() << 21) | transmit;
    }
    assert(cmsg.getDat().size() <= 8);
    bsend[j+1] = cmsg.getDat().size() | (src << 4);
    memcpy(&bsend[j + 2], cmsg.getDat().begin(), cmsg.getDat().size());
  }

  // release msg
  zmq_msg_close(&msg);

  // send to board
  int sent;
  pthread_mutex_lock(&usb_lock);

  if (!fake_send) {
    for (int i = 0; i < num_boards; i++) {
      int msgs = board_msgs[i];
      if (msgs == 0) {
        continue;
      }

      libusb_device_handle *dev_handle = boards[i].dev_handle;
      if (dev_handle == NULL) {
        LOGE("target board %d (pid %x) unavailable", i, boards[i].board_pid);
        continue;
      }

      do {
        err = libusb_bulk_transfer(dev_handle, 3, (uint8_t*)board_send[i], msgs*0x10, &sent, TIMEOUT);
        if (err != 0 || msgs*0x10 != sent) { handle_usb_issue(err, __func__); }
      } while(err != 0);
    }
  }

  pthread_mutex_unlock(&usb_lock);

  // done
  free(send);
}


// **** threads ****

void *can_send_thread(void *crap) {
  LOGD("start send thread");

  // sendcan = 8017
  void *context = zmq_ctx_new();
  void *subscriber = zmq_socket(context, ZMQ_SUB);
  zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "", 0);
  zmq_connect(subscriber, "tcp://127.0.0.1:8017");

  // run as fast as messages come in
  while (!do_exit) {
    can_send(subscriber);
  }
  return NULL;
}

void *can_recv_thread(void *crap) {
  LOGD("start recv thread");

  // can = 8006
  void *context = zmq_ctx_new();
  void *publisher = zmq_socket(context, ZMQ_PUB);
  zmq_bind(publisher, "tcp://*:8006");

  // run at ~200hz
  while (!do_exit) {
    can_recv(publisher);
    // 5ms
    usleep(5*1000);
  }
  return NULL;
}

void *can_health_thread(void *crap) {
  LOGD("start health thread");

  // health = 8011
  void *context = zmq_ctx_new();
  void *publisher = zmq_socket(context, ZMQ_PUB);
  zmq_bind(publisher, "tcp://*:8011");

  // run at 1hz
  while (!do_exit) {
    can_health(publisher);
    usleep(1000*1000);
  }
  return NULL;
}

int set_realtime_priority(int level) {
  // should match python using chrt
  struct sched_param sa;
  memset(&sa, 0, sizeof(sa));
  sa.sched_priority = level;
  return sched_setscheduler(getpid(), SCHED_FIFO, &sa);
}

int main() {
  int err;
  LOGW("starting boardd");

  // set process priority
  err = set_realtime_priority(4);
  LOG("setpriority returns %d", err);

  // check the environment
  if (getenv("STARTED")) {
    spoofing_started = true;
  }

  if (getenv("FAKESEND")) {
    fake_send = true;
  }

  // init libusb
  err = libusb_init(&ctx);
  assert(err == 0);
  libusb_set_debug(ctx, 3);

  // connect to the board
  usb_retry_connect();


  // create threads
  pthread_t can_health_thread_handle;
  err = pthread_create(&can_health_thread_handle, NULL,
                       can_health_thread, NULL);
  assert(err == 0);

  pthread_t can_send_thread_handle;
  err = pthread_create(&can_send_thread_handle, NULL,
                       can_send_thread, NULL);
  assert(err == 0);

  pthread_t can_recv_thread_handle;
  err = pthread_create(&can_recv_thread_handle, NULL,
                       can_recv_thread, NULL);
  assert(err == 0);

  // join threads

  err = pthread_join(can_recv_thread_handle, NULL);
  assert(err == 0);

  err = pthread_join(can_send_thread_handle, NULL);
  assert(err == 0);

  err = pthread_join(can_health_thread_handle, NULL);
  assert(err == 0);

  // destruct libusb

  for (int i = 0; i < num_boards; i++) {
    libusb_device_handle *dev_handle = boards[i].dev_handle;
    if (dev_handle != NULL) {
      libusb_close(dev_handle);
    }
  }

  libusb_exit(ctx);
}

