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
#include "cereal/gen/cpp/car.capnp.h"

#include "common/params.h"
#include "common/swaglog.h"
#include "common/timing.h"

#include <algorithm>

// double the FIFO size
#define RECV_SIZE (0x1000)
#define TIMEOUT 0

#define SAFETY_NOOUTPUT  0
#define SAFETY_HONDA 1
#define SAFETY_TOYOTA 2
#define SAFETY_ELM327 0xE327
#define SAFETY_GM 3
#define SAFETY_HONDA_BOSCH 4
#define SAFETY_FORD 5
#define SAFETY_CADILLAC 6
#define SAFETY_HYUNDAI 7
#define SAFETY_TESLA 8
#define SAFETY_TOYOTA_IPAS 0x1335
#define SAFETY_TOYOTA_NOLIMITS 0x1336
#define SAFETY_ALLOUTPUT 0x1337
#define SAFETY_ELM327 0xE327

namespace {

volatile int do_exit = 0;

libusb_context *ctx = NULL;
libusb_device_handle *dev_handle;
pthread_mutex_t usb_lock;

bool spoofing_started = false;
bool fake_send = false;
bool loopback_can = false;
bool is_grey_panda = false;

pthread_t safety_setter_thread_handle = -1;
pthread_t pigeon_thread_handle = -1;
bool pigeon_needs_init;

void pigeon_init();
void *pigeon_thread(void *crap);

void *safety_setter_thread(void *s) {
  char *value;
  size_t value_sz = 0;

  LOGW("waiting for params to set safety model");
  while (1) {
    if (do_exit) return NULL;

    const int result = read_db_value(NULL, "CarParams", &value, &value_sz);
    if (value_sz > 0) break;
    usleep(100*1000);
  }
  LOGW("got %d bytes CarParams", value_sz);

  // format for board, make copy due to alignment issues, will be freed on out of scope
  auto amsg = kj::heapArray<capnp::word>((value_sz / sizeof(capnp::word)) + 1);
  memcpy(amsg.begin(), value, value_sz);

  capnp::FlatArrayMessageReader cmsg(amsg);
  cereal::CarParams::Reader car_params = cmsg.getRoot<cereal::CarParams>();

  auto safety_model = car_params.getSafetyModel();
  auto safety_param = car_params.getSafetyParam();
  LOGW("setting safety model: %d with param %d", safety_model, safety_param);

  int safety_setting = 0;
  switch (safety_model) {
  case (int)cereal::CarParams::SafetyModels::NO_OUTPUT:
    safety_setting = SAFETY_NOOUTPUT;
    break;
  case (int)cereal::CarParams::SafetyModels::HONDA:
    safety_setting = SAFETY_HONDA;
    break;
  case (int)cereal::CarParams::SafetyModels::TOYOTA:
    safety_setting = SAFETY_TOYOTA;
    break;
  case (int)cereal::CarParams::SafetyModels::ELM327:
    safety_setting = SAFETY_ELM327;
    break;
  case (int)cereal::CarParams::SafetyModels::GM:
    safety_setting = SAFETY_GM;
    break;
  case (int)cereal::CarParams::SafetyModels::HONDA_BOSCH:
    safety_setting = SAFETY_HONDA_BOSCH;
    break;
  case (int)cereal::CarParams::SafetyModels::FORD:
    safety_setting = SAFETY_FORD;
    break;
  case (int)cereal::CarParams::SafetyModels::CADILLAC:
    safety_setting = SAFETY_CADILLAC;
    break;
  case (int)cereal::CarParams::SafetyModels::HYUNDAI:
    safety_setting = SAFETY_HYUNDAI;
    break;
  default:
    LOGE("unknown safety model: %d", safety_model);
  }

  pthread_mutex_lock(&usb_lock);

  // set in the mutex to avoid race
  safety_setter_thread_handle = -1;

  libusb_control_transfer(dev_handle, 0x40, 0xdc, safety_setting, safety_param, NULL, 0, TIMEOUT);

  pthread_mutex_unlock(&usb_lock);

  return NULL;
}

// must be called before threads or with mutex
bool usb_connect() {
  int err;
  unsigned char is_pigeon[1] = {0};

  dev_handle = libusb_open_device_with_vid_pid(ctx, 0xbbaa, 0xddcc);
  if (dev_handle == NULL) { goto fail; }

  err = libusb_set_configuration(dev_handle, 1);
  if (err != 0) { goto fail; }

  err = libusb_claim_interface(dev_handle, 0);
  if (err != 0) { goto fail; }

  if (loopback_can) {
    libusb_control_transfer(dev_handle, 0xc0, 0xe5, 1, 0, NULL, 0, TIMEOUT);
  }

  // power off ESP
  libusb_control_transfer(dev_handle, 0xc0, 0xd9, 0, 0, NULL, 0, TIMEOUT);

  // power on charging (may trigger a reconnection, should be okay)
  #ifndef __x86_64__
    libusb_control_transfer(dev_handle, 0xc0, 0xe6, 1, 0, NULL, 0, TIMEOUT);
  #else
    LOGW("not enabling charging on x86_64");
  #endif

  // no output is the default
  if (getenv("RECVMOCK")) {
    libusb_control_transfer(dev_handle, 0x40, 0xdc, SAFETY_ELM327, 0, NULL, 0, TIMEOUT);
  } else {
    libusb_control_transfer(dev_handle, 0x40, 0xdc, SAFETY_NOOUTPUT, 0, NULL, 0, TIMEOUT);
  }

  if (safety_setter_thread_handle == -1) {
    err = pthread_create(&safety_setter_thread_handle, NULL, safety_setter_thread, NULL);
    assert(err == 0);
  }

  libusb_control_transfer(dev_handle, 0xc0, 0xc1, 0, 0, is_pigeon, 1, TIMEOUT);

  if (is_pigeon[0]) {
    LOGW("grey panda detected");
    is_grey_panda = true;
    pigeon_needs_init = true;
    if (pigeon_thread_handle == -1) {
      err = pthread_create(&pigeon_thread_handle, NULL, pigeon_thread, NULL);
      assert(err == 0);
    }
  }

  return true;
fail:
  return false;
}

void usb_retry_connect() {
  LOG("attempting to connect");
  while (!usb_connect()) { usleep(100*1000); }
  LOGW("connected to board");
}

void handle_usb_issue(int err, const char func[]) {
  LOGE_100("usb error %d \"%s\" in %s", err, libusb_strerror((enum libusb_error)err), func);
  if (err == -4) {
    LOGE("lost connection");
    usb_retry_connect();
  }
  // TODO: check other errors, is simply retrying okay?
}

void can_recv(void *s) {
  int err;
  uint32_t data[RECV_SIZE/4];
  int recv;
  uint32_t f1, f2;

  // do recv
  pthread_mutex_lock(&usb_lock);

  do {
    err = libusb_bulk_transfer(dev_handle, 0x81, (uint8_t*)data, RECV_SIZE, &recv, TIMEOUT);
    if (err != 0) { handle_usb_issue(err, __func__); }
    if (err == -8) { LOGE_100("overflow got 0x%x", recv); };

    // timeout is okay to exit, recv still happened
    if (err == -7) { break; }
  } while(err != 0);

  pthread_mutex_unlock(&usb_lock);

  // return if length is 0
  if (recv <= 0) {
    return;
  }

  // create message
  capnp::MallocMessageBuilder msg;
  cereal::Event::Builder event = msg.initRoot<cereal::Event>();
  event.setLogMonoTime(nanos_since_boot());

  auto canData = event.initCan(recv/0x10);

  // populate message
  for (int i = 0; i<(recv/0x10); i++) {
    if (data[i*4] & 4) {
      // extended
      canData[i].setAddress(data[i*4] >> 3);
      //printf("got extended: %x\n", data[i*4] >> 3);
    } else {
      // normal
      canData[i].setAddress(data[i*4] >> 21);
    }
    canData[i].setBusTime(data[i*4+1] >> 16);
    int len = data[i*4+1]&0xF;
    canData[i].setDat(kj::arrayPtr((uint8_t*)&data[i*4+2], len));
    canData[i].setSrc((data[i*4+1] >> 4) & 0xff);
  }

  // send to can
  auto words = capnp::messageToFlatArray(msg);
  auto bytes = words.asBytes();
  zmq_send(s, bytes.begin(), bytes.size(), 0);
}

void can_health(void *s) {
  int cnt;

  // copied from board/main.c
  struct __attribute__((packed)) health {
    uint32_t voltage;
    uint32_t current;
    uint8_t started;
    uint8_t controls_allowed;
    uint8_t gas_interceptor_detected;
    uint8_t started_signal_detected;
    uint8_t started_alt;
  } health;

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
  healthData.setIsGreyPanda(is_grey_panda);

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

  uint32_t *send = (uint32_t*)malloc(msg_count*0x10);
  memset(send, 0, msg_count*0x10);

  for (int i = 0; i < msg_count; i++) {
    auto cmsg = event.getSendcan()[i];
    if (cmsg.getAddress() >= 0x800) {
      // extended
      send[i*4] = (cmsg.getAddress() << 3) | 5;
    } else {
      // normal
      send[i*4] = (cmsg.getAddress() << 21) | 1;
    }
    assert(cmsg.getDat().size() <= 8);
    send[i*4+1] = cmsg.getDat().size() | (cmsg.getSrc() << 4);
    memcpy(&send[i*4+2], cmsg.getDat().begin(), cmsg.getDat().size());
  }

  // release msg
  zmq_msg_close(&msg);

  // send to board
  int sent;
  pthread_mutex_lock(&usb_lock);

  if (!fake_send) {
    do {
      err = libusb_bulk_transfer(dev_handle, 3, (uint8_t*)send, msg_count*0x10, &sent, TIMEOUT);
      if (err != 0 || msg_count*0x10 != sent) { handle_usb_issue(err, __func__); }
    } while(err != 0);
  }

  pthread_mutex_unlock(&usb_lock);

  // done
  free(send);
}


// **** threads ****

void *thermal_thread(void *crap) {
  int err;
  LOGD("start thermal thread");

  // thermal = 8005
  void *context = zmq_ctx_new();
  void *subscriber = zmq_socket(context, ZMQ_SUB);
  zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "", 0);
  zmq_connect(subscriber, "tcp://127.0.0.1:8005");

  // run as fast as messages come in
  while (!do_exit) {
    // recv from thermal
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    err = zmq_msg_recv(&msg, subscriber, 0);
    assert(err >= 0);

    // format for board, make copy due to alignment issues, will be freed on out of scope
    // copied from send thread...
    auto amsg = kj::heapArray<capnp::word>((zmq_msg_size(&msg) / sizeof(capnp::word)) + 1);
    memcpy(amsg.begin(), zmq_msg_data(&msg), zmq_msg_size(&msg));

    capnp::FlatArrayMessageReader cmsg(amsg);
    cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();

    uint16_t target_fan_speed = event.getThermal().getFanSpeed();
    //LOGW("setting fan speed %d", target_fan_speed);

    pthread_mutex_lock(&usb_lock);
    libusb_control_transfer(dev_handle, 0xc0, 0xd3, target_fan_speed, 0, NULL, 0, TIMEOUT);
    pthread_mutex_unlock(&usb_lock);

    zmq_msg_close(&msg);
  }

  // turn the fan off when we exit
  libusb_control_transfer(dev_handle, 0xc0, 0xd3, 0, 0, NULL, 0, TIMEOUT);

  return NULL;
}

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

#define pigeon_send(x) _pigeon_send(x, sizeof(x)-1)

void hexdump(unsigned char *d, int l) {
  for (int i = 0; i < l; i++) {
    if (i!=0 && i%0x10 == 0) printf("\n");
    printf("%2.2X ", d[i]);
  }
  printf("\n");
}

void _pigeon_send(const char *dat, int len) {
  int sent;
  unsigned char a[0x20];
  int err;
  a[0] = 1;
  for (int i=0; i<len; i+=0x20) {
    int ll = std::min(0x20, len-i);
    memcpy(&a[1], &dat[i], ll);
    pthread_mutex_lock(&usb_lock);
    err = libusb_bulk_transfer(dev_handle, 2, a, ll+1, &sent, TIMEOUT);
    if (err < 0) { handle_usb_issue(err, __func__); }
    /*assert(err == 0);
    assert(sent == ll+1);*/
    //hexdump(a, ll+1);
    pthread_mutex_unlock(&usb_lock);
  }
}

void pigeon_set_power(int power) {
  pthread_mutex_lock(&usb_lock);
  int err = libusb_control_transfer(dev_handle, 0xc0, 0xd9, power, 0, NULL, 0, TIMEOUT);
  if (err < 0) { handle_usb_issue(err, __func__); }
  pthread_mutex_unlock(&usb_lock);
}

void pigeon_set_baud(int baud) {
  int err;
  pthread_mutex_lock(&usb_lock);
  err = libusb_control_transfer(dev_handle, 0xc0, 0xe2, 1, 0, NULL, 0, TIMEOUT);
  if (err < 0) { handle_usb_issue(err, __func__); }
  err = libusb_control_transfer(dev_handle, 0xc0, 0xe4, 1, baud/300, NULL, 0, TIMEOUT);
  if (err < 0) { handle_usb_issue(err, __func__); }
  pthread_mutex_unlock(&usb_lock);
}

void pigeon_init() {
  usleep(1000*1000);
  LOGW("grey panda start");

  // power off pigeon
  pigeon_set_power(0);
  usleep(100*1000);

  // 9600 baud at init
  pigeon_set_baud(9600);

  // power on pigeon
  pigeon_set_power(1);
  usleep(500*1000);

  // baud rate upping
  pigeon_send("\x24\x50\x55\x42\x58\x2C\x34\x31\x2C\x31\x2C\x30\x30\x30\x37\x2C\x30\x30\x30\x33\x2C\x34\x36\x30\x38\x30\x30\x2C\x30\x2A\x31\x35\x0D\x0A");
  usleep(100*1000);

  // set baud rate to 460800
  pigeon_set_baud(460800);
  usleep(100*1000);

  // init from ubloxd
  pigeon_send("\xB5\x62\x06\x00\x14\x00\x03\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x1E\x7F");
  pigeon_send("\xB5\x62\x06\x3E\x00\x00\x44\xD2");
  pigeon_send("\xB5\x62\x06\x00\x14\x00\x00\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x19\x35");
  pigeon_send("\xB5\x62\x06\x00\x14\x00\x01\x00\x00\x00\xC0\x08\x00\x00\x00\x08\x07\x00\x01\x00\x01\x00\x00\x00\x00\x00\xF4\x80");
  pigeon_send("\xB5\x62\x06\x00\x14\x00\x04\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1D\x85");
  pigeon_send("\xB5\x62\x06\x00\x00\x00\x06\x18");
  pigeon_send("\xB5\x62\x06\x00\x01\x00\x01\x08\x22");
  pigeon_send("\xB5\x62\x06\x00\x01\x00\x02\x09\x23");
  pigeon_send("\xB5\x62\x06\x00\x01\x00\x03\x0A\x24");
  pigeon_send("\xB5\x62\x06\x08\x06\x00\x64\x00\x01\x00\x00\x00\x79\x10");
  pigeon_send("\xB5\x62\x06\x24\x24\x00\x05\x00\x04\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x5A\x63");
  pigeon_send("\xB5\x62\x06\x1E\x14\x00\x00\x00\x00\x00\x01\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x3C\x37");
  pigeon_send("\xB5\x62\x06\x24\x00\x00\x2A\x84");
  pigeon_send("\xB5\x62\x06\x23\x00\x00\x29\x81");
  pigeon_send("\xB5\x62\x06\x1E\x00\x00\x24\x72");
  pigeon_send("\xB5\x62\x06\x01\x03\x00\x01\x07\x01\x13\x51");
  pigeon_send("\xB5\x62\x06\x01\x03\x00\x02\x15\x01\x22\x70");
  pigeon_send("\xB5\x62\x06\x01\x03\x00\x02\x13\x01\x20\x6C");

  LOGW("grey panda is ready to fly");
}

static void pigeon_publish_raw(void *publisher, unsigned char *dat, int alen) {
  // create message
  capnp::MallocMessageBuilder msg;
  cereal::Event::Builder event = msg.initRoot<cereal::Event>();
  event.setLogMonoTime(nanos_since_boot());
  auto ublox_raw = event.initUbloxRaw(alen);
  memcpy(ublox_raw.begin(), dat, alen);

  // send to ubloxRaw
  auto words = capnp::messageToFlatArray(msg);
  auto bytes = words.asBytes();
  zmq_send(publisher, bytes.begin(), bytes.size(), 0);
}


void *pigeon_thread(void *crap) {
  // ubloxRaw = 8042
  void *context = zmq_ctx_new();
  void *publisher = zmq_socket(context, ZMQ_PUB);
  zmq_bind(publisher, "tcp://*:8042");

  // run at ~100hz
  unsigned char dat[0x1000];
  uint64_t cnt = 0;
  while (!do_exit) {
    if (pigeon_needs_init) {
      pigeon_needs_init = false;
      pigeon_init();
    }
    int alen = 0;
    while (alen < 0xfc0) {
      pthread_mutex_lock(&usb_lock);
      int len = libusb_control_transfer(dev_handle, 0xc0, 0xe0, 1, 0, dat+alen, 0x40, TIMEOUT);
      if (len < 0) { handle_usb_issue(len, __func__); }
      pthread_mutex_unlock(&usb_lock);
      if (len <= 0) break;

      //printf("got %d\n", len);
      alen += len;
    }
    if (alen > 0) {
      if (dat[0] == (char)0x00){
        LOGW("received invalid ublox message, resetting pigeon");
        pigeon_init();
      } else {
        pigeon_publish_raw(publisher, dat, alen);
      }
    }

    // 10ms
    usleep(10*1000);
    cnt++;
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

  if (getenv("BOARDD_LOOPBACK")){
    loopback_can = true;
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

  pthread_t thermal_thread_handle;
  err = pthread_create(&thermal_thread_handle, NULL,
                       thermal_thread, NULL);
  assert(err == 0);

  // join threads

  err = pthread_join(thermal_thread_handle, NULL);
  assert(err == 0);

  err = pthread_join(can_recv_thread_handle, NULL);
  assert(err == 0);

  err = pthread_join(can_send_thread_handle, NULL);
  assert(err == 0);

  err = pthread_join(can_health_thread_handle, NULL);
  assert(err == 0);

  //while (!do_exit) usleep(1000);

  // destruct libusb

  libusb_close(dev_handle);
  libusb_exit(ctx);
}
