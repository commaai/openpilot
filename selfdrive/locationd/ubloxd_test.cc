#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <assert.h>
#include <math.h>
#include <ctime>
#include <chrono>
#include <iostream>

#include "messaging.hpp"
#include "impl_zmq.hpp"

#include "common/params.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "common/util.h"
#include "ublox_msg.h"

using namespace ublox;
extern volatile sig_atomic_t do_exit;

void write_file(std::string fpath, uint8_t *to_write, int length) {
  FILE* f = fopen(fpath.c_str(), "wb");
  if (!f) {
    std::cout << "Open " << fpath << " failed" << std::endl;
    return;
  }
  fwrite(to_write, length, 1, f);
  fclose(f);
}

static size_t len = 0U;
static size_t consumed = 0U;
static uint8_t *data = NULL;
static int save_idx = 0;
static std::string prefix;

Message * poll_ubloxraw_msg(Poller * poller) {
  assert(poller);

  size_t consuming  = min(len - consumed, 128);
  if(consumed < len) {
    // create message
    capnp::MallocMessageBuilder msg_builder;
    cereal::Event::Builder event = msg_builder.initRoot<cereal::Event>();
    event.setLogMonoTime(nanos_since_boot());

    auto ublox_raw = event.initUbloxRaw(consuming);
    memcpy(ublox_raw.begin(), (void *)(data + consumed), consuming);

    auto words = capnp::messageToFlatArray(msg_builder);
    auto bytes = words.asBytes();

    Message * msg = new ZMQMessage();
    msg->init((char*)bytes.begin(), bytes.size());
    consumed += consuming;
    return msg;
  } else {
    do_exit = 1;
    return NULL;
  }
}

int send_gps_event(PubSocket *s, const void *buf, size_t length) {
  assert(s);
  write_file(prefix + "/" + std::to_string(save_idx), (uint8_t *)buf, length);
  save_idx++;
  return length;
}

int main(int argc, char** argv) {
  if(argc < 3) {
    printf("Format: ubloxd_test stream_file_path save_prefix\n");
    return 0;
  }
  // Parse 11360 msgs, generate 9452 events
  data = (uint8_t *)read_file(argv[1], &len);
  if(data == NULL) {
    LOGE("Read file %s failed\n", argv[1]);
    return -1;
  }
  prefix = argv[2];
  ubloxd_main(poll_ubloxraw_msg, send_gps_event);
  free(data);
  printf("Generated %d cereal events\n", save_idx);
  if(save_idx != 9452) {
    printf("Event count error: %d\n", save_idx);
    return -1;
  }
  return 0;
}
