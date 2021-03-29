#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>

#include <iostream>
#include <fstream>
#include <streambuf>
#ifdef QCOM
#include <cutils/properties.h>
#endif

#include "common/swaglog.h"
#include "common/params.h"
#include "common/version.h"
#include "messaging.hpp"
#include "logger.h"


// ***** logging helpers *****

void append_property(const char* key, const char* value, void *cookie) {
  std::vector<std::pair<std::string, std::string> > *properties =
    (std::vector<std::pair<std::string, std::string> > *)cookie;

  properties->push_back(std::make_pair(std::string(key), std::string(value)));
}

int logger_mkpath(char* file_path) {
  assert(file_path && *file_path);
  char* p;
  for (p=strchr(file_path+1, '/'); p; p=strchr(p+1, '/')) {
    *p = '\0';
    if (mkdir(file_path, 0777)==-1) {
      if (errno != EEXIST) {
        *p = '/';
        return -1;
      }
    }
    *p = '/';
  }
  return 0;
}

// ***** log metadata *****
kj::Array<capnp::word> logger_build_init_data() {
  MessageBuilder msg;
  auto init = msg.initEvent().initInitData();

  if (util::file_exists("/EON")) {
    init.setDeviceType(cereal::InitData::DeviceType::NEO);
  } else if (util::file_exists("/TICI")) {
    init.setDeviceType(cereal::InitData::DeviceType::TICI);
  } else {
    init.setDeviceType(cereal::InitData::DeviceType::PC);
  }

  init.setVersion(capnp::Text::Reader(COMMA_VERSION));

  std::ifstream cmdline_stream("/proc/cmdline");
  std::vector<std::string> kernel_args;
  std::string buf;
  while (cmdline_stream >> buf) {
    kernel_args.push_back(buf);
  }

  auto lkernel_args = init.initKernelArgs(kernel_args.size());
  for (int i=0; i<kernel_args.size(); i++) {
    lkernel_args.set(i, kernel_args[i]);
  }

  init.setKernelVersion(util::read_file("/proc/version"));

#ifdef QCOM
  {
    std::vector<std::pair<std::string, std::string> > properties;
    property_list(append_property, (void*)&properties);

    auto lentries = init.initAndroidProperties().initEntries(properties.size());
    for (int i=0; i<properties.size(); i++) {
      auto lentry = lentries[i];
      lentry.setKey(properties[i].first);
      lentry.setValue(properties[i].second);
    }
  }
#endif

  const char* dongle_id = getenv("DONGLE_ID");
  if (dongle_id) {
    init.setDongleId(std::string(dongle_id));
  }
  init.setDirty(!getenv("CLEAN"));

  // log params
  Params params = Params();
  init.setGitCommit(params.get("GitCommit"));
  init.setGitBranch(params.get("GitBranch"));
  init.setGitRemote(params.get("GitRemote"));
  init.setPassive(params.read_db_bool("Passive"));
  {
    std::map<std::string, std::string> params_map;
    params.read_db_all(&params_map);
    auto lparams = init.initParams().initEntries(params_map.size());
    int i = 0;
    for (auto& kv : params_map) {
      auto lentry = lparams[i];
      lentry.setKey(kv.first);
      lentry.setValue(capnp::Data::Reader((const kj::byte*)kv.second.data(), kv.second.size()));
      i++;
    }
  }
  return capnp::messageToFlatArray(msg);
}

std::string logger_get_route_name() {
  char route_name[64] = {'\0'};
  time_t rawtime = time(NULL);
  struct tm timeinfo;
  localtime_r(&rawtime, &timeinfo);
  strftime(route_name, sizeof(route_name), "%Y-%m-%d--%H-%M-%S", &timeinfo);
  return route_name;
}

void log_init_data(LoggerState *s) {
  auto bytes = s->init_data.asBytes();
  logger_log(s, bytes.begin(), bytes.size(), s->has_qlog);
}


static void log_sentinel(LoggerState *s, cereal::Sentinel::SentinelType type) {
  MessageBuilder msg;
  auto sen = msg.initEvent().initSentinel();
  sen.setType(type);
  auto bytes = msg.toBytes();

  logger_log(s, bytes.begin(), bytes.size(), true);
}

// ***** logging functions *****

void logger_init(LoggerState *s, const char* log_name, bool has_qlog) {
  umask(0);

  pthread_mutex_init(&s->lock, NULL);

  s->part = -1;
  s->has_qlog = has_qlog;
  s->route_name = logger_get_route_name();
  snprintf(s->log_name, sizeof(s->log_name), "%s", log_name);
  s->init_data = logger_build_init_data();
}

static LoggerHandle* logger_open(LoggerState *s, const char* root_path) {
  int err;

  LoggerHandle *h = NULL;
  for (int i=0; i<LOGGER_MAX_HANDLES; i++) {
    if (s->handles[i].refcnt == 0) {
      h = &s->handles[i];
      break;
    }
  }
  assert(h);

  snprintf(h->segment_path, sizeof(h->segment_path),
          "%s/%s--%d", root_path, s->route_name.c_str(), s->part);

  snprintf(h->log_path, sizeof(h->log_path), "%s/%s.bz2", h->segment_path, s->log_name);
  snprintf(h->qlog_path, sizeof(h->qlog_path), "%s/qlog.bz2", h->segment_path);
  snprintf(h->lock_path, sizeof(h->lock_path), "%s.lock", h->log_path);

  err = logger_mkpath(h->log_path);
  if (err) return NULL;

  FILE* lock_file = fopen(h->lock_path, "wb");
  if (lock_file == NULL) return NULL;
  fclose(lock_file);

  h->log = std::make_unique<BZFile>(h->log_path);
  if (s->has_qlog) {
    h->q_log = std::make_unique<BZFile>(h->qlog_path);
  }

  pthread_mutex_init(&h->lock, NULL);
  h->refcnt++;
  return h;
}

int logger_next(LoggerState *s, const char* root_path,
                            char* out_segment_path, size_t out_segment_path_len,
                            int* out_part) {
  bool is_start_of_route = !s->cur_handle;
  if (!is_start_of_route) log_sentinel(s, cereal::Sentinel::SentinelType::END_OF_SEGMENT);

  pthread_mutex_lock(&s->lock);
  s->part++;

  LoggerHandle* next_h = logger_open(s, root_path);
  if (!next_h) {
    pthread_mutex_unlock(&s->lock);
    return -1;
  }

  if (s->cur_handle) {
    lh_close(s->cur_handle);
  }
  s->cur_handle = next_h;

  if (out_segment_path) {
    snprintf(out_segment_path, out_segment_path_len, "%s", next_h->segment_path);
  }
  if (out_part) {
    *out_part = s->part;
  }

  pthread_mutex_unlock(&s->lock);

  // write beggining of log metadata
  log_init_data(s);
  log_sentinel(s, is_start_of_route ? cereal::Sentinel::SentinelType::START_OF_ROUTE : cereal::Sentinel::SentinelType::START_OF_SEGMENT);
  return 0;
}

LoggerHandle* logger_get_handle(LoggerState *s) {
  pthread_mutex_lock(&s->lock);
  LoggerHandle* h = s->cur_handle;
  if (h) {
    pthread_mutex_lock(&h->lock);
    h->refcnt++;
    pthread_mutex_unlock(&h->lock);
  }
  pthread_mutex_unlock(&s->lock);
  return h;
}

void logger_log(LoggerState *s, uint8_t* data, size_t data_size, bool in_qlog) {
  pthread_mutex_lock(&s->lock);
  if (s->cur_handle) {
    lh_log(s->cur_handle, data, data_size, in_qlog);
  }
  pthread_mutex_unlock(&s->lock);
}

void logger_close(LoggerState *s) {
  log_sentinel(s, cereal::Sentinel::SentinelType::END_OF_ROUTE);

  pthread_mutex_lock(&s->lock);
  if (s->cur_handle) {
    lh_close(s->cur_handle);
  }
  pthread_mutex_unlock(&s->lock);
}

void lh_log(LoggerHandle* h, uint8_t* data, size_t data_size, bool in_qlog) {
  pthread_mutex_lock(&h->lock);
  assert(h->refcnt > 0);
  h->log->write(data, data_size);
  if (in_qlog && h->q_log) {
    h->q_log->write(data, data_size);
  }
  pthread_mutex_unlock(&h->lock);
}

void lh_close(LoggerHandle* h) {
  pthread_mutex_lock(&h->lock);
  assert(h->refcnt > 0);
  h->refcnt--;
  if (h->refcnt == 0) {
    h->log.reset(nullptr);
    h->q_log.reset(nullptr);
    unlink(h->lock_path);
    pthread_mutex_unlock(&h->lock);
    pthread_mutex_destroy(&h->lock);
    return;
  }
  pthread_mutex_unlock(&h->lock);
}
