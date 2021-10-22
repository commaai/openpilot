#include "selfdrive/ui/replay/util.h"

#include <array>
#include <cassert>
#include <iostream>
#include <mutex>
#include <numeric>

#include <bzlib.h>
#include <curl/curl.h>

#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"

struct CURLGlobalInitializer {
  CURLGlobalInitializer() { curl_global_init(CURL_GLOBAL_DEFAULT); }
  ~CURLGlobalInitializer() { curl_global_cleanup(); }
};

struct MultiPartWriter {
  size_t offset;
  size_t end;
  size_t written;
  std::ostream *os;
};

static size_t write_cb(char *data, size_t size, size_t count, void *userp) {
  MultiPartWriter *w = (MultiPartWriter *)userp;
  w->os->seekp(w->offset);
  size_t bytes = size * count;
  w->os->write(data, bytes);
  w->offset += bytes;
  w->written += bytes;
  return bytes;
}

static size_t dumy_write_cb(char *data, size_t size, size_t count, void *userp) { return size * count; }

size_t getRemoteFileSize(const std::string &url) {
  CURL *curl = curl_easy_init();
  if (!curl) return -1;

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, dumy_write_cb);
  curl_easy_setopt(curl, CURLOPT_HEADER, 1);
  curl_easy_setopt(curl, CURLOPT_NOBODY, 1);
  CURLcode res = curl_easy_perform(curl);
  double content_length = -1;
  if (res == CURLE_OK) {
    res = curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD, &content_length);
  } else {
    std::cout << "Download failed: error code: " << res << std::endl;
  }
  curl_easy_cleanup(curl);
  return content_length > 0 ? content_length : 0;
}

std::string formattedDataSize(size_t size) {
  if (size < 1024) {
    return std::to_string(size) + " B";
  } else if (size < 1024 * 1024) {
    return util::string_format("%.2f KB", (float)size / 1024);
  } else {
    return util::string_format("%.2f MB", (float)size / (1024 * 1024));
  }
}

static std::atomic<bool> enable_http_logging = false;

void enableHttpLogging(bool enable) {
  enable_http_logging = enable;
}

bool httpMultiPartDownload(const std::string &url, std::ostream &os, int parts, size_t content_length, std::atomic<bool> *abort) {
  static CURLGlobalInitializer curl_initializer;
  static std::mutex lock;
  static uint64_t total_written = 0, prev_total_written = 0;
  static double last_print_ts = 0;

  os.seekp(content_length - 1);
  os.write("\0", 1);

  CURLM *cm = curl_multi_init();

  std::map<CURL *, MultiPartWriter> writers;
  const int part_size = content_length / parts;
  for (int i = 0; i < parts; ++i) {
    CURL *eh = curl_easy_init();
    writers[eh] = {
        .os = &os,
        .offset = (size_t)(i * part_size),
        .end = i == parts - 1 ? content_length - 1 : (i + 1) * part_size - 1,
    };
    curl_easy_setopt(eh, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(eh, CURLOPT_WRITEDATA, (void *)(&writers[eh]));
    curl_easy_setopt(eh, CURLOPT_URL, url.c_str());
    curl_easy_setopt(eh, CURLOPT_RANGE, util::string_format("%d-%d", writers[eh].offset, writers[eh].end).c_str());
    curl_easy_setopt(eh, CURLOPT_HTTPGET, 1);
    curl_easy_setopt(eh, CURLOPT_NOSIGNAL, 1);
    curl_easy_setopt(eh, CURLOPT_FOLLOWLOCATION, 1);

    curl_multi_add_handle(cm, eh);
  }

  int still_running = 1;
  size_t prev_written = 0;
  while (still_running > 0 && !(abort && *abort)) {
    curl_multi_wait(cm, nullptr, 0, 1000, nullptr);
    curl_multi_perform(cm, &still_running);

    size_t written = std::accumulate(writers.begin(), writers.end(), 0, [=](int v, auto &w) { return v + w.second.written; });
    int cur_written = written - prev_written;
    prev_written = written;

    std::lock_guard lk(lock);
    double ts = millis_since_boot();
    total_written += cur_written;
    if ((ts - last_print_ts) > 2 * 1000) {
      if (enable_http_logging && last_print_ts > 0) {
        size_t average = (total_written - prev_total_written) / ((ts - last_print_ts) / 1000.);
        int progress = std::min<int>(100, 100.0 * (double)written / (double)content_length);

        size_t idx = url.find("?");
        std::cout << "downloading " << (idx == std::string::npos ? url : url.substr(0, idx)) << " - " << progress << "% (" << formattedDataSize(average) << "/s)" << std::endl;
      }
      prev_total_written = total_written;
      last_print_ts = ts;
    }
  }

  CURLMsg *msg;
  int msgs_left = -1;
  int complete = 0;
  while ((msg = curl_multi_info_read(cm, &msgs_left)) && !(abort && *abort)) {
    if (msg->msg == CURLMSG_DONE) {
      if (msg->data.result == CURLE_OK) {
        long res_status = 0;
        curl_easy_getinfo(msg->easy_handle, CURLINFO_RESPONSE_CODE, &res_status);
        if (res_status == 206) {
          complete++;
        } else {
          std::cout << "Download failed: http error code: " << res_status << std::endl;
        }
      } else {
        std::cout << "Download failed: connection failure: " << msg->data.result << std::endl;
      }
    }
  }

  for (auto &[e, w] : writers) {
    curl_multi_remove_handle(cm, e);
    curl_easy_cleanup(e);
  }

  curl_multi_cleanup(cm);
  return complete == parts;
}

std::string decompressBZ2(const std::string &in) {
  bz_stream strm = {};
  int bzerror = BZ2_bzDecompressInit(&strm, 0, 0);
  assert(bzerror == BZ_OK);

  strm.next_in = (char *)in.data();
  strm.avail_in = in.size();
  std::string out(in.size() * 5, '\0');
  do {
    strm.next_out = (char *)(&out[strm.total_out_lo32]);
    strm.avail_out = out.size() - strm.total_out_lo32;
    bzerror = BZ2_bzDecompress(&strm);
    if (bzerror == BZ_OK && strm.avail_in > 0 && strm.avail_out == 0) {
      out.resize(out.size() * 2);
    }
  } while (bzerror == BZ_OK);

  BZ2_bzDecompressEnd(&strm);
  if (bzerror == BZ_STREAM_END) {
    out.resize(strm.total_out_lo32);
    return out;
  }
  return {};
}

void precise_nano_sleep(long sleep_ns) {
  const long estimate_ns = 1 * 1e6;  // 1ms
  struct timespec req = {.tv_nsec = estimate_ns};
  uint64_t start_sleep = nanos_since_boot();
  while (sleep_ns > estimate_ns) {
    nanosleep(&req, nullptr);
    uint64_t end_sleep = nanos_since_boot();
    sleep_ns -= (end_sleep - start_sleep);
    start_sleep = end_sleep;
  }
  // spin wait
  if (sleep_ns > 0) {
    while ((nanos_since_boot() - start_sleep) <= sleep_ns) {
      usleep(0);
    }
  }
}
