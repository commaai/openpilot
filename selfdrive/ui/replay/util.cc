#include "selfdrive/ui/replay/util.h"

#include <bzlib.h>
#include <curl/curl.h>
#include <openssl/sha.h>

#include <cstring>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>

#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"

namespace {

static std::atomic<bool> enable_http_logging = false;

struct CURLGlobalInitializer {
  CURLGlobalInitializer() { curl_global_init(CURL_GLOBAL_DEFAULT); }
  ~CURLGlobalInitializer() { curl_global_cleanup(); }
};

template <class T>
struct MultiPartWriter {
  T *buf;
  size_t *total_written;
  size_t offset;
  size_t end;

  size_t write(char *data, size_t size, size_t count) {
    size_t bytes = size * count;
    if ((offset + bytes) > end) return 0;

    if constexpr (std::is_same<T, std::string>::value) {
      memcpy(buf->data() + offset, data, bytes);
    } else if constexpr (std::is_same<T, std::ofstream>::value) {
      buf->seekp(offset);
      buf->write(data, bytes);
    }

    offset += bytes;
    *total_written += bytes;
    return bytes;
  }
};

template <class T>
size_t write_cb(char *data, size_t size, size_t count, void *userp) {
  auto w = (MultiPartWriter<T> *)userp;
  return w->write(data, size, count);
}

size_t dumy_write_cb(char *data, size_t size, size_t count, void *userp) { return size * count; }

std::string formattedDataSize(size_t size) {
  if (size < 1024) {
    return std::to_string(size) + " B";
  } else if (size < 1024 * 1024) {
    return util::string_format("%.2f KB", (float)size / 1024);
  } else {
    return util::string_format("%.2f MB", (float)size / (1024 * 1024));
  }
}

} // namespace

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
    curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD, &content_length);
  } else {
    rError("Download failed: error code: " << res);
  }
  curl_easy_cleanup(curl);
  return content_length > 0 ? (size_t)content_length : 0;
}

std::string getUrlWithoutQuery(const std::string &url) {
  size_t idx = url.find("?");
  return (idx == std::string::npos ? url : url.substr(0, idx));
}

void enableHttpLogging(bool enable) {
  enable_http_logging = enable;
}

template <class T>
bool httpDownload(const std::string &url, T &buf, size_t chunk_size, size_t content_length, std::atomic<bool> *abort) {
  static CURLGlobalInitializer curl_initializer;

  int parts = 1;
  if (chunk_size > 0 && content_length > 10 * 1024 * 1024) {
    parts = std::nearbyint(content_length / (float)chunk_size);
    parts = std::clamp(parts, 1, 5);
  }

  CURLM *cm = curl_multi_init();
  size_t written = 0;
  std::map<CURL *, MultiPartWriter<T>> writers;
  const int part_size = content_length / parts;
  for (int i = 0; i < parts; ++i) {
    CURL *eh = curl_easy_init();
    writers[eh] = {
        .buf = &buf,
        .total_written = &written,
        .offset = (size_t)(i * part_size),
        .end = i == parts - 1 ? content_length : (i + 1) * part_size,
    };
    curl_easy_setopt(eh, CURLOPT_WRITEFUNCTION, write_cb<T>);
    curl_easy_setopt(eh, CURLOPT_WRITEDATA, (void *)(&writers[eh]));
    curl_easy_setopt(eh, CURLOPT_URL, url.c_str());
    curl_easy_setopt(eh, CURLOPT_RANGE, util::string_format("%d-%d", writers[eh].offset, writers[eh].end - 1).c_str());
    curl_easy_setopt(eh, CURLOPT_HTTPGET, 1);
    curl_easy_setopt(eh, CURLOPT_NOSIGNAL, 1);
    curl_easy_setopt(eh, CURLOPT_FOLLOWLOCATION, 1);

    curl_multi_add_handle(cm, eh);
  }

  size_t prev_written = 0;
  double last_print = millis_since_boot();
  int still_running = 1;
  while (still_running > 0 && !(abort && *abort)) {
    curl_multi_wait(cm, nullptr, 0, 1000, nullptr);
    curl_multi_perform(cm, &still_running);

    if (enable_http_logging) {
      if (double ts = millis_since_boot(); (ts - last_print) > 2 * 1000) {
        size_t average = (written - prev_written) / ((ts - last_print) / 1000.);
        int progress = std::min<int>(100, 100.0 * (double)written / (double)content_length);
        rDebug("downloading " << getUrlWithoutQuery(url) << " - " << progress
                  << "% (" << formattedDataSize(average) << "/s)");
        last_print = ts;
        prev_written = written;
      }
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
          rError("Download failed: http error code: " << res_status);
        }
      } else {
        rError("Download failed: connection failure: " << msg->data.result);
      }
    }
  }

  for (const auto &[e, w] : writers) {
    curl_multi_remove_handle(cm, e);
    curl_easy_cleanup(e);
  }
  curl_multi_cleanup(cm);

  return complete == parts;
}

std::string httpGet(const std::string &url, size_t chunk_size, std::atomic<bool> *abort) {
  size_t size = getRemoteFileSize(url);
  if (size == 0) return {};

  std::string result(size, '\0');
  return httpDownload(url, result, chunk_size, size, abort) ? result : "";
}

bool httpDownload(const std::string &url, const std::string &file, size_t chunk_size, std::atomic<bool> *abort) {
  size_t size = getRemoteFileSize(url);
  if (size == 0) return false;

  std::ofstream of(file, std::ios::binary | std::ios::out);
  of.seekp(size - 1).write("\0", 1);
  return httpDownload(url, of, chunk_size, size, abort);
}

std::string decompressBZ2(const std::string &in) {
  return decompressBZ2((std::byte *)in.data(), in.size());
}

std::string decompressBZ2(const std::byte *in, size_t in_size) {
  if (in_size == 0) return {};

  bz_stream strm = {};
  int bzerror = BZ2_bzDecompressInit(&strm, 0, 0);
  assert(bzerror == BZ_OK);

  strm.next_in = (char *)in;
  strm.avail_in = in_size;
  std::string out(in_size * 5, '\0');
  do {
    strm.next_out = (char *)(&out[strm.total_out_lo32]);
    strm.avail_out = out.size() - strm.total_out_lo32;

    const char *prev_write_pos = strm.next_out;
    bzerror = BZ2_bzDecompress(&strm);
    if (bzerror == BZ_OK && prev_write_pos == strm.next_out) {
      // content is corrupt
      bzerror = BZ_STREAM_END;
      rError("decompressBZ2 error : content is corrupt");
      break;
    }

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

std::string sha256(const std::string &str) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256_CTX sha256;
  SHA256_Init(&sha256);
  SHA256_Update(&sha256, str.c_str(), str.size());
  SHA256_Final(hash, &sha256);
  return util::hexdump(hash, SHA256_DIGEST_LENGTH);
}
