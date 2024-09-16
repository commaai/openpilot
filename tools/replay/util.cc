#include "tools/replay/util.h"

#include <bzlib.h>
#include <curl/curl.h>
#include <openssl/sha.h>

#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <utility>
#include <zstd.h>

#include "common/timing.h"
#include "common/util.h"

ReplayMessageHandler message_handler = nullptr;
void installMessageHandler(ReplayMessageHandler handler) { message_handler = handler; }

void logMessage(ReplyMsgType type, const char *fmt, ...) {
  static std::mutex lock;
  std::lock_guard lk(lock);

  char *msg_buf = nullptr;
  va_list args;
  va_start(args, fmt);
  int ret = vasprintf(&msg_buf, fmt, args);
  va_end(args);
  if (ret <= 0 || !msg_buf) return;

  if (message_handler) {
    message_handler(type, msg_buf);
  } else {
    if (type == ReplyMsgType::Debug) {
      std::cout << "\033[38;5;248m" << msg_buf << "\033[00m" << std::endl;
    } else if (type == ReplyMsgType::Warning) {
      std::cout << "\033[38;5;227m" << msg_buf << "\033[00m" << std::endl;
    } else if (type == ReplyMsgType::Critical) {
      std::cout << "\033[38;5;196m" << msg_buf << "\033[00m" << std::endl;
    } else {
      std::cout << msg_buf << std::endl;
    }
  }

  free(msg_buf);
}

namespace {

struct CURLGlobalInitializer {
  CURLGlobalInitializer() { curl_global_init(CURL_GLOBAL_DEFAULT); }
  ~CURLGlobalInitializer() { curl_global_cleanup(); }
};

static CURLGlobalInitializer curl_initializer;

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

struct DownloadStats {
  void installDownloadProgressHandler(DownloadProgressHandler handler) {
    std::lock_guard lk(lock);
    download_progress_handler = handler;
  }

  void add(const std::string &url, uint64_t total_bytes) {
    std::lock_guard lk(lock);
    items[url] = {0, total_bytes};
  }

  void remove(const std::string &url) {
    std::lock_guard lk(lock);
    items.erase(url);
  }

  void update(const std::string &url, uint64_t downloaded, bool success = true) {
    std::lock_guard lk(lock);
    items[url].first = downloaded;

    auto stat = std::accumulate(items.begin(), items.end(), std::pair<int, int>{}, [=](auto &a, auto &b){
      return std::pair{a.first + b.second.first, a.second + b.second.second};
    });
    double tm = millis_since_boot();
    if (download_progress_handler && ((tm - prev_tm) > 500 || !success || stat.first >= stat.second)) {
      download_progress_handler(stat.first, stat.second, success);
      prev_tm = tm;
    }
  }

  std::mutex lock;
  std::map<std::string, std::pair<uint64_t, uint64_t>> items;
  double prev_tm = 0;
  DownloadProgressHandler download_progress_handler = nullptr;
};

static DownloadStats download_stats;

} // namespace

void installDownloadProgressHandler(DownloadProgressHandler handler) {
  download_stats.installDownloadProgressHandler(handler);
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

size_t getRemoteFileSize(const std::string &url, std::atomic<bool> *abort) {
  CURL *curl = curl_easy_init();
  if (!curl) return -1;

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, dumy_write_cb);
  curl_easy_setopt(curl, CURLOPT_HEADER, 1);
  curl_easy_setopt(curl, CURLOPT_NOBODY, 1);

  CURLM *cm = curl_multi_init();
  curl_multi_add_handle(cm, curl);
  int still_running = 1;
  while (still_running > 0 && !(abort && *abort)) {
    CURLMcode mc = curl_multi_perform(cm, &still_running);
    if (mc != CURLM_OK) break;
    if (still_running > 0) {
      curl_multi_wait(cm, nullptr, 0, 1000, nullptr);
    }
  }

  double content_length = -1;
  curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD, &content_length);
  curl_multi_remove_handle(cm, curl);
  curl_easy_cleanup(curl);
  curl_multi_cleanup(cm);
  return content_length > 0 ? (size_t)content_length : 0;
}

std::string getUrlWithoutQuery(const std::string &url) {
  size_t idx = url.find("?");
  return (idx == std::string::npos ? url : url.substr(0, idx));
}

template <class T>
bool httpDownload(const std::string &url, T &buf, size_t chunk_size, size_t content_length, std::atomic<bool> *abort) {
  download_stats.add(url, content_length);

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

  int still_running = 1;
  size_t prev_written = 0;
  while (still_running > 0 && !(abort && *abort)) {
    CURLMcode mc = curl_multi_perform(cm, &still_running);
    if (mc != CURLM_OK) {
      break;
    }
    if (still_running > 0) {
      curl_multi_wait(cm, nullptr, 0, 1000, nullptr);
    }

    if (((written - prev_written) / (double)content_length) >= 0.01) {
      download_stats.update(url, written);
      prev_written = written;
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
          rWarning("Download failed: http error code: %d", res_status);
        }
      } else {
        rWarning("Download failed: connection failure: %d",  msg->data.result);
      }
    }
  }

  bool success = complete == parts;
  download_stats.update(url, written, success);
  download_stats.remove(url);

  for (const auto &[e, w] : writers) {
    curl_multi_remove_handle(cm, e);
    curl_easy_cleanup(e);
  }
  curl_multi_cleanup(cm);

  return success;
}

std::string httpGet(const std::string &url, size_t chunk_size, std::atomic<bool> *abort) {
  size_t size = getRemoteFileSize(url, abort);
  if (size == 0) return {};

  std::string result(size, '\0');
  return httpDownload(url, result, chunk_size, size, abort) ? result : "";
}

bool httpDownload(const std::string &url, const std::string &file, size_t chunk_size, std::atomic<bool> *abort) {
  size_t size = getRemoteFileSize(url, abort);
  if (size == 0) return false;

  std::ofstream of(file, std::ios::binary | std::ios::out);
  of.seekp(size - 1).write("\0", 1);
  return httpDownload(url, of, chunk_size, size, abort);
}

std::string decompressBZ2(const std::string &in, std::atomic<bool> *abort) {
  return decompressBZ2((std::byte *)in.data(), in.size(), abort);
}

std::string decompressBZ2(const std::byte *in, size_t in_size, std::atomic<bool> *abort) {
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
      rWarning("decompressBZ2 error: content is corrupt");
      break;
    }

    if (bzerror == BZ_OK && strm.avail_in > 0 && strm.avail_out == 0) {
      out.resize(out.size() * 2);
    }
  } while (bzerror == BZ_OK && !(abort && *abort));

  BZ2_bzDecompressEnd(&strm);
  if (bzerror == BZ_STREAM_END && !(abort && *abort)) {
    out.resize(strm.total_out_lo32);
    out.shrink_to_fit();
    return out;
  }
  return {};
}

std::string decompressZST(const std::string &in, std::atomic<bool> *abort) {
  return decompressZST((std::byte *)in.data(), in.size(), abort);
}

std::string decompressZST(const std::byte *in, size_t in_size, std::atomic<bool> *abort) {
  ZSTD_DCtx *dctx = ZSTD_createDCtx();
  assert(dctx != nullptr);

  // Initialize input and output buffers
  ZSTD_inBuffer input = {in, in_size, 0};

  // Estimate and reserve memory for decompressed data
  size_t estimatedDecompressedSize = ZSTD_getFrameContentSize(in, in_size);
  if (estimatedDecompressedSize == ZSTD_CONTENTSIZE_ERROR || estimatedDecompressedSize == ZSTD_CONTENTSIZE_UNKNOWN) {
    estimatedDecompressedSize = in_size * 2;  // Use a fallback size
  }

  std::string decompressedData;
  decompressedData.reserve(estimatedDecompressedSize);

  const size_t bufferSize = ZSTD_DStreamOutSize();  // Recommended output buffer size
  std::string outputBuffer(bufferSize, '\0');

  while (input.pos < input.size && !(abort && *abort)) {
    ZSTD_outBuffer output = {outputBuffer.data(), bufferSize, 0};

    size_t result = ZSTD_decompressStream(dctx, &output, &input);
    if (ZSTD_isError(result)) {
      rWarning("decompressZST error: content is corrupt");
      break;
    }

    decompressedData.append(outputBuffer.data(), output.pos);
  }

  ZSTD_freeDCtx(dctx);
  if (!(abort && *abort)) {
    decompressedData.shrink_to_fit();
    return decompressedData;
  }
  return {};
}

void precise_nano_sleep(int64_t nanoseconds, std::atomic<bool> &should_exit) {
  struct timespec req, rem;
  req.tv_sec = nanoseconds / 1000000000;
  req.tv_nsec = nanoseconds % 1000000000;
  while (!should_exit) {
#ifdef __APPLE__
    int ret = nanosleep(&req, &rem);
    if (ret == 0 || errno != EINTR)
      break;
#else
    int ret = clock_nanosleep(CLOCK_MONOTONIC, 0, &req, &rem);
    if (ret == 0 || ret != EINTR)
      break;
#endif
    // Retry sleep if interrupted by a signal
    req = rem;
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

// MonotonicBuffer

void *MonotonicBuffer::allocate(size_t bytes, size_t alignment) {
  assert(bytes > 0);
  void *p = std::align(alignment, bytes, current_buf, available);
  if (p == nullptr) {
    available = next_buffer_size = std::max(next_buffer_size, bytes);
    current_buf = buffers.emplace_back(std::aligned_alloc(alignment, next_buffer_size));
    next_buffer_size *= growth_factor;
    p = current_buf;
  }

  current_buf = (char *)current_buf + bytes;
  available -= bytes;
  return p;
}

MonotonicBuffer::~MonotonicBuffer() {
  for (auto buf : buffers) {
    free(buf);
  }
}
