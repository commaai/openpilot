#include "tools/replay/util.h"

#include <bzlib.h>
#include <openssl/sha.h>

#include <cassert>
#include <cstdarg>
#include <cstring>
#include <iostream>
#include <mutex>
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

std::string formattedDataSize(size_t size) {
  if (size < 1024) {
    return std::to_string(size) + " B";
  } else if (size < 1024 * 1024) {
    return util::string_format("%.2f KB", (float)size / 1024);
  } else {
    return util::string_format("%.2f MB", (float)size / (1024 * 1024));
  }
}

std::string getUrlWithoutQuery(const std::string &url) {
  size_t idx = url.find("?");
  return (idx == std::string::npos ? url : url.substr(0, idx));
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

void precise_nano_sleep(int64_t nanoseconds, std::atomic<bool> &interrupt_requested) {
  struct timespec req, rem;
  req.tv_sec = nanoseconds / 1000000000;
  req.tv_nsec = nanoseconds % 1000000000;
  while (!interrupt_requested) {
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

std::vector<std::string> split(std::string_view source, char delimiter) {
  std::vector<std::string> fields;
  size_t last = 0;
  for (size_t i = 0; i < source.length(); ++i) {
    if (source[i] == delimiter) {
      fields.emplace_back(source.substr(last, i - last));
      last = i + 1;
    }
  }
  fields.emplace_back(source.substr(last));
  return fields;
}

std::string extractFileName(const std::string &file) {
  size_t queryPos = file.find_first_of("?");
  std::string path = (queryPos != std::string::npos) ? file.substr(0, queryPos) : file;
  size_t lastSlash = path.find_last_of("/\\");
  return (lastSlash != std::string::npos) ? path.substr(lastSlash + 1) : path;
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
