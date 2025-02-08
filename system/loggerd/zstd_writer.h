#pragma once

#include <zstd.h>

#include <string>
#include <vector>

class ZstdFileWriter {
public:
  ZstdFileWriter(const std::string &filename, int compression_level);
  ~ZstdFileWriter();
  void write(void* data, size_t size);

private:
  void finishCompression();

  std::vector<char> output_buffer_;
  ZSTD_CStream *cstream_;
  FILE* file_ = nullptr;
};
