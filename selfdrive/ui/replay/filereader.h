#pragma once

#include <atomic>
#include <string>

class FileReader {
public:
  FileReader(bool cache_to_local, size_t chunk_size = 0, int retries = 3)
      : cache_to_local_(cache_to_local), chunk_size_(chunk_size), max_retries_(retries) {}
  virtual ~FileReader() {}
  std::string read(const std::string &file, std::atomic<bool> *abort = nullptr);

private:
  std::string download(const std::string &url, std::atomic<bool> *abort);
  size_t chunk_size_;
  int max_retries_;
  bool cache_to_local_;
};

std::string cacheFilePath(const std::string &url);
