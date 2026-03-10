#pragma once

#include <atomic>
#include <string>

class FileReader {
public:
  FileReader(bool cache_to_local) : cache_to_local_(cache_to_local) {}
  virtual ~FileReader() {}
  std::string read(const std::string &file, std::atomic<bool> *abort = nullptr);

private:
  bool cache_to_local_;
};
