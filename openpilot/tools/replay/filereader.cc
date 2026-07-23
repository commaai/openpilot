#include "tools/replay/filereader.h"

#include <fstream>
#include <string_view>
#include <unistd.h>

#include "common/util.h"
#include "tools/replay/py_downloader.h"

std::string FileReader::read(const std::string &file, std::atomic<bool> *abort) {
  const bool is_remote = (file.find("https://") == 0) || (file.find("http://") == 0);
  if (is_remote) {
    std::string local_path = PyDownloader::download(file, cache_to_local_, abort);
    if (local_path.empty()) return {};
    return util::read_file(local_path);
  }
  char header[3] = {};
  std::ifstream stream(file, std::ios::binary);
  stream.read(header, sizeof(header));
  if (util::ends_with(file, ".bz2") || std::string_view(header, stream.gcount()) == "BZh") {
    std::string local_path = PyDownloader::decompress(file, abort);
    if (local_path.empty()) return {};
    std::string data = util::read_file(local_path);
    unlink(local_path.c_str());
    return data;
  }
  return util::read_file(file);
}
