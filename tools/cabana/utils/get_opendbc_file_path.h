#include <string>
#include <filesystem>
namespace fs = std::filesystem;

const char* get_opendbc_file_path(int dist_from_root) {
  std::string binary_path = fs::canonical("/proc/self/exe").parent_path();
  std::string to_root;
  if (!fs::exists(binary_path + "/" + "opendbc")) {
    for (int i = 0; i < dist_from_root; i++){
      to_root += "../";
    }
  }
  static const std::string opendbc_file_path = binary_path + "/" + to_root + "opendbc";
  return opendbc_file_path.c_str();
}
