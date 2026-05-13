#include "tools/jotpluggler/util.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <sys/wait.h>

std::string read_file_or_throw(const std::filesystem::path &path) {
  const std::string contents = util::read_file(path.string());
  if (!contents.empty() || std::filesystem::exists(path)) {
    return contents;
  }
  throw std::runtime_error("Failed to read " + path.string());
}

void write_file_or_throw(const std::filesystem::path &path, const void *data, size_t size) {
  ensure_parent_dir(path);
  const std::string path_string = path.string();
  const void *bytes = size == 0 ? static_cast<const void *>("") : data;
  if (util::write_file(path_string.c_str(), bytes, size, O_WRONLY | O_CREAT | O_TRUNC) != 0) {
    throw std::runtime_error("Failed to write " + path_string);
  }
}

void write_file_or_throw(const std::filesystem::path &path, std::string_view contents) {
  write_file_or_throw(path, contents.data(), contents.size());
}

void run_system_or_throw(const std::string &command, std::string_view action) {
  const int ret = std::system(command.c_str());
  if (ret != 0) {
    throw std::runtime_error(util::string_format("%.*s failed with exit code %d",
                                                 static_cast<int>(action.size()), action.data(), ret));
  }
}

CommandResult run_process_capture_output(const std::vector<std::string> &args) {
  std::string command;
  for (const std::string &arg : args) {
    if (!command.empty()) command += ' ';
    command += shell_quote(arg);
  }
  command += " 2>&1";

  FILE *pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    throw std::runtime_error("popen() failed");
  }

  CommandResult result;
  std::array<char, 4096> buf = {};
  while (fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
    result.output += buf.data();
  }

  const int status = pclose(pipe);
  result.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
  return result;
}
