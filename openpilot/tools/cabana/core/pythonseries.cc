#include "tools/cabana/core/analysis.h"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <spawn.h>
#include <stdexcept>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#ifndef CABANA_REPO_ROOT
#define CABANA_REPO_ROOT "."
#endif
extern char **environ;

namespace cabana {
namespace {
struct TempDirectory {
  std::filesystem::path path;
  TempDirectory() {
    std::string name = (std::filesystem::temp_directory_path() / "cabana-math-XXXXXX").string();
    auto *created = mkdtemp(name.data());
    if (!created) throw std::runtime_error("Cannot create Python evaluation directory");
    path = created;
  }
  ~TempDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};
void write(const std::filesystem::path &file, const std::string &text) {
  std::ofstream output(file, std::ios::binary);
  output.write(text.data(), text.size());
  if (!output) throw std::runtime_error("Cannot write Python evaluation input");
}
void writeVector(const std::filesystem::path &path, const std::vector<double> &values) {
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(double));
  if (!file) throw std::runtime_error("Cannot write Python series");
}
std::vector<double> readVector(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("Python did not produce a series");
  auto bytes = file.tellg();
  if (bytes < 0 || bytes % sizeof(double)) throw std::runtime_error("Invalid Python series output");
  std::vector<double> result(static_cast<size_t>(bytes) / sizeof(double));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(result.data()), bytes);
  if (!file) throw std::runtime_error("Cannot read Python series output");
  return result;
}
}  // namespace

CerealPoints evaluatePythonSeries(const PythonSeries &spec, const std::map<std::string, CerealPoints> &inputs,
                                 const CerealCancel &cancelled, double timeout_seconds, uint64_t time_origin) {
  if (cancelled && cancelled()) throw std::runtime_error("Python evaluation cancelled");
  if (!std::isfinite(timeout_seconds) || timeout_seconds <= 0) throw std::invalid_argument("Invalid Python timeout");
  std::vector<std::string> sources{spec.linked_source};
  sources.insert(sources.end(), spec.additional_sources.begin(), spec.additional_sources.end());
  uint64_t start = 0;
  for (const auto &path : sources) {
    auto it = inputs.find(path);
    if (it == inputs.end() || it->second.empty()) throw std::runtime_error("Waiting for Python input: /" + path);
    start = std::max(start, it->second.front().mono_time);
  }
  TempDirectory temp;
  json11::Json::array series, additional;
  for (const auto &path : spec.additional_sources) additional.push_back("/" + path);
  size_t index = 0;
  for (const auto &path : sources) {
    auto times_file = temp.path / (std::to_string(index) + ".t");
    auto values_file = temp.path / (std::to_string(index++) + ".v");
    std::vector<double> times, values;
    for (const auto &point : inputs.at(path)) {
      // Crop the reference before evaluation: a stateful function must not see
      // future auxiliary values while processing its initial samples.
      if (path == spec.linked_source && point.mono_time < start) continue;
      if (point.mono_time < time_origin) throw std::invalid_argument("Python input precedes route start");
      times.push_back((point.mono_time - time_origin) * 1e-9);
      values.push_back(point.value);
    }
    if (times.empty()) throw std::runtime_error("Python inputs do not overlap");
    writeVector(times_file, times);
    writeVector(values_file, values);
    series.push_back(json11::Json::object{{"path", "/" + path}, {"t", times_file.string()}, {"v", values_file.string()}});
  }
  write(temp.path / "manifest.json", json11::Json(json11::Json::object{
    {"linked_source", "/" + spec.linked_source}, {"additional_sources", additional}, {"series", series}}).dump());
  write(temp.path / "globals.py", spec.globals_code);
  write(temp.path / "code.py", spec.function_code);
  const char *venv = std::getenv("VIRTUAL_ENV");
  std::string python = venv ? std::string(venv) + "/bin/python" : "python3";
  std::vector<std::string> arguments{python, std::string(CABANA_REPO_ROOT) + "/openpilot/tools/jotpluggler/math_eval.py",
    (temp.path / "manifest.json").string(), (temp.path / "globals.py").string(), (temp.path / "code.py").string(),
    (temp.path / "out.t").string(), (temp.path / "out.v").string()};
  std::vector<char *> argv;
  for (auto &argument : arguments) argv.push_back(argument.data());
  argv.push_back(nullptr);
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, (temp.path / "python.log").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
  posix_spawnattr_t attributes;
  posix_spawnattr_init(&attributes);
  posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
  posix_spawnattr_setpgroup(&attributes, 0);
  pid_t child;
  int error = posix_spawnp(&child, python.c_str(), &actions, &attributes, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  posix_spawnattr_destroy(&attributes);
  if (error) throw std::runtime_error("Cannot start Python evaluator: " + std::to_string(error));
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_seconds);
  int status = 0;
  for (;;) {
    const auto result = waitpid(child, &status, WNOHANG);
    if (result == child) break;
    if (result < 0 && errno != EINTR) {
      kill(-child, SIGKILL);
      while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
      throw std::runtime_error("Cannot wait for Python evaluator");
    }
    if ((cancelled && cancelled()) || std::chrono::steady_clock::now() >= deadline) {
      kill(-child, SIGKILL);
      while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
      throw std::runtime_error(cancelled && cancelled() ? "Python evaluation cancelled" : "Python evaluation timed out");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::ifstream log(temp.path / "python.log");
    std::string message(4096, '\0');
    log.read(message.data(), message.size());
    message.resize(log.gcount());
    throw std::runtime_error("Python evaluation failed:\n" + message);
  }
  auto times = readVector(temp.path / "out.t"), values = readVector(temp.path / "out.v");
  if (times.size() != values.size()) throw std::runtime_error("Python returned mismatched time/value lengths");
  CerealPoints result;
  for (size_t i = 0; i < times.size(); ++i) {
    if (!std::isfinite(times[i]) || times[i] < 0 || times[i] >= static_cast<double>(UINT64_MAX) * 1e-9) {
      throw std::runtime_error("Python returned an invalid timestamp");
    }
    uint64_t time = static_cast<uint64_t>(std::round(times[i] * 1e9));
    if (time > UINT64_MAX - time_origin) throw std::runtime_error("Python timestamp exceeds route clock range");
    time += time_origin;
    if (!result.empty() && time < result.back().mono_time) throw std::runtime_error("Python returned unsorted timestamps");
    result.push_back({time, values[i]});
  }
  return result;
}
}  // namespace cabana
