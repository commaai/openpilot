#include "tools/cabana/ui/custom_eval.h"

#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <sys/wait.h>

#include "json11/json11.hpp"
#include "tools/cabana/utils/util.h"

namespace fs = std::filesystem;

namespace cabana {
namespace {

void write_binary_vector(const fs::path &path, const std::vector<double> &values) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("Failed to open binary file: " + path.string());
  out.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(double));
}

std::vector<double> read_binary_vector(const fs::path &path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("Failed to open binary file: " + path.string());
  auto size = in.tellg();
  if (size % sizeof(double) != 0) throw std::runtime_error("Invalid binary vector: " + path.string());
  in.seekg(0);
  std::vector<double> values(size / sizeof(double));
  if (!values.empty()) {
    in.read(reinterpret_cast<char *>(values.data()), size);
  }
  return values;
}

void write_text_file(const fs::path &path, const std::string &text) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("Failed to open file: " + path.string());
  out << text;
}

fs::path create_custom_series_temp_dir() {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path dir = fs::temp_directory_path() / ("cabana_math_" + std::to_string(::getpid()) + "_" + std::to_string(stamp));
  fs::create_directories(dir);
  return dir;
}

std::string shell_quote(const std::string &val) {
  std::string out = "'";
  for (char c : val) {
    if (c == '\'') out += "'\\''";
    else out += c;
  }
  out += "'";
  return out;
}

struct CommandResult {
  int exit_code = 0;
  std::string output;
};

CommandResult run_process_capture_output(const std::vector<std::string> &args) {
  std::string command;
  for (const std::string &arg : args) {
    if (!command.empty()) command += ' ';
    command += shell_quote(arg);
  }
  command += " 2>&1";

  FILE *pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) throw std::runtime_error("popen() failed");

  CommandResult result;
  std::array<char, 4096> buf = {};
  while (fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
    result.output += buf.data();
  }
  const int status = pclose(pipe);
  result.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
  return result;
}

}  // namespace

PythonEvalResult evaluateCustomPythonSeries(
    const CustomPythonSeries &spec,
    const std::map<std::string, std::pair<std::vector<double>, std::vector<double>>> &inputs) {
  const fs::path temp_dir = create_custom_series_temp_dir();
  try {
    const fs::path globals_path = temp_dir / "globals.py";
    const fs::path code_path = temp_dir / "code.py";
    const fs::path manifest_path = temp_dir / "manifest.json";
    const fs::path out_t_path = temp_dir / "result.t.bin";
    const fs::path out_v_path = temp_dir / "result.v.bin";

    write_text_file(globals_path, spec.globals_code);
    write_text_file(code_path, spec.function_code);

    json11::Json::array paths_json;
    json11::Json::array series_json;
    size_t idx = 0;
    for (const auto &[path, tv] : inputs) {
      paths_json.push_back(path);
      const std::string prefix = "series_" + std::to_string(idx++);
      const fs::path t_path = temp_dir / (prefix + ".t.bin");
      const fs::path v_path = temp_dir / (prefix + ".v.bin");
      write_binary_vector(t_path, tv.first);
      write_binary_vector(v_path, tv.second);
      series_json.push_back(json11::Json::object{
        {"path", path}, {"t", t_path.string()}, {"v", v_path.string()}
      });
    }

    json11::Json::array additional_json;
    for (const auto &src : spec.additional_sources) additional_json.push_back(src);

    const json11::Json manifest = json11::Json::object{
      {"paths", std::move(paths_json)},
      {"linked_source", spec.linked_source},
      {"additional_sources", std::move(additional_json)},
      {"series", std::move(series_json)}
    };
    write_text_file(manifest_path, manifest.dump());

    fs::path math_eval_path = executableDir() / "utils" / "math_eval.py";
    if (!fs::exists(math_eval_path)) {
      math_eval_path = executableDir() / "openpilot" / "tools" / "cabana" / "utils" / "math_eval.py";
    }

    const CommandResult proc = run_process_capture_output({
      "python3",
      math_eval_path.string(),
      manifest_path.string(),
      globals_path.string(),
      code_path.string(),
      out_t_path.string(),
      out_v_path.string()
    });

    if (proc.exit_code != 0) {
      throw std::runtime_error(proc.output.empty() ? "Python evaluation failed" : proc.output);
    }

    PythonEvalResult result;
    result.xs = read_binary_vector(out_t_path);
    result.ys = read_binary_vector(out_v_path);
    fs::remove_all(temp_dir);
    return result;
  } catch (...) {
    std::error_code ec;
    fs::remove_all(temp_dir, ec);
    throw;
  }
}

}  // namespace cabana
