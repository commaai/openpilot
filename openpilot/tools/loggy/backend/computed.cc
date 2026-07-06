#include "tools/loggy/backend/computed.h"

#include "tools/loggy/backend/route.h"

#include "json11/json11.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <sys/wait.h>
#include <unordered_set>
#include <utility>
#include <unistd.h>

namespace loggy {
namespace {

namespace fs = std::filesystem;

struct CommandResult {
  int exit_code = 0;
  std::string output;
};

std::string slug_text(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  bool last_dash = false;
  for (unsigned char c : text) {
    if (std::isalnum(c)) {
      out.push_back(static_cast<char>(std::tolower(c)));
      last_dash = false;
    } else if (!last_dash && !out.empty()) {
      out.push_back('-');
      last_dash = true;
    }
  }
  while (!out.empty() && out.back() == '-') out.pop_back();
  return out.empty() ? "series" : out;
}

std::string hash_suffix(std::string_view text) {
  std::ostringstream out;
  out << std::hex << std::nouppercase << std::setfill('0') << std::setw(12)
      << (std::hash<std::string_view>{}(text) & 0xffffffffffffULL);
  return out.str();
}

void set_status(ComputedSeriesStatus *status,
                const ComputedSeriesSpec &spec,
                bool ok,
                std::string message,
                size_t input_points,
                size_t output_points) {
  if (status == nullptr) return;
  status->output_path = spec.output_path;
  status->ok = ok;
  status->message = std::move(message);
  status->input_points = input_points;
  status->output_points = output_points;
}

std::string shell_quote(const std::string &value) {
  std::string quoted;
  quoted.reserve(value.size() + 8);
  quoted.push_back('\'');
  for (char c : value) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(c);
    }
  }
  quoted.push_back('\'');
  return quoted;
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
  std::array<char, 4096> buf{};
  while (fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
    result.output += buf.data();
  }
  const int status = pclose(pipe);
  result.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
  return result;
}

void write_binary_vector(const fs::path &path, const std::vector<double> &values) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) throw std::runtime_error("failed to open " + path.string());
  if (!values.empty()) {
    out.write(reinterpret_cast<const char *>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(double)));
  }
  if (!out.good()) throw std::runtime_error("failed to write " + path.string());
}

std::vector<double> read_binary_vector(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) throw std::runtime_error("failed to open " + path.string());
  std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (raw.size() % sizeof(double) != 0) throw std::runtime_error("invalid binary vector " + path.string());
  std::vector<double> values(raw.size() / sizeof(double));
  if (!values.empty()) std::memcpy(values.data(), raw.data(), raw.size());
  return values;
}

void write_text_file(const fs::path &path, std::string_view text) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) throw std::runtime_error("failed to open " + path.string());
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!out.good()) throw std::runtime_error("failed to write " + path.string());
}

fs::path create_computed_temp_dir() {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path dir = fs::temp_directory_path() /
                       ("loggy_math_" + std::to_string(::getpid()) + "_" + std::to_string(stamp));
  fs::create_directories(dir);
  return dir;
}

std::string trim_process_output(std::string text) {
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) text.pop_back();
  size_t start_ = 0;
  while (start_ < text.size() && std::isspace(static_cast<unsigned char>(text[start_]))) ++start_;
  return start_ == 0 ? text : text.substr(start_);
}

std::vector<SeriesPoint> make_points(std::vector<double> xs, std::vector<double> ys) {
  const size_t count = std::min(xs.size(), ys.size());
  std::vector<SeriesPoint> points;
  points.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(xs[i]) || !std::isfinite(ys[i])) continue;
    points.push_back({xs[i], ys[i]});
  }
  return points;
}

TimeRange point_range(const std::vector<SeriesPoint> &points) {
  if (points.empty()) return {};
  auto [lo, hi] = std::minmax_element(points.begin(), points.end(), [](const SeriesPoint &a, const SeriesPoint &b) {
    return a.t < b.t;
  });
  return {lo->t, hi->t};
}

void append_paths_from_code(std::string_view code, std::vector<std::string> *paths) {
  if (paths == nullptr || code.empty()) return;
  static const std::regex kPathRegex(R"([tv]\(\s*["']([^"']+)["']\s*\))");
  const std::string owned(code);
  for (std::sregex_iterator it(owned.begin(), owned.end(), kPathRegex), end; it != end; ++it) {
    const std::string path = (*it)[1].str();
    if (!path.empty() && std::find(paths->begin(), paths->end(), path) == paths->end()) paths->push_back(path);
  }
}

}  // namespace

std::string computed_output_path(std::string_view source_path,
                                 std::string_view label,
                                 std::string_view operation) {
  const std::string base = slug_text(label.empty() ? source_path : label);
  const std::string identity = std::string(source_path) + "|" + std::string(label) + "|" + std::string(operation);
  return "/computed/" + base + "-" + hash_suffix(identity);
}

std::vector<std::string> computed_dependencies(const ComputedSeriesSpec &spec) {
  std::vector<std::string> out;
  auto add = [&](const std::string &path) {
    if (!path.empty() && std::find(out.begin(), out.end(), path) == out.end()) out.push_back(path);
  };
  if (spec.kind == ComputedSeriesKind::CustomPython) {
    add(spec.python_linked_source);
    for (const std::string &path : spec.python_additional_sources) add(path);
    append_paths_from_code(spec.python_globals_code, &out);
    append_paths_from_code(spec.python_function_code, &out);
  } else {
    add(spec.source_path);
  }
  return out;
}

bool computed_spec_references_path(const ComputedSeriesSpec &spec, std::string_view path) {
  if (path.empty()) return false;
  const std::vector<std::string> dependencies = computed_dependencies(spec);
  return std::find(dependencies.begin(), dependencies.end(), path) != dependencies.end();
}

bool computed_spec_needs_recompute(const ComputedSeriesSpec &spec,
                                   const std::vector<std::string> &touched_paths) {
  if (touched_paths.empty()) return false;
  for (const std::string &path : touched_paths) {
    if (computed_spec_references_path(spec, path)) return true;
  }
  return false;
}

SeriesChunk materialize_computed_transform(const SeriesView &source,
                                           const ComputedSeriesSpec &spec,
                                           ComputedSeriesStatus *status) {
  SeriesChunk chunk;
  chunk.path = spec.output_path;
  chunk.segment = -1;

  if (spec.kind != ComputedSeriesKind::Transform) {
    set_status(status, spec, false, "unsupported computed kind", source.points.size(), 0);
    return chunk;
  }
  if (spec.output_path.rfind("/computed/", 0) != 0) {
    set_status(status, spec, false, "computed output path must start_ with /computed/", source.points.size(), 0);
    return chunk;
  }
  if (source.points.empty()) {
    set_status(status, spec, false, "missing source series", 0, 0);
    return chunk;
  }

  if (spec.transform == ComputedTransformKind::Derivative) {
    chunk.points.reserve(source.points.size() > 0 ? source.points.size() - 1 : 0);
    for (size_t i = 1; i < source.points.size(); ++i) {
      const SeriesPoint &prev = source.points[i - 1];
      const SeriesPoint &cur = source.points[i];
      const double dt = spec.derivative_dt > 0.0 ? spec.derivative_dt : (cur.t - prev.t);
      if (dt <= 0.0 || !std::isfinite(dt)) continue;
      const double value = ((cur.value - prev.value) / dt) * spec.scale + spec.offset;
      if (!std::isfinite(value)) continue;
      chunk.points.push_back({cur.t, value});
    }
  } else {
    chunk.points.reserve(source.points.size());
    for (const SeriesPoint &point : source.points) {
      const double value = point.value * spec.scale + spec.offset;
      if (!std::isfinite(value)) continue;
      chunk.points.push_back({point.t, value});
    }
  }

  if (!chunk.points.empty()) {
    chunk.range = {chunk.points.front().t, chunk.points.back().t};
  }
  set_status(status, spec, !chunk.points.empty(), chunk.points.empty() ? "no computed points" : "ok",
             source.points.size(), chunk.points.size());
  return chunk;
}

SeriesChunk materialize_computed_python(const Store &store,
                                        TimeRange range,
                                        const ComputedSeriesSpec &spec,
                                        ComputedSeriesStatus *status) {
  SeriesChunk chunk;
  chunk.path = spec.output_path;
  chunk.segment = -1;

  if (spec.output_path.rfind("/computed/", 0) != 0) {
    set_status(status, spec, false, "computed output path must start_ with /computed/", 0, 0);
    return chunk;
  }

  const std::vector<std::string> dependencies = computed_dependencies(spec);
  if (dependencies.empty()) {
    set_status(status, spec, false, "no input series referenced", 0, 0);
    return chunk;
  }

  const fs::path temp_dir = create_computed_temp_dir();
  size_t input_points = 0;
  try {
    const fs::path globals_path = temp_dir / "globals.py";
    const fs::path code_path = temp_dir / "code.py";
    const fs::path manifest_path = temp_dir / "manifest.json";
    const fs::path out_t_path = temp_dir / "result.t.bin";
    const fs::path out_v_path = temp_dir / "result.v.bin";

    write_text_file(globals_path, spec.python_globals_code);
    write_text_file(code_path, spec.python_function_code.empty() ? "return value" : spec.python_function_code);

    json11::Json::array paths_json;
    for (const std::string &path : store.series_paths()) paths_json.push_back(path);
    json11::Json::array additional_json;
    for (const std::string &path : spec.python_additional_sources) additional_json.push_back(path);

    json11::Json::array series_json;
    size_t series_index = 0;
    for (const std::string &path : dependencies) {
      const SeriesView view = store.series_full(path, range);
      if (view.points.size() < 2) throw std::runtime_error("Missing route series " + path);

      std::vector<double> times;
      std::vector<double> values;
      times.reserve(view.points.size());
      values.reserve(view.points.size());
      for (const SeriesPoint &point : view.points) {
        times.push_back(point.t);
        values.push_back(point.value);
      }
      input_points += view.points.size();

      const std::string prefix = "series_" + std::to_string(series_index++);
      const fs::path time_path = temp_dir / (prefix + ".t.bin");
      const fs::path value_path = temp_dir / (prefix + ".v.bin");
      write_binary_vector(time_path, times);
      write_binary_vector(value_path, values);
      series_json.push_back(json11::Json::object{
        {"path", path},
        {"t", time_path.string()},
        {"v", value_path.string()},
      });
    }

    const json11::Json manifest_json = json11::Json::object{
      {"paths", std::move(paths_json)},
      {"linked_source", spec.python_linked_source},
      {"additional_sources", std::move(additional_json)},
      {"series", std::move(series_json)},
    };
    write_text_file(manifest_path, manifest_json.dump());

    const fs::path script = loggy_repo_root_path() / "openpilot" / "tools" / "loggy" / "backend" / "math_eval.py";
    const CommandResult process = run_process_capture_output({
      "python3",
      script.string(),
      manifest_path.string(),
      globals_path.string(),
      code_path.string(),
      out_t_path.string(),
      out_v_path.string(),
    });
    if (process.exit_code != 0) {
      const std::string error_text = trim_process_output(process.output);
      throw std::runtime_error(error_text.empty() ? "Python evaluation failed" : error_text);
    }

    chunk.points = make_points(read_binary_vector(out_t_path), read_binary_vector(out_v_path));
    if (chunk.points.size() < 2) throw std::runtime_error("Custom series returned invalid output");
    chunk.range = point_range(chunk.points);
    fs::remove_all(temp_dir);
    set_status(status, spec, true, "ok", input_points, chunk.points.size());
    return chunk;
  } catch (const std::exception &err) {
    std::error_code ignore;
    fs::remove_all(temp_dir, ignore);
    chunk.points.clear();
    chunk.range = {};
    set_status(status, spec, false, err.what(), input_points, 0);
    return chunk;
  }
}

StoreBatch materialize_computed_series_batch(const Store &store,
                                             const std::vector<ComputedSeriesSpec> &specs,
                                             TimeRange range,
                                             std::vector<ComputedSeriesStatus> *statuses) {
  StoreBatch batch;
  batch.segment = -1;
  batch.coverage = {range};
  std::unordered_set<std::string> replaced;

  for (const ComputedSeriesSpec &spec : specs) {
    if (statuses != nullptr) statuses->push_back({.output_path = spec.output_path});
    ComputedSeriesStatus *status = statuses == nullptr ? nullptr : &statuses->back();
    if (spec.kind == ComputedSeriesKind::CustomPython) {
      SeriesChunk chunk = materialize_computed_python(store, range, spec, status);
      if (chunk.path.empty() || chunk.points.empty()) continue;
      if (replaced.insert(chunk.path).second) batch.replace_series_paths.push_back(chunk.path);
      batch.series.push_back(std::move(chunk));
      continue;
    }

    const SeriesView source = store.series_full(spec.source_path, range);
    SeriesChunk chunk = materialize_computed_transform(source, spec, status);
    if (chunk.path.empty() || chunk.points.empty()) continue;
    if (replaced.insert(chunk.path).second) batch.replace_series_paths.push_back(chunk.path);
    batch.series.push_back(std::move(chunk));
  }

  return batch;
}

}  // namespace loggy
