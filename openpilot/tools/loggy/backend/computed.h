#pragma once

#include "tools/loggy/backend/store.h"

#include <string>
#include <string_view>
#include <vector>

namespace loggy {

enum class ComputedSeriesKind {
  Transform,
  CustomPython,
};

enum class ComputedTransformKind {
  Scale,
  Derivative,
};

struct ComputedPythonSpec {
  std::string linked_source;
  std::vector<std::string> additional_sources;
  std::string globals_code;
  std::string function_code;
};

struct ComputedSeriesSpec {
  std::string output_path;
  std::string label;
  ComputedSeriesKind kind = ComputedSeriesKind::Transform;
  ComputedTransformKind transform = ComputedTransformKind::Scale;
  std::string source_path;
  double derivative_dt = 0.0;
  double scale = 1.0;
  double offset = 0.0;
  ComputedPythonSpec python;
};

struct ComputedSeriesStatus {
  std::string output_path;
  bool ok = false;
  std::string message;
  size_t input_points = 0;
  size_t output_points = 0;
};

std::string computed_output_path(std::string_view source_path,
                                 std::string_view label,
                                 std::string_view operation);
std::vector<std::string> computed_dependencies(const ComputedSeriesSpec &spec);
bool computed_spec_references_path(const ComputedSeriesSpec &spec, std::string_view path);
bool computed_spec_needs_recompute(const ComputedSeriesSpec &spec,
                                   const std::vector<std::string> &touched_paths);
SeriesChunk materialize_computed_transform(const SeriesView &source,
                                           const ComputedSeriesSpec &spec,
                                           ComputedSeriesStatus *status = nullptr);
SeriesChunk materialize_computed_python(const Store &store,
                                        TimeRange range,
                                        const ComputedSeriesSpec &spec,
                                        ComputedSeriesStatus *status = nullptr);
StoreBatch materialize_computed_series_batch(const Store &store,
                                             const std::vector<ComputedSeriesSpec> &specs,
                                             TimeRange range,
                                             std::vector<ComputedSeriesStatus> *statuses = nullptr);

}  // namespace loggy
