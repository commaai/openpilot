#include "catch2/catch.hpp"

#include "tools/loggy/backend/computed.h"

#include <utility>

namespace {

void populateSource(loggy::Store *store) {
  loggy::StoreBatch batch;
  batch.segment = 0;
  batch.coverage = {{0.0, 4.0}};
  batch.series.push_back({
    .path = "/carState/vEgo",
    .range = {0.0, 4.0},
    .points = {{0.0, 2.0}, {1.0, 4.0}, {3.0, 10.0}},
    .segment = 0,
  });
  batch.series.push_back({
    .path = "/carState/aEgo",
    .range = {0.0, 4.0},
    .points = {{0.5, 100.0}, {2.0, 200.0}, {3.5, 300.0}},
    .segment = 0,
  });
  store->stage(std::move(batch));
  store->begin_frame();
}

}  // namespace

TEST_CASE("computed transform helpers materialize scale and derivative series") {
  loggy::Store store;
  populateSource(&store);

  const std::string scaled_path = loggy::computed_output_path("/carState/vEgo", "vEgo scaled", "scale");
  CHECK(scaled_path.rfind("/computed/vego-scaled-", 0) == 0);

  loggy::ComputedSeriesSpec scale_spec;
  scale_spec.output_path = scaled_path;
  scale_spec.label = "vEgo scaled";
  scale_spec.source_path = "/carState/vEgo";
  scale_spec.transform = loggy::ComputedTransformKind::Scale;
  scale_spec.scale = 2.0;
  scale_spec.offset = -1.0;

  std::vector<loggy::ComputedSeriesStatus> statuses;
  loggy::StoreBatch computed = loggy::materialize_computed_series_batch(store, {scale_spec}, {0.0, 4.0}, &statuses);
  REQUIRE(statuses.size() == 1);
  CHECK(statuses[0].ok);
  CHECK(statuses[0].input_points == 3);
  CHECK(statuses[0].output_points == 3);
  REQUIRE(computed.replace_series_paths.size() == 1);
  CHECK(computed.replace_series_paths[0] == scaled_path);
  REQUIRE(computed.series.size() == 1);
  CHECK(computed.series[0].points[0].value == 3.0);
  CHECK(computed.series[0].points[2].value == 19.0);

  store.stage(std::move(computed));
  store.begin_frame();
  REQUIRE(store.series_full(scaled_path, {0.0, 4.0}).points.size() == 3);

  scale_spec.scale = 3.0;
  statuses.clear();
  computed = loggy::materialize_computed_series_batch(store, {scale_spec}, {0.0, 4.0}, &statuses);
  store.stage(std::move(computed));
  store.begin_frame();
  const loggy::SeriesView replaced = store.series_full(scaled_path, {0.0, 4.0});
  REQUIRE(replaced.points.size() == 3);
  CHECK(replaced.points[0].value == 5.0);
  CHECK(replaced.points[2].value == 29.0);

  loggy::ComputedSeriesSpec derivative_spec;
  derivative_spec.output_path = loggy::computed_output_path("/carState/vEgo", "vEgo derivative", "derivative");
  derivative_spec.label = "vEgo derivative";
  derivative_spec.source_path = "/carState/vEgo";
  derivative_spec.transform = loggy::ComputedTransformKind::Derivative;
  statuses.clear();
  computed = loggy::materialize_computed_series_batch(store, {derivative_spec}, {0.0, 4.0}, &statuses);
  REQUIRE(statuses.size() == 1);
  CHECK(statuses[0].ok);
  REQUIRE(computed.series.size() == 1);
  REQUIRE(computed.series[0].points.size() == 2);
  CHECK(computed.series[0].points[0].t == 1.0);
  CHECK(computed.series[0].points[0].value == 2.0);
  CHECK(computed.series[0].points[1].t == 3.0);
  CHECK(computed.series[0].points[1].value == 3.0);

  derivative_spec.derivative_dt = 1.0;
  statuses.clear();
  computed = loggy::materialize_computed_series_batch(store, {derivative_spec}, {0.0, 4.0}, &statuses);
  REQUIRE(computed.series.size() == 1);
  REQUIRE(computed.series[0].points.size() == 2);
  CHECK(computed.series[0].points[1].value == 6.0);
}

TEST_CASE("computed custom Python specs materialize through math_eval") {
  loggy::ComputedSeriesSpec spec;
  spec.output_path = "/computed/custom";
  spec.kind = loggy::ComputedSeriesKind::CustomPython;
  spec.python_linked_source = "/carState/vEgo";
  spec.python_additional_sources = {"/carState/aEgo", "/carState/vEgo"};
  spec.python_function_code = "return value * 2";

  const std::vector<std::string> deps = loggy::computed_dependencies(spec);
  REQUIRE(deps.size() == 2);
  CHECK(deps[0] == "/carState/vEgo");
  CHECK(deps[1] == "/carState/aEgo");
  CHECK(loggy::computed_spec_references_path(spec, "/carState/vEgo"));
  CHECK(loggy::computed_spec_references_path(spec, "/carState/aEgo"));
  CHECK_FALSE(loggy::computed_spec_references_path(spec, "/carState/brakePressed"));
  CHECK(loggy::computed_spec_needs_recompute(spec, {"/carState/aEgo"}));
  CHECK(loggy::computed_spec_needs_recompute(spec, {"/carState/vEgo"}));
  CHECK_FALSE(loggy::computed_spec_needs_recompute(spec, {"/carState/brakePressed"}));

  loggy::Store store;
  populateSource(&store);
  std::vector<loggy::ComputedSeriesStatus> statuses;
  loggy::StoreBatch batch = loggy::materialize_computed_series_batch(store, {spec}, {0.0, 4.0}, &statuses);
  REQUIRE(statuses.size() == 1);
  CHECK(statuses[0].ok);
  REQUIRE(batch.series.size() == 1);
  REQUIRE(batch.series[0].points.size() == 3);
  CHECK(batch.series[0].points[0].value == 4.0);
  CHECK(batch.series[0].points[2].value == 20.0);

  spec.output_path = "/computed/custom_tuple";
  spec.python_additional_sources.clear();
  spec.python_function_code = "return (time + 1, value + 5)";
  statuses.clear();
  batch = loggy::materialize_computed_series_batch(store, {spec}, {0.0, 4.0}, &statuses);
  REQUIRE(statuses.size() == 1);
  CHECK(statuses[0].ok);
  REQUIRE(batch.series.size() == 1);
  REQUIRE(batch.series[0].points.size() == 3);
  CHECK(batch.series[0].points[0].t == 1.0);
  CHECK(batch.series[0].points[0].value == 7.0);
  CHECK(batch.series[0].points[2].t == 4.0);
  CHECK(batch.series[0].points[2].value == 15.0);

  spec.output_path = "/computed/custom_additional";
  spec.python_additional_sources = {"/carState/aEgo"};
  spec.python_function_code = "return value + v1";
  statuses.clear();
  batch = loggy::materialize_computed_series_batch(store, {spec}, {0.0, 4.0}, &statuses);
  REQUIRE(statuses.size() == 1);
  CHECK(statuses[0].ok);
  REQUIRE(batch.series.size() == 1);
  REQUIRE(batch.series[0].points.size() == 3);
  CHECK(batch.series[0].points[0].value == 102.0);
  CHECK(batch.series[0].points[1].value == 104.0);
  CHECK(batch.series[0].points[2].value == 210.0);

  spec.output_path = "/computed/custom_referenced";
  spec.python_linked_source.clear();
  spec.python_additional_sources.clear();
  spec.python_function_code = "return (t(\"/carState/aEgo\"), v(\"/carState/aEgo\") - 1)";
  const std::vector<std::string> referenced_deps = loggy::computed_dependencies(spec);
  REQUIRE(referenced_deps.size() == 1);
  CHECK(referenced_deps[0] == "/carState/aEgo");
  CHECK(loggy::computed_spec_references_path(spec, "/carState/aEgo"));
  CHECK(loggy::computed_spec_needs_recompute(spec, {"/carState/aEgo"}));
  statuses.clear();
  batch = loggy::materialize_computed_series_batch(store, {spec}, {0.0, 4.0}, &statuses);
  REQUIRE(statuses.size() == 1);
  CHECK(statuses[0].ok);
  REQUIRE(batch.series.size() == 1);
  REQUIRE(batch.series[0].points.size() == 3);
  CHECK(batch.series[0].points[0].t == 0.5);
  CHECK(batch.series[0].points[0].value == 99.0);

  loggy::ComputedSeriesSpec missing;
  missing.output_path = "/computed/missing";
  missing.kind = loggy::ComputedSeriesKind::CustomPython;
  missing.python_linked_source = "/missing/source";
  missing.python_function_code = "return value";
  statuses.clear();
  loggy::materialize_computed_series_batch(store, {missing}, {0.0, 4.0}, &statuses);
  REQUIRE(statuses.size() == 1);
  CHECK_FALSE(statuses[0].ok);
  CHECK(statuses[0].message == "Missing route series /missing/source");

  loggy::ComputedSeriesSpec bad_output;
  bad_output.output_path = "/not-computed/vEgo";
  bad_output.kind = loggy::ComputedSeriesKind::CustomPython;
  bad_output.python_linked_source = "/carState/vEgo";
  bad_output.python_function_code = "return value";
  statuses.clear();
  loggy::materialize_computed_series_batch(store, {bad_output}, {0.0, 4.0}, &statuses);
  REQUIRE(statuses.size() == 1);
  CHECK_FALSE(statuses[0].ok);
  CHECK(statuses[0].message.find("/computed/") != std::string::npos);
}
