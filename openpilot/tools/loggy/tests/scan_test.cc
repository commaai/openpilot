#include "catch2/catch.hpp"

#include "tools/loggy/backend/csv.h"
#include "tools/loggy/backend/scan.h"
#include "tools/loggy/backend/store.h"

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace {

using loggy::CanEvent;
using loggy::MessageId;
using loggy::Store;
using loggy::StoreBatch;
using loggy::TimeRange;

// Stages one CAN event chunk per id and drains it so it is immediately queryable.
// Store isn't movable/copyable (it owns a mutex), so this fills a caller-owned instance.
void stageEvents(Store *store, std::vector<std::pair<MessageId, std::vector<CanEvent>>> per_id, TimeRange coverage) {
  StoreBatch batch;
  batch.segment = 0;
  batch.coverage = {coverage};
  for (auto &[id, events] : per_id) {
    batch.can_events.push_back({.id = id, .range = coverage, .events = std::move(events), .segment = 0});
  }
  store->stage(std::move(batch));
  store->begin_frame();
}

}  // namespace

// -- (a) scan_find_bits_events: exact mismatch counts, ordering, and row cap. --

TEST_CASE("scan_find_bits_events computes exact mismatch counts and orders best-match first") {
  using loggy::FindBitsEvent;
  using loggy::FindBitsParams;
  using loggy::FindBitsRow;

  const MessageId perfect_match{.source = 1, .address = 0x10};
  const MessageId perfect_mismatch{.source = 1, .address = 0x20};

  std::vector<FindBitsEvent> events;
  // Every bit of 0xFF equals source_value=1: all 8 (byte,bit) rows for this address are 0% mismatch.
  events.push_back({.mono_time = 0.0, .source_value = 1, .id = perfect_match, .data = {0xFF}});
  // Every bit of 0x00 differs from source_value=1: all 8 rows for this address are 100% mismatch.
  events.push_back({.mono_time = 0.0, .source_value = 1, .id = perfect_mismatch, .data = {0x00}});

  FindBitsParams params;
  params.equal = true;
  params.min_msgs = 0;
  params.max_rows = 512;

  const std::vector<FindBitsRow> rows = loggy::scan_find_bits_events(events, params);
  REQUIRE(rows.size() == 16);

  // Best match (lowest mismatch percent) sorts first.
  CHECK(rows.front().address == perfect_match.address);
  CHECK(rows.front().byte_idx == 0);
  CHECK(rows.front().bit_idx == 0);
  CHECK(rows.front().total == 1);
  CHECK(rows.front().mismatches == 0);
  CHECK(rows.front().percent == Approx(0.0f));

  // Worst match sorts last.
  CHECK(rows.back().address == perfect_mismatch.address);
  CHECK(rows.back().byte_idx == 0);
  CHECK(rows.back().bit_idx == 7);
  CHECK(rows.back().total == 1);
  CHECK(rows.back().mismatches == 1);
  CHECK(rows.back().percent == Approx(100.0f));
}

TEST_CASE("scan_find_bits_events caps output at max_rows, keeping the best matches") {
  using loggy::FindBitsEvent;
  using loggy::FindBitsParams;
  using loggy::FindBitsRow;

  const MessageId perfect_match{.source = 1, .address = 0x10};
  const MessageId perfect_mismatch{.source = 1, .address = 0x20};
  std::vector<FindBitsEvent> events;
  events.push_back({.mono_time = 0.0, .source_value = 1, .id = perfect_match, .data = {0xFF}});
  events.push_back({.mono_time = 0.0, .source_value = 1, .id = perfect_mismatch, .data = {0x00}});

  FindBitsParams params;
  params.equal = true;
  params.min_msgs = 0;
  params.max_rows = 3;

  const std::vector<FindBitsRow> rows = loggy::scan_find_bits_events(events, params);
  REQUIRE(rows.size() == 3);
  for (const FindBitsRow &row : rows) {
    CHECK(row.address == perfect_match.address);
    CHECK(row.mismatches == 0);
  }
}

TEST_CASE("scan_find_bits_events drops rows at or below min_msgs") {
  using loggy::FindBitsEvent;
  using loggy::FindBitsParams;

  const MessageId target{.source = 1, .address = 0x30};
  std::vector<FindBitsEvent> events = {
    {.mono_time = 0.0, .source_value = 1, .id = target, .data = {0xFF}},
  };

  FindBitsParams params;
  params.equal = true;
  params.min_msgs = 1;  // requires total > 1, but this row only has total == 1
  params.max_rows = 512;

  CHECK(loggy::scan_find_bits_events(events, params).empty());
}

// -- (b) step_find_bits_job chunking: max_messages=1 must match one unbounded step. --

TEST_CASE("step_find_bits_job chunked by one message matches a single unbounded step") {
  using loggy::FindBitsParams;
  using loggy::FindBitsRow;

  const MessageId source_id{.source = 0, .address = 0x47};
  const MessageId target_a{.source = 0, .address = 0x100};
  const MessageId target_b{.source = 0, .address = 0x101};
  const MessageId target_c{.source = 0, .address = 0x102};
  const TimeRange range{0.0, 10.0};

  std::vector<std::pair<MessageId, std::vector<CanEvent>>> per_id;
  per_id.push_back({source_id, {
    {.mono_time = 1.0, .data = {0x01}},
    {.mono_time = 2.0, .data = {0x00}},
    {.mono_time = 3.0, .data = {0x01}},
  }});
  per_id.push_back({target_a, {
    {.mono_time = 1.0, .data = {0xFF}},
    {.mono_time = 2.0, .data = {0xFF}},
    {.mono_time = 3.0, .data = {0x00}},
  }});
  per_id.push_back({target_b, {
    {.mono_time = 1.5, .data = {0x00}},
    {.mono_time = 3.5, .data = {0xFF}},
  }});
  per_id.push_back({target_c, {
    {.mono_time = 2.5, .data = {0xAA}},
  }});

  Store store;
  stageEvents(&store, per_id, range);

  FindBitsParams params;
  params.range = range;
  params.source_bus = source_id.source;
  params.source_address = source_id.address;
  params.byte_idx = 0;
  params.bit_idx = 7;  // LSB of byte0, matches how the source data above was written (0x01 -> bit set)
  params.find_bus = 0;
  params.equal = true;
  params.min_msgs = 0;
  params.max_rows = 512;

  loggy::FindBitsJob unbounded_job = loggy::make_find_bits_job(store, params);
  REQUIRE(loggy::step_find_bits_job(unbounded_job, std::numeric_limits<size_t>::max()));
  REQUIRE(unbounded_job.done);

  loggy::FindBitsJob chunked_job = loggy::make_find_bits_job(store, params);
  int guard = 0;
  while (!loggy::step_find_bits_job(chunked_job, 1)) {
    REQUIRE(++guard < 1000);
  }
  REQUIRE(chunked_job.done);

  REQUIRE(chunked_job.rows.size() == unbounded_job.rows.size());
  for (size_t i = 0; i < unbounded_job.rows.size(); ++i) {
    const FindBitsRow &a = unbounded_job.rows[i];
    const FindBitsRow &b = chunked_job.rows[i];
    CHECK(a.address == b.address);
    CHECK(a.byte_idx == b.byte_idx);
    CHECK(a.bit_idx == b.bit_idx);
    CHECK(a.total == b.total);
    CHECK(a.mismatches == b.mismatches);
    CHECK(a.percent == b.percent);
  }
  // Sanity: the chunked run actually visited every target message, not zero.
  CHECK(chunked_job.id_index == chunked_job.ids.size());
  CHECK(chunked_job.ids.size() == 3);
}

// -- (c) find_signal_compare_value: table test for every comparator, incl. NaN -> false. --

TEST_CASE("find_signal_compare_value table for every comparator") {
  using loggy::FindSignalCompare;

  const double nan = std::numeric_limits<double>::quiet_NaN();

  CHECK(loggy::find_signal_compare_value(5.0, FindSignalCompare::Any, 5.0));
  CHECK(loggy::find_signal_compare_value(5.0, FindSignalCompare::Any, 999.0));
  CHECK_FALSE(loggy::find_signal_compare_value(nan, FindSignalCompare::Any, 5.0));

  CHECK(loggy::find_signal_compare_value(5.0, FindSignalCompare::Equal, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(5.0, FindSignalCompare::Equal, 5.1));
  CHECK_FALSE(loggy::find_signal_compare_value(nan, FindSignalCompare::Equal, 5.0));

  CHECK(loggy::find_signal_compare_value(5.0, FindSignalCompare::NotEqual, 5.1));
  CHECK_FALSE(loggy::find_signal_compare_value(5.0, FindSignalCompare::NotEqual, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(nan, FindSignalCompare::NotEqual, 5.0));

  CHECK(loggy::find_signal_compare_value(6.0, FindSignalCompare::Greater, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(5.0, FindSignalCompare::Greater, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(nan, FindSignalCompare::Greater, 5.0));

  CHECK(loggy::find_signal_compare_value(5.0, FindSignalCompare::GreaterEqual, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(4.9, FindSignalCompare::GreaterEqual, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(nan, FindSignalCompare::GreaterEqual, 5.0));

  CHECK(loggy::find_signal_compare_value(4.0, FindSignalCompare::Less, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(5.0, FindSignalCompare::Less, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(nan, FindSignalCompare::Less, 5.0));

  CHECK(loggy::find_signal_compare_value(5.0, FindSignalCompare::LessEqual, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(5.1, FindSignalCompare::LessEqual, 5.0));
  CHECK_FALSE(loggy::find_signal_compare_value(nan, FindSignalCompare::LessEqual, 5.0));
}

// -- (d) find_signal job: finds a known planted 8-bit signal, no false positives when widening size range. --

TEST_CASE("find_signal job finds a planted 8-bit signal without false positives across a widened size range") {
  using loggy::FindSignalCompare;
  using loggy::FindSignalParams;

  const MessageId id{.source = 0, .address = 0x100};
  const TimeRange range{0.0, 10.0};
  // byte0 = 0xAB (171): the planted 8-bit unsigned LE value at start_bit 0.
  // byte1 = 0xFF: chosen so any wider/narrower window that spills into it changes the decoded
  // value (never coincidentally re-produces 171), proving there's no false positive.
  std::vector<std::pair<MessageId, std::vector<CanEvent>>> per_id;
  per_id.push_back({id, {{.mono_time = 1.0, .data = {0xAB, 0xFF}}}});
  Store store;
  stageEvents(&store, per_id, range);

  FindSignalParams params;
  params.range = range;
  params.buses = {0};
  params.addresses = {id.address};
  params.min_size = 1;
  params.max_size = 16;  // widened well past the true 8-bit signal
  params.little_endian = true;
  params.is_signed = false;
  params.factor = 1.0;
  params.offset = 0.0;
  params.target_value = 171.0;
  params.compare = FindSignalCompare::Equal;
  params.max_results = 512;

  loggy::FindSignalJob job = loggy::make_find_signal_job(store, params);
  int guard = 0;
  while (!loggy::step_find_signal_job(job, 32)) {
    REQUIRE(++guard < 1000);
  }

  REQUIRE(job.results.size() == 1);
  const auto &result = job.results.front();
  CHECK(result.id == id);
  CHECK(result.sig.start_bit == 0);
  CHECK(result.sig.size == 8);
  CHECK(result.msg_size == 2);
  REQUIRE(result.matches.size() == 1);
  CHECK(result.matches.front().second == Approx(171.0));
}

// -- (e) prepare_history_log_page: page-clamp, truncated flag, missing-signal doesn't crash. --

TEST_CASE("prepare_history_log_page clamps an out-of-range page index") {
  using loggy::HistoryLogParams;

  const MessageId id{.source = 0, .address = 0x200};
  const TimeRange range{0.0, 100.0};
  std::vector<CanEvent> events;
  for (int i = 0; i < 10; ++i) events.push_back({.mono_time = double(i), .data = {uint8_t(i)}});
  Store store;
  stageEvents(&store, {{id, events}}, range);

  HistoryLogParams params;
  params.page_size = 3;
  params.page_index = 99;  // far beyond page_count (4 pages for 10 rows at page_size 3)
  params.max_rows = 1000;

  const loggy::HistoryLogPage page = loggy::prepare_history_log_page(store, id, range, params);
  REQUIRE(page.page_count == 4);
  CHECK(page.page_index == 3);
  CHECK(page.total_rows == 10);
  CHECK(page.rows.size() == 1);  // last page holds the 1 remaining row (10 - 3*3)
  CHECK_FALSE(page.truncated);
}

TEST_CASE("prepare_history_log_page sets truncated when max_rows caps the match set") {
  using loggy::HistoryLogParams;

  const MessageId id{.source = 0, .address = 0x201};
  const TimeRange range{0.0, 100.0};
  std::vector<CanEvent> events;
  for (int i = 0; i < 10; ++i) events.push_back({.mono_time = double(i), .data = {uint8_t(i)}});
  Store store;
  stageEvents(&store, {{id, events}}, range);

  HistoryLogParams params;
  params.page_size = 100;
  params.page_index = 0;
  params.max_rows = 5;

  const loggy::HistoryLogPage page = loggy::prepare_history_log_page(store, id, range, params);
  CHECK(page.total_rows == 5);
  CHECK(page.truncated);
}

TEST_CASE("prepare_history_log_page with an unresolved compare signal filters everything, no crash") {
  using loggy::HistoryLogParams;

  const MessageId id{.source = 0, .address = 0x202};
  const TimeRange range{0.0, 100.0};
  std::vector<CanEvent> events = {
    {.mono_time = 1.0, .data = {0x01, 0x02}},
    {.mono_time = 2.0, .data = {0x03, 0x04}},
  };
  Store store;
  stageEvents(&store, {{id, events}}, range);

  HistoryLogParams params;
  params.compare_enabled = true;
  params.compare_signal = "does_not_exist";
  params.max_rows = 1000;
  params.page_size = 250;

  // msg == nullptr here (no DBC loaded) is the "missing signal" case: get_value can never be
  // called, but the page must still build cleanly instead of crashing.
  const loggy::HistoryLogPage page = loggy::prepare_history_log_page(store, id, range, params, nullptr);
  CHECK(page.total_rows == 0);
  CHECK(page.rows.empty());
  CHECK(page.page_count == 1);
  CHECK(page.page_index == 0);
}
