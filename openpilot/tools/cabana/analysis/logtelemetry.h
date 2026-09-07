#pragma once

#include <atomic>

#include "tools/cabana/analysis/telemetry.h"
#include "tools/cabana/ui/threadpool.h"
#include "tools/replay/logreader.h"

namespace cabana {
inline Telemetry extractLogTelemetry(const LogReader &log, const std::atomic<bool> &stopping) {
  Telemetry telemetry_batch;
  // Each worker owns its extraction paths and samples. Contiguous event ranges let
  // us concatenate the results in timestamp order without sorting individual samples.
  const size_t batch_count = std::clamp<size_t>(std::thread::hardware_concurrency(), 1, 4);
  std::vector<Telemetry> batches(batch_count);
  const auto &events = log.events;
  parallelFor(batch_count, [&](size_t begin, size_t end) {
    for (size_t batch = begin; batch < end; ++batch) {
      TelemetryExtractor extractor(batches[batch]);
      for (size_t i = events.size() * batch / batch_count; i < events.size() * (batch + 1) / batch_count; ++i) {
        if (stopping.load(std::memory_order_relaxed)) return;
        if (events[i].which == cereal::Event::Which::CAN || events[i].which == cereal::Event::Which::SENDCAN) continue;
        capnp::FlatArrayMessageReader reader(events[i].data);
        extractor.extract(reader.getRoot<cereal::Event>());
      }
    }
  });
  if (stopping) return {};
  for (auto &batch : batches) {
    for (auto &[path, samples] : batch) {
      auto [it, inserted] = telemetry_batch.try_emplace(path);
      if (inserted) it->second = std::move(samples);
      else it->second.insert(it->second.end(), samples.begin(), samples.end());
    }
  }
  return telemetry_batch;
}

}  // namespace cabana
