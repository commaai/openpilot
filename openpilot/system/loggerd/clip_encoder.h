#pragma once

#include <string>
#include <vector>

// inputs are consecutive 60-second loggerd HEVC segments; start_time is
// relative to the beginning of the first input.
int encode_clip(const std::vector<std::string> &inputs, const std::string &output,
                double start_time, double duration, int bitrate = 5'000'000,
                int speedup = 1, const std::string &metadata = {});
