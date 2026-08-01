#pragma once

#include <string>
#include <vector>

int encode_clip(const std::vector<std::string> &inputs, const std::string &output,
                double start_time, double duration);
