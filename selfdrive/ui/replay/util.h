#pragma once

#include <atomic>
#include <ostream>
#include <string>
#include <vector>

bool httpMultiPartDownload(const std::string &url, const std::string &target_file, int parts, std::atomic<bool> *abort = nullptr);
bool decompressBZ2(std::vector<uint8_t> &dest, const char srcData[], size_t srcSize, size_t outputSizeIncrement = 0x100000U);
