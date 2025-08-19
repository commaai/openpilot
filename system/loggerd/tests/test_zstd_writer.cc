#include <zstd.h>

#include <catch2/catch.hpp>
#include <cstring>
#include <vector>

#include "common/util.h"
#include "system/loggerd/logger.h"
#include "system/loggerd/zstd_writer.h"

TEST_CASE("ZstdFileWriter writes and compresses data correctly in loops", "[ZstdFileWriter]") {
  const std::string filename = "test_zstd_file.zst";
  const int iterations = 100;
  const size_t dataSize = 1024;

  std::string totalTestData;

  // Step 1: Write compressed data to file in a loop
  {
    ZstdFileWriter writer(filename, LOG_COMPRESSION_LEVEL);
    // Write various data sizes including edge cases
    std::vector<size_t> testSizes = {dataSize, 1, 0, dataSize * 2};  // Normal, minimal, empty, large
    for (int i = 0; i < iterations; ++i) {
      size_t currentSize = testSizes[i % testSizes.size()];
      std::string testData = util::random_string(currentSize);
      totalTestData.append(testData);

      writer.write((void *)testData.c_str(), testData.size());
    }
  }

  // Step 2: Decompress the file and verify the data
  auto compressedContent = util::read_file(filename);
  REQUIRE(compressedContent.size() > 0);
  REQUIRE(compressedContent.size() < totalTestData.size());
  std::string decompressedData = zstd_decompress(compressedContent);

  // Step 3: Verify that the decompressed data matches the original accumulated data
  REQUIRE(decompressedData.size() == totalTestData.size());
  REQUIRE(std::memcmp(decompressedData.data(), totalTestData.c_str(), totalTestData.size()) == 0);

  // Clean up the test file
  std::remove(filename.c_str());
}
