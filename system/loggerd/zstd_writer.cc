
#include "system/loggerd/zstd_writer.h"

#include <cassert>

#include "common/util.h"

// Constructor: Initializes compression stream and opens file
ZstdFileWriter::ZstdFileWriter(const std::string& filename, int compression_level) {
  // Create the compression stream
  cstream_ = ZSTD_createCStream();
  assert(cstream_);

  size_t initResult = ZSTD_initCStream(cstream_, compression_level);
  assert(!ZSTD_isError(initResult));

  output_buffer_.resize(ZSTD_CStreamOutSize());

  file_ = util::safe_fopen(filename.c_str(), "wb");
  assert(file_ != nullptr);
}

// Destructor: Finalizes compression and closes file
ZstdFileWriter::~ZstdFileWriter() {
  finishCompression();

  util::safe_fflush(file_);
  int err = fclose(file_);
  assert(err == 0);

  ZSTD_freeCStream(cstream_);
}

// Compresses and writes data to file
void ZstdFileWriter::write(void* data, size_t size) {
  ZSTD_inBuffer input = {data, size, 0};
  while (input.pos < input.size) {
    ZSTD_outBuffer output = {output_buffer_.data(), output_buffer_.size(), 0};
    size_t remaining = ZSTD_compressStream(cstream_, &output, &input);
    assert(!ZSTD_isError(remaining));
    size_t written = util::safe_fwrite(output_buffer_.data(), 1, output.pos, file_);
    assert(written == output.pos);
  }
}

// Finalizes compression and writes remaining data
void ZstdFileWriter::finishCompression() {
  ZSTD_outBuffer output = {output_buffer_.data(), output_buffer_.size(), 0};
  size_t remaining;
  do {
    remaining = ZSTD_endStream(cstream_, &output);  // Flush remaining data
    assert(!ZSTD_isError(remaining));
    size_t written = util::safe_fwrite(output_buffer_.data(), 1, output.pos, file_);
    assert(written == output.pos);
  } while (remaining > 0);
}
