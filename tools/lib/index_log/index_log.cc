#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>

#include <unistd.h>
#include <fcntl.h>

#include <sys/stat.h>
#include <sys/mman.h>

#include <kj/io.h>
#include <capnp/serialize.h>

int main(int argc, char** argv) {

  if (argc != 3) {
    printf("usage: %s <log_path> <index_output_path>\n", argv[0]);
    return 1;
  }

  const std::string log_fn = argv[1];
  const std::string index_fn = argv[2];

  int log_fd = open(log_fn.c_str(), O_RDONLY, 0);
  assert(log_fd >= 0);

  off_t log_size = lseek(log_fd, 0, SEEK_END);
  lseek(log_fd, 0, SEEK_SET);

  FILE* index_f = NULL;
  if (index_fn == "-") {
    index_f = stdout;
  } else {
    index_f = fopen(index_fn.c_str(), "wb");
  }
  assert(index_f);

  void* log_data = mmap(NULL, log_size, PROT_READ, MAP_PRIVATE, log_fd, 0);
  assert(log_data);

  auto words = kj::arrayPtr((const capnp::word*)log_data, log_size/sizeof(capnp::word));
  while (words.size() > 0) {
    uint64_t idx = ((uintptr_t)words.begin() - (uintptr_t)log_data);
    // printf("%llu - %ld\n", idx, words.size());
    const char* idx_bytes = (const char*)&idx;
    fwrite(idx_bytes, 8, 1, index_f);
    try {
      capnp::FlatArrayMessageReader reader(words);
      words = kj::arrayPtr(reader.getEnd(), words.end());
    } catch (const kj::Exception& exc) {
      break;
    }
  }

  munmap(log_data, log_size);

  fclose(index_f);

  close(log_fd);

  return 0;
}
