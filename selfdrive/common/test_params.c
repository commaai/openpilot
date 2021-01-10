#include "selfdrive/common/params.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* const kUsage = "%s: read|write|read_block params_path key [value]\n";

int main(int argc, const char* argv[]) {
  if (argc < 4) {
    printf(kUsage, argv[0]);
    return 0;
  }

  const char* params_path = argv[2];
  const char* key = argv[3];
  if (strcmp(argv[1], "read") == 0) {
    char* value;
    size_t value_size;
    int result = read_db_value(params_path, key, &value, &value_size);
    if (result >= 0) {
      fprintf(stdout, "Read %zu bytes: ", value_size);
      fwrite(value, 1, value_size, stdout);
      fprintf(stdout, "\n");
      free(value);
    } else {
      fprintf(stderr, "Error reading: %d\n", result);
      return -1;
    }
  } else if (strcmp(argv[1], "write") == 0) {
    if (argc < 5) {
      fprintf(stderr, "Error: write value required\n");
      return 1;
    }

    const char* value = argv[4];
    const size_t value_size = strlen(value);
    int result = write_db_value(params_path, key, value, value_size);
    if (result >= 0) {
      fprintf(stdout, "Wrote %s to %s\n", value, key);
    } else {
      fprintf(stderr, "Error writing: %d\n", result);
      return -1;
    }
  } else if (strcmp(argv[1], "read_block") == 0) {
    char* value;
    size_t value_size;
    read_db_value_blocking(params_path, key, &value, &value_size);
    fprintf(stdout, "Read %zu bytes: ", value_size);
    fwrite(value, 1, value_size, stdout);
    fprintf(stdout, "\n");
    free(value);
  } else {
    printf(kUsage, argv[0]);
    return 1;
  }

  return 0;
}

// BUILD:
// $ gcc -I$HOME/one selfdrive/common/test_params.c selfdrive/common/params.c selfdrive/common/util.cc -o ./test_params
// $ seq 0 100000 | xargs -P20 -I{} ./test_params write /data/params DongleId {} && sleep 0.1 &
// $ while ./test_params read /data/params DongleId; do sleep 0.05; done
