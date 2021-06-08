#include <cassert>
#include <cstdio>
#include <cstdlib>
#include "tensorflow/c/c_api.h"

void* read_file(const char* path, size_t* out_len) {
  FILE* f = fopen(path, "r");
  if (!f) {
    return NULL;
  }
  fseek(f, 0, SEEK_END);
  long f_len = ftell(f);
  rewind(f);

  char* buf = (char*)calloc(f_len, 1);
  assert(buf);

  size_t num_read = fread(buf, f_len, 1, f);
  fclose(f);

  if (num_read != 1) {
    free(buf);
    return NULL;
  }

  if (out_len) {
    *out_len = f_len;
  }

  return buf;
}

static void DeallocateBuffer(void* data, size_t) {
  free(data);
}

int main(int argc, char* argv[]) {
  TF_Buffer* buf;
	TF_Graph* graph;
	TF_Status* status;
	char *path = argv[1];

  // load model
  {
    size_t model_size;
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s.pb", path);
    printf("loading model %s\n", tmp);
    uint8_t *model_data = (uint8_t *)read_file(tmp, &model_size);
    buf = TF_NewBuffer();
    buf->data = model_data;
    buf->length = model_size;
    buf->data_deallocator = DeallocateBuffer;
    printf("loaded model of size %d\n", model_size);
  }

  // import graph
  status = TF_NewStatus();
  graph = TF_NewGraph();
  TF_ImportGraphDefOptions *opts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graph, buf, opts, status);
  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteBuffer(buf);
  if (TF_GetCode(status) != TF_OK) {
    printf("FAIL: %s\n", TF_Message(status));
  } else {
    printf("SUCCESS\n");
  }
}
