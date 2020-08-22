#include "../thneed.h"
#include "../../runners/snpemodel.h"

#define TEMPORAL_SIZE 512
#define DESIRE_LEN 8
#define TRAFFIC_CONVENTION_LEN 2

void hexdump(uint32_t *d, int len);

int main(int argc, char* argv[]) {
  #define OUTPUT_SIZE 0x10000
  float *output = (float*)calloc(OUTPUT_SIZE, sizeof(float));
  float *golden = (float*)calloc(OUTPUT_SIZE, sizeof(float));
  SNPEModel mdl(argv[1], output, 0, USE_GPU_RUNTIME);

  // cmd line test
  if (argc > 2) {
    for (int i = 2; i < argc; i++) {
      float *buf[5];
      FILE *f = fopen(argv[i], "rb");

      size_t sz;
      for (int j = 0; j < 5; j++) {
        fread(&sz, 1, sizeof(sz), f);
        printf("reading %zu\n", sz);
        buf[j] = (float*)malloc(sz);
        fread(buf[j], 1, sz, f);
      }

      if (sz != 9532) continue;

      mdl.addRecurrent(buf[0], TEMPORAL_SIZE);
      mdl.addTrafficConvention(buf[1], TRAFFIC_CONVENTION_LEN);
      mdl.addDesire(buf[2], DESIRE_LEN);
      mdl.execute(buf[3], 0);

      hexdump((uint32_t*)buf[4], 0x100);
      hexdump((uint32_t*)output, 0x100);

      for (int j = 0; j < sz/4; j++) {
        if (buf[4][j] != output[j]) {
          printf("MISMATCH %d real:%f comp:%f\n", j, buf[4][j], output[j]);
        }
      }
    }

    return 0;
  }

  float state[TEMPORAL_SIZE];
  mdl.addRecurrent(state, TEMPORAL_SIZE);

  float desire[DESIRE_LEN];
  mdl.addDesire(desire, DESIRE_LEN);

  float traffic_convention[TRAFFIC_CONVENTION_LEN];
  mdl.addTrafficConvention(traffic_convention, TRAFFIC_CONVENTION_LEN);

  float *input = (float*)calloc(0x1000000, sizeof(float));;

  // first run
  printf("************** execute 1 **************\n");
  memset(output, 0, OUTPUT_SIZE * sizeof(float));
  mdl.execute(input, 0);
  hexdump((uint32_t *)output, 0x100);
  memcpy(golden, output, OUTPUT_SIZE * sizeof(float));

  // second run
  printf("************** execute 2 **************\n");
  memset(output, 0, OUTPUT_SIZE * sizeof(float));
  Thneed *t = new Thneed();
  t->record = 7;  // debug print with record
  mdl.execute(input, 0);
  t->stop();
  hexdump((uint32_t *)output, 0x100);
  if (memcmp(golden, output, OUTPUT_SIZE * sizeof(float)) != 0) { printf("FAILURE\n"); return -1; }

  // third run
  printf("************** execute 3 **************\n");
  memset(output, 0, OUTPUT_SIZE * sizeof(float));
  t->record = 2;  // debug print w/o record
  float *inputs[4] = {state, traffic_convention, desire, input};
  t->execute(inputs, output, true);
  hexdump((uint32_t *)output, 0x100);
  if (memcmp(golden, output, OUTPUT_SIZE * sizeof(float)) != 0) { printf("FAILURE\n"); return -1; }

  printf("************** execute 4 **************\n");
  while (1) {
    memset(output, 0, OUTPUT_SIZE * sizeof(float));
    //t->record = 2;  // debug print w/o record
    t->execute(inputs, output);
    hexdump((uint32_t *)output, 0x100);
    if (memcmp(golden, output, OUTPUT_SIZE * sizeof(float)) != 0) { printf("FAILURE\n"); return -1; }
    break;
  }

  printf("************** execute done **************\n");
}

