#include "../thneed.h"
#include "../../runners/snpemodel.h"

#define TEMPORAL_SIZE 512
#define DESIRE_LEN 8
#define TRAFFIC_CONVENTION_LEN 2

void hexdump(uint32_t *d, int len);

int main(int argc, char* argv[]) {
  float *output = (float*)calloc(0x10000, sizeof(float));
  SNPEModel mdl(argv[1], output, 0, USE_GPU_RUNTIME);

  float state[TEMPORAL_SIZE];
  mdl.addRecurrent(state, TEMPORAL_SIZE);

  float desire[DESIRE_LEN];
  mdl.addDesire(desire, DESIRE_LEN);

  float traffic_convention[TRAFFIC_CONVENTION_LEN];
  mdl.addTrafficConvention(traffic_convention, TRAFFIC_CONVENTION_LEN);

  float *input = (float*)calloc(0x1000000, sizeof(float));;

  // first run
  printf("************** execute 1 **************\n");
  memset(input, 0xCC, 0x1000000);
  mdl.execute(input, 0);
  hexdump((uint32_t *)output, 0x100);

  // second run
  printf("************** execute 2 **************\n");
  memset(input, 0xBB, 0x1000000);
  Thneed *t = new Thneed();
  t->record = 3;  // debug print with record
  mdl.execute(input, 0);
  t->stop();
  hexdump((uint32_t *)output, 0x100);

  // third run
  printf("************** execute 3 **************\n");
  t->record = 2;  // debug print w/o record
  memset(input, 0xAA, 0x1000000);
  float *inputs[4] = {state, traffic_convention, desire, input};
  t->execute(inputs, output);
  hexdump((uint32_t *)output, 0x100);

  printf("************** execute 4 **************\n");
  //t->record = 2;  // debug print w/o record
  memset(input, 0, 0x1000000);
  t->execute(inputs, output);
  hexdump((uint32_t *)output, 0x100);

  printf("************** execute done **************\n");
}

