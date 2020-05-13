#include "thneed.h"
#include "../runners/snpemodel.h"

#define TEMPORAL_SIZE 512
#define DESIRE_LEN 8
#define TRAFFIC_CONVENTION_LEN 2

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
  mdl.execute(input, 0);

  // second run
  printf("************** execute 2 **************\n");
  Thneed *t = new Thneed();
  mdl.execute(input, 0);
  t->stop();

  // third run
  printf("************** execute 3 **************\n");
  t->execute();
}

