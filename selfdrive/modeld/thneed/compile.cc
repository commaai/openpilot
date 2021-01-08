#include <string.h>
#include "thneed.h"
#include "../runners/snpemodel.h"

#define TEMPORAL_SIZE 512
#define DESIRE_LEN 8
#define TRAFFIC_CONVENTION_LEN 2

int main(int argc, char* argv[]) {
  #define OUTPUT_SIZE 0x10000
  float *output = (float*)calloc(OUTPUT_SIZE, sizeof(float));
  SNPEModel mdl("/data/openpilot/models/supercombo.dlc", output, 0, USE_GPU_RUNTIME);

  float state[TEMPORAL_SIZE] = {0};
  float desire[DESIRE_LEN] = {0};
  float traffic_convention[TRAFFIC_CONVENTION_LEN] = {0};
  float *input = (float*)calloc(0x1000000, sizeof(float));;

  mdl.addRecurrent(state, TEMPORAL_SIZE);
  mdl.addDesire(desire, DESIRE_LEN);
  mdl.addTrafficConvention(traffic_convention, TRAFFIC_CONVENTION_LEN);

  // first run
  printf("************** execute 1 **************\n");
  memset(output, 0, OUTPUT_SIZE * sizeof(float));
  mdl.execute(input, 0);

  // save model
  mdl.thneed->save("/data/openpilot/models/supercombo.thneed");
  return 0;
}

