#include <cstring>

#include "selfdrive/modeld/runners/snpemodel.h"
#include "selfdrive/modeld/thneed/thneed.h"
#include "selfdrive/hardware/hw.h"

#define TEMPORAL_SIZE 512
#define DESIRE_LEN 8
#define TRAFFIC_CONVENTION_LEN 2

// TODO: This should probably use SNPE directly.
int main(int argc, char* argv[]) {
  #define OUTPUT_SIZE 0x10000

  float *output = (float*)calloc(OUTPUT_SIZE, sizeof(float));
  SNPEModel mdl(argv[1], output, 0, USE_GPU_RUNTIME, true);

  float state[TEMPORAL_SIZE] = {0};
  float desire[DESIRE_LEN] = {0};
  float traffic_convention[TRAFFIC_CONVENTION_LEN] = {0};
  float *input = (float*)calloc(0x1000000, sizeof(float));
  float *extra = (float*)calloc(0x1000000, sizeof(float));

  mdl.addRecurrent(state, TEMPORAL_SIZE);
  mdl.addDesire(desire, DESIRE_LEN);
  mdl.addTrafficConvention(traffic_convention, TRAFFIC_CONVENTION_LEN);
  mdl.addImage(input, 0);
  mdl.addExtra(extra, 0);

  // first run
  printf("************** execute 1 **************\n");
  memset(output, 0, OUTPUT_SIZE * sizeof(float));
  mdl.execute();

  // save model
  bool save_binaries = (argc > 3) && (strcmp(argv[3], "--binary") == 0);
  mdl.thneed->save(argv[2], save_binaries);

  // test model
  auto thneed = new Thneed(true);
  thneed->record = false;
  thneed->load(argv[2]);
  thneed->clexec();
  thneed->find_inputs_outputs();

  return 0;
}

