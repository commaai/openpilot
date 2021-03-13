#include <string.h>
#include "thneed.h"
#include "../runners/snpemodel.h"

#define TEMPORAL_SIZE 512
#define DESIRE_LEN 8
#define TRAFFIC_CONVENTION_LEN 2

// TODO: This should probably use SNPE directly.
int main(int argc, char* argv[]) {
  #define OUTPUT_SIZE 0x10000
  std::unique_ptr<float[]> output = std::make_unique<float[]>(OUTPUT_SIZE);
  SNPEModel mdl(argv[1], output.get(), 0, USE_GPU_RUNTIME);

  float state[TEMPORAL_SIZE] = {0};
  float desire[DESIRE_LEN] = {0};
  float traffic_convention[TRAFFIC_CONVENTION_LEN] = {0};
  std::unique_ptr<float[]> input = std::make_unique<float []>(0x1000000);

  mdl.addRecurrent(state, TEMPORAL_SIZE);
  mdl.addDesire(desire, DESIRE_LEN);
  mdl.addTrafficConvention(traffic_convention, TRAFFIC_CONVENTION_LEN);

  // first run
  printf("************** execute 1 **************\n");
  mdl.execute(input.get(), 0);

  // save model
  bool save_binaries = (argc > 3) && (strcmp(argv[3], "--binary") == 0);
  mdl.thneed->save(argv[2], save_binaries);
  return 0;
}
