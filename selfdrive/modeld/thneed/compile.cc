#include <cstring>
#include <getopt.h>

#include "selfdrive/modeld/runners/snpemodel.h"
#include "selfdrive/modeld/thneed/thneed.h"
#include "system/hardware/hw.h"

#define TEMPORAL_SIZE 512+256
#define DESIRE_LEN 8
#define TRAFFIC_CONVENTION_LEN 2

// TODO: This should probably use SNPE directly.
int main(int argc, char* argv[]) {
  bool run_optimizer = false, save_binaries = false;
  const char *input_file = NULL, *output_file = NULL;
  static struct option long_options[] = {
      {"in",       required_argument, 0,  'i' },
      {"out",      required_argument, 0,  'o' },
      {"binary",   no_argument,       0,  'b' },
      {"optimize", no_argument,       0,  'f' },
      {0,          0,                 0,  0 }
  };
  int long_index = 0, opt = 0;
  while ((opt = getopt_long_only(argc, argv,"", long_options, &long_index)) != -1) {
    switch (opt) {
      case 'i': input_file = optarg; break;
      case 'o': output_file = optarg; break;
      case 'b': save_binaries = true; break;
      case 'f': run_optimizer = true; break;
    }
  }

  // no input?
  if (!input_file) {
    printf("usage: -i <input file> -o <output file> --binary --optimize\n");
    return -1;
  }

  #define OUTPUT_SIZE 0x10000

  float *output = (float*)calloc(OUTPUT_SIZE, sizeof(float));
  SNPEModel mdl(input_file, output, 0, USE_GPU_RUNTIME, true);
  mdl.thneed->run_optimizer = run_optimizer;

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

  // don't save?
  if (!output_file) {
    printf("no output file, exiting\n");
    return 0;
  }

  // save model
  printf("saving %s with binary %d\n", output_file, save_binaries);
  mdl.thneed->save(output_file, save_binaries);

  // test model
  auto thneed = new Thneed(true);
  thneed->record = false;
  thneed->load(output_file);
  thneed->clexec();
  thneed->find_inputs_outputs();

  return 0;
}

