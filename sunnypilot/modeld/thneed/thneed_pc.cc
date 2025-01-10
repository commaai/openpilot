#include "sunnypilot/modeld/thneed/thneed.h"

#include <cassert>

#include "common/clutil.h"
#include "common/timing.h"

Thneed::Thneed(bool do_clinit, cl_context _context) {
  context = _context;
  if (do_clinit) clinit();
  char *thneed_debug_env = getenv("THNEED_DEBUG");
  debug = (thneed_debug_env != NULL) ? atoi(thneed_debug_env) : 0;
}

void Thneed::execute(float **finputs, float *foutput, bool slow) {
  uint64_t tb, te;
  if (debug >= 1) tb = nanos_since_boot();

  // ****** copy inputs
  copy_inputs(finputs);

  // ****** run commands
  clexec();

  // ****** copy outputs
  copy_output(foutput);

  if (debug >= 1) {
    te = nanos_since_boot();
    printf("model exec in %lu us\n", (te-tb)/1000);
  }
}
