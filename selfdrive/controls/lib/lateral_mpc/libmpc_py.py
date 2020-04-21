import os

from cffi import FFI
from common.ffi_wrapper import suffix

mpc_dir = os.path.dirname(os.path.abspath(__file__))
libmpc_fn = os.path.join(mpc_dir, "libmpc"+suffix())

ffi = FFI()
ffi.cdef("""
typedef struct {
    double x, y, psi, delta, t;
} state_t;

typedef struct {
    double x[21];
    double y[21];
    double psi[21];
    double delta[21];
    double rate[20];
    double cost;
} log_t;

void init(double pathCost, double steerRateCost);
void init_weights(double pathCost, double steerRateCost);
int run_mpc(state_t * x0, log_t * solution,
             double d_poly[4],
             double curvature_factor, double v_ref);
""")

libmpc = ffi.dlopen(libmpc_fn)
