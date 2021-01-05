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
int N = 16;

typedef struct {
    double x[N+1];
    double y[N+1];
    double psi[N+1];
    double delta[N+1];
    double rate[N];
    double cost;
} log_t;

void init(double pathCost, double laneCost, double headingCost, double steerRateCost);
void init_weights(double pathCost, double laneCost, double headingCost, double steerRateCost);
int run_mpc(state_t * x0, log_t * solution,
             double l_poly[4], double r_poly[4], double d_poly[4],
             double l_prob, double r_prob, double curvature_factor, double v_ref, double lane_width);
""")

libmpc = ffi.dlopen(libmpc_fn)
