import os

from cffi import FFI
from common.ffi_wrapper import suffix

mpc_dir = os.path.dirname(os.path.abspath(__file__))
libmpc_fn = os.path.join(mpc_dir, "libmpc"+suffix())

ffi = FFI()
# FIXME N HAS TO MATCH GENERATOR.CPP
# FAILS SILENTLY IF MISMATCHED
ffi.cdef("""
typedef struct {
    double x, y, psi, delta, rate;
} state_t;
int N = 15;

typedef struct {
    double x[N+1];
    double y[N+1];
    double psi[N+1];
    double delta[N+1];
    double rate[N];
    double cost;
} log_t;

void init(double pathCost, double headingCost, double yawRateCost, double steerRateCost);
void set_weights(double pathCost, double headingCost, double yawRateCost, double steerRateCost);
int run_mpc(state_t * x0, log_t * solution,
             double v_ego, double target_y[N+1],
             double target_psi[N+1], double target_dpsi[N+1],
             double curvature_factor);
""")

libmpc = ffi.dlopen(libmpc_fn)
