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

void init(double pathCost, double headingCost, double steerRateCost);
void init_weights(double pathCost, double headingCost, double steerRateCost);
int run_mpc(state_t * x0, log_t * solution,
             double v_poly[4], double curvature_factor, double rotation_radius,
             double target_y[N+1], double target_psi[N+1]);
""")

libmpc = ffi.dlopen(libmpc_fn)
