import os

from cffi import FFI
from common.ffi_wrapper import suffix

mpc_dir = os.path.dirname(os.path.abspath(__file__))
libmpc_fn = os.path.join(mpc_dir, "libmpc"+suffix())

ffi = FFI()
# FIXME THESES SIZES HAVE TO MATCH GENERATOR.CPP
# FAILS SILENTLY IF MISMATCHED
ffi.cdef("""
typedef struct {
    double x, y, psi, dpsi, ddpsi;
} state_t;

typedef struct {
    double x[16];
    double y[16];
    double psi[16];
    double dpsi[16];
    double ddpsi[15];
    double cost;
} log_t;

void init(double pathCost, double yawRateCost, double steerRateCost);
void init_weights(double pathCost, double yawRateCost, double steerRateCost);
int run_mpc(state_t * x0, log_t * solution,
             double d_poly[4], double dpsi_poly[4], double v_ref );
""")

libmpc = ffi.dlopen(libmpc_fn)
