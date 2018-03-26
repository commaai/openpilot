import os
import sys
import subprocess

from cffi import FFI

mpc_dir = os.path.dirname(os.path.abspath(__file__))
libmpc_fn = os.path.join(mpc_dir, "libcommampc.so")
subprocess.check_call(["make", "-j4"], stdout=sys.stderr, cwd=mpc_dir)

ffi = FFI()
ffi.cdef("""
typedef struct {
    double x, y, psi, delta, t;
} state_t;

typedef struct {
    double x[20];
    double y[20];
    double psi[20];
    double delta[20];
} log_t;

void init(double steer_rate_cost);
int run_mpc(state_t * x0, log_t * solution,
             double l_poly[4], double r_poly[4], double p_poly[4],
             double l_prob, double r_prob, double p_prob, double curvature_factor, double v_ref, double lane_width);
""")

libmpc = ffi.dlopen(libmpc_fn)
