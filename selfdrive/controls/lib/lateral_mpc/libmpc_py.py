import os

from cffi import FFI

mpc_dir = os.path.dirname(os.path.abspath(__file__))
libmpc_fn = os.path.join(mpc_dir, "libmpc.so")

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

void init(double pathCost, double laneCost, double headingCost, double steerRateCost);
void init_weights(double pathCost, double laneCost, double headingCost, double steerRateCost);
int run_mpc(state_t * x0, log_t * solution,
             double l_poly[4], double r_poly[4], double d_poly[4],
             double l_prob, double r_prob, double curvature_factor, double v_ref, double lane_width);
""")

libmpc = ffi.dlopen(libmpc_fn)
