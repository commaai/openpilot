import os

from cffi import FFI
from common.ffi_wrapper import suffix

mpc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
libmpc_fn = os.path.join(mpc_dir, "libmpc"+suffix())

ffi = FFI()
ffi.cdef("""

typedef struct {
double x_ego, v_ego, a_ego;
} state_t;
int N = 20;


typedef struct {
double x_ego[N+1];
double v_ego[N+1];
double a_ego[N+1];
double t[N+1];
double j_ego[N];
double cost;
} log_t;


void init(double xCost, double vCost, double aCost, double jerkCost);
void init_with_simulation(double v_ego);
int run_mpc(state_t * x0, log_t * solution,
            double target_x[N+1], double target_v[N+1], double target_a[N+1],
            double min_a, double max_a);
""")

libmpc = ffi.dlopen(libmpc_fn)
