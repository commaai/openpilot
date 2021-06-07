import os

from cffi import FFI
from common.ffi_wrapper import suffix

mpc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

def _get_libmpc(mpc_id):
    libmpc_fn = os.path.join(mpc_dir, "libmpc%d%s" % (mpc_id, suffix()))

    ffi = FFI()
    ffi.cdef("""
    typedef struct {
    double x_ego, v_ego, a_ego, x_l, v_l, a_l;
    } state_t;
    int N = 20;

    typedef struct {
    double x_ego[N+1];
    double v_ego[N+1];
    double a_ego[N+1];
    double j_ego[N];
    double x_l[N+1];
    double v_l[N+1];
    double t[N+1];
    double cost;
    } log_t;

    void init(double ttcCost, double distanceCost, double accelerationCost, double jerkCost);
    int run_mpc(state_t * x0, log_t * solution,
                double x_l[N+1], double v_l[N+1]);
    """)

    return (ffi, ffi.dlopen(libmpc_fn))

mpcs = [_get_libmpc(1), _get_libmpc(2)]

def get_libmpc(mpc_id):
    return mpcs[mpc_id - 1]
