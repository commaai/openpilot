import os
import platform
import subprocess

from cffi import FFI

mpc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

if platform.machine() == "x86_64":
  try:
    FFI().dlopen(os.path.join(mpc_dir, "libmpc1.so"))
  except OSError:
    # libmpc1.so is likely built for aarch64. cleaning...
    subprocess.check_call(["make", "clean"], cwd=mpc_dir)

subprocess.check_call(["make", "-j4"], cwd=mpc_dir)

def _get_libmpc(mpc_id):
    libmpc_fn = os.path.join(mpc_dir, "libmpc%d.so" % mpc_id)

    ffi = FFI()
    ffi.cdef("""
    typedef struct {
    double x_ego, v_ego, a_ego, x_l, v_l, a_l;
    } state_t;


    typedef struct {
    double x_ego[21];
    double v_ego[21];
    double a_ego[21];
    double j_ego[20];
    double x_l[21];
    double v_l[21];
    double a_l[21];
    double t[21];
    double cost;
    } log_t;

    void init(double ttcCost, double distanceCost, double accelerationCost, double jerkCost);
    void init_with_simulation(double v_ego, double x_l, double v_l, double a_l, double l);
    int run_mpc(state_t * x0, log_t * solution,
                double l, double a_l_0);
    """)

    return (ffi, ffi.dlopen(libmpc_fn))

mpcs = [_get_libmpc(1), _get_libmpc(2)]

def get_libmpc(mpc_id):
    return mpcs[mpc_id - 1]
