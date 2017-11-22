import os
import subprocess

from cffi import FFI

mpc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
subprocess.check_output(["make", "-j4"], cwd=mpc_dir)


def _get_libmpc(mpc_id):
    libmpc_fn = os.path.join(mpc_dir, "libcommampc%d.so" % mpc_id)

    ffi = FFI()
    ffi.cdef("""
    typedef struct {
    double x_ego, v_ego, a_ego, x_l, v_l, a_l;
    } state_t;


    typedef struct {
    double x_ego[50];
    double v_ego[50];
    double a_ego[50];
    double j_ego[50];
    double x_l[50];
    double v_l[50];
    double a_l[50];
    } log_t;

    void init();
    void init_with_simulation(double v_ego, double x_l, double v_l, double a_l, double l);
    int run_mpc(state_t * x0, log_t * solution,
                double l);
    """)

    return (ffi, ffi.dlopen(libmpc_fn))


mpcs = [_get_libmpc(1), _get_libmpc(2)]


def get_libmpc(mpc_id):
    return mpcs[mpc_id - 1]
