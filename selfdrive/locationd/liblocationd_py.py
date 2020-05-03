import os
from common.basedir import BASEDIR

from cffi import FFI

locationd_dir = os.path.dirname(os.path.abspath(__file__))
liblocationd_fn = os.path.join(locationd_dir, "liblocationd.so")


ffi = FFI()
ffi.cdef("""
void *localizer_init(void);
void localizer_handle_log(void * localizer, const unsigned char * data, size_t len);
double localizer_get_yaw(void * localizer);
double localizer_get_bias(void * localizer);
double localizer_get_t(void * localizer);
void *params_learner_init(size_t len, char * params, double angle_offset, double stiffness_factor, double steer_ratio, double learning_rate);
bool params_learner_update(void * params_learner, double psi, double u, double sa);
double params_learner_get_ao(void * params_learner);
double params_learner_get_slow_ao(void * params_learner);
double params_learner_get_x(void * params_learner);
double params_learner_get_sR(void * params_learner);
double * localizer_get_P(void * localizer);
void localizer_set_P(void * localizer, double * P);
double * localizer_get_state(void * localizer);
void localizer_set_state(void * localizer, double * state);
""")

liblocationd = ffi.dlopen(liblocationd_fn)
