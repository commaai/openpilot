/******************************************************************************
 *                       Code generated with sympy 1.4                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_7280050432103966623);
void inv_err_fun(double *nom_x, double *true_x, double *out_3047360746371905783);
void H_mod_fun(double *state, double *out_6308920866961272604);
void f_fun(double *state, double dt, double *out_3308595574725787248);
void F_fun(double *state, double dt, double *out_4160370806524182836);
void h_25(double *state, double *unused, double *out_8731921353915695165);
void H_25(double *state, double *unused, double *out_3358246505016874748);
void h_24(double *state, double *unused, double *out_291084043109330954);
void H_24(double *state, double *unused, double *out_8184636798247949712);
void h_30(double *state, double *unused, double *out_303723421636649087);
void H_30(double *state, double *unused, double *out_6693907007073320002);
void h_26(double *state, double *unused, double *out_6620125193237786428);
void H_26(double *state, double *unused, double *out_5876036152884916164);
void h_27(double *state, double *unused, double *out_1541777291655224576);
void H_27(double *state, double *unused, double *out_7363913394360118048);
void h_29(double *state, double *unused, double *out_1735585773123112638);
void H_29(double *state, double *unused, double *out_6594505521491184084);
void h_28(double *state, double *unused, double *out_5837464036911225666);
void H_28(double *state, double *unused, double *out_1866410131138996922);
#define DIM 8
#define EDIM 8
#define MEDIM 8
typedef void (*Hfun)(double *, double *, double *);

void predict(double *x, double *P, double *Q, double dt);
const static double MAHA_THRESH_25 = 3.841459;
void update_25(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_24 = 5.991465;
void update_24(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_30 = 3.841459;
void update_30(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_26 = 3.841459;
void update_26(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_27 = 3.841459;
void update_27(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_29 = 3.841459;
void update_29(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_28 = 5.991465;
void update_28(double *, double *, double *, double *, double *);
void set_mass(double x);

void set_rotational_inertia(double x);

void set_center_to_front(double x);

void set_center_to_rear(double x);

void set_stiffness_front(double x);

void set_stiffness_rear(double x);
