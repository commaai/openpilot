/******************************************************************************
 *                       Code generated with sympy 1.4                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_7794183793822969988);
void inv_err_fun(double *nom_x, double *true_x, double *out_7355221333369501264);
void H_mod_fun(double *state, double *out_5744445918609299259);
void f_fun(double *state, double dt, double *out_9039064146508570210);
void F_fun(double *state, double dt, double *out_8072223598153394438);
void h_25(double *state, double *unused, double *out_4065255527585992767);
void H_25(double *state, double *unused, double *out_8165269990197921171);
void h_24(double *state, double *unused, double *out_1976043640039013749);
void H_24(double *state, double *unused, double *out_413298986514378767);
void h_26(double *state, double *unused, double *out_8261513777496151526);
void H_26(double *state, double *unused, double *out_8923028299312733787);
void h_27(double *state, double *unused, double *out_2457131146622328962);
void H_27(double *state, double *unused, double *out_1366834396023443895);
void h_29(double *state, double *unused, double *out_189784329795850816);
void H_29(double *state, double *unused, double *out_5664192392506963901);
void h_28(double *state, double *unused, double *out_5232440403228794693);
void H_28(double *state, double *unused, double *out_1801946368970198955);
#define DIM 8
#define EDIM 8
#define MEDIM 8
typedef void (*Hfun)(double *, double *, double *);

void predict(double *x, double *P, double *Q, double dt);
const static double MAHA_THRESH_25 = 3.841459;
void update_25(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_24 = 5.991465;
void update_24(double *, double *, double *, double *, double *);
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
