/******************************************************************************
 *                      Code generated with sympy 1.7.1                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_4116456328380618754);
void inv_err_fun(double *nom_x, double *true_x, double *out_6229172458057010821);
void H_mod_fun(double *state, double *out_6663468603065277721);
void f_fun(double *state, double dt, double *out_1108163707767927737);
void F_fun(double *state, double dt, double *out_6783042894754934255);
void h_25(double *state, double *unused, double *out_3432551924314513333);
void H_25(double *state, double *unused, double *out_8585452173030056914);
void h_24(double *state, double *unused, double *out_2216813462697559184);
void H_24(double *state, double *unused, double *out_5393105767067412132);
void h_30(double *state, double *unused, double *out_3707745986599019222);
void H_30(double *state, double *unused, double *out_247392989730311422);
void h_26(double *state, double *unused, double *out_8339956569696683248);
void H_26(double *state, double *unused, double *out_2880859018090847641);
void h_27(double *state, double *unused, double *out_8434225191975358978);
void H_27(double *state, double *unused, double *out_1040188998106313890);
void h_29(double *state, double *unused, double *out_670959935154686537);
void H_29(double *state, double *unused, double *out_3469623509236336294);
void h_28(double *state, double *unused, double *out_101455398566109503);
void H_28(double *state, double *unused, double *out_6035805828405644618);
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
