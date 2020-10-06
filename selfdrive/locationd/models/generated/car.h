/******************************************************************************
 *                       Code generated with sympy 1.4                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_2030068162292336593);
void inv_err_fun(double *nom_x, double *true_x, double *out_537171033141946999);
void H_mod_fun(double *state, double *out_7461128603485662222);
void f_fun(double *state, double dt, double *out_7427120013843204316);
void F_fun(double *state, double dt, double *out_537719153909518512);
void h_25(double *state, double *unused, double *out_5730183952204665335);
void H_25(double *state, double *unused, double *out_6850079430537253322);
void h_24(double *state, double *unused, double *out_2156192023864214624);
void H_24(double *state, double *unused, double *out_6745777559073209274);
void h_30(double *state, double *unused, double *out_8532523385339766571);
void H_30(double *state, double *unused, double *out_2577225185951832736);
void h_26(double *state, double *unused, double *out_2234723025716285968);
void H_26(double *state, double *unused, double *out_4924351094218360094);
void h_27(double *state, double *unused, double *out_5815685395346792460);
void H_27(double *state, double *unused, double *out_2622377755614588354);
void h_29(double *state, double *unused, double *out_2052512895026669290);
void H_29(double *state, double *unused, double *out_8160466959469793030);
void h_28(double *state, double *unused, double *out_2232376890138292912);
void H_28(double *state, double *unused, double *out_2704987191912913244);
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
