/******************************************************************************
 *                       Code generated with sympy 1.4                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_8900724936196313707);
void inv_err_fun(double *nom_x, double *true_x, double *out_7977763515281484839);
void H_mod_fun(double *state, double *out_6060248653239228829);
void f_fun(double *state, double dt, double *out_6120496469287493804);
void F_fun(double *state, double dt, double *out_5230264909827851820);
void h_3(double *state, double *unused, double *out_6568460050549030718);
void H_3(double *state, double *unused, double *out_946786442953426365);
void h_4(double *state, double *unused, double *out_7341462235099715835);
void H_4(double *state, double *unused, double *out_5708352876022396058);
void h_9(double *state, double *unused, double *out_368062729746095277);
void H_9(double *state, double *unused, double *out_7131118677423599767);
void h_10(double *state, double *unused, double *out_2254376106876256481);
void H_10(double *state, double *unused, double *out_1169144658584652285);
void h_12(double *state, double *unused, double *out_8624202941148333689);
void H_12(double *state, double *unused, double *out_5372901294693848953);
void h_31(double *state, double *unused, double *out_5549841177629691742);
void H_31(double *state, double *unused, double *out_7999788167863417633);
void h_32(double *state, double *unused, double *out_3155132724619494487);
void H_32(double *state, double *unused, double *out_2575980467354797887);
void h_13(double *state, double *unused, double *out_4956049755813539351);
void H_13(double *state, double *unused, double *out_6996870794441281148);
void h_14(double *state, double *unused, double *out_368062729746095277);
void H_14(double *state, double *unused, double *out_7131118677423599767);
void h_19(double *state, double *unused, double *out_8136788542865316821);
void H_19(double *state, double *unused, double *out_4830128544457818515);
#define DIM 23
#define EDIM 22
#define MEDIM 22
typedef void (*Hfun)(double *, double *, double *);

void predict(double *x, double *P, double *Q, double dt);
const static double MAHA_THRESH_3 = 3.841459;
void update_3(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_4 = 7.814728;
void update_4(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_9 = 7.814728;
void update_9(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_10 = 7.814728;
void update_10(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_12 = 7.814728;
void update_12(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_31 = 7.814728;
void update_31(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_32 = 9.487729;
void update_32(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_13 = 7.814728;
void update_13(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_14 = 7.814728;
void update_14(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_19 = 7.814728;
void update_19(double *, double *, double *, double *, double *);