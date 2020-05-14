/******************************************************************************
 *                      Code generated with sympy 1.5.1                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_4893295704109867170);
void inv_err_fun(double *nom_x, double *true_x, double *out_8416321681820466281);
void H_mod_fun(double *state, double *out_462850411114629672);
void f_fun(double *state, double dt, double *out_793761718918406765);
void F_fun(double *state, double dt, double *out_218762218435608313);
void h_3(double *state, double *unused, double *out_5177375843913594329);
void H_3(double *state, double *unused, double *out_612731026909448861);
void h_4(double *state, double *unused, double *out_8879199284893495296);
void H_4(double *state, double *unused, double *out_2716866069264098879);
void h_9(double *state, double *unused, double *out_1834646685490159049);
void H_9(double *state, double *unused, double *out_2001743734408262921);
void h_10(double *state, double *unused, double *out_2826441252690492484);
void H_10(double *state, double *unused, double *out_2249359780824616317);
void h_12(double *state, double *unused, double *out_2829107275878131469);
void H_12(double *state, double *unused, double *out_5260235586358240305);
void h_13(double *state, double *unused, double *out_7221105317765519708);
void H_13(double *state, double *unused, double *out_4543079268205798191);
void h_14(double *state, double *unused, double *out_1834646685490159049);
void H_14(double *state, double *unused, double *out_2001743734408262921);
void h_19(double *state, double *unused, double *out_3333104463935752152);
void H_19(double *state, double *unused, double *out_6434423585510694102);
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
const static double MAHA_THRESH_13 = 7.814728;
void update_13(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_14 = 7.814728;
void update_14(double *, double *, double *, double *, double *);
const static double MAHA_THRESH_19 = 7.814728;
void update_19(double *, double *, double *, double *, double *);