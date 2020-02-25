/******************************************************************************
 *                       Code generated with sympy 1.4                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_793318766921778757);
void inv_err_fun(double *nom_x, double *true_x, double *out_5402299630070885707);
void H_mod_fun(double *state, double *out_2522671031779474405);
void f_fun(double *state, double dt, double *out_378098923817338870);
void F_fun(double *state, double dt, double *out_8782550063293226640);
void h_3(double *state, double *unused, double *out_3302685666909389357);
void H_3(double *state, double *unused, double *out_8690330570363964837);
void h_4(double *state, double *unused, double *out_557092936251020754);
void H_4(double *state, double *unused, double *out_7147312946450067033);
void h_9(double *state, double *unused, double *out_4276545558886999701);
void H_9(double *state, double *unused, double *out_6376307510006402895);
void h_10(double *state, double *unused, double *out_1554040295281382172);
void H_10(double *state, double *unused, double *out_4551493278263167736);
void h_12(double *state, double *unused, double *out_3623617127578838799);
void H_12(double *state, double *unused, double *out_3191300629580742751);
void h_13(double *state, double *unused, double *out_4732068992027723369);
void H_13(double *state, double *unused, double *out_4088276358986605924);
void h_14(double *state, double *unused, double *out_4276545558886999701);
void H_14(double *state, double *unused, double *out_6376307510006402895);
void h_19(double *state, double *unused, double *out_1440879188724072987);
void H_19(double *state, double *unused, double *out_8270819139758291157);
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