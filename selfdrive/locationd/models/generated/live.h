/******************************************************************************
 *                      Code generated with sympy 1.6.1                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_2881535217378572505);
void inv_err_fun(double *nom_x, double *true_x, double *out_6982354823973209510);
void H_mod_fun(double *state, double *out_4734628292416748153);
void f_fun(double *state, double dt, double *out_202748511544268255);
void F_fun(double *state, double dt, double *out_3030002625149166559);
void h_3(double *state, double *unused, double *out_1508304444960141513);
void H_3(double *state, double *unused, double *out_2565766130081680703);
void h_4(double *state, double *unused, double *out_1837542008826379368);
void H_4(double *state, double *unused, double *out_1822943689399447903);
void h_9(double *state, double *unused, double *out_5866413323935688268);
void H_9(double *state, double *unused, double *out_7545591143106264892);
void h_10(double *state, double *unused, double *out_856278124557846577);
void H_10(double *state, double *unused, double *out_7840028771914449723);
void h_12(double *state, double *unused, double *out_5277063221784290535);
void H_12(double *state, double *unused, double *out_283611822339761666);
void h_31(double *state, double *unused, double *out_6375323783593327690);
void H_31(double *state, double *unused, double *out_2391477952803658155);
void h_32(double *state, double *unused, double *out_388716143553508675);
void H_32(double *state, double *unused, double *out_8691843697064059341);
void h_13(double *state, double *unused, double *out_8981811330497740659);
void H_13(double *state, double *unused, double *out_8492796277005188832);
void h_14(double *state, double *unused, double *out_5866413323935688268);
void H_14(double *state, double *unused, double *out_7545591143106264892);
void h_19(double *state, double *unused, double *out_2038495679218932262);
void H_19(double *state, double *unused, double *out_6468473079500855543);
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