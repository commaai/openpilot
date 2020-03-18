/******************************************************************************
 *                       Code generated with sympy 1.4                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_3259377041439391293);
void inv_err_fun(double *nom_x, double *true_x, double *out_68172643959344193);
void H_mod_fun(double *state, double *out_38284492468958156);
void f_fun(double *state, double dt, double *out_3815066989981844339);
void F_fun(double *state, double dt, double *out_7424907043672160838);
void h_3(double *state, double *unused, double *out_1229988802839845909);
void H_3(double *state, double *unused, double *out_6679490204903323283);
void h_4(double *state, double *unused, double *out_5647162554835604562);
void H_4(double *state, double *unused, double *out_8506744682740913932);
void h_9(double *state, double *unused, double *out_4459836647641077847);
void H_9(double *state, double *unused, double *out_8925767257136624014);
void h_10(double *state, double *unused, double *out_9070397578161175042);
void H_10(double *state, double *unused, double *out_1748964158189932574);
void h_12(double *state, double *unused, double *out_6521141449751668803);
void H_12(double *state, double *unused, double *out_8458097209359346526);
void h_13(double *state, double *unused, double *out_3526910856013430002);
void H_13(double *state, double *unused, double *out_7864941421191853340);
void h_14(double *state, double *unused, double *out_4459836647641077847);
void H_14(double *state, double *unused, double *out_8925767257136624014);
void h_19(double *state, double *unused, double *out_6190249993410873977);
void H_19(double *state, double *unused, double *out_4524439862299680294);
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