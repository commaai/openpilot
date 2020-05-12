/******************************************************************************
 *                       Code generated with sympy 1.4                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_6984294585133015189);
void inv_err_fun(double *nom_x, double *true_x, double *out_6789192750565377785);
void H_mod_fun(double *state, double *out_4814388503184708105);
void f_fun(double *state, double dt, double *out_2160837932202987216);
void F_fun(double *state, double dt, double *out_5183711748696818220);
void h_3(double *state, double *unused, double *out_6343399653673618493);
void H_3(double *state, double *unused, double *out_5092099246839707065);
void h_4(double *state, double *unused, double *out_2929807715366999974);
void H_4(double *state, double *unused, double *out_8417274009086201443);
void h_9(double *state, double *unused, double *out_4887380865297738989);
void H_9(double *state, double *unused, double *out_2215096800156305355);
void h_10(double *state, double *unused, double *out_2382488801300693052);
void H_10(double *state, double *unused, double *out_6414374174391957608);
void h_12(double *state, double *unused, double *out_8480975742075338993);
void H_12(double *state, double *unused, double *out_100474230096376165);
void h_13(double *state, double *unused, double *out_6586684637762242075);
void H_13(double *state, double *unused, double *out_6892547764146259716);
void h_14(double *state, double *unused, double *out_4887380865297738989);
void H_14(double *state, double *unused, double *out_2215096800156305355);
void h_19(double *state, double *unused, double *out_8857247847418984403);
void H_19(double *state, double *unused, double *out_9146524052143216033);
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