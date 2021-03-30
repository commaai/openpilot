/******************************************************************************
 *                      Code generated with sympy 1.7.1                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_6950268572851752455);
void inv_err_fun(double *nom_x, double *true_x, double *out_4255607517521366588);
void H_mod_fun(double *state, double *out_9101789641950081223);
void f_fun(double *state, double dt, double *out_8065546771995148323);
void F_fun(double *state, double dt, double *out_7338680300367955279);
void h_3(double *state, double *unused, double *out_744929963730460454);
void H_3(double *state, double *unused, double *out_7001390638340323824);
void h_4(double *state, double *unused, double *out_2827067238847056154);
void H_4(double *state, double *unused, double *out_7940325287044802466);
void h_9(double *state, double *unused, double *out_907146022988570509);
void H_9(double *state, double *unused, double *out_1428209545460910329);
void h_10(double *state, double *unused, double *out_5693832136999998389);
void H_10(double *state, double *unused, double *out_5009502699981707998);
void h_12(double *state, double *unused, double *out_995554410106486465);
void H_12(double *state, double *unused, double *out_1212259513329263928);
void h_31(double *state, double *unused, double *out_915484525720524238);
void H_31(double *state, double *unused, double *out_4110502167464644590);
void h_32(double *state, double *unused, double *out_7324229075683226179);
void H_32(double *state, double *unused, double *out_985025399522460588);
void h_13(double *state, double *unused, double *out_837326564915423285);
void H_13(double *state, double *unused, double *out_3004867798354010718);
void h_14(double *state, double *unused, double *out_907146022988570509);
void H_14(double *state, double *unused, double *out_1428209545460910329);
void h_19(double *state, double *unused, double *out_6426269517784467286);
void H_19(double *state, double *unused, double *out_8187497294161841978);
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