/******************************************************************************
 *                       Code generated with sympy 1.4                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_6788465772816668761);
void inv_err_fun(double *nom_x, double *true_x, double *out_5601436874753092679);
void H_mod_fun(double *state, double *out_3068454842243606904);
void f_fun(double *state, double dt, double *out_6548865548862311877);
void F_fun(double *state, double dt, double *out_2795244210510601530);
void h_3(double *state, double *unused, double *out_5890016392798834445);
void H_3(double *state, double *unused, double *out_5951574930511872297);
void h_4(double *state, double *unused, double *out_8526004289681937483);
void H_4(double *state, double *unused, double *out_2015221744362692841);
void h_9(double *state, double *unused, double *out_6849354457489368967);
void H_9(double *state, double *unused, double *out_7925197860050783002);
void h_10(double *state, double *unused, double *out_6748519270298518309);
void H_10(double *state, double *unused, double *out_409868221626246429);
void h_12(double *state, double *unused, double *out_1066610582664881043);
void H_12(double *state, double *unused, double *out_6435812489740243594);
void h_31(double *state, double *unused, double *out_7114311135786219984);
void H_31(double *state, double *unused, double *out_5420128406978512594);
void h_32(double *state, double *unused, double *out_6656960917418393346);
void H_32(double *state, double *unused, double *out_5465957462164714200);
void h_13(double *state, double *unused, double *out_7669147739270683552);
void H_13(double *state, double *unused, double *out_3028851125003744530);
void h_14(double *state, double *unused, double *out_6849354457489368967);
void H_14(double *state, double *unused, double *out_7925197860050783002);
void h_19(double *state, double *unused, double *out_2880896277160693049);
void H_19(double *state, double *unused, double *out_623983078872615106);
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