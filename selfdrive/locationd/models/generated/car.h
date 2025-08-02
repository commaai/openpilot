#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_err_fun(double *nom_x, double *delta_x, double *out_5914414296846804453);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_5011141162173190918);
void car_H_mod_fun(double *state, double *out_4291403483710066446);
void car_f_fun(double *state, double dt, double *out_6295743357823790773);
void car_F_fun(double *state, double dt, double *out_428584837545743588);
void car_h_25(double *state, double *unused, double *out_2854088158423216169);
void car_H_25(double *state, double *unused, double *out_5240524405738548655);
void car_h_24(double *state, double *unused, double *out_359654079315781340);
void car_H_24(double *state, double *unused, double *out_371709540710841803);
void car_h_30(double *state, double *unused, double *out_3011274826774345314);
void car_H_30(double *state, double *unused, double *out_712828075610940457);
void car_h_26(double *state, double *unused, double *out_7436635742072576847);
void car_H_26(double *state, double *unused, double *out_1499021086864492431);
void car_h_27(double *state, double *unused, double *out_1005994483947410829);
void car_H_27(double *state, double *unused, double *out_2936422146794883674);
void car_h_29(double *state, double *unused, double *out_1990398440886500331);
void car_H_29(double *state, double *unused, double *out_1223059419925332641);
void car_h_28(double *state, double *unused, double *out_7070785832501170307);
void car_H_28(double *state, double *unused, double *out_3186689691490658892);
void car_h_31(double *state, double *unused, double *out_4606470605825672156);
void car_H_31(double *state, double *unused, double *out_872812984631140955);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}