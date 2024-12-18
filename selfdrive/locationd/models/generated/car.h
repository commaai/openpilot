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
void car_err_fun(double *nom_x, double *delta_x, double *out_1778852089894645104);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_2123954078847546568);
void car_H_mod_fun(double *state, double *out_1868636510444762077);
void car_f_fun(double *state, double dt, double *out_5217702743341681626);
void car_F_fun(double *state, double dt, double *out_2769937760945802053);
void car_h_25(double *state, double *unused, double *out_8414728814641436601);
void car_H_25(double *state, double *unused, double *out_7023695843139760480);
void car_h_24(double *state, double *unused, double *out_4219916024026276113);
void car_H_24(double *state, double *unused, double *out_822663163288599054);
void car_h_30(double *state, double *unused, double *out_2828799302082129045);
void car_H_30(double *state, double *unused, double *out_2495999513012152282);
void car_h_26(double *state, double *unused, double *out_8603047268603794642);
void car_H_26(double *state, double *unused, double *out_3282192524265704256);
void car_h_27(double *state, double *unused, double *out_3413055546980590956);
void car_H_27(double *state, double *unused, double *out_4719593584196095499);
void car_h_29(double *state, double *unused, double *out_224393981273431394);
void car_H_29(double *state, double *unused, double *out_3006230857326544466);
void car_h_28(double *state, double *unused, double *out_6449841598424923641);
void car_H_28(double *state, double *unused, double *out_2076168159742986108);
void car_h_31(double *state, double *unused, double *out_8689922876925942490);
void car_H_31(double *state, double *unused, double *out_2655984422032352780);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}