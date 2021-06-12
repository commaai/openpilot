#pragma once
#include "rednose/helpers/common_ekf.h"
extern "C" {
void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_err_fun(double *nom_x, double *delta_x, double *out_3157583653505174283);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_5329952054111744571);
void car_H_mod_fun(double *state, double *out_8803128394076082262);
void car_f_fun(double *state, double dt, double *out_5487969453112289401);
void car_F_fun(double *state, double dt, double *out_249163952273201207);
void car_h_25(double *state, double *unused, double *out_1166715342447388350);
void car_H_25(double *state, double *unused, double *out_7931236138849448777);
void car_h_24(double *state, double *unused, double *out_3877885127749003657);
void car_H_24(double *state, double *unused, double *out_6235472662875167742);
void car_h_30(double *state, double *unused, double *out_891521280162882461);
void car_H_30(double *state, double *unused, double *out_1682662772099734503);
void car_h_26(double *state, double *unused, double *out_6111241423630183318);
void car_H_26(double *state, double *unused, double *out_4810914779920893566);
void car_h_27(double *state, double *unused, double *out_4711570145111326738);
void car_H_27(double *state, double *unused, double *out_2970244759936359815);
void car_h_29(double *state, double *unused, double *out_3928307331607215146);
void car_H_29(double *state, double *unused, double *out_1539567747406290369);
void car_h_28(double *state, double *unused, double *out_6192292826639898941);
void car_H_28(double *state, double *unused, double *out_782359815361327124);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}