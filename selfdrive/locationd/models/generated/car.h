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
void car_err_fun(double *nom_x, double *delta_x, double *out_7082197508459355294);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_7553533010452763308);
void car_H_mod_fun(double *state, double *out_2748508627650133070);
void car_f_fun(double *state, double dt, double *out_1858329490310860980);
void car_F_fun(double *state, double dt, double *out_8266273866488598058);
void car_h_25(double *state, double *unused, double *out_8442152877268712832);
void car_H_25(double *state, double *unused, double *out_6916230363893539613);
void car_h_24(double *state, double *unused, double *out_450177087820241572);
void car_H_24(double *state, double *unused, double *out_6784004382849502969);
void car_h_30(double *state, double *unused, double *out_9043571034882076386);
void car_H_30(double *state, double *unused, double *out_4397897405386290986);
void car_h_26(double *state, double *unused, double *out_4401499771166629585);
void car_H_26(double *state, double *unused, double *out_7789010390941955779);
void car_h_27(double *state, double *unused, double *out_3819966847812604518);
void car_H_27(double *state, double *unused, double *out_6572660717186715897);
void car_h_29(double *state, double *unused, double *out_3147348316868427976);
void car_H_29(double *state, double *unused, double *out_3887666061071898802);
void car_h_28(double *state, double *unused, double *out_8227735708483097952);
void car_H_28(double *state, double *unused, double *out_8970065078141429376);
void car_h_31(double *state, double *unused, double *out_6367141668186816434);
void car_H_31(double *state, double *unused, double *out_7162802288708604303);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}