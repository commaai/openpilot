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
void car_err_fun(double *nom_x, double *delta_x, double *out_6387606906665158225);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_1940653918789291055);
void car_H_mod_fun(double *state, double *out_4168849259471386666);
void car_f_fun(double *state, double dt, double *out_7308117309923687076);
void car_F_fun(double *state, double dt, double *out_7708045199441947550);
void car_h_25(double *state, double *unused, double *out_8572254133923072558);
void car_H_25(double *state, double *unused, double *out_8963680102399255775);
void car_h_24(double *state, double *unused, double *out_4030352194828355617);
void car_H_24(double *state, double *unused, double *out_7310414372304796275);
void car_h_30(double *state, double *unused, double *out_6926763949430994097);
void car_H_30(double *state, double *unused, double *out_6445347143892007148);
void car_h_26(double *state, double *unused, double *out_5983254096529331662);
void car_H_26(double *state, double *unused, double *out_5741560652436239617);
void car_h_27(double *state, double *unused, double *out_6598642127039022469);
void car_H_27(double *state, double *unused, double *out_2780604329382262732);
void car_h_29(double *state, double *unused, double *out_6323448064754516580);
void car_H_29(double *state, double *unused, double *out_5935115799577614964);
void car_h_28(double *state, double *unused, double *out_5449060429256680001);
void car_H_28(double *state, double *unused, double *out_7429229257062406078);
void car_h_31(double *state, double *unused, double *out_3103654713436128122);
void car_H_31(double *state, double *unused, double *out_5115352550202888141);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}