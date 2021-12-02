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
void car_err_fun(double *nom_x, double *delta_x, double *out_6314627675369028590);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_7796662908045054598);
void car_H_mod_fun(double *state, double *out_7490673381729093055);
void car_f_fun(double *state, double dt, double *out_1191313618120458023);
void car_F_fun(double *state, double dt, double *out_3988374560714266677);
void car_h_25(double *state, double *unused, double *out_3077195206092010461);
void car_H_25(double *state, double *unused, double *out_8769480316130288681);
void car_h_24(double *state, double *unused, double *out_357861895765212515);
void car_H_24(double *state, double *unused, double *out_3828366448772007202);
void car_h_30(double *state, double *unused, double *out_3089918833879123842);
void car_H_30(double *state, double *unused, double *out_63364846630079655);
void car_h_26(double *state, double *unused, double *out_8091592101539969487);
void car_H_26(double *state, double *unused, double *out_7463244544175447536);
void car_h_27(double *state, double *unused, double *out_4178948511001441248);
void car_H_27(double *state, double *unused, double *out_1224217141206545657);
void car_h_29(double *state, double *unused, double *out_6730426071702673160);
void car_H_29(double *state, double *unused, double *out_3285595366136104527);
void car_h_28(double *state, double *unused, double *out_917016906443250257);
void car_H_28(double *state, double *unused, double *out_3202187763716681420);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}