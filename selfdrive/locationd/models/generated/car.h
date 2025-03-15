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
void car_err_fun(double *nom_x, double *delta_x, double *out_3675669119552892075);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_1525509250774896015);
void car_H_mod_fun(double *state, double *out_8908179234073782012);
void car_f_fun(double *state, double dt, double *out_2817676241617045765);
void car_F_fun(double *state, double dt, double *out_587119351607645481);
void car_h_25(double *state, double *unused, double *out_2948496957738460434);
void car_H_25(double *state, double *unused, double *out_345376613723532367);
void car_h_24(double *state, double *unused, double *out_1253312341000543437);
void car_H_24(double *state, double *unused, double *out_8873302273916824024);
void car_h_30(double *state, double *unused, double *out_4399128350367343601);
void car_H_30(double *state, double *unused, double *out_216037666580292297);
void car_h_26(double *state, double *unused, double *out_1362342298923010916);
void car_H_26(double *state, double *unused, double *out_3396126705150523857);
void car_h_27(double *state, double *unused, double *out_9204321679697507688);
void car_H_27(double *state, double *unused, double *out_1958725645220132614);
void car_h_29(double *state, double *unused, double *out_602544917293502044);
void car_H_29(double *state, double *unused, double *out_3672088372089683647);
void car_h_28(double *state, double *unused, double *out_4281120956433966857);
void car_H_28(double *state, double *unused, double *out_1708458100524357396);
void car_h_31(double *state, double *unused, double *out_5944889882572371594);
void car_H_31(double *state, double *unused, double *out_376022575600492795);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}