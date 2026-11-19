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
void car_err_fun(double *nom_x, double *delta_x, double *out_790466696376099935);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_529756539852428822);
void car_H_mod_fun(double *state, double *out_9219675536978344786);
void car_f_fun(double *state, double dt, double *out_6134589376329078738);
void car_F_fun(double *state, double dt, double *out_4746845148006465944);
void car_h_25(double *state, double *unused, double *out_8027414408540613172);
void car_H_25(double *state, double *unused, double *out_33880310818969593);
void car_h_24(double *state, double *unused, double *out_113766384020535879);
void car_H_24(double *state, double *unused, double *out_8741012096993201301);
void car_h_30(double *state, double *unused, double *out_8968698272540055277);
void car_H_30(double *state, double *unused, double *out_95458636324270477);
void car_h_26(double *state, double *unused, double *out_6441259749725163654);
void car_H_26(double *state, double *unused, double *out_3707623008055086631);
void car_h_27(double *state, double *unused, double *out_3416566961375872665);
void car_H_27(double *state, double *unused, double *out_2270221948124695388);
void car_h_29(double *state, double *unused, double *out_4476372533508650694);
void car_H_29(double *state, double *unused, double *out_414772707990121707);
void car_h_28(double *state, double *unused, double *out_5565402210630968100);
void car_H_28(double *state, double *unused, double *out_2019954403428920170);
void car_h_31(double *state, double *unused, double *out_1860300368569478869);
void car_H_31(double *state, double *unused, double *out_64526272695930021);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}