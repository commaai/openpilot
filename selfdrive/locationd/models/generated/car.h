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
void car_err_fun(double *nom_x, double *delta_x, double *out_2195023721757572954);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_6339939555658911788);
void car_H_mod_fun(double *state, double *out_8639855113291280281);
void car_f_fun(double *state, double dt, double *out_8753740640370834672);
void car_F_fun(double *state, double dt, double *out_4380248598819438044);
void car_h_25(double *state, double *unused, double *out_5041160612363327269);
void car_H_25(double *state, double *unused, double *out_2934563808916292903);
void car_h_24(double *state, double *unused, double *out_6427212538985091517);
void car_H_24(double *state, double *unused, double *out_7214599080901228000);
void car_h_30(double *state, double *unused, double *out_1729674613987023667);
void car_H_30(double *state, double *unused, double *out_6679335102032890377);
void car_h_26(double *state, double *unused, double *out_2332990498663501184);
void car_H_26(double *state, double *unused, double *out_8639156963855502176);
void car_h_27(double *state, double *unused, double *out_4097650759461664834);
void car_H_27(double *state, double *unused, double *out_7966917089869515689);
void car_h_29(double *state, double *unused, double *out_7735100287715687184);
void car_H_29(double *state, double *unused, double *out_6104776488177354202);
void car_h_28(double *state, double *unused, double *out_3201468729684098971);
void car_H_28(double *state, double *unused, double *out_4214312514571828750);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}