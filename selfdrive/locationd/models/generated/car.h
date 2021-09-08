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
void car_err_fun(double *nom_x, double *delta_x, double *out_4238289070007684364);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_3393139237430767898);
void car_H_mod_fun(double *state, double *out_6977086138309336193);
void car_f_fun(double *state, double dt, double *out_8805650115117454659);
void car_F_fun(double *state, double dt, double *out_6686524369442219331);
void car_h_25(double *state, double *unused, double *out_3742018707143453848);
void car_H_25(double *state, double *unused, double *out_5681767162363916568);
void car_h_24(double *state, double *unused, double *out_5471145934099093935);
void car_H_24(double *state, double *unused, double *out_8296790777733552478);
void car_h_30(double *state, double *unused, double *out_7933890140215746832);
void car_H_30(double *state, double *unused, double *out_1247279382587916360);
void car_h_26(double *state, double *unused, double *out_2932216872554901187);
void car_H_26(double *state, double *unused, double *out_4375531390409075423);
void car_h_27(double *state, double *unused, double *out_7213026032994887806);
void car_H_27(double *state, double *unused, double *out_1863496012559826456);
void car_h_29(double *state, double *unused, double *out_5121338036945893449);
void car_H_29(double *state, double *unused, double *out_6373308519902476640);
void car_h_28(double *state, double *unused, double *out_4307520237050144928);
void car_H_28(double *state, double *unused, double *out_3132120817739504272);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}