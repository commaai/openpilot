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
void car_err_fun(double *nom_x, double *delta_x, double *out_7661067751022070458);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_5136938064863750921);
void car_H_mod_fun(double *state, double *out_3671347944419902585);
void car_f_fun(double *state, double dt, double *out_7888740610946713678);
void car_F_fun(double *state, double dt, double *out_3995884459612552931);
void car_h_25(double *state, double *unused, double *out_3968497935547833524);
void car_H_25(double *state, double *unused, double *out_1232343880764762255);
void car_h_24(double *state, double *unused, double *out_3248956284435332617);
void car_H_24(double *state, double *unused, double *out_6210330712223954065);
void car_h_30(double *state, double *unused, double *out_5370462393698290492);
void car_H_30(double *state, double *unused, double *out_3295352449362845943);
void car_h_26(double *state, double *unused, double *out_1313132500629109286);
void car_H_26(double *state, double *unused, double *out_2509159438109293969);
void car_h_27(double *state, double *unused, double *out_1033175332113012121);
void car_H_27(double *state, double *unused, double *out_5470115761163270854);
void car_h_29(double *state, double *unused, double *out_875988663761882976);
void car_H_29(double *state, double *unused, double *out_2785121105048453759);
void car_h_28(double *state, double *unused, double *out_5458536247411243654);
void car_H_28(double *state, double *unused, double *out_821490833483127508);
void car_h_31(double *state, double *unused, double *out_3492060828701054801);
void car_H_31(double *state, double *unused, double *out_3135367540342645445);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}