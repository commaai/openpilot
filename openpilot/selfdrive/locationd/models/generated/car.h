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
void car_err_fun(double *nom_x, double *delta_x, double *out_23523217466685504);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_490723024718224285);
void car_H_mod_fun(double *state, double *out_4030678833409053918);
void car_f_fun(double *state, double dt, double *out_7376853175200153600);
void car_F_fun(double *state, double dt, double *out_732326874719211251);
void car_h_25(double *state, double *unused, double *out_1982449565641168842);
void car_H_25(double *state, double *unused, double *out_5162509392503183319);
void car_h_24(double *state, double *unused, double *out_5761352688166713048);
void car_H_24(double *state, double *unused, double *out_8537725354239859263);
void car_h_30(double *state, double *unused, double *out_2257643627925674731);
void car_H_30(double *state, double *unused, double *out_1754180948988433436);
void car_h_26(double *state, double *unused, double *out_3425387429001816468);
void car_H_26(double *state, double *unused, double *out_8904012711377239543);
void car_h_27(double *state, double *unused, double *out_7338031019540344707);
void car_H_27(double *state, double *unused, double *out_420582362811991475);
void car_h_29(double *state, double *unused, double *out_4953265858328546780);
void car_H_29(double *state, double *unused, double *out_2133945089681542508);
void car_h_28(double *state, double *unused, double *out_6107039786508361264);
void car_H_28(double *state, double *unused, double *out_7216344106751073082);
void car_h_31(double *state, double *unused, double *out_6080752863920540706);
void car_H_31(double *state, double *unused, double *out_5131863430626222891);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}