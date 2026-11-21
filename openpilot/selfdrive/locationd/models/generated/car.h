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
void car_err_fun(double *nom_x, double *delta_x, double *out_52242666316804912);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_4570214122495112138);
void car_H_mod_fun(double *state, double *out_1673187515601145384);
void car_f_fun(double *state, double dt, double *out_609770788174560294);
void car_F_fun(double *state, double dt, double *out_4074794192207865105);
void car_h_25(double *state, double *unused, double *out_5272859061006755531);
void car_H_25(double *state, double *unused, double *out_7520000710311091853);
void car_h_24(double *state, double *unused, double *out_2158751071353599023);
void car_H_24(double *state, double *unused, double *out_8754093764392960197);
void car_h_30(double *state, double *unused, double *out_6403049786352445149);
void car_H_30(double *state, double *unused, double *out_603310368819475098);
void car_h_26(double *state, double *unused, double *out_3695140238103757981);
void car_H_26(double *state, double *unused, double *out_7185240044524403539);
void car_h_27(double *state, double *unused, double *out_7607783828642286220);
void car_H_27(double *state, double *unused, double *out_2778073680619900009);
void car_h_29(double *state, double *unused, double *out_645064872358707193);
void car_H_29(double *state, double *unused, double *out_4491436407489451042);
void car_h_28(double *state, double *unused, double *out_5613053029231940449);
void car_H_28(double *state, double *unused, double *out_8872908649150570000);
void car_h_31(double *state, double *unused, double *out_2490406195813916910);
void car_H_31(double *state, double *unused, double *out_7489354748434131425);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}