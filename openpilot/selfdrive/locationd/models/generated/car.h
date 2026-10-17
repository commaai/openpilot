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
void car_err_fun(double *nom_x, double *delta_x, double *out_1193858814788666420);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_8463344000198758499);
void car_H_mod_fun(double *state, double *out_5310208349491607054);
void car_f_fun(double *state, double dt, double *out_3672197714364501904);
void car_F_fun(double *state, double dt, double *out_3045272393170548372);
void car_h_25(double *state, double *unused, double *out_770246915850585039);
void car_H_25(double *state, double *unused, double *out_406516524306942214);
void car_h_24(double *state, double *unused, double *out_3584346622172585855);
void car_H_24(double *state, double *unused, double *out_6169055282284575887);
void car_h_30(double *state, double *unused, double *out_9036619681140424693);
void car_H_30(double *state, double *unused, double *out_6510173817184674541);
void car_h_26(double *state, double *unused, double *out_8323397744212662596);
void car_H_26(double *state, double *unused, double *out_4148019843180998438);
void car_h_27(double *state, double *unused, double *out_2505600896896871550);
void car_H_27(double *state, double *unused, double *out_2710618783250607195);
void car_h_29(double *state, double *unused, double *out_4265234329453479386);
void car_H_29(double *state, double *unused, double *out_2622047778514698597);
void car_h_28(double *state, double *unused, double *out_7507274164557717195);
void car_H_28(double *state, double *unused, double *out_2460351238554831977);
void car_h_31(double *state, double *unused, double *out_1045440978135090928);
void car_H_31(double *state, double *unused, double *out_375870562429981786);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}