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
void car_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_err_fun(double *nom_x, double *delta_x, double *out_1299726308890531037);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_2964139867777846776);
void car_H_mod_fun(double *state, double *out_8544953491798085886);
void car_f_fun(double *state, double dt, double *out_5256147688816920328);
void car_F_fun(double *state, double dt, double *out_6834940091299106646);
void car_h_25(double *state, double *unused, double *out_6166873754981284210);
void car_H_25(double *state, double *unused, double *out_7142658040254357010);
void car_h_24(double *state, double *unused, double *out_2192149979727043903);
void car_H_24(double *state, double *unused, double *out_1331272409231580045);
void car_h_30(double *state, double *unused, double *out_4319993932698147994);
void car_H_30(double *state, double *unused, double *out_7271996987397597080);
void car_h_26(double *state, double *unused, double *out_7644508679924030782);
void car_H_26(double *state, double *unused, double *out_7562582714581138382);
void car_h_27(double *state, double *unused, double *out_3874213499084040753);
void car_H_27(double *state, double *unused, double *out_8999983774511529625);
void car_h_29(double *state, double *unused, double *out_5263503785599810429);
void car_H_29(double *state, double *unused, double *out_7286621047641978592);
void car_h_28(double *state, double *unused, double *out_407350342159619755);
void car_H_28(double *state, double *unused, double *out_9196492754502246773);
void car_h_31(double *state, double *unused, double *out_2647431620660638604);
void car_H_31(double *state, double *unused, double *out_7112012078377396582);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}