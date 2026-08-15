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
void car_err_fun(double *nom_x, double *delta_x, double *out_6823495906263670498);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_2832333554644134208);
void car_H_mod_fun(double *state, double *out_8294511297811177194);
void car_f_fun(double *state, double dt, double *out_156104586604879915);
void car_F_fun(double *state, double dt, double *out_221487960878628462);
void car_h_25(double *state, double *unused, double *out_319137421637609022);
void car_H_25(double *state, double *unused, double *out_910996624291208289);
void car_h_24(double *state, double *unused, double *out_2686288490551722631);
void car_H_24(double *state, double *unused, double *out_4912708702921716922);
void car_h_30(double *state, double *unused, double *out_7573802073246914337);
void car_H_30(double *state, double *unused, double *out_7827686965782825044);
void car_h_26(double *state, double *unused, double *out_2284779318657795871);
void car_H_26(double *state, double *unused, double *out_2830506694582847935);
void car_h_27(double *state, double *unused, double *out_5320810689298454667);
void car_H_27(double *state, double *unused, double *out_5652923653982400133);
void car_h_29(double *state, double *unused, double *out_3870179296669571500);
void car_H_29(double *state, double *unused, double *out_8337918310097217228);
void car_h_28(double *state, double *unused, double *out_3661158482708386098);
void car_H_28(double *state, double *unused, double *out_1142838089956681474);
void car_h_31(double *state, double *unused, double *out_7089972647987959958);
void car_H_31(double *state, double *unused, double *out_941642586168168717);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}