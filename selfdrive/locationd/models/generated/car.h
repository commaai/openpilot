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
void car_err_fun(double *nom_x, double *delta_x, double *out_3990407830206607826);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_596620799835131520);
void car_H_mod_fun(double *state, double *out_4704152346071038473);
void car_f_fun(double *state, double dt, double *out_2233432862498219035);
void car_F_fun(double *state, double dt, double *out_8517999407047013030);
void car_h_25(double *state, double *unused, double *out_3158706812719367378);
void car_H_25(double *state, double *unused, double *out_2847211384764115064);
void car_h_24(double *state, double *unused, double *out_2356632080120596950);
void car_H_24(double *state, double *unused, double *out_3353821295087046362);
void car_h_30(double *state, double *unused, double *out_1340624676461203017);
void car_H_30(double *state, double *unused, double *out_2717872437620874994);
void car_h_26(double *state, double *unused, double *out_8343387380254268796);
void car_H_26(double *state, double *unused, double *out_894291934109941160);
void car_h_27(double *state, double *unused, double *out_121920761275034693);
void car_H_27(double *state, double *unused, double *out_543109125820450083);
void car_h_29(double *state, double *unused, double *out_397114823559540582);
void car_H_29(double *state, double *unused, double *out_3228103781935267178);
void car_h_28(double *state, double *unused, double *out_5123594028935880338);
void car_H_28(double *state, double *unused, double *out_6252652618118631524);
void car_h_31(double *state, double *unused, double *out_8517202034639833302);
void car_H_31(double *state, double *unused, double *out_2877857346641075492);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}