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
void car_err_fun(double *nom_x, double *delta_x, double *out_7126119498051021768);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_2773982190367778391);
void car_H_mod_fun(double *state, double *out_8542714350044848421);
void car_f_fun(double *state, double dt, double *out_7713290282379442360);
void car_F_fun(double *state, double dt, double *out_2189258202331065399);
void car_h_25(double *state, double *unused, double *out_2256658881339624696);
void car_H_25(double *state, double *unused, double *out_5019733700055356744);
void car_h_24(double *state, double *unused, double *out_1108309987830345920);
void car_H_24(double *state, double *unused, double *out_1461808101253033608);
void car_h_30(double *state, double *unused, double *out_1981464819055118807);
void car_H_30(double *state, double *unused, double *out_1896956641436260011);
void car_h_26(double *state, double *unused, double *out_3461392923629465767);
void car_H_26(double *state, double *unused, double *out_8761237018929412968);
void car_h_27(double *state, double *unused, double *out_2969248334495700251);
void car_H_27(double *state, double *unused, double *out_277806670364164900);
void car_h_29(double *state, double *unused, double *out_3020208448605726838);
void car_H_29(double *state, double *unused, double *out_2407187985750652195);
void car_h_28(double *state, double *unused, double *out_364843013687002600);
void car_H_28(double *state, double *unused, double *out_2675211031318878379);
void car_h_31(double *state, double *unused, double *out_5894108409593647046);
void car_H_31(double *state, double *unused, double *out_4989087738178396316);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}