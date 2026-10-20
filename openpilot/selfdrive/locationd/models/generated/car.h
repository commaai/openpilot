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
void car_err_fun(double *nom_x, double *delta_x, double *out_3479962638771710567);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_3560122421157608575);
void car_H_mod_fun(double *state, double *out_7086273587120705954);
void car_f_fun(double *state, double dt, double *out_6844953383171278992);
void car_F_fun(double *state, double dt, double *out_6778960402696977626);
void car_h_25(double *state, double *unused, double *out_5169296545033589210);
void car_H_25(double *state, double *unused, double *out_7837489836599353985);
void car_h_24(double *state, double *unused, double *out_2794263254641248906);
void car_H_24(double *state, double *unused, double *out_4279564237797030849);
void car_h_30(double *state, double *unused, double *out_5012109876682460065);
void car_H_30(double *state, double *unused, double *out_920799495107737230);
void car_h_26(double *state, double *unused, double *out_903520877051402271);
void car_H_26(double *state, double *unused, double *out_6867750918236141407);
void car_h_27(double *state, double *unused, double *out_7831542771752882352);
void car_H_27(double *state, double *unused, double *out_3095562806908162141);
void car_h_29(double *state, double *unused, double *out_8432960929366245906);
void car_H_29(double *state, double *unused, double *out_4808925533777713174);
void car_h_28(double *state, double *unused, double *out_1189000640687594090);
void car_H_28(double *state, double *unused, double *out_8555419522862307868);
void car_h_31(double *state, double *unused, double *out_6506612302325611470);
void car_H_31(double *state, double *unused, double *out_7806843874722393557);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}