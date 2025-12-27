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
void car_err_fun(double *nom_x, double *delta_x, double *out_6612010329800967159);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_8935913149389536496);
void car_H_mod_fun(double *state, double *out_1202674038686624606);
void car_f_fun(double *state, double dt, double *out_2072276713079294601);
void car_F_fun(double *state, double dt, double *out_6787727984604679414);
void car_h_25(double *state, double *unused, double *out_2354092351086383892);
void car_H_25(double *state, double *unused, double *out_9124012071660484411);
void car_h_24(double *state, double *unused, double *out_5283557851803415774);
void car_H_24(double *state, double *unused, double *out_5764806403246744069);
void car_h_30(double *state, double *unused, double *out_1507591125342117760);
void car_H_30(double *state, double *unused, double *out_4596315741532876213);
void car_h_26(double *state, double *unused, double *out_2451100978243780195);
void car_H_26(double *state, double *unused, double *out_5382508752786428187);
void car_h_27(double *state, double *unused, double *out_1372008257751085780);
void car_H_27(double *state, double *unused, double *out_6819909812716819430);
void car_h_29(double *state, double *unused, double *out_6509264393002963405);
void car_H_29(double *state, double *unused, double *out_5106547085847268397);
void car_h_28(double *state, double *unused, double *out_768692373074608263);
void car_H_28(double *state, double *unused, double *out_7070177357412594648);
void car_h_31(double *state, double *unused, double *out_1389583731408741016);
void car_H_31(double *state, double *unused, double *out_4756300650553076711);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}