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
void car_err_fun(double *nom_x, double *delta_x, double *out_2069516698642958211);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_8998358058995224576);
void car_H_mod_fun(double *state, double *out_7695547058271246574);
void car_f_fun(double *state, double dt, double *out_2845810596498337567);
void car_F_fun(double *state, double dt, double *out_6746448386310637026);
void car_h_25(double *state, double *unused, double *out_707314913295055078);
void car_H_25(double *state, double *unused, double *out_1198524098381216623);
void car_h_24(double *state, double *unused, double *out_6878495411227386244);
void car_H_24(double *state, double *unused, double *out_8616915732058143516);
void car_h_30(double *state, double *unused, double *out_2240411077060429883);
void car_H_30(double *state, double *unused, double *out_1069185151237976553);
void car_h_26(double *state, double *unused, double *out_5133980333049960676);
void car_H_26(double *state, double *unused, double *out_2542979220492839601);
void car_h_27(double *state, double *unused, double *out_7320798468675099859);
void car_H_27(double *state, double *unused, double *out_1105578160562448358);
void car_h_29(double *state, double *unused, double *out_3244672954280402572);
void car_H_29(double *state, double *unused, double *out_2818940887431999391);
void car_h_28(double *state, double *unused, double *out_6124272337373606112);
void car_H_28(double *state, double *unused, double *out_7901339904501529965);
void car_h_31(double *state, double *unused, double *out_8364353615874624961);
void car_H_31(double *state, double *unused, double *out_1229170060258177051);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}