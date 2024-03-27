#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void live_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_9(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_12(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_35(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_32(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_33(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_H(double *in_vec, double *out_7641177276838842515);
void live_err_fun(double *nom_x, double *delta_x, double *out_2532183155020331779);
void live_inv_err_fun(double *nom_x, double *true_x, double *out_4705040342203024337);
void live_H_mod_fun(double *state, double *out_5358584358519148047);
void live_f_fun(double *state, double dt, double *out_2483488690285755121);
void live_F_fun(double *state, double dt, double *out_6736351039582247501);
void live_h_4(double *state, double *unused, double *out_2376661880563825318);
void live_H_4(double *state, double *unused, double *out_6207540533751071916);
void live_h_9(double *state, double *unused, double *out_2065860645518724890);
void live_H_9(double *state, double *unused, double *out_1079678401513375554);
void live_h_10(double *state, double *unused, double *out_6788541366407009480);
void live_H_10(double *state, double *unused, double *out_4717283011623182729);
void live_h_12(double *state, double *unused, double *out_2109637504871451338);
void live_H_12(double *state, double *unused, double *out_1188084125719110121);
void live_h_35(double *state, double *unused, double *out_5362197287562187074);
void live_H_35(double *state, double *unused, double *out_2840878476378464540);
void live_h_32(double *state, double *unused, double *out_3168274671828253638);
void live_H_32(double *state, double *unused, double *out_8440288889245546773);
void live_h_13(double *state, double *unused, double *out_7772958891132362007);
void live_H_13(double *state, double *unused, double *out_6618699489849958875);
void live_h_14(double *state, double *unused, double *out_2065860645518724890);
void live_H_14(double *state, double *unused, double *out_1079678401513375554);
void live_h_33(double *state, double *unused, double *out_6155803893915517555);
void live_H_33(double *state, double *unused, double *out_309678528260393064);
void live_predict(double *in_x, double *in_P, double *in_Q, double dt);
}