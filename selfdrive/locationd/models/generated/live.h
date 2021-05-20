#pragma once
#include "rednose/helpers/common_ekf.h"
extern "C" {
void live_update_3(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_9(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_12(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_32(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_19(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_H(double *in_vec, double *out_6069105127087503274);
void live_err_fun(double *nom_x, double *delta_x, double *out_7840132191283171662);
void live_inv_err_fun(double *nom_x, double *true_x, double *out_5918052630055149403);
void live_H_mod_fun(double *state, double *out_5935626302997998690);
void live_f_fun(double *state, double dt, double *out_5659457342976443075);
void live_F_fun(double *state, double dt, double *out_6621988335735070412);
void live_h_3(double *state, double *unused, double *out_4195337123801230203);
void live_H_3(double *state, double *unused, double *out_6176339591529763545);
void live_h_4(double *state, double *unused, double *out_2265658319195655908);
void live_H_4(double *state, double *unused, double *out_3462673777983642645);
void live_h_9(double *state, double *unused, double *out_1765330615151165563);
void live_H_9(double *state, double *unused, double *out_8142525636203113338);
void live_h_10(double *state, double *unused, double *out_2300116803469151225);
void live_H_10(double *state, double *unused, double *out_4148224349411515613);
void live_h_12(double *state, double *unused, double *out_5881586180038087301);
void live_H_12(double *state, double *unused, double *out_8358475668334759739);
void live_h_31(double *state, double *unused, double *out_1231433603799948957);
void live_H_31(double *state, double *unused, double *out_367149341596515231);
void live_h_32(double *state, double *unused, double *out_8175259575514077269);
void live_H_32(double *state, double *unused, double *out_7272350460975888741);
void live_h_13(double *state, double *unused, double *out_7157488464707671805);
void live_H_13(double *state, double *unused, double *out_165012951736613355);
void live_h_14(double *state, double *unused, double *out_1765330615151165563);
void live_H_14(double *state, double *unused, double *out_8142525636203113338);
void live_h_19(double *state, double *unused, double *out_6733255088328622280);
void live_H_19(double *state, double *unused, double *out_3709845785100682157);
void live_predict(double *in_x, double *in_P, double *in_Q, double dt);
}