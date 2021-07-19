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
void live_H(double *in_vec, double *out_1290796793546397921);
void live_err_fun(double *nom_x, double *delta_x, double *out_8015454969191180433);
void live_inv_err_fun(double *nom_x, double *true_x, double *out_8436131724044148520);
void live_H_mod_fun(double *state, double *out_1141367288153725546);
void live_f_fun(double *state, double dt, double *out_728053839780956570);
void live_F_fun(double *state, double dt, double *out_3912237817192903500);
void live_h_3(double *state, double *unused, double *out_5297274283000343270);
void live_H_3(double *state, double *unused, double *out_7559209817469584163);
void live_h_4(double *state, double *unused, double *out_8103984854313015970);
void live_H_4(double *state, double *unused, double *out_6056554309960341744);
void live_h_9(double *state, double *unused, double *out_6001131901424453768);
void live_H_9(double *state, double *unused, double *out_8379059853831197714);
void live_h_10(double *state, double *unused, double *out_3821507716443431332);
void live_H_10(double *state, double *unused, double *out_1921616749995426926);
void live_h_12(double *state, double *unused, double *out_6443218251506020026);
void live_H_12(double *state, double *unused, double *out_8163109821699551313);
void live_h_31(double *state, double *unused, double *out_2363526457636194341);
void live_H_31(double *state, double *unused, double *out_5488020046556131492);
void live_h_32(double *state, double *unused, double *out_661403799137613573);
void live_H_32(double *state, double *unused, double *out_812345697704269694);
void live_h_13(double *state, double *unused, double *out_1133152421767922579);
void live_H_13(double *state, double *unused, double *out_788371292687742634);
void live_h_14(double *state, double *unused, double *out_6001131901424453768);
void live_H_14(double *state, double *unused, double *out_8379059853831197714);
void live_h_19(double *state, double *unused, double *out_6703514544593698245);
void live_H_19(double *state, double *unused, double *out_5635004368775922721);
void live_predict(double *in_x, double *in_P, double *in_Q, double dt);
}