#pragma once
#include "rednose/helpers/common_ekf.h"
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
void live_H(double *in_vec, double *out_1353179780026901961);
void live_err_fun(double *nom_x, double *delta_x, double *out_7751269372535998420);
void live_inv_err_fun(double *nom_x, double *true_x, double *out_1182572260918191243);
void live_H_mod_fun(double *state, double *out_4484216252147071045);
void live_f_fun(double *state, double dt, double *out_3264449968128188014);
void live_F_fun(double *state, double dt, double *out_6359784474284521014);
void live_h_4(double *state, double *unused, double *out_8752866696884087506);
void live_H_4(double *state, double *unused, double *out_5123057213746350251);
void live_h_9(double *state, double *unused, double *out_2208756653883987899);
void live_H_9(double *state, double *unused, double *out_5364246860375940896);
void live_h_10(double *state, double *unused, double *out_5423925561549761455);
void live_H_10(double *state, double *unused, double *out_2468338539411567376);
void live_h_12(double *state, double *unused, double *out_8595711851316088480);
void live_H_12(double *state, double *unused, double *out_8304230451931239570);
void live_h_35(double *state, double *unused, double *out_2795028386863618620);
void live_H_35(double *state, double *unused, double *out_5558667419606225861);
void live_h_32(double *state, double *unused, double *out_6062655590965651811);
void live_H_32(double *state, double *unused, double *out_242636952601386697);
void live_h_13(double *state, double *unused, double *out_4841202820403640116);
void live_H_13(double *state, double *unused, double *out_7088075903825718386);
void live_h_14(double *state, double *unused, double *out_2208756653883987899);
void live_H_14(double *state, double *unused, double *out_5364246860375940896);
void live_h_33(double *state, double *unused, double *out_9047983107143661110);
void live_H_33(double *state, double *unused, double *out_6806467797951736385);
void live_predict(double *in_x, double *in_P, double *in_Q, double dt);
}