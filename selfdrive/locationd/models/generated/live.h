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
void live_H(double *in_vec, double *out_2488790301073913633);
void live_err_fun(double *nom_x, double *delta_x, double *out_3367948852930505733);
void live_inv_err_fun(double *nom_x, double *true_x, double *out_7411058887648722315);
void live_H_mod_fun(double *state, double *out_1105391377250207785);
void live_f_fun(double *state, double dt, double *out_2945186301542193362);
void live_F_fun(double *state, double dt, double *out_6924942018240878733);
void live_h_3(double *state, double *unused, double *out_1864141882008389075);
void live_H_3(double *state, double *unused, double *out_6235344005769230067);
void live_h_4(double *state, double *unused, double *out_6814177113217099097);
void live_H_4(double *state, double *unused, double *out_1401833084360613006);
void live_h_9(double *state, double *unused, double *out_2479120142832280908);
void live_H_9(double *state, double *unused, double *out_3278018773858857687);
void live_h_10(double *state, double *unused, double *out_3386973186807228775);
void live_H_10(double *state, double *unused, double *out_269553952332063068);
void live_h_12(double *state, double *unused, double *out_7538694857600801567);
void live_H_12(double *state, double *unused, double *out_7892326188974872216);
void live_h_31(double *state, double *unused, double *out_7913255707812459020);
void live_H_31(double *state, double *unused, double *out_5231656203940770882);
void live_h_32(double *state, double *unused, double *out_6513779334431292886);
void live_H_32(double *state, double *unused, double *out_7309537563138681095);
void live_h_13(double *state, double *unused, double *out_2868278194920684367);
void live_H_13(double *state, double *unused, double *out_8811383553922433037);
void live_h_14(double *state, double *unused, double *out_2479120142832280908);
void live_H_14(double *state, double *unused, double *out_3278018773858857687);
void live_h_19(double *state, double *unused, double *out_4498204190364601756);
void live_H_19(double *state, double *unused, double *out_1154661077243573494);
void live_predict(double *in_x, double *in_P, double *in_Q, double dt);
}