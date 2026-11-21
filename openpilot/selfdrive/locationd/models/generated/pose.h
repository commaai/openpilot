#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_6544138591762530502);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_5883871402494837793);
void pose_H_mod_fun(double *state, double *out_7964293451172914980);
void pose_f_fun(double *state, double dt, double *out_7677670493198828015);
void pose_F_fun(double *state, double dt, double *out_2599667736124097937);
void pose_h_4(double *state, double *unused, double *out_6486377135734027290);
void pose_H_4(double *state, double *unused, double *out_4956672416897748544);
void pose_h_10(double *state, double *unused, double *out_2228325365749495789);
void pose_H_10(double *state, double *unused, double *out_6859217750491902718);
void pose_h_13(double *state, double *unused, double *out_4348549909198884608);
void pose_H_13(double *state, double *unused, double *out_1744398591565415743);
void pose_h_14(double *state, double *unused, double *out_5034700456740961966);
void pose_H_14(double *state, double *unused, double *out_8039460849193120840);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}