#include "pose.h"

namespace {
#define DIM 18
#define EDIM 18
#define MEDIM 18
typedef void (*Hfun)(double *, double *, double *);
const static double MAHA_THRESH_4 = 7.814727903251177;
const static double MAHA_THRESH_10 = 7.814727903251177;
const static double MAHA_THRESH_13 = 7.814727903251177;
const static double MAHA_THRESH_14 = 7.814727903251177;

/******************************************************************************
 *                      Code generated with SymPy 1.14.0                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_4069511595981512906) {
   out_4069511595981512906[0] = delta_x[0] + nom_x[0];
   out_4069511595981512906[1] = delta_x[1] + nom_x[1];
   out_4069511595981512906[2] = delta_x[2] + nom_x[2];
   out_4069511595981512906[3] = delta_x[3] + nom_x[3];
   out_4069511595981512906[4] = delta_x[4] + nom_x[4];
   out_4069511595981512906[5] = delta_x[5] + nom_x[5];
   out_4069511595981512906[6] = delta_x[6] + nom_x[6];
   out_4069511595981512906[7] = delta_x[7] + nom_x[7];
   out_4069511595981512906[8] = delta_x[8] + nom_x[8];
   out_4069511595981512906[9] = delta_x[9] + nom_x[9];
   out_4069511595981512906[10] = delta_x[10] + nom_x[10];
   out_4069511595981512906[11] = delta_x[11] + nom_x[11];
   out_4069511595981512906[12] = delta_x[12] + nom_x[12];
   out_4069511595981512906[13] = delta_x[13] + nom_x[13];
   out_4069511595981512906[14] = delta_x[14] + nom_x[14];
   out_4069511595981512906[15] = delta_x[15] + nom_x[15];
   out_4069511595981512906[16] = delta_x[16] + nom_x[16];
   out_4069511595981512906[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_6977798052227057242) {
   out_6977798052227057242[0] = -nom_x[0] + true_x[0];
   out_6977798052227057242[1] = -nom_x[1] + true_x[1];
   out_6977798052227057242[2] = -nom_x[2] + true_x[2];
   out_6977798052227057242[3] = -nom_x[3] + true_x[3];
   out_6977798052227057242[4] = -nom_x[4] + true_x[4];
   out_6977798052227057242[5] = -nom_x[5] + true_x[5];
   out_6977798052227057242[6] = -nom_x[6] + true_x[6];
   out_6977798052227057242[7] = -nom_x[7] + true_x[7];
   out_6977798052227057242[8] = -nom_x[8] + true_x[8];
   out_6977798052227057242[9] = -nom_x[9] + true_x[9];
   out_6977798052227057242[10] = -nom_x[10] + true_x[10];
   out_6977798052227057242[11] = -nom_x[11] + true_x[11];
   out_6977798052227057242[12] = -nom_x[12] + true_x[12];
   out_6977798052227057242[13] = -nom_x[13] + true_x[13];
   out_6977798052227057242[14] = -nom_x[14] + true_x[14];
   out_6977798052227057242[15] = -nom_x[15] + true_x[15];
   out_6977798052227057242[16] = -nom_x[16] + true_x[16];
   out_6977798052227057242[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_2633911611712179105) {
   out_2633911611712179105[0] = 1.0;
   out_2633911611712179105[1] = 0.0;
   out_2633911611712179105[2] = 0.0;
   out_2633911611712179105[3] = 0.0;
   out_2633911611712179105[4] = 0.0;
   out_2633911611712179105[5] = 0.0;
   out_2633911611712179105[6] = 0.0;
   out_2633911611712179105[7] = 0.0;
   out_2633911611712179105[8] = 0.0;
   out_2633911611712179105[9] = 0.0;
   out_2633911611712179105[10] = 0.0;
   out_2633911611712179105[11] = 0.0;
   out_2633911611712179105[12] = 0.0;
   out_2633911611712179105[13] = 0.0;
   out_2633911611712179105[14] = 0.0;
   out_2633911611712179105[15] = 0.0;
   out_2633911611712179105[16] = 0.0;
   out_2633911611712179105[17] = 0.0;
   out_2633911611712179105[18] = 0.0;
   out_2633911611712179105[19] = 1.0;
   out_2633911611712179105[20] = 0.0;
   out_2633911611712179105[21] = 0.0;
   out_2633911611712179105[22] = 0.0;
   out_2633911611712179105[23] = 0.0;
   out_2633911611712179105[24] = 0.0;
   out_2633911611712179105[25] = 0.0;
   out_2633911611712179105[26] = 0.0;
   out_2633911611712179105[27] = 0.0;
   out_2633911611712179105[28] = 0.0;
   out_2633911611712179105[29] = 0.0;
   out_2633911611712179105[30] = 0.0;
   out_2633911611712179105[31] = 0.0;
   out_2633911611712179105[32] = 0.0;
   out_2633911611712179105[33] = 0.0;
   out_2633911611712179105[34] = 0.0;
   out_2633911611712179105[35] = 0.0;
   out_2633911611712179105[36] = 0.0;
   out_2633911611712179105[37] = 0.0;
   out_2633911611712179105[38] = 1.0;
   out_2633911611712179105[39] = 0.0;
   out_2633911611712179105[40] = 0.0;
   out_2633911611712179105[41] = 0.0;
   out_2633911611712179105[42] = 0.0;
   out_2633911611712179105[43] = 0.0;
   out_2633911611712179105[44] = 0.0;
   out_2633911611712179105[45] = 0.0;
   out_2633911611712179105[46] = 0.0;
   out_2633911611712179105[47] = 0.0;
   out_2633911611712179105[48] = 0.0;
   out_2633911611712179105[49] = 0.0;
   out_2633911611712179105[50] = 0.0;
   out_2633911611712179105[51] = 0.0;
   out_2633911611712179105[52] = 0.0;
   out_2633911611712179105[53] = 0.0;
   out_2633911611712179105[54] = 0.0;
   out_2633911611712179105[55] = 0.0;
   out_2633911611712179105[56] = 0.0;
   out_2633911611712179105[57] = 1.0;
   out_2633911611712179105[58] = 0.0;
   out_2633911611712179105[59] = 0.0;
   out_2633911611712179105[60] = 0.0;
   out_2633911611712179105[61] = 0.0;
   out_2633911611712179105[62] = 0.0;
   out_2633911611712179105[63] = 0.0;
   out_2633911611712179105[64] = 0.0;
   out_2633911611712179105[65] = 0.0;
   out_2633911611712179105[66] = 0.0;
   out_2633911611712179105[67] = 0.0;
   out_2633911611712179105[68] = 0.0;
   out_2633911611712179105[69] = 0.0;
   out_2633911611712179105[70] = 0.0;
   out_2633911611712179105[71] = 0.0;
   out_2633911611712179105[72] = 0.0;
   out_2633911611712179105[73] = 0.0;
   out_2633911611712179105[74] = 0.0;
   out_2633911611712179105[75] = 0.0;
   out_2633911611712179105[76] = 1.0;
   out_2633911611712179105[77] = 0.0;
   out_2633911611712179105[78] = 0.0;
   out_2633911611712179105[79] = 0.0;
   out_2633911611712179105[80] = 0.0;
   out_2633911611712179105[81] = 0.0;
   out_2633911611712179105[82] = 0.0;
   out_2633911611712179105[83] = 0.0;
   out_2633911611712179105[84] = 0.0;
   out_2633911611712179105[85] = 0.0;
   out_2633911611712179105[86] = 0.0;
   out_2633911611712179105[87] = 0.0;
   out_2633911611712179105[88] = 0.0;
   out_2633911611712179105[89] = 0.0;
   out_2633911611712179105[90] = 0.0;
   out_2633911611712179105[91] = 0.0;
   out_2633911611712179105[92] = 0.0;
   out_2633911611712179105[93] = 0.0;
   out_2633911611712179105[94] = 0.0;
   out_2633911611712179105[95] = 1.0;
   out_2633911611712179105[96] = 0.0;
   out_2633911611712179105[97] = 0.0;
   out_2633911611712179105[98] = 0.0;
   out_2633911611712179105[99] = 0.0;
   out_2633911611712179105[100] = 0.0;
   out_2633911611712179105[101] = 0.0;
   out_2633911611712179105[102] = 0.0;
   out_2633911611712179105[103] = 0.0;
   out_2633911611712179105[104] = 0.0;
   out_2633911611712179105[105] = 0.0;
   out_2633911611712179105[106] = 0.0;
   out_2633911611712179105[107] = 0.0;
   out_2633911611712179105[108] = 0.0;
   out_2633911611712179105[109] = 0.0;
   out_2633911611712179105[110] = 0.0;
   out_2633911611712179105[111] = 0.0;
   out_2633911611712179105[112] = 0.0;
   out_2633911611712179105[113] = 0.0;
   out_2633911611712179105[114] = 1.0;
   out_2633911611712179105[115] = 0.0;
   out_2633911611712179105[116] = 0.0;
   out_2633911611712179105[117] = 0.0;
   out_2633911611712179105[118] = 0.0;
   out_2633911611712179105[119] = 0.0;
   out_2633911611712179105[120] = 0.0;
   out_2633911611712179105[121] = 0.0;
   out_2633911611712179105[122] = 0.0;
   out_2633911611712179105[123] = 0.0;
   out_2633911611712179105[124] = 0.0;
   out_2633911611712179105[125] = 0.0;
   out_2633911611712179105[126] = 0.0;
   out_2633911611712179105[127] = 0.0;
   out_2633911611712179105[128] = 0.0;
   out_2633911611712179105[129] = 0.0;
   out_2633911611712179105[130] = 0.0;
   out_2633911611712179105[131] = 0.0;
   out_2633911611712179105[132] = 0.0;
   out_2633911611712179105[133] = 1.0;
   out_2633911611712179105[134] = 0.0;
   out_2633911611712179105[135] = 0.0;
   out_2633911611712179105[136] = 0.0;
   out_2633911611712179105[137] = 0.0;
   out_2633911611712179105[138] = 0.0;
   out_2633911611712179105[139] = 0.0;
   out_2633911611712179105[140] = 0.0;
   out_2633911611712179105[141] = 0.0;
   out_2633911611712179105[142] = 0.0;
   out_2633911611712179105[143] = 0.0;
   out_2633911611712179105[144] = 0.0;
   out_2633911611712179105[145] = 0.0;
   out_2633911611712179105[146] = 0.0;
   out_2633911611712179105[147] = 0.0;
   out_2633911611712179105[148] = 0.0;
   out_2633911611712179105[149] = 0.0;
   out_2633911611712179105[150] = 0.0;
   out_2633911611712179105[151] = 0.0;
   out_2633911611712179105[152] = 1.0;
   out_2633911611712179105[153] = 0.0;
   out_2633911611712179105[154] = 0.0;
   out_2633911611712179105[155] = 0.0;
   out_2633911611712179105[156] = 0.0;
   out_2633911611712179105[157] = 0.0;
   out_2633911611712179105[158] = 0.0;
   out_2633911611712179105[159] = 0.0;
   out_2633911611712179105[160] = 0.0;
   out_2633911611712179105[161] = 0.0;
   out_2633911611712179105[162] = 0.0;
   out_2633911611712179105[163] = 0.0;
   out_2633911611712179105[164] = 0.0;
   out_2633911611712179105[165] = 0.0;
   out_2633911611712179105[166] = 0.0;
   out_2633911611712179105[167] = 0.0;
   out_2633911611712179105[168] = 0.0;
   out_2633911611712179105[169] = 0.0;
   out_2633911611712179105[170] = 0.0;
   out_2633911611712179105[171] = 1.0;
   out_2633911611712179105[172] = 0.0;
   out_2633911611712179105[173] = 0.0;
   out_2633911611712179105[174] = 0.0;
   out_2633911611712179105[175] = 0.0;
   out_2633911611712179105[176] = 0.0;
   out_2633911611712179105[177] = 0.0;
   out_2633911611712179105[178] = 0.0;
   out_2633911611712179105[179] = 0.0;
   out_2633911611712179105[180] = 0.0;
   out_2633911611712179105[181] = 0.0;
   out_2633911611712179105[182] = 0.0;
   out_2633911611712179105[183] = 0.0;
   out_2633911611712179105[184] = 0.0;
   out_2633911611712179105[185] = 0.0;
   out_2633911611712179105[186] = 0.0;
   out_2633911611712179105[187] = 0.0;
   out_2633911611712179105[188] = 0.0;
   out_2633911611712179105[189] = 0.0;
   out_2633911611712179105[190] = 1.0;
   out_2633911611712179105[191] = 0.0;
   out_2633911611712179105[192] = 0.0;
   out_2633911611712179105[193] = 0.0;
   out_2633911611712179105[194] = 0.0;
   out_2633911611712179105[195] = 0.0;
   out_2633911611712179105[196] = 0.0;
   out_2633911611712179105[197] = 0.0;
   out_2633911611712179105[198] = 0.0;
   out_2633911611712179105[199] = 0.0;
   out_2633911611712179105[200] = 0.0;
   out_2633911611712179105[201] = 0.0;
   out_2633911611712179105[202] = 0.0;
   out_2633911611712179105[203] = 0.0;
   out_2633911611712179105[204] = 0.0;
   out_2633911611712179105[205] = 0.0;
   out_2633911611712179105[206] = 0.0;
   out_2633911611712179105[207] = 0.0;
   out_2633911611712179105[208] = 0.0;
   out_2633911611712179105[209] = 1.0;
   out_2633911611712179105[210] = 0.0;
   out_2633911611712179105[211] = 0.0;
   out_2633911611712179105[212] = 0.0;
   out_2633911611712179105[213] = 0.0;
   out_2633911611712179105[214] = 0.0;
   out_2633911611712179105[215] = 0.0;
   out_2633911611712179105[216] = 0.0;
   out_2633911611712179105[217] = 0.0;
   out_2633911611712179105[218] = 0.0;
   out_2633911611712179105[219] = 0.0;
   out_2633911611712179105[220] = 0.0;
   out_2633911611712179105[221] = 0.0;
   out_2633911611712179105[222] = 0.0;
   out_2633911611712179105[223] = 0.0;
   out_2633911611712179105[224] = 0.0;
   out_2633911611712179105[225] = 0.0;
   out_2633911611712179105[226] = 0.0;
   out_2633911611712179105[227] = 0.0;
   out_2633911611712179105[228] = 1.0;
   out_2633911611712179105[229] = 0.0;
   out_2633911611712179105[230] = 0.0;
   out_2633911611712179105[231] = 0.0;
   out_2633911611712179105[232] = 0.0;
   out_2633911611712179105[233] = 0.0;
   out_2633911611712179105[234] = 0.0;
   out_2633911611712179105[235] = 0.0;
   out_2633911611712179105[236] = 0.0;
   out_2633911611712179105[237] = 0.0;
   out_2633911611712179105[238] = 0.0;
   out_2633911611712179105[239] = 0.0;
   out_2633911611712179105[240] = 0.0;
   out_2633911611712179105[241] = 0.0;
   out_2633911611712179105[242] = 0.0;
   out_2633911611712179105[243] = 0.0;
   out_2633911611712179105[244] = 0.0;
   out_2633911611712179105[245] = 0.0;
   out_2633911611712179105[246] = 0.0;
   out_2633911611712179105[247] = 1.0;
   out_2633911611712179105[248] = 0.0;
   out_2633911611712179105[249] = 0.0;
   out_2633911611712179105[250] = 0.0;
   out_2633911611712179105[251] = 0.0;
   out_2633911611712179105[252] = 0.0;
   out_2633911611712179105[253] = 0.0;
   out_2633911611712179105[254] = 0.0;
   out_2633911611712179105[255] = 0.0;
   out_2633911611712179105[256] = 0.0;
   out_2633911611712179105[257] = 0.0;
   out_2633911611712179105[258] = 0.0;
   out_2633911611712179105[259] = 0.0;
   out_2633911611712179105[260] = 0.0;
   out_2633911611712179105[261] = 0.0;
   out_2633911611712179105[262] = 0.0;
   out_2633911611712179105[263] = 0.0;
   out_2633911611712179105[264] = 0.0;
   out_2633911611712179105[265] = 0.0;
   out_2633911611712179105[266] = 1.0;
   out_2633911611712179105[267] = 0.0;
   out_2633911611712179105[268] = 0.0;
   out_2633911611712179105[269] = 0.0;
   out_2633911611712179105[270] = 0.0;
   out_2633911611712179105[271] = 0.0;
   out_2633911611712179105[272] = 0.0;
   out_2633911611712179105[273] = 0.0;
   out_2633911611712179105[274] = 0.0;
   out_2633911611712179105[275] = 0.0;
   out_2633911611712179105[276] = 0.0;
   out_2633911611712179105[277] = 0.0;
   out_2633911611712179105[278] = 0.0;
   out_2633911611712179105[279] = 0.0;
   out_2633911611712179105[280] = 0.0;
   out_2633911611712179105[281] = 0.0;
   out_2633911611712179105[282] = 0.0;
   out_2633911611712179105[283] = 0.0;
   out_2633911611712179105[284] = 0.0;
   out_2633911611712179105[285] = 1.0;
   out_2633911611712179105[286] = 0.0;
   out_2633911611712179105[287] = 0.0;
   out_2633911611712179105[288] = 0.0;
   out_2633911611712179105[289] = 0.0;
   out_2633911611712179105[290] = 0.0;
   out_2633911611712179105[291] = 0.0;
   out_2633911611712179105[292] = 0.0;
   out_2633911611712179105[293] = 0.0;
   out_2633911611712179105[294] = 0.0;
   out_2633911611712179105[295] = 0.0;
   out_2633911611712179105[296] = 0.0;
   out_2633911611712179105[297] = 0.0;
   out_2633911611712179105[298] = 0.0;
   out_2633911611712179105[299] = 0.0;
   out_2633911611712179105[300] = 0.0;
   out_2633911611712179105[301] = 0.0;
   out_2633911611712179105[302] = 0.0;
   out_2633911611712179105[303] = 0.0;
   out_2633911611712179105[304] = 1.0;
   out_2633911611712179105[305] = 0.0;
   out_2633911611712179105[306] = 0.0;
   out_2633911611712179105[307] = 0.0;
   out_2633911611712179105[308] = 0.0;
   out_2633911611712179105[309] = 0.0;
   out_2633911611712179105[310] = 0.0;
   out_2633911611712179105[311] = 0.0;
   out_2633911611712179105[312] = 0.0;
   out_2633911611712179105[313] = 0.0;
   out_2633911611712179105[314] = 0.0;
   out_2633911611712179105[315] = 0.0;
   out_2633911611712179105[316] = 0.0;
   out_2633911611712179105[317] = 0.0;
   out_2633911611712179105[318] = 0.0;
   out_2633911611712179105[319] = 0.0;
   out_2633911611712179105[320] = 0.0;
   out_2633911611712179105[321] = 0.0;
   out_2633911611712179105[322] = 0.0;
   out_2633911611712179105[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_7060402792772463216) {
   out_7060402792772463216[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_7060402792772463216[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_7060402792772463216[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_7060402792772463216[3] = dt*state[12] + state[3];
   out_7060402792772463216[4] = dt*state[13] + state[4];
   out_7060402792772463216[5] = dt*state[14] + state[5];
   out_7060402792772463216[6] = state[6];
   out_7060402792772463216[7] = state[7];
   out_7060402792772463216[8] = state[8];
   out_7060402792772463216[9] = state[9];
   out_7060402792772463216[10] = state[10];
   out_7060402792772463216[11] = state[11];
   out_7060402792772463216[12] = state[12];
   out_7060402792772463216[13] = state[13];
   out_7060402792772463216[14] = state[14];
   out_7060402792772463216[15] = state[15];
   out_7060402792772463216[16] = state[16];
   out_7060402792772463216[17] = state[17];
}
void F_fun(double *state, double dt, double *out_4609062820950943518) {
   out_4609062820950943518[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_4609062820950943518[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_4609062820950943518[2] = 0;
   out_4609062820950943518[3] = 0;
   out_4609062820950943518[4] = 0;
   out_4609062820950943518[5] = 0;
   out_4609062820950943518[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_4609062820950943518[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_4609062820950943518[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_4609062820950943518[9] = 0;
   out_4609062820950943518[10] = 0;
   out_4609062820950943518[11] = 0;
   out_4609062820950943518[12] = 0;
   out_4609062820950943518[13] = 0;
   out_4609062820950943518[14] = 0;
   out_4609062820950943518[15] = 0;
   out_4609062820950943518[16] = 0;
   out_4609062820950943518[17] = 0;
   out_4609062820950943518[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_4609062820950943518[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_4609062820950943518[20] = 0;
   out_4609062820950943518[21] = 0;
   out_4609062820950943518[22] = 0;
   out_4609062820950943518[23] = 0;
   out_4609062820950943518[24] = 0;
   out_4609062820950943518[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_4609062820950943518[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_4609062820950943518[27] = 0;
   out_4609062820950943518[28] = 0;
   out_4609062820950943518[29] = 0;
   out_4609062820950943518[30] = 0;
   out_4609062820950943518[31] = 0;
   out_4609062820950943518[32] = 0;
   out_4609062820950943518[33] = 0;
   out_4609062820950943518[34] = 0;
   out_4609062820950943518[35] = 0;
   out_4609062820950943518[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_4609062820950943518[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_4609062820950943518[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_4609062820950943518[39] = 0;
   out_4609062820950943518[40] = 0;
   out_4609062820950943518[41] = 0;
   out_4609062820950943518[42] = 0;
   out_4609062820950943518[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_4609062820950943518[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_4609062820950943518[45] = 0;
   out_4609062820950943518[46] = 0;
   out_4609062820950943518[47] = 0;
   out_4609062820950943518[48] = 0;
   out_4609062820950943518[49] = 0;
   out_4609062820950943518[50] = 0;
   out_4609062820950943518[51] = 0;
   out_4609062820950943518[52] = 0;
   out_4609062820950943518[53] = 0;
   out_4609062820950943518[54] = 0;
   out_4609062820950943518[55] = 0;
   out_4609062820950943518[56] = 0;
   out_4609062820950943518[57] = 1;
   out_4609062820950943518[58] = 0;
   out_4609062820950943518[59] = 0;
   out_4609062820950943518[60] = 0;
   out_4609062820950943518[61] = 0;
   out_4609062820950943518[62] = 0;
   out_4609062820950943518[63] = 0;
   out_4609062820950943518[64] = 0;
   out_4609062820950943518[65] = 0;
   out_4609062820950943518[66] = dt;
   out_4609062820950943518[67] = 0;
   out_4609062820950943518[68] = 0;
   out_4609062820950943518[69] = 0;
   out_4609062820950943518[70] = 0;
   out_4609062820950943518[71] = 0;
   out_4609062820950943518[72] = 0;
   out_4609062820950943518[73] = 0;
   out_4609062820950943518[74] = 0;
   out_4609062820950943518[75] = 0;
   out_4609062820950943518[76] = 1;
   out_4609062820950943518[77] = 0;
   out_4609062820950943518[78] = 0;
   out_4609062820950943518[79] = 0;
   out_4609062820950943518[80] = 0;
   out_4609062820950943518[81] = 0;
   out_4609062820950943518[82] = 0;
   out_4609062820950943518[83] = 0;
   out_4609062820950943518[84] = 0;
   out_4609062820950943518[85] = dt;
   out_4609062820950943518[86] = 0;
   out_4609062820950943518[87] = 0;
   out_4609062820950943518[88] = 0;
   out_4609062820950943518[89] = 0;
   out_4609062820950943518[90] = 0;
   out_4609062820950943518[91] = 0;
   out_4609062820950943518[92] = 0;
   out_4609062820950943518[93] = 0;
   out_4609062820950943518[94] = 0;
   out_4609062820950943518[95] = 1;
   out_4609062820950943518[96] = 0;
   out_4609062820950943518[97] = 0;
   out_4609062820950943518[98] = 0;
   out_4609062820950943518[99] = 0;
   out_4609062820950943518[100] = 0;
   out_4609062820950943518[101] = 0;
   out_4609062820950943518[102] = 0;
   out_4609062820950943518[103] = 0;
   out_4609062820950943518[104] = dt;
   out_4609062820950943518[105] = 0;
   out_4609062820950943518[106] = 0;
   out_4609062820950943518[107] = 0;
   out_4609062820950943518[108] = 0;
   out_4609062820950943518[109] = 0;
   out_4609062820950943518[110] = 0;
   out_4609062820950943518[111] = 0;
   out_4609062820950943518[112] = 0;
   out_4609062820950943518[113] = 0;
   out_4609062820950943518[114] = 1;
   out_4609062820950943518[115] = 0;
   out_4609062820950943518[116] = 0;
   out_4609062820950943518[117] = 0;
   out_4609062820950943518[118] = 0;
   out_4609062820950943518[119] = 0;
   out_4609062820950943518[120] = 0;
   out_4609062820950943518[121] = 0;
   out_4609062820950943518[122] = 0;
   out_4609062820950943518[123] = 0;
   out_4609062820950943518[124] = 0;
   out_4609062820950943518[125] = 0;
   out_4609062820950943518[126] = 0;
   out_4609062820950943518[127] = 0;
   out_4609062820950943518[128] = 0;
   out_4609062820950943518[129] = 0;
   out_4609062820950943518[130] = 0;
   out_4609062820950943518[131] = 0;
   out_4609062820950943518[132] = 0;
   out_4609062820950943518[133] = 1;
   out_4609062820950943518[134] = 0;
   out_4609062820950943518[135] = 0;
   out_4609062820950943518[136] = 0;
   out_4609062820950943518[137] = 0;
   out_4609062820950943518[138] = 0;
   out_4609062820950943518[139] = 0;
   out_4609062820950943518[140] = 0;
   out_4609062820950943518[141] = 0;
   out_4609062820950943518[142] = 0;
   out_4609062820950943518[143] = 0;
   out_4609062820950943518[144] = 0;
   out_4609062820950943518[145] = 0;
   out_4609062820950943518[146] = 0;
   out_4609062820950943518[147] = 0;
   out_4609062820950943518[148] = 0;
   out_4609062820950943518[149] = 0;
   out_4609062820950943518[150] = 0;
   out_4609062820950943518[151] = 0;
   out_4609062820950943518[152] = 1;
   out_4609062820950943518[153] = 0;
   out_4609062820950943518[154] = 0;
   out_4609062820950943518[155] = 0;
   out_4609062820950943518[156] = 0;
   out_4609062820950943518[157] = 0;
   out_4609062820950943518[158] = 0;
   out_4609062820950943518[159] = 0;
   out_4609062820950943518[160] = 0;
   out_4609062820950943518[161] = 0;
   out_4609062820950943518[162] = 0;
   out_4609062820950943518[163] = 0;
   out_4609062820950943518[164] = 0;
   out_4609062820950943518[165] = 0;
   out_4609062820950943518[166] = 0;
   out_4609062820950943518[167] = 0;
   out_4609062820950943518[168] = 0;
   out_4609062820950943518[169] = 0;
   out_4609062820950943518[170] = 0;
   out_4609062820950943518[171] = 1;
   out_4609062820950943518[172] = 0;
   out_4609062820950943518[173] = 0;
   out_4609062820950943518[174] = 0;
   out_4609062820950943518[175] = 0;
   out_4609062820950943518[176] = 0;
   out_4609062820950943518[177] = 0;
   out_4609062820950943518[178] = 0;
   out_4609062820950943518[179] = 0;
   out_4609062820950943518[180] = 0;
   out_4609062820950943518[181] = 0;
   out_4609062820950943518[182] = 0;
   out_4609062820950943518[183] = 0;
   out_4609062820950943518[184] = 0;
   out_4609062820950943518[185] = 0;
   out_4609062820950943518[186] = 0;
   out_4609062820950943518[187] = 0;
   out_4609062820950943518[188] = 0;
   out_4609062820950943518[189] = 0;
   out_4609062820950943518[190] = 1;
   out_4609062820950943518[191] = 0;
   out_4609062820950943518[192] = 0;
   out_4609062820950943518[193] = 0;
   out_4609062820950943518[194] = 0;
   out_4609062820950943518[195] = 0;
   out_4609062820950943518[196] = 0;
   out_4609062820950943518[197] = 0;
   out_4609062820950943518[198] = 0;
   out_4609062820950943518[199] = 0;
   out_4609062820950943518[200] = 0;
   out_4609062820950943518[201] = 0;
   out_4609062820950943518[202] = 0;
   out_4609062820950943518[203] = 0;
   out_4609062820950943518[204] = 0;
   out_4609062820950943518[205] = 0;
   out_4609062820950943518[206] = 0;
   out_4609062820950943518[207] = 0;
   out_4609062820950943518[208] = 0;
   out_4609062820950943518[209] = 1;
   out_4609062820950943518[210] = 0;
   out_4609062820950943518[211] = 0;
   out_4609062820950943518[212] = 0;
   out_4609062820950943518[213] = 0;
   out_4609062820950943518[214] = 0;
   out_4609062820950943518[215] = 0;
   out_4609062820950943518[216] = 0;
   out_4609062820950943518[217] = 0;
   out_4609062820950943518[218] = 0;
   out_4609062820950943518[219] = 0;
   out_4609062820950943518[220] = 0;
   out_4609062820950943518[221] = 0;
   out_4609062820950943518[222] = 0;
   out_4609062820950943518[223] = 0;
   out_4609062820950943518[224] = 0;
   out_4609062820950943518[225] = 0;
   out_4609062820950943518[226] = 0;
   out_4609062820950943518[227] = 0;
   out_4609062820950943518[228] = 1;
   out_4609062820950943518[229] = 0;
   out_4609062820950943518[230] = 0;
   out_4609062820950943518[231] = 0;
   out_4609062820950943518[232] = 0;
   out_4609062820950943518[233] = 0;
   out_4609062820950943518[234] = 0;
   out_4609062820950943518[235] = 0;
   out_4609062820950943518[236] = 0;
   out_4609062820950943518[237] = 0;
   out_4609062820950943518[238] = 0;
   out_4609062820950943518[239] = 0;
   out_4609062820950943518[240] = 0;
   out_4609062820950943518[241] = 0;
   out_4609062820950943518[242] = 0;
   out_4609062820950943518[243] = 0;
   out_4609062820950943518[244] = 0;
   out_4609062820950943518[245] = 0;
   out_4609062820950943518[246] = 0;
   out_4609062820950943518[247] = 1;
   out_4609062820950943518[248] = 0;
   out_4609062820950943518[249] = 0;
   out_4609062820950943518[250] = 0;
   out_4609062820950943518[251] = 0;
   out_4609062820950943518[252] = 0;
   out_4609062820950943518[253] = 0;
   out_4609062820950943518[254] = 0;
   out_4609062820950943518[255] = 0;
   out_4609062820950943518[256] = 0;
   out_4609062820950943518[257] = 0;
   out_4609062820950943518[258] = 0;
   out_4609062820950943518[259] = 0;
   out_4609062820950943518[260] = 0;
   out_4609062820950943518[261] = 0;
   out_4609062820950943518[262] = 0;
   out_4609062820950943518[263] = 0;
   out_4609062820950943518[264] = 0;
   out_4609062820950943518[265] = 0;
   out_4609062820950943518[266] = 1;
   out_4609062820950943518[267] = 0;
   out_4609062820950943518[268] = 0;
   out_4609062820950943518[269] = 0;
   out_4609062820950943518[270] = 0;
   out_4609062820950943518[271] = 0;
   out_4609062820950943518[272] = 0;
   out_4609062820950943518[273] = 0;
   out_4609062820950943518[274] = 0;
   out_4609062820950943518[275] = 0;
   out_4609062820950943518[276] = 0;
   out_4609062820950943518[277] = 0;
   out_4609062820950943518[278] = 0;
   out_4609062820950943518[279] = 0;
   out_4609062820950943518[280] = 0;
   out_4609062820950943518[281] = 0;
   out_4609062820950943518[282] = 0;
   out_4609062820950943518[283] = 0;
   out_4609062820950943518[284] = 0;
   out_4609062820950943518[285] = 1;
   out_4609062820950943518[286] = 0;
   out_4609062820950943518[287] = 0;
   out_4609062820950943518[288] = 0;
   out_4609062820950943518[289] = 0;
   out_4609062820950943518[290] = 0;
   out_4609062820950943518[291] = 0;
   out_4609062820950943518[292] = 0;
   out_4609062820950943518[293] = 0;
   out_4609062820950943518[294] = 0;
   out_4609062820950943518[295] = 0;
   out_4609062820950943518[296] = 0;
   out_4609062820950943518[297] = 0;
   out_4609062820950943518[298] = 0;
   out_4609062820950943518[299] = 0;
   out_4609062820950943518[300] = 0;
   out_4609062820950943518[301] = 0;
   out_4609062820950943518[302] = 0;
   out_4609062820950943518[303] = 0;
   out_4609062820950943518[304] = 1;
   out_4609062820950943518[305] = 0;
   out_4609062820950943518[306] = 0;
   out_4609062820950943518[307] = 0;
   out_4609062820950943518[308] = 0;
   out_4609062820950943518[309] = 0;
   out_4609062820950943518[310] = 0;
   out_4609062820950943518[311] = 0;
   out_4609062820950943518[312] = 0;
   out_4609062820950943518[313] = 0;
   out_4609062820950943518[314] = 0;
   out_4609062820950943518[315] = 0;
   out_4609062820950943518[316] = 0;
   out_4609062820950943518[317] = 0;
   out_4609062820950943518[318] = 0;
   out_4609062820950943518[319] = 0;
   out_4609062820950943518[320] = 0;
   out_4609062820950943518[321] = 0;
   out_4609062820950943518[322] = 0;
   out_4609062820950943518[323] = 1;
}
void h_4(double *state, double *unused, double *out_1304819822248380504) {
   out_1304819822248380504[0] = state[6] + state[9];
   out_1304819822248380504[1] = state[7] + state[10];
   out_1304819822248380504[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_8481993418464137826) {
   out_8481993418464137826[0] = 0;
   out_8481993418464137826[1] = 0;
   out_8481993418464137826[2] = 0;
   out_8481993418464137826[3] = 0;
   out_8481993418464137826[4] = 0;
   out_8481993418464137826[5] = 0;
   out_8481993418464137826[6] = 1;
   out_8481993418464137826[7] = 0;
   out_8481993418464137826[8] = 0;
   out_8481993418464137826[9] = 1;
   out_8481993418464137826[10] = 0;
   out_8481993418464137826[11] = 0;
   out_8481993418464137826[12] = 0;
   out_8481993418464137826[13] = 0;
   out_8481993418464137826[14] = 0;
   out_8481993418464137826[15] = 0;
   out_8481993418464137826[16] = 0;
   out_8481993418464137826[17] = 0;
   out_8481993418464137826[18] = 0;
   out_8481993418464137826[19] = 0;
   out_8481993418464137826[20] = 0;
   out_8481993418464137826[21] = 0;
   out_8481993418464137826[22] = 0;
   out_8481993418464137826[23] = 0;
   out_8481993418464137826[24] = 0;
   out_8481993418464137826[25] = 1;
   out_8481993418464137826[26] = 0;
   out_8481993418464137826[27] = 0;
   out_8481993418464137826[28] = 1;
   out_8481993418464137826[29] = 0;
   out_8481993418464137826[30] = 0;
   out_8481993418464137826[31] = 0;
   out_8481993418464137826[32] = 0;
   out_8481993418464137826[33] = 0;
   out_8481993418464137826[34] = 0;
   out_8481993418464137826[35] = 0;
   out_8481993418464137826[36] = 0;
   out_8481993418464137826[37] = 0;
   out_8481993418464137826[38] = 0;
   out_8481993418464137826[39] = 0;
   out_8481993418464137826[40] = 0;
   out_8481993418464137826[41] = 0;
   out_8481993418464137826[42] = 0;
   out_8481993418464137826[43] = 0;
   out_8481993418464137826[44] = 1;
   out_8481993418464137826[45] = 0;
   out_8481993418464137826[46] = 0;
   out_8481993418464137826[47] = 1;
   out_8481993418464137826[48] = 0;
   out_8481993418464137826[49] = 0;
   out_8481993418464137826[50] = 0;
   out_8481993418464137826[51] = 0;
   out_8481993418464137826[52] = 0;
   out_8481993418464137826[53] = 0;
}
void h_10(double *state, double *unused, double *out_3476711082558028639) {
   out_3476711082558028639[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_3476711082558028639[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_3476711082558028639[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_436246387152335361) {
   out_436246387152335361[0] = 0;
   out_436246387152335361[1] = 9.8100000000000005*cos(state[1]);
   out_436246387152335361[2] = 0;
   out_436246387152335361[3] = 0;
   out_436246387152335361[4] = -state[8];
   out_436246387152335361[5] = state[7];
   out_436246387152335361[6] = 0;
   out_436246387152335361[7] = state[5];
   out_436246387152335361[8] = -state[4];
   out_436246387152335361[9] = 0;
   out_436246387152335361[10] = 0;
   out_436246387152335361[11] = 0;
   out_436246387152335361[12] = 1;
   out_436246387152335361[13] = 0;
   out_436246387152335361[14] = 0;
   out_436246387152335361[15] = 1;
   out_436246387152335361[16] = 0;
   out_436246387152335361[17] = 0;
   out_436246387152335361[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_436246387152335361[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_436246387152335361[20] = 0;
   out_436246387152335361[21] = state[8];
   out_436246387152335361[22] = 0;
   out_436246387152335361[23] = -state[6];
   out_436246387152335361[24] = -state[5];
   out_436246387152335361[25] = 0;
   out_436246387152335361[26] = state[3];
   out_436246387152335361[27] = 0;
   out_436246387152335361[28] = 0;
   out_436246387152335361[29] = 0;
   out_436246387152335361[30] = 0;
   out_436246387152335361[31] = 1;
   out_436246387152335361[32] = 0;
   out_436246387152335361[33] = 0;
   out_436246387152335361[34] = 1;
   out_436246387152335361[35] = 0;
   out_436246387152335361[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_436246387152335361[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_436246387152335361[38] = 0;
   out_436246387152335361[39] = -state[7];
   out_436246387152335361[40] = state[6];
   out_436246387152335361[41] = 0;
   out_436246387152335361[42] = state[4];
   out_436246387152335361[43] = -state[3];
   out_436246387152335361[44] = 0;
   out_436246387152335361[45] = 0;
   out_436246387152335361[46] = 0;
   out_436246387152335361[47] = 0;
   out_436246387152335361[48] = 0;
   out_436246387152335361[49] = 0;
   out_436246387152335361[50] = 1;
   out_436246387152335361[51] = 0;
   out_436246387152335361[52] = 0;
   out_436246387152335361[53] = 1;
}
void h_13(double *state, double *unused, double *out_5988377314978994220) {
   out_5988377314978994220[0] = state[3];
   out_5988377314978994220[1] = state[4];
   out_5988377314978994220[2] = state[5];
}
void H_13(double *state, double *unused, double *out_5269719593131805025) {
   out_5269719593131805025[0] = 0;
   out_5269719593131805025[1] = 0;
   out_5269719593131805025[2] = 0;
   out_5269719593131805025[3] = 1;
   out_5269719593131805025[4] = 0;
   out_5269719593131805025[5] = 0;
   out_5269719593131805025[6] = 0;
   out_5269719593131805025[7] = 0;
   out_5269719593131805025[8] = 0;
   out_5269719593131805025[9] = 0;
   out_5269719593131805025[10] = 0;
   out_5269719593131805025[11] = 0;
   out_5269719593131805025[12] = 0;
   out_5269719593131805025[13] = 0;
   out_5269719593131805025[14] = 0;
   out_5269719593131805025[15] = 0;
   out_5269719593131805025[16] = 0;
   out_5269719593131805025[17] = 0;
   out_5269719593131805025[18] = 0;
   out_5269719593131805025[19] = 0;
   out_5269719593131805025[20] = 0;
   out_5269719593131805025[21] = 0;
   out_5269719593131805025[22] = 1;
   out_5269719593131805025[23] = 0;
   out_5269719593131805025[24] = 0;
   out_5269719593131805025[25] = 0;
   out_5269719593131805025[26] = 0;
   out_5269719593131805025[27] = 0;
   out_5269719593131805025[28] = 0;
   out_5269719593131805025[29] = 0;
   out_5269719593131805025[30] = 0;
   out_5269719593131805025[31] = 0;
   out_5269719593131805025[32] = 0;
   out_5269719593131805025[33] = 0;
   out_5269719593131805025[34] = 0;
   out_5269719593131805025[35] = 0;
   out_5269719593131805025[36] = 0;
   out_5269719593131805025[37] = 0;
   out_5269719593131805025[38] = 0;
   out_5269719593131805025[39] = 0;
   out_5269719593131805025[40] = 0;
   out_5269719593131805025[41] = 1;
   out_5269719593131805025[42] = 0;
   out_5269719593131805025[43] = 0;
   out_5269719593131805025[44] = 0;
   out_5269719593131805025[45] = 0;
   out_5269719593131805025[46] = 0;
   out_5269719593131805025[47] = 0;
   out_5269719593131805025[48] = 0;
   out_5269719593131805025[49] = 0;
   out_5269719593131805025[50] = 0;
   out_5269719593131805025[51] = 0;
   out_5269719593131805025[52] = 0;
   out_5269719593131805025[53] = 0;
}
void h_14(double *state, double *unused, double *out_4290951141142713711) {
   out_4290951141142713711[0] = state[6];
   out_4290951141142713711[1] = state[7];
   out_4290951141142713711[2] = state[8];
}
void H_14(double *state, double *unused, double *out_4518752562124653297) {
   out_4518752562124653297[0] = 0;
   out_4518752562124653297[1] = 0;
   out_4518752562124653297[2] = 0;
   out_4518752562124653297[3] = 0;
   out_4518752562124653297[4] = 0;
   out_4518752562124653297[5] = 0;
   out_4518752562124653297[6] = 1;
   out_4518752562124653297[7] = 0;
   out_4518752562124653297[8] = 0;
   out_4518752562124653297[9] = 0;
   out_4518752562124653297[10] = 0;
   out_4518752562124653297[11] = 0;
   out_4518752562124653297[12] = 0;
   out_4518752562124653297[13] = 0;
   out_4518752562124653297[14] = 0;
   out_4518752562124653297[15] = 0;
   out_4518752562124653297[16] = 0;
   out_4518752562124653297[17] = 0;
   out_4518752562124653297[18] = 0;
   out_4518752562124653297[19] = 0;
   out_4518752562124653297[20] = 0;
   out_4518752562124653297[21] = 0;
   out_4518752562124653297[22] = 0;
   out_4518752562124653297[23] = 0;
   out_4518752562124653297[24] = 0;
   out_4518752562124653297[25] = 1;
   out_4518752562124653297[26] = 0;
   out_4518752562124653297[27] = 0;
   out_4518752562124653297[28] = 0;
   out_4518752562124653297[29] = 0;
   out_4518752562124653297[30] = 0;
   out_4518752562124653297[31] = 0;
   out_4518752562124653297[32] = 0;
   out_4518752562124653297[33] = 0;
   out_4518752562124653297[34] = 0;
   out_4518752562124653297[35] = 0;
   out_4518752562124653297[36] = 0;
   out_4518752562124653297[37] = 0;
   out_4518752562124653297[38] = 0;
   out_4518752562124653297[39] = 0;
   out_4518752562124653297[40] = 0;
   out_4518752562124653297[41] = 0;
   out_4518752562124653297[42] = 0;
   out_4518752562124653297[43] = 0;
   out_4518752562124653297[44] = 1;
   out_4518752562124653297[45] = 0;
   out_4518752562124653297[46] = 0;
   out_4518752562124653297[47] = 0;
   out_4518752562124653297[48] = 0;
   out_4518752562124653297[49] = 0;
   out_4518752562124653297[50] = 0;
   out_4518752562124653297[51] = 0;
   out_4518752562124653297[52] = 0;
   out_4518752562124653297[53] = 0;
}
#include <eigen3/Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<double, DIM, DIM, Eigen::RowMajor> DDM;
typedef Eigen::Matrix<double, EDIM, EDIM, Eigen::RowMajor> EEM;
typedef Eigen::Matrix<double, DIM, EDIM, Eigen::RowMajor> DEM;

void predict(double *in_x, double *in_P, double *in_Q, double dt) {
  typedef Eigen::Matrix<double, MEDIM, MEDIM, Eigen::RowMajor> RRM;

  double nx[DIM] = {0};
  double in_F[EDIM*EDIM] = {0};

  // functions from sympy
  f_fun(in_x, dt, nx);
  F_fun(in_x, dt, in_F);


  EEM F(in_F);
  EEM P(in_P);
  EEM Q(in_Q);

  RRM F_main = F.topLeftCorner(MEDIM, MEDIM);
  P.topLeftCorner(MEDIM, MEDIM) = (F_main * P.topLeftCorner(MEDIM, MEDIM)) * F_main.transpose();
  P.topRightCorner(MEDIM, EDIM - MEDIM) = F_main * P.topRightCorner(MEDIM, EDIM - MEDIM);
  P.bottomLeftCorner(EDIM - MEDIM, MEDIM) = P.bottomLeftCorner(EDIM - MEDIM, MEDIM) * F_main.transpose();

  P = P + dt*Q;

  // copy out state
  memcpy(in_x, nx, DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
}

// note: extra_args dim only correct when null space projecting
// otherwise 1
template <int ZDIM, int EADIM, bool MAHA_TEST>
void update(double *in_x, double *in_P, Hfun h_fun, Hfun H_fun, Hfun Hea_fun, double *in_z, double *in_R, double *in_ea, double MAHA_THRESHOLD) {
  typedef Eigen::Matrix<double, ZDIM, ZDIM, Eigen::RowMajor> ZZM;
  typedef Eigen::Matrix<double, ZDIM, DIM, Eigen::RowMajor> ZDM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, EDIM, Eigen::RowMajor> XEM;
  //typedef Eigen::Matrix<double, EDIM, ZDIM, Eigen::RowMajor> EZM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> X1M;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> XXM;

  double in_hx[ZDIM] = {0};
  double in_H[ZDIM * DIM] = {0};
  double in_H_mod[EDIM * DIM] = {0};
  double delta_x[EDIM] = {0};
  double x_new[DIM] = {0};


  // state x, P
  Eigen::Matrix<double, ZDIM, 1> z(in_z);
  EEM P(in_P);
  ZZM pre_R(in_R);

  // functions from sympy
  h_fun(in_x, in_ea, in_hx);
  H_fun(in_x, in_ea, in_H);
  ZDM pre_H(in_H);

  // get y (y = z - hx)
  Eigen::Matrix<double, ZDIM, 1> pre_y(in_hx); pre_y = z - pre_y;
  X1M y; XXM H; XXM R;
  if (Hea_fun){
    typedef Eigen::Matrix<double, ZDIM, EADIM, Eigen::RowMajor> ZAM;
    double in_Hea[ZDIM * EADIM] = {0};
    Hea_fun(in_x, in_ea, in_Hea);
    ZAM Hea(in_Hea);
    XXM A = Hea.transpose().fullPivLu().kernel();


    y = A.transpose() * pre_y;
    H = A.transpose() * pre_H;
    R = A.transpose() * pre_R * A;
  } else {
    y = pre_y;
    H = pre_H;
    R = pre_R;
  }
  // get modified H
  H_mod_fun(in_x, in_H_mod);
  DEM H_mod(in_H_mod);
  XEM H_err = H * H_mod;

  // Do mahalobis distance test
  if (MAHA_TEST){
    XXM a = (H_err * P * H_err.transpose() + R).inverse();
    double maha_dist = y.transpose() * a * y;
    if (maha_dist > MAHA_THRESHOLD){
      R = 1.0e16 * R;
    }
  }

  // Outlier resilient weighting
  double weight = 1;//(1.5)/(1 + y.squaredNorm()/R.sum());

  // kalman gains and I_KH
  XXM S = ((H_err * P) * H_err.transpose()) + R/weight;
  XEM KT = S.fullPivLu().solve(H_err * P.transpose());
  //EZM K = KT.transpose(); TODO: WHY DOES THIS NOT COMPILE?
  //EZM K = S.fullPivLu().solve(H_err * P.transpose()).transpose();
  //std::cout << "Here is the matrix rot:\n" << K << std::endl;
  EEM I_KH = Eigen::Matrix<double, EDIM, EDIM>::Identity() - (KT.transpose() * H_err);

  // update state by injecting dx
  Eigen::Matrix<double, EDIM, 1> dx(delta_x);
  dx  = (KT.transpose() * y);
  memcpy(delta_x, dx.data(), EDIM * sizeof(double));
  err_fun(in_x, delta_x, x_new);
  Eigen::Matrix<double, DIM, 1> x(x_new);

  // update cov
  P = ((I_KH * P) * I_KH.transpose()) + ((KT.transpose() * R) * KT);

  // copy out state
  memcpy(in_x, x.data(), DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
  memcpy(in_z, y.data(), y.rows() * sizeof(double));
}




}
extern "C" {

void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_4, H_4, NULL, in_z, in_R, in_ea, MAHA_THRESH_4);
}
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_10, H_10, NULL, in_z, in_R, in_ea, MAHA_THRESH_10);
}
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_13, H_13, NULL, in_z, in_R, in_ea, MAHA_THRESH_13);
}
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_14, H_14, NULL, in_z, in_R, in_ea, MAHA_THRESH_14);
}
void pose_err_fun(double *nom_x, double *delta_x, double *out_4069511595981512906) {
  err_fun(nom_x, delta_x, out_4069511595981512906);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_6977798052227057242) {
  inv_err_fun(nom_x, true_x, out_6977798052227057242);
}
void pose_H_mod_fun(double *state, double *out_2633911611712179105) {
  H_mod_fun(state, out_2633911611712179105);
}
void pose_f_fun(double *state, double dt, double *out_7060402792772463216) {
  f_fun(state,  dt, out_7060402792772463216);
}
void pose_F_fun(double *state, double dt, double *out_4609062820950943518) {
  F_fun(state,  dt, out_4609062820950943518);
}
void pose_h_4(double *state, double *unused, double *out_1304819822248380504) {
  h_4(state, unused, out_1304819822248380504);
}
void pose_H_4(double *state, double *unused, double *out_8481993418464137826) {
  H_4(state, unused, out_8481993418464137826);
}
void pose_h_10(double *state, double *unused, double *out_3476711082558028639) {
  h_10(state, unused, out_3476711082558028639);
}
void pose_H_10(double *state, double *unused, double *out_436246387152335361) {
  H_10(state, unused, out_436246387152335361);
}
void pose_h_13(double *state, double *unused, double *out_5988377314978994220) {
  h_13(state, unused, out_5988377314978994220);
}
void pose_H_13(double *state, double *unused, double *out_5269719593131805025) {
  H_13(state, unused, out_5269719593131805025);
}
void pose_h_14(double *state, double *unused, double *out_4290951141142713711) {
  h_14(state, unused, out_4290951141142713711);
}
void pose_H_14(double *state, double *unused, double *out_4518752562124653297) {
  H_14(state, unused, out_4518752562124653297);
}
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
}

const EKF pose = {
  .name = "pose",
  .kinds = { 4, 10, 13, 14 },
  .feature_kinds = {  },
  .f_fun = pose_f_fun,
  .F_fun = pose_F_fun,
  .err_fun = pose_err_fun,
  .inv_err_fun = pose_inv_err_fun,
  .H_mod_fun = pose_H_mod_fun,
  .predict = pose_predict,
  .hs = {
    { 4, pose_h_4 },
    { 10, pose_h_10 },
    { 13, pose_h_13 },
    { 14, pose_h_14 },
  },
  .Hs = {
    { 4, pose_H_4 },
    { 10, pose_H_10 },
    { 13, pose_H_13 },
    { 14, pose_H_14 },
  },
  .updates = {
    { 4, pose_update_4 },
    { 10, pose_update_10 },
    { 13, pose_update_13 },
    { 14, pose_update_14 },
  },
  .Hes = {
  },
  .sets = {
  },
  .extra_routines = {
  },
};

ekf_lib_init(pose)
