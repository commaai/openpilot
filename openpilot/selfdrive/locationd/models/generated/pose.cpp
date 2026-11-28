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
void err_fun(double *nom_x, double *delta_x, double *out_8093069977717794784) {
   out_8093069977717794784[0] = delta_x[0] + nom_x[0];
   out_8093069977717794784[1] = delta_x[1] + nom_x[1];
   out_8093069977717794784[2] = delta_x[2] + nom_x[2];
   out_8093069977717794784[3] = delta_x[3] + nom_x[3];
   out_8093069977717794784[4] = delta_x[4] + nom_x[4];
   out_8093069977717794784[5] = delta_x[5] + nom_x[5];
   out_8093069977717794784[6] = delta_x[6] + nom_x[6];
   out_8093069977717794784[7] = delta_x[7] + nom_x[7];
   out_8093069977717794784[8] = delta_x[8] + nom_x[8];
   out_8093069977717794784[9] = delta_x[9] + nom_x[9];
   out_8093069977717794784[10] = delta_x[10] + nom_x[10];
   out_8093069977717794784[11] = delta_x[11] + nom_x[11];
   out_8093069977717794784[12] = delta_x[12] + nom_x[12];
   out_8093069977717794784[13] = delta_x[13] + nom_x[13];
   out_8093069977717794784[14] = delta_x[14] + nom_x[14];
   out_8093069977717794784[15] = delta_x[15] + nom_x[15];
   out_8093069977717794784[16] = delta_x[16] + nom_x[16];
   out_8093069977717794784[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_2890339221041424080) {
   out_2890339221041424080[0] = -nom_x[0] + true_x[0];
   out_2890339221041424080[1] = -nom_x[1] + true_x[1];
   out_2890339221041424080[2] = -nom_x[2] + true_x[2];
   out_2890339221041424080[3] = -nom_x[3] + true_x[3];
   out_2890339221041424080[4] = -nom_x[4] + true_x[4];
   out_2890339221041424080[5] = -nom_x[5] + true_x[5];
   out_2890339221041424080[6] = -nom_x[6] + true_x[6];
   out_2890339221041424080[7] = -nom_x[7] + true_x[7];
   out_2890339221041424080[8] = -nom_x[8] + true_x[8];
   out_2890339221041424080[9] = -nom_x[9] + true_x[9];
   out_2890339221041424080[10] = -nom_x[10] + true_x[10];
   out_2890339221041424080[11] = -nom_x[11] + true_x[11];
   out_2890339221041424080[12] = -nom_x[12] + true_x[12];
   out_2890339221041424080[13] = -nom_x[13] + true_x[13];
   out_2890339221041424080[14] = -nom_x[14] + true_x[14];
   out_2890339221041424080[15] = -nom_x[15] + true_x[15];
   out_2890339221041424080[16] = -nom_x[16] + true_x[16];
   out_2890339221041424080[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_3258004078251541594) {
   out_3258004078251541594[0] = 1.0;
   out_3258004078251541594[1] = 0.0;
   out_3258004078251541594[2] = 0.0;
   out_3258004078251541594[3] = 0.0;
   out_3258004078251541594[4] = 0.0;
   out_3258004078251541594[5] = 0.0;
   out_3258004078251541594[6] = 0.0;
   out_3258004078251541594[7] = 0.0;
   out_3258004078251541594[8] = 0.0;
   out_3258004078251541594[9] = 0.0;
   out_3258004078251541594[10] = 0.0;
   out_3258004078251541594[11] = 0.0;
   out_3258004078251541594[12] = 0.0;
   out_3258004078251541594[13] = 0.0;
   out_3258004078251541594[14] = 0.0;
   out_3258004078251541594[15] = 0.0;
   out_3258004078251541594[16] = 0.0;
   out_3258004078251541594[17] = 0.0;
   out_3258004078251541594[18] = 0.0;
   out_3258004078251541594[19] = 1.0;
   out_3258004078251541594[20] = 0.0;
   out_3258004078251541594[21] = 0.0;
   out_3258004078251541594[22] = 0.0;
   out_3258004078251541594[23] = 0.0;
   out_3258004078251541594[24] = 0.0;
   out_3258004078251541594[25] = 0.0;
   out_3258004078251541594[26] = 0.0;
   out_3258004078251541594[27] = 0.0;
   out_3258004078251541594[28] = 0.0;
   out_3258004078251541594[29] = 0.0;
   out_3258004078251541594[30] = 0.0;
   out_3258004078251541594[31] = 0.0;
   out_3258004078251541594[32] = 0.0;
   out_3258004078251541594[33] = 0.0;
   out_3258004078251541594[34] = 0.0;
   out_3258004078251541594[35] = 0.0;
   out_3258004078251541594[36] = 0.0;
   out_3258004078251541594[37] = 0.0;
   out_3258004078251541594[38] = 1.0;
   out_3258004078251541594[39] = 0.0;
   out_3258004078251541594[40] = 0.0;
   out_3258004078251541594[41] = 0.0;
   out_3258004078251541594[42] = 0.0;
   out_3258004078251541594[43] = 0.0;
   out_3258004078251541594[44] = 0.0;
   out_3258004078251541594[45] = 0.0;
   out_3258004078251541594[46] = 0.0;
   out_3258004078251541594[47] = 0.0;
   out_3258004078251541594[48] = 0.0;
   out_3258004078251541594[49] = 0.0;
   out_3258004078251541594[50] = 0.0;
   out_3258004078251541594[51] = 0.0;
   out_3258004078251541594[52] = 0.0;
   out_3258004078251541594[53] = 0.0;
   out_3258004078251541594[54] = 0.0;
   out_3258004078251541594[55] = 0.0;
   out_3258004078251541594[56] = 0.0;
   out_3258004078251541594[57] = 1.0;
   out_3258004078251541594[58] = 0.0;
   out_3258004078251541594[59] = 0.0;
   out_3258004078251541594[60] = 0.0;
   out_3258004078251541594[61] = 0.0;
   out_3258004078251541594[62] = 0.0;
   out_3258004078251541594[63] = 0.0;
   out_3258004078251541594[64] = 0.0;
   out_3258004078251541594[65] = 0.0;
   out_3258004078251541594[66] = 0.0;
   out_3258004078251541594[67] = 0.0;
   out_3258004078251541594[68] = 0.0;
   out_3258004078251541594[69] = 0.0;
   out_3258004078251541594[70] = 0.0;
   out_3258004078251541594[71] = 0.0;
   out_3258004078251541594[72] = 0.0;
   out_3258004078251541594[73] = 0.0;
   out_3258004078251541594[74] = 0.0;
   out_3258004078251541594[75] = 0.0;
   out_3258004078251541594[76] = 1.0;
   out_3258004078251541594[77] = 0.0;
   out_3258004078251541594[78] = 0.0;
   out_3258004078251541594[79] = 0.0;
   out_3258004078251541594[80] = 0.0;
   out_3258004078251541594[81] = 0.0;
   out_3258004078251541594[82] = 0.0;
   out_3258004078251541594[83] = 0.0;
   out_3258004078251541594[84] = 0.0;
   out_3258004078251541594[85] = 0.0;
   out_3258004078251541594[86] = 0.0;
   out_3258004078251541594[87] = 0.0;
   out_3258004078251541594[88] = 0.0;
   out_3258004078251541594[89] = 0.0;
   out_3258004078251541594[90] = 0.0;
   out_3258004078251541594[91] = 0.0;
   out_3258004078251541594[92] = 0.0;
   out_3258004078251541594[93] = 0.0;
   out_3258004078251541594[94] = 0.0;
   out_3258004078251541594[95] = 1.0;
   out_3258004078251541594[96] = 0.0;
   out_3258004078251541594[97] = 0.0;
   out_3258004078251541594[98] = 0.0;
   out_3258004078251541594[99] = 0.0;
   out_3258004078251541594[100] = 0.0;
   out_3258004078251541594[101] = 0.0;
   out_3258004078251541594[102] = 0.0;
   out_3258004078251541594[103] = 0.0;
   out_3258004078251541594[104] = 0.0;
   out_3258004078251541594[105] = 0.0;
   out_3258004078251541594[106] = 0.0;
   out_3258004078251541594[107] = 0.0;
   out_3258004078251541594[108] = 0.0;
   out_3258004078251541594[109] = 0.0;
   out_3258004078251541594[110] = 0.0;
   out_3258004078251541594[111] = 0.0;
   out_3258004078251541594[112] = 0.0;
   out_3258004078251541594[113] = 0.0;
   out_3258004078251541594[114] = 1.0;
   out_3258004078251541594[115] = 0.0;
   out_3258004078251541594[116] = 0.0;
   out_3258004078251541594[117] = 0.0;
   out_3258004078251541594[118] = 0.0;
   out_3258004078251541594[119] = 0.0;
   out_3258004078251541594[120] = 0.0;
   out_3258004078251541594[121] = 0.0;
   out_3258004078251541594[122] = 0.0;
   out_3258004078251541594[123] = 0.0;
   out_3258004078251541594[124] = 0.0;
   out_3258004078251541594[125] = 0.0;
   out_3258004078251541594[126] = 0.0;
   out_3258004078251541594[127] = 0.0;
   out_3258004078251541594[128] = 0.0;
   out_3258004078251541594[129] = 0.0;
   out_3258004078251541594[130] = 0.0;
   out_3258004078251541594[131] = 0.0;
   out_3258004078251541594[132] = 0.0;
   out_3258004078251541594[133] = 1.0;
   out_3258004078251541594[134] = 0.0;
   out_3258004078251541594[135] = 0.0;
   out_3258004078251541594[136] = 0.0;
   out_3258004078251541594[137] = 0.0;
   out_3258004078251541594[138] = 0.0;
   out_3258004078251541594[139] = 0.0;
   out_3258004078251541594[140] = 0.0;
   out_3258004078251541594[141] = 0.0;
   out_3258004078251541594[142] = 0.0;
   out_3258004078251541594[143] = 0.0;
   out_3258004078251541594[144] = 0.0;
   out_3258004078251541594[145] = 0.0;
   out_3258004078251541594[146] = 0.0;
   out_3258004078251541594[147] = 0.0;
   out_3258004078251541594[148] = 0.0;
   out_3258004078251541594[149] = 0.0;
   out_3258004078251541594[150] = 0.0;
   out_3258004078251541594[151] = 0.0;
   out_3258004078251541594[152] = 1.0;
   out_3258004078251541594[153] = 0.0;
   out_3258004078251541594[154] = 0.0;
   out_3258004078251541594[155] = 0.0;
   out_3258004078251541594[156] = 0.0;
   out_3258004078251541594[157] = 0.0;
   out_3258004078251541594[158] = 0.0;
   out_3258004078251541594[159] = 0.0;
   out_3258004078251541594[160] = 0.0;
   out_3258004078251541594[161] = 0.0;
   out_3258004078251541594[162] = 0.0;
   out_3258004078251541594[163] = 0.0;
   out_3258004078251541594[164] = 0.0;
   out_3258004078251541594[165] = 0.0;
   out_3258004078251541594[166] = 0.0;
   out_3258004078251541594[167] = 0.0;
   out_3258004078251541594[168] = 0.0;
   out_3258004078251541594[169] = 0.0;
   out_3258004078251541594[170] = 0.0;
   out_3258004078251541594[171] = 1.0;
   out_3258004078251541594[172] = 0.0;
   out_3258004078251541594[173] = 0.0;
   out_3258004078251541594[174] = 0.0;
   out_3258004078251541594[175] = 0.0;
   out_3258004078251541594[176] = 0.0;
   out_3258004078251541594[177] = 0.0;
   out_3258004078251541594[178] = 0.0;
   out_3258004078251541594[179] = 0.0;
   out_3258004078251541594[180] = 0.0;
   out_3258004078251541594[181] = 0.0;
   out_3258004078251541594[182] = 0.0;
   out_3258004078251541594[183] = 0.0;
   out_3258004078251541594[184] = 0.0;
   out_3258004078251541594[185] = 0.0;
   out_3258004078251541594[186] = 0.0;
   out_3258004078251541594[187] = 0.0;
   out_3258004078251541594[188] = 0.0;
   out_3258004078251541594[189] = 0.0;
   out_3258004078251541594[190] = 1.0;
   out_3258004078251541594[191] = 0.0;
   out_3258004078251541594[192] = 0.0;
   out_3258004078251541594[193] = 0.0;
   out_3258004078251541594[194] = 0.0;
   out_3258004078251541594[195] = 0.0;
   out_3258004078251541594[196] = 0.0;
   out_3258004078251541594[197] = 0.0;
   out_3258004078251541594[198] = 0.0;
   out_3258004078251541594[199] = 0.0;
   out_3258004078251541594[200] = 0.0;
   out_3258004078251541594[201] = 0.0;
   out_3258004078251541594[202] = 0.0;
   out_3258004078251541594[203] = 0.0;
   out_3258004078251541594[204] = 0.0;
   out_3258004078251541594[205] = 0.0;
   out_3258004078251541594[206] = 0.0;
   out_3258004078251541594[207] = 0.0;
   out_3258004078251541594[208] = 0.0;
   out_3258004078251541594[209] = 1.0;
   out_3258004078251541594[210] = 0.0;
   out_3258004078251541594[211] = 0.0;
   out_3258004078251541594[212] = 0.0;
   out_3258004078251541594[213] = 0.0;
   out_3258004078251541594[214] = 0.0;
   out_3258004078251541594[215] = 0.0;
   out_3258004078251541594[216] = 0.0;
   out_3258004078251541594[217] = 0.0;
   out_3258004078251541594[218] = 0.0;
   out_3258004078251541594[219] = 0.0;
   out_3258004078251541594[220] = 0.0;
   out_3258004078251541594[221] = 0.0;
   out_3258004078251541594[222] = 0.0;
   out_3258004078251541594[223] = 0.0;
   out_3258004078251541594[224] = 0.0;
   out_3258004078251541594[225] = 0.0;
   out_3258004078251541594[226] = 0.0;
   out_3258004078251541594[227] = 0.0;
   out_3258004078251541594[228] = 1.0;
   out_3258004078251541594[229] = 0.0;
   out_3258004078251541594[230] = 0.0;
   out_3258004078251541594[231] = 0.0;
   out_3258004078251541594[232] = 0.0;
   out_3258004078251541594[233] = 0.0;
   out_3258004078251541594[234] = 0.0;
   out_3258004078251541594[235] = 0.0;
   out_3258004078251541594[236] = 0.0;
   out_3258004078251541594[237] = 0.0;
   out_3258004078251541594[238] = 0.0;
   out_3258004078251541594[239] = 0.0;
   out_3258004078251541594[240] = 0.0;
   out_3258004078251541594[241] = 0.0;
   out_3258004078251541594[242] = 0.0;
   out_3258004078251541594[243] = 0.0;
   out_3258004078251541594[244] = 0.0;
   out_3258004078251541594[245] = 0.0;
   out_3258004078251541594[246] = 0.0;
   out_3258004078251541594[247] = 1.0;
   out_3258004078251541594[248] = 0.0;
   out_3258004078251541594[249] = 0.0;
   out_3258004078251541594[250] = 0.0;
   out_3258004078251541594[251] = 0.0;
   out_3258004078251541594[252] = 0.0;
   out_3258004078251541594[253] = 0.0;
   out_3258004078251541594[254] = 0.0;
   out_3258004078251541594[255] = 0.0;
   out_3258004078251541594[256] = 0.0;
   out_3258004078251541594[257] = 0.0;
   out_3258004078251541594[258] = 0.0;
   out_3258004078251541594[259] = 0.0;
   out_3258004078251541594[260] = 0.0;
   out_3258004078251541594[261] = 0.0;
   out_3258004078251541594[262] = 0.0;
   out_3258004078251541594[263] = 0.0;
   out_3258004078251541594[264] = 0.0;
   out_3258004078251541594[265] = 0.0;
   out_3258004078251541594[266] = 1.0;
   out_3258004078251541594[267] = 0.0;
   out_3258004078251541594[268] = 0.0;
   out_3258004078251541594[269] = 0.0;
   out_3258004078251541594[270] = 0.0;
   out_3258004078251541594[271] = 0.0;
   out_3258004078251541594[272] = 0.0;
   out_3258004078251541594[273] = 0.0;
   out_3258004078251541594[274] = 0.0;
   out_3258004078251541594[275] = 0.0;
   out_3258004078251541594[276] = 0.0;
   out_3258004078251541594[277] = 0.0;
   out_3258004078251541594[278] = 0.0;
   out_3258004078251541594[279] = 0.0;
   out_3258004078251541594[280] = 0.0;
   out_3258004078251541594[281] = 0.0;
   out_3258004078251541594[282] = 0.0;
   out_3258004078251541594[283] = 0.0;
   out_3258004078251541594[284] = 0.0;
   out_3258004078251541594[285] = 1.0;
   out_3258004078251541594[286] = 0.0;
   out_3258004078251541594[287] = 0.0;
   out_3258004078251541594[288] = 0.0;
   out_3258004078251541594[289] = 0.0;
   out_3258004078251541594[290] = 0.0;
   out_3258004078251541594[291] = 0.0;
   out_3258004078251541594[292] = 0.0;
   out_3258004078251541594[293] = 0.0;
   out_3258004078251541594[294] = 0.0;
   out_3258004078251541594[295] = 0.0;
   out_3258004078251541594[296] = 0.0;
   out_3258004078251541594[297] = 0.0;
   out_3258004078251541594[298] = 0.0;
   out_3258004078251541594[299] = 0.0;
   out_3258004078251541594[300] = 0.0;
   out_3258004078251541594[301] = 0.0;
   out_3258004078251541594[302] = 0.0;
   out_3258004078251541594[303] = 0.0;
   out_3258004078251541594[304] = 1.0;
   out_3258004078251541594[305] = 0.0;
   out_3258004078251541594[306] = 0.0;
   out_3258004078251541594[307] = 0.0;
   out_3258004078251541594[308] = 0.0;
   out_3258004078251541594[309] = 0.0;
   out_3258004078251541594[310] = 0.0;
   out_3258004078251541594[311] = 0.0;
   out_3258004078251541594[312] = 0.0;
   out_3258004078251541594[313] = 0.0;
   out_3258004078251541594[314] = 0.0;
   out_3258004078251541594[315] = 0.0;
   out_3258004078251541594[316] = 0.0;
   out_3258004078251541594[317] = 0.0;
   out_3258004078251541594[318] = 0.0;
   out_3258004078251541594[319] = 0.0;
   out_3258004078251541594[320] = 0.0;
   out_3258004078251541594[321] = 0.0;
   out_3258004078251541594[322] = 0.0;
   out_3258004078251541594[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_3851408165110359671) {
   out_3851408165110359671[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_3851408165110359671[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_3851408165110359671[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_3851408165110359671[3] = dt*state[12] + state[3];
   out_3851408165110359671[4] = dt*state[13] + state[4];
   out_3851408165110359671[5] = dt*state[14] + state[5];
   out_3851408165110359671[6] = state[6];
   out_3851408165110359671[7] = state[7];
   out_3851408165110359671[8] = state[8];
   out_3851408165110359671[9] = state[9];
   out_3851408165110359671[10] = state[10];
   out_3851408165110359671[11] = state[11];
   out_3851408165110359671[12] = state[12];
   out_3851408165110359671[13] = state[13];
   out_3851408165110359671[14] = state[14];
   out_3851408165110359671[15] = state[15];
   out_3851408165110359671[16] = state[16];
   out_3851408165110359671[17] = state[17];
}
void F_fun(double *state, double dt, double *out_5176413023006983401) {
   out_5176413023006983401[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_5176413023006983401[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_5176413023006983401[2] = 0;
   out_5176413023006983401[3] = 0;
   out_5176413023006983401[4] = 0;
   out_5176413023006983401[5] = 0;
   out_5176413023006983401[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_5176413023006983401[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_5176413023006983401[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_5176413023006983401[9] = 0;
   out_5176413023006983401[10] = 0;
   out_5176413023006983401[11] = 0;
   out_5176413023006983401[12] = 0;
   out_5176413023006983401[13] = 0;
   out_5176413023006983401[14] = 0;
   out_5176413023006983401[15] = 0;
   out_5176413023006983401[16] = 0;
   out_5176413023006983401[17] = 0;
   out_5176413023006983401[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_5176413023006983401[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_5176413023006983401[20] = 0;
   out_5176413023006983401[21] = 0;
   out_5176413023006983401[22] = 0;
   out_5176413023006983401[23] = 0;
   out_5176413023006983401[24] = 0;
   out_5176413023006983401[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_5176413023006983401[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_5176413023006983401[27] = 0;
   out_5176413023006983401[28] = 0;
   out_5176413023006983401[29] = 0;
   out_5176413023006983401[30] = 0;
   out_5176413023006983401[31] = 0;
   out_5176413023006983401[32] = 0;
   out_5176413023006983401[33] = 0;
   out_5176413023006983401[34] = 0;
   out_5176413023006983401[35] = 0;
   out_5176413023006983401[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_5176413023006983401[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_5176413023006983401[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_5176413023006983401[39] = 0;
   out_5176413023006983401[40] = 0;
   out_5176413023006983401[41] = 0;
   out_5176413023006983401[42] = 0;
   out_5176413023006983401[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_5176413023006983401[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_5176413023006983401[45] = 0;
   out_5176413023006983401[46] = 0;
   out_5176413023006983401[47] = 0;
   out_5176413023006983401[48] = 0;
   out_5176413023006983401[49] = 0;
   out_5176413023006983401[50] = 0;
   out_5176413023006983401[51] = 0;
   out_5176413023006983401[52] = 0;
   out_5176413023006983401[53] = 0;
   out_5176413023006983401[54] = 0;
   out_5176413023006983401[55] = 0;
   out_5176413023006983401[56] = 0;
   out_5176413023006983401[57] = 1;
   out_5176413023006983401[58] = 0;
   out_5176413023006983401[59] = 0;
   out_5176413023006983401[60] = 0;
   out_5176413023006983401[61] = 0;
   out_5176413023006983401[62] = 0;
   out_5176413023006983401[63] = 0;
   out_5176413023006983401[64] = 0;
   out_5176413023006983401[65] = 0;
   out_5176413023006983401[66] = dt;
   out_5176413023006983401[67] = 0;
   out_5176413023006983401[68] = 0;
   out_5176413023006983401[69] = 0;
   out_5176413023006983401[70] = 0;
   out_5176413023006983401[71] = 0;
   out_5176413023006983401[72] = 0;
   out_5176413023006983401[73] = 0;
   out_5176413023006983401[74] = 0;
   out_5176413023006983401[75] = 0;
   out_5176413023006983401[76] = 1;
   out_5176413023006983401[77] = 0;
   out_5176413023006983401[78] = 0;
   out_5176413023006983401[79] = 0;
   out_5176413023006983401[80] = 0;
   out_5176413023006983401[81] = 0;
   out_5176413023006983401[82] = 0;
   out_5176413023006983401[83] = 0;
   out_5176413023006983401[84] = 0;
   out_5176413023006983401[85] = dt;
   out_5176413023006983401[86] = 0;
   out_5176413023006983401[87] = 0;
   out_5176413023006983401[88] = 0;
   out_5176413023006983401[89] = 0;
   out_5176413023006983401[90] = 0;
   out_5176413023006983401[91] = 0;
   out_5176413023006983401[92] = 0;
   out_5176413023006983401[93] = 0;
   out_5176413023006983401[94] = 0;
   out_5176413023006983401[95] = 1;
   out_5176413023006983401[96] = 0;
   out_5176413023006983401[97] = 0;
   out_5176413023006983401[98] = 0;
   out_5176413023006983401[99] = 0;
   out_5176413023006983401[100] = 0;
   out_5176413023006983401[101] = 0;
   out_5176413023006983401[102] = 0;
   out_5176413023006983401[103] = 0;
   out_5176413023006983401[104] = dt;
   out_5176413023006983401[105] = 0;
   out_5176413023006983401[106] = 0;
   out_5176413023006983401[107] = 0;
   out_5176413023006983401[108] = 0;
   out_5176413023006983401[109] = 0;
   out_5176413023006983401[110] = 0;
   out_5176413023006983401[111] = 0;
   out_5176413023006983401[112] = 0;
   out_5176413023006983401[113] = 0;
   out_5176413023006983401[114] = 1;
   out_5176413023006983401[115] = 0;
   out_5176413023006983401[116] = 0;
   out_5176413023006983401[117] = 0;
   out_5176413023006983401[118] = 0;
   out_5176413023006983401[119] = 0;
   out_5176413023006983401[120] = 0;
   out_5176413023006983401[121] = 0;
   out_5176413023006983401[122] = 0;
   out_5176413023006983401[123] = 0;
   out_5176413023006983401[124] = 0;
   out_5176413023006983401[125] = 0;
   out_5176413023006983401[126] = 0;
   out_5176413023006983401[127] = 0;
   out_5176413023006983401[128] = 0;
   out_5176413023006983401[129] = 0;
   out_5176413023006983401[130] = 0;
   out_5176413023006983401[131] = 0;
   out_5176413023006983401[132] = 0;
   out_5176413023006983401[133] = 1;
   out_5176413023006983401[134] = 0;
   out_5176413023006983401[135] = 0;
   out_5176413023006983401[136] = 0;
   out_5176413023006983401[137] = 0;
   out_5176413023006983401[138] = 0;
   out_5176413023006983401[139] = 0;
   out_5176413023006983401[140] = 0;
   out_5176413023006983401[141] = 0;
   out_5176413023006983401[142] = 0;
   out_5176413023006983401[143] = 0;
   out_5176413023006983401[144] = 0;
   out_5176413023006983401[145] = 0;
   out_5176413023006983401[146] = 0;
   out_5176413023006983401[147] = 0;
   out_5176413023006983401[148] = 0;
   out_5176413023006983401[149] = 0;
   out_5176413023006983401[150] = 0;
   out_5176413023006983401[151] = 0;
   out_5176413023006983401[152] = 1;
   out_5176413023006983401[153] = 0;
   out_5176413023006983401[154] = 0;
   out_5176413023006983401[155] = 0;
   out_5176413023006983401[156] = 0;
   out_5176413023006983401[157] = 0;
   out_5176413023006983401[158] = 0;
   out_5176413023006983401[159] = 0;
   out_5176413023006983401[160] = 0;
   out_5176413023006983401[161] = 0;
   out_5176413023006983401[162] = 0;
   out_5176413023006983401[163] = 0;
   out_5176413023006983401[164] = 0;
   out_5176413023006983401[165] = 0;
   out_5176413023006983401[166] = 0;
   out_5176413023006983401[167] = 0;
   out_5176413023006983401[168] = 0;
   out_5176413023006983401[169] = 0;
   out_5176413023006983401[170] = 0;
   out_5176413023006983401[171] = 1;
   out_5176413023006983401[172] = 0;
   out_5176413023006983401[173] = 0;
   out_5176413023006983401[174] = 0;
   out_5176413023006983401[175] = 0;
   out_5176413023006983401[176] = 0;
   out_5176413023006983401[177] = 0;
   out_5176413023006983401[178] = 0;
   out_5176413023006983401[179] = 0;
   out_5176413023006983401[180] = 0;
   out_5176413023006983401[181] = 0;
   out_5176413023006983401[182] = 0;
   out_5176413023006983401[183] = 0;
   out_5176413023006983401[184] = 0;
   out_5176413023006983401[185] = 0;
   out_5176413023006983401[186] = 0;
   out_5176413023006983401[187] = 0;
   out_5176413023006983401[188] = 0;
   out_5176413023006983401[189] = 0;
   out_5176413023006983401[190] = 1;
   out_5176413023006983401[191] = 0;
   out_5176413023006983401[192] = 0;
   out_5176413023006983401[193] = 0;
   out_5176413023006983401[194] = 0;
   out_5176413023006983401[195] = 0;
   out_5176413023006983401[196] = 0;
   out_5176413023006983401[197] = 0;
   out_5176413023006983401[198] = 0;
   out_5176413023006983401[199] = 0;
   out_5176413023006983401[200] = 0;
   out_5176413023006983401[201] = 0;
   out_5176413023006983401[202] = 0;
   out_5176413023006983401[203] = 0;
   out_5176413023006983401[204] = 0;
   out_5176413023006983401[205] = 0;
   out_5176413023006983401[206] = 0;
   out_5176413023006983401[207] = 0;
   out_5176413023006983401[208] = 0;
   out_5176413023006983401[209] = 1;
   out_5176413023006983401[210] = 0;
   out_5176413023006983401[211] = 0;
   out_5176413023006983401[212] = 0;
   out_5176413023006983401[213] = 0;
   out_5176413023006983401[214] = 0;
   out_5176413023006983401[215] = 0;
   out_5176413023006983401[216] = 0;
   out_5176413023006983401[217] = 0;
   out_5176413023006983401[218] = 0;
   out_5176413023006983401[219] = 0;
   out_5176413023006983401[220] = 0;
   out_5176413023006983401[221] = 0;
   out_5176413023006983401[222] = 0;
   out_5176413023006983401[223] = 0;
   out_5176413023006983401[224] = 0;
   out_5176413023006983401[225] = 0;
   out_5176413023006983401[226] = 0;
   out_5176413023006983401[227] = 0;
   out_5176413023006983401[228] = 1;
   out_5176413023006983401[229] = 0;
   out_5176413023006983401[230] = 0;
   out_5176413023006983401[231] = 0;
   out_5176413023006983401[232] = 0;
   out_5176413023006983401[233] = 0;
   out_5176413023006983401[234] = 0;
   out_5176413023006983401[235] = 0;
   out_5176413023006983401[236] = 0;
   out_5176413023006983401[237] = 0;
   out_5176413023006983401[238] = 0;
   out_5176413023006983401[239] = 0;
   out_5176413023006983401[240] = 0;
   out_5176413023006983401[241] = 0;
   out_5176413023006983401[242] = 0;
   out_5176413023006983401[243] = 0;
   out_5176413023006983401[244] = 0;
   out_5176413023006983401[245] = 0;
   out_5176413023006983401[246] = 0;
   out_5176413023006983401[247] = 1;
   out_5176413023006983401[248] = 0;
   out_5176413023006983401[249] = 0;
   out_5176413023006983401[250] = 0;
   out_5176413023006983401[251] = 0;
   out_5176413023006983401[252] = 0;
   out_5176413023006983401[253] = 0;
   out_5176413023006983401[254] = 0;
   out_5176413023006983401[255] = 0;
   out_5176413023006983401[256] = 0;
   out_5176413023006983401[257] = 0;
   out_5176413023006983401[258] = 0;
   out_5176413023006983401[259] = 0;
   out_5176413023006983401[260] = 0;
   out_5176413023006983401[261] = 0;
   out_5176413023006983401[262] = 0;
   out_5176413023006983401[263] = 0;
   out_5176413023006983401[264] = 0;
   out_5176413023006983401[265] = 0;
   out_5176413023006983401[266] = 1;
   out_5176413023006983401[267] = 0;
   out_5176413023006983401[268] = 0;
   out_5176413023006983401[269] = 0;
   out_5176413023006983401[270] = 0;
   out_5176413023006983401[271] = 0;
   out_5176413023006983401[272] = 0;
   out_5176413023006983401[273] = 0;
   out_5176413023006983401[274] = 0;
   out_5176413023006983401[275] = 0;
   out_5176413023006983401[276] = 0;
   out_5176413023006983401[277] = 0;
   out_5176413023006983401[278] = 0;
   out_5176413023006983401[279] = 0;
   out_5176413023006983401[280] = 0;
   out_5176413023006983401[281] = 0;
   out_5176413023006983401[282] = 0;
   out_5176413023006983401[283] = 0;
   out_5176413023006983401[284] = 0;
   out_5176413023006983401[285] = 1;
   out_5176413023006983401[286] = 0;
   out_5176413023006983401[287] = 0;
   out_5176413023006983401[288] = 0;
   out_5176413023006983401[289] = 0;
   out_5176413023006983401[290] = 0;
   out_5176413023006983401[291] = 0;
   out_5176413023006983401[292] = 0;
   out_5176413023006983401[293] = 0;
   out_5176413023006983401[294] = 0;
   out_5176413023006983401[295] = 0;
   out_5176413023006983401[296] = 0;
   out_5176413023006983401[297] = 0;
   out_5176413023006983401[298] = 0;
   out_5176413023006983401[299] = 0;
   out_5176413023006983401[300] = 0;
   out_5176413023006983401[301] = 0;
   out_5176413023006983401[302] = 0;
   out_5176413023006983401[303] = 0;
   out_5176413023006983401[304] = 1;
   out_5176413023006983401[305] = 0;
   out_5176413023006983401[306] = 0;
   out_5176413023006983401[307] = 0;
   out_5176413023006983401[308] = 0;
   out_5176413023006983401[309] = 0;
   out_5176413023006983401[310] = 0;
   out_5176413023006983401[311] = 0;
   out_5176413023006983401[312] = 0;
   out_5176413023006983401[313] = 0;
   out_5176413023006983401[314] = 0;
   out_5176413023006983401[315] = 0;
   out_5176413023006983401[316] = 0;
   out_5176413023006983401[317] = 0;
   out_5176413023006983401[318] = 0;
   out_5176413023006983401[319] = 0;
   out_5176413023006983401[320] = 0;
   out_5176413023006983401[321] = 0;
   out_5176413023006983401[322] = 0;
   out_5176413023006983401[323] = 1;
}
void h_4(double *state, double *unused, double *out_9158595119508026012) {
   out_9158595119508026012[0] = state[6] + state[9];
   out_9158595119508026012[1] = state[7] + state[10];
   out_9158595119508026012[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_7087198156485597472) {
   out_7087198156485597472[0] = 0;
   out_7087198156485597472[1] = 0;
   out_7087198156485597472[2] = 0;
   out_7087198156485597472[3] = 0;
   out_7087198156485597472[4] = 0;
   out_7087198156485597472[5] = 0;
   out_7087198156485597472[6] = 1;
   out_7087198156485597472[7] = 0;
   out_7087198156485597472[8] = 0;
   out_7087198156485597472[9] = 1;
   out_7087198156485597472[10] = 0;
   out_7087198156485597472[11] = 0;
   out_7087198156485597472[12] = 0;
   out_7087198156485597472[13] = 0;
   out_7087198156485597472[14] = 0;
   out_7087198156485597472[15] = 0;
   out_7087198156485597472[16] = 0;
   out_7087198156485597472[17] = 0;
   out_7087198156485597472[18] = 0;
   out_7087198156485597472[19] = 0;
   out_7087198156485597472[20] = 0;
   out_7087198156485597472[21] = 0;
   out_7087198156485597472[22] = 0;
   out_7087198156485597472[23] = 0;
   out_7087198156485597472[24] = 0;
   out_7087198156485597472[25] = 1;
   out_7087198156485597472[26] = 0;
   out_7087198156485597472[27] = 0;
   out_7087198156485597472[28] = 1;
   out_7087198156485597472[29] = 0;
   out_7087198156485597472[30] = 0;
   out_7087198156485597472[31] = 0;
   out_7087198156485597472[32] = 0;
   out_7087198156485597472[33] = 0;
   out_7087198156485597472[34] = 0;
   out_7087198156485597472[35] = 0;
   out_7087198156485597472[36] = 0;
   out_7087198156485597472[37] = 0;
   out_7087198156485597472[38] = 0;
   out_7087198156485597472[39] = 0;
   out_7087198156485597472[40] = 0;
   out_7087198156485597472[41] = 0;
   out_7087198156485597472[42] = 0;
   out_7087198156485597472[43] = 0;
   out_7087198156485597472[44] = 1;
   out_7087198156485597472[45] = 0;
   out_7087198156485597472[46] = 0;
   out_7087198156485597472[47] = 1;
   out_7087198156485597472[48] = 0;
   out_7087198156485597472[49] = 0;
   out_7087198156485597472[50] = 0;
   out_7087198156485597472[51] = 0;
   out_7087198156485597472[52] = 0;
   out_7087198156485597472[53] = 0;
}
void h_10(double *state, double *unused, double *out_1976892447920578365) {
   out_1976892447920578365[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_1976892447920578365[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_1976892447920578365[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_2856556414602702172) {
   out_2856556414602702172[0] = 0;
   out_2856556414602702172[1] = 9.8100000000000005*cos(state[1]);
   out_2856556414602702172[2] = 0;
   out_2856556414602702172[3] = 0;
   out_2856556414602702172[4] = -state[8];
   out_2856556414602702172[5] = state[7];
   out_2856556414602702172[6] = 0;
   out_2856556414602702172[7] = state[5];
   out_2856556414602702172[8] = -state[4];
   out_2856556414602702172[9] = 0;
   out_2856556414602702172[10] = 0;
   out_2856556414602702172[11] = 0;
   out_2856556414602702172[12] = 1;
   out_2856556414602702172[13] = 0;
   out_2856556414602702172[14] = 0;
   out_2856556414602702172[15] = 1;
   out_2856556414602702172[16] = 0;
   out_2856556414602702172[17] = 0;
   out_2856556414602702172[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_2856556414602702172[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_2856556414602702172[20] = 0;
   out_2856556414602702172[21] = state[8];
   out_2856556414602702172[22] = 0;
   out_2856556414602702172[23] = -state[6];
   out_2856556414602702172[24] = -state[5];
   out_2856556414602702172[25] = 0;
   out_2856556414602702172[26] = state[3];
   out_2856556414602702172[27] = 0;
   out_2856556414602702172[28] = 0;
   out_2856556414602702172[29] = 0;
   out_2856556414602702172[30] = 0;
   out_2856556414602702172[31] = 1;
   out_2856556414602702172[32] = 0;
   out_2856556414602702172[33] = 0;
   out_2856556414602702172[34] = 1;
   out_2856556414602702172[35] = 0;
   out_2856556414602702172[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_2856556414602702172[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_2856556414602702172[38] = 0;
   out_2856556414602702172[39] = -state[7];
   out_2856556414602702172[40] = state[6];
   out_2856556414602702172[41] = 0;
   out_2856556414602702172[42] = state[4];
   out_2856556414602702172[43] = -state[3];
   out_2856556414602702172[44] = 0;
   out_2856556414602702172[45] = 0;
   out_2856556414602702172[46] = 0;
   out_2856556414602702172[47] = 0;
   out_2856556414602702172[48] = 0;
   out_2856556414602702172[49] = 0;
   out_2856556414602702172[50] = 1;
   out_2856556414602702172[51] = 0;
   out_2856556414602702172[52] = 0;
   out_2856556414602702172[53] = 1;
}
void h_13(double *state, double *unused, double *out_7124925249639989703) {
   out_7124925249639989703[0] = state[3];
   out_7124925249639989703[1] = state[4];
   out_7124925249639989703[2] = state[5];
}
void H_13(double *state, double *unused, double *out_8147272091891621343) {
   out_8147272091891621343[0] = 0;
   out_8147272091891621343[1] = 0;
   out_8147272091891621343[2] = 0;
   out_8147272091891621343[3] = 1;
   out_8147272091891621343[4] = 0;
   out_8147272091891621343[5] = 0;
   out_8147272091891621343[6] = 0;
   out_8147272091891621343[7] = 0;
   out_8147272091891621343[8] = 0;
   out_8147272091891621343[9] = 0;
   out_8147272091891621343[10] = 0;
   out_8147272091891621343[11] = 0;
   out_8147272091891621343[12] = 0;
   out_8147272091891621343[13] = 0;
   out_8147272091891621343[14] = 0;
   out_8147272091891621343[15] = 0;
   out_8147272091891621343[16] = 0;
   out_8147272091891621343[17] = 0;
   out_8147272091891621343[18] = 0;
   out_8147272091891621343[19] = 0;
   out_8147272091891621343[20] = 0;
   out_8147272091891621343[21] = 0;
   out_8147272091891621343[22] = 1;
   out_8147272091891621343[23] = 0;
   out_8147272091891621343[24] = 0;
   out_8147272091891621343[25] = 0;
   out_8147272091891621343[26] = 0;
   out_8147272091891621343[27] = 0;
   out_8147272091891621343[28] = 0;
   out_8147272091891621343[29] = 0;
   out_8147272091891621343[30] = 0;
   out_8147272091891621343[31] = 0;
   out_8147272091891621343[32] = 0;
   out_8147272091891621343[33] = 0;
   out_8147272091891621343[34] = 0;
   out_8147272091891621343[35] = 0;
   out_8147272091891621343[36] = 0;
   out_8147272091891621343[37] = 0;
   out_8147272091891621343[38] = 0;
   out_8147272091891621343[39] = 0;
   out_8147272091891621343[40] = 0;
   out_8147272091891621343[41] = 1;
   out_8147272091891621343[42] = 0;
   out_8147272091891621343[43] = 0;
   out_8147272091891621343[44] = 0;
   out_8147272091891621343[45] = 0;
   out_8147272091891621343[46] = 0;
   out_8147272091891621343[47] = 0;
   out_8147272091891621343[48] = 0;
   out_8147272091891621343[49] = 0;
   out_8147272091891621343[50] = 0;
   out_8147272091891621343[51] = 0;
   out_8147272091891621343[52] = 0;
   out_8147272091891621343[53] = 0;
}
void h_14(double *state, double *unused, double *out_1921870948443217106) {
   out_1921870948443217106[0] = state[6];
   out_1921870948443217106[1] = state[7];
   out_1921870948443217106[2] = state[8];
}
void H_14(double *state, double *unused, double *out_7396305060884469615) {
   out_7396305060884469615[0] = 0;
   out_7396305060884469615[1] = 0;
   out_7396305060884469615[2] = 0;
   out_7396305060884469615[3] = 0;
   out_7396305060884469615[4] = 0;
   out_7396305060884469615[5] = 0;
   out_7396305060884469615[6] = 1;
   out_7396305060884469615[7] = 0;
   out_7396305060884469615[8] = 0;
   out_7396305060884469615[9] = 0;
   out_7396305060884469615[10] = 0;
   out_7396305060884469615[11] = 0;
   out_7396305060884469615[12] = 0;
   out_7396305060884469615[13] = 0;
   out_7396305060884469615[14] = 0;
   out_7396305060884469615[15] = 0;
   out_7396305060884469615[16] = 0;
   out_7396305060884469615[17] = 0;
   out_7396305060884469615[18] = 0;
   out_7396305060884469615[19] = 0;
   out_7396305060884469615[20] = 0;
   out_7396305060884469615[21] = 0;
   out_7396305060884469615[22] = 0;
   out_7396305060884469615[23] = 0;
   out_7396305060884469615[24] = 0;
   out_7396305060884469615[25] = 1;
   out_7396305060884469615[26] = 0;
   out_7396305060884469615[27] = 0;
   out_7396305060884469615[28] = 0;
   out_7396305060884469615[29] = 0;
   out_7396305060884469615[30] = 0;
   out_7396305060884469615[31] = 0;
   out_7396305060884469615[32] = 0;
   out_7396305060884469615[33] = 0;
   out_7396305060884469615[34] = 0;
   out_7396305060884469615[35] = 0;
   out_7396305060884469615[36] = 0;
   out_7396305060884469615[37] = 0;
   out_7396305060884469615[38] = 0;
   out_7396305060884469615[39] = 0;
   out_7396305060884469615[40] = 0;
   out_7396305060884469615[41] = 0;
   out_7396305060884469615[42] = 0;
   out_7396305060884469615[43] = 0;
   out_7396305060884469615[44] = 1;
   out_7396305060884469615[45] = 0;
   out_7396305060884469615[46] = 0;
   out_7396305060884469615[47] = 0;
   out_7396305060884469615[48] = 0;
   out_7396305060884469615[49] = 0;
   out_7396305060884469615[50] = 0;
   out_7396305060884469615[51] = 0;
   out_7396305060884469615[52] = 0;
   out_7396305060884469615[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_8093069977717794784) {
  err_fun(nom_x, delta_x, out_8093069977717794784);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_2890339221041424080) {
  inv_err_fun(nom_x, true_x, out_2890339221041424080);
}
void pose_H_mod_fun(double *state, double *out_3258004078251541594) {
  H_mod_fun(state, out_3258004078251541594);
}
void pose_f_fun(double *state, double dt, double *out_3851408165110359671) {
  f_fun(state,  dt, out_3851408165110359671);
}
void pose_F_fun(double *state, double dt, double *out_5176413023006983401) {
  F_fun(state,  dt, out_5176413023006983401);
}
void pose_h_4(double *state, double *unused, double *out_9158595119508026012) {
  h_4(state, unused, out_9158595119508026012);
}
void pose_H_4(double *state, double *unused, double *out_7087198156485597472) {
  H_4(state, unused, out_7087198156485597472);
}
void pose_h_10(double *state, double *unused, double *out_1976892447920578365) {
  h_10(state, unused, out_1976892447920578365);
}
void pose_H_10(double *state, double *unused, double *out_2856556414602702172) {
  H_10(state, unused, out_2856556414602702172);
}
void pose_h_13(double *state, double *unused, double *out_7124925249639989703) {
  h_13(state, unused, out_7124925249639989703);
}
void pose_H_13(double *state, double *unused, double *out_8147272091891621343) {
  H_13(state, unused, out_8147272091891621343);
}
void pose_h_14(double *state, double *unused, double *out_1921870948443217106) {
  h_14(state, unused, out_1921870948443217106);
}
void pose_H_14(double *state, double *unused, double *out_7396305060884469615) {
  H_14(state, unused, out_7396305060884469615);
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
